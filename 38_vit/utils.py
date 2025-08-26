"""
Training Utilities for Vision Transformer

Includes metrics tracking, data augmentation (mixup/cutmix), label smoothing,
learning rate scheduling, and other utilities for modern ViT training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math
from collections import defaultdict
from typing import Tuple, Optional


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """Display training progress."""
    
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        
    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes accuracy for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss.
    
    Reduces overconfidence and improves model calibration by smoothing
    the target distribution.
    
    Reference: Szegedy et al. (2016) "Rethinking the Inception Architecture"
    """
    
    def __init__(self, smoothing=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
        
    def forward(self, pred, target):
        """
        Args:
            pred: Predictions (B, num_classes)
            target: Ground truth labels (B,)
        """
        num_classes = pred.size(-1)
        log_pred = F.log_softmax(pred, dim=-1)
        
        # Create smooth target distribution
        with torch.no_grad():
            true_dist = torch.zeros_like(log_pred)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1.0 - self.smoothing)
            
        loss = torch.sum(-true_dist * log_pred, dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def mixup_data(x, y, alpha=0.2):
    """
    Mixup data augmentation.
    
    Reference: Zhang et al. (2017) "mixup: Beyond Empirical Risk Minimization"
    
    Args:
        x: Input images (B, C, H, W)
        y: Labels (B,)
        alpha: Mixup strength parameter
        
    Returns:
        mixed_x: Mixed images
        y_a, y_b: Original labels
        lam: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
        
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss function."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cutmix_data(x, y, alpha=1.0):
    """
    CutMix data augmentation.
    
    Reference: Yun et al. (2019) "CutMix: Regularization Strategy"
    
    Args:
        x: Input images (B, C, H, W)
        y: Labels (B,)
        alpha: CutMix strength parameter
        
    Returns:
        Mixed images and labels
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
        
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    y_a, y_b = y, y[index]
    
    # Generate random bounding box
    W, H = x.size(3), x.size(2)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # Uniform sampling of center point
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Apply cutmix
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    return x, y_a, y_b, lam


class CosineAnnealingWarmupRestarts:
    """
    Cosine annealing with warmup and restarts.
    
    Combines linear warmup with cosine annealing schedule, commonly used for ViT training.
    """
    
    def __init__(self, optimizer, first_cycle_steps, cycle_mult=1.0, max_lr=0.1,
                 min_lr=0.001, warmup_steps=0, gamma=1.0):
        """
        Args:
            optimizer: Optimizer to schedule
            first_cycle_steps: Number of steps in first cycle
            cycle_mult: Cycle steps multiplier after each restart
            max_lr: Maximum learning rate
            min_lr: Minimum learning rate 
            warmup_steps: Number of warmup steps
            gamma: Decrease rate of max_lr after each restart
        """
        self.optimizer = optimizer
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = 0
        self.step_count = 0
        
    def step(self):
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Warmup phase
            lr = self.max_lr * self.step_count / self.warmup_steps
        else:
            # Cosine annealing phase
            self.step_in_cycle += 1
            
            if self.step_in_cycle > self.cur_cycle_steps:
                # Start new cycle
                self.cycle += 1
                self.step_in_cycle = 1
                self.cur_cycle_steps = int(self.cur_cycle_steps * self.cycle_mult)
                self.max_lr *= self.gamma
                
            lr = self.min_lr + (self.max_lr - self.min_lr) * \
                 (1 + math.cos(math.pi * self.step_in_cycle / self.cur_cycle_steps)) / 2
                
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        return lr


class MetricsTracker:
    """Track training and validation metrics."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.metrics = defaultdict(list)
        
    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.metrics[key].append(value)
            
    def get_metric(self, name):
        return self.metrics[name] if name in self.metrics else []
        
    def get_latest(self, name, default=0):
        values = self.get_metric(name)
        return values[-1] if values else default
        
    def get_best(self, name, maximize=True):
        values = self.get_metric(name)
        if not values:
            return None
        return max(values) if maximize else min(values)
        
    def summary(self):
        """Print summary of all metrics."""
        for name, values in self.metrics.items():
            if values:
                print(f"{name}: latest={values[-1]:.4f}, "
                      f"best={max(values):.4f}, avg={np.mean(values):.4f}")


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
        
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', 
                   best_filename='model_best.pth.tar'):
    """Save training checkpoint."""
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)


def load_checkpoint(model, optimizer, filename):
    """Load training checkpoint."""
    checkpoint = torch.load(filename, map_location='cpu')
    
    start_epoch = checkpoint.get('epoch', 0)
    best_acc = checkpoint.get('best_acc', 0.0)
    
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        
    return start_epoch, best_acc


def get_lr_scheduler(optimizer, scheduler_type='cosine', **kwargs):
    """
    Get learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler ('cosine', 'step', 'warmup_cosine')
        **kwargs: Additional scheduler parameters
        
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=kwargs.get('epochs', 100),
            eta_min=kwargs.get('min_lr', 0.0)
        )
    elif scheduler_type == 'step':
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=kwargs.get('milestones', [60, 120, 160]),
            gamma=kwargs.get('gamma', 0.2)
        )
    elif scheduler_type == 'warmup_cosine':
        return CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=kwargs.get('epochs', 100),
            max_lr=kwargs.get('max_lr', 0.001),
            min_lr=kwargs.get('min_lr', 0.0),
            warmup_steps=kwargs.get('warmup_steps', 5)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def model_info(model, verbose=False, input_size=(1, 3, 32, 32)):
    """
    Print model information including parameters and memory usage.
    """
    n_params = sum(p.numel() for p in model.parameters())
    n_gradients = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Summary:")
    print(f"  Total params: {n_params:,}")
    print(f"  Trainable params: {n_gradients:,}")
    print(f"  Non-trainable params: {n_params - n_gradients:,}")
    
    # Memory estimation
    param_size = n_params * 4 / (1024 ** 2)  # Assume float32
    print(f"  Estimated size: {param_size:.2f}MB")
    
    if verbose:
        print("\nLayer-wise parameter count:")
        for name, param in model.named_parameters():
            print(f"  {name}: {param.numel():,}")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        x = torch.randn(input_size)
        try:
            y = model(x)
            print(f"  Input shape: {tuple(x.shape)}")
            print(f"  Output shape: {tuple(y.shape)}")
        except Exception as e:
            print(f"  Forward pass failed: {e}")


def compute_flops_vit(model, input_size=(3, 32, 32)):
    """
    Estimate FLOPs for Vision Transformer.
    
    This is a simplified estimation focusing on the main components.
    """
    C, H, W = input_size
    patch_size = model.patch_embed.patch_size
    embed_dim = model.embed_dim
    num_heads = model.blocks[0].attn.num_heads
    depth = model.depth
    
    # Number of patches
    num_patches = (H // patch_size) * (W // patch_size)
    seq_len = num_patches + 1  # +1 for CLS token
    
    flops = 0
    
    # Patch embedding FLOPs
    flops += num_patches * (patch_size * patch_size * C) * embed_dim
    
    # Transformer blocks FLOPs
    for _ in range(depth):
        # Multi-head attention FLOPs
        # QKV projection: 3 * seq_len * embed_dim^2
        flops += 3 * seq_len * embed_dim * embed_dim
        
        # Attention computation: seq_len^2 * embed_dim
        flops += seq_len * seq_len * embed_dim
        
        # Output projection: seq_len * embed_dim^2
        flops += seq_len * embed_dim * embed_dim
        
        # MLP FLOPs
        mlp_ratio = 4.0  # Standard ViT MLP ratio
        hidden_dim = int(embed_dim * mlp_ratio)
        flops += seq_len * embed_dim * hidden_dim  # First linear layer
        flops += seq_len * hidden_dim * embed_dim  # Second linear layer
    
    # Classification head FLOPs
    num_classes = model.num_classes
    flops += embed_dim * num_classes
    
    return flops


# Utility functions for data augmentation
class RandomResizedCrop:
    """Random resized crop for CIFAR-10 (simplified version)."""
    
    def __init__(self, size=32, scale=(0.8, 1.0), ratio=(0.75, 1.33)):
        self.size = size
        self.scale = scale
        self.ratio = ratio
        
    def __call__(self, img):
        # Simplified implementation - in practice use torchvision transforms
        return img


class AutoAugment:
    """
    Placeholder for AutoAugment data augmentation.
    In practice, use torchvision.transforms.AutoAugment or RandAugment.
    """
    
    def __init__(self, policy='cifar10'):
        self.policy = policy
        
    def __call__(self, img):
        # Placeholder - implement specific augmentations
        return img


if __name__ == "__main__":
    # Test utilities
    print("Testing Vision Transformer training utilities...")
    
    # Test accuracy computation
    output = torch.randn(10, 10)  # 10 samples, 10 classes
    target = torch.randint(0, 10, (10,))
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    print(f"Top-1 accuracy: {acc1[0]:.2f}%")
    print(f"Top-5 accuracy: {acc5[0]:.2f}%")
    
    # Test label smoothing
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    loss = criterion(output, target)
    print(f"Label smoothing loss: {loss:.4f}")
    
    # Test mixup
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)
    print(f"Mixup lambda: {lam:.4f}")
    
    print("Utilities test completed! ✅")