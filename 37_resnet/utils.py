"""
Training Utilities for ResNet

Includes metrics tracking, data augmentation, regularization techniques,
and other utilities for modern deep learning training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from collections import defaultdict


class AverageMeter:
    """Computes and stores the average and current value"""
    
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
    """Display progress during training"""
    
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
    """Computes accuracy for the specified values of k"""
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


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """
    Mixup data augmentation
    
    Reference: Zhang et al. (2017) "mixup: Beyond Empirical Risk Minimization"
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
        
    batch_size = x.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
        
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss function"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss
    
    Reference: Szegedy et al. (2016) "Rethinking the Inception Architecture"
    """
    
    def __init__(self, epsilon=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.ignore_index = ignore_index
        
    def forward(self, preds, target):
        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction, 
                         ignore_index=self.ignore_index)
        return self.linear_combination(loss / n, nll, self.epsilon)
        
    def linear_combination(self, x, y, epsilon):
        return epsilon * x + (1 - epsilon) * y
        
    def reduce_loss(self, loss, reduction='mean'):
        return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


def cutmix_data(x, y, alpha=1.0, use_cuda=True):
    """
    CutMix data augmentation
    
    Reference: Yun et al. (2019) "CutMix: Regularization Strategy to Train Strong Classifiers"
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
        
    batch_size = x.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
        
    y_a, y_b = y, y[index]
    
    # Generate random bounding box
    W = x.size(2)
    H = x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    
    # Uniform sampling
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    return x, y_a, y_b, lam


class WarmupCosineScheduler:
    """
    Cosine annealing with warmup
    
    Learning rate starts from 0, warms up linearly to max_lr,
    then follows cosine annealing to min_lr.
    """
    
    def __init__(self, optimizer, warmup_epochs, max_epochs, max_lr, min_lr=0, warmup_start_lr=0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.epoch = 0
        
    def step(self):
        if self.epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.warmup_start_lr + (self.max_lr - self.warmup_start_lr) * self.epoch / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        self.epoch += 1
        return lr


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
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


class MetricsTracker:
    """Track training and validation metrics"""
    
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
        """Print summary of all metrics"""
        for name, values in self.metrics.items():
            if values:
                print(f"{name}: latest={values[-1]:.4f}, "
                      f"best={max(values):.4f}, avg={np.mean(values):.4f}")


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_filename='model_best.pth.tar'):
    """Save training checkpoint"""
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)


def load_checkpoint(model, optimizer, filename):
    """Load training checkpoint"""
    if torch.cuda.is_available():
        checkpoint = torch.load(filename)
    else:
        checkpoint = torch.load(filename, map_location='cpu')
        
    start_epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    return start_epoch, best_acc


def adjust_learning_rate(optimizer, epoch, lr, schedule, gamma=0.1):
    """Adjust learning rate according to schedule"""
    if epoch in schedule:
        lr *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


class GradualWarmupScheduler:
    """Gradually warm-up learning rate"""
    
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.optimizer = optimizer
        
    def step(self, epoch):
        if epoch < self.total_epoch:
            multiplier = 1.0 * epoch / self.total_epoch
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * multiplier
        else:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [group['lr'] for group in self.optimizer.param_groups]
                    self.finished = True
                self.after_scheduler.step(epoch - self.total_epoch)


def model_info(model, verbose=False, input_size=(1, 3, 32, 32)):
    """
    Print model information including parameters and FLOPs estimation
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


def count_flops(model, input_size=(1, 3, 32, 32)):
    """
    Rough estimation of FLOPs for a forward pass
    Note: This is a simplified estimation
    """
    def conv_flop_count(input_shape, output_shape, kernel_shape, groups=1):
        batch_count = input_shape[0]
        output_dims = output_shape[2:]
        kernel_dims = kernel_shape[2:]
        in_channels = kernel_shape[1]
        out_channels = output_shape[1]
        
        filters_per_channel = out_channels // groups
        conv_per_position_flops = int(np.prod(kernel_dims)) * in_channels // groups
        
        active_elements_count = batch_count * int(np.prod(output_dims))
        overall_conv_flops = conv_per_position_flops * active_elements_count * filters_per_channel
        
        bias_flops = 0
        overall_flops = overall_conv_flops + bias_flops
        return overall_flops
    
    # This would require more sophisticated analysis
    # For now, return a placeholder
    return "FLOP counting requires more detailed implementation"