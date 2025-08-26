"""
Vision Transformer Training on CIFAR-10

Comprehensive training script with modern techniques:
- AdamW optimizer with cosine annealing and warmup
- Label smoothing and mixup/cutmix data augmentation  
- Stochastic depth (DropPath) regularization
- Gradient clipping and weight decay
- Model checkpointing and early stopping
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add models to path
sys.path.append(str(Path(__file__).parent))

from models import vit_tiny, vit_small, vit_base
from utils import (
    AverageMeter, ProgressMeter, accuracy,
    LabelSmoothingCrossEntropy, mixup_data, cutmix_data, mixup_criterion,
    CosineAnnealingWarmupRestarts, MetricsTracker, EarlyStopping,
    save_checkpoint, load_checkpoint, model_info
)


def get_transforms(train=True, use_autoaugment=False):
    """Get data transforms for CIFAR-10."""
    
    if train:
        transforms_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
        
        if use_autoaugment:
            # Add AutoAugment if available
            try:
                transforms_list.append(transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10))
            except AttributeError:
                print("AutoAugment not available, using basic augmentation")
        
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        return transforms.Compose(transforms_list)
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])


def get_cifar10_loaders(batch_size=128, num_workers=4, use_autoaugment=False):
    """Get CIFAR-10 data loaders."""
    
    # Data transforms
    train_transform = get_transforms(train=True, use_autoaugment=use_autoaugment)
    test_transform = get_transforms(train=False)
    
    # Datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader


def train_epoch(train_loader, model, criterion, optimizer, scheduler, epoch, device, args):
    """Train for one epoch."""
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix=f"Epoch: [{epoch}]"
    )
    
    # Switch to train mode
    model.train()
    
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        # Apply mixup/cutmix if enabled
        if args.mixup_alpha > 0 and np.random.rand() < args.mixup_prob:
            if args.cutmix and np.random.rand() < 0.5:
                images, targets_a, targets_b, lam = cutmix_data(images, target, args.mixup_alpha)
            else:
                images, targets_a, targets_b, lam = mixup_data(images, target, args.mixup_alpha)
            
            # Compute output
            output = model(images)
            loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
        else:
            # Regular forward pass
            output = model(images)
            loss = criterion(output, target)
        
        # Measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        
        # Compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping if enabled
        if args.clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            
        optimizer.step()
        
        # Update learning rate scheduler (step-wise)
        if hasattr(scheduler, 'step') and args.scheduler == 'warmup_cosine':
            scheduler.step()
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            progress.display(i)
            if args.debug and i > 10:  # Quick debug mode
                break
    
    return losses.avg, top1.avg.item(), top5.avg.item()


def validate(test_loader, model, criterion, device, args):
    """Validate model."""
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: '
    )
    
    # Switch to evaluate mode
    model.eval()
    
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(test_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # Compute output
            output = model(images)
            loss = criterion(output, target)
            
            # Measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % args.print_freq == 0:
                progress.display(i)
                if args.debug and i > 5:  # Quick debug mode
                    break
    
    print(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')
    
    return losses.avg, top1.avg.item(), top5.avg.item()


def plot_training_curves(metrics, save_path):
    """Plot and save training curves."""
    epochs = range(1, len(metrics.get_metric('train_loss')) + 1)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curves
    ax1.plot(epochs, metrics.get_metric('train_loss'), label='Train Loss', color='blue')
    ax1.plot(epochs, metrics.get_metric('val_loss'), label='Val Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(epochs, metrics.get_metric('train_acc1'), label='Train Acc@1', color='blue')
    ax2.plot(epochs, metrics.get_metric('val_acc1'), label='Val Acc@1', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Learning rate
    lr_values = metrics.get_metric('lr')
    if lr_values:
        ax3.plot(epochs, lr_values, label='Learning Rate', color='green')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True)
    
    # Top-5 accuracy
    ax4.plot(epochs, metrics.get_metric('train_acc5'), label='Train Acc@5', color='blue')
    ax4.plot(epochs, metrics.get_metric('val_acc5'), label='Val Acc@5', color='red')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_title('Top-5 Accuracy')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Vision Transformer CIFAR-10 Training')
    
    # Model architecture
    parser.add_argument('--model', default='vit_tiny', choices=['vit_tiny', 'vit_small', 'vit_base'],
                        help='ViT model variant')
    parser.add_argument('--patch-size', default=4, type=int, help='patch size')
    parser.add_argument('--embed-dim', default=192, type=int, help='embedding dimension')
    parser.add_argument('--depth', default=6, type=int, help='number of transformer blocks')
    parser.add_argument('--num-heads', default=3, type=int, help='number of attention heads')
    
    # Training parameters
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs')
    parser.add_argument('--batch-size', default=128, type=int, help='mini-batch size')
    parser.add_argument('--lr', default=3e-4, type=float, help='initial learning rate')
    parser.add_argument('--min-lr', default=1e-6, type=float, help='minimum learning rate')
    parser.add_argument('--weight-decay', default=0.05, type=float, help='weight decay')
    parser.add_argument('--warmup-epochs', default=5, type=int, help='warmup epochs')
    
    # Regularization
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout rate')
    parser.add_argument('--attn-dropout', default=0.1, type=float, help='attention dropout rate')
    parser.add_argument('--drop-path', default=0.1, type=float, help='drop path rate')
    parser.add_argument('--mixup-alpha', default=0.2, type=float, help='mixup alpha')
    parser.add_argument('--mixup-prob', default=0.8, type=float, help='mixup probability')
    parser.add_argument('--cutmix', action='store_true', help='use cutmix')
    parser.add_argument('--label-smoothing', default=0.1, type=float, help='label smoothing')
    parser.add_argument('--clip-grad', default=1.0, type=float, help='gradient clipping')
    
    # Scheduler
    parser.add_argument('--scheduler', default='warmup_cosine', 
                        choices=['cosine', 'step', 'warmup_cosine'],
                        help='learning rate scheduler')
    
    # Data augmentation
    parser.add_argument('--autoaugment', action='store_true', help='use AutoAugment')
    
    # Training settings
    parser.add_argument('--num-workers', default=4, type=int, help='number of workers')
    parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
    parser.add_argument('--save-freq', default=20, type=int, help='save frequency')
    parser.add_argument('--resume', default='', type=str, help='path to checkpoint')
    parser.add_argument('--evaluate', action='store_true', help='evaluate model only')
    
    # Hardware
    parser.add_argument('--device', default='auto', help='device to use')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    
    # Output
    parser.add_argument('--save-dir', default='./checkpoints', type=str, help='save directory')
    parser.add_argument('--experiment-name', default='vit_cifar10', type=str, help='experiment name')
    parser.add_argument('--debug', action='store_true', help='debug mode (quick run)')
    
    args = parser.parse_args()
    
    # Create save directory
    save_dir = Path(args.save_dir) / args.experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device configuration
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
        
    if torch.cuda.is_available() and device.type == 'cuda':
        cudnn.benchmark = True
        
    print(f"Using device: {device}")
    
    # Data loading
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_autoaugment=args.autoaugment
    )
    
    # Model creation
    print(f"Creating model: {args.model}")
    
    model_builders = {
        'vit_tiny': vit_tiny,
        'vit_small': vit_small,
        'vit_base': vit_base
    }
    
    model = model_builders[args.model](
        num_classes=10,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        drop_path_rate=args.drop_path
    )
    
    model = model.to(device)
    
    # Print model info
    model_info(model, verbose=False)
    
    # Loss function
    if args.label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    
    criterion = criterion.to(device)
    
    # Optimizer (AdamW for ViT)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    if args.scheduler == 'warmup_cosine':
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=args.epochs,
            max_lr=args.lr,
            min_lr=args.min_lr,
            warmup_steps=args.warmup_epochs
        )
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr
        )
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[60, 80], gamma=0.1
        )
    else:
        scheduler = None
    
    # Resume from checkpoint
    start_epoch = 0
    best_acc1 = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            start_epoch, best_acc1 = load_checkpoint(model, optimizer, args.resume)
            print(f"Loaded checkpoint (epoch {start_epoch}, best_acc1 {best_acc1:.3f})")
        else:
            print(f"No checkpoint found at '{args.resume}'")
    
    # Evaluate only
    if args.evaluate:
        val_loss, val_acc1, val_acc5 = validate(test_loader, model, criterion, device, args)
        print(f"Validation - Loss: {val_loss:.4f}, Acc@1: {val_acc1:.2f}%, Acc@5: {val_acc5:.2f}%")
        return
    
    # Metrics tracking
    metrics = MetricsTracker()
    
    # Early stopping
    early_stopping = EarlyStopping(patience=15, min_delta=0.01)
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    print("=" * 80)
    
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        # Training
        train_loss, train_acc1, train_acc5 = train_epoch(
            train_loader, model, criterion, optimizer, scheduler, epoch, device, args
        )
        
        # Validation
        val_loss, val_acc1, val_acc5 = validate(test_loader, model, criterion, device, args)
        
        # Update learning rate scheduler (epoch-wise)
        if scheduler is not None and args.scheduler != 'warmup_cosine':
            scheduler.step()
        
        # Track metrics
        lr = optimizer.param_groups[0]['lr']
        metrics.update(
            epoch=epoch,
            train_loss=train_loss,
            train_acc1=train_acc1,
            train_acc5=train_acc5,
            val_loss=val_loss,
            val_acc1=val_acc1,
            val_acc5=val_acc5,
            lr=lr
        )
        
        # Check for best accuracy
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)
        
        print(f'Epoch: [{epoch}/{args.epochs}] '
              f'Train Acc@1: {train_acc1:.2f}% '
              f'Val Acc@1: {val_acc1:.2f}% '
              f'Best Acc@1: {best_acc1:.2f}% '
              f'LR: {lr:.6f}')
        
        # Save checkpoint
        if epoch % args.save_freq == 0 or is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'model': args.model,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'args': args,
                'metrics': metrics.metrics
            }, is_best,
            filename=save_dir / f'checkpoint_epoch_{epoch}.pth',
            best_filename=save_dir / 'model_best.pth')
        
        # Early stopping
        if early_stopping(val_acc1, model):
            print(f"Early stopping at epoch {epoch}")
            break
            
        if args.debug and epoch >= 2:  # Quick debug mode
            break
    
    total_time = time.time() - start_time
    
    print("=" * 80)
    print("Training completed!")
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Best validation accuracy: {best_acc1:.2f}%")
    
    # Final metrics summary
    print("\nTraining Summary:")
    print("-" * 40)
    metrics.summary()
    
    # Plot training curves
    plot_training_curves(metrics, save_dir / 'training_curves.png')
    
    # Save final model
    torch.save({
        'model': args.model,
        'state_dict': model.state_dict(),
        'best_acc1': best_acc1,
        'args': args,
        'metrics': metrics.metrics
    }, save_dir / 'final_model.pth')
    
    print(f"Model saved to: {save_dir}")


if __name__ == '__main__':
    main()