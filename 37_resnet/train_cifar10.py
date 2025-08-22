"""
ResNet Training on CIFAR-10

End-to-end training script with modern techniques:
- Data augmentation (RandAugment, Mixup, CutMix)
- Label smoothing
- Cosine annealing with warmup
- He initialization with zero-gamma
- Weight decay and gradient clipping
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
from pathlib import Path

# Add models to path
sys.path.append(str(Path(__file__).parent))

from models import resnet18, resnet34, resnet50, resnet101, init_weights
from utils import (
    AverageMeter, ProgressMeter, accuracy, 
    mixup_data, mixup_criterion, cutmix_data,
    LabelSmoothingCrossEntropy, WarmupCosineScheduler,
    EarlyStopping, MetricsTracker, save_checkpoint, model_info
)


def get_transforms(train=True, use_randaugment=False):
    """Get data transforms for CIFAR-10"""
    if train:
        transforms_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        
        if use_randaugment:
            # Note: RandAugment requires torchvision >= 0.11
            try:
                from torchvision.transforms import RandAugment
                transforms_list.append(RandAugment(num_ops=2, magnitude=14))
            except ImportError:
                print("RandAugment not available, using basic augmentation")
        
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        return transforms.Compose(transforms_list)
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])


def get_cifar10_loaders(batch_size=128, num_workers=4, use_randaugment=False):
    """Get CIFAR-10 data loaders"""
    
    # Data transforms
    train_transform = get_transforms(train=True, use_randaugment=use_randaugment)
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
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader


def train_epoch(train_loader, model, criterion, optimizer, epoch, device, args, scheduler=None):
    """Train for one epoch"""
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
                images, targets_a, targets_b, lam = cutmix_data(images, target, args.mixup_alpha, device.type == 'cuda')
            else:
                images, targets_a, targets_b, lam = mixup_data(images, target, args.mixup_alpha, device.type == 'cuda')
            
            # Compute output
            output = model(images)
            loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
        else:
            # Regular forward pass
            output = model(images)
            loss = criterion(output, target)
        
        # Measure accuracy
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        
        # Compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping if enabled
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
        optimizer.step()
        
        # Update learning rate scheduler
        if scheduler is not None and args.scheduler == 'warmup_cosine':
            scheduler.step()
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            progress.display(i)
    
    return losses.avg, top1.avg.item(), top5.avg.item()


def validate(test_loader, model, criterion, device, args):
    """Validate model"""
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
            
            # Measure accuracy
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if i % args.print_freq == 0:
                progress.display(i)
    
    print(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')
    
    return losses.avg, top1.avg.item(), top5.avg.item()


def main():
    parser = argparse.ArgumentParser(description='ResNet CIFAR-10 Training')
    
    # Model architecture
    parser.add_argument('--arch', default='resnet18', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'],
                        help='model architecture')
    parser.add_argument('--pre-activation', action='store_true', help='use pre-activation ResNet')
    
    # Training parameters
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--batch-size', default=128, type=int, help='mini-batch size')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--weight-decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    
    # Learning rate scheduling
    parser.add_argument('--scheduler', default='warmup_cosine', choices=['step', 'cosine', 'warmup_cosine'],
                        help='learning rate scheduler')
    parser.add_argument('--warmup-epochs', default=5, type=int, help='warmup epochs')
    parser.add_argument('--min-lr', default=0, type=float, help='minimum learning rate')
    
    # Regularization
    parser.add_argument('--mixup-alpha', default=0.2, type=float, help='mixup alpha')
    parser.add_argument('--mixup-prob', default=1.0, type=float, help='mixup probability')
    parser.add_argument('--cutmix', action='store_true', help='use cutmix instead of mixup')
    parser.add_argument('--label-smoothing', default=0.1, type=float, help='label smoothing epsilon')
    parser.add_argument('--grad-clip', default=0, type=float, help='gradient clipping max norm')
    
    # Data augmentation
    parser.add_argument('--randaugment', action='store_true', help='use RandAugment')
    
    # Training settings
    parser.add_argument('--num-workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('--save-freq', default=50, type=int, help='save frequency')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
    
    # Hardware
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    
    # Output
    parser.add_argument('--save-dir', default='./checkpoints', type=str, help='directory to save checkpoints')
    parser.add_argument('--experiment-name', default='resnet_cifar10', type=str, help='experiment name')
    
    args = parser.parse_args()
    
    # Create save directory
    save_dir = Path(args.save_dir) / args.experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device configuration
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        cudnn.benchmark = True
        
    print(f"Using device: {device}")
    
    # Data loading
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        use_randaugment=args.randaugment
    )
    
    # Model creation
    print(f"Creating model: {args.arch} (pre_activation={args.pre_activation})")
    
    model_builders = {
        'resnet18': resnet18,
        'resnet34': resnet34, 
        'resnet50': resnet50,
        'resnet101': resnet101
    }
    
    model = model_builders[args.arch](num_classes=10, pre_activation=args.pre_activation)
    
    # Initialize weights
    init_weights(model, init_type='he_normal', zero_gamma=True)
    
    model = model.to(device)
    
    # Print model info
    model_info(model, verbose=False)
    
    # Loss function
    if args.label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(epsilon=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    
    criterion = criterion.to(device)
    
    # Optimizer
    optimizer = optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = None
    if args.scheduler == 'warmup_cosine':
        scheduler = WarmupCosineScheduler(
            optimizer, 
            warmup_epochs=args.warmup_epochs,
            max_epochs=args.epochs,
            max_lr=args.lr,
            min_lr=args.min_lr
        )
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    
    # Resume from checkpoint
    start_epoch = 0
    best_acc1 = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at '{args.resume}'")
    
    # Metrics tracking
    metrics = MetricsTracker()
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    print("=" * 80)
    
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        # Training
        train_loss, train_acc1, train_acc5 = train_epoch(
            train_loader, model, criterion, optimizer, epoch, device, args, scheduler
        )
        
        # Validation
        val_loss, val_acc1, val_acc5 = validate(test_loader, model, criterion, device, args)
        
        # Update learning rate scheduler (except warmup_cosine which updates per step)
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
        
        # Remember best accuracy
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
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'args': args,
                'metrics': metrics.metrics
            }, is_best, 
            filename=save_dir / f'checkpoint_epoch_{epoch}.pth.tar',
            best_filename=save_dir / 'model_best.pth.tar')
    
    total_time = time.time() - start_time
    
    print("=" * 80)
    print("Training completed!")
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Best validation accuracy: {best_acc1:.2f}%")
    
    # Final metrics summary
    print("\nTraining Summary:")
    print("-" * 40)
    metrics.summary()
    
    # Save final model
    torch.save({
        'arch': args.arch,
        'state_dict': model.state_dict(),
        'best_acc1': best_acc1,
        'args': args,
        'metrics': metrics.metrics
    }, save_dir / 'final_model.pth.tar')
    
    print(f"Model saved to: {save_dir}")


if __name__ == '__main__':
    main()