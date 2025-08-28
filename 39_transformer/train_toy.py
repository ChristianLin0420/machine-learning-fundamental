"""
Training script for Transformer on toy seq2seq tasks (copy/reverse).
Uses teacher forcing, label smoothing, and proper evaluation metrics.
"""

import torch
import torch.nn as nn
import argparse
import time
import os
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path

from model import Transformer
from utils import (
    ToyDataset,
    LabelSmoothingLoss,
    create_padding_mask,
    create_combined_mask,
    compute_accuracy,
    compute_sequence_accuracy,
    compute_perplexity,
    count_parameters,
    create_optimizer,
    NoamScheduler,
    save_checkpoint,
    load_checkpoint
)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def evaluate_model(
    model: nn.Module,
    dataset: ToyDataset,
    task_type: str,
    num_batches: int,
    batch_size: int,
    criterion: nn.Module,
    device: torch.device,
    pad_token_id: int = 0
) -> dict:
    """
    Evaluate model on validation data.
    
    Args:
        model: Transformer model
        dataset: Toy dataset generator
        task_type: 'copy' or 'reverse'
        num_batches: Number of evaluation batches
        batch_size: Batch size
        criterion: Loss function
        device: Device to run on
        pad_token_id: Padding token ID
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    total_loss = 0
    total_token_acc = 0
    total_seq_acc = 0
    
    with torch.no_grad():
        for _ in range(num_batches):
            # Generate batch
            if task_type == 'copy':
                src, tgt = dataset.generate_copy_task(batch_size)
            else:  # reverse
                src, tgt = dataset.generate_reverse_task(batch_size)
                
            src = src.to(device)
            tgt = tgt.to(device)
            
            # Create masks
            src_mask = create_padding_mask(src, pad_token_id)
            tgt_input = tgt[:, :-1]  # Remove last token for input
            tgt_output = tgt[:, 1:]  # Remove first token for target
            tgt_mask = create_combined_mask(tgt_input, pad_token_id)
            
            # Forward pass
            logits, _ = model(src, tgt_input, src_mask, tgt_mask)
            
            # Compute loss
            loss = criterion(logits, tgt_output)
            total_loss += loss.item()
            
            # Compute metrics
            token_acc = compute_accuracy(logits, tgt_output, ignore_index=pad_token_id)
            seq_acc = compute_sequence_accuracy(logits, tgt_output, ignore_index=pad_token_id)
            
            total_token_acc += token_acc
            total_seq_acc += seq_acc
    
    # Average metrics
    avg_loss = total_loss / num_batches
    avg_token_acc = total_token_acc / num_batches
    avg_seq_acc = total_seq_acc / num_batches
    perplexity = compute_perplexity(avg_loss)
    
    return {
        'loss': avg_loss,
        'token_accuracy': avg_token_acc,
        'sequence_accuracy': avg_seq_acc,
        'perplexity': perplexity
    }


def train_epoch(
    model: nn.Module,
    dataset: ToyDataset,
    task_type: str,
    num_batches: int,
    batch_size: int,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: NoamScheduler,
    device: torch.device,
    pad_token_id: int = 0
) -> dict:
    """
    Train model for one epoch.
    
    Returns:
        Dictionary with training metrics
    """
    model.train()
    
    total_loss = 0
    total_token_acc = 0
    total_seq_acc = 0
    
    pbar = tqdm(range(num_batches), desc="Training")
    
    for batch_idx in pbar:
        # Generate batch
        if task_type == 'copy':
            src, tgt = dataset.generate_copy_task(batch_size)
        else:  # reverse
            src, tgt = dataset.generate_reverse_task(batch_size)
            
        src = src.to(device)
        tgt = tgt.to(device)
        
        # Create masks
        src_mask = create_padding_mask(src, pad_token_id)
        tgt_input = tgt[:, :-1]  # Remove last token for input (teacher forcing)
        tgt_output = tgt[:, 1:]  # Remove first token for target
        tgt_mask = create_combined_mask(tgt_input, pad_token_id)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits, _ = model(src, tgt_input, src_mask, tgt_mask)
        
        # Compute loss
        loss = criterion(logits, tgt_output)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update parameters
        optimizer.step()
        scheduler.step()
        
        # Update metrics
        total_loss += loss.item()
        token_acc = compute_accuracy(logits, tgt_output, ignore_index=pad_token_id)
        seq_acc = compute_sequence_accuracy(logits, tgt_output, ignore_index=pad_token_id)
        
        total_token_acc += token_acc
        total_seq_acc += seq_acc
        
        # Update progress bar
        if batch_idx % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'token_acc': f'{token_acc:.3f}',
                'seq_acc': f'{seq_acc:.3f}',
                'lr': f'{current_lr:.2e}'
            })
    
    # Average metrics
    avg_loss = total_loss / num_batches
    avg_token_acc = total_token_acc / num_batches
    avg_seq_acc = total_seq_acc / num_batches
    
    return {
        'loss': avg_loss,
        'token_accuracy': avg_token_acc,
        'sequence_accuracy': avg_seq_acc
    }


def main():
    parser = argparse.ArgumentParser(description='Train Transformer on toy seq2seq tasks')
    
    # Model hyperparameters
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads') 
    parser.add_argument('--num_encoder_layers', type=int, default=4, help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=4, help='Number of decoder layers')
    parser.add_argument('--d_ff', type=int, default=1024, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--use_geglu', action='store_true', help='Use GEGLU in FFN')
    
    # Dataset parameters
    parser.add_argument('--vocab_size', type=int, default=100, help='Vocabulary size')
    parser.add_argument('--max_seq_len', type=int, default=64, help='Maximum sequence length')
    parser.add_argument('--seq_len_range', type=str, default='5,20', help='Sequence length range (min,max)')
    parser.add_argument('--task', type=str, default='copy', choices=['copy', 'reverse'], help='Task type')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batches_per_epoch', type=int, default=100, help='Batches per epoch')
    parser.add_argument('--eval_batches', type=int, default=20, help='Evaluation batches')
    parser.add_argument('--eval_interval', type=int, default=10, help='Evaluation interval (epochs)')
    
    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps for Noam scheduler')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing epsilon')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Save directory')
    parser.add_argument('--save_interval', type=int, default=50, help='Save interval (epochs)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Parse sequence length range
    seq_len_min, seq_len_max = map(int, args.seq_len_range.split(','))
    
    # Create dataset
    dataset = ToyDataset(
        vocab_size=args.vocab_size,
        seq_len_range=(seq_len_min, seq_len_max)
    )
    
    # Create model
    model = Transformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        use_geglu=args.use_geglu,
        pad_token_id=0
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Create loss function
    criterion = LabelSmoothingLoss(
        vocab_size=args.vocab_size,
        smoothing=args.label_smoothing,
        ignore_index=0  # Ignore padding tokens
    )
    
    # Create optimizer
    optimizer = create_optimizer(model, args.lr, args.weight_decay)
    
    # Create scheduler
    scheduler = NoamScheduler(optimizer, args.d_model, args.warmup_steps)
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, args.resume, device)
        print(f"Resumed training from epoch {start_epoch}")
    
    print(f"Training Transformer on {args.task} task")
    print(f"Task: {args.task}, Vocab: {args.vocab_size}, Seq len: {seq_len_min}-{seq_len_max}")
    print("-" * 60)
    
    # Training loop
    best_seq_acc = 0.0
    
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Train for one epoch
        train_metrics = train_epoch(
            model, dataset, args.task, args.batches_per_epoch, args.batch_size,
            criterion, optimizer, scheduler, device
        )
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Token Acc: {train_metrics['token_accuracy']:.3f}, "
              f"Seq Acc: {train_metrics['sequence_accuracy']:.3f}")
        
        # Evaluate periodically
        if (epoch + 1) % args.eval_interval == 0:
            eval_metrics = evaluate_model(
                model, dataset, args.task, args.eval_batches, args.batch_size,
                criterion, device
            )
            
            print(f"Eval  - Loss: {eval_metrics['loss']:.4f}, "
                  f"Token Acc: {eval_metrics['token_accuracy']:.3f}, "
                  f"Seq Acc: {eval_metrics['sequence_accuracy']:.3f}, "
                  f"PPL: {eval_metrics['perplexity']:.2f}")
            
            # Save best model
            if eval_metrics['sequence_accuracy'] > best_seq_acc:
                best_seq_acc = eval_metrics['sequence_accuracy']
                save_checkpoint(
                    model, optimizer, epoch, eval_metrics['loss'],
                    save_dir / 'best_model.pt'
                )
                print(f"Saved best model (seq acc: {best_seq_acc:.3f})")
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                model, optimizer, epoch, train_metrics['loss'],
                save_dir / f'checkpoint_epoch_{epoch+1}.pt'
            )
    
    print(f"\\nTraining completed! Best sequence accuracy: {best_seq_acc:.3f}")


if __name__ == '__main__':
    main()
