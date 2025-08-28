"""
Utility functions for Transformer training and evaluation.
Includes masks, label smoothing, batching, metrics, and toy dataset generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Tuple, Dict, Optional, Union
from collections import Counter
import math


# ========================================
# Mask Generation Functions
# ========================================

def create_padding_mask(tokens: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """
    Create padding mask for attention.
    
    Args:
        tokens: Input tokens of shape (batch_size, seq_len)
        pad_token_id: ID of padding token
        
    Returns:
        Mask of shape (batch_size, 1, 1, seq_len) where 1 = valid, 0 = padded
    """
    mask = (tokens != pad_token_id).unsqueeze(1).unsqueeze(2)
    return mask.float()


def create_causal_mask(size: int, device: torch.device) -> torch.Tensor:
    """
    Create causal (subsequent) mask for decoder self-attention.
    
    Args:
        size: Sequence length
        device: Device to create mask on
        
    Returns:
        Lower triangular mask of shape (1, 1, size, size)
    """
    mask = torch.tril(torch.ones(size, size, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


def create_combined_mask(
    tgt_tokens: torch.Tensor, 
    pad_token_id: int = 0
) -> torch.Tensor:
    """
    Create combined causal and padding mask for decoder.
    
    Args:
        tgt_tokens: Target tokens of shape (batch_size, seq_len)
        pad_token_id: ID of padding token
        
    Returns:
        Combined mask of shape (batch_size, 1, seq_len, seq_len)
    """
    batch_size, seq_len = tgt_tokens.size()
    device = tgt_tokens.device
    
    # Create padding mask
    padding_mask = create_padding_mask(tgt_tokens, pad_token_id)  # (B, 1, 1, seq_len)
    
    # Create causal mask
    causal_mask = create_causal_mask(seq_len, device)  # (1, 1, seq_len, seq_len)
    
    # Combine masks: both conditions must be true
    # Convert to boolean first, then combine
    padding_mask_bool = padding_mask.bool()
    causal_mask_bool = causal_mask.bool()
    combined_mask = padding_mask_bool & causal_mask_bool
    
    return combined_mask


# ========================================
# Label Smoothing Loss
# ========================================

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss for training.
    Reduces overconfidence and improves generalization.
    """
    
    def __init__(self, vocab_size: int, smoothing: float = 0.1, ignore_index: int = -100):
        """
        Initialize label smoothing loss.
        
        Args:
            vocab_size: Size of vocabulary
            smoothing: Smoothing parameter (epsilon)
            ignore_index: Index to ignore in loss computation (e.g., padding)
        """
        super(LabelSmoothingLoss, self).__init__()
        
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing loss.
        
        Args:
            pred: Predictions of shape (batch_size, seq_len, vocab_size) or (N, vocab_size)
            target: Target labels of shape (batch_size, seq_len) or (N,)
            
        Returns:
            Label smoothing loss
        """
        # Reshape if needed
        if pred.dim() == 3:
            pred = pred.reshape(-1, pred.size(-1))
            target = target.reshape(-1)
            
        # Create smooth target distribution
        smooth_target = torch.zeros_like(pred)
        smooth_target.fill_(self.smoothing / (self.vocab_size - 1))
        
        # Set confidence for true labels
        smooth_target.scatter_(1, target.unsqueeze(1), self.confidence)
        
        # Mask out ignore_index
        mask = (target != self.ignore_index)
        smooth_target = smooth_target * mask.unsqueeze(1)
        
        # Compute KL divergence
        loss = F.kl_div(F.log_softmax(pred, dim=1), smooth_target, reduction='none')
        loss = loss.sum(dim=1)  # Sum over vocab dimension
        
        # Average over non-ignored positions
        loss = loss.masked_select(mask).mean()
        
        return loss


# ========================================
# Metrics
# ========================================

def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> float:
    """
    Compute token-level accuracy.
    
    Args:
        predictions: Model predictions of shape (batch_size, seq_len, vocab_size)
        targets: Target tokens of shape (batch_size, seq_len)
        ignore_index: Index to ignore (e.g., padding)
        
    Returns:
        Token-level accuracy
    """
    pred_tokens = predictions.argmax(dim=-1)
    
    # Mask out ignored positions
    mask = (targets != ignore_index)
    correct = (pred_tokens == targets) & mask
    
    accuracy = correct.sum().float() / mask.sum().float()
    return accuracy.item()


def compute_sequence_accuracy(predictions: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> float:
    """
    Compute sequence-level accuracy (exact match).
    
    Args:
        predictions: Model predictions of shape (batch_size, seq_len, vocab_size)
        targets: Target tokens of shape (batch_size, seq_len)
        ignore_index: Index to ignore (e.g., padding)
        
    Returns:
        Sequence-level accuracy
    """
    pred_tokens = predictions.argmax(dim=-1)
    batch_size = targets.size(0)
    
    correct_sequences = 0
    for i in range(batch_size):
        # Get non-ignored positions
        mask = targets[i] != ignore_index
        pred_seq = pred_tokens[i][mask]
        target_seq = targets[i][mask]
        
        # Check if sequences match exactly
        if torch.equal(pred_seq, target_seq):
            correct_sequences += 1
            
    return correct_sequences / batch_size


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from loss."""
    return math.exp(loss)


# ========================================
# Toy Dataset Generation
# ========================================

class ToyDataset:
    """
    Generate toy seq2seq tasks (copy, reverse) for Transformer training.
    """
    
    def __init__(
        self,
        vocab_size: int = 100,
        seq_len_range: Tuple[int, int] = (5, 20),
        pad_token_id: int = 0,
        sos_token_id: int = 1,
        eos_token_id: int = 2
    ):
        """
        Initialize toy dataset generator.
        
        Args:
            vocab_size: Size of vocabulary
            seq_len_range: Range of sequence lengths (min, max)
            pad_token_id: Padding token ID
            sos_token_id: Start-of-sequence token ID  
            eos_token_id: End-of-sequence token ID
        """
        self.vocab_size = vocab_size
        self.seq_len_range = seq_len_range
        self.pad_token_id = pad_token_id
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        
        # Regular tokens (excluding special tokens)
        self.regular_tokens = list(range(3, vocab_size))
        
    def generate_copy_task(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate copy task: output = input.
        
        Args:
            batch_size: Number of sequences to generate
            
        Returns:
            source: Source sequences (batch_size, max_seq_len)
            target: Target sequences (batch_size, max_seq_len)
        """
        sequences = []
        max_len = 0
        
        for _ in range(batch_size):
            # Random sequence length
            seq_len = random.randint(*self.seq_len_range)
            max_len = max(max_len, seq_len + 2)  # +2 for SOS/EOS
            
            # Generate random sequence
            seq = random.choices(self.regular_tokens, k=seq_len)
            sequences.append(seq)
            
        # Create tensors
        source = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        target = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        
        for i, seq in enumerate(sequences):
            seq_len = len(seq)
            
            # Source: seq + EOS  
            source[i, :seq_len] = torch.tensor(seq)
            source[i, seq_len] = self.eos_token_id
            
            # Target: SOS + seq + EOS
            target[i, 0] = self.sos_token_id
            target[i, 1:seq_len+1] = torch.tensor(seq) 
            target[i, seq_len+1] = self.eos_token_id
            
        return source, target
        
    def generate_reverse_task(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate reverse task: output = reverse(input).
        
        Args:
            batch_size: Number of sequences to generate
            
        Returns:
            source: Source sequences (batch_size, max_seq_len)  
            target: Target sequences (batch_size, max_seq_len)
        """
        sequences = []
        max_len = 0
        
        for _ in range(batch_size):
            # Random sequence length
            seq_len = random.randint(*self.seq_len_range)
            max_len = max(max_len, seq_len + 2)  # +2 for SOS/EOS
            
            # Generate random sequence
            seq = random.choices(self.regular_tokens, k=seq_len)
            sequences.append(seq)
            
        # Create tensors
        source = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        target = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        
        for i, seq in enumerate(sequences):
            seq_len = len(seq)
            reversed_seq = seq[::-1]  # Reverse the sequence
            
            # Source: seq + EOS
            source[i, :seq_len] = torch.tensor(seq)
            source[i, seq_len] = self.eos_token_id
            
            # Target: SOS + reversed_seq + EOS  
            target[i, 0] = self.sos_token_id
            target[i, 1:seq_len+1] = torch.tensor(reversed_seq)
            target[i, seq_len+1] = self.eos_token_id
            
        return source, target


# ========================================
# Training Utilities  
# ========================================

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_optimizer(model: nn.Module, lr: float = 3e-4, weight_decay: float = 1e-2) -> torch.optim.Optimizer:
    """Create AdamW optimizer with recommended settings."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.98),
        eps=1e-9
    )


class NoamScheduler:
    """
    Noam learning rate scheduler from "Attention Is All You Need".
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int = 4000):
        """
        Initialize Noam scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            d_model: Model dimension
            warmup_steps: Number of warmup steps
        """
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
        
    def step(self):
        """Update learning rate."""
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


def save_checkpoint(
    model: nn.Module, 
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str
):
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
    device: torch.device
) -> int:
    """
    Load training checkpoint.
    
    Returns:
        Last epoch number
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    return checkpoint['epoch']
