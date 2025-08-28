"""
Attention mechanisms for Transformer architecture.
Implements Scaled Dot-Product Attention and Multi-Head Attention.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor, 
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.
    
    Args:
        query: Query tensor of shape (batch_size, seq_len, d_k) or (batch_size, n_heads, seq_len, d_k)
        key: Key tensor of shape (batch_size, seq_len, d_k) or (batch_size, n_heads, seq_len, d_k) 
        value: Value tensor of shape (batch_size, seq_len, d_v) or (batch_size, n_heads, seq_len, d_v)
        mask: Attention mask tensor. 1 for valid positions, 0 for masked positions
        dropout: Optional dropout layer
        
    Returns:
        attention_output: Weighted values (same shape as value)
        attention_weights: Attention weights (batch_size, n_heads, seq_len, seq_len)
    """
    d_k = query.size(-1)
    
    # Compute attention scores: QK^T / sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask if provided (set masked positions to large negative value)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Apply softmax to get attention weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply dropout if provided
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    
    # Apply attention weights to values
    attention_output = torch.matmul(attention_weights, value)
    
    return attention_output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Initialize Multi-Head Attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False) 
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Multi-Head Attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, d_model)
            key: Key tensor of shape (batch_size, seq_len, d_model)
            value: Value tensor of shape (batch_size, seq_len, d_model)
            mask: Attention mask of shape (batch_size, 1, seq_len, seq_len) or (batch_size, n_heads, seq_len, seq_len)
            
        Returns:
            output: Multi-head attention output (batch_size, seq_len, d_model)
            attention_weights: Attention weights (batch_size, n_heads, seq_len, seq_len)
        """
        batch_size = query.size(0)
        query_len = query.size(1)
        key_len = key.size(1)
        value_len = value.size(1)
        
        # Linear projections in batch from d_model => h x d_k
        Q = self.w_q(query).view(batch_size, query_len, self.n_heads, self.d_k).transpose(1, 2)  # (B, n_heads, query_len, d_k)
        K = self.w_k(key).view(batch_size, key_len, self.n_heads, self.d_k).transpose(1, 2)      # (B, n_heads, key_len, d_k)
        V = self.w_v(value).view(batch_size, value_len, self.n_heads, self.d_k).transpose(1, 2)  # (B, n_heads, value_len, d_k)
        
        # Apply attention on all the projected vectors in batch
        attn_output, attention_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout
        )
        
        # Concatenate heads and put through final linear layer
        # (B, n_heads, query_len, d_k) -> (B, query_len, n_heads, d_k) -> (B, query_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, query_len, self.d_model
        )
        
        output = self.w_o(attn_output)
        
        return output, attention_weights


class MultiHeadCrossAttention(MultiHeadAttention):
    """
    Multi-Head Cross-Attention for Decoder layers.
    Same as MultiHeadAttention but designed for cross-attention between encoder-decoder.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super(MultiHeadCrossAttention, self).__init__(d_model, n_heads, dropout)
        
    def forward(
        self,
        query: torch.Tensor,      # from decoder
        key: torch.Tensor,        # from encoder  
        value: torch.Tensor,      # from encoder
        mask: Optional[torch.Tensor] = None  # encoder padding mask
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-attention forward pass.
        Query comes from decoder, Key and Value from encoder.
        """
        return super().forward(query, key, value, mask)
