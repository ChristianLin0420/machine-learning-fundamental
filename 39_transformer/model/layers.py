"""
Transformer layers: EncoderLayer, DecoderLayer, Feed-Forward Networks, and Positional Encoding.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .attention import MultiHeadAttention, MultiHeadCrossAttention


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer models.
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_seq_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        
        # Create div_term for sine and cosine
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices  
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_seq_len, 1, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            x with positional encoding added
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].transpose(0, 1)  # (batch_size, seq_len, d_model)


class FeedForwardNetwork(nn.Module):
    """
    Position-wise feed-forward network with GELU activation.
    Optionally supports GEGLU (Gated Linear Units).
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, use_geglu: bool = False):
        """
        Initialize feed-forward network.
        
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            use_geglu: Whether to use GEGLU activation
        """
        super(FeedForwardNetwork, self).__init__()
        
        self.use_geglu = use_geglu
        
        if use_geglu:
            # GEGLU: FFN_GEGLU(x) = (xW1 ⊙ gelu(xW2))W3
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_model, d_ff) 
            self.linear3 = nn.Linear(d_ff, d_model)
        else:
            # Standard FFN: FFN(x) = max(0, xW1 + b1)W2 + b2
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        if self.use_geglu:
            # GEGLU variant
            gate = F.gelu(self.linear1(x))
            proj = self.linear2(x)
            hidden = gate * proj
            output = self.linear3(self.dropout(hidden))
        else:
            # Standard FFN with GELU
            hidden = F.gelu(self.linear1(x))
            output = self.linear2(self.dropout(hidden))
            
        return output


class EncoderLayer(nn.Module):
    """
    Single encoder layer with multi-head self-attention and feed-forward network.
    Uses Pre-LN architecture (LayerNorm before sublayers).
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        d_ff: int, 
        dropout: float = 0.1,
        use_geglu: bool = False
    ):
        """
        Initialize encoder layer.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            use_geglu: Whether to use GEGLU in FFN
        """
        super(EncoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout, use_geglu)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of encoder layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Self-attention mask
            
        Returns:
            output: Encoded output (batch_size, seq_len, d_model)
            attention_weights: Self-attention weights
        """
        # Pre-LN: Self-attention sublayer
        norm_x = self.norm1(x)
        attn_output, attention_weights = self.self_attention(norm_x, norm_x, norm_x, mask)
        x = x + self.dropout(attn_output)  # Residual connection
        
        # Pre-LN: Feed-forward sublayer  
        norm_x = self.norm2(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout(ff_output)    # Residual connection
        
        return x, attention_weights


class DecoderLayer(nn.Module):
    """
    Single decoder layer with masked self-attention, cross-attention, and feed-forward network.
    Uses Pre-LN architecture.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int, 
        d_ff: int,
        dropout: float = 0.1,
        use_geglu: bool = False
    ):
        """
        Initialize decoder layer.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension 
            dropout: Dropout probability
            use_geglu: Whether to use GEGLU in FFN
        """
        super(DecoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadCrossAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout, use_geglu)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor, 
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of decoder layer.
        
        Args:
            x: Decoder input of shape (batch_size, target_seq_len, d_model)
            encoder_output: Encoder output of shape (batch_size, source_seq_len, d_model)
            self_attn_mask: Causal mask for self-attention
            cross_attn_mask: Padding mask for cross-attention
            
        Returns:
            output: Decoded output (batch_size, target_seq_len, d_model)
            self_attention_weights: Self-attention weights  
            cross_attention_weights: Cross-attention weights
        """
        # Pre-LN: Masked self-attention sublayer
        norm_x = self.norm1(x)
        self_attn_output, self_attention_weights = self.self_attention(
            norm_x, norm_x, norm_x, self_attn_mask
        )
        x = x + self.dropout(self_attn_output)
        
        # Pre-LN: Cross-attention sublayer
        norm_x = self.norm2(x)
        cross_attn_output, cross_attention_weights = self.cross_attention(
            norm_x, encoder_output, encoder_output, cross_attn_mask
        )
        x = x + self.dropout(cross_attn_output)
        
        # Pre-LN: Feed-forward sublayer
        norm_x = self.norm3(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout(ff_output)
        
        return x, self_attention_weights, cross_attention_weights
