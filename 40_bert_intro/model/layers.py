"""
Core Transformer layers for BERT implementation.
Includes Multi-Head Attention, Feed-Forward Network, Layer Normalization, and Positional Embeddings.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention mechanism."""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x, attention_mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            attention_mask: (batch_size, seq_len) - 1 for real tokens, 0 for padding
        """
        batch_size, seq_len = x.size(0), x.size(1)
        
        # Linear transformations and split into heads
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply attention mask
        if attention_mask is not None:
            # Convert mask to attention mask format
            # attention_mask: (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
            mask = attention_mask.unsqueeze(1).unsqueeze(1).float()
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.w_o(context)
        
        return output


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network."""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))


class EncoderLayer(nn.Module):
    """Single Transformer encoder layer with pre-normalization."""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attention_mask=None):
        # Pre-norm self-attention with residual connection
        normed = self.norm1(x)
        attn_output = self.self_attention(normed, attention_mask)
        x = x + self.dropout(attn_output)
        
        # Pre-norm feed-forward with residual connection
        normed = self.norm2(x)
        ff_output = self.feed_forward(normed)
        x = x + self.dropout(ff_output)
        
        return x


class PositionalEmbedding(nn.Module):
    """Learned positional embeddings."""
    
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        self.embeddings = nn.Embedding(max_seq_len, d_model)
        
    def forward(self, seq_len):
        positions = torch.arange(seq_len, device=self.embeddings.weight.device)
        return self.embeddings(positions)


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings (from original Transformer paper)."""
    
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, seq_len):
        return self.pe[:seq_len]


class BERTEmbeddings(nn.Module):
    """BERT-style embeddings: token + segment + position."""
    
    def __init__(self, vocab_size, d_model, max_seq_len=512, n_segments=2, 
                 pad_token_id=0, dropout=0.1, use_sinusoidal_pos=False):
        super().__init__()
        
        self.token_embeddings = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.segment_embeddings = nn.Embedding(n_segments, d_model)
        
        if use_sinusoidal_pos:
            self.position_embeddings = SinusoidalPositionalEmbedding(max_seq_len, d_model)
        else:
            self.position_embeddings = PositionalEmbedding(max_seq_len, d_model)
            
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, segment_ids=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            segment_ids: (batch_size, seq_len)
        """
        batch_size, seq_len = input_ids.size()
        
        # Token embeddings
        token_embeds = self.token_embeddings(input_ids)
        
        # Position embeddings - ensure we don't exceed max length and match exactly
        if hasattr(self.position_embeddings, 'embeddings'):
            max_pos_len = self.position_embeddings.embeddings.num_embeddings
        else:
            max_pos_len = 512  # Default fallback
        
        pos_embeds = self.position_embeddings(min(seq_len, max_pos_len))
        
        # Truncate or pad position embeddings to match input length
        if pos_embeds.size(0) > seq_len:
            pos_embeds = pos_embeds[:seq_len]
        elif pos_embeds.size(0) < seq_len:
            # If position embeddings are shorter, pad with the last position
            last_pos = pos_embeds[-1:].expand(seq_len - pos_embeds.size(0), -1)
            pos_embeds = torch.cat([pos_embeds, last_pos], dim=0)
            
        pos_embeds = pos_embeds.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Segment embeddings
        if segment_ids is not None:
            segment_embeds = self.segment_embeddings(segment_ids)
        else:
            segment_embeds = torch.zeros_like(token_embeds)
        
        # Ensure all embeddings have the same shape
        assert token_embeds.shape == pos_embeds.shape == segment_embeds.shape, \
            f"Shape mismatch: token {token_embeds.shape}, pos {pos_embeds.shape}, segment {segment_embeds.shape}"
            
        # Combine all embeddings
        embeddings = token_embeds + pos_embeds + segment_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings
