"""
Complete Transformer architecture with Encoder, Decoder, and full Transformer model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from .layers import EncoderLayer, DecoderLayer, PositionalEncoding


class Encoder(nn.Module):
    """
    Transformer Encoder: Stack of encoder layers.
    """
    
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_geglu: bool = False
    ):
        """
        Initialize Transformer encoder.
        
        Args:
            num_layers: Number of encoder layers
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            use_geglu: Whether to use GEGLU in FFN
        """
        super(Encoder, self).__init__()
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, use_geglu) 
            for _ in range(num_layers)
        ])
        
        # Final layer norm (Pre-LN architecture)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list]:
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Self-attention mask
            
        Returns:
            output: Encoded representation (batch_size, seq_len, d_model)
            attention_weights: List of attention weights from each layer
        """
        attention_weights = []
        
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attention_weights.append(attn_weights)
            
        # Apply final layer norm
        x = self.norm(x)
        
        return x, attention_weights


class Decoder(nn.Module):
    """
    Transformer Decoder: Stack of decoder layers.
    """
    
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_geglu: bool = False
    ):
        """
        Initialize Transformer decoder.
        
        Args:
            num_layers: Number of decoder layers
            d_model: Model dimension 
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout: Dropout probability
            use_geglu: Whether to use GEGLU in FFN
        """
        super(Decoder, self).__init__()
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout, use_geglu)
            for _ in range(num_layers)
        ])
        
        # Final layer norm (Pre-LN architecture)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list, list]:
        """
        Forward pass through decoder.
        
        Args:
            x: Decoder input of shape (batch_size, target_seq_len, d_model)
            encoder_output: Encoder output of shape (batch_size, source_seq_len, d_model)
            self_attn_mask: Causal mask for self-attention
            cross_attn_mask: Padding mask for cross-attention
            
        Returns:
            output: Decoded representation (batch_size, target_seq_len, d_model)
            self_attention_weights: List of self-attention weights from each layer
            cross_attention_weights: List of cross-attention weights from each layer
        """
        self_attention_weights = []
        cross_attention_weights = []
        
        for layer in self.layers:
            x, self_attn_weights, cross_attn_weights = layer(
                x, encoder_output, self_attn_mask, cross_attn_mask
            )
            self_attention_weights.append(self_attn_weights)
            cross_attention_weights.append(cross_attn_weights)
            
        # Apply final layer norm
        x = self.norm(x)
        
        return x, self_attention_weights, cross_attention_weights


class Transformer(nn.Module):
    """
    Complete Transformer model for sequence-to-sequence tasks.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 64,
        dropout: float = 0.1,
        use_geglu: bool = False,
        pad_token_id: int = 0
    ):
        """
        Initialize complete Transformer model.
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            n_heads: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            use_geglu: Whether to use GEGLU in FFN
            pad_token_id: Padding token ID
        """
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        
        # Embedding layers
        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Encoder and Decoder
        self.encoder = Encoder(
            num_encoder_layers, d_model, n_heads, d_ff, dropout, use_geglu
        )
        self.decoder = Decoder(
            num_decoder_layers, d_model, n_heads, d_ff, dropout, use_geglu  
        )
        
        # Output projection layer
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def encode(
        self, 
        src: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list]:
        """
        Encode source sequence.
        
        Args:
            src: Source sequence of shape (batch_size, src_seq_len)
            src_mask: Source padding mask
            
        Returns:
            encoder_output: Encoded representation (batch_size, src_seq_len, d_model)
            attention_weights: Encoder attention weights
        """
        # Embedding + positional encoding
        x = self.encoder_embedding(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Encode
        encoder_output, attention_weights = self.encoder(x, src_mask)
        
        return encoder_output, attention_weights
        
    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, list, list]:
        """
        Decode target sequence.
        
        Args:
            tgt: Target sequence of shape (batch_size, tgt_seq_len)
            encoder_output: Encoder output of shape (batch_size, src_seq_len, d_model)
            tgt_mask: Target causal mask
            src_mask: Source padding mask
            
        Returns:
            decoder_output: Decoded representation (batch_size, tgt_seq_len, d_model)
            self_attention_weights: Decoder self-attention weights
            cross_attention_weights: Decoder cross-attention weights
        """
        # Embedding + positional encoding
        x = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Decode
        decoder_output, self_attn_weights, cross_attn_weights = self.decoder(
            x, encoder_output, tgt_mask, src_mask
        )
        
        return decoder_output, self_attn_weights, cross_attn_weights
        
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass of complete Transformer.
        
        Args:
            src: Source sequence of shape (batch_size, src_seq_len)
            tgt: Target sequence of shape (batch_size, tgt_seq_len)
            src_mask: Source padding mask
            tgt_mask: Target causal mask
            
        Returns:
            output: Logits of shape (batch_size, tgt_seq_len, vocab_size)
            attention_weights: Dictionary containing all attention weights
        """
        # Encode
        encoder_output, enc_attn_weights = self.encode(src, src_mask)
        
        # Decode
        decoder_output, self_attn_weights, cross_attn_weights = self.decode(
            tgt, encoder_output, tgt_mask, src_mask
        )
        
        # Project to vocabulary
        output = self.output_projection(decoder_output)
        
        # Collect attention weights
        attention_weights = {
            'encoder_attention': enc_attn_weights,
            'decoder_self_attention': self_attn_weights,  
            'decoder_cross_attention': cross_attn_weights
        }
        
        return output, attention_weights
        
    def generate_square_subsequent_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """
        Generate causal mask for decoder self-attention.
        
        Args:
            size: Sequence length
            device: Device to create mask on
            
        Returns:
            Causal mask of shape (size, size)
        """
        mask = torch.tril(torch.ones(size, size, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)
        
    def create_padding_mask(
        self, 
        tokens: torch.Tensor, 
        pad_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Create padding mask for attention.
        
        Args:
            tokens: Input tokens of shape (batch_size, seq_len)
            pad_token_id: Padding token ID (defaults to self.pad_token_id)
            
        Returns:
            Padding mask of shape (batch_size, 1, 1, seq_len)
        """
        if pad_token_id is None:
            pad_token_id = self.pad_token_id
            
        mask = (tokens != pad_token_id).unsqueeze(1).unsqueeze(2)
        return mask  # (batch_size, 1, 1, seq_len)
