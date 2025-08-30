"""
BERT Encoder implementation using transformer encoder layers.
"""

import torch
import torch.nn as nn
from .layers import BERTEmbeddings, EncoderLayer


class BERTEncoder(nn.Module):
    """BERT Encoder: embeddings + N×EncoderLayer."""
    
    def __init__(self, vocab_size, d_model=256, n_layers=4, n_heads=4, d_ff=1024, 
                 max_seq_len=512, n_segments=2, pad_token_id=0, dropout=0.1,
                 use_sinusoidal_pos=False):
        super().__init__()
        
        self.embeddings = BERTEmbeddings(
            vocab_size=vocab_size,
            d_model=d_model,
            max_seq_len=max_seq_len,
            n_segments=n_segments,
            pad_token_id=pad_token_id,
            dropout=dropout,
            use_sinusoidal_pos=use_sinusoidal_pos
        )
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, input_ids, segment_ids=None, attention_mask=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            segment_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len) - 1 for real tokens, 0 for padding
        
        Returns:
            hidden_states: (batch_size, seq_len, d_model)
        """
        # Ensure inputs don't exceed max sequence length (512 is the default)
        seq_len = input_ids.size(1)
        max_len = 512  # Use default max length
        
        if seq_len > max_len:
            input_ids = input_ids[:, :max_len]
            if segment_ids is not None:
                segment_ids = segment_ids[:, :max_len]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :max_len]
        
        # Get embeddings
        hidden_states = self.embeddings(input_ids, segment_ids)
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask)
            
        # Final layer normalization
        hidden_states = self.final_norm(hidden_states)
        
        return hidden_states


class BERTModel(nn.Module):
    """Complete BERT model with encoder and optional heads."""
    
    def __init__(self, config):
        super().__init__()
        
        self.encoder = BERTEncoder(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
            d_ff=config['d_ff'],
            max_seq_len=config['max_seq_length'],
            n_segments=config.get('n_segments', 2),
            pad_token_id=config.get('pad_token_id', 0),
            dropout=config.get('dropout', 0.1),
            use_sinusoidal_pos=config.get('use_sinusoidal_pos', False)
        )
        
        self.config = config
        
    def forward(self, input_ids, segment_ids=None, attention_mask=None):
        """Forward pass through BERT encoder."""
        return self.encoder(input_ids, segment_ids, attention_mask)
    
    def get_input_embeddings(self):
        """Get token embeddings for weight tying."""
        return self.encoder.embeddings.token_embeddings
    
    def get_output_embeddings(self):
        """Get output embeddings (same as input for weight tying)."""
        return self.encoder.embeddings.token_embeddings
        
    @classmethod
    def from_config(cls, config_dict):
        """Create model from configuration dictionary."""
        return cls(config_dict)


def create_attention_mask(input_ids, pad_token_id=0):
    """Create attention mask from input_ids."""
    return (input_ids != pad_token_id).long()


def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
