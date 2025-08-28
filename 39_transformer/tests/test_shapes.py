"""
Test suite for checking tensor shapes and mask functionality in Transformer model.
Run these tests to ensure proper implementation before training.
"""

import torch
import pytest
import sys
import os

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Transformer, MultiHeadAttention, EncoderLayer, DecoderLayer
from utils import (
    create_padding_mask, 
    create_causal_mask,
    create_combined_mask,
    ToyDataset,
    LabelSmoothingLoss
)


class TestMasks:
    """Test mask creation functions."""
    
    def test_padding_mask_shape(self):
        """Test padding mask has correct shape."""
        batch_size, seq_len = 4, 10
        tokens = torch.randint(0, 100, (batch_size, seq_len))
        tokens[:, -2:] = 0  # Add some padding
        
        mask = create_padding_mask(tokens, pad_token_id=0)
        
        expected_shape = (batch_size, 1, 1, seq_len)
        assert mask.shape == expected_shape, f"Expected {expected_shape}, got {mask.shape}"
        assert mask.dtype == torch.float32, f"Expected float32, got {mask.dtype}"
        
    def test_padding_mask_values(self):
        """Test padding mask has correct values."""
        tokens = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]])
        mask = create_padding_mask(tokens, pad_token_id=0)
        
        # Should be 1 for non-padding, 0 for padding
        expected = torch.tensor([
            [[[1., 1., 1., 0., 0.]]],
            [[[1., 1., 0., 0., 0.]]]
        ])
        
        assert torch.equal(mask, expected), f"Mask values incorrect: {mask}"
        
    def test_causal_mask_shape(self):
        """Test causal mask has correct shape."""
        seq_len = 5
        device = torch.device('cpu')
        
        mask = create_causal_mask(seq_len, device)
        
        expected_shape = (1, 1, seq_len, seq_len)
        assert mask.shape == expected_shape, f"Expected {expected_shape}, got {mask.shape}"
        
    def test_causal_mask_values(self):
        """Test causal mask is lower triangular."""
        seq_len = 4
        device = torch.device('cpu')
        
        mask = create_causal_mask(seq_len, device)
        
        # Should be lower triangular
        expected = torch.tensor([[[
            [1., 0., 0., 0.],
            [1., 1., 0., 0.], 
            [1., 1., 1., 0.],
            [1., 1., 1., 1.]
        ]]])
        
        assert torch.equal(mask, expected), f"Causal mask incorrect: {mask}"
        
    def test_combined_mask(self):
        """Test combined mask combines causal and padding correctly."""
        tgt_tokens = torch.tensor([[1, 2, 3, 0], [1, 2, 0, 0]])
        mask = create_combined_mask(tgt_tokens, pad_token_id=0)
        
        batch_size, seq_len = tgt_tokens.shape
        assert mask.shape == (batch_size, 1, seq_len, seq_len)
        
        # Check that it respects both causal and padding constraints
        # First sequence: valid positions [0,1,2], padding [3]
        # Second sequence: valid positions [0,1], padding [2,3]
        
        # Position (2,3) should be 0 due to padding in first sequence
        assert mask[0, 0, 2, 3] == 0
        # Position (1,2) should be 0 due to padding in second sequence  
        assert mask[1, 0, 1, 2] == 0
        # Position (2,1) should be 1 (valid causal connection) in first sequence
        assert mask[0, 0, 2, 1] == 1


class TestTransformerShapes:
    """Test Transformer model tensor shapes."""
    
    def setup_method(self):
        """Setup test model."""
        self.vocab_size = 100
        self.d_model = 64
        self.n_heads = 8
        self.d_ff = 256
        self.max_seq_len = 32
        
        self.model = Transformer(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            num_encoder_layers=2,
            num_decoder_layers=2,
            d_ff=self.d_ff,
            max_seq_len=self.max_seq_len,
            dropout=0.1
        )
        
    def test_embedding_shapes(self):
        """Test embedding layer output shapes."""
        batch_size, seq_len = 4, 10
        tokens = torch.randint(1, self.vocab_size, (batch_size, seq_len))
        
        # Test encoder embedding
        enc_emb = self.model.encoder_embedding(tokens)
        assert enc_emb.shape == (batch_size, seq_len, self.d_model)
        
        # Test decoder embedding
        dec_emb = self.model.decoder_embedding(tokens)  
        assert dec_emb.shape == (batch_size, seq_len, self.d_model)
        
    def test_positional_encoding_shape(self):
        """Test positional encoding preserves shape."""
        batch_size, seq_len = 4, 10
        x = torch.randn(batch_size, seq_len, self.d_model)
        
        pos_encoded = self.model.positional_encoding(x)
        assert pos_encoded.shape == x.shape
        
    def test_encoder_shapes(self):
        """Test encoder output shapes."""
        batch_size, src_len = 4, 12
        src = torch.randint(1, self.vocab_size, (batch_size, src_len))
        src_mask = create_padding_mask(src)
        
        encoder_output, attention_weights = self.model.encode(src, src_mask)
        
        # Check encoder output shape
        assert encoder_output.shape == (batch_size, src_len, self.d_model)
        
        # Check attention weights shape
        assert len(attention_weights) == 2  # 2 encoder layers
        for attn_w in attention_weights:
            assert attn_w.shape == (batch_size, self.n_heads, src_len, src_len)
            
    def test_decoder_shapes(self):
        """Test decoder output shapes."""
        batch_size, src_len, tgt_len = 4, 12, 8
        src = torch.randint(1, self.vocab_size, (batch_size, src_len))
        tgt = torch.randint(1, self.vocab_size, (batch_size, tgt_len))
        
        src_mask = create_padding_mask(src)
        tgt_mask = create_combined_mask(tgt)
        
        # Encode first
        encoder_output, _ = self.model.encode(src, src_mask)
        
        # Then decode
        decoder_output, self_attn_weights, cross_attn_weights = self.model.decode(
            tgt, encoder_output, tgt_mask, src_mask
        )
        
        # Check decoder output shape
        assert decoder_output.shape == (batch_size, tgt_len, self.d_model)
        
        # Check self-attention weights
        assert len(self_attn_weights) == 2  # 2 decoder layers
        for attn_w in self_attn_weights:
            assert attn_w.shape == (batch_size, self.n_heads, tgt_len, tgt_len)
            
        # Check cross-attention weights
        assert len(cross_attn_weights) == 2  # 2 decoder layers
        for attn_w in cross_attn_weights:
            assert attn_w.shape == (batch_size, self.n_heads, tgt_len, src_len)
            
    def test_full_forward_shapes(self):
        """Test complete forward pass shapes."""
        batch_size, src_len, tgt_len = 4, 12, 8
        src = torch.randint(1, self.vocab_size, (batch_size, src_len))
        tgt = torch.randint(1, self.vocab_size, (batch_size, tgt_len))
        
        src_mask = create_padding_mask(src)
        tgt_mask = create_combined_mask(tgt)
        
        output, attention_weights = self.model(src, tgt, src_mask, tgt_mask)
        
        # Check output shape (logits over vocabulary)
        assert output.shape == (batch_size, tgt_len, self.vocab_size)
        
        # Check attention weights structure
        assert 'encoder_attention' in attention_weights
        assert 'decoder_self_attention' in attention_weights
        assert 'decoder_cross_attention' in attention_weights
        
    def test_attention_layer_shapes(self):
        """Test individual attention layer shapes."""
        batch_size, seq_len = 4, 10
        x = torch.randn(batch_size, seq_len, self.d_model)
        
        mha = MultiHeadAttention(self.d_model, self.n_heads)
        output, attention_weights = mha(x, x, x)
        
        assert output.shape == (batch_size, seq_len, self.d_model)
        assert attention_weights.shape == (batch_size, self.n_heads, seq_len, seq_len)
        
    def test_encoder_layer_shapes(self):
        """Test encoder layer shapes."""
        batch_size, seq_len = 4, 10
        x = torch.randn(batch_size, seq_len, self.d_model)
        
        encoder_layer = EncoderLayer(self.d_model, self.n_heads, self.d_ff)
        output, attention_weights = encoder_layer(x)
        
        assert output.shape == (batch_size, seq_len, self.d_model)
        assert attention_weights.shape == (batch_size, self.n_heads, seq_len, seq_len)
        
    def test_decoder_layer_shapes(self):
        """Test decoder layer shapes."""
        batch_size, tgt_len, src_len = 4, 8, 10
        x = torch.randn(batch_size, tgt_len, self.d_model)
        encoder_output = torch.randn(batch_size, src_len, self.d_model)
        
        decoder_layer = DecoderLayer(self.d_model, self.n_heads, self.d_ff)
        output, self_attn, cross_attn = decoder_layer(x, encoder_output)
        
        assert output.shape == (batch_size, tgt_len, self.d_model)
        assert self_attn.shape == (batch_size, self.n_heads, tgt_len, tgt_len)
        assert cross_attn.shape == (batch_size, self.n_heads, tgt_len, src_len)


class TestToyDataset:
    """Test toy dataset generation."""
    
    def setup_method(self):
        """Setup dataset."""
        self.dataset = ToyDataset(vocab_size=100, seq_len_range=(5, 15))
        
    def test_copy_task_shapes(self):
        """Test copy task generation shapes."""
        batch_size = 8
        src, tgt = self.dataset.generate_copy_task(batch_size)
        
        assert src.shape[0] == batch_size
        assert tgt.shape[0] == batch_size
        assert src.shape[1] == tgt.shape[1]  # Same max length
        
    def test_reverse_task_shapes(self):
        """Test reverse task generation shapes."""
        batch_size = 8
        src, tgt = self.dataset.generate_reverse_task(batch_size)
        
        assert src.shape[0] == batch_size
        assert tgt.shape[0] == batch_size
        assert src.shape[1] == tgt.shape[1]  # Same max length
        
    def test_copy_task_correctness(self):
        """Test copy task generates correct sequences."""
        batch_size = 1
        src, tgt = self.dataset.generate_copy_task(batch_size)
        
        src_seq = src[0]
        tgt_seq = tgt[0]
        
        # Find actual sequence lengths (excluding padding/special tokens)
        src_len = (src_seq != 0).sum().item() - 1  # -1 for EOS
        tgt_len = (tgt_seq != 0).sum().item() - 2  # -2 for SOS/EOS
        
        # Extract actual sequences
        src_actual = src_seq[:src_len]
        tgt_actual = tgt_seq[1:tgt_len+1]  # Remove SOS
        
        assert torch.equal(src_actual, tgt_actual), "Copy task should produce identical sequences"
        
    def test_reverse_task_correctness(self):
        """Test reverse task generates correctly reversed sequences."""
        batch_size = 1
        src, tgt = self.dataset.generate_reverse_task(batch_size)
        
        src_seq = src[0]
        tgt_seq = tgt[0]
        
        # Find actual sequence lengths (excluding padding/special tokens)
        src_len = (src_seq != 0).sum().item() - 1  # -1 for EOS
        tgt_len = (tgt_seq != 0).sum().item() - 2  # -2 for SOS/EOS
        
        # Extract actual sequences
        src_actual = src_seq[:src_len]
        tgt_actual = tgt_seq[1:tgt_len+1]  # Remove SOS
        
        # Check if target is reverse of source
        expected_reversed = torch.flip(src_actual, [0])
        assert torch.equal(tgt_actual, expected_reversed), "Reverse task should produce reversed sequences"


class TestLabelSmoothing:
    """Test label smoothing loss."""
    
    def test_label_smoothing_shapes(self):
        """Test label smoothing loss computation."""
        vocab_size = 100
        batch_size, seq_len = 4, 10
        
        criterion = LabelSmoothingLoss(vocab_size, smoothing=0.1)
        
        # Random predictions and targets
        pred = torch.randn(batch_size, seq_len, vocab_size)
        target = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        loss = criterion(pred, target)
        
        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Loss should be non-negative"


def run_shape_tests():
    """Run all shape tests manually."""
    print("Running Transformer shape tests...")
    
    # Test masks
    print("✓ Testing mask functions...")
    mask_tests = TestMasks()
    mask_tests.test_padding_mask_shape()
    mask_tests.test_padding_mask_values()
    mask_tests.test_causal_mask_shape()
    mask_tests.test_causal_mask_values()
    mask_tests.test_combined_mask()
    
    # Test transformer shapes
    print("✓ Testing Transformer shapes...")
    transformer_tests = TestTransformerShapes()
    transformer_tests.setup_method()
    transformer_tests.test_embedding_shapes()
    transformer_tests.test_positional_encoding_shape()
    transformer_tests.test_encoder_shapes()
    transformer_tests.test_decoder_shapes()
    transformer_tests.test_full_forward_shapes()
    transformer_tests.test_attention_layer_shapes()
    transformer_tests.test_encoder_layer_shapes()
    transformer_tests.test_decoder_layer_shapes()
    
    # Test dataset
    print("✓ Testing toy dataset...")
    dataset_tests = TestToyDataset()
    dataset_tests.setup_method()
    dataset_tests.test_copy_task_shapes()
    dataset_tests.test_reverse_task_shapes()
    dataset_tests.test_copy_task_correctness()
    dataset_tests.test_reverse_task_correctness()
    
    # Test label smoothing
    print("✓ Testing label smoothing...")
    ls_tests = TestLabelSmoothing()
    ls_tests.test_label_smoothing_shapes()
    
    print("All tests passed! ✅")


if __name__ == '__main__':
    # Run tests manually if pytest not available
    try:
        import pytest
        print("Run with: pytest tests/test_shapes.py -v")
    except ImportError:
        print("pytest not available, running tests manually...")
        run_shape_tests()