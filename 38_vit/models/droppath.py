"""
Stochastic Depth (DropPath) Implementation

Stochastic depth randomly drops residual branches during training,
improving regularization and reducing overfitting in deep networks.

Reference: Huang et al. (2016) "Deep Networks with Stochastic Depth"
"""

import torch
import torch.nn as nn


def drop_path(x, drop_prob=0., training=False, scale_by_keep=True):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    
    Args:
        x: Input tensor
        drop_prob: Probability of dropping path
        training: Whether in training mode
        scale_by_keep: Whether to scale by keep probability for expected value preservation
        
    Returns:
        Output tensor with stochastic depth applied
    """
    if drop_prob == 0. or not training:
        return x
        
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
        
    return x * random_tensor


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    
    This implementation supports:
    - Stochastic depth during training
    - Identity during inference
    - Proper scaling to maintain expected output
    """
    
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
        
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3) if self.drop_prob else None}'


def make_drop_path_schedule(drop_path_rate, depth):
    """
    Create a drop path schedule that linearly increases with depth.
    
    Args:
        drop_path_rate: Maximum drop path rate
        depth: Number of transformer blocks
        
    Returns:
        List of drop path rates for each block
    """
    if drop_path_rate <= 0:
        return [0.0] * depth
        
    # Linear scaling: earlier layers have lower drop rates
    drop_rates = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
    return drop_rates


class StochasticDepth(nn.Module):
    """
    Alternative implementation of stochastic depth with survival probability.
    
    This version directly uses survival probability (complementary to drop probability)
    and includes options for different sampling strategies.
    """
    
    def __init__(self, survival_prob=0.9, mode='batch'):
        """
        Args:
            survival_prob: Probability of keeping the layer (1 - drop_prob)
            mode: 'batch' for per-batch sampling, 'sample' for per-sample sampling
        """
        super(StochasticDepth, self).__init__()
        self.survival_prob = survival_prob
        self.mode = mode
        
    def forward(self, x):
        if not self.training or self.survival_prob == 1.0:
            return x
            
        if self.mode == 'batch':
            # Single random value for entire batch
            if torch.rand(1).item() < self.survival_prob:
                return x / self.survival_prob  # Scale for expected value
            else:
                return x.new_zeros(x.shape)
        else:
            # Per-sample random values
            batch_size = x.shape[0]
            shape = (batch_size,) + (1,) * (x.ndim - 1)
            random_tensor = x.new_empty(shape).bernoulli_(self.survival_prob)
            return x * random_tensor / self.survival_prob
    
    def extra_repr(self):
        return f'survival_prob={self.survival_prob}, mode={self.mode}'


# Utility function for creating drop path layers
def create_drop_path_layers(drop_path_rate, depth):
    """
    Create a list of DropPath layers with linearly increasing rates.
    
    Args:
        drop_path_rate: Maximum drop path rate
        depth: Number of layers
        
    Returns:
        List of DropPath modules
    """
    drop_rates = make_drop_path_schedule(drop_path_rate, depth)
    return [DropPath(drop_prob) for drop_prob in drop_rates]


if __name__ == "__main__":
    # Test DropPath implementation
    print("Testing DropPath implementation...")
    
    # Create test tensor
    x = torch.randn(8, 64, 196)  # (batch_size, dim, num_patches)
    
    # Test DropPath
    drop_path_layer = DropPath(drop_prob=0.1)
    
    # Training mode
    drop_path_layer.train()
    output_train = drop_path_layer(x)
    print(f"Training mode - Input shape: {x.shape}, Output shape: {output_train.shape}")
    print(f"Training mode - Input mean: {x.mean():.4f}, Output mean: {output_train.mean():.4f}")
    
    # Evaluation mode
    drop_path_layer.eval()
    output_eval = drop_path_layer(x)
    print(f"Eval mode - Input shape: {x.shape}, Output shape: {output_eval.shape}")
    print(f"Eval mode - Input mean: {x.mean():.4f}, Output mean: {output_eval.mean():.4f}")
    print(f"Eval mode - Input equals output: {torch.equal(x, output_eval)}")
    
    # Test drop path schedule
    drop_rates = make_drop_path_schedule(0.1, 12)
    print(f"\nDrop path schedule for depth=12, max_rate=0.1:")
    for i, rate in enumerate(drop_rates):
        print(f"  Layer {i}: {rate:.3f}")
    
    print("\nDropPath implementation test completed! ✅")