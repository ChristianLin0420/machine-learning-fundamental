"""
Vision Transformer (ViT) Implementation

Complete implementation of Vision Transformer from "An Image is Worth 16x16 Words"
with modern improvements including stochastic depth, proper initialization, and
CIFAR-10 specific configurations.

Key Components:
- PatchEmbedding: Convert image patches to token embeddings
- MultiHeadSelfAttention: Core attention mechanism
- MLP: Feed-forward network with GELU activation
- TransformerBlock: Attention + MLP with pre-norm and residual connections
- VisionTransformer: Complete model with classification head

Reference: Dosovitskiy et al. (2020) "An Image is Worth 16x16 Words"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple
from .droppath import DropPath, make_drop_path_schedule


class PatchEmbedding(nn.Module):
    """
    Convert 2D image patches into 1D token embeddings.
    
    For CIFAR-10 (32x32), with patch_size=4, we get 8x8=64 patches.
    Each patch becomes a token with dimension embed_dim.
    """
    
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super(PatchEmbedding, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Convolution to extract patches and embed them
        # Each patch becomes a token through linear projection (implemented as conv)
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
        # Layer normalization (optional, often helps with training)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Token embeddings (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}x{W}) doesn't match model ({self.img_size}x{self.img_size})"
        
        # Extract patches and embed: (B, C, H, W) -> (B, embed_dim, H//P, W//P)
        x = self.proj(x)
        
        # Flatten spatial dimensions: (B, embed_dim, H//P, W//P) -> (B, embed_dim, num_patches)
        x = x.flatten(2)
        
        # Transpose to get token sequence: (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        
        # Apply layer normalization
        x = self.norm(x)
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism for Vision Transformer.
    
    Implements scaled dot-product attention with multiple heads:
    Attention(Q,K,V) = softmax(QK^T/√d)V
    """
    
    def __init__(self, embed_dim=192, num_heads=3, attn_dropout=0.1, proj_dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5  # 1/sqrt(head_dim)
        
        # Linear projections for Q, K, V (combined for efficiency)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        
        # Attention dropout
        self.attn_dropout = nn.Dropout(attn_dropout)
        
        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(proj_dropout)
        
    def forward(self, x, return_attention=False):
        """
        Args:
            x: Input tokens (B, N, D) where N = num_patches + 1 (including CLS token)
            return_attention: Whether to return attention weights
            
        Returns:
            output: Transformed tokens (B, N, D)
            attention: Attention weights if return_attention=True (B, H, N, N)
        """
        B, N, D = x.shape
        
        # Generate Q, K, V through linear projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, H, N, head_dim)
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)  # (B, N, D)
        
        # Output projection
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        if return_attention:
            return x, attn
        return x


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Feed Forward Network) for Transformer.
    
    Standard architecture: Linear -> GELU -> Dropout -> Linear -> Dropout
    """
    
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 activation=nn.GELU, dropout=0.1):
        super(MLP, self).__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = activation()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer Block with Multi-Head Self-Attention and MLP.
    
    Architecture: LayerNorm -> Attention -> Residual -> LayerNorm -> MLP -> Residual
    Uses pre-normalization (LayerNorm before sublayers) for better training stability.
    """
    
    def __init__(self, embed_dim=192, num_heads=3, mlp_ratio=4.0, 
                 dropout=0.1, attn_dropout=0.1, drop_path=0.0):
        super(TransformerBlock, self).__init__()
        
        # Layer normalization (pre-norm)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Multi-head self-attention
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            proj_dropout=dropout
        )
        
        # MLP (Feed Forward)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=mlp_hidden_dim,
            dropout=dropout
        )
        
        # Stochastic depth (DropPath)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x, return_attention=False):
        """
        Args:
            x: Input tokens (B, N, D)
            return_attention: Whether to return attention weights
            
        Returns:
            output: Transformed tokens (B, N, D)
            attention: Attention weights if return_attention=True
        """
        # Self-attention with residual connection
        if return_attention:
            attn_out, attention = self.attn(self.norm1(x), return_attention=True)
            x = x + self.drop_path(attn_out)
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            attention = None
            
        # MLP with residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        if return_attention:
            return x, attention
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) for image classification.
    
    Complete ViT implementation with:
    - Patch embedding
    - Learnable positional embeddings
    - CLS token for classification
    - Transformer blocks with stochastic depth
    - Classification head
    """
    
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 embed_dim=192, depth=6, num_heads=3, mlp_ratio=4.0,
                 dropout=0.1, attn_dropout=0.1, drop_path_rate=0.1,
                 representation_size=None):
        super(VisionTransformer, self).__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size, 
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # CLS token (learnable classification token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embeddings (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(dropout)
        
        # Transformer blocks with stochastic depth
        drop_path_rates = make_drop_path_schedule(drop_path_rate, depth)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
                drop_path=drop_path_rates[i]
            )
            for i in range(depth)
        ])
        
        # Layer normalization before classifier
        self.norm = nn.LayerNorm(embed_dim)
        
        # Representation layer (optional, used in some variants)
        if representation_size:
            self.pre_logits = nn.Sequential(
                nn.Linear(embed_dim, representation_size),
                nn.Tanh()
            )
            classifier_input_dim = representation_size
        else:
            self.pre_logits = nn.Identity()
            classifier_input_dim = embed_dim
            
        # Classification head
        self.head = nn.Linear(classifier_input_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights following ViT paper recommendations."""
        
        # Initialize CLS token and positional embeddings
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Initialize other parameters
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
        self.apply(_init)
        
        # Initialize classification head with smaller std
        if hasattr(self.head, 'weight'):
            nn.init.trunc_normal_(self.head.weight, std=0.02)
            
    def forward_features(self, x):
        """
        Forward pass through patch embedding and transformer blocks.
        
        Args:
            x: Input images (B, C, H, W)
            
        Returns:
            features: Feature representations (B, N+1, D)
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, N, D)
        
        # Add CLS token
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat((cls_token, x), dim=1)  # (B, N+1, D)
        
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Final layer normalization
        x = self.norm(x)
        
        return x
    
    def forward(self, x, return_attention=False):
        """
        Complete forward pass.
        
        Args:
            x: Input images (B, C, H, W)
            return_attention: Whether to return attention maps
            
        Returns:
            logits: Classification logits (B, num_classes)
            attention: Attention maps if return_attention=True
        """
        if return_attention:
            return self.forward_with_attention(x)
            
        # Feature extraction
        x = self.forward_features(x)
        
        # Classification head (use CLS token)
        x = self.pre_logits(x[:, 0])  # (B, D) - CLS token features
        x = self.head(x)  # (B, num_classes)
        
        return x
    
    def forward_with_attention(self, x):
        """Forward pass that returns attention maps from all layers."""
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add CLS token and positional embeddings
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_dropout(x)
        
        # Collect attention maps from all blocks
        attention_maps = []
        for block in self.blocks:
            x, attention = block(x, return_attention=True)
            attention_maps.append(attention)
            
        # Final normalization and classification
        x = self.norm(x)
        x = self.pre_logits(x[:, 0])
        logits = self.head(x)
        
        return logits, attention_maps
    
    def get_attention_maps(self, x, layer_idx=None):
        """
        Get attention maps for visualization.
        
        Args:
            x: Input images (B, C, H, W)
            layer_idx: Specific layer index (if None, returns all layers)
            
        Returns:
            attention_maps: Attention weights
        """
        _, attention_maps = self.forward_with_attention(x)
        
        if layer_idx is not None:
            return attention_maps[layer_idx]
        return attention_maps


# Model variants for different scales
def vit_tiny(num_classes=10, **kwargs):
    """ViT-Tiny for CIFAR-10"""
    model = VisionTransformer(
        img_size=32, patch_size=4, embed_dim=192, depth=6, num_heads=3,
        mlp_ratio=4.0, num_classes=num_classes, **kwargs
    )
    return model


def vit_small(num_classes=10, **kwargs):
    """ViT-Small for CIFAR-10"""
    model = VisionTransformer(
        img_size=32, patch_size=4, embed_dim=384, depth=8, num_heads=6,
        mlp_ratio=4.0, num_classes=num_classes, **kwargs
    )
    return model


def vit_base(num_classes=10, **kwargs):
    """ViT-Base for CIFAR-10 (adapted from ImageNet)"""
    model = VisionTransformer(
        img_size=32, patch_size=4, embed_dim=768, depth=12, num_heads=12,
        mlp_ratio=4.0, num_classes=num_classes, **kwargs
    )
    return model


def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model, input_size=(1, 3, 32, 32)):
    """Print model summary."""
    total_params = count_parameters(model)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Total trainable parameters: {total_params:,}")
    
    # Test forward pass
    with torch.no_grad():
        x = torch.randn(input_size)
        try:
            output = model(x)
            print(f"Input shape: {x.shape}")
            print(f"Output shape: {output.shape}")
        except Exception as e:
            print(f"Forward pass failed: {e}")


if __name__ == "__main__":
    # Test ViT implementation
    print("Testing Vision Transformer implementation...")
    print("=" * 60)
    
    # Test different model variants
    models = {
        'ViT-Tiny': vit_tiny(),
        'ViT-Small': vit_small(),
        'ViT-Base': vit_base()
    }
    
    for name, model in models.items():
        print(f"\n{name}:")
        print(f"  Parameters: {count_parameters(model):,}")
        
        # Test forward pass
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            output = model(x)
            print(f"  Input: {x.shape} -> Output: {output.shape}")
            
        # Test attention maps
        with torch.no_grad():
            _, attention_maps = model.forward_with_attention(x[:1])
            print(f"  Attention maps: {len(attention_maps)} layers, "
                  f"shape: {attention_maps[0].shape}")
    
    print("\nViT implementation test completed! ✅")