"""
Vision Transformer Demo Script

Quick demonstration of ViT models and their capabilities.
Tests model creation, forward pass, and provides usage examples.
"""

import torch
import torch.nn as nn
import time
import numpy as np
from models import vit_tiny, vit_small, vit_base
from utils import model_info, accuracy, LabelSmoothingCrossEntropy, mixup_data


def test_model_creation():
    """Test creation of all ViT variants."""
    print("🔧 Testing ViT Model Creation")
    print("=" * 50)
    
    models = {
        'ViT-Tiny': vit_tiny(),
        'ViT-Small': vit_small(),
        'ViT-Base': vit_base()
    }
    
    for name, model in models.items():
        param_count = sum(p.numel() for p in model.parameters())
        print(f"{name:12} | Parameters: {param_count:,}")
        
        # Test forward pass
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            output = model(x)
            print(f"{'':12} | Input: {x.shape} -> Output: {output.shape}")
    
    return models


def test_patch_embedding():
    """Test patch embedding component."""
    print("\n🧩 Testing Patch Embedding")
    print("=" * 50)
    
    from models.vit import PatchEmbedding
    
    # Create patch embedding layer
    patch_embed = PatchEmbedding(img_size=32, patch_size=4, in_channels=3, embed_dim=192)
    
    x = torch.randn(4, 3, 32, 32)  # CIFAR-10 batch
    patches = patch_embed(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Patches shape: {patches.shape}")
    print(f"Number of patches: {patch_embed.num_patches}")
    print(f"Patches per side: {int(np.sqrt(patch_embed.num_patches))}")
    
    return patch_embed


def test_attention_mechanism():
    """Test multi-head self-attention."""
    print("\n👁️ Testing Multi-Head Self-Attention")
    print("=" * 50)
    
    from models.vit import MultiHeadSelfAttention
    
    # Create attention layer
    attn = MultiHeadSelfAttention(embed_dim=192, num_heads=3)
    
    # Test input: batch_size=2, seq_len=65 (64 patches + 1 CLS), embed_dim=192
    x = torch.randn(2, 65, 192)
    
    # Forward pass
    output = attn(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test with attention return
    output_with_attn, attention = attn(x, return_attention=True)
    print(f"Attention shape: {attention.shape}")  # (batch, heads, seq_len, seq_len)
    
    return attn


def test_transformer_block():
    """Test transformer block."""
    print("\n🏗️ Testing Transformer Block")
    print("=" * 50)
    
    from models.vit import TransformerBlock
    
    # Create transformer block
    block = TransformerBlock(embed_dim=192, num_heads=3, drop_path=0.1)
    
    x = torch.randn(2, 65, 192)  # (batch, seq_len, embed_dim)
    
    # Forward pass
    output = block(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test with attention return
    output_with_attn, attention = block(x, return_attention=True)
    print(f"Attention shape: {attention.shape}")
    
    return block


def test_complete_vit():
    """Test complete ViT model with various features."""
    print("\n🤖 Testing Complete ViT Model")
    print("=" * 50)
    
    # Create ViT-Tiny model
    model = vit_tiny(num_classes=10, dropout=0.1, drop_path_rate=0.1)
    model.eval()
    
    # Test batch of CIFAR-10 images
    x = torch.randn(8, 3, 32, 32)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Test feature extraction
        features = model.forward_features(x)
        print(f"Features shape: {features.shape}")  # (batch, seq_len, embed_dim)
        
        # Test attention maps
        logits, attention_maps = model.forward_with_attention(x)
        print(f"Number of attention layers: {len(attention_maps)}")
        print(f"Attention map shape: {attention_maps[0].shape}")  # (batch, heads, seq_len, seq_len)
    
    return model


def test_training_components():
    """Test training-related components."""
    print("\n🎯 Testing Training Components")
    print("=" * 50)
    
    # Create model and sample data
    model = vit_tiny()
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 10, (16,))
    
    print(f"Batch shape: {x.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Test forward pass
    output = model(x)
    print(f"Model output shape: {output.shape}")
    
    # Test accuracy computation
    acc1, acc5 = accuracy(output, y, topk=(1, 5))
    print(f"Random accuracy - Top1: {acc1[0]:.2f}%, Top5: {acc5[0]:.2f}%")
    
    # Test label smoothing loss
    criterion_smooth = LabelSmoothingCrossEntropy(smoothing=0.1)
    criterion_regular = nn.CrossEntropyLoss()
    
    loss_smooth = criterion_smooth(output, y)
    loss_regular = criterion_regular(output, y)
    
    print(f"Regular CE loss: {loss_regular:.4f}")
    print(f"Label smoothing loss: {loss_smooth:.4f}")
    
    # Test mixup
    mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)
    print(f"Mixup lambda: {lam:.4f}")
    print(f"Mixed input shape: {mixed_x.shape}")
    
    return model


def benchmark_models():
    """Benchmark different ViT variants."""
    print("\n⏱️ Model Benchmarking")
    print("=" * 50)
    
    models = {
        'ViT-Tiny': vit_tiny(),
        'ViT-Small': vit_small(), 
        'ViT-Base': vit_base()
    }
    
    batch_size = 16
    x = torch.randn(batch_size, 3, 32, 32)
    
    results = []
    
    for name, model in models.items():
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(x)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(50):
                start = time.time()
                _ = model(x)
                times.append(time.time() - start)
        
        avg_time = np.mean(times) * 1000  # Convert to ms
        param_count = sum(p.numel() for p in model.parameters())
        
        results.append((name, param_count, avg_time))
        print(f"{name:10} | {param_count:8,} params | {avg_time:6.2f}ms/batch")
    
    return results


def test_stochastic_depth():
    """Test stochastic depth (DropPath) functionality."""
    print("\n🌊 Testing Stochastic Depth")
    print("=" * 50)
    
    from models.droppath import DropPath, make_drop_path_schedule
    
    # Test DropPath layer
    drop_path = DropPath(drop_prob=0.2)
    x = torch.randn(4, 65, 192)
    
    # Training mode
    drop_path.train()
    output_train = drop_path(x)
    print(f"Training mode - Input: {x.shape}, Output: {output_train.shape}")
    print(f"Training mode - Input mean: {x.mean():.4f}, Output mean: {output_train.mean():.4f}")
    
    # Evaluation mode
    drop_path.eval()
    output_eval = drop_path(x)
    print(f"Eval mode - Input equals output: {torch.equal(x, output_eval)}")
    
    # Test drop path schedule
    drop_rates = make_drop_path_schedule(0.2, 6)  # For ViT-Tiny with 6 layers
    print(f"Drop path schedule: {[f'{rate:.3f}' for rate in drop_rates]}")
    
    return drop_path


def demonstrate_training_step():
    """Demonstrate a single training step."""
    print("\n🎓 Training Step Demonstration")
    print("=" * 50)
    
    # Create model and optimizer
    model = vit_tiny(num_classes=10, dropout=0.1, drop_path_rate=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    
    # Create sample batch
    x = torch.randn(32, 3, 32, 32)
    y = torch.randint(0, 10, (32,))
    
    print(f"Batch shape: {x.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    output = model(x)
    loss = criterion(output, y)
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    
    # Compute accuracy
    acc1, acc5 = accuracy(output, y, topk=(1, 5))
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Top-1 Accuracy: {acc1[0]:.2f}%")
    print(f"Top-5 Accuracy: {acc5[0]:.2f}%")
    
    print("Training step completed successfully! ✅")


def main():
    """Run all demonstrations."""
    print("🚀 Vision Transformer Implementation Demo")
    print("=" * 70)
    
    try:
        # Test model creation
        models = test_model_creation()
        
        # Test individual components
        patch_embed = test_patch_embedding()
        attention = test_attention_mechanism()
        block = test_transformer_block()
        
        # Test complete model
        model = test_complete_vit()
        
        # Test training components
        test_training_components()
        
        # Test stochastic depth
        test_stochastic_depth()
        
        # Benchmark models
        benchmark_models()
        
        # Demonstrate training step
        demonstrate_training_step()
        
        print("\n" + "=" * 70)
        print("✅ All tests passed! ViT implementation is working correctly.")
        print("\nQuick Start Commands:")
        print("  # Train ViT-Tiny on CIFAR-10:")
        print("  python train_cifar.py --model vit_tiny --epochs 100 --batch-size 128")
        print("  ")
        print("  # Visualize attention maps:")
        print("  python visualize_attention.py")
        print("  ")
        print("  # Debug mode (quick test):")
        print("  python train_cifar.py --debug --epochs 3")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()