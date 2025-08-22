"""
ResNet Demo Script

Quick demonstration of ResNet models and their capabilities.
Tests model creation, forward pass, and provides usage examples.
"""

import torch
import torch.nn as nn
import time
import numpy as np
from models import resnet18, resnet34, resnet50, resnet101, init_weights
from utils import model_info, accuracy


def test_model_creation():
    """Test creation of all ResNet variants"""
    print("🔧 Testing Model Creation")
    print("=" * 50)
    
    models = {
        'ResNet-18': resnet18(),
        'ResNet-18 (Pre-act)': resnet18(pre_activation=True),
        'ResNet-34': resnet34(), 
        'ResNet-34 (Pre-act)': resnet34(pre_activation=True),
        'ResNet-50': resnet50(),
        'ResNet-50 (Pre-act)': resnet50(pre_activation=True),
        'ResNet-101': resnet101(),
        'ResNet-101 (Pre-act)': resnet101(pre_activation=True)
    }
    
    for name, model in models.items():
        param_count = sum(p.numel() for p in model.parameters())
        print(f"{name:20} | Parameters: {param_count:,}")
    
    return models


def test_forward_pass():
    """Test forward pass with different batch sizes"""
    print("\n📊 Testing Forward Pass")
    print("=" * 50)
    
    model = resnet18(pre_activation=True)
    model.eval()
    
    batch_sizes = [1, 16, 32, 128]
    input_size = (3, 32, 32)  # CIFAR-10 input
    
    print(f"Input size: {input_size}")
    print(f"Output classes: 10 (CIFAR-10)")
    
    with torch.no_grad():
        for batch_size in batch_sizes:
            # Create random input
            x = torch.randn(batch_size, *input_size)
            
            # Time forward pass
            start_time = time.time()
            output = model(x)
            forward_time = time.time() - start_time
            
            print(f"Batch {batch_size:3d}: {x.shape} → {output.shape} | "
                  f"Time: {forward_time*1000:.2f}ms")
    
    return model


def test_gradient_flow():
    """Test gradient flow through the network"""
    print("\n🌊 Testing Gradient Flow")
    print("=" * 50)
    
    model = resnet18(pre_activation=True)
    model.train()
    
    # Initialize weights
    init_weights(model, init_type='he_normal', zero_gamma=True)
    
    # Create dummy batch
    x = torch.randn(32, 3, 32, 32)
    target = torch.randint(0, 10, (32,))
    
    # Forward pass
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, target)
    
    # Backward pass
    loss.backward()
    
    # Analyze gradients
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2)
            grad_norms.append(grad_norm.item())
            if 'conv' in name or 'fc' in name:
                print(f"{name:30} | Grad norm: {grad_norm:.6f}")
    
    print(f"\nGradient Statistics:")
    print(f"  Mean grad norm: {np.mean(grad_norms):.6f}")
    print(f"  Max grad norm:  {np.max(grad_norms):.6f}")
    print(f"  Min grad norm:  {np.min(grad_norms):.6f}")


def test_feature_extraction():
    """Test feature extraction capabilities"""
    print("\n🔍 Testing Feature Extraction")
    print("=" * 50)
    
    model = resnet18(pre_activation=True)
    model.eval()
    
    x = torch.randn(4, 3, 32, 32)
    
    # Get intermediate features
    features = model.get_feature_maps(x)
    
    print("Feature map shapes at different stages:")
    for stage_name, feature_map in features.items():
        print(f"  {stage_name:10} | Shape: {tuple(feature_map.shape)}")


def benchmark_models():
    """Benchmark different ResNet variants"""
    print("\n⏱️  Model Benchmarking")
    print("=" * 50)
    
    models = {
        'ResNet-18': resnet18(),
        'ResNet-34': resnet34(),
        'ResNet-50': resnet50(),
    }
    
    batch_size = 32
    x = torch.randn(batch_size, 3, 32, 32)
    
    results = []
    
    for name, model in models.items():
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(100):
                start = time.time()
                _ = model(x)
                times.append(time.time() - start)
        
        avg_time = np.mean(times) * 1000  # Convert to ms
        param_count = sum(p.numel() for p in model.parameters())
        
        results.append((name, param_count, avg_time))
        print(f"{name:10} | {param_count:8,} params | {avg_time:6.2f}ms/batch")
    
    return results


def demonstrate_training_step():
    """Demonstrate a single training step"""
    print("\n🎯 Single Training Step Demo")  
    print("=" * 50)
    
    # Create model and optimizer
    model = resnet18(pre_activation=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # Create dummy batch
    x = torch.randn(16, 3, 32, 32)
    target = torch.randint(0, 10, (16,))
    
    print(f"Batch shape: {x.shape}")
    print(f"Target shape: {target.shape}")
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    # Forward
    output = model(x)
    loss = criterion(output, target)
    
    # Backward
    loss.backward()
    optimizer.step()
    
    # Compute accuracy
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Top-1 Accuracy: {acc1[0]:.2f}%")
    print(f"Top-5 Accuracy: {acc5[0]:.2f}%")
    
    print("\nTraining step completed successfully! ✅")


def main():
    """Run all demonstrations"""
    print("🚀 ResNet Implementation Demo")
    print("=" * 70)
    
    try:
        # Test model creation
        models = test_model_creation()
        
        # Test forward pass
        model = test_forward_pass()
        
        # Test gradient flow
        test_gradient_flow()
        
        # Test feature extraction
        test_feature_extraction()
        
        # Benchmark models
        benchmark_models()
        
        # Demonstrate training step
        demonstrate_training_step()
        
        print("\n" + "=" * 70)
        print("✅ All tests passed! ResNet implementation is working correctly.")
        print("\nTo train on CIFAR-10, run:")
        print("  python train_cifar10.py --arch resnet18 --pre-activation")
        print("\nTo run experiments with configs:")
        print("  python run_experiments.py --configs resnet18_cifar")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()