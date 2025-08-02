"""
PyTorch Tensor Basics
====================

This module demonstrates fundamental PyTorch tensor operations:
- Tensor creation and manipulation
- GPU acceleration with CUDA
- Matrix operations
- Tensor properties and methods
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Tuple, List

def demonstrate_tensor_creation():
    """Demonstrate various ways to create tensors"""
    print("=" * 60)
    print("TENSOR CREATION METHODS")
    print("=" * 60)
    
    # 1. From lists
    tensor_from_list = torch.tensor([1, 2, 3, 4, 5])
    print(f"From list: {tensor_from_list}")
    print(f"Shape: {tensor_from_list.shape}, dtype: {tensor_from_list.dtype}")
    
    # 2. From numpy arrays
    numpy_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
    tensor_from_numpy = torch.from_numpy(numpy_array)
    print(f"\nFrom NumPy: {tensor_from_numpy}")
    print(f"Shares memory: {tensor_from_numpy.data_ptr() == numpy_array.__array_interface__['data'][0]}")
    
    # 3. Random tensors
    random_tensor = torch.rand(3, 4)
    print(f"\nRandom tensor:\n{random_tensor}")
    
    # 4. Zeros and ones
    zeros_tensor = torch.zeros(2, 3)
    ones_tensor = torch.ones(2, 3)
    print(f"\nZeros:\n{zeros_tensor}")
    print(f"Ones:\n{ones_tensor}")
    
    # 5. Ranges
    range_tensor = torch.arange(0, 10, 2)
    linspace_tensor = torch.linspace(0, 1, 5)
    print(f"\nRange: {range_tensor}")
    print(f"Linspace: {linspace_tensor}")
    
    # 6. Identity matrix
    identity = torch.eye(3)
    print(f"\nIdentity matrix:\n{identity}")
    
    # 7. Normal distribution
    normal_tensor = torch.randn(2, 3)
    print(f"\nNormal distribution:\n{normal_tensor}")
    
    return {
        'from_list': tensor_from_list,
        'from_numpy': tensor_from_numpy,
        'random': random_tensor,
        'zeros': zeros_tensor,
        'ones': ones_tensor,
        'range': range_tensor,
        'linspace': linspace_tensor,
        'identity': identity,
        'normal': normal_tensor
    }

def demonstrate_tensor_operations():
    """Demonstrate tensor manipulation and operations"""
    print("\n" + "=" * 60)
    print("TENSOR OPERATIONS")
    print("=" * 60)
    
    # Create sample tensors
    a = torch.rand(3, 4)
    b = torch.rand(3, 4)
    
    print(f"Tensor a:\n{a}")
    print(f"Tensor b:\n{b}")
    
    # Basic arithmetic
    print(f"\nAddition: a + b =\n{a + b}")
    print(f"Element-wise multiplication: a * b =\n{a * b}")
    print(f"Division: a / b =\n{a / b}")
    
    # Mathematical functions
    print(f"\nSquare root of a:\n{torch.sqrt(a)}")
    print(f"Exponential of a:\n{torch.exp(a)}")
    print(f"Logarithm of a:\n{torch.log(a)}")
    
    # Aggregation operations
    print(f"\nSum of a: {torch.sum(a)}")
    print(f"Mean of a: {torch.mean(a)}")
    print(f"Max of a: {torch.max(a)}")
    print(f"Min of a: {torch.min(a)}")
    
    # Dimension-specific operations
    print(f"Sum along dim 0: {torch.sum(a, dim=0)}")
    print(f"Mean along dim 1: {torch.mean(a, dim=1)}")
    
    # Broadcasting
    scalar = 5
    print(f"\nBroadcasting a + {scalar}:\n{a + scalar}")
    
    vector = torch.tensor([1, 2, 3, 4])
    print(f"Broadcasting a + vector:\n{a + vector}")
    
    return a, b

def demonstrate_tensor_indexing():
    """Demonstrate tensor indexing and slicing"""
    print("\n" + "=" * 60)
    print("TENSOR INDEXING AND SLICING")
    print("=" * 60)
    
    # Create a 3D tensor
    tensor = torch.rand(4, 5, 6)
    print(f"Original tensor shape: {tensor.shape}")
    
    # Basic indexing
    print(f"First element: {tensor[0, 0, 0]}")
    print(f"First row: {tensor[0, 0, :]}")
    print(f"First matrix:\n{tensor[0]}")
    
    # Slicing
    print(f"Slice [1:3, 2:4, :]:\n{tensor[1:3, 2:4, :]}")
    
    # Advanced indexing
    indices = torch.tensor([0, 2])
    print(f"Advanced indexing with {indices}: shape {tensor[indices].shape}")
    
    # Boolean indexing
    mask = tensor > 0.5
    print(f"Boolean mask shape: {mask.shape}")
    print(f"Elements > 0.5: {tensor[mask][:10]}...")  # Show first 10
    
    # Negative indexing
    print(f"Last element: {tensor[-1, -1, -1]}")
    
    return tensor

def demonstrate_tensor_reshaping():
    """Demonstrate tensor reshaping operations"""
    print("\n" + "=" * 60)
    print("TENSOR RESHAPING")
    print("=" * 60)
    
    # Create original tensor
    original = torch.rand(2, 3, 4)
    print(f"Original shape: {original.shape}")
    
    # Reshape
    reshaped = original.reshape(6, 4)
    print(f"Reshaped to (6, 4): {reshaped.shape}")
    
    # View (shares memory)
    viewed = original.view(8, 3)
    print(f"Viewed as (8, 3): {viewed.shape}")
    print(f"Shares memory with original: {viewed.data_ptr() == original.data_ptr()}")
    
    # Flatten
    flattened = original.flatten()
    print(f"Flattened: {flattened.shape}")
    
    # Squeeze and unsqueeze
    squeezed = torch.rand(1, 5, 1, 3)
    print(f"Before squeeze: {squeezed.shape}")
    after_squeeze = squeezed.squeeze()
    print(f"After squeeze: {after_squeeze.shape}")
    
    unsqueezed = after_squeeze.unsqueeze(0)
    print(f"After unsqueeze(0): {unsqueezed.shape}")
    
    # Transpose
    transposed = original.transpose(0, 2)
    print(f"Transposed (0, 2): {transposed.shape}")
    
    # Permute
    permuted = original.permute(2, 0, 1)
    print(f"Permuted (2, 0, 1): {permuted.shape}")
    
    return original, reshaped, viewed, flattened

def demonstrate_matrix_operations():
    """Demonstrate matrix multiplication and linear algebra operations"""
    print("\n" + "=" * 60)
    print("MATRIX OPERATIONS")
    print("=" * 60)
    
    # Create matrices
    A = torch.rand(3, 4)
    B = torch.rand(4, 5)
    
    print(f"Matrix A shape: {A.shape}")
    print(f"Matrix B shape: {B.shape}")
    
    # Matrix multiplication
    C1 = torch.mm(A, B)          # 2D only
    C2 = torch.matmul(A, B)      # Works with batches
    C3 = A @ B                   # Operator overload
    
    print(f"A @ B shape: {C1.shape}")
    print(f"All methods equal: {torch.allclose(C1, C2) and torch.allclose(C2, C3)}")
    
    # Batch matrix multiplication
    batch_A = torch.rand(10, 3, 4)
    batch_B = torch.rand(10, 4, 5)
    batch_C = torch.bmm(batch_A, batch_B)
    print(f"Batch multiplication result shape: {batch_C.shape}")
    
    # Element-wise vs matrix multiplication
    square_A = torch.rand(3, 3)
    square_B = torch.rand(3, 3)
    
    element_wise = square_A * square_B
    matrix_mult = square_A @ square_B
    
    print(f"\nElement-wise multiplication shape: {element_wise.shape}")
    print(f"Matrix multiplication shape: {matrix_mult.shape}")
    print(f"Results are different: {not torch.allclose(element_wise, matrix_mult)}")
    
    # Linear algebra operations
    square_matrix = torch.rand(4, 4)
    
    # Determinant
    det = torch.det(square_matrix)
    print(f"\nDeterminant: {det}")
    
    # Eigenvalues and eigenvectors
    try:
        eigenvals, eigenvecs = torch.linalg.eig(square_matrix)
        print(f"Eigenvalues shape: {eigenvals.shape}")
        print(f"Eigenvectors shape: {eigenvecs.shape}")
        print(f"Eigenvalues (real part): {eigenvals.real[:3]}...")  # Show only first 3
        if torch.any(eigenvals.imag != 0):
            print(f"Has complex eigenvalues: True")
        else:
            print(f"All eigenvalues are real")
    except Exception as e:
        print(f"Eigenvalue computation failed: {e}")
    
    # SVD
    try:
        U, S, Vh = torch.linalg.svd(square_matrix)
        print(f"SVD - U: {U.shape}, S: {S.shape}, Vh: {Vh.shape}")
    except:
        # Fallback to old method if new one fails
        try:
            U, S, V = torch.svd(square_matrix)
            print(f"SVD - U: {U.shape}, S: {S.shape}, V: {V.shape}")
        except Exception as e:
            print(f"SVD computation failed: {e}")
    
    # Inverse
    try:
        # Use torch.linalg.inv if available, fallback to torch.inverse
        try:
            inv = torch.linalg.inv(square_matrix)
        except:
            inv = torch.inverse(square_matrix)
        
        print(f"Inverse shape: {inv.shape}")
        
        # Verify A * A^-1 = I
        identity_check = square_matrix @ inv
        is_identity = torch.allclose(identity_check, torch.eye(4), atol=1e-6)
        print(f"A @ A^-1 ≈ I: {is_identity}")
    except Exception as e:
        print(f"Matrix is not invertible or computation failed: {e}")
    
    return A, B, C1

def check_gpu_availability():
    """Check CUDA availability and GPU information"""
    print("\n" + "=" * 60)
    print("GPU AVAILABILITY CHECK")
    print("=" * 60)
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")
        
        current_device = torch.cuda.current_device()
        print(f"Current device: {current_device}")
        
        # Memory info
        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        memory_cached = torch.cuda.memory_reserved() / 1024**2      # MB
        print(f"Memory allocated: {memory_allocated:.2f} MB")
        print(f"Memory cached: {memory_cached:.2f} MB")
    else:
        print("CUDA not available. Using CPU.")
    
    # MPS (Apple Silicon) check
    mps_available = torch.backends.mps.is_available()
    print(f"MPS (Apple Silicon) available: {mps_available}")
    
    return cuda_available, mps_available

def demonstrate_gpu_operations():
    """Demonstrate GPU tensor operations"""
    print("\n" + "=" * 60)
    print("GPU OPERATIONS")
    print("=" * 60)
    
    cuda_available, mps_available = check_gpu_availability()
    
    # Determine device
    if cuda_available:
        device = torch.device("cuda")
    elif mps_available:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # Create tensors on CPU
    cpu_tensor = torch.rand(1000, 1000)
    print(f"CPU tensor device: {cpu_tensor.device}")
    
    # Move to GPU/device
    gpu_tensor = cpu_tensor.to(device)
    print(f"GPU tensor device: {gpu_tensor.device}")
    
    # Create tensor directly on GPU
    direct_gpu = torch.rand(1000, 1000, device=device)
    print(f"Direct GPU tensor device: {direct_gpu.device}")
    
    # Performance comparison
    def time_operation(tensor1, tensor2, operation_name):
        start_time = time.time()
        result = tensor1 @ tensor2
        if device.type == "cuda":
            torch.cuda.synchronize()  # Wait for GPU operations to complete
        end_time = time.time()
        return end_time - start_time, result
    
    # CPU computation
    cpu_tensor2 = torch.rand(1000, 1000)
    cpu_time, cpu_result = time_operation(cpu_tensor, cpu_tensor2, "CPU")
    
    # GPU computation
    gpu_tensor2 = torch.rand(1000, 1000, device=device)
    gpu_time, gpu_result = time_operation(gpu_tensor, gpu_tensor2, "GPU")
    
    print(f"\nPerformance comparison (1000x1000 matrix multiplication):")
    print(f"CPU time: {cpu_time:.4f} seconds")
    print(f"GPU time: {gpu_time:.4f} seconds")
    
    if device.type != "cpu":
        speedup = cpu_time / gpu_time
        print(f"GPU speedup: {speedup:.2f}x")
    
    # Move back to CPU
    result_cpu = gpu_result.cpu()
    print(f"Result moved back to CPU: {result_cpu.device}")
    
    # Memory management
    if cuda_available:
        print(f"\nMemory after operations:")
        memory_allocated = torch.cuda.memory_allocated() / 1024**2
        print(f"Memory allocated: {memory_allocated:.2f} MB")
        
        # Clear cache
        torch.cuda.empty_cache()
        memory_after_clear = torch.cuda.memory_allocated() / 1024**2
        print(f"Memory after clearing cache: {memory_after_clear:.2f} MB")
    
    return device, gpu_tensor, cpu_result, gpu_result

def demonstrate_tensor_properties():
    """Demonstrate tensor properties and metadata"""
    print("\n" + "=" * 60)
    print("TENSOR PROPERTIES")
    print("=" * 60)
    
    # Create various tensors
    int_tensor = torch.tensor([1, 2, 3], dtype=torch.int32)
    float_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    double_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    
    tensors = [
        ("Integer", int_tensor),
        ("Float", float_tensor),
        ("Double", double_tensor)
    ]
    
    for name, tensor in tensors:
        print(f"\n{name} tensor:")
        print(f"  Data: {tensor}")
        print(f"  Shape: {tensor.shape}")
        print(f"  Size: {tensor.size()}")
        print(f"  Dtype: {tensor.dtype}")
        print(f"  Device: {tensor.device}")
        print(f"  Requires grad: {tensor.requires_grad}")
        print(f"  Memory layout: {tensor.layout}")
        print(f"  Number of elements: {tensor.numel()}")
        print(f"  Element size (bytes): {tensor.element_size()}")
        print(f"  Storage size: {tensor.storage().size()}")
    
    # Type conversion
    print(f"\nType conversions:")
    original = torch.tensor([1.7, 2.8, 3.9])
    print(f"Original (float): {original}")
    print(f"To int: {original.int()}")
    print(f"To long: {original.long()}")
    print(f"To double: {original.double()}")
    print(f"To bool: {original.bool()}")
    
    return tensors

def visualize_tensor_operations():
    """Create visualizations of tensor operations"""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('PyTorch Tensor Operations Visualization', fontsize=16)
    
    # 1. Random tensor heatmap
    random_tensor = torch.rand(10, 10)
    im1 = axes[0, 0].imshow(random_tensor.numpy(), cmap='viridis')
    axes[0, 0].set_title('Random Tensor Heatmap')
    axes[0, 0].set_xlabel('Column')
    axes[0, 0].set_ylabel('Row')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 2. Normal distribution histogram
    normal_tensor = torch.randn(10000)
    axes[0, 1].hist(normal_tensor.numpy(), bins=50, alpha=0.7, color='blue')
    axes[0, 1].set_title('Normal Distribution Tensor')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Matrix multiplication visualization
    A = torch.rand(5, 3)
    B = torch.rand(3, 4)
    C = A @ B
    
    # Show dimensions
    axes[0, 2].text(0.1, 0.8, f'A: {A.shape}', fontsize=12, transform=axes[0, 2].transAxes)
    axes[0, 2].text(0.1, 0.6, f'B: {B.shape}', fontsize=12, transform=axes[0, 2].transAxes)
    axes[0, 2].text(0.1, 0.4, f'C = A @ B: {C.shape}', fontsize=12, transform=axes[0, 2].transAxes)
    axes[0, 2].text(0.1, 0.2, 'Matrix Multiplication', fontsize=14, weight='bold', 
                   transform=axes[0, 2].transAxes)
    axes[0, 2].set_xlim(0, 1)
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].axis('off')
    
    # 4. Tensor reshaping visualization
    original_shape = torch.arange(24).reshape(2, 3, 4)
    flattened = original_shape.flatten()
    
    axes[1, 0].plot(flattened.numpy(), 'o-', markersize=4)
    axes[1, 0].set_title(f'Tensor Reshaping\nOriginal: {original_shape.shape} → Flat: {flattened.shape}')
    axes[1, 0].set_xlabel('Index')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Broadcasting example
    x = torch.arange(5).float()
    y = torch.arange(3).float().reshape(3, 1)
    result = x + y
    
    im2 = axes[1, 1].imshow(result.numpy(), cmap='coolwarm')
    axes[1, 1].set_title(f'Broadcasting Example\n{y.shape} + {x.shape} = {result.shape}')
    axes[1, 1].set_xlabel('x dimension')
    axes[1, 1].set_ylabel('y dimension')
    plt.colorbar(im2, ax=axes[1, 1])
    
    # 6. Activation functions
    x = torch.linspace(-5, 5, 100)
    relu = torch.relu(x)
    sigmoid = torch.sigmoid(x)
    tanh = torch.tanh(x)
    
    axes[1, 2].plot(x.numpy(), relu.numpy(), label='ReLU', linewidth=2)
    axes[1, 2].plot(x.numpy(), sigmoid.numpy(), label='Sigmoid', linewidth=2)
    axes[1, 2].plot(x.numpy(), tanh.numpy(), label='Tanh', linewidth=2)
    axes[1, 2].set_title('Common Activation Functions')
    axes[1, 2].set_xlabel('Input')
    axes[1, 2].set_ylabel('Output')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/tensor_operations_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def run_comprehensive_demo():
    """Run all tensor demonstrations"""
    print("PyTorch Tensor Basics Comprehensive Demo")
    print("========================================")
    
    # Run all demonstrations
    tensor_examples = demonstrate_tensor_creation()
    a, b = demonstrate_tensor_operations()
    indexed_tensor = demonstrate_tensor_indexing()
    reshaped_tensors = demonstrate_tensor_reshaping()
    matrix_results = demonstrate_matrix_operations()
    device, gpu_tensor, cpu_result, gpu_result = demonstrate_gpu_operations()
    tensor_props = demonstrate_tensor_properties()
    
    # Create visualizations
    fig = visualize_tensor_operations()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✅ Tensor creation methods demonstrated")
    print("✅ Basic tensor operations and broadcasting")
    print("✅ Indexing and slicing operations")
    print("✅ Tensor reshaping and views")
    print("✅ Matrix operations and linear algebra")
    print("✅ GPU operations and performance comparison")
    print("✅ Tensor properties and metadata")
    print("✅ Visualizations created and saved")
    
    return {
        'tensor_examples': tensor_examples,
        'operations': (a, b),
        'indexing': indexed_tensor,
        'reshaping': reshaped_tensors,
        'matrix_ops': matrix_results,
        'gpu_demo': (device, gpu_tensor),
        'properties': tensor_props,
        'visualization': fig
    }

if __name__ == "__main__":
    results = run_comprehensive_demo() 