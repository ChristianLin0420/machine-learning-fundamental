"""
PyTorch Autograd Demo
====================

This module demonstrates PyTorch's automatic differentiation system:
- Creating tensors with gradient tracking
- Building computational graphs
- Computing gradients with backward()
- Understanding gradient flow
- Advanced autograd features
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from typing import List, Tuple, Dict

def basic_autograd_demo():
    """Demonstrate basic autograd functionality"""
    print("=" * 60)
    print("BASIC AUTOGRAD DEMONSTRATION")
    print("=" * 60)
    
    # 1. Simple scalar example
    print("1. Scalar gradient computation:")
    x = torch.tensor(2.0, requires_grad=True)
    print(f"x = {x}")
    print(f"x.requires_grad = {x.requires_grad}")
    
    # Compute y = x^2
    y = x ** 2
    print(f"y = x^2 = {y}")
    print(f"y.requires_grad = {y.requires_grad}")
    
    # Compute gradient dy/dx = 2x
    y.backward()
    print(f"dy/dx = {x.grad}")
    print(f"Expected: 2 * {x.data} = {2 * x.data}")
    
    # 2. Vector example
    print("\n2. Vector gradient computation:")
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    print(f"x = {x}")
    
    # Compute y = sum(x^2)
    y = torch.sum(x ** 2)
    print(f"y = sum(x^2) = {y}")
    
    # Compute gradient
    y.backward()
    print(f"dy/dx = {x.grad}")
    print(f"Expected: 2*x = {2 * x.data}")
    
    # 3. Multiple operations
    print("\n3. Multiple operations:")
    x = torch.tensor(3.0, requires_grad=True)
    y = x * 2
    z = y ** 3
    w = z + 5
    
    print(f"x = {x}")
    print(f"y = 2*x = {y}")
    print(f"z = y^3 = {z}")
    print(f"w = z + 5 = {w}")
    
    w.backward()
    print(f"dw/dx = {x.grad}")
    # Manual calculation: dw/dx = dw/dz * dz/dy * dy/dx = 1 * 3*y^2 * 2 = 6*y^2 = 6*(2*3)^2 = 6*36 = 216
    expected = 6 * (2 * 3) ** 2
    print(f"Expected: 6*(2*3)^2 = {expected}")
    
    return x, y, z, w

def computational_graph_demo():
    """Demonstrate computational graph concepts"""
    print("\n" + "=" * 60)
    print("COMPUTATIONAL GRAPH CONCEPTS")
    print("=" * 60)
    
    # Create a more complex computation
    print("Building computational graph...")
    
    # Input variables
    a = torch.tensor(2.0, requires_grad=True)
    b = torch.tensor(3.0, requires_grad=True)
    
    print(f"Inputs: a = {a}, b = {b}")
    
    # Intermediate computations
    c = a + b
    d = a * b
    e = c ** 2
    f = torch.sin(d)
    
    # Final output
    output = e + f
    
    print(f"c = a + b = {c}")
    print(f"d = a * b = {d}")
    print(f"e = c^2 = {e}")
    print(f"f = sin(d) = {f}")
    print(f"output = e + f = {output}")
    
    # Compute gradients
    output.backward()
    
    print(f"\nGradients:")
    print(f"∂output/∂a = {a.grad}")
    print(f"∂output/∂b = {b.grad}")
    
    # Manual verification for ∂output/∂a:
    # output = (a+b)^2 + sin(a*b)
    # ∂output/∂a = 2*(a+b)*1 + cos(a*b)*b = 2*(a+b) + b*cos(a*b)
    manual_grad_a = 2 * (a + b) + b * torch.cos(a * b)
    print(f"Manual ∂output/∂a = {manual_grad_a}")
    
    return a, b, output

def gradient_accumulation_demo():
    """Demonstrate gradient accumulation"""
    print("\n" + "=" * 60)
    print("GRADIENT ACCUMULATION")
    print("=" * 60)
    
    # Multiple backward passes accumulate gradients
    x = torch.tensor(1.0, requires_grad=True)
    
    print("First computation: y1 = x^2")
    y1 = x ** 2
    y1.backward()
    print(f"After first backward: x.grad = {x.grad}")
    
    print("\nSecond computation: y2 = 3*x")
    y2 = 3 * x
    y2.backward()
    print(f"After second backward: x.grad = {x.grad}")
    print("Note: Gradients accumulated! 2*1 + 3 = 5")
    
    # Zero gradients
    print("\nZeroing gradients...")
    x.grad.zero_()
    print(f"After zeroing: x.grad = {x.grad}")
    
    # Or use grad = None
    print("\nThird computation: y3 = x^3")
    y3 = x ** 3
    y3.backward()
    print(f"After third backward: x.grad = {x.grad}")
    
    return x

def higher_order_gradients_demo():
    """Demonstrate higher-order gradients"""
    print("\n" + "=" * 60)
    print("HIGHER-ORDER GRADIENTS")
    print("=" * 60)
    
    # Second derivative example
    x = torch.tensor(2.0, requires_grad=True)
    
    # Function: f(x) = x^3
    y = x ** 3
    print(f"f(x) = x^3, x = {x}, f(x) = {y}")
    
    # First derivative: f'(x) = 3*x^2
    grad1 = torch.autograd.grad(y, x, create_graph=True)[0]
    print(f"f'(x) = {grad1}")
    print(f"Expected f'(2) = 3*2^2 = {3 * 2**2}")
    
    # Second derivative: f''(x) = 6*x
    grad2 = torch.autograd.grad(grad1, x)[0]
    print(f"f''(x) = {grad2}")
    print(f"Expected f''(2) = 6*2 = {6 * 2}")
    
    return x, grad1, grad2

def jacobian_demo():
    """Demonstrate Jacobian computation"""
    print("\n" + "=" * 60)
    print("JACOBIAN COMPUTATION")
    print("=" * 60)
    
    # Vector-valued function
    def vector_function(x):
        """
        f: R^2 -> R^2
        f([x1, x2]) = [x1^2 + x2, x1 * x2^2]
        """
        x1, x2 = x[0], x[1]
        return torch.stack([x1**2 + x2, x1 * x2**2])
    
    # Input point
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    print(f"Input: x = {x}")
    
    # Compute function value
    y = vector_function(x)
    print(f"f(x) = {y}")
    
    # Compute Jacobian manually
    jacobian = torch.zeros(2, 2)
    
    for i in range(2):  # For each output component
        # Zero gradients
        if x.grad is not None:
            x.grad.zero_()
            
        # Compute gradient of i-th output component
        y_i = vector_function(x)[i]
        y_i.backward(retain_graph=True)
        jacobian[i] = x.grad.clone()
    
    print(f"Jacobian matrix:\n{jacobian}")
    
    # Manual verification:
    # ∂f1/∂x1 = 2*x1 = 2*2 = 4
    # ∂f1/∂x2 = 1
    # ∂f2/∂x1 = x2^2 = 3^2 = 9
    # ∂f2/∂x2 = 2*x1*x2 = 2*2*3 = 12
    x1, x2 = x[0].item(), x[1].item()
    manual_jacobian = torch.tensor([
        [2*x1, 1],
        [x2**2, 2*x1*x2]
    ])
    print(f"Manual Jacobian:\n{manual_jacobian}")
    
    return x, y, jacobian

def functional_jacobian_demo():
    """Demonstrate functional Jacobian computation"""
    print("\n" + "=" * 60)
    print("FUNCTIONAL JACOBIAN")
    print("=" * 60)
    
    # Using torch.autograd.functional.jacobian
    def func(x):
        return torch.stack([
            x[0]**2 + x[1],
            x[0] * x[1]**2,
            torch.sin(x[0]) + torch.cos(x[1])
        ])
    
    x = torch.tensor([1.0, 2.0])
    
    # Compute Jacobian using functional API
    jacobian = torch.autograd.functional.jacobian(func, x)
    print(f"Input: {x}")
    print(f"Function output: {func(x)}")
    print(f"Jacobian shape: {jacobian.shape}")
    print(f"Jacobian:\n{jacobian}")
    
    return x, jacobian

def no_grad_demo():
    """Demonstrate no_grad context and gradient disabling"""
    print("\n" + "=" * 60)
    print("GRADIENT DISABLING")
    print("=" * 60)
    
    x = torch.tensor(2.0, requires_grad=True)
    
    # Normal computation with gradients
    print("1. Normal computation:")
    y1 = x ** 2
    print(f"y1.requires_grad = {y1.requires_grad}")
    
    # Computation with no_grad
    print("\n2. With torch.no_grad():")
    with torch.no_grad():
        y2 = x ** 2
        print(f"y2.requires_grad = {y2.requires_grad}")
    
    # Using detach()
    print("\n3. Using detach():")
    y3 = (x ** 2).detach()
    print(f"y3.requires_grad = {y3.requires_grad}")
    
    # Mixed computation
    print("\n4. Mixed computation:")
    with torch.no_grad():
        temp = x * 2
    y4 = temp + x  # This will still require grad due to x
    print(f"temp.requires_grad = {temp.requires_grad}")
    print(f"y4.requires_grad = {y4.requires_grad}")
    
    return x, y1, y2, y3, y4

def custom_function_demo():
    """Demonstrate custom autograd functions"""
    print("\n" + "=" * 60)
    print("CUSTOM AUTOGRAD FUNCTIONS")
    print("=" * 60)
    
    class SquareFunction(torch.autograd.Function):
        """Custom square function with custom backward pass"""
        
        @staticmethod
        def forward(ctx, input):
            """Forward pass: f(x) = x^2"""
            ctx.save_for_backward(input)
            return input ** 2
        
        @staticmethod
        def backward(ctx, grad_output):
            """Backward pass: df/dx = 2*x"""
            input, = ctx.saved_tensors
            return grad_output * 2 * input
    
    # Use custom function
    square = SquareFunction.apply
    
    x = torch.tensor(3.0, requires_grad=True)
    y = square(x)
    
    print(f"Custom square function:")
    print(f"x = {x}")
    print(f"y = square(x) = {y}")
    
    y.backward()
    print(f"dy/dx = {x.grad}")
    print(f"Expected: 2*3 = 6")
    
    return x, y

def gradient_flow_visualization():
    """Visualize gradient flow through a simple network"""
    print("\n" + "=" * 60)
    print("GRADIENT FLOW VISUALIZATION")
    print("=" * 60)
    
    # Create a simple 2-layer network manually
    x = torch.randn(100, 1, requires_grad=True)
    
    # Layer 1: Linear transformation
    w1 = torch.randn(1, 5, requires_grad=True)
    b1 = torch.randn(5, requires_grad=True)
    z1 = x @ w1 + b1
    a1 = torch.relu(z1)
    
    # Layer 2: Linear transformation
    w2 = torch.randn(5, 1, requires_grad=True)
    b2 = torch.randn(1, requires_grad=True)
    z2 = a1 @ w2 + b2
    
    # Loss (mean squared)
    target = torch.randn(100, 1)
    loss = torch.mean((z2 - target) ** 2)
    
    print(f"Forward pass completed")
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Analyze gradients
    params = [('w1', w1), ('b1', b1), ('w2', w2), ('b2', b2)]
    
    print(f"\nGradient analysis:")
    gradient_norms = {}
    for name, param in params:
        grad_norm = torch.norm(param.grad).item()
        gradient_norms[name] = grad_norm
        print(f"{name}: shape {param.shape}, grad_norm = {grad_norm:.4f}")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot gradient norms
    names = list(gradient_norms.keys())
    norms = list(gradient_norms.values())
    
    ax1.bar(names, norms)
    ax1.set_title('Gradient Norms by Parameter')
    ax1.set_ylabel('Gradient Norm')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot gradient distributions
    all_grads = torch.cat([param.grad.flatten() for _, param in params])
    ax2.hist(all_grads.detach().numpy(), bins=30, alpha=0.7)
    ax2.set_title('Distribution of All Gradients')
    ax2.set_xlabel('Gradient Value')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('27_pytorch_intro/plots/gradient_flow_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return loss, gradient_norms, fig

def autograd_profiling():
    """Profile autograd operations"""
    print("\n" + "=" * 60)
    print("AUTOGRAD PROFILING")
    print("=" * 60)
    
    import time
    
    # Large tensor operations
    x = torch.randn(1000, 1000, requires_grad=True)
    
    # Forward pass timing
    start_time = time.time()
    y = torch.sum(x ** 3 + torch.sin(x) * torch.cos(x))
    forward_time = time.time() - start_time
    
    # Backward pass timing
    start_time = time.time()
    y.backward()
    backward_time = time.time() - start_time
    
    print(f"Forward pass time: {forward_time:.4f} seconds")
    print(f"Backward pass time: {backward_time:.4f} seconds")
    print(f"Backward/Forward ratio: {backward_time/forward_time:.2f}")
    
    # Memory usage
    print(f"\nMemory usage:")
    print(f"x requires_grad: {x.requires_grad}")
    print(f"y requires_grad: {y.requires_grad}")
    print(f"x.grad shape: {x.grad.shape}")
    
    return forward_time, backward_time

def comprehensive_autograd_demo():
    """Run all autograd demonstrations"""
    print("PyTorch Autograd Comprehensive Demo")
    print("===================================")
    
    # Run all demonstrations
    basic_results = basic_autograd_demo()
    graph_results = computational_graph_demo()
    accum_results = gradient_accumulation_demo()
    higher_order_results = higher_order_gradients_demo()
    jacobian_results = jacobian_demo()
    functional_jac_results = functional_jacobian_demo()
    no_grad_results = no_grad_demo()
    custom_func_results = custom_function_demo()
    viz_results = gradient_flow_visualization()
    profile_results = autograd_profiling()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✅ Basic autograd operations")
    print("✅ Computational graph concepts")
    print("✅ Gradient accumulation")
    print("✅ Higher-order gradients")
    print("✅ Jacobian computation")
    print("✅ Functional autograd API")
    print("✅ Gradient disabling techniques")
    print("✅ Custom autograd functions")
    print("✅ Gradient flow visualization")
    print("✅ Performance profiling")
    
    return {
        'basic': basic_results,
        'graph': graph_results,
        'accumulation': accum_results,
        'higher_order': higher_order_results,
        'jacobian': jacobian_results,
        'functional_jacobian': functional_jac_results,
        'no_grad': no_grad_results,
        'custom_function': custom_func_results,
        'visualization': viz_results,
        'profiling': profile_results
    }

if __name__ == "__main__":
    results = comprehensive_autograd_demo() 