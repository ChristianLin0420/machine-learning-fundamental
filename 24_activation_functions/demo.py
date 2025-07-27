"""
Simple Activation Functions Demo
===============================

A quick demonstration of the activation functions implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from activations import *

def demo_activations():
    """Simple demo of activation functions."""
    print("🚀 Activation Functions Demo")
    print("=" * 40)
    
    # Test basic functionality
    x_test = np.array([-2, -1, 0, 1, 2])
    print(f"Test input: {x_test}")
    print()
    
    # Test each activation
    activations = ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'elu', 'gelu', 'swish']
    
    for name in activations:
        func = get_activation_function(name)
        deriv = get_activation_derivative(name)
        
        y = func(x_test)
        dy = deriv(x_test)
        
        print(f"{name.upper():12s}: f(x) = {y}")
        print(f"{'':12s}: f'(x) = {dy}")
        print()
    
    # Create simple visualization
    print("Creating simple comparison plot...")
    
    x = np.linspace(-3, 3, 100)
    
    plt.figure(figsize=(15, 5))
    
    # Plot functions
    plt.subplot(1, 2, 1)
    for name in ['sigmoid', 'tanh', 'relu', 'gelu']:
        func = get_activation_function(name)
        y = func(x)
        plt.plot(x, y, label=name.upper(), linewidth=2)
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Activation Functions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot derivatives
    plt.subplot(1, 2, 2)
    for name in ['sigmoid', 'tanh', 'relu', 'gelu']:
        deriv = get_activation_derivative(name)
        dy = deriv(x)
        plt.plot(x, dy, label=f"{name.upper()}'", linewidth=2)
    
    plt.xlabel('x')
    plt.ylabel("f'(x)")
    plt.title('Activation Function Derivatives')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/activation_demo.png', dpi=150, bbox_inches='tight')
    print("✅ Demo plot saved to plots/activation_demo.png")
    
    # Print properties summary
    print("\n" + compare_activations_summary())

if __name__ == "__main__":
    demo_activations() 