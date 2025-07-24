"""
Activation Functions for Neural Networks

This module provides various activation functions and their derivatives
commonly used in neural networks, along with utility functions for
output layer activations like softmax.
"""

import numpy as np


def sigmoid(x):
    """
    Sigmoid activation function.
    
    f(x) = 1 / (1 + exp(-x))
    
    Args:
        x (np.ndarray): Input array
        
    Returns:
        np.ndarray: Sigmoid of input
    """
    # Clip x to prevent overflow
    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_clipped))


def sigmoid_derivative(x):
    """
    Derivative of sigmoid function.
    
    f'(x) = f(x) * (1 - f(x))
    
    Args:
        x (np.ndarray): Input array (can be pre-computed sigmoid values)
        
    Returns:
        np.ndarray: Derivative of sigmoid
    """
    s = sigmoid(x)
    return s * (1 - s)


def relu(x):
    """
    ReLU (Rectified Linear Unit) activation function.
    
    f(x) = max(0, x)
    
    Args:
        x (np.ndarray): Input array
        
    Returns:
        np.ndarray: ReLU of input
    """
    return np.maximum(0, x)


def relu_derivative(x):
    """
    Derivative of ReLU function.
    
    f'(x) = 1 if x > 0, 0 otherwise
    
    Args:
        x (np.ndarray): Input array
        
    Returns:
        np.ndarray: Derivative of ReLU
    """
    return (x > 0).astype(float)


def leaky_relu(x, alpha=0.01):
    """
    Leaky ReLU activation function.
    
    f(x) = x if x > 0, alpha * x otherwise
    
    Args:
        x (np.ndarray): Input array
        alpha (float): Slope for negative values
        
    Returns:
        np.ndarray: Leaky ReLU of input
    """
    return np.where(x > 0, x, alpha * x)


def leaky_relu_derivative(x, alpha=0.01):
    """
    Derivative of Leaky ReLU function.
    
    Args:
        x (np.ndarray): Input array
        alpha (float): Slope for negative values
        
    Returns:
        np.ndarray: Derivative of Leaky ReLU
    """
    return np.where(x > 0, 1, alpha)


def tanh(x):
    """
    Hyperbolic tangent activation function.
    
    f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    
    Args:
        x (np.ndarray): Input array
        
    Returns:
        np.ndarray: Tanh of input
    """
    return np.tanh(x)


def tanh_derivative(x):
    """
    Derivative of tanh function.
    
    f'(x) = 1 - tanh²(x)
    
    Args:
        x (np.ndarray): Input array
        
    Returns:
        np.ndarray: Derivative of tanh
    """
    t = np.tanh(x)
    return 1 - t**2


def linear(x):
    """
    Linear activation function (identity).
    
    f(x) = x
    
    Args:
        x (np.ndarray): Input array
        
    Returns:
        np.ndarray: Input unchanged
    """
    return x


def linear_derivative(x):
    """
    Derivative of linear function.
    
    f'(x) = 1
    
    Args:
        x (np.ndarray): Input array
        
    Returns:
        np.ndarray: Array of ones
    """
    return np.ones_like(x)


def softmax(x):
    """
    Softmax activation function for multi-class classification.
    
    f(x_i) = exp(x_i) / Σ(exp(x_j))
    
    Args:
        x (np.ndarray): Input array (n_samples, n_classes) or (n_classes,)
        
    Returns:
        np.ndarray: Softmax probabilities
    """
    # Subtract max for numerical stability
    if x.ndim == 1:
        x_stable = x - np.max(x)
        exp_x = np.exp(x_stable)
        return exp_x / np.sum(exp_x)
    else:
        x_stable = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_stable)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def softmax_derivative(x):
    """
    Derivative of softmax function.
    Note: This is typically computed in conjunction with cross-entropy loss.
    
    Args:
        x (np.ndarray): Input array
        
    Returns:
        np.ndarray: Jacobian matrix of softmax
    """
    s = softmax(x)
    if x.ndim == 1:
        return np.diag(s) - np.outer(s, s)
    else:
        # For batch processing, return simplified form
        # In practice, softmax derivative is usually combined with loss
        return s * (1 - s)


# Dictionary mapping activation names to functions
ACTIVATION_FUNCTIONS = {
    'sigmoid': sigmoid,
    'relu': relu,
    'leaky_relu': leaky_relu,
    'tanh': tanh,
    'linear': linear,
    'softmax': softmax
}

# Dictionary mapping activation names to their derivatives
ACTIVATION_DERIVATIVES = {
    'sigmoid': sigmoid_derivative,
    'relu': relu_derivative,
    'leaky_relu': leaky_relu_derivative,
    'tanh': tanh_derivative,
    'linear': linear_derivative,
    'softmax': softmax_derivative
}


def get_activation_function(name):
    """
    Get activation function by name.
    
    Args:
        name (str): Name of activation function
        
    Returns:
        callable: Activation function
    """
    if name not in ACTIVATION_FUNCTIONS:
        raise ValueError(f"Unknown activation function: {name}. "
                        f"Available: {list(ACTIVATION_FUNCTIONS.keys())}")
    return ACTIVATION_FUNCTIONS[name]


def get_activation_derivative(name):
    """
    Get activation derivative function by name.
    
    Args:
        name (str): Name of activation function
        
    Returns:
        callable: Activation derivative function
    """
    if name not in ACTIVATION_DERIVATIVES:
        raise ValueError(f"Unknown activation function: {name}. "
                        f"Available: {list(ACTIVATION_DERIVATIVES.keys())}")
    return ACTIVATION_DERIVATIVES[name]


def plot_activation_functions(x_range=(-5, 5), save_path=None):
    """
    Plot all activation functions and their derivatives.
    
    Args:
        x_range (tuple): Range of x values to plot
        save_path (str): Path to save plot
    """
    import matplotlib.pyplot as plt
    
    x = np.linspace(x_range[0], x_range[1], 1000)
    
    # Activation functions to plot (excluding softmax for simplicity)
    functions_to_plot = ['sigmoid', 'relu', 'leaky_relu', 'tanh', 'linear']
    
    fig, axes = plt.subplots(2, len(functions_to_plot), figsize=(15, 8))
    
    for i, func_name in enumerate(functions_to_plot):
        func = ACTIVATION_FUNCTIONS[func_name]
        deriv = ACTIVATION_DERIVATIVES[func_name]
        
        # Plot activation function
        if func_name == 'leaky_relu':
            y = func(x, alpha=0.01)
            y_deriv = deriv(x, alpha=0.01)
        else:
            y = func(x)
            y_deriv = deriv(x)
        
        axes[0, i].plot(x, y, 'b-', linewidth=2)
        axes[0, i].set_title(f'{func_name.replace("_", " ").title()}')
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].set_xlabel('x')
        axes[0, i].set_ylabel('f(x)')
        
        # Plot derivative
        axes[1, i].plot(x, y_deriv, 'r-', linewidth=2)
        axes[1, i].set_title(f'{func_name.replace("_", " ").title()} Derivative')
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].set_xlabel('x')
        axes[1, i].set_ylabel("f'(x)")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def compare_activation_properties():
    """
    Compare key properties of different activation functions.
    
    Returns:
        dict: Properties of each activation function
    """
    properties = {
        'sigmoid': {
            'range': '(0, 1)',
            'zero_centered': False,
            'monotonic': True,
            'saturating': True,
            'vanishing_gradient': True,
            'use_case': 'Binary classification output'
        },
        'tanh': {
            'range': '(-1, 1)',
            'zero_centered': True,
            'monotonic': True,
            'saturating': True,
            'vanishing_gradient': True,
            'use_case': 'Hidden layers (better than sigmoid)'
        },
        'relu': {
            'range': '[0, ∞)',
            'zero_centered': False,
            'monotonic': True,
            'saturating': False,
            'vanishing_gradient': False,
            'use_case': 'Most hidden layers (default choice)'
        },
        'leaky_relu': {
            'range': '(-∞, ∞)',
            'zero_centered': False,
            'monotonic': True,
            'saturating': False,
            'vanishing_gradient': False,
            'use_case': 'When ReLU causes dead neurons'
        },
        'linear': {
            'range': '(-∞, ∞)',
            'zero_centered': True,
            'monotonic': True,
            'saturating': False,
            'vanishing_gradient': False,
            'use_case': 'Regression output layer'
        },
        'softmax': {
            'range': '(0, 1) with sum=1',
            'zero_centered': False,
            'monotonic': False,
            'saturating': True,
            'vanishing_gradient': True,
            'use_case': 'Multi-class classification output'
        }
    }
    
    return properties


if __name__ == "__main__":
    # Create plots directory
    import os
    os.makedirs("plots", exist_ok=True)
    
    print("Activation Functions Analysis")
    print("=" * 50)
    
    # Plot all activation functions
    plot_activation_functions(save_path="plots/activation_functions.png")
    
    # Display properties comparison
    properties = compare_activation_properties()
    
    print("\nActivation Function Properties:")
    print("-" * 80)
    print(f"{'Function':<12} {'Range':<15} {'Zero-Centered':<13} {'Saturating':<10} {'Use Case':<25}")
    print("-" * 80)
    
    for func_name, props in properties.items():
        print(f"{func_name:<12} {props['range']:<15} {str(props['zero_centered']):<13} "
              f"{str(props['saturating']):<10} {props['use_case']:<25}")
    
    # Test numerical stability
    print(f"\nNumerical Stability Tests:")
    print("-" * 30)
    
    large_vals = np.array([100, 500, 1000])
    print(f"Sigmoid(large values): {sigmoid(large_vals)}")
    print(f"Tanh(large values): {tanh(large_vals)}")
    
    # Test softmax stability
    large_logits = np.array([1000, 1001, 1002])
    print(f"Softmax(large logits): {softmax(large_logits)}")
    
    print("\nActivation functions implementation complete!") 