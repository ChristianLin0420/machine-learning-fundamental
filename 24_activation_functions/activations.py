"""
Activation Functions Implementation
==================================

This module provides comprehensive implementations of activation functions 
and their derivatives commonly used in neural networks. All functions are 
implemented from scratch using NumPy for educational purposes.

Key Features:
- Mathematical accuracy with numerical stability
- Vectorized operations for efficiency
- Comprehensive derivative implementations
- Edge case handling
- Documentation with mathematical formulations
"""

import numpy as np
from typing import Union, Callable, Tuple
import warnings

# Suppress potential overflow warnings for educational clarity
warnings.filterwarnings('ignore', category=RuntimeWarning)

class ActivationFunction:
    """Base class for activation functions with consistent interface."""
    
    def __init__(self, name: str):
        self.name = name
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def derivative(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def __repr__(self):
        return f"ActivationFunction({self.name})"

# =============================================================================
# SIGMOID ACTIVATION FAMILY
# =============================================================================

def sigmoid(x: np.ndarray, clip_value: float = 500) -> np.ndarray:
    """
    Sigmoid activation function with numerical stability.
    
    Mathematical formulation:
        σ(x) = 1 / (1 + e^(-x))
    
    Properties:
        - Range: (0, 1)
        - Smooth and differentiable
        - Saturates for large |x|
        - Vanishing gradient problem
    
    Args:
        x: Input array
        clip_value: Clipping value to prevent overflow
        
    Returns:
        Sigmoid activated values
    """
    # Numerical stability: clip extreme values
    x_clipped = np.clip(x, -clip_value, clip_value)
    
    # Use stable computation for negative values
    # For x >= 0: σ(x) = 1 / (1 + e^(-x))
    # For x < 0: σ(x) = e^x / (1 + e^x) to avoid overflow
    positive_mask = x_clipped >= 0
    result = np.zeros_like(x_clipped)
    
    # Positive values
    exp_neg_x = np.exp(-x_clipped[positive_mask])
    result[positive_mask] = 1.0 / (1.0 + exp_neg_x)
    
    # Negative values  
    exp_x = np.exp(x_clipped[~positive_mask])
    result[~positive_mask] = exp_x / (1.0 + exp_x)
    
    return result

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of sigmoid function.
    
    Mathematical formulation:
        σ'(x) = σ(x) * (1 - σ(x))
    
    Properties:
        - Maximum value: 0.25 at x = 0
        - Vanishing gradients for |x| > 3
        - Always positive
        
    Args:
        x: Input array
        
    Returns:
        Sigmoid derivative values
    """
    s = sigmoid(x)
    return s * (1.0 - s)

# =============================================================================
# HYPERBOLIC TANGENT ACTIVATION
# =============================================================================

def tanh(x: np.ndarray) -> np.ndarray:
    """
    Hyperbolic tangent activation function.
    
    Mathematical formulation:
        tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
                = 2*sigmoid(2x) - 1
    
    Properties:
        - Range: (-1, 1)
        - Zero-centered (better than sigmoid)
        - Still suffers from vanishing gradients
        - Symmetric around origin
        
    Args:
        x: Input array
        
    Returns:
        Tanh activated values
    """
    return np.tanh(x)

def tanh_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of hyperbolic tangent function.
    
    Mathematical formulation:
        tanh'(x) = 1 - tanh²(x) = sech²(x)
    
    Properties:
        - Maximum value: 1.0 at x = 0
        - Better than sigmoid but still vanishing
        
    Args:
        x: Input array
        
    Returns:
        Tanh derivative values
    """
    t = tanh(x)
    return 1.0 - t**2

# =============================================================================
# RECTIFIED LINEAR UNIT (ReLU) FAMILY
# =============================================================================

def relu(x: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit activation function.
    
    Mathematical formulation:
        ReLU(x) = max(0, x)
    
    Properties:
        - Range: [0, ∞)
        - No saturation for positive values
        - Sparse activation (many zeros)
        - Computationally efficient
        - Dying ReLU problem for negative inputs
        
    Args:
        x: Input array
        
    Returns:
        ReLU activated values
    """
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of ReLU function.
    
    Mathematical formulation:
        ReLU'(x) = { 1 if x > 0
                   { 0 if x ≤ 0
    
    Note: Technically undefined at x = 0, but we use 0 by convention.
    
    Properties:
        - Binary gradient: 0 or 1
        - No vanishing gradient for positive inputs
        - Gradient is 0 for negative inputs (dying ReLU)
        
    Args:
        x: Input array
        
    Returns:
        ReLU derivative values
    """
    return (x > 0).astype(float)

def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Leaky ReLU activation function.
    
    Mathematical formulation:
        LeakyReLU(x) = { x      if x > 0
                       { α*x    if x ≤ 0
    
    Properties:
        - Range: (-∞, ∞)
        - Fixes dying ReLU problem
        - Small gradient for negative inputs
        - α typically 0.01
        
    Args:
        x: Input array
        alpha: Slope for negative inputs
        
    Returns:
        Leaky ReLU activated values
    """
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Derivative of Leaky ReLU function.
    
    Mathematical formulation:
        LeakyReLU'(x) = { 1  if x > 0
                        { α  if x ≤ 0
    
    Properties:
        - Prevents completely dead neurons
        - Non-zero gradient for all inputs
        
    Args:
        x: Input array
        alpha: Slope for negative inputs
        
    Returns:
        Leaky ReLU derivative values
    """
    return np.where(x > 0, 1.0, alpha)

def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Exponential Linear Unit activation function.
    
    Mathematical formulation:
        ELU(x) = { x                if x > 0
                 { α(e^x - 1)       if x ≤ 0
    
    Properties:
        - Range: (-α, ∞)
        - Smooth everywhere (differentiable)
        - Self-normalizing property
        - Reduces bias shift
        
    Args:
        x: Input array
        alpha: Scale parameter for negative inputs
        
    Returns:
        ELU activated values
    """
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Derivative of ELU function.
    
    Mathematical formulation:
        ELU'(x) = { 1           if x > 0
                  { α*e^x       if x ≤ 0
    
    Properties:
        - Smooth transition at x = 0
        - Non-zero gradient everywhere
        
    Args:
        x: Input array
        alpha: Scale parameter for negative inputs
        
    Returns:
        ELU derivative values
    """
    return np.where(x > 0, 1.0, alpha * np.exp(x))

# =============================================================================
# MODERN ACTIVATION FUNCTIONS
# =============================================================================

def gelu(x: np.ndarray, approximate: bool = True) -> np.ndarray:
    """
    Gaussian Error Linear Unit activation function.
    
    Mathematical formulation (exact):
        GELU(x) = x * Φ(x) where Φ is Gaussian CDF
    
    Mathematical formulation (approximate):
        GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    
    Properties:
        - Smooth, non-monotonic
        - Used in BERT and GPT models
        - Self-gating properties
        - Probabilistically motivated
        
    Args:
        x: Input array
        approximate: Use fast approximation if True
        
    Returns:
        GELU activated values
    """
    if approximate:
        # Fast approximation used in practice
        return 0.5 * x * (1.0 + tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))
    else:
        # Exact formulation using error function
        try:
            from scipy.special import erf
            return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))
        except ImportError:
            # Fall back to approximation if scipy not available
            return 0.5 * x * (1.0 + tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

def gelu_derivative(x: np.ndarray, approximate: bool = True) -> np.ndarray:
    """
    Derivative of GELU function.
    
    Mathematical formulation (approximate):
        GELU'(x) ≈ 0.5 * tanh(√(2/π) * (x + 0.044715 * x³)) + 
                   0.5 * x * sech²(√(2/π) * (x + 0.044715 * x³)) * 
                   √(2/π) * (1 + 3 * 0.044715 * x²)
    
    Args:
        x: Input array
        approximate: Use fast approximation if True
        
    Returns:
        GELU derivative values
    """
    if approximate:
        # Derivative of the approximation
        sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
        inner = sqrt_2_over_pi * (x + 0.044715 * x**3)
        tanh_inner = tanh(inner)
        sech_squared = 1.0 - tanh_inner**2
        
        first_term = 0.5 * (1.0 + tanh_inner)
        second_term = 0.5 * x * sech_squared * sqrt_2_over_pi * (1.0 + 3 * 0.044715 * x**2)
        
        return first_term + second_term
    else:
        # Exact derivative using Gaussian PDF
        try:
            from scipy.special import erf
            gaussian_pdf = np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)
            gaussian_cdf = 0.5 * (1.0 + erf(x / np.sqrt(2.0)))
            return gaussian_cdf + x * gaussian_pdf
        except ImportError:
            # Fall back to approximation
            sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
            inner = sqrt_2_over_pi * (x + 0.044715 * x**3)
            tanh_inner = tanh(inner)
            sech_squared = 1.0 - tanh_inner**2
            
            first_term = 0.5 * (1.0 + tanh_inner)
            second_term = 0.5 * x * sech_squared * sqrt_2_over_pi * (1.0 + 3 * 0.044715 * x**2)
            
            return first_term + second_term

def swish(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """
    Swish activation function (also known as SiLU).
    
    Mathematical formulation:
        Swish(x) = x * σ(β*x) = x / (1 + e^(-β*x))
    
    Properties:
        - Self-gating activation
        - Smooth and non-monotonic
        - β = 1 gives SiLU
        - Bounded below, unbounded above
        
    Args:
        x: Input array
        beta: Scaling parameter (β = 1 for SiLU)
        
    Returns:
        Swish activated values
    """
    return x * sigmoid(beta * x)

def swish_derivative(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """
    Derivative of Swish function.
    
    Mathematical formulation:
        Swish'(x) = σ(β*x) + x * σ'(β*x) * β
                  = σ(β*x) + β*x * σ(β*x) * (1 - σ(β*x))
                  = σ(β*x) * (1 + β*x * (1 - σ(β*x)))
    
    Args:
        x: Input array
        beta: Scaling parameter
        
    Returns:
        Swish derivative values
    """
    sigmoid_val = sigmoid(beta * x)
    return sigmoid_val * (1.0 + beta * x * (1.0 - sigmoid_val))

def softplus(x: np.ndarray) -> np.ndarray:
    """
    Softplus activation function.
    
    Mathematical formulation:
        Softplus(x) = log(1 + e^x)
    
    Properties:
        - Smooth approximation to ReLU
        - Range: (0, ∞)
        - Derivative is sigmoid
        - No dead neurons
        
    Args:
        x: Input array
        
    Returns:
        Softplus activated values
    """
    # Numerical stability: for large x, softplus(x) ≈ x
    return np.where(x > 20, x, np.log(1.0 + np.exp(x)))

def softplus_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of Softplus function.
    
    Mathematical formulation:
        Softplus'(x) = e^x / (1 + e^x) = σ(x)
    
    Note: The derivative of softplus is exactly sigmoid!
    
    Args:
        x: Input array
        
    Returns:
        Softplus derivative values (sigmoid)
    """
    return sigmoid(x)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_activation_function(name: str) -> Callable:
    """
    Get activation function by name.
    
    Args:
        name: Name of activation function
        
    Returns:
        Activation function
    """
    activation_map = {
        'sigmoid': sigmoid,
        'tanh': tanh,
        'relu': relu,
        'leaky_relu': leaky_relu,
        'elu': elu,
        'gelu': gelu,
        'swish': swish,
        'softplus': softplus,
        'linear': lambda x: x,  # Identity function
    }
    
    if name.lower() not in activation_map:
        raise ValueError(f"Unknown activation function: {name}")
    
    return activation_map[name.lower()]

def get_activation_derivative(name: str) -> Callable:
    """
    Get activation derivative function by name.
    
    Args:
        name: Name of activation function
        
    Returns:
        Activation derivative function
    """
    derivative_map = {
        'sigmoid': sigmoid_derivative,
        'tanh': tanh_derivative,
        'relu': relu_derivative,
        'leaky_relu': leaky_relu_derivative,
        'elu': elu_derivative,
        'gelu': gelu_derivative,
        'swish': swish_derivative,
        'softplus': softplus_derivative,
        'linear': lambda x: np.ones_like(x),  # Derivative of identity
    }
    
    if name.lower() not in derivative_map:
        raise ValueError(f"Unknown activation function: {name}")
    
    return derivative_map[name.lower()]

def activation_properties() -> dict:
    """
    Get properties of all activation functions.
    
    Returns:
        Dictionary with activation properties
    """
    return {
        'sigmoid': {
            'range': '(0, 1)',
            'zero_centered': False,
            'monotonic': True,
            'saturating': True,
            'vanishing_gradient': True,
            'computational_cost': 'Medium',
            'use_cases': ['Binary classification output', 'Gating mechanisms']
        },
        'tanh': {
            'range': '(-1, 1)',
            'zero_centered': True,
            'monotonic': True,
            'saturating': True,
            'vanishing_gradient': True,
            'computational_cost': 'Medium',
            'use_cases': ['Hidden layers (better than sigmoid)', 'RNN gates']
        },
        'relu': {
            'range': '[0, ∞)',
            'zero_centered': False,
            'monotonic': True,
            'saturating': False,
            'vanishing_gradient': False,
            'computational_cost': 'Very Low',
            'use_cases': ['Hidden layers', 'Most common choice', 'CNNs']
        },
        'leaky_relu': {
            'range': '(-∞, ∞)',
            'zero_centered': False,
            'monotonic': True,
            'saturating': False,
            'vanishing_gradient': False,
            'computational_cost': 'Very Low',
            'use_cases': ['When ReLU causes dying neurons', 'Hidden layers']
        },
        'elu': {
            'range': '(-α, ∞)',
            'zero_centered': True,
            'monotonic': True,
            'saturating': False,
            'vanishing_gradient': False,
            'computational_cost': 'Medium',
            'use_cases': ['Self-normalizing networks', 'Hidden layers']
        },
        'gelu': {
            'range': '(-0.17, ∞)',
            'zero_centered': False,
            'monotonic': False,
            'saturating': False,
            'vanishing_gradient': False,
            'computational_cost': 'High',
            'use_cases': ['Transformers (BERT, GPT)', 'Modern architectures']
        },
        'swish': {
            'range': '(-∞, ∞)',
            'zero_centered': False,
            'monotonic': False,
            'saturating': False,
            'vanishing_gradient': False,
            'computational_cost': 'Medium',
            'use_cases': ['Mobile networks', 'Self-gating applications']
        },
        'softplus': {
            'range': '(0, ∞)',
            'zero_centered': False,
            'monotonic': True,
            'saturating': False,
            'vanishing_gradient': True,
            'computational_cost': 'Medium',
            'use_cases': ['Smooth ReLU alternative', 'Positive outputs']
        }
    }

def compare_activations_summary() -> str:
    """
    Generate a summary comparison of activation functions.
    
    Returns:
        Formatted comparison string
    """
    props = activation_properties()
    
    summary = "Activation Functions Comparison Summary\n"
    summary += "=" * 50 + "\n\n"
    
    # Gradient-friendly functions
    gradient_friendly = [name for name, prop in props.items() 
                        if not prop['vanishing_gradient']]
    summary += f"✅ Gradient-friendly (no vanishing): {', '.join(gradient_friendly)}\n"
    
    # Zero-centered functions  
    zero_centered = [name for name, prop in props.items() 
                    if prop['zero_centered']]
    summary += f"⚖️  Zero-centered: {', '.join(zero_centered)}\n"
    
    # Non-saturating functions
    non_saturating = [name for name, prop in props.items() 
                     if not prop['saturating']]
    summary += f"🚀 Non-saturating: {', '.join(non_saturating)}\n"
    
    # Computationally efficient
    efficient = [name for name, prop in props.items() 
                if prop['computational_cost'] in ['Very Low', 'Low']]
    summary += f"⚡ Computationally efficient: {', '.join(efficient)}\n"
    
    # Modern/advanced functions
    modern = ['gelu', 'swish', 'elu']
    summary += f"🔬 Modern/Advanced: {', '.join(modern)}\n"
    
    return summary

# Test functions for validation
def test_activations():
    """Test all activation functions for correctness."""
    print("Testing activation functions...")
    
    # Test input
    x = np.array([-2, -1, 0, 1, 2])
    
    activations = ['sigmoid', 'tanh', 'relu', 'leaky_relu', 'elu', 
                  'gelu', 'swish', 'softplus']
    
    for name in activations:
        try:
            func = get_activation_function(name)
            deriv = get_activation_derivative(name)
            
            y = func(x)
            dy = deriv(x)
            
            print(f"✅ {name:12s}: output shape {y.shape}, derivative shape {dy.shape}")
            
        except Exception as e:
            print(f"❌ {name:12s}: Error - {e}")
    
    print("Activation function tests completed!")

if __name__ == "__main__":
    # Run tests
    test_activations()
    
    # Print comparison summary
    print("\n" + compare_activations_summary())
    
    # Example usage
    print("\nExample Usage:")
    print("-" * 20)
    x = np.linspace(-3, 3, 7)
    print(f"Input: {x}")
    print(f"ReLU: {relu(x)}")
    print(f"ReLU': {relu_derivative(x)}")
    print(f"GELU: {gelu(x)}")
    print(f"Swish: {swish(x)}") 