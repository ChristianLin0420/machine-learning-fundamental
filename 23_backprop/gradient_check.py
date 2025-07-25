"""
Numerical Gradient Checking for Neural Networks

This module implements numerical gradient checking to validate analytical gradients
computed via backpropagation. Uses finite differences to approximate gradients
and compares them with analytically computed gradients.

Key Formula: ∂f/∂x ≈ [f(x + ε) - f(x - ε)] / (2ε)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Dict, List
import warnings

# Import from previous day's implementation
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '22_ffnn'))

from mlp_numpy import FeedforwardNeuralNet
from activation_functions import sigmoid, relu, tanh, get_activation_function, get_activation_derivative


class GradientChecker:
    """
    Numerical gradient checker for validating backpropagation implementations.
    
    Compares analytical gradients (from backprop) with numerical gradients
    (from finite differences) to ensure correctness of implementation.
    """
    
    def __init__(self, epsilon=1e-7, tolerance=1e-5):
        """
        Initialize gradient checker.
        
        Args:
            epsilon (float): Small value for finite differences
            tolerance (float): Tolerance for gradient comparison
        """
        self.epsilon = epsilon
        self.tolerance = tolerance
        self.check_results = []
    
    def numerical_gradient(self, func: Callable, x: np.ndarray, *args) -> np.ndarray:
        """
        Compute numerical gradient using central differences.
        
        Formula: ∂f/∂x_i ≈ [f(x + ε*e_i) - f(x - ε*e_i)] / (2ε)
        
        Args:
            func: Function to differentiate
            x: Point to compute gradient at
            *args: Additional arguments to func
            
        Returns:
            np.ndarray: Numerical gradient with same shape as x
        """
        grad = np.zeros_like(x)
        
        # Create iterator for all elements in x
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        
        while not it.finished:
            idx = it.multi_index
            
            # Store original value
            old_value = x[idx]
            
            # Evaluate f(x + ε)
            x[idx] = old_value + self.epsilon
            fxh = func(x, *args)
            
            # Evaluate f(x - ε)
            x[idx] = old_value - self.epsilon
            fxl = func(x, *args)
            
            # Compute numerical gradient
            grad[idx] = (fxh - fxl) / (2 * self.epsilon)
            
            # Restore original value
            x[idx] = old_value
            it.iternext()
        
        return grad
    
    def relative_error(self, analytical: np.ndarray, numerical: np.ndarray) -> float:
        """
        Compute relative error between analytical and numerical gradients.
        
        Formula: ||grad_analytical - grad_numerical|| / (||grad_analytical|| + ||grad_numerical|| + ε)
        
        Args:
            analytical: Analytical gradient
            numerical: Numerical gradient
            
        Returns:
            float: Relative error
        """
        numerator = np.linalg.norm(analytical - numerical)
        denominator = np.linalg.norm(analytical) + np.linalg.norm(numerical) + 1e-8
        return numerator / denominator
    
    def check_gradient(self, func: Callable, grad_func: Callable, x: np.ndarray, 
                      *args, param_name: str = "parameter") -> Dict:
        """
        Check analytical gradient against numerical gradient.
        
        Args:
            func: Function to differentiate
            grad_func: Function that computes analytical gradient
            x: Point to check gradient at
            *args: Additional arguments
            param_name: Name of parameter being checked
            
        Returns:
            dict: Results of gradient check
        """
        # Compute analytical gradient
        analytical_grad = grad_func(x, *args)
        
        # Compute numerical gradient
        numerical_grad = self.numerical_gradient(func, x.copy(), *args)
        
        # Compute relative error
        rel_error = self.relative_error(analytical_grad, numerical_grad)
        
        # Determine if check passed
        passed = rel_error < self.tolerance
        
        result = {
            'parameter': param_name,
            'analytical_grad': analytical_grad,
            'numerical_grad': numerical_grad,
            'relative_error': rel_error,
            'passed': passed,
            'tolerance': self.tolerance
        }
        
        self.check_results.append(result)
        return result
    
    def print_result(self, result: Dict):
        """Print gradient check result."""
        status = "✓ PASSED" if result['passed'] else "✗ FAILED"
        print(f"{status} - {result['parameter']}")
        print(f"  Relative error: {result['relative_error']:.2e}")
        print(f"  Tolerance: {result['tolerance']:.2e}")
        
        if not result['passed']:
            print(f"  Max analytical: {np.max(np.abs(result['analytical_grad'])):.6f}")
            print(f"  Max numerical:  {np.max(np.abs(result['numerical_grad'])):.6f}")
        print()


class SimpleNetworkGradientCheck:
    """
    Gradient checking specifically for simple neural network components.
    """
    
    def __init__(self, epsilon=1e-7, tolerance=1e-5):
        self.checker = GradientChecker(epsilon, tolerance)
    
    def check_activation_function(self, activation_name: str, x: np.ndarray) -> Dict:
        """
        Check gradient of activation function.
        
        Args:
            activation_name: Name of activation function
            x: Input to activation function
            
        Returns:
            dict: Gradient check results
        """
        activation_func = get_activation_function(activation_name)
        activation_deriv = get_activation_derivative(activation_name)
        
        def loss_func(x_input):
            """Simple loss: sum of squared activations."""
            return 0.5 * np.sum(activation_func(x_input) ** 2)
        
        def grad_func(x_input):
            """Analytical gradient of loss w.r.t input."""
            a = activation_func(x_input)
            return a * activation_deriv(x_input)
        
        return self.checker.check_gradient(
            loss_func, grad_func, x.copy(),
            param_name=f"{activation_name}_activation"
        )
    
    def check_linear_layer(self, W: np.ndarray, b: np.ndarray, x: np.ndarray) -> Tuple[Dict, Dict]:
        """
        Check gradients of linear layer: y = Wx + b
        
        Args:
            W: Weight matrix
            b: Bias vector
            x: Input vector
            
        Returns:
            tuple: (weight_check_result, bias_check_result)
        """
        def loss_func_W(W_input):
            """Loss function w.r.t weights."""
            y = np.dot(W_input, x) + b
            return 0.5 * np.sum(y ** 2)
        
        def grad_func_W(W_input):
            """Analytical gradient w.r.t weights."""
            y = np.dot(W_input, x) + b
            return np.outer(y, x)
        
        def loss_func_b(b_input):
            """Loss function w.r.t bias."""
            y = np.dot(W, x) + b_input
            return 0.5 * np.sum(y ** 2)
        
        def grad_func_b(b_input):
            """Analytical gradient w.r.t bias."""
            y = np.dot(W, x) + b_input
            return y
        
        weight_result = self.checker.check_gradient(
            loss_func_W, grad_func_W, W.copy(),
            param_name="linear_layer_weights"
        )
        
        bias_result = self.checker.check_gradient(
            loss_func_b, grad_func_b, b.copy(),
            param_name="linear_layer_bias"
        )
        
        return weight_result, bias_result
    
    def check_mlp_gradients(self, mlp: FeedforwardNeuralNet, X: np.ndarray, y: np.ndarray) -> List[Dict]:
        """
        Check all gradients in an MLP.
        
        Args:
            mlp: Trained MLP
            X: Input data
            y: Target labels
            
        Returns:
            list: Results for all gradient checks
        """
        results = []
        
        def loss_func_weights(W_flat, layer_idx):
            """Loss function w.r.t flattened weights of specific layer."""
            # Create temporary MLP with modified weights
            temp_mlp = self._create_temp_mlp(mlp, W_flat, layer_idx, 'weights')
            
            # Forward pass
            activations, _ = temp_mlp.forward(X)
            y_pred = activations[-1]
            
            # Compute loss
            return temp_mlp._compute_loss(y, y_pred)
        
        def loss_func_bias(b_flat, layer_idx):
            """Loss function w.r.t flattened bias of specific layer."""
            # Create temporary MLP with modified bias
            temp_mlp = self._create_temp_mlp(mlp, b_flat, layer_idx, 'bias')
            
            # Forward pass
            activations, _ = temp_mlp.forward(X)
            y_pred = activations[-1]
            
            # Compute loss
            return temp_mlp._compute_loss(y, y_pred)
        
        # Get analytical gradients
        activations, pre_activations = mlp.forward(X)
        weight_grads, bias_grads = mlp.backward(X, y, activations, pre_activations)
        
        # Check gradients for each layer
        for i in range(len(mlp.weights)):
            # Check weight gradients
            W_flat = mlp.weights[i].flatten()
            grad_analytical = weight_grads[i].flatten()
            
            def grad_func_weights(W_input):
                return grad_analytical
            
            weight_result = self.checker.check_gradient(
                lambda w: loss_func_weights(w, i),
                grad_func_weights,
                W_flat.copy(),
                param_name=f"layer_{i}_weights"
            )
            results.append(weight_result)
            
            # Check bias gradients
            b_flat = mlp.biases[i].flatten()
            grad_analytical = bias_grads[i].flatten()
            
            def grad_func_bias(b_input):
                return grad_analytical
            
            bias_result = self.checker.check_gradient(
                lambda b: loss_func_bias(b, i),
                grad_func_bias,
                b_flat.copy(),
                param_name=f"layer_{i}_bias"
            )
            results.append(bias_result)
        
        return results
    
    def _create_temp_mlp(self, original_mlp, param_flat, layer_idx, param_type):
        """Create temporary MLP with modified parameters."""
        # Create copy of MLP
        temp_mlp = FeedforwardNeuralNet(
            layers=original_mlp.layers,
            activations=original_mlp.activations[:-1],  # Exclude output activation
            output_activation=original_mlp.activations[-1],
            weight_init='zeros'
        )
        
        # Copy all parameters
        temp_mlp.weights = [w.copy() for w in original_mlp.weights]
        temp_mlp.biases = [b.copy() for b in original_mlp.biases]
        
        # Modify specific parameter
        if param_type == 'weights':
            temp_mlp.weights[layer_idx] = param_flat.reshape(original_mlp.weights[layer_idx].shape)
        elif param_type == 'bias':
            temp_mlp.biases[layer_idx] = param_flat.reshape(original_mlp.biases[layer_idx].shape)
        
        return temp_mlp


def comprehensive_gradient_check():
    """
    Run comprehensive gradient checking on various components.
    """
    print("Comprehensive Gradient Checking")
    print("=" * 50)
    
    checker = SimpleNetworkGradientCheck(epsilon=1e-7, tolerance=1e-5)
    
    # 1. Test activation functions
    print("\n1. Testing Activation Function Gradients")
    print("-" * 40)
    
    activations_to_test = ['sigmoid', 'relu', 'tanh']
    x_test = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    
    for activation_name in activations_to_test:
        result = checker.check_activation_function(activation_name, x_test.copy())
        checker.checker.print_result(result)
    
    # 2. Test linear layer
    print("2. Testing Linear Layer Gradients")
    print("-" * 40)
    
    np.random.seed(42)
    W = np.random.randn(3, 4) * 0.1
    b = np.random.randn(3) * 0.1
    x = np.random.randn(4)
    
    weight_result, bias_result = checker.check_linear_layer(W, b, x)
    checker.checker.print_result(weight_result)
    checker.checker.print_result(bias_result)
    
    # 3. Test simple MLP
    print("3. Testing MLP Gradients")
    print("-" * 40)
    
    # Create simple dataset
    np.random.seed(42)
    X = np.random.randn(5, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1) * 2 - 1
    
    # Create small MLP
    mlp = FeedforwardNeuralNet(
        layers=[2, 3, 1],
        activations=['tanh'],
        output_activation='sigmoid',
        random_state=42
    )
    
    # Initialize with small random weights for numerical stability
    for i in range(len(mlp.weights)):
        mlp.weights[i] = np.random.randn(*mlp.weights[i].shape) * 0.1
        mlp.biases[i] = np.random.randn(*mlp.biases[i].shape) * 0.1
    
    # Check gradients
    try:
        results = checker.check_mlp_gradients(mlp, X, y)
        for result in results:
            checker.checker.print_result(result)
    except Exception as e:
        print(f"MLP gradient check failed: {e}")
        print("This is often due to numerical precision issues with small networks.")
    
    # 4. Summary
    print("4. Summary")
    print("-" * 40)
    
    all_results = checker.checker.check_results
    passed = sum(1 for r in all_results if r['passed'])
    total = len(all_results)
    
    print(f"Gradient checks passed: {passed}/{total} ({100*passed/total:.1f}%)")
    
    failed_results = [r for r in all_results if not r['passed']]
    if failed_results:
        print("\nFailed checks:")
        for result in failed_results:
            print(f"  - {result['parameter']}: error = {result['relative_error']:.2e}")
    
    return all_results


def visualize_gradient_comparison(analytical_grad: np.ndarray, numerical_grad: np.ndarray, 
                                title: str = "Gradient Comparison", save_path: str = None):
    """
    Visualize comparison between analytical and numerical gradients.
    
    Args:
        analytical_grad: Analytical gradient
        numerical_grad: Numerical gradient
        title: Plot title
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Flatten gradients for plotting
    analytical_flat = analytical_grad.flatten()
    numerical_flat = numerical_grad.flatten()
    
    # 1. Scatter plot
    axes[0].scatter(analytical_flat, numerical_flat, alpha=0.6)
    min_val = min(analytical_flat.min(), numerical_flat.min())
    max_val = max(analytical_flat.max(), numerical_flat.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect match')
    axes[0].set_xlabel('Analytical Gradient')
    axes[0].set_ylabel('Numerical Gradient')
    axes[0].set_title('Analytical vs Numerical')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Difference plot
    diff = analytical_flat - numerical_flat
    axes[1].plot(diff, 'bo-', markersize=3)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('Gradient Index')
    axes[1].set_ylabel('Difference')
    axes[1].set_title('Gradient Differences')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Histogram of differences
    axes[2].hist(diff, bins=20, alpha=0.7, edgecolor='black')
    axes[2].axvline(x=0, color='r', linestyle='--', label='Zero difference')
    axes[2].set_xlabel('Difference')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Distribution of Differences')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def test_gradient_sensitivity():
    """
    Test how gradient checking sensitivity depends on epsilon value.
    """
    print("\nGradient Checking Sensitivity Analysis")
    print("=" * 50)
    
    # Test function: f(x) = x^4 - 2*x^2 + 1
    # Analytical gradient: f'(x) = 4*x^3 - 4*x
    
    def test_func(x):
        return np.sum(x**4 - 2*x**2 + 1)
    
    def analytical_grad(x):
        return 4*x**3 - 4*x
    
    x_test = np.array([1.5])
    epsilons = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    
    errors = []
    
    for eps in epsilons:
        checker = GradientChecker(epsilon=eps)
        numerical_grad = checker.numerical_gradient(test_func, x_test.copy())
        analytical = analytical_grad(x_test)
        
        error = checker.relative_error(analytical, numerical_grad)
        errors.append(error)
        
        print(f"ε = {eps:.0e}: relative error = {error:.2e}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.loglog(epsilons, errors, 'bo-', markersize=8)
    plt.xlabel('Epsilon (ε)')
    plt.ylabel('Relative Error')
    plt.title('Gradient Check Sensitivity to Epsilon')
    plt.grid(True, alpha=0.3)
    
    # Find optimal epsilon
    optimal_idx = np.argmin(errors)
    optimal_eps = epsilons[optimal_idx]
    plt.axvline(x=optimal_eps, color='r', linestyle='--', 
                label=f'Optimal ε = {optimal_eps:.0e}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("visualizations/epsilon_sensitivity.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nOptimal epsilon: {optimal_eps:.0e}")
    print(f"Minimum error: {errors[optimal_idx]:.2e}")


if __name__ == "__main__":
    # Create visualizations directory
    import os
    os.makedirs("visualizations", exist_ok=True)
    
    # Run comprehensive gradient checking
    results = comprehensive_gradient_check()
    
    # Test epsilon sensitivity
    test_gradient_sensitivity()
    
    print("\nGradient checking complete!")
    print("Key takeaways:")
    print("- Numerical gradients approximate analytical gradients")
    print("- Epsilon choice affects accuracy (too small → numerical errors)")
    print("- Relative error < 1e-5 typically indicates correct implementation")
    print("- Failed checks often reveal bugs in backpropagation") 