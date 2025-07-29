"""
Gradient Descent Optimizers from Scratch

This module implements various gradient descent optimization algorithms
used in training neural networks and machine learning models.

Key Optimizers Implemented:
- SGD (Stochastic Gradient Descent)
- SGD with Momentum
- Nesterov Accelerated Gradient (NAG)
- AdaGrad (Adaptive Gradient)
- RMSProp (Root Mean Square Propagation)
- Adam (Adaptive Moment Estimation)

Each optimizer follows a consistent interface:
- __init__: Initialize with hyperparameters
- step: Update parameters given gradients
- reset: Reset internal state (for momentum-based optimizers)
- get_state: Get current optimizer state
"""

import numpy as np
from typing import Dict, List, Optional, Any
import copy

class BaseOptimizer:
    """Base class for all optimizers providing common interface"""
    
    def __init__(self, learning_rate: float = 0.01):
        """
        Initialize base optimizer
        
        Args:
            learning_rate: Learning rate for parameter updates
        """
        self.learning_rate = learning_rate
        self.step_count = 0
        
    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        """
        Update parameters using gradients
        
        Args:
            params: Dictionary of parameter arrays
            grads: Dictionary of gradient arrays (same keys as params)
        """
        raise NotImplementedError("Subclasses must implement step method")
        
    def reset(self) -> None:
        """Reset optimizer state"""
        self.step_count = 0
        
    def get_state(self) -> Dict[str, Any]:
        """Get current optimizer state"""
        return {
            'learning_rate': self.learning_rate,
            'step_count': self.step_count
        }

class SGD(BaseOptimizer):
    """
    Stochastic Gradient Descent
    
    Update rule: θ ← θ - η∇θ
    
    Simple gradient descent without any modifications.
    Prone to oscillations and slow convergence.
    """
    
    def __init__(self, learning_rate: float = 0.01):
        """
        Initialize SGD optimizer
        
        Args:
            learning_rate: Step size for parameter updates
        """
        super().__init__(learning_rate)
        
    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        """
        Perform SGD parameter update
        
        Args:
            params: Parameters to update (modified in-place)
            grads: Gradients for each parameter
        """
        self.step_count += 1
        
        for param_name in params:
            if param_name in grads:
                params[param_name] -= self.learning_rate * grads[param_name]

class Momentum(BaseOptimizer):
    """
    SGD with Momentum
    
    Update rule:
    v_t = β * v_{t-1} + ∇θ
    θ ← θ - η * v_t
    
    Accelerates convergence and reduces oscillations by accumulating
    momentum in directions of consistent gradients.
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        """
        Initialize Momentum optimizer
        
        Args:
            learning_rate: Step size for parameter updates
            momentum: Momentum coefficient (typically 0.9)
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocities = {}
        
    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        """
        Perform momentum-based parameter update
        
        Args:
            params: Parameters to update (modified in-place)
            grads: Gradients for each parameter
        """
        self.step_count += 1
        
        for param_name in params:
            if param_name in grads:
                # Initialize velocity if first time
                if param_name not in self.velocities:
                    self.velocities[param_name] = np.zeros_like(params[param_name])
                
                # Update velocity: v = β*v + ∇θ
                self.velocities[param_name] = (self.momentum * self.velocities[param_name] + 
                                             grads[param_name])
                
                # Update parameters: θ = θ - η*v
                params[param_name] -= self.learning_rate * self.velocities[param_name]
                
    def reset(self) -> None:
        """Reset optimizer state including velocities"""
        super().reset()
        self.velocities = {}
        
    def get_state(self) -> Dict[str, Any]:
        """Get current optimizer state including velocities"""
        state = super().get_state()
        state.update({
            'momentum': self.momentum,
            'velocities': copy.deepcopy(self.velocities)
        })
        return state

class Nesterov(BaseOptimizer):
    """
    Nesterov Accelerated Gradient (NAG)
    
    Update rule:
    v_t = β * v_{t-1} + ∇θ(θ - η * β * v_{t-1})
    θ ← θ - η * v_t
    
    "Look-ahead" momentum that evaluates gradient at the anticipated
    position, providing better convergence properties than standard momentum.
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        """
        Initialize Nesterov optimizer
        
        Args:
            learning_rate: Step size for parameter updates
            momentum: Momentum coefficient (typically 0.9)
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocities = {}
        
    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        """
        Perform Nesterov momentum parameter update
        
        Args:
            params: Parameters to update (modified in-place)
            grads: Gradients for each parameter
        """
        self.step_count += 1
        
        for param_name in params:
            if param_name in grads:
                # Initialize velocity if first time
                if param_name not in self.velocities:
                    self.velocities[param_name] = np.zeros_like(params[param_name])
                
                # Save old velocity
                v_prev = self.velocities[param_name].copy()
                
                # Update velocity: v = β*v + ∇θ
                self.velocities[param_name] = (self.momentum * self.velocities[param_name] + 
                                             grads[param_name])
                
                # Nesterov update: θ = θ - η*(β*v_old + ∇θ)
                # Equivalent to: θ = θ - η*β*v_old - η*∇θ
                params[param_name] -= (self.learning_rate * 
                                     (self.momentum * v_prev + grads[param_name]))
                
    def reset(self) -> None:
        """Reset optimizer state including velocities"""
        super().reset()
        self.velocities = {}
        
    def get_state(self) -> Dict[str, Any]:
        """Get current optimizer state including velocities"""
        state = super().get_state()
        state.update({
            'momentum': self.momentum,
            'velocities': copy.deepcopy(self.velocities)
        })
        return state

class AdaGrad(BaseOptimizer):
    """
    Adaptive Gradient Algorithm (AdaGrad)
    
    Update rule:
    G_t = G_{t-1} + ∇θ²
    θ ← θ - η * ∇θ / (√G_t + ε)
    
    Adapts learning rate based on historical gradients.
    Performs larger updates for infrequent parameters and smaller
    updates for frequent parameters.
    """
    
    def __init__(self, learning_rate: float = 0.01, eps: float = 1e-8):
        """
        Initialize AdaGrad optimizer
        
        Args:
            learning_rate: Initial learning rate
            eps: Small constant to avoid division by zero
        """
        super().__init__(learning_rate)
        self.eps = eps
        self.accumulated_grads = {}
        
    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        """
        Perform AdaGrad parameter update
        
        Args:
            params: Parameters to update (modified in-place)
            grads: Gradients for each parameter
        """
        self.step_count += 1
        
        for param_name in params:
            if param_name in grads:
                # Initialize accumulated gradients if first time
                if param_name not in self.accumulated_grads:
                    self.accumulated_grads[param_name] = np.zeros_like(params[param_name])
                
                # Accumulate squared gradients: G = G + ∇θ²
                self.accumulated_grads[param_name] += grads[param_name] ** 2
                
                # Adaptive learning rate: η / √(G + ε)
                adapted_lr = self.learning_rate / (np.sqrt(self.accumulated_grads[param_name]) + self.eps)
                
                # Update parameters
                params[param_name] -= adapted_lr * grads[param_name]
                
    def reset(self) -> None:
        """Reset optimizer state including accumulated gradients"""
        super().reset()
        self.accumulated_grads = {}
        
    def get_state(self) -> Dict[str, Any]:
        """Get current optimizer state including accumulated gradients"""
        state = super().get_state()
        state.update({
            'eps': self.eps,
            'accumulated_grads': copy.deepcopy(self.accumulated_grads)
        })
        return state

class RMSProp(BaseOptimizer):
    """
    Root Mean Square Propagation (RMSProp)
    
    Update rule:
    E[g²]_t = β * E[g²]_{t-1} + (1-β) * ∇θ²
    θ ← θ - η * ∇θ / (√E[g²]_t + ε)
    
    Addresses AdaGrad's diminishing learning rates by using exponential
    moving average of squared gradients instead of cumulative sum.
    """
    
    def __init__(self, learning_rate: float = 0.001, decay: float = 0.9, eps: float = 1e-8):
        """
        Initialize RMSProp optimizer
        
        Args:
            learning_rate: Learning rate
            decay: Decay rate for moving average (typically 0.9)
            eps: Small constant to avoid division by zero
        """
        super().__init__(learning_rate)
        self.decay = decay
        self.eps = eps
        self.squared_grads = {}
        
    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        """
        Perform RMSProp parameter update
        
        Args:
            params: Parameters to update (modified in-place)
            grads: Gradients for each parameter
        """
        self.step_count += 1
        
        for param_name in params:
            if param_name in grads:
                # Initialize squared gradients if first time
                if param_name not in self.squared_grads:
                    self.squared_grads[param_name] = np.zeros_like(params[param_name])
                
                # Update exponential moving average of squared gradients
                # E[g²] = β*E[g²] + (1-β)*∇θ²
                self.squared_grads[param_name] = (
                    self.decay * self.squared_grads[param_name] + 
                    (1 - self.decay) * grads[param_name] ** 2
                )
                
                # Adaptive learning rate: η / √(E[g²] + ε)
                adapted_lr = self.learning_rate / (np.sqrt(self.squared_grads[param_name]) + self.eps)
                
                # Update parameters
                params[param_name] -= adapted_lr * grads[param_name]
                
    def reset(self) -> None:
        """Reset optimizer state including squared gradients"""
        super().reset()
        self.squared_grads = {}
        
    def get_state(self) -> Dict[str, Any]:
        """Get current optimizer state including squared gradients"""
        state = super().get_state()
        state.update({
            'decay': self.decay,
            'eps': self.eps,
            'squared_grads': copy.deepcopy(self.squared_grads)
        })
        return state

class Adam(BaseOptimizer):
    """
    Adaptive Moment Estimation (Adam)
    
    Update rule:
    m_t = β₁ * m_{t-1} + (1-β₁) * ∇θ
    v_t = β₂ * v_{t-1} + (1-β₂) * ∇θ²
    m̂_t = m_t / (1 - β₁ᵗ)  [bias correction]
    v̂_t = v_t / (1 - β₂ᵗ)  [bias correction]
    θ ← θ - η * m̂_t / (√v̂_t + ε)
    
    Combines momentum and adaptive learning rates with bias correction
    for the initial timesteps. Generally considered the most robust optimizer.
    """
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, eps: float = 1e-8):
        """
        Initialize Adam optimizer
        
        Args:
            learning_rate: Learning rate (typically 0.001)
            beta1: Exponential decay rate for first moment (typically 0.9)
            beta2: Exponential decay rate for second moment (typically 0.999)
            eps: Small constant to avoid division by zero
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}  # First moment (momentum)
        self.v = {}  # Second moment (squared gradients)
        
    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> None:
        """
        Perform Adam parameter update
        
        Args:
            params: Parameters to update (modified in-place)
            grads: Gradients for each parameter
        """
        self.step_count += 1
        
        for param_name in params:
            if param_name in grads:
                # Initialize moments if first time
                if param_name not in self.m:
                    self.m[param_name] = np.zeros_like(params[param_name])
                    self.v[param_name] = np.zeros_like(params[param_name])
                
                # Update biased first moment estimate: m = β₁*m + (1-β₁)*∇θ
                self.m[param_name] = (self.beta1 * self.m[param_name] + 
                                    (1 - self.beta1) * grads[param_name])
                
                # Update biased second raw moment estimate: v = β₂*v + (1-β₂)*∇θ²
                self.v[param_name] = (self.beta2 * self.v[param_name] + 
                                    (1 - self.beta2) * grads[param_name] ** 2)
                
                # Compute bias-corrected first moment estimate
                m_hat = self.m[param_name] / (1 - self.beta1 ** self.step_count)
                
                # Compute bias-corrected second raw moment estimate
                v_hat = self.v[param_name] / (1 - self.beta2 ** self.step_count)
                
                # Update parameters: θ = θ - η * m̂ / (√v̂ + ε)
                params[param_name] -= (self.learning_rate * m_hat / 
                                     (np.sqrt(v_hat) + self.eps))
                
    def reset(self) -> None:
        """Reset optimizer state including moments"""
        super().reset()
        self.m = {}
        self.v = {}
        
    def get_state(self) -> Dict[str, Any]:
        """Get current optimizer state including moments"""
        state = super().get_state()
        state.update({
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
            'm': copy.deepcopy(self.m),
            'v': copy.deepcopy(self.v)
        })
        return state

# Optimizer factory function
def get_optimizer(name: str, **kwargs) -> BaseOptimizer:
    """
    Factory function to create optimizers by name
    
    Args:
        name: Optimizer name ('sgd', 'momentum', 'nesterov', 'adagrad', 'rmsprop', 'adam')
        **kwargs: Optimizer-specific parameters
        
    Returns:
        Initialized optimizer instance
        
    Raises:
        ValueError: If optimizer name is not recognized
    """
    optimizers = {
        'sgd': SGD,
        'momentum': Momentum,
        'nesterov': Nesterov,
        'adagrad': AdaGrad,
        'rmsprop': RMSProp,
        'adam': Adam
    }
    
    name = name.lower()
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}. Available: {list(optimizers.keys())}")
    
    return optimizers[name](**kwargs)

# Helper functions for testing and analysis
def compare_optimizer_convergence(optimizers: List[BaseOptimizer], 
                                objective_func, gradient_func,
                                initial_params: np.ndarray,
                                num_steps: int = 1000) -> Dict[str, Dict]:
    """
    Compare convergence of different optimizers on a test function
    
    Args:
        optimizers: List of optimizer instances
        objective_func: Function to minimize f(x) -> scalar
        gradient_func: Gradient function ∇f(x) -> array
        initial_params: Starting parameter values
        num_steps: Number of optimization steps
        
    Returns:
        Dictionary containing convergence history for each optimizer
    """
    results = {}
    
    for optimizer in optimizers:
        # Reset optimizer state
        optimizer.reset()
        
        # Initialize parameters (copy to avoid modification, ensure float dtype)
        params = {'weights': initial_params.copy().astype(np.float64)}
        
        # Track convergence
        history = {
            'params': [],
            'losses': [],
            'gradients': []
        }
        
        for step in range(num_steps):
            # Compute loss and gradients
            loss = objective_func(params['weights'])
            grad = gradient_func(params['weights'])
            
            # Store history
            history['params'].append(params['weights'].copy())
            history['losses'].append(loss)
            history['gradients'].append(np.linalg.norm(grad))
            
            # Update parameters
            grads = {'weights': grad}
            optimizer.step(params, grads)
        
        # Store results
        optimizer_name = optimizer.__class__.__name__
        results[optimizer_name] = history
    
    return results

if __name__ == "__main__":
    # Simple demonstration of optimizers
    print("Optimizer Implementations Test")
    print("=" * 50)
    
    # Test function: simple quadratic f(x) = x²
    def quadratic(x):
        return np.sum(x ** 2)
    
    def quadratic_grad(x):
        return 2 * x
    
    # Initial parameters (ensure float dtype)
    x0 = np.array([5.0, -3.0], dtype=np.float64)
    
    # Create optimizers
    optimizers = [
        SGD(learning_rate=0.1),
        Momentum(learning_rate=0.1, momentum=0.9),
        Nesterov(learning_rate=0.1, momentum=0.9),
        AdaGrad(learning_rate=1.0),
        RMSProp(learning_rate=0.1),
        Adam(learning_rate=0.1)
    ]
    
    # Compare convergence
    results = compare_optimizer_convergence(
        optimizers, quadratic, quadratic_grad, x0, num_steps=50
    )
    
    # Print final results
    print(f"Starting point: {x0}")
    print(f"True minimum: [0, 0]")
    print(f"Initial loss: {quadratic(x0):.6f}")
    print()
    
    for name, history in results.items():
        final_params = history['params'][-1]
        final_loss = history['losses'][-1]
        print(f"{name:12} - Final: {final_params} (loss: {final_loss:.6f})")
    
    print("\nAll optimizers implemented successfully!") 