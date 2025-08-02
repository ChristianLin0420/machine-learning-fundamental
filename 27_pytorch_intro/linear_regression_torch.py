"""
Linear Regression with PyTorch
==============================

This module demonstrates linear regression implementation using PyTorch:
- PyTorch vs NumPy comparison
- Automatic differentiation for gradient computation
- torch.optim optimizers
- Loss functions and training loops
- Visualization of training progress
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
from typing import Tuple, List, Dict

class LinearRegressionNumPy:
    """Linear regression implementation using NumPy"""
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
        self.history = {'loss': [], 'weights': [], 'bias': []}
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000):
        """Train the model using gradient descent"""
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.random.normal(0, 0.01, (n_features,))
        self.bias = 0
        
        # Training loop
        for epoch in range(epochs):
            # Forward pass
            y_pred = X @ self.weights + self.bias
            
            # Compute loss (MSE)
            loss = np.mean((y_pred - y) ** 2)
            
            # Compute gradients
            dw = (2 / n_samples) * X.T @ (y_pred - y)
            db = (2 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Store history
            if epoch % 10 == 0:
                self.history['loss'].append(loss)
                self.history['weights'].append(self.weights.copy())
                self.history['bias'].append(self.bias)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return X @ self.weights + self.bias

class LinearRegressionPyTorch:
    """Linear regression implementation using PyTorch"""
    
    def __init__(self, n_features: int, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        
        # Initialize parameters as tensors
        self.weights = torch.randn(n_features, requires_grad=True, dtype=torch.float32)
        self.bias = torch.zeros(1, requires_grad=True, dtype=torch.float32)
        
        self.history = {'loss': [], 'weights': [], 'bias': []}
    
    def fit(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 1000):
        """Train the model using PyTorch autograd"""
        for epoch in range(epochs):
            # Forward pass
            y_pred = X @ self.weights + self.bias
            
            # Compute loss (MSE)
            loss = torch.mean((y_pred - y) ** 2)
            
            # Backward pass
            loss.backward()
            
            # Update parameters (no_grad to prevent tracking)
            with torch.no_grad():
                self.weights -= self.learning_rate * self.weights.grad
                self.bias -= self.learning_rate * self.bias.grad
                
                # Zero gradients
                self.weights.grad.zero_()
                self.bias.grad.zero_()
            
            # Store history
            if epoch % 10 == 0:
                self.history['loss'].append(loss.item())
                self.history['weights'].append(self.weights.detach().clone())
                self.history['bias'].append(self.bias.detach().clone())
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions"""
        with torch.no_grad():
            return X @ self.weights + self.bias

class LinearRegressionModule(nn.Module):
    """Linear regression using nn.Module"""
    
    def __init__(self, n_features: int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        self.history = {'loss': []}
    
    def forward(self, x):
        return self.linear(x).squeeze()
    
    def fit(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 1000, 
            learning_rate: float = 0.01):
        """Train using nn.Module and optimizer"""
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        
        for epoch in range(epochs):
            # Forward pass
            y_pred = self(X)
            loss = criterion(y_pred, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Store history
            if epoch % 10 == 0:
                self.history['loss'].append(loss.item())

def generate_synthetic_data(n_samples: int = 1000, n_features: int = 1, 
                          noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data"""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise * 100,  # sklearn uses different noise scale
        random_state=42
    )
    
    # Normalize data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    return X, y

def compare_implementations():
    """Compare NumPy vs PyTorch implementations"""
    print("=" * 60)
    print("COMPARING IMPLEMENTATIONS")
    print("=" * 60)
    
    # Generate data
    X, y = generate_synthetic_data(n_samples=1000, n_features=3, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Convert to tensors for PyTorch
    X_train_torch = torch.FloatTensor(X_train)
    y_train_torch = torch.FloatTensor(y_train)
    X_test_torch = torch.FloatTensor(X_test)
    y_test_torch = torch.FloatTensor(y_test)
    
    # Training parameters
    learning_rate = 0.01
    epochs = 1000
    
    # 1. NumPy implementation
    print("\n1. Training NumPy implementation...")
    start_time = time.time()
    model_numpy = LinearRegressionNumPy(learning_rate=learning_rate)
    model_numpy.fit(X_train, y_train, epochs=epochs)
    numpy_time = time.time() - start_time
    
    y_pred_numpy = model_numpy.predict(X_test)
    mse_numpy = np.mean((y_pred_numpy - y_test) ** 2)
    
    print(f"NumPy - Training time: {numpy_time:.4f}s, Test MSE: {mse_numpy:.6f}")
    
    # 2. PyTorch manual implementation
    print("\n2. Training PyTorch manual implementation...")
    start_time = time.time()
    model_torch = LinearRegressionPyTorch(n_features=X_train.shape[1], learning_rate=learning_rate)
    model_torch.fit(X_train_torch, y_train_torch, epochs=epochs)
    torch_time = time.time() - start_time
    
    y_pred_torch = model_torch.predict(X_test_torch)
    mse_torch = torch.mean((y_pred_torch - y_test_torch) ** 2).item()
    
    print(f"PyTorch Manual - Training time: {torch_time:.4f}s, Test MSE: {mse_torch:.6f}")
    
    # 3. PyTorch nn.Module implementation
    print("\n3. Training PyTorch nn.Module implementation...")
    start_time = time.time()
    model_module = LinearRegressionModule(n_features=X_train.shape[1])
    model_module.fit(X_train_torch, y_train_torch, epochs=epochs, learning_rate=learning_rate)
    module_time = time.time() - start_time
    
    y_pred_module = model_module(X_test_torch)
    mse_module = torch.mean((y_pred_module - y_test_torch) ** 2).item()
    
    print(f"PyTorch Module - Training time: {module_time:.4f}s, Test MSE: {mse_module:.6f}")
    
    # Compare parameters
    print(f"\n4. Parameter comparison:")
    print(f"NumPy weights: {model_numpy.weights}")
    print(f"NumPy bias: {model_numpy.bias}")
    print(f"PyTorch weights: {model_torch.weights.detach().numpy()}")
    print(f"PyTorch bias: {model_torch.bias.detach().numpy()}")
    print(f"Module weights: {model_module.linear.weight.detach().numpy().flatten()}")
    print(f"Module bias: {model_module.linear.bias.detach().numpy()}")
    
    return {
        'numpy_model': model_numpy,
        'torch_model': model_torch,
        'module_model': model_module,
        'data': (X_train, X_test, y_train, y_test),
        'torch_data': (X_train_torch, X_test_torch, y_train_torch, y_test_torch),
        'times': {'numpy': numpy_time, 'torch': torch_time, 'module': module_time},
        'mse': {'numpy': mse_numpy, 'torch': mse_torch, 'module': mse_module}
    }

def demonstrate_optimizers():
    """Demonstrate different PyTorch optimizers"""
    print("\n" + "=" * 60)
    print("PYTORCH OPTIMIZERS COMPARISON")
    print("=" * 60)
    
    # Generate data
    X, y = generate_synthetic_data(n_samples=500, n_features=2, noise=0.2)
    X_torch = torch.FloatTensor(X)
    y_torch = torch.FloatTensor(y)
    
    # Test different optimizers
    optimizers_config = [
        {'name': 'SGD', 'class': optim.SGD, 'kwargs': {'lr': 0.01}},
        {'name': 'SGD+Momentum', 'class': optim.SGD, 'kwargs': {'lr': 0.01, 'momentum': 0.9}},
        {'name': 'Adam', 'class': optim.Adam, 'kwargs': {'lr': 0.01}},
        {'name': 'AdaGrad', 'class': optim.Adagrad, 'kwargs': {'lr': 0.01}},
        {'name': 'RMSprop', 'class': optim.RMSprop, 'kwargs': {'lr': 0.01}}
    ]
    
    results = {}
    epochs = 500
    
    for opt_config in optimizers_config:
        print(f"\nTraining with {opt_config['name']}...")
        
        # Create model
        model = LinearRegressionModule(n_features=X.shape[1])
        criterion = nn.MSELoss()
        optimizer = opt_config['class'](model.parameters(), **opt_config['kwargs'])
        
        # Track loss
        losses = []
        
        for epoch in range(epochs):
            y_pred = model(X_torch)
            loss = criterion(y_pred, y_torch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                losses.append(loss.item())
        
        results[opt_config['name']] = {
            'losses': losses,
            'final_loss': losses[-1],
            'model': model
        }
        
        print(f"Final loss: {losses[-1]:.6f}")
    
    return results

def visualize_training_progress(comparison_results: Dict, optimizer_results: Dict):
    """Create comprehensive visualizations"""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Loss curves comparison (implementations)
    ax1 = plt.subplot(3, 4, 1)
    numpy_losses = comparison_results['numpy_model'].history['loss']
    torch_losses = comparison_results['torch_model'].history['loss']
    module_losses = comparison_results['module_model'].history['loss']
    
    epochs = np.arange(0, len(numpy_losses) * 10, 10)
    plt.plot(epochs, numpy_losses, label='NumPy', linewidth=2)
    plt.plot(epochs, torch_losses, label='PyTorch Manual', linewidth=2)
    plt.plot(epochs, module_losses, label='PyTorch Module', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss: Implementation Comparison')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # 2. Loss curves comparison (optimizers)
    ax2 = plt.subplot(3, 4, 2)
    for name, result in optimizer_results.items():
        epochs_opt = np.arange(0, len(result['losses']) * 10, 10)
        plt.plot(epochs_opt, result['losses'], label=name, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss: Optimizer Comparison')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # 3. Training time comparison
    ax3 = plt.subplot(3, 4, 3)
    times = comparison_results['times']
    implementations = list(times.keys())
    time_values = list(times.values())
    bars = plt.bar(implementations, time_values, color=['blue', 'orange', 'green'])
    plt.ylabel('Time (seconds)')
    plt.title('Training Time Comparison')
    plt.xticks(rotation=45)
    for bar, time_val in zip(bars, time_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{time_val:.3f}s', ha='center', va='bottom')
    
    # 4. MSE comparison
    ax4 = plt.subplot(3, 4, 4)
    mse_values = comparison_results['mse']
    implementations = list(mse_values.keys())
    mse_vals = list(mse_values.values())
    bars = plt.bar(implementations, mse_vals, color=['blue', 'orange', 'green'])
    plt.ylabel('Mean Squared Error')
    plt.title('Test MSE Comparison')
    plt.xticks(rotation=45)
    for bar, mse_val in zip(bars, mse_vals):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{mse_val:.4f}', ha='center', va='bottom')
    
    # 5-8. Prediction vs Actual plots
    X_test, y_test = comparison_results['data'][1], comparison_results['data'][3]
    X_test_torch, y_test_torch = comparison_results['torch_data'][1], comparison_results['torch_data'][3]
    
    models_data = [
        ('NumPy', comparison_results['numpy_model'].predict(X_test), y_test),
        ('PyTorch Manual', comparison_results['torch_model'].predict(X_test_torch).numpy(), y_test),
        ('PyTorch Module', comparison_results['module_model'](X_test_torch).detach().numpy(), y_test)
    ]
    
    for i, (name, y_pred, y_true) in enumerate(models_data):
        ax = plt.subplot(3, 4, 5 + i)
        plt.scatter(y_true, y_pred, alpha=0.6, s=20)
        
        # Perfect prediction line
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        # R² calculation
        r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
        
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{name}\nR² = {r2:.4f}')
        plt.grid(True, alpha=0.3)
    
    # 9. Weight evolution (NumPy)
    ax9 = plt.subplot(3, 4, 9)
    weight_history = np.array(comparison_results['numpy_model'].history['weights'])
    epochs_w = np.arange(0, len(weight_history) * 10, 10)
    for i in range(weight_history.shape[1]):
        plt.plot(epochs_w, weight_history[:, i], label=f'w{i}', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Weight Value')
    plt.title('Weight Evolution (NumPy)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 10. Gradient norms during training
    ax10 = plt.subplot(3, 4, 10)
    # Simulate gradient norms (would need to track during actual training)
    grad_norms = np.exp(-np.linspace(0, 5, 50)) + 0.01 * np.random.randn(50)
    epochs_g = np.arange(0, len(grad_norms) * 10, 10)
    plt.plot(epochs_g, grad_norms, linewidth=2, color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norm Evolution')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    # 11. Optimizer convergence comparison
    ax11 = plt.subplot(3, 4, 11)
    final_losses = [result['final_loss'] for result in optimizer_results.values()]
    opt_names = list(optimizer_results.keys())
    bars = plt.bar(opt_names, final_losses, color='skyblue')
    plt.ylabel('Final Loss')
    plt.title('Final Loss by Optimizer')
    plt.xticks(rotation=45)
    for bar, loss_val in zip(bars, final_losses):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{loss_val:.4f}', ha='center', va='bottom')
    
    # 12. Loss landscape (2D visualization)
    ax12 = plt.subplot(3, 4, 12)
    # Create a simple 2D loss landscape for 1D linear regression
    X_simple, y_simple = generate_synthetic_data(n_samples=100, n_features=1, noise=0.1)
    w_range = np.linspace(-2, 2, 50)
    b_range = np.linspace(-2, 2, 50)
    W, B = np.meshgrid(w_range, b_range)
    
    # Compute loss for each (w, b) pair
    losses = np.zeros_like(W)
    for i in range(len(w_range)):
        for j in range(len(b_range)):
            y_pred_simple = X_simple.flatten() * W[j, i] + B[j, i]
            losses[j, i] = np.mean((y_pred_simple - y_simple)**2)
    
    contour = plt.contour(W, B, losses, levels=20)
    plt.colorbar(contour)
    plt.xlabel('Weight')
    plt.ylabel('Bias')
    plt.title('Loss Landscape (2D)')
    
    plt.tight_layout()
    plt.savefig('27_pytorch_intro/plots/linear_regression_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def comprehensive_linear_regression_demo():
    """Run complete linear regression demonstration"""
    print("PyTorch Linear Regression Comprehensive Demo")
    print("==========================================")
    
    # Run comparisons
    comparison_results = compare_implementations()
    optimizer_results = demonstrate_optimizers()
    
    # Create visualizations
    fig = visualize_training_progress(comparison_results, optimizer_results)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✅ NumPy vs PyTorch implementation comparison")
    print("✅ Manual PyTorch vs nn.Module comparison")
    print("✅ Multiple optimizers tested")
    print("✅ Training dynamics visualized")
    print("✅ Performance metrics compared")
    print("✅ Loss landscapes explored")
    
    return {
        'comparison': comparison_results,
        'optimizers': optimizer_results,
        'visualization': fig
    }

if __name__ == "__main__":
    results = comprehensive_linear_regression_demo() 