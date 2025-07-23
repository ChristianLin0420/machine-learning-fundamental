"""
Binary Perceptron Implementation from Scratch

This module implements the classic perceptron algorithm for binary classification
using only numpy. Includes convergence analysis and visualization capabilities.

The perceptron learning rule:
w ← w + η * y_i * x_i if y_i * (w · x_i) ≤ 0

Where:
- w: weight vector
- η: learning rate
- y_i: true label {-1, 1}
- x_i: input vector
"""

import numpy as np
import matplotlib.pyplot as plt
from synthetic_data import make_linearly_separable_data, make_noisy_linear_data


class Perceptron:
    """
    Binary Perceptron Classifier
    
    A linear classifier that learns a separating hyperplane for binary classification
    using the perceptron learning algorithm.
    """
    
    def __init__(self, learning_rate=1.0, max_epochs=1000, random_state=42):
        """
        Initialize the perceptron.
        
        Args:
            learning_rate (float): Learning rate (η)
            max_epochs (int): Maximum number of training epochs
            random_state (int): Random seed for weight initialization
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.random_state = random_state
        
        # Will be set during training
        self.weights = None
        self.bias = None
        self.history = {
            'weights': [],
            'bias': [],
            'errors': [],
            'converged_epoch': None
        }
    
    def _add_bias_term(self, X):
        """Add bias term to input matrix."""
        return np.c_[np.ones(X.shape[0]), X]
    
    def _initialize_weights(self, n_features):
        """
        Initialize weights using different strategies.
        
        Args:
            n_features (int): Number of features (including bias)
        """
        np.random.seed(self.random_state)
        
        # Initialize weights to small random values
        self.weights = np.random.normal(0, 0.01, n_features)
        
        # Alternative initializations (uncomment to try):
        # self.weights = np.zeros(n_features)  # Zero initialization
        # self.weights = np.random.uniform(-1, 1, n_features)  # Uniform [-1, 1]
    
    def _predict_sample(self, x):
        """
        Predict class for a single sample.
        
        Args:
            x (np.ndarray): Single sample with bias term
            
        Returns:
            int: Predicted class {-1, 1}
        """
        activation = np.dot(self.weights, x)
        return 1 if activation > 0 else -1
    
    def predict(self, X):
        """
        Predict classes for multiple samples.
        
        Args:
            X (np.ndarray): Input features (n_samples, n_features)
            
        Returns:
            np.ndarray: Predicted classes
        """
        if self.weights is None:
            raise ValueError("Model must be trained before making predictions")
        
        X_with_bias = self._add_bias_term(X)
        predictions = []
        
        for x in X_with_bias:
            predictions.append(self._predict_sample(x))
        
        return np.array(predictions)
    
    def fit(self, X, y, verbose=True):
        """
        Train the perceptron using the perceptron learning algorithm.
        
        Args:
            X (np.ndarray): Training features (n_samples, n_features)
            y (np.ndarray): Training labels {-1, 1}
            verbose (bool): Whether to print training progress
            
        Returns:
            self: Returns self for method chaining
        """
        # Add bias term to input
        X_with_bias = self._add_bias_term(X)
        n_samples, n_features = X_with_bias.shape
        
        # Initialize weights
        self._initialize_weights(n_features)
        
        # Reset history
        self.history = {
            'weights': [],
            'bias': [],
            'errors': [],
            'converged_epoch': None
        }
        
        if verbose:
            print(f"Training perceptron with {n_samples} samples, {n_features-1} features")
            print(f"Learning rate: {self.learning_rate}, Max epochs: {self.max_epochs}")
        
        # Training loop
        for epoch in range(self.max_epochs):
            errors = 0
            
            # Store current state
            self.history['weights'].append(self.weights.copy())
            
            # Go through each training sample
            for i in range(n_samples):
                x_i = X_with_bias[i]
                y_i = y[i]
                
                # Make prediction
                prediction = self._predict_sample(x_i)
                
                # Check if misclassified
                if prediction != y_i:
                    # Apply perceptron learning rule
                    # w ← w + η * y_i * x_i
                    self.weights += self.learning_rate * y_i * x_i
                    errors += 1
            
            self.history['errors'].append(errors)
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: {errors} errors")
            
            # Check for convergence (no errors)
            if errors == 0:
                self.history['converged_epoch'] = epoch
                if verbose:
                    print(f"Converged after {epoch + 1} epochs!")
                break
        
        if self.history['converged_epoch'] is None and verbose:
            print(f"Did not converge after {self.max_epochs} epochs")
        
        return self
    
    def score(self, X, y):
        """
        Calculate accuracy on given data.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): True labels
            
        Returns:
            float: Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def get_decision_boundary_params(self):
        """
        Get parameters for plotting decision boundary.
        For 2D case: w0 + w1*x1 + w2*x2 = 0
        Solved for x2: x2 = -(w0 + w1*x1) / w2
        
        Returns:
            tuple: (slope, intercept) for plotting
        """
        if self.weights is None or len(self.weights) != 3:
            raise ValueError("Model must be trained with 2D data")
        
        w0, w1, w2 = self.weights
        
        if abs(w2) < 1e-10:  # Nearly vertical line
            return None, -w0/w1  # x1 = -w0/w1
        
        slope = -w1 / w2
        intercept = -w0 / w2
        
        return slope, intercept


def plot_perceptron_convergence(perceptron, X, y, title="Perceptron Convergence", save_path=None):
    """
    Plot the convergence behavior of the perceptron.
    
    Args:
        perceptron (Perceptron): Trained perceptron
        X (np.ndarray): Training data
        y (np.ndarray): Training labels
        title (str): Plot title
        save_path (str): Path to save plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Error curve
    axes[0].plot(perceptron.history['errors'], 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Number of Errors')
    axes[0].set_title('Training Errors Over Time')
    axes[0].grid(True, alpha=0.3)
    
    if perceptron.history['converged_epoch'] is not None:
        axes[0].axvline(x=perceptron.history['converged_epoch'], color='r', 
                       linestyle='--', label=f'Converged at epoch {perceptron.history["converged_epoch"]}')
        axes[0].legend()
    
    # Plot 2: Weight evolution (for 2D case)
    if len(perceptron.weights) == 3:  # bias + 2 features
        weights_history = np.array(perceptron.history['weights'])
        axes[1].plot(weights_history[:, 0], label='Bias (w0)', linewidth=2)
        axes[1].plot(weights_history[:, 1], label='Weight 1 (w1)', linewidth=2)
        axes[1].plot(weights_history[:, 2], label='Weight 2 (w2)', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Weight Value')
        axes[1].set_title('Weight Evolution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Final decision boundary
    if X.shape[1] == 2:  # 2D data
        # Plot data points
        mask_pos = y == 1
        mask_neg = y == -1
        
        axes[2].scatter(X[mask_pos, 0], X[mask_pos, 1], c='red', marker='o', 
                       label='Class +1', alpha=0.7, s=50)
        axes[2].scatter(X[mask_neg, 0], X[mask_neg, 1], c='blue', marker='s', 
                       label='Class -1', alpha=0.7, s=50)
        
        # Plot decision boundary
        try:
            slope, intercept = perceptron.get_decision_boundary_params()
            x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
            
            if slope is not None:
                y_boundary = slope * x_range + intercept
                axes[2].plot(x_range, y_boundary, 'g-', linewidth=2, 
                           label='Decision Boundary')
            else:
                # Vertical line
                axes[2].axvline(x=intercept, color='g', linewidth=2, 
                              label='Decision Boundary')
        except:
            pass
        
        axes[2].set_xlabel('Feature 1')
        axes[2].set_ylabel('Feature 2')
        axes[2].set_title('Final Decision Boundary')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def compare_datasets():
    """
    Compare perceptron performance on different types of datasets.
    """
    # Create plots directory
    import os
    os.makedirs("plots", exist_ok=True)
    
    # Test datasets
    datasets = {
        "Linearly Separable": make_linearly_separable_data(n_samples=100, random_state=42),
        "Noisy Linear": make_noisy_linear_data(n_samples=100, noise_level=0.2, random_state=42),
        "Very Noisy": make_noisy_linear_data(n_samples=100, noise_level=0.4, random_state=42)
    }
    
    results = {}
    
    for name, (X, y) in datasets.items():
        print(f"\n{'='*50}")
        print(f"Testing on {name} Data")
        print(f"{'='*50}")
        
        # Train perceptron
        perceptron = Perceptron(learning_rate=1.0, max_epochs=1000, random_state=42)
        perceptron.fit(X, y, verbose=True)
        
        # Evaluate
        accuracy = perceptron.score(X, y)
        converged = perceptron.history['converged_epoch'] is not None
        
        results[name] = {
            'accuracy': accuracy,
            'converged': converged,
            'epochs': perceptron.history['converged_epoch'] if converged else 1000,
            'final_errors': perceptron.history['errors'][-1]
        }
        
        print(f"Final accuracy: {accuracy:.3f}")
        print(f"Converged: {converged}")
        if converged:
            print(f"Converged in {perceptron.history['converged_epoch']} epochs")
        else:
            print(f"Final errors: {perceptron.history['errors'][-1]}")
        
        # Plot convergence
        plot_perceptron_convergence(
            perceptron, X, y, 
            title=f"Perceptron on {name} Data",
            save_path=f"plots/perceptron_{name.lower().replace(' ', '_')}.png"
        )
    
    # Summary plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in names]
    epochs = [results[name]['epochs'] for name in names]
    
    # Accuracy comparison
    bars1 = axes[0].bar(names, accuracies, color=['green', 'orange', 'red'], alpha=0.7)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Final Accuracy Comparison')
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True, alpha=0.3)
    
    # Add accuracy values on bars
    for bar, acc in zip(bars1, accuracies):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{acc:.3f}', ha='center', va='bottom')
    
    # Epochs to convergence
    bars2 = axes[1].bar(names, epochs, color=['green', 'orange', 'red'], alpha=0.7)
    axes[1].set_ylabel('Epochs to Convergence')
    axes[1].set_title('Convergence Speed Comparison')
    axes[1].grid(True, alpha=0.3)
    
    # Add epoch values on bars
    for bar, epoch in zip(bars2, epochs):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
                    f'{epoch}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("plots/perceptron_comparison_summary.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return results


def test_learning_rates():
    """
    Test different learning rates and their effect on convergence.
    """
    print(f"\n{'='*50}")
    print("Testing Different Learning Rates")
    print(f"{'='*50}")
    
    # Generate linearly separable data
    X, y = make_linearly_separable_data(n_samples=100, random_state=42)
    
    learning_rates = [0.1, 0.5, 1.0, 2.0, 5.0]
    results = []
    
    fig, axes = plt.subplots(1, len(learning_rates), figsize=(20, 4))
    
    for i, lr in enumerate(learning_rates):
        perceptron = Perceptron(learning_rate=lr, max_epochs=1000, random_state=42)
        perceptron.fit(X, y, verbose=False)
        
        accuracy = perceptron.score(X, y)
        converged = perceptron.history['converged_epoch'] is not None
        epochs = perceptron.history['converged_epoch'] if converged else 1000
        
        results.append({
            'lr': lr,
            'accuracy': accuracy,
            'converged': converged,
            'epochs': epochs
        })
        
        print(f"Learning rate {lr}: Accuracy={accuracy:.3f}, "
              f"Converged={converged}, Epochs={epochs}")
        
        # Plot error curve
        axes[i].plot(perceptron.history['errors'], 'b-', linewidth=2)
        axes[i].set_title(f'LR = {lr}')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Errors')
        axes[i].grid(True, alpha=0.3)
        
        if converged:
            axes[i].axvline(x=epochs, color='r', linestyle='--', alpha=0.7)
    
    plt.suptitle('Effect of Learning Rate on Convergence', fontsize=16)
    plt.tight_layout()
    plt.savefig("plots/learning_rate_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return results


if __name__ == "__main__":
    # Create plots directory
    import os
    os.makedirs("plots", exist_ok=True)
    
    print("Perceptron Algorithm Implementation")
    print("="*50)
    
    # Test 1: Compare different datasets
    dataset_results = compare_datasets()
    
    # Test 2: Learning rate analysis
    lr_results = test_learning_rates()
    
    # Test 3: Simple example
    print(f"\n{'='*50}")
    print("Simple Example: Manual Training")
    print(f"{'='*50}")
    
    # Create simple 2D example
    X_simple = np.array([[1, 1], [2, 2], [1, -1], [2, -2]])
    y_simple = np.array([1, 1, -1, -1])
    
    print("Training data:")
    for i, (x, label) in enumerate(zip(X_simple, y_simple)):
        print(f"  Sample {i+1}: {x} → {label}")
    
    # Train perceptron
    perceptron = Perceptron(learning_rate=1.0, max_epochs=100, random_state=42)
    perceptron.fit(X_simple, y_simple, verbose=True)
    
    print(f"\nFinal weights: {perceptron.weights}")
    print(f"Final accuracy: {perceptron.score(X_simple, y_simple):.3f}")
    
    # Test predictions
    test_points = np.array([[0, 0], [1, 0], [0, 1], [-1, -1]])
    predictions = perceptron.predict(test_points)
    
    print("\nTest predictions:")
    for point, pred in zip(test_points, predictions):
        print(f"  {point} → {pred}")
    
    print("\nPerceptron implementation complete!") 