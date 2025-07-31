"""
Regularization Techniques for Neural Networks - NumPy Implementation

This module implements various regularization techniques to prevent overfitting:
- L1 Regularization (Lasso)
- L2 Regularization (Ridge/Weight Decay)
- Dropout
- Early Stopping Integration

Author: ML Fundamentals Course
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Callable
from enum import Enum
import warnings

class ActivationType(Enum):
    SIGMOID = "sigmoid"
    RELU = "relu"
    TANH = "tanh"
    LINEAR = "linear"

class RegularizationType(Enum):
    NONE = "none"
    L1 = "l1"
    L2 = "l2"
    L1_L2 = "elastic_net"

class ActivationFunctions:
    """Collection of activation functions and their derivatives"""
    
    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function with numerical stability"""
        x = np.clip(x, -500, 500)  # Prevent overflow
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid function"""
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """ReLU activation function"""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of ReLU function"""
        return (x > 0).astype(float)
    
    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        """Tanh activation function"""
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of tanh function"""
        return 1 - np.tanh(x)**2
    
    @staticmethod
    def linear(x: np.ndarray) -> np.ndarray:
        """Linear activation function"""
        return x
    
    @staticmethod
    def linear_derivative(x: np.ndarray) -> np.ndarray:
        """Derivative of linear function"""
        return np.ones_like(x)

class RegularizedMLP:
    """
    Multi-Layer Perceptron with comprehensive regularization techniques
    
    Features:
    - L1, L2, and Elastic Net regularization
    - Dropout during training
    - Early stopping integration
    - Training history tracking
    - Flexible architecture
    """
    
    def __init__(self, 
                 layers: List[int],
                 activation: str = 'relu',
                 output_activation: str = 'sigmoid',
                 l1_lambda: float = 0.0,
                 l2_lambda: float = 0.0,
                 dropout_rate: float = 0.0,
                 random_state: int = 42):
        """
        Initialize the regularized MLP
        
        Args:
            layers: List of layer sizes [input_size, hidden1, hidden2, ..., output_size]
            activation: Activation function for hidden layers
            output_activation: Activation function for output layer
            l1_lambda: L1 regularization strength
            l2_lambda: L2 regularization strength
            dropout_rate: Dropout probability (0 = no dropout)
            random_state: Random seed for reproducibility
        """
        self.layers = layers
        self.num_layers = len(layers)
        self.activation_type = ActivationType(activation)
        self.output_activation_type = ActivationType(output_activation)
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        
        # Set random seed
        np.random.seed(random_state)
        
        # Get activation functions
        self.activation_func = getattr(ActivationFunctions, activation)
        self.activation_derivative = getattr(ActivationFunctions, f"{activation}_derivative")
        self.output_activation_func = getattr(ActivationFunctions, output_activation)
        self.output_activation_derivative = getattr(ActivationFunctions, f"{output_activation}_derivative")
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self._initialize_parameters()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'regularization_loss': []
        }
        
        # Cache for forward pass
        self.z_values = []  # Pre-activation values
        self.a_values = []  # Post-activation values
        self.dropout_masks = []  # Dropout masks
        
    def _initialize_parameters(self):
        """Initialize weights and biases using Xavier/He initialization"""
        for i in range(self.num_layers - 1):
            input_size = self.layers[i]
            output_size = self.layers[i + 1]
            
            # Xavier initialization for sigmoid/tanh, He for ReLU
            if self.activation_type == ActivationType.RELU:
                # He initialization
                std = np.sqrt(2.0 / input_size)
            else:
                # Xavier initialization
                std = np.sqrt(2.0 / (input_size + output_size))
            
            weight = np.random.normal(0, std, (input_size, output_size))
            bias = np.zeros((1, output_size))
            
            self.weights.append(weight)
            self.biases.append(bias)
    
    def _apply_dropout(self, x: np.ndarray, training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply dropout to input
        
        Args:
            x: Input array
            training: Whether in training mode
            
        Returns:
            Tuple of (dropped_x, dropout_mask)
        """
        if not training or self.dropout_rate == 0:
            return x, np.ones_like(x)
        
        # Generate dropout mask
        keep_prob = 1 - self.dropout_rate
        mask = np.random.binomial(1, keep_prob, size=x.shape) / keep_prob
        
        return x * mask, mask
    
    def _compute_regularization_loss(self) -> float:
        """Compute L1 and L2 regularization losses"""
        l1_loss = 0.0
        l2_loss = 0.0
        
        for weight in self.weights:
            if self.l1_lambda > 0:
                l1_loss += np.sum(np.abs(weight))
            if self.l2_lambda > 0:
                l2_loss += np.sum(weight ** 2)
        
        return self.l1_lambda * l1_loss + self.l2_lambda * l2_loss
    
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through the network
        
        Args:
            X: Input data of shape (batch_size, input_features)
            training: Whether in training mode (affects dropout)
            
        Returns:
            Network output
        """
        # Clear cache
        self.z_values = []
        self.a_values = []
        self.dropout_masks = []
        
        # Input layer
        current_input = X
        self.a_values.append(current_input)
        
        # Hidden layers
        for i in range(self.num_layers - 2):
            # Linear transformation
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            
            # Activation
            a = self.activation_func(z)
            
            # Apply dropout
            a, mask = self._apply_dropout(a, training)
            
            self.a_values.append(a)
            self.dropout_masks.append(mask)
            current_input = a
        
        # Output layer
        z_output = np.dot(current_input, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z_output)
        output = self.output_activation_func(z_output)
        self.a_values.append(output)
        
        return output
    
    def backward(self, X: np.ndarray, y: np.ndarray, output: np.ndarray) -> List[np.ndarray]:
        """
        Backward pass to compute gradients
        
        Args:
            X: Input data
            y: True labels
            output: Network output from forward pass
            
        Returns:
            List of weight gradients
        """
        batch_size = X.shape[0]
        
        # Initialize gradient lists
        weight_gradients = [np.zeros_like(w) for w in self.weights]
        bias_gradients = [np.zeros_like(b) for b in self.biases]
        
        # Output layer error
        if self.output_activation_type == ActivationType.SIGMOID:
            delta = output - y  # For sigmoid + binary cross-entropy
        else:
            # General case: dL/da * da/dz
            delta = (output - y) * self.output_activation_derivative(self.z_values[-1])
        
        # Output layer gradients
        weight_gradients[-1] = np.dot(self.a_values[-2].T, delta) / batch_size
        bias_gradients[-1] = np.sum(delta, axis=0, keepdims=True) / batch_size
        
        # Add L2 regularization to weights
        if self.l2_lambda > 0:
            weight_gradients[-1] += 2 * self.l2_lambda * self.weights[-1]
        
        # Add L1 regularization to weights
        if self.l1_lambda > 0:
            weight_gradients[-1] += self.l1_lambda * np.sign(self.weights[-1])
        
        # Backpropagate through hidden layers
        for i in range(self.num_layers - 2, 0, -1):
            # Error propagation
            delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(self.z_values[i-1])
            
            # Apply dropout mask if used during forward pass
            if i-1 < len(self.dropout_masks):
                delta = delta * self.dropout_masks[i-1]
            
            # Gradients
            weight_gradients[i-1] = np.dot(self.a_values[i-1].T, delta) / batch_size
            bias_gradients[i-1] = np.sum(delta, axis=0, keepdims=True) / batch_size
            
            # Add regularization
            if self.l2_lambda > 0:
                weight_gradients[i-1] += 2 * self.l2_lambda * self.weights[i-1]
            if self.l1_lambda > 0:
                weight_gradients[i-1] += self.l1_lambda * np.sign(self.weights[i-1])
        
        return weight_gradients, bias_gradients
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute loss including regularization"""
        batch_size = y_true.shape[0]
        
        # Primary loss (binary cross-entropy or MSE)
        if self.output_activation_type == ActivationType.SIGMOID:
            # Binary cross-entropy
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Prevent log(0)
            primary_loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        else:
            # Mean squared error
            primary_loss = np.mean((y_true - y_pred) ** 2)
        
        # Regularization loss
        regularization_loss = self._compute_regularization_loss()
        
        return primary_loss + regularization_loss
    
    def compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute classification accuracy"""
        if self.output_activation_type == ActivationType.SIGMOID:
            predictions = (y_pred > 0.5).astype(int)
        else:
            predictions = np.argmax(y_pred, axis=1)
            y_true = np.argmax(y_true, axis=1) if y_true.ndim > 1 else y_true
        
        return np.mean(predictions == y_true.ravel())
    
    def fit(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = 100,
            learning_rate: float = 0.01,
            batch_size: int = 32,
            early_stopping: Optional[Callable] = None,
            verbose: bool = True) -> Dict:
        """
        Train the neural network
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            early_stopping: Early stopping callback
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        # Reset history
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'regularization_loss': []
        }
        
        n_samples = X_train.shape[0]
        n_batches = max(1, n_samples // batch_size)
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            epoch_loss = 0.0
            epoch_reg_loss = 0.0
            
            # Mini-batch training
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                X_batch = X_train_shuffled[start_idx:end_idx]
                y_batch = y_train_shuffled[start_idx:end_idx]
                
                # Forward pass
                output = self.forward(X_batch, training=True)
                
                # Compute loss
                loss = self.compute_loss(y_batch, output)
                reg_loss = self._compute_regularization_loss()
                
                epoch_loss += loss
                epoch_reg_loss += reg_loss
                
                # Backward pass
                weight_grads, bias_grads = self.backward(X_batch, y_batch, output)
                
                # Update parameters
                for i in range(len(self.weights)):
                    self.weights[i] -= learning_rate * weight_grads[i]
                    self.biases[i] -= learning_rate * bias_grads[i]
            
            # Average losses
            epoch_loss /= n_batches
            epoch_reg_loss /= n_batches
            
            # Compute training metrics
            train_output = self.forward(X_train, training=False)
            train_loss = self.compute_loss(y_train, train_output)
            train_accuracy = self.compute_accuracy(y_train, train_output)
            
            # Store training metrics
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_accuracy)
            self.history['regularization_loss'].append(epoch_reg_loss)
            
            # Compute validation metrics
            if X_val is not None and y_val is not None:
                val_output = self.forward(X_val, training=False)
                val_loss = self.compute_loss(y_val, val_output)
                val_accuracy = self.compute_accuracy(y_val, val_output)
                
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_accuracy)
                
                # Early stopping check
                if early_stopping is not None:
                    if early_stopping(val_loss):
                        if verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                if X_val is not None:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, "
                          f"Reg Loss: {epoch_reg_loss:.4f}")
                else:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                          f"Reg Loss: {epoch_reg_loss:.4f}")
        
        return self.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data"""
        return self.forward(X, training=False)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        return self.predict(X)
    
    def get_weight_statistics(self) -> Dict:
        """Get statistics about the weights"""
        all_weights = np.concatenate([w.flatten() for w in self.weights])
        
        return {
            'mean': np.mean(all_weights),
            'std': np.std(all_weights),
            'l1_norm': np.sum(np.abs(all_weights)),
            'l2_norm': np.sqrt(np.sum(all_weights ** 2)),
            'max': np.max(np.abs(all_weights)),
            'sparsity': np.mean(np.abs(all_weights) < 1e-6)
        }

def create_synthetic_dataset(n_samples: int = 1000, 
                           n_features: int = 20, 
                           noise: float = 0.1,
                           random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Create a synthetic dataset for testing regularization"""
    np.random.seed(random_state)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Only first 5 features are relevant
    true_weights = np.zeros(n_features)
    true_weights[:5] = np.random.randn(5) * 2
    
    # Generate target with some relevant features
    y_continuous = X @ true_weights + noise * np.random.randn(n_samples)
    
    # Convert to binary classification
    y = (y_continuous > np.median(y_continuous)).astype(int)
    
    return X, y.reshape(-1, 1)

def compare_regularization_techniques(X_train: np.ndarray, 
                                   y_train: np.ndarray,
                                   X_val: np.ndarray,
                                   y_val: np.ndarray,
                                   epochs: int = 100) -> Dict:
    """Compare different regularization techniques"""
    
    configs = {
        'No Regularization': {
            'l1_lambda': 0.0,
            'l2_lambda': 0.0,
            'dropout_rate': 0.0
        },
        'L1 Regularization': {
            'l1_lambda': 0.01,
            'l2_lambda': 0.0,
            'dropout_rate': 0.0
        },
        'L2 Regularization': {
            'l1_lambda': 0.0,
            'l2_lambda': 0.01,
            'dropout_rate': 0.0
        },
        'Dropout': {
            'l1_lambda': 0.0,
            'l2_lambda': 0.0,
            'dropout_rate': 0.3
        },
        'L2 + Dropout': {
            'l1_lambda': 0.0,
            'l2_lambda': 0.01,
            'dropout_rate': 0.3
        },
        'Elastic Net': {
            'l1_lambda': 0.005,
            'l2_lambda': 0.005,
            'dropout_rate': 0.0
        }
    }
    
    results = {}
    input_size = X_train.shape[1]
    
    for name, config in configs.items():
        print(f"\nTraining {name}...")
        
        # Create model
        model = RegularizedMLP(
            layers=[input_size, 64, 32, 1],
            activation='relu',
            **config
        )
        
        # Train model
        history = model.fit(
            X_train, y_train, X_val, y_val,
            epochs=epochs, learning_rate=0.001,
            verbose=False
        )
        
        # Store results
        results[name] = {
            'model': model,
            'history': history,
            'config': config,
            'final_val_loss': history['val_loss'][-1],
            'final_val_accuracy': history['val_accuracy'][-1],
            'weight_stats': model.get_weight_statistics()
        }
        
        print(f"Final Validation Accuracy: {history['val_accuracy'][-1]:.4f}")
        print(f"Final Validation Loss: {history['val_loss'][-1]:.4f}")
    
    return results

def plot_regularization_comparison(results: Dict, save_path: str = None):
    """Plot comparison of regularization techniques"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Training vs Validation Loss
    ax1 = axes[0, 0]
    for name, result in results.items():
        history = result['history']
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], '--', alpha=0.7, label=f'{name} (Train)')
        ax1.plot(epochs, history['val_loss'], '-', label=f'{name} (Val)')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training vs Validation Loss')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation Accuracy
    ax2 = axes[0, 1]
    for name, result in results.items():
        history = result['history']
        epochs = range(1, len(history['val_accuracy']) + 1)
        ax2.plot(epochs, history['val_accuracy'], label=name)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title('Validation Accuracy Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final Performance Comparison
    ax3 = axes[0, 2]
    names = list(results.keys())
    val_accs = [results[name]['final_val_accuracy'] for name in names]
    val_losses = [results[name]['final_val_loss'] for name in names]
    
    x_pos = np.arange(len(names))
    ax3_twin = ax3.twinx()
    
    bars1 = ax3.bar(x_pos - 0.2, val_accs, 0.4, label='Accuracy', alpha=0.7)
    bars2 = ax3_twin.bar(x_pos + 0.2, val_losses, 0.4, label='Loss', alpha=0.7, color='red')
    
    ax3.set_xlabel('Regularization Method')
    ax3.set_ylabel('Validation Accuracy', color='blue')
    ax3_twin.set_ylabel('Validation Loss', color='red')
    ax3.set_title('Final Performance Comparison')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(names, rotation=45, ha='right')
    
    # Plot 4: Weight Statistics
    ax4 = axes[1, 0]
    weight_l2_norms = [results[name]['weight_stats']['l2_norm'] for name in names]
    sparsity = [results[name]['weight_stats']['sparsity'] for name in names]
    
    ax4_twin = ax4.twinx()
    bars3 = ax4.bar(x_pos - 0.2, weight_l2_norms, 0.4, label='L2 Norm', alpha=0.7)
    bars4 = ax4_twin.bar(x_pos + 0.2, sparsity, 0.4, label='Sparsity', alpha=0.7, color='green')
    
    ax4.set_xlabel('Regularization Method')
    ax4.set_ylabel('Weight L2 Norm', color='blue')
    ax4_twin.set_ylabel('Weight Sparsity', color='green')
    ax4.set_title('Weight Statistics')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(names, rotation=45, ha='right')
    
    # Plot 5: Regularization Loss (where applicable)
    ax5 = axes[1, 1]
    for name, result in results.items():
        history = result['history']
        if any(loss > 0 for loss in history['regularization_loss']):
            epochs = range(1, len(history['regularization_loss']) + 1)
            ax5.plot(epochs, history['regularization_loss'], label=name)
    
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Regularization Loss')
    ax5.set_title('Regularization Loss Over Time')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Overfitting Analysis (Train-Val Loss Gap)
    ax6 = axes[1, 2]
    for name, result in results.items():
        history = result['history']
        epochs = range(1, len(history['train_loss']) + 1)
        gap = np.array(history['val_loss']) - np.array(history['train_loss'])
        ax6.plot(epochs, gap, label=name)
    
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Validation - Training Loss')
    ax6.set_title('Overfitting Analysis')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def demonstrate_regularization():
    """Comprehensive demonstration of regularization techniques"""
    print("Regularization Techniques Demonstration")
    print("=" * 50)
    
    # Generate synthetic dataset
    print("1. Generating synthetic dataset...")
    X, y = create_synthetic_dataset(n_samples=2000, n_features=50, noise=0.1)
    
    # Split into train/validation
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Normalize features
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train = (X_train - mean) / (std + 1e-8)
    X_val = (X_val - mean) / (std + 1e-8)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    
    # Compare regularization techniques
    print("\n2. Comparing regularization techniques...")
    results = compare_regularization_techniques(X_train, y_train, X_val, y_val, epochs=150)
    
    # Create visualization
    print("\n3. Creating comparison plots...")
    plot_regularization_comparison(results, save_path='plots/regularization_comparison.png')
    
    # Print summary
    print("\n4. Summary of Results:")
    print("-" * 60)
    print(f"{'Method':<20} {'Val Acc':<10} {'Val Loss':<10} {'L2 Norm':<10} {'Sparsity':<10}")
    print("-" * 60)
    
    for name, result in results.items():
        acc = result['final_val_accuracy']
        loss = result['final_val_loss']
        l2_norm = result['weight_stats']['l2_norm']
        sparsity = result['weight_stats']['sparsity']
        print(f"{name:<20} {acc:<10.4f} {loss:<10.4f} {l2_norm:<10.4f} {sparsity:<10.4f}")
    
    return results

if __name__ == "__main__":
    # Run demonstration
    results = demonstrate_regularization() 