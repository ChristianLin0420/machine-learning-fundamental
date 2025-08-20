#!/usr/bin/env python3
"""
CNN from Scratch Implementation
===============================

A complete Convolutional Neural Network implementation from scratch using only NumPy.
Includes convolution, pooling, and fully connected layers with proper backpropagation.

Author: ML Fundamentals Course
Day: 36 - CNNs
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict, Any
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import pickle


class ConvolutionLayer:
    """
    2D Convolution layer implementation.
    
    Applies filters to input feature maps and computes gradients during backpropagation.
    """
    
    def __init__(self, input_channels: int, output_channels: int, 
                 kernel_size: int, stride: int = 1, padding: int = 0):
        """
        Initialize convolution layer.
        
        Args:
            input_channels: Number of input channels
            output_channels: Number of output filters
            kernel_size: Size of convolution kernel (assumed square)
            stride: Stride of convolution
            padding: Padding around input
        """
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Xavier initialization for filters
        fan_in = input_channels * kernel_size * kernel_size
        fan_out = output_channels * kernel_size * kernel_size
        limit = np.sqrt(6 / (fan_in + fan_out))
        
        self.filters = np.random.uniform(-limit, limit, 
                                       (output_channels, input_channels, kernel_size, kernel_size))
        self.biases = np.zeros((output_channels, 1))
        
        # For backpropagation
        self.last_input = None
        self.filter_gradients = None
        self.bias_gradients = None
    
    def add_padding(self, x: np.ndarray) -> np.ndarray:
        """Add zero padding to input."""
        if self.padding == 0:
            return x
        
        return np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), 
                         (self.padding, self.padding)), mode='constant')
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through convolution layer.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor after convolution
        """
        self.last_input = x
        batch_size, input_channels, input_height, input_width = x.shape
        
        # Add padding
        padded_input = self.add_padding(x)
        
        # Calculate output dimensions
        output_height = (input_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_width = (input_width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, self.output_channels, output_height, output_width))
        
        # Convolution operation
        for b in range(batch_size):
            for f in range(self.output_channels):
                for h in range(output_height):
                    for w in range(output_width):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        
                        # Extract region and apply filter
                        region = padded_input[b, :, h_start:h_end, w_start:w_end]
                        output[b, f, h, w] = np.sum(region * self.filters[f]) + self.biases[f]
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through convolution layer.
        
        Args:
            grad_output: Gradient from next layer
            
        Returns:
            Gradient with respect to input
        """
        batch_size, input_channels, input_height, input_width = self.last_input.shape
        _, output_channels, output_height, output_width = grad_output.shape
        
        # Initialize gradients
        self.filter_gradients = np.zeros_like(self.filters)
        self.bias_gradients = np.zeros_like(self.biases)
        grad_input = np.zeros_like(self.last_input)
        
        # Add padding to input
        padded_input = self.add_padding(self.last_input)
        padded_grad_input = np.zeros_like(padded_input)
        
        # Compute gradients
        for b in range(batch_size):
            for f in range(output_channels):
                for h in range(output_height):
                    for w in range(output_width):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                        
                        # Region in padded input
                        region = padded_input[b, :, h_start:h_end, w_start:w_end]
                        
                        # Gradient w.r.t. filters
                        self.filter_gradients[f] += grad_output[b, f, h, w] * region
                        
                        # Gradient w.r.t. input
                        padded_grad_input[b, :, h_start:h_end, w_start:w_end] += \
                            grad_output[b, f, h, w] * self.filters[f]
        
        # Remove padding from input gradients
        if self.padding > 0:
            grad_input = padded_grad_input[:, :, self.padding:-self.padding, 
                                         self.padding:-self.padding]
        else:
            grad_input = padded_grad_input
        
        # Bias gradients
        self.bias_gradients = np.sum(grad_output, axis=(0, 2, 3)).reshape(-1, 1)
        
        return grad_input


class PoolingLayer:
    """
    Pooling layer implementation (Max and Average pooling).
    """
    
    def __init__(self, pool_size: int = 2, stride: int = 2, mode: str = 'max'):
        """
        Initialize pooling layer.
        
        Args:
            pool_size: Size of pooling window
            stride: Stride of pooling
            mode: 'max' or 'avg'
        """
        self.pool_size = pool_size
        self.stride = stride
        self.mode = mode
        self.last_input = None
        self.mask = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through pooling layer.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Pooled output tensor
        """
        self.last_input = x
        batch_size, channels, input_height, input_width = x.shape
        
        # Calculate output dimensions
        output_height = (input_height - self.pool_size) // self.stride + 1
        output_width = (input_width - self.pool_size) // self.stride + 1
        
        # Initialize output and mask for max pooling
        output = np.zeros((batch_size, channels, output_height, output_width))
        if self.mode == 'max':
            self.mask = np.zeros_like(x)
        
        # Pooling operation
        for b in range(batch_size):
            for c in range(channels):
                for h in range(output_height):
                    for w in range(output_width):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size
                        
                        region = x[b, c, h_start:h_end, w_start:w_end]
                        
                        if self.mode == 'max':
                            output[b, c, h, w] = np.max(region)
                            
                            # Store mask for backpropagation
                            max_pos = np.unravel_index(np.argmax(region), region.shape)
                            self.mask[b, c, h_start + max_pos[0], w_start + max_pos[1]] = 1
                        
                        elif self.mode == 'avg':
                            output[b, c, h, w] = np.mean(region)
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through pooling layer.
        
        Args:
            grad_output: Gradient from next layer
            
        Returns:
            Gradient with respect to input
        """
        batch_size, channels, output_height, output_width = grad_output.shape
        grad_input = np.zeros_like(self.last_input)
        
        for b in range(batch_size):
            for c in range(channels):
                for h in range(output_height):
                    for w in range(output_width):
                        h_start = h * self.stride
                        h_end = h_start + self.pool_size
                        w_start = w * self.stride
                        w_end = w_start + self.pool_size
                        
                        if self.mode == 'max':
                            # Use stored mask
                            grad_input[b, c, h_start:h_end, w_start:w_end] += \
                                grad_output[b, c, h, w] * self.mask[b, c, h_start:h_end, w_start:w_end]
                        
                        elif self.mode == 'avg':
                            # Distribute gradient equally
                            grad_input[b, c, h_start:h_end, w_start:w_end] += \
                                grad_output[b, c, h, w] / (self.pool_size * self.pool_size)
        
        return grad_input


class ReLULayer:
    """ReLU activation layer."""
    
    def __init__(self):
        self.last_input = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through ReLU."""
        self.last_input = x
        return np.maximum(0, x)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through ReLU."""
        return grad_output * (self.last_input > 0)


class FlattenLayer:
    """Flatten layer to convert 4D tensor to 2D for fully connected layers."""
    
    def __init__(self):
        self.input_shape = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through flatten layer."""
        self.input_shape = x.shape
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through flatten layer."""
        return grad_output.reshape(self.input_shape)


class FullyConnectedLayer:
    """Fully connected (dense) layer."""
    
    def __init__(self, input_size: int, output_size: int):
        """
        Initialize fully connected layer.
        
        Args:
            input_size: Number of input features
            output_size: Number of output features
        """
        # Xavier initialization
        limit = np.sqrt(6 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, (output_size, input_size))
        self.biases = np.zeros((output_size, 1))
        
        # For backpropagation
        self.last_input = None
        self.weight_gradients = None
        self.bias_gradients = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through fully connected layer."""
        self.last_input = x
        return np.dot(self.weights, x.T).T + self.biases.T
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through fully connected layer."""
        # Gradients w.r.t. weights and biases
        self.weight_gradients = np.dot(grad_output.T, self.last_input)
        self.bias_gradients = np.sum(grad_output, axis=0, keepdims=True).T
        
        # Gradient w.r.t. input
        grad_input = np.dot(grad_output, self.weights)
        return grad_input


class SoftmaxLayer:
    """Softmax activation layer."""
    
    def __init__(self):
        self.last_output = None
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through softmax."""
        # Numerical stability
        exp_scores = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.last_output = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.last_output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass through softmax."""
        batch_size, num_classes = self.last_output.shape
        grad_input = np.zeros_like(grad_output)
        
        for i in range(batch_size):
            s = self.last_output[i].reshape(-1, 1)
            jacobian = np.diagflat(s) - np.dot(s, s.T)
            grad_input[i] = np.dot(jacobian, grad_output[i])
        
        return grad_input


class CrossEntropyLoss:
    """Cross-entropy loss function."""
    
    def __init__(self):
        self.last_predictions = None
        self.last_targets = None
    
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute cross-entropy loss.
        
        Args:
            predictions: Predicted probabilities (batch_size, num_classes)
            targets: One-hot encoded targets (batch_size, num_classes)
            
        Returns:
            Average loss over the batch
        """
        self.last_predictions = predictions
        self.last_targets = targets
        
        # Clip predictions to prevent log(0)
        predictions_clipped = np.clip(predictions, 1e-15, 1 - 1e-15)
        
        # Compute loss
        loss = -np.sum(targets * np.log(predictions_clipped)) / targets.shape[0]
        return loss
    
    def backward(self) -> np.ndarray:
        """Compute gradient of loss w.r.t. predictions."""
        batch_size = self.last_targets.shape[0]
        grad = -(self.last_targets / self.last_predictions) / batch_size
        return grad


class CNN:
    """
    Complete Convolutional Neural Network implementation from scratch.
    
    Architecture: Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Flatten -> FC -> Softmax
    """
    
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int):
        """
        Initialize CNN.
        
        Args:
            input_shape: (channels, height, width)
            num_classes: Number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Build layers
        self.conv1 = ConvolutionLayer(input_channels=input_shape[0], output_channels=16, 
                                    kernel_size=3, stride=1, padding=1)
        self.relu1 = ReLULayer()
        self.pool1 = PoolingLayer(pool_size=2, stride=2, mode='max')
        
        self.conv2 = ConvolutionLayer(input_channels=16, output_channels=32, 
                                    kernel_size=3, stride=1, padding=1)
        self.relu2 = ReLULayer()
        self.pool2 = PoolingLayer(pool_size=2, stride=2, mode='max')
        
        self.flatten = FlattenLayer()
        
        # Calculate flattened size
        self.flattened_size = self._calculate_flattened_size()
        
        self.fc1 = FullyConnectedLayer(self.flattened_size, 128)
        self.relu3 = ReLULayer()
        self.fc2 = FullyConnectedLayer(128, num_classes)
        self.softmax = SoftmaxLayer()
        
        # Loss function
        self.loss_fn = CrossEntropyLoss()
        
        # Training history
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def _calculate_flattened_size(self) -> int:
        """Calculate the size after convolution and pooling layers."""
        # Simulate forward pass to get dimensions
        dummy_input = np.zeros((1, *self.input_shape))
        
        x = self.conv1.forward(dummy_input)
        x = self.pool1.forward(x)
        x = self.conv2.forward(x)
        x = self.pool2.forward(x)
        
        return x.shape[1] * x.shape[2] * x.shape[3]
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the entire network.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            Output probabilities (batch_size, num_classes)
        """
        # Convolutional layers
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)
        
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)
        
        # Fully connected layers
        x = self.flatten.forward(x)
        x = self.fc1.forward(x)
        x = self.relu3.forward(x)
        x = self.fc2.forward(x)
        x = self.softmax.forward(x)
        
        return x
    
    def backward(self, grad_output: np.ndarray) -> None:
        """
        Backward pass through the entire network.
        
        Args:
            grad_output: Gradient from loss function
        """
        # Backward through fully connected layers
        grad = self.softmax.backward(grad_output)
        grad = self.fc2.backward(grad)
        grad = self.relu3.backward(grad)
        grad = self.fc1.backward(grad)
        grad = self.flatten.backward(grad)
        
        # Backward through convolutional layers
        grad = self.pool2.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.conv2.backward(grad)
        
        grad = self.pool1.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.conv1.backward(grad)
    
    def update_weights(self, learning_rate: float) -> None:
        """Update all weights using computed gradients."""
        # Update convolution layers
        self.conv1.filters -= learning_rate * self.conv1.filter_gradients
        self.conv1.biases -= learning_rate * self.conv1.bias_gradients
        
        self.conv2.filters -= learning_rate * self.conv2.filter_gradients
        self.conv2.biases -= learning_rate * self.conv2.bias_gradients
        
        # Update fully connected layers
        self.fc1.weights -= learning_rate * self.fc1.weight_gradients
        self.fc1.biases -= learning_rate * self.fc1.bias_gradients
        
        self.fc2.weights -= learning_rate * self.fc2.weight_gradients
        self.fc2.biases -= learning_rate * self.fc2.bias_gradients
    
    def compute_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute cross-entropy loss."""
        return self.loss_fn.forward(predictions, targets)
    
    def compute_accuracy(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute classification accuracy."""
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(targets, axis=1)
        return np.mean(predicted_classes == true_classes)
    
    def train_epoch(self, X_train: np.ndarray, y_train: np.ndarray, 
                   learning_rate: float, batch_size: int = 32) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            X_train: Training images
            y_train: Training labels (one-hot encoded)
            learning_rate: Learning rate
            batch_size: Batch size
            
        Returns:
            Average loss and accuracy for the epoch
        """
        n_samples = X_train.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        
        # Shuffle data
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            X_batch = X_shuffled[start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            
            # Forward pass
            predictions = self.forward(X_batch)
            
            # Compute loss and accuracy
            loss = self.compute_loss(predictions, y_batch)
            accuracy = self.compute_accuracy(predictions, y_batch)
            
            epoch_loss += loss
            epoch_accuracy += accuracy
            
            # Backward pass
            grad_loss = self.loss_fn.backward()
            self.backward(grad_loss)
            
            # Update weights
            self.update_weights(learning_rate)
        
        return epoch_loss / n_batches, epoch_accuracy / n_batches
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
        """
        Evaluate model on test data.
        
        Args:
            X_test: Test images
            y_test: Test labels (one-hot encoded)
            
        Returns:
            Test loss and accuracy
        """
        predictions = self.forward(X_test)
        loss = self.compute_loss(predictions, y_test)
        accuracy = self.compute_accuracy(predictions, y_test)
        return loss, accuracy
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            epochs: int = 10, learning_rate: float = 0.001, 
            batch_size: int = 32, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the CNN.
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            epochs: Number of epochs
            learning_rate: Learning rate
            batch_size: Batch size
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        if verbose:
            print(f"Training CNN for {epochs} epochs...")
            print(f"Architecture: Conv(16) -> Pool -> Conv(32) -> Pool -> FC(128) -> FC({self.num_classes})")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train epoch
            train_loss, train_acc = self.train_epoch(X_train, y_train, learning_rate, batch_size)
            
            # Validation
            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.evaluate(X_val, y_val)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_accuracy'].append(val_acc)
            
            # Store history
            self.training_history['loss'].append(train_loss)
            self.training_history['accuracy'].append(train_acc)
            
            if verbose:
                epoch_time = time.time() - start_time
                if X_val is not None:
                    print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.1f}s - "
                          f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - "
                          f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.1f}s - "
                          f"loss: {train_loss:.4f} - acc: {train_acc:.4f}")
        
        return self.training_history


def load_mnist_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess MNIST dataset.
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("Loading MNIST dataset...")
    
    # Load MNIST (this might take a moment on first run)
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target.astype(int)
    
    # Use a subset for faster training
    X = X[:10000]
    y = y[:10000]
    
    # Reshape to image format: (samples, channels, height, width)
    X = X.reshape(-1, 1, 28, 28)
    
    # Normalize pixel values
    X = X.astype(np.float32) / 255.0
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert labels to one-hot encoding
    num_classes = 10
    y_train_onehot = np.eye(num_classes)[y_train]
    y_test_onehot = np.eye(num_classes)[y_test]
    
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Classes: {num_classes}")
    
    return X_train, X_test, y_train_onehot, y_test_onehot


def create_simple_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a simple synthetic dataset for quick testing.
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("Creating simple synthetic dataset...")
    
    # Create simple patterns
    n_samples = 1000
    img_size = 8
    
    X = np.zeros((n_samples, 1, img_size, img_size))
    y = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        if i < n_samples // 4:
            # Class 0: horizontal line
            y[i] = 0
            row = np.random.randint(2, 6)
            X[i, 0, row, 2:6] = 1.0
            
        elif i < n_samples // 2:
            # Class 1: vertical line
            y[i] = 1
            col = np.random.randint(2, 6)
            X[i, 0, 2:6, col] = 1.0
            
        elif i < 3 * n_samples // 4:
            # Class 2: diagonal line
            y[i] = 2
            for j in range(4):
                X[i, 0, j+2, j+2] = 1.0
                
        else:
            # Class 3: square
            y[i] = 3
            X[i, 0, 2:6, 2:6] = 1.0
            X[i, 0, 3:5, 3:5] = 0.0
    
    # Add noise
    noise = np.random.normal(0, 0.1, X.shape)
    X += noise
    X = np.clip(X, 0, 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert to one-hot
    num_classes = 4
    y_train_onehot = np.eye(num_classes)[y_train]
    y_test_onehot = np.eye(num_classes)[y_test]
    
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    print(f"Classes: {num_classes}")
    
    return X_train, X_test, y_train_onehot, y_test_onehot


def visualize_training_history(history: Dict[str, List[float]], save_path: str = None) -> None:
    """Visualize training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(history['loss'], label='Training Loss', marker='o')
    if 'val_loss' in history and history['val_loss']:
        axes[0].plot(history['val_loss'], label='Validation Loss', marker='s')
    axes[0].set_title('Training History - Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history['accuracy'], label='Training Accuracy', marker='o')
    if 'val_accuracy' in history and history['val_accuracy']:
        axes[1].plot(history['val_accuracy'], label='Validation Accuracy', marker='s')
    axes[1].set_title('Training History - Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def visualize_filters(cnn: CNN, save_path: str = None) -> None:
    """Visualize learned filters."""
    # Get first layer filters
    filters = cnn.conv1.filters  # Shape: (num_filters, channels, height, width)
    
    num_filters = filters.shape[0]
    cols = min(8, num_filters)
    rows = (num_filters + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 1.5))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_filters):
        row = i // cols
        col = i % cols
        
        # Normalize filter for visualization
        filter_img = filters[i, 0]  # First channel
        filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min() + 1e-8)
        
        axes[row, col].imshow(filter_img, cmap='gray')
        axes[row, col].set_title(f'Filter {i+1}')
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(num_filters, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.suptitle('Learned Convolution Filters (Layer 1)', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Filter visualization saved to {save_path}")
    
    plt.show()


def visualize_feature_maps(cnn: CNN, X_sample: np.ndarray, save_path: str = None) -> None:
    """Visualize feature maps for a sample input."""
    # Forward pass to get intermediate activations
    x = X_sample
    
    # Conv1 + ReLU
    conv1_output = cnn.conv1.forward(x)
    relu1_output = cnn.relu1.forward(conv1_output)
    
    # Visualize some feature maps from first conv layer
    num_maps_to_show = min(16, relu1_output.shape[1])
    cols = 4
    rows = (num_maps_to_show + cols - 1) // cols
    
    fig, axes = plt.subplots(rows + 1, cols, figsize=(12, (rows + 1) * 2))
    
    # Original image
    axes[0, 0].imshow(X_sample[0, 0], cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Hide other subplots in first row
    for i in range(1, cols):
        axes[0, i].axis('off')
    
    # Feature maps
    for i in range(num_maps_to_show):
        row = (i // cols) + 1
        col = i % cols
        
        feature_map = relu1_output[0, i]
        axes[row, col].imshow(feature_map, cmap='viridis')
        axes[row, col].set_title(f'Feature Map {i+1}')
        axes[row, col].axis('off')
    
    plt.suptitle('Feature Maps from First Convolution Layer', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature map visualization saved to {save_path}")
    
    plt.show()


def compare_architectures() -> Dict[str, Any]:
    """Compare different CNN architectures."""
    print("\nComparing CNN Architectures...")
    print("=" * 50)
    
    # Create simple dataset for quick comparison
    X_train, X_test, y_train, y_test = create_simple_dataset()
    
    architectures = {
        'Shallow CNN': {
            'description': 'Single conv layer',
            'layers': [(16, 3), 'pool', 'fc']
        },
        'Deep CNN': {
            'description': 'Two conv layers',
            'layers': [(16, 3), 'pool', (32, 3), 'pool', 'fc']
        }
    }
    
    results = {}
    
    for arch_name, arch_config in architectures.items():
        print(f"\nTraining {arch_name}...")
        
        # Create model (for now, just use our standard architecture)
        input_shape = X_train.shape[1:]  # (channels, height, width)
        num_classes = y_train.shape[1]
        
        cnn = CNN(input_shape, num_classes)
        
        # Quick training
        history = cnn.fit(X_train, y_train, X_test, y_test, 
                         epochs=5, learning_rate=0.01, batch_size=32, verbose=False)
        
        # Final evaluation
        test_loss, test_acc = cnn.evaluate(X_test, y_test)
        
        results[arch_name] = {
            'history': history,
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'model': cnn
        }
        
        print(f"  Final Test Accuracy: {test_acc:.3f}")
    
    return results


def run_comprehensive_experiments() -> Dict[str, Any]:
    """Run comprehensive CNN experiments."""
    print("Starting CNN from Scratch - Comprehensive Experiments")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    all_results = {}
    
    # Experiment 1: Simple dataset
    print("\n" + "=" * 50)
    print("EXPERIMENT 1: Simple Synthetic Dataset")
    print("=" * 50)
    
    X_train, X_test, y_train, y_test = create_simple_dataset()
    
    # Train CNN
    input_shape = X_train.shape[1:]  # (channels, height, width)
    num_classes = y_train.shape[1]
    
    cnn = CNN(input_shape, num_classes)
    
    print(f"\nCNN Architecture:")
    print(f"Input shape: {input_shape}")
    print(f"Conv1: {cnn.conv1.input_channels} -> {cnn.conv1.output_channels} channels")
    print(f"Conv2: {cnn.conv2.input_channels} -> {cnn.conv2.output_channels} channels")
    print(f"FC1: {cnn.flattened_size} -> 128")
    print(f"FC2: 128 -> {num_classes}")
    
    # Train model
    history = cnn.fit(X_train, y_train, X_test, y_test, 
                     epochs=10, learning_rate=0.01, batch_size=32)
    
    # Final evaluation
    test_loss, test_acc = cnn.evaluate(X_test, y_test)
    print(f"\nFinal Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    all_results['simple_dataset'] = {
        'history': history,
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'model': cnn,
        'data': (X_train, X_test, y_train, y_test)
    }
    
    # Visualizations
    print("\nCreating visualizations...")
    
    # Training history
    visualize_training_history(history, 'plots/cnn_training_history.png')
    
    # Filters
    visualize_filters(cnn, 'plots/cnn_learned_filters.png')
    
    # Feature maps
    sample_input = X_test[:1]  # First test sample
    visualize_feature_maps(cnn, sample_input, 'plots/cnn_feature_maps.png')
    
    # Experiment 2: Architecture comparison
    print("\n" + "=" * 50)
    print("EXPERIMENT 2: Architecture Comparison")
    print("=" * 50)
    
    arch_results = compare_architectures()
    all_results['architecture_comparison'] = arch_results
    
    # Save results
    print("\nSaving results...")
    with open('plots/cnn_results.pkl', 'wb') as f:
        # Don't save the models (too large), just the metrics
        results_to_save = {}
        for exp_name, exp_results in all_results.items():
            if exp_name == 'simple_dataset':
                results_to_save[exp_name] = {
                    'history': exp_results['history'],
                    'test_accuracy': exp_results['test_accuracy'],
                    'test_loss': exp_results['test_loss']
                }
            else:
                results_to_save[exp_name] = {
                    arch_name: {
                        'history': arch_data['history'],
                        'test_accuracy': arch_data['test_accuracy'],
                        'test_loss': arch_data['test_loss']
                    }
                    for arch_name, arch_data in exp_results.items()
                }
        pickle.dump(results_to_save, f)
    
    print("CNN from scratch experiments completed!")
    print("Check the plots/ directory for visualizations.")
    
    return all_results


if __name__ == "__main__":
    results = run_comprehensive_experiments()