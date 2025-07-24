"""
PyTorch MLP Implementation for Comparison

This module implements the same MLP architecture using PyTorch
to compare performance, training time, and ease of implementation
with the from-scratch NumPy version.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from mlp_numpy import FeedforwardNeuralNet
from datasets import (make_xor_dataset, make_spiral_dataset, make_moons_dataset, 
                     create_train_test_split, normalize_features)


class PyTorchMLP(nn.Module):
    """
    PyTorch implementation of Multi-Layer Perceptron.
    
    Equivalent to the NumPy implementation but using PyTorch's
    automatic differentiation and optimized operations.
    """
    
    def __init__(self, layers, activations=None, output_activation='sigmoid', dropout=0.0):
        """
        Initialize the PyTorch MLP.
        
        Args:
            layers (list): Number of neurons in each layer
            activations (list): Activation functions for hidden layers
            output_activation (str): Output activation function
            dropout (float): Dropout probability
        """
        super(PyTorchMLP, self).__init__()
        
        self.layers = layers
        self.n_layers = len(layers)
        self.dropout = dropout
        
        # Set default activations
        if activations is None:
            activations = ['relu'] * (self.n_layers - 2)
        
        self.activations = activations + [output_activation]
        
        # Create layers
        self.linear_layers = nn.ModuleList()
        for i in range(self.n_layers - 1):
            self.linear_layers.append(
                nn.Linear(layers[i], layers[i + 1])
            )
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize weights
        self._initialize_weights()
        
        # Training history
        self.history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for layer in self.linear_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def _get_activation(self, name):
        """Get PyTorch activation function by name."""
        if name == 'relu':
            return F.relu
        elif name == 'sigmoid':
            return torch.sigmoid
        elif name == 'tanh':
            return torch.tanh
        elif name == 'linear':
            return lambda x: x
        elif name == 'softmax':
            return lambda x: F.softmax(x, dim=1)
        else:
            raise ValueError(f"Unknown activation: {name}")
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        for i, layer in enumerate(self.linear_layers):
            x = layer(x)
            
            # Apply activation function
            activation_func = self._get_activation(self.activations[i])
            x = activation_func(x)
            
            # Apply dropout (except for output layer)
            if i < len(self.linear_layers) - 1 and self.dropout > 0:
                x = self.dropout_layer(x)
        
        return x
    
    def predict_proba(self, X):
        """
        Predict probabilities.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predicted probabilities
        """
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.forward(X_tensor)
            return outputs.numpy()
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predicted labels
        """
        probabilities = self.predict_proba(X)
        
        if probabilities.shape[1] == 1:
            # Binary classification
            return np.where(probabilities > 0.5, 1, -1)
        else:
            # Multi-class classification
            return np.argmax(probabilities, axis=1).reshape(-1, 1)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=1000, 
            learning_rate=0.01, batch_size=32, verbose=True):
        """
        Train the PyTorch model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation labels
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate
            batch_size (int): Batch size
            verbose (bool): Print training progress
            
        Returns:
            self: Returns self for method chaining
        """
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        
        # Handle labels - ensure proper shape
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        
        if y_train.min() == -1:
            # Convert {-1, 1} to {0, 1} for PyTorch
            y_train_tensor = torch.FloatTensor((y_train + 1) / 2)
        else:
            y_train_tensor = torch.FloatTensor(y_train)
        
        # Create data loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer and loss function
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        if y_train_tensor.shape[1] == 1:
            criterion = nn.BCELoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Reset history
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        if verbose:
            print(f"Training PyTorch MLP:")
            print(f"  Architecture: {self.layers}")
            print(f"  Epochs: {epochs}, Learning rate: {learning_rate}")
            print(f"  Batch size: {batch_size}")
        
        # Training loop
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            n_batches = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.forward(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Compute metrics
                epoch_loss += loss.item()
                
                if batch_y.shape[1] == 1:
                    # Binary classification accuracy
                    predicted = (outputs > 0.5).float()
                    accuracy = (predicted == batch_y).float().mean().item()
                else:
                    # Multi-class accuracy
                    _, predicted = torch.max(outputs, 1)
                    accuracy = (predicted == torch.argmax(batch_y, 1)).float().mean().item()
                
                epoch_accuracy += accuracy
                n_batches += 1
            
            # Average metrics
            avg_loss = epoch_loss / n_batches
            avg_accuracy = epoch_accuracy / n_batches
            
            self.history['loss'].append(avg_loss)
            self.history['accuracy'].append(avg_accuracy)
            
            # Validation metrics
            if X_val is not None and y_val is not None:
                # Use the same criterion for validation
                if y_train_tensor.shape[1] == 1:
                    val_criterion = nn.BCELoss()
                else:
                    val_criterion = nn.CrossEntropyLoss()
                
                val_loss, val_accuracy = self._evaluate(X_val, y_val, val_criterion)
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_accuracy)
            
            # Print progress
            if verbose and (epoch + 1) % (epochs // 10 if epochs >= 10 else 1) == 0:
                msg = f"Epoch {epoch + 1}/{epochs} - "
                msg += f"loss: {avg_loss:.4f} - accuracy: {avg_accuracy:.4f}"
                
                if X_val is not None:
                    msg += f" - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}"
                
                print(msg)
        
        return self
    
    def _evaluate(self, X, y, criterion):
        """Evaluate the model on validation data."""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            
            # Ensure proper shape
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            
            if y.min() == -1:
                y_tensor = torch.FloatTensor((y + 1) / 2)
            else:
                y_tensor = torch.FloatTensor(y)
            
            outputs = self.forward(X_tensor)
            loss = criterion(outputs, y_tensor).item()
            
            if y_tensor.shape[1] == 1:
                predicted = (outputs > 0.5).float()
                accuracy = (predicted == y_tensor).float().mean().item()
            else:
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == torch.argmax(y_tensor, 1)).float().mean().item()
            
            return loss, accuracy
    
    def evaluate(self, X, y):
        """
        Evaluate the model and return metrics.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Labels
            
        Returns:
            dict: Evaluation metrics
        """
        # Ensure proper shape
        if y.ndim == 1:
            y = y.reshape(-1, 1)
            
        if y.min() == -1:
            criterion = nn.BCELoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        loss, accuracy = self._evaluate(X, y, criterion)
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'predictions': self.predict_proba(X),
            'predicted_classes': self.predict(X)
        }
    
    def get_model_summary(self):
        """Get model summary."""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'architecture': self.layers,
            'total_parameters': total_params,
            'framework': 'PyTorch'
        }


def compare_numpy_vs_pytorch(X, y, dataset_name="Dataset"):
    """
    Compare NumPy and PyTorch implementations on the same dataset.
    
    Args:
        X (np.ndarray): Features
        y (np.ndarray): Labels
        dataset_name (str): Name of the dataset
        
    Returns:
        dict: Comparison results
    """
    print(f"\nComparing NumPy vs PyTorch on {dataset_name}")
    print("-" * 60)
    
    # Split and normalize data
    X_train, X_test, y_train, y_test = create_train_test_split(X, y, test_size=0.2)
    X_train_norm, X_test_norm, _, _ = normalize_features(X_train, X_test)
    
    # Common architecture
    layers = [2, 16, 16, 1]
    activations = ['relu', 'relu']
    epochs = 300
    learning_rate = 0.01
    
    results = {}
    
    # Test NumPy implementation
    print("\nTraining NumPy implementation...")
    numpy_mlp = FeedforwardNeuralNet(
        layers=layers,
        activations=activations,
        output_activation='sigmoid',
        weight_init='xavier',
        random_state=42
    )
    
    start_time = time.time()
    numpy_mlp.fit(X_train_norm, y_train, epochs=epochs, learning_rate=learning_rate,
                  validation_data=(X_test_norm, y_test), verbose=False)
    numpy_time = time.time() - start_time
    
    numpy_results = numpy_mlp.evaluate(X_test_norm, y_test)
    
    results['NumPy'] = {
        'model': numpy_mlp,
        'training_time': numpy_time,
        'test_accuracy': numpy_results['accuracy'],
        'test_loss': numpy_results['loss'],
        'framework': 'NumPy'
    }
    
    print(f"  Training time: {numpy_time:.2f}s")
    print(f"  Test accuracy: {numpy_results['accuracy']:.4f}")
    print(f"  Test loss: {numpy_results['loss']:.4f}")
    
    # Test PyTorch implementation
    print("\nTraining PyTorch implementation...")
    pytorch_mlp = PyTorchMLP(
        layers=layers,
        activations=activations,
        output_activation='sigmoid'
    )
    
    start_time = time.time()
    pytorch_mlp.fit(X_train_norm, y_train, X_test_norm, y_test, 
                   epochs=epochs, learning_rate=learning_rate, batch_size=32, verbose=False)
    pytorch_time = time.time() - start_time
    
    pytorch_results = pytorch_mlp.evaluate(X_test_norm, y_test)
    
    results['PyTorch'] = {
        'model': pytorch_mlp,
        'training_time': pytorch_time,
        'test_accuracy': pytorch_results['accuracy'],
        'test_loss': pytorch_results['loss'],
        'framework': 'PyTorch'
    }
    
    print(f"  Training time: {pytorch_time:.2f}s")
    print(f"  Test accuracy: {pytorch_results['accuracy']:.4f}")
    print(f"  Test loss: {pytorch_results['loss']:.4f}")
    
    # Speed comparison
    speedup = numpy_time / pytorch_time
    print(f"\nSpeedup: {speedup:.2f}x ({'PyTorch' if speedup > 1 else 'NumPy'} faster)")
    
    return results, X_train_norm, X_test_norm, y_train, y_test


def plot_framework_comparison(results, X_train, y_train, dataset_name):
    """
    Plot comparison between NumPy and PyTorch implementations.
    
    Args:
        results (dict): Results from compare_numpy_vs_pytorch
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        dataset_name (str): Name of the dataset
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Training curves comparison
    for framework, result in results.items():
        model = result['model']
        axes[0, 0].plot(model.history['loss'], label=f'{framework} (Train)', linewidth=2)
        if model.history['val_loss']:
            axes[0, 0].plot(model.history['val_loss'], '--', 
                           label=f'{framework} (Val)', linewidth=2)
    
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # Accuracy comparison
    for framework, result in results.items():
        model = result['model']
        axes[0, 1].plot(model.history['accuracy'], label=f'{framework} (Train)', linewidth=2)
        if model.history['val_accuracy']:
            axes[0, 1].plot(model.history['val_accuracy'], '--', 
                           label=f'{framework} (Val)', linewidth=2)
    
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training Accuracy Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Performance metrics comparison
    frameworks = list(results.keys())
    test_accs = [results[fw]['test_accuracy'] for fw in frameworks]
    training_times = [results[fw]['training_time'] for fw in frameworks]
    
    x_pos = np.arange(len(frameworks))
    
    bars1 = axes[1, 0].bar(x_pos, test_accs, alpha=0.8, color=['skyblue', 'lightcoral'])
    axes[1, 0].set_xlabel('Framework')
    axes[1, 0].set_ylabel('Test Accuracy')
    axes[1, 0].set_title('Test Accuracy Comparison')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(frameworks)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add values on bars
    for bar, acc in zip(bars1, test_accs):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{acc:.3f}', ha='center', va='bottom')
    
    bars2 = axes[1, 1].bar(x_pos, training_times, alpha=0.8, color=['skyblue', 'lightcoral'])
    axes[1, 1].set_xlabel('Framework')
    axes[1, 1].set_ylabel('Training Time (seconds)')
    axes[1, 1].set_title('Training Time Comparison')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(frameworks)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add values on bars
    for bar, time_val in zip(bars2, training_times):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                       f'{time_val:.2f}s', ha='center', va='bottom')
    
    plt.suptitle(f'NumPy vs PyTorch Comparison: {dataset_name}', fontsize=16)
    plt.tight_layout()
    
    save_path = f"plots/framework_comparison_{dataset_name.lower().replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def comprehensive_framework_comparison():
    """
    Run comprehensive comparison between NumPy and PyTorch implementations.
    """
    # Create plots directory
    import os
    os.makedirs("plots", exist_ok=True)
    
    print("Comprehensive Framework Comparison")
    print("=" * 60)
    
    # Test datasets
    datasets = {
        "XOR": make_xor_dataset(n_samples=1000, noise=0.05, random_state=42),
        "Spiral": make_spiral_dataset(n_samples=1000, noise=0.1, random_state=42),
        "Moons": make_moons_dataset(n_samples=1000, noise=0.1, random_state=42)
    }
    
    all_results = {}
    
    for dataset_name, (X, y) in datasets.items():
        print(f"\n{'='*60}")
        print(f"Testing on {dataset_name} Dataset")
        print(f"{'='*60}")
        
        # Compare frameworks
        results, X_train, X_test, y_train, y_test = compare_numpy_vs_pytorch(X, y, dataset_name)
        plot_framework_comparison(results, X_train, y_train, dataset_name)
        
        all_results[dataset_name] = results
    
    # Create overall summary
    create_framework_summary(all_results)
    
    return all_results


def create_framework_summary(all_results):
    """
    Create summary of framework comparison results.
    
    Args:
        all_results (dict): Results from all dataset comparisons
    """
    print(f"\n{'='*60}")
    print("FRAMEWORK COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    # Calculate averages
    numpy_avg_time = np.mean([results['NumPy']['training_time'] 
                             for results in all_results.values()])
    pytorch_avg_time = np.mean([results['PyTorch']['training_time'] 
                               for results in all_results.values()])
    
    numpy_avg_acc = np.mean([results['NumPy']['test_accuracy'] 
                            for results in all_results.values()])
    pytorch_avg_acc = np.mean([results['PyTorch']['test_accuracy'] 
                              for results in all_results.values()])
    
    print(f"\nAverage Performance:")
    print("-" * 30)
    print(f"{'Framework':<10} {'Avg Time (s)':<12} {'Avg Accuracy':<12}")
    print("-" * 30)
    print(f"{'NumPy':<10} {numpy_avg_time:<12.2f} {numpy_avg_acc:<12.4f}")
    print(f"{'PyTorch':<10} {pytorch_avg_time:<12.2f} {pytorch_avg_acc:<12.4f}")
    
    speedup = numpy_avg_time / pytorch_avg_time
    print(f"\nOverall Speedup: {speedup:.2f}x ({'PyTorch' if speedup > 1 else 'NumPy'} faster)")
    
    # Detailed comparison table
    print(f"\nDetailed Results by Dataset:")
    print("-" * 80)
    print(f"{'Dataset':<12} {'NumPy Time':<12} {'PyTorch Time':<14} {'NumPy Acc':<12} {'PyTorch Acc':<12}")
    print("-" * 80)
    
    for dataset_name, results in all_results.items():
        numpy_time = results['NumPy']['training_time']
        pytorch_time = results['PyTorch']['training_time']
        numpy_acc = results['NumPy']['test_accuracy']
        pytorch_acc = results['PyTorch']['test_accuracy']
        
        print(f"{dataset_name:<12} {numpy_time:<12.2f} {pytorch_time:<14.2f} "
              f"{numpy_acc:<12.4f} {pytorch_acc:<12.4f}")
    
    # Summary visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    datasets = list(all_results.keys())
    
    # Time comparison
    numpy_times = [all_results[d]['NumPy']['training_time'] for d in datasets]
    pytorch_times = [all_results[d]['PyTorch']['training_time'] for d in datasets]
    
    x_pos = np.arange(len(datasets))
    width = 0.35
    
    axes[0].bar(x_pos - width/2, numpy_times, width, label='NumPy', alpha=0.8)
    axes[0].bar(x_pos + width/2, pytorch_times, width, label='PyTorch', alpha=0.8)
    axes[0].set_xlabel('Dataset')
    axes[0].set_ylabel('Training Time (seconds)')
    axes[0].set_title('Training Time Comparison')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(datasets)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy comparison
    numpy_accs = [all_results[d]['NumPy']['test_accuracy'] for d in datasets]
    pytorch_accs = [all_results[d]['PyTorch']['test_accuracy'] for d in datasets]
    
    axes[1].bar(x_pos - width/2, numpy_accs, width, label='NumPy', alpha=0.8)
    axes[1].bar(x_pos + width/2, pytorch_accs, width, label='PyTorch', alpha=0.8)
    axes[1].set_xlabel('Dataset')
    axes[1].set_ylabel('Test Accuracy')
    axes[1].set_title('Accuracy Comparison')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(datasets)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("plots/framework_summary_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Check if PyTorch is available
    try:
        import torch
        print("PyTorch is available. Running framework comparison...")
        results = comprehensive_framework_comparison()
        print("\nFramework comparison complete!")
    except ImportError:
        print("PyTorch is not available. Install with: pip install torch")
        print("Running basic PyTorch test with synthetic data...")
        
        # Create simple test
        X, y = make_xor_dataset(n_samples=400, noise=0.1, random_state=42)
        print(f"Created XOR dataset: {X.shape[0]} samples")
        
        # Test NumPy version only
        from mlp_numpy import FeedforwardNeuralNet
        mlp = FeedforwardNeuralNet(layers=[2, 8, 1], random_state=42)
        mlp.fit(X, y, epochs=200, verbose=True)
        
        results = mlp.evaluate(X, y)
        print(f"NumPy MLP accuracy: {results['accuracy']:.4f}") 