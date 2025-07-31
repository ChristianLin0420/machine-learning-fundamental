"""
Regularization Techniques for Neural Networks - PyTorch Implementation

This module implements various regularization techniques using PyTorch:
- Weight Decay (L2 regularization in optimizer)
- Dropout layers
- Batch Normalization
- Early Stopping integration
- Learning rate scheduling

Author: ML Fundamentals Course
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time
from collections import defaultdict

# Import early stopping
from early_stopping import EarlyStopping

class RegularizedMLP(nn.Module):
    """
    Multi-Layer Perceptron with comprehensive regularization techniques
    
    Features:
    - Configurable dropout layers
    - Batch normalization option
    - Flexible architecture
    - Weight decay through optimizer
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_sizes: List[int],
                 output_size: int,
                 dropout_rate: float = 0.0,
                 use_batch_norm: bool = False,
                 activation: str = 'relu'):
        """
        Initialize the regularized MLP
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output units
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
            activation: Activation function ('relu', 'tanh', 'sigmoid')
        """
        super(RegularizedMLP, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Create layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Input to first hidden layer
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            # Linear layer
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            self.layers.append(layer)
            
            # Batch normalization (not for output layer)
            if use_batch_norm and i < len(layer_sizes) - 2:
                bn = nn.BatchNorm1d(layer_sizes[i + 1])
                self.batch_norms.append(bn)
            
            # Dropout (not for output layer)
            if dropout_rate > 0 and i < len(layer_sizes) - 2:
                dropout = nn.Dropout(dropout_rate)
                self.dropouts.append(dropout)
        
        # Activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/He initialization"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # He initialization for ReLU, Xavier for others
                if self.activation == F.relu:
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        for i, layer in enumerate(self.layers[:-1]):
            # Linear transformation
            x = layer(x)
            
            # Batch normalization
            if self.use_batch_norm and i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            
            # Activation
            x = self.activation(x)
            
            # Dropout
            if self.dropout_rate > 0 and i < len(self.dropouts):
                x = self.dropouts[i](x)
        
        # Output layer (no activation)
        x = self.layers[-1](x)
        return x

class RegularizationTrainer:
    """
    Trainer class for regularized neural networks with comprehensive tracking
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = 'cpu',
                 verbose: bool = True):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model to train
            device: Device to use ('cpu' or 'cuda')
            verbose: Whether to print training progress
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.verbose = verbose
        
        # Training history
        self.history = defaultdict(list)
        
    def train_epoch(self, 
                   train_loader: DataLoader,
                   optimizer: optim.Optimizer,
                   criterion: nn.Module) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            
            # Compute loss
            if self.model.output_size == 1:
                # Binary classification
                output = torch.sigmoid(output)
                target = target.float().view(-1, 1)  # Ensure target has same shape as output
                loss = criterion(output, target)
                pred = (output > 0.5).float()
            else:
                # Multi-class classification
                loss = criterion(output, target.long())
                pred = output.argmax(dim=1)
                target = target.long()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            if self.model.output_size == 1:
                correct += pred.eq(target).sum().item()
            else:
                correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, 
                      val_loader: DataLoader,
                      criterion: nn.Module) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # Compute loss
                if self.model.output_size == 1:
                    # Binary classification
                    output = torch.sigmoid(output)
                    target = target.float().view(-1, 1)  # Ensure target has same shape as output
                    loss = criterion(output, target)
                    pred = (output > 0.5).float()
                else:
                    # Multi-class classification
                    loss = criterion(output, target.long())
                    pred = output.argmax(dim=1)
                    target = target.long()
                
                # Statistics
                total_loss += loss.item()
                if self.model.output_size == 1:
                    correct += pred.eq(target).sum().item()
                else:
                    correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def fit(self,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            epochs: int = 100,
            learning_rate: float = 0.001,
            weight_decay: float = 0.0,
            early_stopping: Optional[EarlyStopping] = None,
            lr_scheduler: Optional[optim.lr_scheduler._LRScheduler] = None) -> Dict:
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            learning_rate: Learning rate
            weight_decay: L2 regularization strength
            early_stopping: Early stopping callback
            lr_scheduler: Learning rate scheduler
            
        Returns:
            Training history dictionary
        """
        # Setup optimizer and criterion
        optimizer = optim.Adam(self.model.parameters(), 
                             lr=learning_rate, 
                             weight_decay=weight_decay)
        
        if self.model.output_size == 1:
            criterion = nn.BCELoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Reset history
        self.history = defaultdict(list)
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validate
            if val_loader is not None:
                val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            else:
                val_loss, val_acc = train_loss, train_acc
            
            # Update learning rate
            if lr_scheduler is not None:
                if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(val_loss)
                else:
                    lr_scheduler.step()
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(optimizer.param_groups[0]['lr'])
            
            # Early stopping
            if early_stopping is not None:
                early_stopping.set_current_epoch(epoch + 1)
                if early_stopping.check(val_loss, [p.data.cpu().numpy() for p in self.model.parameters()]):
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    
                    # Restore best weights if available
                    if early_stopping.get_best_weights() is not None:
                        for param, best_weight in zip(self.model.parameters(), early_stopping.get_best_weights()):
                            param.data = torch.from_numpy(best_weight).to(self.device)
                    break
            
            # Print progress
            if self.verbose and (epoch + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch + 1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                      f"LR: {optimizer.param_groups[0]['lr']:.6f}, "
                      f"Time: {elapsed:.1f}s")
        
        return dict(self.history)
    
    def get_weight_statistics(self) -> Dict:
        """Get statistics about model weights"""
        all_weights = []
        for param in self.model.parameters():
            if param.requires_grad and len(param.shape) >= 2:  # Weight matrices only
                all_weights.append(param.data.cpu().numpy().flatten())
        
        if not all_weights:
            return {}
        
        all_weights = np.concatenate(all_weights)
        
        return {
            'mean': float(np.mean(all_weights)),
            'std': float(np.std(all_weights)),
            'l1_norm': float(np.sum(np.abs(all_weights))),
            'l2_norm': float(np.sqrt(np.sum(all_weights ** 2))),
            'max': float(np.max(np.abs(all_weights))),
            'sparsity': float(np.mean(np.abs(all_weights) < 1e-6))
        }

def create_data_loaders(X: np.ndarray, 
                       y: np.ndarray,
                       batch_size: int = 32,
                       val_split: float = 0.2) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders"""
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    # Create dataset
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # Split into train/validation
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def compare_regularization_pytorch(X: np.ndarray, 
                                  y: np.ndarray,
                                  epochs: int = 100,
                                  batch_size: int = 32) -> Dict:
    """Compare different regularization techniques in PyTorch"""
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(X, y, batch_size)
    
    configs = {
        'No Regularization': {
            'dropout_rate': 0.0,
            'use_batch_norm': False,
            'weight_decay': 0.0,
            'early_stopping': None
        },
        'Dropout': {
            'dropout_rate': 0.3,
            'use_batch_norm': False,
            'weight_decay': 0.0,
            'early_stopping': None
        },
        'Weight Decay': {
            'dropout_rate': 0.0,
            'use_batch_norm': False,
            'weight_decay': 0.01,
            'early_stopping': None
        },
        'Batch Normalization': {
            'dropout_rate': 0.0,
            'use_batch_norm': True,
            'weight_decay': 0.0,
            'early_stopping': None
        },
        'Dropout + Weight Decay': {
            'dropout_rate': 0.3,
            'use_batch_norm': False,
            'weight_decay': 0.01,
            'early_stopping': None
        },
        'All Techniques': {
            'dropout_rate': 0.2,
            'use_batch_norm': True,
            'weight_decay': 0.005,
            'early_stopping': EarlyStopping(patience=10, min_delta=0.001, verbose=False)
        }
    }
    
    results = {}
    input_size = X.shape[1]
    output_size = 1 if len(y.shape) == 1 or y.shape[1] == 1 else y.shape[1]
    
    for name, config in configs.items():
        print(f"\nTraining {name}...")
        
        # Extract early stopping
        early_stopping = config.pop('early_stopping', None)
        if early_stopping is not None:
            early_stopping.reset()
        
        # Create model
        model = RegularizedMLP(
            input_size=input_size,
            hidden_sizes=[64, 32],
            output_size=output_size,
            dropout_rate=config['dropout_rate'],
            use_batch_norm=config['use_batch_norm']
        )
        
        # Create trainer
        trainer = RegularizationTrainer(model, verbose=False)
        
        # Train model
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            learning_rate=0.001,
            weight_decay=config['weight_decay'],
            early_stopping=early_stopping
        )
        
        # Store results
        results[name] = {
            'model': model,
            'history': history,
            'config': config,
            'final_val_loss': history['val_loss'][-1],
            'final_val_acc': history['val_acc'][-1],
            'weight_stats': trainer.get_weight_statistics()
        }
        
        print(f"Final Validation Accuracy: {history['val_acc'][-1]:.4f}")
        print(f"Final Validation Loss: {history['val_loss'][-1]:.4f}")
        print(f"Training completed in {len(history['train_loss'])} epochs")
    
    return results

def plot_pytorch_comparison(results: Dict, save_path: str = None):
    """Plot comparison of PyTorch regularization techniques"""
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
    ax1.set_title('Training vs Validation Loss (PyTorch)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation Accuracy
    ax2 = axes[0, 1]
    for name, result in results.items():
        history = result['history']
        epochs = range(1, len(history['val_acc']) + 1)
        ax2.plot(epochs, history['val_acc'], label=name)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title('Validation Accuracy Comparison (PyTorch)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate Schedule
    ax3 = axes[0, 2]
    for name, result in results.items():
        history = result['history']
        if 'lr' in history:
            epochs = range(1, len(history['lr']) + 1)
            ax3.plot(epochs, history['lr'], label=name)
    
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Final Performance
    ax4 = axes[1, 0]
    names = list(results.keys())
    val_accs = [results[name]['final_val_acc'] for name in names]
    val_losses = [results[name]['final_val_loss'] for name in names]
    
    x_pos = np.arange(len(names))
    ax4_twin = ax4.twinx()
    
    bars1 = ax4.bar(x_pos - 0.2, val_accs, 0.4, label='Accuracy', alpha=0.7)
    bars2 = ax4_twin.bar(x_pos + 0.2, val_losses, 0.4, label='Loss', alpha=0.7, color='red')
    
    ax4.set_xlabel('Regularization Method')
    ax4.set_ylabel('Validation Accuracy', color='blue')
    ax4_twin.set_ylabel('Validation Loss', color='red')
    ax4.set_title('Final Performance (PyTorch)')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(names, rotation=45, ha='right')
    
    # Plot 5: Weight Statistics
    ax5 = axes[1, 1]
    weight_l2_norms = [results[name]['weight_stats'].get('l2_norm', 0) for name in names]
    sparsity = [results[name]['weight_stats'].get('sparsity', 0) for name in names]
    
    ax5_twin = ax5.twinx()
    bars3 = ax5.bar(x_pos - 0.2, weight_l2_norms, 0.4, label='L2 Norm', alpha=0.7)
    bars4 = ax5_twin.bar(x_pos + 0.2, sparsity, 0.4, label='Sparsity', alpha=0.7, color='green')
    
    ax5.set_xlabel('Regularization Method')
    ax5.set_ylabel('Weight L2 Norm', color='blue')
    ax5_twin.set_ylabel('Weight Sparsity', color='green')
    ax5.set_title('Weight Statistics (PyTorch)')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(names, rotation=45, ha='right')
    
    # Plot 6: Training Efficiency (Epochs to Convergence)
    ax6 = axes[1, 2]
    epochs_to_convergence = [len(results[name]['history']['train_loss']) for name in names]
    
    bars5 = ax6.bar(names, epochs_to_convergence, alpha=0.7)
    ax6.set_xlabel('Regularization Method')
    ax6.set_ylabel('Epochs to Convergence')
    ax6.set_title('Training Efficiency (PyTorch)')
    ax6.set_xticklabels(names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars5, epochs_to_convergence):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def demonstrate_pytorch_regularization():
    """Comprehensive demonstration of PyTorch regularization techniques"""
    print("PyTorch Regularization Techniques Demonstration")
    print("=" * 55)
    
    # Generate synthetic dataset (same as NumPy version for comparison)
    print("1. Generating synthetic dataset...")
    np.random.seed(42)
    n_samples, n_features = 2000, 50
    X = np.random.randn(n_samples, n_features)
    
    # Only first 5 features are relevant
    true_weights = np.zeros(n_features)
    true_weights[:5] = np.random.randn(5) * 2
    
    # Generate target
    y_continuous = X @ true_weights + 0.1 * np.random.randn(n_samples)
    y = (y_continuous > np.median(y_continuous)).astype(float)
    
    # Normalize features
    X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y.astype(int))}")
    
    # Compare regularization techniques
    print("\n2. Comparing PyTorch regularization techniques...")
    results = compare_regularization_pytorch(X, y, epochs=100, batch_size=32)
    
    # Create visualization
    print("\n3. Creating comparison plots...")
    plot_pytorch_comparison(results, save_path='plots/pytorch_regularization_comparison.png')
    
    # Print summary
    print("\n4. Summary of PyTorch Results:")
    print("-" * 70)
    print(f"{'Method':<25} {'Val Acc':<10} {'Val Loss':<10} {'Epochs':<8} {'L2 Norm':<10}")
    print("-" * 70)
    
    for name, result in results.items():
        acc = result['final_val_acc']
        loss = result['final_val_loss']
        epochs = len(result['history']['train_loss'])
        l2_norm = result['weight_stats'].get('l2_norm', 0)
        print(f"{name:<25} {acc:<10.4f} {loss:<10.4f} {epochs:<8d} {l2_norm:<10.4f}")
    
    return results

if __name__ == "__main__":
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Run demonstration
    results = demonstrate_pytorch_regularization() 