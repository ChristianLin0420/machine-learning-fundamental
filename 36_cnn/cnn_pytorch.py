#!/usr/bin/env python3
"""
CNN PyTorch Implementation
==========================

Modern CNN implementation using PyTorch with various architectures
and comprehensive experiments for image classification.

Author: ML Fundamentals Course
Day: 36 - CNNs
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Any, Optional
import time
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class SimpleCNN(nn.Module):
    """Simple CNN architecture."""
    
    def __init__(self, input_channels: int = 1, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Calculate the size of flattened features
        self.feature_size = self._get_feature_size(input_channels)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def _get_feature_size(self, input_channels: int) -> int:
        """Calculate the size of features after conv layers."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 28, 28)
            features = self.features(dummy_input)
            return features.numel()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x


class DeepCNN(nn.Module):
    """Deeper CNN architecture with batch normalization."""
    
    def __init__(self, input_channels: int = 1, num_classes: int = 10):
        super(DeepCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
        )
        
        # Calculate feature size
        self.feature_size = self._get_feature_size(input_channels)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def _get_feature_size(self, input_channels: int) -> int:
        """Calculate the size of features after conv layers."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 28, 28)
            features = self.features(dummy_input)
            return features.numel()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    """Simple residual block for ResNet-like architecture."""
    
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection."""
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual  # Skip connection
        out = F.relu(out)
        
        return out


class ResNetCNN(nn.Module):
    """CNN with residual connections."""
    
    def __init__(self, input_channels: int = 1, num_classes: int = 10):
        super(ResNetCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.res_block1 = ResidualBlock(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.res_block2 = ResidualBlock(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate feature size
        self.feature_size = self._get_feature_size(input_channels)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def _get_feature_size(self, input_channels: int) -> int:
        """Calculate the size of features after conv layers."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 28, 28)
            x = self.conv1(dummy_input)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.res_block1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.res_block2(x)
            x = self.pool2(x)
            return x.numel()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.res_block1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.res_block2(x)
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x


class CNNTrainer:
    """Training utility for CNN models."""
    
    def __init__(self, model: nn.Module, device: torch.device = device):
        self.model = model.to(device)
        self.device = device
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, train_loader: DataLoader, criterion: nn.Module, 
                   optimizer: optim.Optimizer) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader = None,
            epochs: int = 10, learning_rate: float = 0.001, 
            weight_decay: float = 1e-4, verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            learning_rate: Learning rate
            weight_decay: L2 regularization strength
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        if verbose:
            print(f"Training {self.model.__class__.__name__} for {epochs} epochs...")
            print(f"Learning rate: {learning_rate}, Weight decay: {weight_decay}")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validate
            if val_loader is not None:
                val_loss, val_acc = self.validate(val_loader, criterion)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
            
            # Update learning rate
            scheduler.step()
            
            if verbose:
                epoch_time = time.time() - start_time
                if val_loader is not None:
                    print(f"Epoch {epoch+1:2d}/{epochs} - {epoch_time:.1f}s - "
                          f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - "
                          f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
                else:
                    print(f"Epoch {epoch+1:2d}/{epochs} - {epoch_time:.1f}s - "
                          f"loss: {train_loss:.4f} - acc: {train_acc:.4f}")
        
        return self.history


def load_mnist_pytorch() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load MNIST dataset for PyTorch.
    
    Returns:
        train_loader, val_loader, test_loader
    """
    print("Loading MNIST dataset...")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Create validation split
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def load_cifar10_pytorch() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load CIFAR-10 dataset for PyTorch.
    
    Returns:
        train_loader, val_loader, test_loader
    """
    print("Loading CIFAR-10 dataset...")
    
    # Data transforms with augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    # Create validation split
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Update validation dataset transform
    val_dataset.dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=False, transform=transform_test
    )
    
    # Create data loaders
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def visualize_pytorch_training(history: Dict[str, List[float]], title: str = "Training History",
                              save_path: str = None) -> None:
    """Visualize PyTorch training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Training Loss', marker='o')
    if 'val_loss' in history and history['val_loss']:
        axes[0].plot(history['val_loss'], label='Validation Loss', marker='s')
    axes[0].set_title(f'{title} - Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Training Accuracy', marker='o')
    if 'val_acc' in history and history['val_acc']:
        axes[1].plot(history['val_acc'], label='Validation Accuracy', marker='s')
    axes[1].set_title(f'{title} - Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    
    plt.show()


def visualize_pytorch_filters(model: nn.Module, save_path: str = None) -> None:
    """Visualize learned filters from first conv layer."""
    # Get first conv layer
    first_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            first_conv = module
            break
    
    if first_conv is None:
        print("No convolution layer found!")
        return
    
    # Get filter weights
    filters = first_conv.weight.data.cpu().numpy()  # Shape: (out_channels, in_channels, h, w)
    
    num_filters = min(16, filters.shape[0])  # Show max 16 filters
    cols = 4
    rows = (num_filters + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_filters):
        row = i // cols
        col = i % cols
        
        # Get first input channel of filter
        filter_img = filters[i, 0]  # Shape: (h, w)
        
        # Normalize for visualization
        filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min() + 1e-8)
        
        axes[row, col].imshow(filter_img, cmap='RdBu')
        axes[row, col].set_title(f'Filter {i+1}')
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(num_filters, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.suptitle('Learned Convolution Filters (PyTorch)', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Filter visualization saved to {save_path}")
    
    plt.show()


def get_feature_maps(model: nn.Module, input_tensor: torch.Tensor, 
                    layer_name: str = 'first_conv') -> torch.Tensor:
    """Extract feature maps from a specific layer."""
    feature_maps = {}
    
    def hook_fn(module, input, output):
        feature_maps[layer_name] = output
    
    # Register hook
    if layer_name == 'first_conv':
        # Find first conv layer
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                handle = module.register_forward_hook(hook_fn)
                break
    
    # Forward pass
    with torch.no_grad():
        model.eval()
        _ = model(input_tensor)
    
    # Remove hook
    handle.remove()
    
    return feature_maps.get(layer_name, None)


def visualize_pytorch_feature_maps(model: nn.Module, input_sample: torch.Tensor,
                                  save_path: str = None) -> None:
    """Visualize feature maps from PyTorch model."""
    # Get feature maps
    feature_maps = get_feature_maps(model, input_sample.unsqueeze(0))
    
    if feature_maps is None:
        print("Could not extract feature maps!")
        return
    
    feature_maps = feature_maps.cpu().numpy()  # Shape: (1, channels, h, w)
    
    # Show subset of feature maps
    num_maps = min(16, feature_maps.shape[1])
    cols = 4
    rows = (num_maps + cols - 1) // cols + 1  # +1 for original image
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 2))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # Original image
    original = input_sample.cpu().numpy()
    if original.shape[0] == 1:  # Grayscale
        axes[0, 0].imshow(original[0], cmap='gray')
    else:  # RGB
        axes[0, 0].imshow(np.transpose(original, (1, 2, 0)))
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Hide other subplots in first row
    for i in range(1, cols):
        axes[0, i].axis('off')
    
    # Feature maps
    for i in range(num_maps):
        row = (i // cols) + 1
        col = i % cols
        
        feature_map = feature_maps[0, i]
        im = axes[row, col].imshow(feature_map, cmap='viridis')
        axes[row, col].set_title(f'Feature {i+1}')
        axes[row, col].axis('off')
        plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
    
    plt.suptitle('Feature Maps from First Conv Layer (PyTorch)', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature map visualization saved to {save_path}")
    
    plt.show()


def compare_cnn_architectures() -> Dict[str, Any]:
    """Compare different CNN architectures on MNIST."""
    print("\nComparing CNN Architectures on MNIST...")
    print("=" * 50)
    
    # Load data
    train_loader, val_loader, test_loader = load_mnist_pytorch()
    
    # Define architectures
    architectures = {
        'SimpleCNN': SimpleCNN(input_channels=1, num_classes=10),
        'DeepCNN': DeepCNN(input_channels=1, num_classes=10),
        'ResNetCNN': ResNetCNN(input_channels=1, num_classes=10)
    }
    
    results = {}
    
    for arch_name, model in architectures.items():
        print(f"\nTraining {arch_name}...")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Train model
        trainer = CNNTrainer(model, device)
        history = trainer.fit(
            train_loader, val_loader, 
            epochs=15, learning_rate=0.001, verbose=False
        )
        
        # Final evaluation
        test_loss, test_acc = trainer.validate(test_loader, nn.CrossEntropyLoss())
        
        print(f"  Final Test Accuracy: {test_acc:.4f}")
        
        results[arch_name] = {
            'model': model,
            'trainer': trainer,
            'history': history,
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'total_params': total_params,
            'trainable_params': trainable_params
        }
    
    return results


def create_confusion_matrix(model: nn.Module, test_loader: DataLoader, 
                          class_names: List[str] = None, save_path: str = None) -> None:
    """Create and visualize confusion matrix."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            
            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.numpy().flatten())
    
    # Create confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names if class_names else range(len(cm)),
                yticklabels=class_names if class_names else range(len(cm)))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def analyze_misclassifications(model: nn.Module, test_loader: DataLoader,
                             save_path: str = None) -> None:
    """Analyze and visualize misclassified examples."""
    model.eval()
    misclassified = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            # Find misclassified examples
            wrong_idx = (pred != target.to(device)).nonzero(as_tuple=True)[0]
            
            for idx in wrong_idx:
                if len(misclassified) < 20:  # Collect max 20 examples
                    misclassified.append({
                        'image': data[idx].cpu(),
                        'predicted': pred[idx].item(),
                        'actual': target[idx].item(),
                        'confidence': F.softmax(output[idx], dim=0).max().item()
                    })
    
    # Visualize misclassifications
    if misclassified:
        cols = 5
        rows = min(4, (len(misclassified) + cols - 1) // cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(min(len(misclassified), rows * cols)):
            row = i // cols
            col = i % cols
            
            example = misclassified[i]
            image = example['image'].numpy()
            
            if image.shape[0] == 1:  # Grayscale
                axes[row, col].imshow(image[0], cmap='gray')
            else:  # RGB
                axes[row, col].imshow(np.transpose(image, (1, 2, 0)))
            
            axes[row, col].set_title(
                f"Pred: {example['predicted']}, Actual: {example['actual']}\n"
                f"Conf: {example['confidence']:.2f}"
            )
            axes[row, col].axis('off')
        
        # Hide unused subplots
        for i in range(len(misclassified), rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.suptitle('Misclassified Examples', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Misclassification analysis saved to {save_path}")
        
        plt.show()


def run_comprehensive_pytorch_experiments() -> Dict[str, Any]:
    """Run comprehensive PyTorch CNN experiments."""
    print("Starting Comprehensive PyTorch CNN Experiments")
    print("=" * 60)
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    all_results = {}
    
    # Experiment 1: Architecture comparison on MNIST
    print("\n" + "=" * 50)
    print("EXPERIMENT 1: Architecture Comparison (MNIST)")
    print("=" * 50)
    
    mnist_results = compare_cnn_architectures()
    all_results['mnist_architectures'] = mnist_results
    
    # Visualize best model
    best_model_name = max(mnist_results.keys(), 
                         key=lambda k: mnist_results[k]['test_accuracy'])
    best_model = mnist_results[best_model_name]['model']
    best_history = mnist_results[best_model_name]['history']
    
    print(f"\nBest model: {best_model_name}")
    print(f"Test accuracy: {mnist_results[best_model_name]['test_accuracy']:.4f}")
    
    # Visualizations for best model
    visualize_pytorch_training(best_history, f"Best Model ({best_model_name})",
                              'plots/pytorch_best_training.png')
    
    visualize_pytorch_filters(best_model, 'plots/pytorch_learned_filters.png')
    
    # Get test data for visualization
    _, _, test_loader = load_mnist_pytorch()
    
    # Feature maps for sample
    test_sample, _ = next(iter(test_loader))
    sample_input = test_sample[0]  # First sample
    visualize_pytorch_feature_maps(best_model, sample_input, 
                                  'plots/pytorch_feature_maps.png')
    
    # Confusion matrix
    create_confusion_matrix(best_model, test_loader, 
                          class_names=[str(i) for i in range(10)],
                          save_path='plots/pytorch_confusion_matrix.png')
    
    # Misclassification analysis
    analyze_misclassifications(best_model, test_loader,
                             'plots/pytorch_misclassifications.png')
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Test accuracies
    plt.subplot(2, 2, 1)
    arch_names = list(mnist_results.keys())
    test_accs = [mnist_results[name]['test_accuracy'] for name in arch_names]
    param_counts = [mnist_results[name]['total_params'] for name in arch_names]
    
    bars = plt.bar(arch_names, test_accs)
    plt.title('Test Accuracy by Architecture')
    plt.ylabel('Test Accuracy')
    plt.xticks(rotation=45)
    
    # Add parameter count labels
    for bar, param_count in zip(bars, param_counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{param_count:,} params', ha='center', va='bottom', fontsize=8)
    
    # Subplot 2: Training curves comparison
    plt.subplot(2, 2, 2)
    for arch_name in arch_names:
        history = mnist_results[arch_name]['history']
        plt.plot(history['val_acc'], label=f'{arch_name}', marker='o')
    plt.title('Validation Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Subplot 3: Loss curves
    plt.subplot(2, 2, 3)
    for arch_name in arch_names:
        history = mnist_results[arch_name]['history']
        plt.plot(history['val_loss'], label=f'{arch_name}', marker='s')
    plt.title('Validation Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Subplot 4: Parameter efficiency
    plt.subplot(2, 2, 4)
    plt.scatter(param_counts, test_accs, s=100, alpha=0.7)
    for i, name in enumerate(arch_names):
        plt.annotate(name, (param_counts[i], test_accs[i]), 
                    xytext=(5, 5), textcoords='offset points')
    plt.title('Parameter Efficiency')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Test Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/pytorch_architecture_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nPyTorch CNN experiments completed!")
    print("Check the plots/ directory for visualizations.")
    
    return all_results


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    results = run_comprehensive_pytorch_experiments()