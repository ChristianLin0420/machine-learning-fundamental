"""
Datasets for Neural Network Testing

This module provides various synthetic datasets commonly used to test
neural network architectures and training algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_blobs, make_classification


def make_xor_dataset(n_samples=1000, noise=0.1, random_state=42):
    """
    Generate XOR dataset - the classic non-linearly separable problem.
    
    Args:
        n_samples (int): Number of samples to generate
        noise (float): Amount of noise to add
        random_state (int): Random seed
        
    Returns:
        tuple: (X, y) where X is features and y is binary labels
    """
    np.random.seed(random_state)
    
    # Generate random points in [0,1] x [0,1]
    X = np.random.uniform(0, 1, (n_samples, 2))
    
    # XOR logic: y = (x1 > 0.5) XOR (x2 > 0.5)
    y = ((X[:, 0] > 0.5) ^ (X[:, 1] > 0.5)).astype(int)
    
    # Add noise
    X += np.random.normal(0, noise, X.shape)
    
    # Convert to -1, 1 labels for easier training
    y = 2 * y - 1
    
    return X, y


def make_spiral_dataset(n_samples=1000, noise=0.1, n_turns=1.5, random_state=42):
    """
    Generate two-class spiral dataset.
    
    Args:
        n_samples (int): Number of samples to generate
        noise (float): Amount of noise to add
        n_turns (float): Number of spiral turns
        random_state (int): Random seed
        
    Returns:
        tuple: (X, y) where X is features and y is binary labels
    """
    np.random.seed(random_state)
    
    n_per_class = n_samples // 2
    
    # Generate spiral parameters
    t = np.linspace(0, n_turns * 2 * np.pi, n_per_class)
    
    # First spiral (class 0)
    x1_0 = t * np.cos(t)
    y1_0 = t * np.sin(t)
    
    # Second spiral (class 1) - offset by π
    x1_1 = t * np.cos(t + np.pi)
    y1_1 = t * np.sin(t + np.pi)
    
    # Combine spirals
    X = np.vstack([np.column_stack([x1_0, y1_0]), 
                   np.column_stack([x1_1, y1_1])])
    
    # Add noise
    X += np.random.normal(0, noise, X.shape)
    
    # Create labels
    y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])
    
    # Convert to -1, 1 labels
    y = 2 * y - 1
    
    # Shuffle the data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y


def make_moons_dataset(n_samples=1000, noise=0.1, random_state=42):
    """
    Generate two interleaving half circles dataset.
    
    Args:
        n_samples (int): Number of samples
        noise (float): Amount of noise
        random_state (int): Random seed
        
    Returns:
        tuple: (X, y) with binary labels {-1, 1}
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    
    # Convert labels to {-1, 1}
    y = 2 * y - 1
    
    return X, y


def make_circles_dataset(n_samples=1000, noise=0.1, factor=0.6, random_state=42):
    """
    Generate concentric circles dataset.
    
    Args:
        n_samples (int): Number of samples
        noise (float): Amount of noise
        factor (float): Scale factor of inner circle
        random_state (int): Random seed
        
    Returns:
        tuple: (X, y) with binary labels {-1, 1}
    """
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)
    
    # Convert labels to {-1, 1}
    y = 2 * y - 1
    
    return X, y


def make_blobs_dataset(n_samples=1000, n_centers=3, n_features=2, cluster_std=1.0, random_state=42):
    """
    Generate isotropic Gaussian blobs for clustering/classification.
    
    Args:
        n_samples (int): Number of samples
        n_centers (int): Number of cluster centers
        n_features (int): Number of features
        cluster_std (float): Standard deviation of clusters
        random_state (int): Random seed
        
    Returns:
        tuple: (X, y) with multi-class labels
    """
    X, y = make_blobs(n_samples=n_samples, centers=n_centers, n_features=n_features,
                      cluster_std=cluster_std, random_state=random_state)
    
    return X, y


def make_multiclass_classification_dataset(n_samples=1000, n_features=2, n_classes=3, 
                                         n_redundant=0, n_informative=2, 
                                         n_clusters_per_class=1, random_state=42):
    """
    Generate a multi-class classification dataset.
    
    Args:
        n_samples (int): Number of samples
        n_features (int): Number of features
        n_classes (int): Number of classes
        n_redundant (int): Number of redundant features
        n_informative (int): Number of informative features
        n_clusters_per_class (int): Number of clusters per class
        random_state (int): Random seed
        
    Returns:
        tuple: (X, y) with multi-class labels
    """
    X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                              n_classes=n_classes, n_redundant=n_redundant,
                              n_informative=n_informative, 
                              n_clusters_per_class=n_clusters_per_class,
                              random_state=random_state)
    
    return X, y


def make_regression_dataset(n_samples=1000, n_features=1, noise=0.1, 
                          function_type='sine', random_state=42):
    """
    Generate synthetic regression datasets.
    
    Args:
        n_samples (int): Number of samples
        n_features (int): Number of input features
        noise (float): Amount of noise to add
        function_type (str): Type of function ('sine', 'polynomial', 'linear')
        random_state (int): Random seed
        
    Returns:
        tuple: (X, y) for regression
    """
    np.random.seed(random_state)
    
    if n_features == 1:
        X = np.random.uniform(-2, 2, (n_samples, 1))
        
        if function_type == 'sine':
            y = np.sin(2 * np.pi * X[:, 0]) + 0.5 * np.sin(4 * np.pi * X[:, 0])
        elif function_type == 'polynomial':
            y = X[:, 0]**3 - 2 * X[:, 0]**2 + X[:, 0]
        elif function_type == 'linear':
            y = 2 * X[:, 0] + 1
        else:
            raise ValueError(f"Unknown function type: {function_type}")
        
        # Add noise
        y += np.random.normal(0, noise, y.shape)
        
    else:
        # Multi-dimensional regression
        X = np.random.uniform(-1, 1, (n_samples, n_features))
        y = np.sum(X**2, axis=1) + np.random.normal(0, noise, n_samples)
    
    return X, y.reshape(-1, 1)


def create_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Split dataset into train and test sets.
    
    Args:
        X (np.ndarray): Features
        y (np.ndarray): Labels
        test_size (float): Fraction for test set
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    # Random permutation
    indices = np.random.permutation(n_samples)
    
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test


def normalize_features(X_train, X_test=None):
    """
    Normalize features to zero mean and unit variance.
    
    Args:
        X_train (np.ndarray): Training features
        X_test (np.ndarray): Test features (optional)
        
    Returns:
        tuple: (X_train_norm, X_test_norm, mean, std) or (X_train_norm, mean, std)
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    
    # Avoid division by zero
    std[std == 0] = 1
    
    X_train_norm = (X_train - mean) / std
    
    if X_test is not None:
        X_test_norm = (X_test - mean) / std
        return X_train_norm, X_test_norm, mean, std
    else:
        return X_train_norm, mean, std


def plot_2d_dataset(X, y, title="Dataset", save_path=None, figsize=(8, 6)):
    """
    Plot a 2D dataset with different colors for each class.
    
    Args:
        X (np.ndarray): 2D features
        y (np.ndarray): Labels
        title (str): Plot title
        save_path (str): Path to save plot
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    
    # Get unique classes and assign colors
    classes = np.unique(y)
    colors = plt.cm.Set1(np.linspace(0, 1, len(classes)))
    
    for i, cls in enumerate(classes):
        mask = y == cls
        plt.scatter(X[mask, 0], X[mask, 1], c=[colors[i]], 
                   label=f'Class {cls}', alpha=0.7, s=50)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_regression_dataset(X, y, title="Regression Dataset", save_path=None, figsize=(8, 6)):
    """
    Plot a regression dataset.
    
    Args:
        X (np.ndarray): 1D features
        y (np.ndarray): Target values
        title (str): Plot title
        save_path (str): Path to save plot
        figsize (tuple): Figure size
    """
    plt.figure(figsize=figsize)
    
    if X.shape[1] == 1:
        # Sort for better visualization
        sort_idx = np.argsort(X[:, 0])
        plt.plot(X[sort_idx, 0], y[sort_idx], 'bo', alpha=0.6, markersize=4)
        plt.xlabel('Input Feature')
        plt.ylabel('Target Value')
    else:
        plt.scatter(range(len(y)), y, alpha=0.6)
        plt.xlabel('Sample Index')
        plt.ylabel('Target Value')
    
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def demonstrate_datasets():
    """
    Create and visualize all available datasets.
    """
    # Create plots directory
    import os
    os.makedirs("plots", exist_ok=True)
    
    print("Neural Network Dataset Demonstration")
    print("=" * 50)
    
    # Binary classification datasets
    binary_datasets = {
        "XOR": make_xor_dataset(n_samples=400, noise=0.05),
        "Spiral": make_spiral_dataset(n_samples=400, noise=0.1),
        "Moons": make_moons_dataset(n_samples=400, noise=0.1),
        "Circles": make_circles_dataset(n_samples=400, noise=0.1)
    }
    
    # Create subplot for binary classification datasets
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i, (name, (X, y)) in enumerate(binary_datasets.items()):
        ax = axes[i]
        
        # Plot data
        classes = np.unique(y)
        colors = ['red' if c == -1 else 'blue' for c in classes]
        
        for j, cls in enumerate(classes):
            mask = y == cls
            color = 'red' if cls == -1 else 'blue'
            label = 'Class -1' if cls == -1 else 'Class +1'
            ax.scatter(X[mask, 0], X[mask, 1], c=color, alpha=0.7, s=30, label=label)
        
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(f'{name} Dataset')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        print(f"{name}: {X.shape[0]} samples, {len(classes)} classes")
    
    plt.tight_layout()
    plt.savefig("plots/binary_classification_datasets.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Multi-class dataset
    print(f"\nMulti-class Classification:")
    X_multi, y_multi = make_blobs_dataset(n_samples=300, n_centers=3)
    plot_2d_dataset(X_multi, y_multi, "Multi-class Blobs Dataset", 
                   "plots/multiclass_dataset.png")
    print(f"Blobs: {X_multi.shape[0]} samples, {len(np.unique(y_multi))} classes")
    
    # Regression datasets
    print(f"\nRegression Datasets:")
    regression_types = ['sine', 'polynomial', 'linear']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, func_type in enumerate(regression_types):
        X_reg, y_reg = make_regression_dataset(n_samples=200, function_type=func_type, noise=0.1)
        
        # Sort for better visualization
        sort_idx = np.argsort(X_reg[:, 0])
        axes[i].plot(X_reg[sort_idx, 0], y_reg[sort_idx], 'bo', alpha=0.6, markersize=4)
        axes[i].set_xlabel('Input Feature')
        axes[i].set_ylabel('Target Value')
        axes[i].set_title(f'{func_type.title()} Function')
        axes[i].grid(True, alpha=0.3)
        
        print(f"{func_type.title()}: {X_reg.shape[0]} samples")
    
    plt.tight_layout()
    plt.savefig("plots/regression_datasets.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Dataset statistics
    print(f"\nDataset Statistics:")
    print("-" * 40)
    
    for name, (X, y) in binary_datasets.items():
        X_norm, _, _ = normalize_features(X)
        print(f"{name}:")
        print(f"  Shape: {X.shape}")
        print(f"  Feature range: [{X.min():.2f}, {X.max():.2f}]")
        print(f"  Normalized range: [{X_norm.min():.2f}, {X_norm.max():.2f}]")
        print(f"  Class balance: {np.bincount((y + 1).astype(int))}")  # +1 to handle -1,1 labels
        print()


if __name__ == "__main__":
    demonstrate_datasets() 