"""
Synthetic Data Generation for Perceptron Testing

This module provides various synthetic datasets to test different aspects
of the perceptron algorithm including linearly separable and non-separable data.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles, make_moons


def make_linearly_separable_data(n_samples=100, n_features=2, random_state=42):
    """
    Generate linearly separable binary classification data.
    
    Args:
        n_samples (int): Number of samples to generate
        n_features (int): Number of features
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X, y) where X is features and y is binary labels {-1, 1}
    """
    np.random.seed(random_state)
    
    # Generate two well-separated clusters
    X, y = make_blobs(n_samples=n_samples, centers=2, n_features=n_features, 
                      cluster_std=1.0, center_box=(-3.0, 3.0), random_state=random_state)
    
    # Convert labels to {-1, 1}
    y = np.where(y == 0, -1, 1)
    
    return X, y


def make_noisy_linear_data(n_samples=100, n_features=2, noise_level=0.3, random_state=42):
    """
    Generate linearly separable data with added noise to make it harder to separate.
    
    Args:
        n_samples (int): Number of samples
        n_features (int): Number of features
        noise_level (float): Amount of noise to add (0-1)
        random_state (int): Random seed
        
    Returns:
        tuple: (X, y) with noisy linearly separable data
    """
    np.random.seed(random_state)
    
    # Start with linearly separable data
    X, y = make_linearly_separable_data(n_samples, n_features, random_state)
    
    # Add noise to features
    noise = np.random.normal(0, noise_level, X.shape)
    X_noisy = X + noise
    
    # Flip some labels to make it non-linearly separable
    flip_indices = np.random.choice(len(y), size=int(noise_level * len(y)), replace=False)
    y_noisy = y.copy()
    y_noisy[flip_indices] *= -1
    
    return X_noisy, y_noisy


def make_xor_data(n_samples=200, noise=0.1, random_state=42):
    """
    Generate XOR-like data that is not linearly separable.
    
    Args:
        n_samples (int): Number of samples
        noise (float): Amount of noise to add
        random_state (int): Random seed
        
    Returns:
        tuple: (X, y) XOR pattern data
    """
    np.random.seed(random_state)
    
    # Create four clusters in XOR pattern
    X = np.random.randn(n_samples, 2)
    
    # Define XOR pattern
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
    y = np.where(y == 0, -1, 1)
    
    # Add some noise
    X += np.random.normal(0, noise, X.shape)
    
    return X, y


def make_concentric_circles_data(n_samples=200, noise=0.1, random_state=42):
    """
    Generate concentric circles data for kernel perceptron testing.
    
    Args:
        n_samples (int): Number of samples
        noise (float): Amount of noise
        random_state (int): Random seed
        
    Returns:
        tuple: (X, y) concentric circles data
    """
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.6, random_state=random_state)
    
    # Convert labels to {-1, 1}
    y = np.where(y == 0, -1, 1)
    
    return X, y


def make_moons_data(n_samples=200, noise=0.1, random_state=42):
    """
    Generate two moons dataset for testing.
    
    Args:
        n_samples (int): Number of samples
        noise (float): Amount of noise
        random_state (int): Random seed
        
    Returns:
        tuple: (X, y) moons data
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    
    # Convert labels to {-1, 1}
    y = np.where(y == 0, -1, 1)
    
    return X, y


def make_multiclass_data(n_samples=300, n_classes=3, n_features=2, random_state=42):
    """
    Generate multi-class classification data.
    
    Args:
        n_samples (int): Number of samples
        n_classes (int): Number of classes
        n_features (int): Number of features
        random_state (int): Random seed
        
    Returns:
        tuple: (X, y) multi-class data with labels {0, 1, 2, ...}
    """
    X, y = make_blobs(n_samples=n_samples, centers=n_classes, n_features=n_features,
                      cluster_std=1.5, random_state=random_state)
    
    return X, y


def visualize_dataset(X, y, title="Dataset Visualization", save_path=None):
    """
    Visualize a 2D dataset with different colors for each class.
    
    Args:
        X (np.ndarray): Features (n_samples, 2)
        y (np.ndarray): Labels
        title (str): Plot title
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Get unique classes
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


if __name__ == "__main__":
    # Create plots directory
    import os
    os.makedirs("plots", exist_ok=True)
    
    # Generate and visualize different datasets
    datasets = {
        "linearly_separable": make_linearly_separable_data(),
        "noisy_linear": make_noisy_linear_data(),
        "xor": make_xor_data(),
        "concentric_circles": make_concentric_circles_data(),
        "moons": make_moons_data(),
        "multiclass": make_multiclass_data()
    }
    
    # Create a subplot for all datasets
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, (name, (X, y)) in enumerate(datasets.items()):
        ax = axes[i]
        
        # Get unique classes and colors
        classes = np.unique(y)
        colors = plt.cm.Set1(np.linspace(0, 1, len(classes)))
        
        for j, cls in enumerate(classes):
            mask = y == cls
            ax.scatter(X[mask, 0], X[mask, 1], c=[colors[j]], 
                      label=f'Class {cls}', alpha=0.7, s=30)
        
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title(name.replace('_', ' ').title())
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("plots/synthetic_datasets_overview.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Generated synthetic datasets:")
    for name, (X, y) in datasets.items():
        print(f"- {name}: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes") 