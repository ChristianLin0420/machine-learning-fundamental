"""
Synthetic Anomaly Data Generation

This module creates synthetic datasets with known anomalies for testing
anomaly detection algorithms.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from typing import Tuple, Dict, Any
import os


class SyntheticAnomalyData:
    """Generate synthetic datasets with anomalies for testing detection algorithms."""
    
    def __init__(self, random_state: int = 42):
        """Initialize the data generator."""
        self.random_state = random_state
        np.random.seed(random_state)
    
    def gaussian_clusters_with_outliers(self, 
                                      n_normal: int = 300, 
                                      n_outliers: int = 30,
                                      centers: int = 3,
                                      cluster_std: float = 1.0,
                                      outlier_factor: float = 4.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate 2D Gaussian clusters with injected outliers.
        
        Args:
            n_normal: Number of normal points
            n_outliers: Number of outlier points
            centers: Number of cluster centers
            cluster_std: Standard deviation of clusters
            outlier_factor: How far outliers are from normal data
            
        Returns:
            X: Feature matrix
            y: Labels (0=normal, 1=anomaly)
        """
        # Generate normal data (Gaussian clusters)
        X_normal, _ = make_blobs(n_samples=n_normal, 
                                centers=centers, 
                                cluster_std=cluster_std,
                                random_state=self.random_state)
        
        # Calculate data bounds for outlier generation
        x_min, x_max = X_normal[:, 0].min(), X_normal[:, 0].max()
        y_min, y_max = X_normal[:, 1].min(), X_normal[:, 1].max()
        
        # Extend bounds for outliers
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # Generate outliers outside the normal data range
        outliers = []
        for _ in range(n_outliers):
            if np.random.rand() < 0.5:
                # Far outliers
                x = np.random.uniform(x_min - outlier_factor * x_range, 
                                    x_max + outlier_factor * x_range)
                y = np.random.uniform(y_min - outlier_factor * y_range, 
                                    y_max + outlier_factor * y_range)
            else:
                # Moderate outliers
                x = np.random.uniform(x_min - 0.5 * outlier_factor * x_range, 
                                    x_max + 0.5 * outlier_factor * x_range)
                y = np.random.uniform(y_min - 0.5 * outlier_factor * y_range, 
                                    y_max + 0.5 * outlier_factor * y_range)
            outliers.append([x, y])
        
        X_outliers = np.array(outliers)
        
        # Combine normal and outlier data
        X = np.vstack([X_normal, X_outliers])
        y = np.hstack([np.zeros(n_normal), np.ones(n_outliers)])
        
        return X, y
    
    def time_series_with_spikes(self, 
                               length: int = 1000,
                               spike_probability: float = 0.02,
                               spike_magnitude: float = 5.0,
                               noise_std: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate time series data with anomalous spikes.
        
        Args:
            length: Length of time series
            spike_probability: Probability of spike at each time point
            spike_magnitude: Magnitude multiplier for spikes
            noise_std: Standard deviation of background noise
            
        Returns:
            t: Time points
            y: Time series values
            anomalies: Binary array indicating anomalies
        """
        t = np.arange(length)
        
        # Base signal: combination of sinusoids + trend
        base_signal = (0.5 * np.sin(2 * np.pi * t / 50) + 
                      0.3 * np.sin(2 * np.pi * t / 20) +
                      0.01 * t)  # slight trend
        
        # Add noise
        y = base_signal + np.random.normal(0, noise_std, length)
        
        # Add spikes (anomalies)
        anomalies = np.random.rand(length) < spike_probability
        spike_values = np.random.choice([-1, 1], size=np.sum(anomalies)) * spike_magnitude
        y[anomalies] += spike_values * np.std(base_signal)
        
        return t, y, anomalies.astype(int)
    
    def mixed_distributions(self, 
                           n_normal: int = 200,
                           n_outliers: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate data from mixed distributions with outliers.
        
        Args:
            n_normal: Number of normal points
            n_outliers: Number of outlier points
            
        Returns:
            X: Feature matrix
            y: Labels (0=normal, 1=anomaly)
        """
        # Normal data: mixture of two Gaussians
        n1, n2 = n_normal // 2, n_normal - n_normal // 2
        
        X1 = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n1)
        X2 = np.random.multivariate_normal([3, 2], [[0.8, -0.3], [-0.3, 0.8]], n2)
        X_normal = np.vstack([X1, X2])
        
        # Outliers: random points far from normal data
        X_outliers = np.random.uniform(-5, 8, (n_outliers, 2))
        
        # Combine
        X = np.vstack([X_normal, X_outliers])
        y = np.hstack([np.zeros(n_normal), np.ones(n_outliers)])
        
        return X, y
    
    def moons_with_outliers(self, 
                           n_normal: int = 200,
                           n_outliers: int = 20,
                           noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate moon-shaped data with outliers.
        
        Args:
            n_normal: Number of normal points
            n_outliers: Number of outlier points
            noise: Noise level in normal data
            
        Returns:
            X: Feature matrix
            y: Labels (0=normal, 1=anomaly)
        """
        # Generate moon-shaped normal data
        X_normal, _ = make_moons(n_samples=n_normal, noise=noise, 
                                random_state=self.random_state)
        
        # Generate outliers
        X_outliers = np.random.uniform(-2, 3, (n_outliers, 2))
        
        # Combine
        X = np.vstack([X_normal, X_outliers])
        y = np.hstack([np.zeros(n_normal), np.ones(n_outliers)])
        
        return X, y
    
    def visualize_2d_data(self, X: np.ndarray, y: np.ndarray, title: str):
        """Visualize 2D anomaly data."""
        plt.figure(figsize=(10, 8))
        
        # Plot normal points
        normal_mask = y == 0
        plt.scatter(X[normal_mask, 0], X[normal_mask, 1], 
                   c='blue', alpha=0.6, s=50, label='Normal')
        
        # Plot anomalies
        anomaly_mask = y == 1
        plt.scatter(X[anomaly_mask, 0], X[anomaly_mask, 1], 
                   c='red', alpha=0.8, s=100, marker='x', 
                   linewidths=2, label='Anomaly')
        
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
    
    def visualize_time_series(self, t: np.ndarray, y: np.ndarray, 
                            anomalies: np.ndarray, title: str):
        """Visualize time series with anomalies."""
        plt.figure(figsize=(12, 6))
        
        # Plot time series
        plt.plot(t, y, 'b-', alpha=0.7, linewidth=1, label='Time Series')
        
        # Highlight anomalies
        anomaly_indices = np.where(anomalies == 1)[0]
        if len(anomaly_indices) > 0:
            plt.scatter(t[anomaly_indices], y[anomaly_indices], 
                       c='red', s=100, marker='o', alpha=0.8,
                       label='Anomalies', zorder=5)
        
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()


def generate_all_datasets() -> Dict[str, Any]:
    """Generate all synthetic datasets for anomaly detection testing."""
    generator = SyntheticAnomalyData()
    datasets = {}
    
    print("Generating synthetic anomaly datasets...")
    
    # 2D datasets
    datasets['gaussian_clusters'] = generator.gaussian_clusters_with_outliers()
    datasets['mixed_distributions'] = generator.mixed_distributions()
    datasets['moons'] = generator.moons_with_outliers()
    
    # Time series
    datasets['time_series'] = generator.time_series_with_spikes()
    
    print(f"Generated {len(datasets)} datasets")
    return datasets


def visualize_all_datasets():
    """Create visualizations for all synthetic datasets."""
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    generator = SyntheticAnomalyData()
    
    # Visualize 2D datasets
    datasets_2d = {
        'Gaussian Clusters with Outliers': generator.gaussian_clusters_with_outliers(),
        'Mixed Distributions': generator.mixed_distributions(),
        'Moons with Outliers': generator.moons_with_outliers()
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (title, (X, y)) in enumerate(datasets_2d.items()):
        ax = axes[i]
        
        # Plot normal points
        normal_mask = y == 0
        ax.scatter(X[normal_mask, 0], X[normal_mask, 1], 
                  c='blue', alpha=0.6, s=50, label='Normal')
        
        # Plot anomalies
        anomaly_mask = y == 1
        ax.scatter(X[anomaly_mask, 0], X[anomaly_mask, 1], 
                  c='red', alpha=0.8, s=100, marker='x', 
                  linewidths=2, label='Anomaly')
        
        ax.set_title(title)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/synthetic_2d_datasets.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualize time series
    t, y, anomalies = generator.time_series_with_spikes(length=500)
    plt.figure(figsize=(12, 6))
    plt.plot(t, y, 'b-', alpha=0.7, linewidth=1, label='Time Series')
    
    anomaly_indices = np.where(anomalies == 1)[0]
    if len(anomaly_indices) > 0:
        plt.scatter(t[anomaly_indices], y[anomaly_indices], 
                   c='red', s=100, marker='o', alpha=0.8,
                   label='Anomalies', zorder=5)
    
    plt.title('Time Series with Anomalous Spikes')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/synthetic_time_series.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Generate and visualize all datasets
    datasets = generate_all_datasets()
    visualize_all_datasets()
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print("-" * 50)
    
    for name, data in datasets.items():
        if name == 'time_series':
            t, y, anomalies = data
            n_total = len(y)
            n_anomalies = int(np.sum(anomalies))
        else:
            X, y = data
            n_total = len(y)
            n_anomalies = int(np.sum(y))
        
        anomaly_rate = n_anomalies / n_total * 100
        print(f"{name:20}: {n_total:4d} points, {n_anomalies:3d} anomalies ({anomaly_rate:.1f}%)") 