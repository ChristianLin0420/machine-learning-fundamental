"""
Model-Based Anomaly Detection Methods

This module implements model-based methods for anomaly detection,
including Isolation Forest, One-Class SVM, and Autoencoder-based detection.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from typing import Tuple, Dict, Any, List
import os
from synthetic_anomaly_data import SyntheticAnomalyData


class SimpleAutoencoder(nn.Module):
    """Simple MLP Autoencoder for anomaly detection."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [8, 4, 2]):
        """
        Initialize autoencoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions (encoding path)
        """
        super(SimpleAutoencoder, self).__init__()
        
        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder layers (reverse of encoder)
        decoder_layers = []
        hidden_dims_reversed = list(reversed(hidden_dims[:-1])) + [input_dim]
        prev_dim = hidden_dims[-1]  # bottleneck dimension
        
        for i, hidden_dim in enumerate(hidden_dims_reversed):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            if i < len(hidden_dims_reversed) - 1:  # No activation on output layer
                decoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        """Forward pass through autoencoder."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """Encode input to latent representation."""
        return self.encoder(x)


class ModelBasedAnomalyDetector:
    """Model-based anomaly detection methods."""
    
    def __init__(self, random_state: int = 42):
        """Initialize the detector."""
        self.random_state = random_state
        self.fitted_ = False
        self.isolation_forest_ = None
        self.one_class_svm_ = None
        self.autoencoder_ = None
        self.scaler_ = None
        
        # Set random seeds
        np.random.seed(random_state)
        torch.manual_seed(random_state)
    
    def isolation_forest_detection(self, 
                                  X: np.ndarray,
                                  contamination: float = 0.1,
                                  n_estimators: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using Isolation Forest.
        
        Args:
            X: Input data
            contamination: Expected proportion of anomalies
            n_estimators: Number of isolation trees
            
        Returns:
            anomaly_scores: Anomaly scores (negative values indicate anomalies)
            anomalies: Boolean array indicating anomalies
        """
        # Initialize and fit Isolation Forest
        self.isolation_forest_ = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=self.random_state
        )
        
        # Fit and predict
        anomaly_labels = self.isolation_forest_.fit_predict(X)
        anomaly_scores = self.isolation_forest_.score_samples(X)
        
        # Convert labels to boolean (1: normal, -1: anomaly)
        anomalies = anomaly_labels == -1
        
        self.fitted_ = True
        return anomaly_scores, anomalies
    
    def one_class_svm_detection(self, 
                               X: np.ndarray,
                               kernel: str = 'rbf',
                               gamma: str = 'scale',
                               nu: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using One-Class SVM.
        
        Args:
            X: Input data
            kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            gamma: Kernel coefficient
            nu: Upper bound on fraction of training errors and lower bound of support vectors
            
        Returns:
            anomaly_scores: Distance to separating hyperplane
            anomalies: Boolean array indicating anomalies
        """
        # Scale the data for SVM
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # Initialize and fit One-Class SVM
        self.one_class_svm_ = OneClassSVM(
            kernel=kernel,
            gamma=gamma,
            nu=nu
        )
        
        # Fit and predict
        anomaly_labels = self.one_class_svm_.fit_predict(X_scaled)
        anomaly_scores = self.one_class_svm_.score_samples(X_scaled)
        
        # Convert labels to boolean (1: normal, -1: anomaly)
        anomalies = anomaly_labels == -1
        
        return anomaly_scores, anomalies
    
    def autoencoder_detection(self, 
                            X: np.ndarray,
                            hidden_dims: List[int] = [8, 4, 2],
                            epochs: int = 100,
                            batch_size: int = 32,
                            learning_rate: float = 0.001,
                            threshold_percentile: float = 95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using Autoencoder reconstruction error.
        
        Args:
            X: Input data
            hidden_dims: Hidden layer dimensions for autoencoder
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            threshold_percentile: Percentile threshold for anomaly detection
            
        Returns:
            reconstruction_errors: Reconstruction errors for each point
            anomalies: Boolean array indicating anomalies
        """
        # Scale the data
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize autoencoder
        input_dim = X.shape[1]
        self.autoencoder_ = SimpleAutoencoder(input_dim, hidden_dims)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.autoencoder_.parameters(), lr=learning_rate)
        
        # Training loop
        self.autoencoder_.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_data, _ in dataloader:
                optimizer.zero_grad()
                reconstructed = self.autoencoder_(batch_data)
                loss = criterion(reconstructed, batch_data)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')
        
        # Calculate reconstruction errors
        self.autoencoder_.eval()
        with torch.no_grad():
            reconstructed = self.autoencoder_(X_tensor)
            reconstruction_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()
        
        # Determine threshold and detect anomalies
        threshold = np.percentile(reconstruction_errors, threshold_percentile)
        anomalies = reconstruction_errors > threshold
        
        return reconstruction_errors, anomalies
    
    def ensemble_detection(self, 
                          X: np.ndarray,
                          methods: List[str] = ['isolation_forest', 'one_class_svm', 'autoencoder'],
                          weights: List[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ensemble of multiple anomaly detection methods.
        
        Args:
            X: Input data
            methods: List of methods to ensemble
            weights: Weights for each method (if None, equal weights)
            
        Returns:
            ensemble_scores: Combined anomaly scores
            anomalies: Boolean array indicating anomalies
        """
        if weights is None:
            weights = [1.0] * len(methods)
        
        all_scores = []
        
        for method in methods:
            if method == 'isolation_forest':
                scores, _ = self.isolation_forest_detection(X)
                # Convert to positive scores (higher = more anomalous)
                scores = -scores
            elif method == 'one_class_svm':
                scores, _ = self.one_class_svm_detection(X)
                # Convert to positive scores
                scores = -scores
            elif method == 'autoencoder':
                scores, _ = self.autoencoder_detection(X, epochs=50)  # Fewer epochs for ensemble
            
            # Normalize scores to [0, 1]
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            all_scores.append(scores)
        
        # Weighted ensemble
        ensemble_scores = np.zeros_like(all_scores[0])
        for i, (scores, weight) in enumerate(zip(all_scores, weights)):
            ensemble_scores += weight * scores
        
        ensemble_scores /= sum(weights)
        
        # Determine threshold and detect anomalies
        threshold = np.percentile(ensemble_scores, 90)
        anomalies = ensemble_scores > threshold
        
        return ensemble_scores, anomalies


def visualize_model_results_2d(X: np.ndarray, 
                             method_results: Dict[str, Tuple],
                             true_anomalies: np.ndarray = None):
    """Visualize 2D model-based anomaly detection results."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    methods = ['isolation_forest', 'one_class_svm', 'autoencoder', 'ensemble']
    titles = ['Isolation Forest', 'One-Class SVM', 'Autoencoder', 'Ensemble Method']
    
    for i, (method, title) in enumerate(zip(methods, titles)):
        if method not in method_results:
            continue
            
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        scores, anomalies = method_results[method]
        
        # Plot normal points
        normal_mask = ~anomalies
        scatter_normal = ax.scatter(X[normal_mask, 0], X[normal_mask, 1], 
                                   c=scores[normal_mask], cmap='viridis',
                                   alpha=0.6, s=50, label='Normal')
        
        # Plot detected anomalies
        if np.any(anomalies):
            ax.scatter(X[anomalies, 0], X[anomalies, 1], 
                      c='red', alpha=0.8, s=100, marker='o', 
                      edgecolors='black', linewidths=1,
                      label='Detected Anomalies')
        
        # Plot true anomalies for comparison if available
        if true_anomalies is not None:
            true_anomaly_mask = true_anomalies.astype(bool)
            ax.scatter(X[true_anomaly_mask, 0], X[true_anomaly_mask, 1], 
                      c='orange', alpha=0.8, s=150, marker='+', 
                      linewidths=4, label='True Anomalies')
        
        ax.set_title(title)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add colorbar for scores
        plt.colorbar(scatter_normal, ax=ax, label='Anomaly Score')
    
    plt.tight_layout()


def compare_model_performance(X: np.ndarray, 
                            true_anomalies: np.ndarray,
                            detector: ModelBasedAnomalyDetector):
    """Compare performance of different model-based methods."""
    methods = {
        'Isolation Forest': detector.isolation_forest_detection,
        'One-Class SVM (RBF)': lambda x: detector.one_class_svm_detection(x, kernel='rbf'),
        'One-Class SVM (Linear)': lambda x: detector.one_class_svm_detection(x, kernel='linear'),
        'Autoencoder': lambda x: detector.autoencoder_detection(x, epochs=50)
    }
    
    results = {}
    
    for method_name, method_func in methods.items():
        print(f"Running {method_name}...")
        scores, anomalies = method_func(X)
        
        # Calculate metrics
        precision = precision_score(true_anomalies, anomalies.astype(int), zero_division=0)
        recall = recall_score(true_anomalies, anomalies.astype(int), zero_division=0)
        f1 = f1_score(true_anomalies, anomalies.astype(int), zero_division=0)
        
        try:
            # For ROC AUC, handle negative scores
            if method_name.startswith('Isolation Forest') or method_name.startswith('One-Class SVM'):
                roc_scores = -scores  # Convert to positive scores
            else:
                roc_scores = scores
            roc_auc = roc_auc_score(true_anomalies, roc_scores)
        except:
            roc_auc = 0.0
        
        results[method_name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'scores': scores,
            'anomalies': anomalies
        }
    
    return results


def plot_roc_curves(true_anomalies: np.ndarray, 
                   performance_results: Dict[str, Dict]):
    """Plot ROC curves for model-based methods."""
    from sklearn.metrics import roc_curve
    
    plt.figure(figsize=(10, 8))
    
    for method_name, results in performance_results.items():
        scores = results['scores']
        
        # Handle negative scores
        if method_name.startswith('Isolation Forest') or method_name.startswith('One-Class SVM'):
            scores = -scores
        
        try:
            fpr, tpr, _ = roc_curve(true_anomalies, scores)
            auc = results['roc_auc']
            plt.plot(fpr, tpr, linewidth=2, label=f'{method_name} (AUC = {auc:.3f})')
        except:
            continue
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Model-Based Anomaly Detection')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)


def plot_precision_at_k(true_anomalies: np.ndarray, 
                       performance_results: Dict[str, Dict]):
    """Plot Precision@k curves."""
    plt.figure(figsize=(12, 8))
    
    # Calculate Precision@k for different k values
    n_samples = len(true_anomalies)
    k_values = range(1, min(51, n_samples + 1))
    
    for method_name, results in performance_results.items():
        scores = results['scores']
        
        # Handle negative scores (higher absolute value = more anomalous)
        if method_name.startswith('Isolation Forest') or method_name.startswith('One-Class SVM'):
            scores = -scores
        
        precisions_at_k = []
        
        for k in k_values:
            # Get top k anomalous points
            top_k_indices = np.argsort(scores)[-k:]
            
            # Calculate precision@k
            true_positives = np.sum(true_anomalies[top_k_indices])
            precision_at_k = true_positives / k if k > 0 else 0
            precisions_at_k.append(precision_at_k)
        
        plt.plot(k_values, precisions_at_k, linewidth=2, 
                marker='o', markersize=4, label=method_name)
    
    plt.xlabel('k (Top k predictions)')
    plt.ylabel('Precision@k')
    plt.title('Precision@k Analysis - Model-Based Methods')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([1, max(k_values)])
    plt.ylim([0, 1.05])


def demonstrate_model_methods():
    """Demonstrate model-based anomaly detection methods."""
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Initialize data generator and detector
    data_generator = SyntheticAnomalyData()
    detector = ModelBasedAnomalyDetector()
    
    print("Model-Based Anomaly Detection Demonstration")
    print("=" * 50)
    
    # Test on different 2D datasets
    datasets = {
        'gaussian_clusters': data_generator.gaussian_clusters_with_outliers(n_normal=200, n_outliers=20),
        'mixed_distributions': data_generator.mixed_distributions(n_normal=200, n_outliers=20)
    }
    
    for dataset_name, (X, true_anomalies) in datasets.items():
        print(f"\n{dataset_name.replace('_', ' ').title()} Dataset")
        print("-" * 40)
        
        # Apply model-based methods
        method_results = {}
        
        print("Running Isolation Forest...")
        iso_scores, iso_anomalies = detector.isolation_forest_detection(X, contamination=0.1)
        method_results['isolation_forest'] = (iso_scores, iso_anomalies)
        
        print("Running One-Class SVM...")
        svm_scores, svm_anomalies = detector.one_class_svm_detection(X, kernel='rbf', nu=0.1)
        method_results['one_class_svm'] = (svm_scores, svm_anomalies)
        
        print("Running Autoencoder...")
        ae_scores, ae_anomalies = detector.autoencoder_detection(X, epochs=100)
        method_results['autoencoder'] = (ae_scores, ae_anomalies)
        
        print("Running Ensemble...")
        ensemble_scores, ensemble_anomalies = detector.ensemble_detection(X)
        method_results['ensemble'] = (ensemble_scores, ensemble_anomalies)
        
        # Visualize results
        plt.figure(figsize=(16, 12))
        visualize_model_results_2d(X, method_results, true_anomalies)
        plt.suptitle(f'Model-Based Anomaly Detection - {dataset_name.replace("_", " ").title()}', 
                    fontsize=16)
        plt.savefig(f'plots/model_methods_{dataset_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Performance comparison
        print(f"\nPerformance Comparison - {dataset_name.replace('_', ' ').title()}")
        print("-" * 50)
        
        performance_results = compare_model_performance(X, true_anomalies, detector)
        
        for method, metrics in performance_results.items():
            print(f"{method:20}: Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}, "
                  f"ROC-AUC={metrics['roc_auc']:.3f}")
        
        # Plot ROC curves
        plot_roc_curves(true_anomalies, performance_results)
        plt.savefig(f'plots/roc_curves_{dataset_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot Precision@k
        plot_precision_at_k(true_anomalies, performance_results)
        plt.savefig(f'plots/precision_at_k_{dataset_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Autoencoder visualization on a simple dataset
    print("\nAutoencoder Latent Space Visualization")
    print("-" * 38)
    
    X_simple, y_simple = data_generator.gaussian_clusters_with_outliers(n_normal=150, n_outliers=15)
    
    # Train autoencoder with 2D latent space for visualization
    ae_scores, ae_anomalies = detector.autoencoder_detection(X_simple, hidden_dims=[4, 2], epochs=150)
    
    # Get latent representations
    X_scaled = detector.scaler_.transform(X_simple)
    X_tensor = torch.FloatTensor(X_scaled)
    
    detector.autoencoder_.eval()
    with torch.no_grad():
        latent = detector.autoencoder_.encode(X_tensor).numpy()
    
    # Plot latent space
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.scatter(X_simple[:, 0], X_simple[:, 1], c=y_simple, cmap='RdYlBu', alpha=0.7)
    plt.title('Original Data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='True Label')
    
    plt.subplot(1, 3, 2)
    plt.scatter(latent[:, 0], latent[:, 1], c=y_simple, cmap='RdYlBu', alpha=0.7)
    plt.title('Latent Space (2D)')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.colorbar(label='True Label')
    
    plt.subplot(1, 3, 3)
    plt.scatter(X_simple[:, 0], X_simple[:, 1], c=ae_scores, cmap='viridis', alpha=0.7)
    plt.title('Reconstruction Error')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Reconstruction Error')
    
    plt.tight_layout()
    plt.savefig('plots/autoencoder_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    demonstrate_model_methods() 