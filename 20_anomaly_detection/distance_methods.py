"""
Distance-Based Anomaly Detection Methods

This module implements distance-based methods for anomaly detection,
including k-NN distance-based detection and DBSCAN clustering-based outlier detection.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from typing import Tuple, Dict, Any, List
import os
from synthetic_anomaly_data import SyntheticAnomalyData


class DistanceAnomalyDetector:
    """Distance-based anomaly detection methods."""
    
    def __init__(self):
        """Initialize the detector."""
        self.fitted_ = False
        self.nn_model_ = None
        self.dbscan_model_ = None
    
    def knn_distance_detection(self, 
                             X: np.ndarray, 
                             k: int = 5,
                             threshold_percentile: float = 95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using k-NN distance method.
        Points with high distance to their k-th nearest neighbor are considered anomalous.
        
        Args:
            X: Input data
            k: Number of nearest neighbors to consider
            threshold_percentile: Percentile threshold for anomaly detection
            
        Returns:
            distances: k-NN distances for each point
            anomalies: Boolean array indicating anomalies
        """
        # Fit k-NN model
        self.nn_model_ = NearestNeighbors(n_neighbors=k+1)  # +1 to exclude the point itself
        self.nn_model_.fit(X)
        
        # Find distances to k-th nearest neighbor
        distances, _ = self.nn_model_.kneighbors(X)
        knn_distances = distances[:, -1]  # Distance to k-th neighbor (excluding self)
        
        # Determine threshold based on percentile
        threshold = np.percentile(knn_distances, threshold_percentile)
        
        # Detect anomalies
        anomalies = knn_distances > threshold
        
        self.fitted_ = True
        return knn_distances, anomalies
    
    def knn_avg_distance_detection(self, 
                                  X: np.ndarray, 
                                  k: int = 5,
                                  threshold_percentile: float = 95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using average k-NN distance method.
        
        Args:
            X: Input data
            k: Number of nearest neighbors to consider
            threshold_percentile: Percentile threshold for anomaly detection
            
        Returns:
            avg_distances: Average k-NN distances for each point
            anomalies: Boolean array indicating anomalies
        """
        # Fit k-NN model
        self.nn_model_ = NearestNeighbors(n_neighbors=k+1)
        self.nn_model_.fit(X)
        
        # Find average distances to k nearest neighbors
        distances, _ = self.nn_model_.kneighbors(X)
        avg_distances = np.mean(distances[:, 1:], axis=1)  # Exclude self (first neighbor)
        
        # Determine threshold
        threshold = np.percentile(avg_distances, threshold_percentile)
        
        # Detect anomalies
        anomalies = avg_distances > threshold
        
        return avg_distances, anomalies
    
    def dbscan_outlier_detection(self, 
                                X: np.ndarray, 
                                eps: float = 0.5,
                                min_samples: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect outliers using DBSCAN clustering.
        Points not assigned to any cluster (noise) are considered anomalies.
        
        Args:
            X: Input data
            eps: Maximum distance between points in the same neighborhood
            min_samples: Minimum number of points required to form a cluster
            
        Returns:
            cluster_labels: Cluster labels (-1 for noise/outliers)
            anomalies: Boolean array indicating anomalies
        """
        # Fit DBSCAN model
        self.dbscan_model_ = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = self.dbscan_model_.fit_predict(X)
        
        # Points labeled as -1 are considered noise (anomalies)
        anomalies = cluster_labels == -1
        
        self.fitted_ = True
        return cluster_labels, anomalies
    
    def local_outlier_factor_scratch(self, 
                                   X: np.ndarray, 
                                   k: int = 5,
                                   threshold: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Implement Local Outlier Factor (LOF) from scratch.
        
        Args:
            X: Input data
            k: Number of nearest neighbors
            threshold: LOF threshold for anomaly detection
            
        Returns:
            lof_scores: LOF scores for each point
            anomalies: Boolean array indicating anomalies
        """
        n_samples = X.shape[0]
        
        # Fit k-NN model
        nn_model = NearestNeighbors(n_neighbors=k+1)
        nn_model.fit(X)
        
        # Find k-nearest neighbors and distances
        distances, neighbors = nn_model.kneighbors(X)
        
        # Calculate k-distance (distance to k-th nearest neighbor)
        k_distances = distances[:, -1]
        
        # Calculate reachability distance
        reachability_distances = np.zeros((n_samples, k))
        for i in range(n_samples):
            for j in range(k):
                neighbor_idx = neighbors[i, j+1]  # Skip self
                reachability_distances[i, j] = max(distances[i, j+1], 
                                                 k_distances[neighbor_idx])
        
        # Calculate local reachability density (LRD)
        lrd = np.zeros(n_samples)
        for i in range(n_samples):
            avg_reachability = np.mean(reachability_distances[i])
            lrd[i] = 1.0 / (avg_reachability + 1e-10)  # Add small epsilon to avoid division by zero
        
        # Calculate Local Outlier Factor (LOF)
        lof_scores = np.zeros(n_samples)
        for i in range(n_samples):
            neighbor_lrds = []
            for j in range(k):
                neighbor_idx = neighbors[i, j+1]  # Skip self
                neighbor_lrds.append(lrd[neighbor_idx])
            
            avg_neighbor_lrd = np.mean(neighbor_lrds)
            lof_scores[i] = avg_neighbor_lrd / (lrd[i] + 1e-10)
        
        # Detect anomalies
        anomalies = lof_scores > threshold
        
        return lof_scores, anomalies


def visualize_distance_methods_2d(X: np.ndarray, 
                                 method_results: Dict[str, Tuple],
                                 true_anomalies: np.ndarray = None):
    """Visualize 2D distance-based anomaly detection results."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    methods = ['knn_distance', 'knn_avg_distance', 'dbscan', 'lof']
    titles = ['k-NN Distance', 'k-NN Average Distance', 'DBSCAN Outliers', 'Local Outlier Factor']
    
    for i, (method, title) in enumerate(zip(methods, titles)):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        if method == 'dbscan':
            cluster_labels, anomalies = method_results[method]
            
            # Plot clusters with different colors
            unique_labels = np.unique(cluster_labels)
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(unique_labels, colors):
                if label == -1:
                    # Outliers in red
                    mask = cluster_labels == label
                    ax.scatter(X[mask, 0], X[mask, 1], 
                             c='red', alpha=0.8, s=100, marker='x',
                             linewidths=2, label='Outliers')
                else:
                    # Regular clusters
                    mask = cluster_labels == label
                    ax.scatter(X[mask, 0], X[mask, 1], 
                             c=[color], alpha=0.6, s=50, 
                             label=f'Cluster {label}')
        else:
            _, anomalies = method_results[method]
            
            # Plot normal points
            normal_mask = ~anomalies
            ax.scatter(X[normal_mask, 0], X[normal_mask, 1], 
                      c='blue', alpha=0.6, s=50, label='Normal')
            
            # Plot detected anomalies
            if np.any(anomalies):
                ax.scatter(X[anomalies, 0], X[anomalies, 1], 
                          c='red', alpha=0.8, s=100, marker='o', 
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
    
    plt.tight_layout()


def visualize_decision_regions_2d(X: np.ndarray, 
                                 detector: DistanceAnomalyDetector,
                                 method: str,
                                 resolution: int = 100):
    """Visualize decision regions for 2D distance-based methods."""
    # Create a mesh of points
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                        np.linspace(y_min, y_max, resolution))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Apply the detection method to mesh points
    if method == 'knn_distance':
        scores, predictions = detector.knn_distance_detection(mesh_points, k=5)
    elif method == 'knn_avg_distance':
        scores, predictions = detector.knn_avg_distance_detection(mesh_points, k=5)
    elif method == 'lof':
        scores, predictions = detector.local_outlier_factor_scratch(mesh_points, k=5)
    else:
        return  # DBSCAN doesn't work well for this visualization
    
    # Reshape predictions for plotting
    predictions = predictions.reshape(xx.shape)
    scores = scores.reshape(xx.shape)
    
    plt.figure(figsize=(12, 5))
    
    # Plot decision regions
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, predictions.astype(int), alpha=0.3, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c='black', alpha=0.7, s=50)
    plt.title(f'{method.replace("_", " ").title()} - Decision Regions')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Anomaly Prediction')
    
    # Plot anomaly scores
    plt.subplot(1, 2, 2)
    contour = plt.contourf(xx, yy, scores, alpha=0.7, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c='red', alpha=0.7, s=50)
    plt.title(f'{method.replace("_", " ").title()} - Anomaly Scores')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(contour, label='Anomaly Score')
    
    plt.tight_layout()


def evaluate_distance_methods(true_anomalies: np.ndarray, 
                            method_results: Dict[str, Tuple]) -> Dict[str, Dict[str, float]]:
    """Evaluate distance-based anomaly detection methods."""
    results = {}
    
    for method_name, method_result in method_results.items():
        if method_name == 'dbscan':
            _, predicted_anomalies = method_result
        else:
            _, predicted_anomalies = method_result
        
        # Convert to binary if needed
        if predicted_anomalies.dtype == bool:
            predicted_binary = predicted_anomalies.astype(int)
        else:
            predicted_binary = predicted_anomalies
        
        # Calculate metrics
        precision = precision_score(true_anomalies, predicted_binary, zero_division=0)
        recall = recall_score(true_anomalies, predicted_binary, zero_division=0)
        f1 = f1_score(true_anomalies, predicted_binary, zero_division=0)
        
        # For ROC AUC, we need scores rather than binary predictions
        try:
            if method_name != 'dbscan':
                scores, _ = method_result
                if scores.ndim > 1:
                    scores = np.max(scores, axis=1)
                roc_auc = roc_auc_score(true_anomalies, scores)
            else:
                # For DBSCAN, we can't compute ROC AUC easily
                roc_auc = 0.0
        except:
            roc_auc = 0.0
        
        results[method_name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
    
    return results


def parameter_sensitivity_analysis(X: np.ndarray, 
                                 true_anomalies: np.ndarray,
                                 detector: DistanceAnomalyDetector):
    """Analyze parameter sensitivity for distance-based methods."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # k-NN parameter sensitivity
    k_values = range(3, 21, 2)
    knn_results = []
    
    for k in k_values:
        _, anomalies = detector.knn_distance_detection(X, k=k, threshold_percentile=95)
        metrics = evaluate_distance_methods(true_anomalies, {'knn': ([], anomalies)})
        knn_results.append({
            'k': k,
            'precision': metrics['knn']['precision'],
            'recall': metrics['knn']['recall'],
            'f1_score': metrics['knn']['f1_score']
        })
    
    ax = axes[0, 0]
    k_vals = [r['k'] for r in knn_results]
    precisions = [r['precision'] for r in knn_results]
    recalls = [r['recall'] for r in knn_results]
    f1_scores = [r['f1_score'] for r in knn_results]
    
    ax.plot(k_vals, precisions, 'b-o', label='Precision', linewidth=2)
    ax.plot(k_vals, recalls, 'r-s', label='Recall', linewidth=2)
    ax.plot(k_vals, f1_scores, 'g-^', label='F1-Score', linewidth=2)
    ax.set_xlabel('k (Number of Neighbors)')
    ax.set_ylabel('Metric Value')
    ax.set_title('k-NN Parameter Sensitivity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Threshold percentile sensitivity
    percentiles = np.arange(80, 99, 2)
    percentile_results = []
    
    for perc in percentiles:
        _, anomalies = detector.knn_distance_detection(X, k=5, threshold_percentile=perc)
        metrics = evaluate_distance_methods(true_anomalies, {'knn': ([], anomalies)})
        percentile_results.append({
            'percentile': perc,
            'precision': metrics['knn']['precision'],
            'recall': metrics['knn']['recall'],
            'f1_score': metrics['knn']['f1_score']
        })
    
    ax = axes[0, 1]
    perc_vals = [r['percentile'] for r in percentile_results]
    precisions = [r['precision'] for r in percentile_results]
    recalls = [r['recall'] for r in percentile_results]
    f1_scores = [r['f1_score'] for r in percentile_results]
    
    ax.plot(perc_vals, precisions, 'b-o', label='Precision', linewidth=2)
    ax.plot(perc_vals, recalls, 'r-s', label='Recall', linewidth=2)
    ax.plot(perc_vals, f1_scores, 'g-^', label='F1-Score', linewidth=2)
    ax.set_xlabel('Threshold Percentile')
    ax.set_ylabel('Metric Value')
    ax.set_title('Threshold Percentile Sensitivity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # DBSCAN eps sensitivity
    eps_values = np.arange(0.3, 2.0, 0.2)
    eps_results = []
    
    for eps in eps_values:
        _, anomalies = detector.dbscan_outlier_detection(X, eps=eps, min_samples=5)
        metrics = evaluate_distance_methods(true_anomalies, {'dbscan': ([], anomalies)})
        eps_results.append({
            'eps': eps,
            'precision': metrics['dbscan']['precision'],
            'recall': metrics['dbscan']['recall'],
            'f1_score': metrics['dbscan']['f1_score']
        })
    
    ax = axes[1, 0]
    eps_vals = [r['eps'] for r in eps_results]
    precisions = [r['precision'] for r in eps_results]
    recalls = [r['recall'] for r in eps_results]
    f1_scores = [r['f1_score'] for r in eps_results]
    
    ax.plot(eps_vals, precisions, 'b-o', label='Precision', linewidth=2)
    ax.plot(eps_vals, recalls, 'r-s', label='Recall', linewidth=2)
    ax.plot(eps_vals, f1_scores, 'g-^', label='F1-Score', linewidth=2)
    ax.set_xlabel('eps (Neighborhood Distance)')
    ax.set_ylabel('Metric Value')
    ax.set_title('DBSCAN eps Sensitivity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # DBSCAN min_samples sensitivity
    min_samples_values = range(3, 11)
    min_samples_results = []
    
    for min_samples in min_samples_values:
        _, anomalies = detector.dbscan_outlier_detection(X, eps=0.8, min_samples=min_samples)
        metrics = evaluate_distance_methods(true_anomalies, {'dbscan': ([], anomalies)})
        min_samples_results.append({
            'min_samples': min_samples,
            'precision': metrics['dbscan']['precision'],
            'recall': metrics['dbscan']['recall'],
            'f1_score': metrics['dbscan']['f1_score']
        })
    
    ax = axes[1, 1]
    min_samples_vals = [r['min_samples'] for r in min_samples_results]
    precisions = [r['precision'] for r in min_samples_results]
    recalls = [r['recall'] for r in min_samples_results]
    f1_scores = [r['f1_score'] for r in min_samples_results]
    
    ax.plot(min_samples_vals, precisions, 'b-o', label='Precision', linewidth=2)
    ax.plot(min_samples_vals, recalls, 'r-s', label='Recall', linewidth=2)
    ax.plot(min_samples_vals, f1_scores, 'g-^', label='F1-Score', linewidth=2)
    ax.set_xlabel('min_samples')
    ax.set_ylabel('Metric Value')
    ax.set_title('DBSCAN min_samples Sensitivity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()


def demonstrate_distance_methods():
    """Demonstrate distance-based anomaly detection methods."""
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Initialize data generator and detector
    data_generator = SyntheticAnomalyData()
    detector = DistanceAnomalyDetector()
    
    print("Distance-Based Anomaly Detection Demonstration")
    print("=" * 55)
    
    # Test on different 2D datasets
    datasets = {
        'gaussian_clusters': data_generator.gaussian_clusters_with_outliers(n_normal=200, n_outliers=20),
        'mixed_distributions': data_generator.mixed_distributions(n_normal=200, n_outliers=20),
        'moons': data_generator.moons_with_outliers(n_normal=200, n_outliers=20)
    }
    
    for dataset_name, (X, true_anomalies) in datasets.items():
        print(f"\n{dataset_name.replace('_', ' ').title()} Dataset")
        print("-" * 40)
        
        # Apply distance-based methods
        method_results = {}
        
        # k-NN distance method
        knn_distances, knn_anomalies = detector.knn_distance_detection(X, k=5, threshold_percentile=90)
        method_results['knn_distance'] = (knn_distances, knn_anomalies)
        
        # k-NN average distance method
        knn_avg_distances, knn_avg_anomalies = detector.knn_avg_distance_detection(X, k=5, threshold_percentile=90)
        method_results['knn_avg_distance'] = (knn_avg_distances, knn_avg_anomalies)
        
        # DBSCAN outlier detection
        cluster_labels, dbscan_anomalies = detector.dbscan_outlier_detection(X, eps=0.8, min_samples=5)
        method_results['dbscan'] = (cluster_labels, dbscan_anomalies)
        
        # LOF from scratch
        lof_scores, lof_anomalies = detector.local_outlier_factor_scratch(X, k=5, threshold=1.5)
        method_results['lof'] = (lof_scores, lof_anomalies)
        
        # Visualize results
        plt.figure(figsize=(16, 12))
        visualize_distance_methods_2d(X, method_results, true_anomalies)
        plt.suptitle(f'Distance-Based Anomaly Detection - {dataset_name.replace("_", " ").title()}', 
                    fontsize=16)
        plt.savefig(f'plots/distance_methods_{dataset_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Evaluate methods
        eval_results = evaluate_distance_methods(true_anomalies, method_results)
        
        print(f"\n{dataset_name.replace('_', ' ').title()} Results:")
        for method, metrics in eval_results.items():
            print(f"{method:15}: Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}, "
                  f"ROC-AUC={metrics['roc_auc']:.3f}")
    
    # Decision region visualization for Gaussian clusters
    print("\nDecision Region Visualization")
    print("-" * 30)
    
    X_demo, _ = data_generator.gaussian_clusters_with_outliers(n_normal=100, n_outliers=10)
    
    methods_for_regions = ['knn_distance', 'knn_avg_distance', 'lof']
    for method in methods_for_regions:
        visualize_decision_regions_2d(X_demo, detector, method)
        plt.suptitle(f'Decision Regions - {method.replace("_", " ").title()}', fontsize=14)
        plt.savefig(f'plots/decision_regions_{method}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Parameter sensitivity analysis
    print("\nParameter Sensitivity Analysis")
    print("-" * 32)
    
    X_sens, true_anomalies_sens = data_generator.gaussian_clusters_with_outliers(n_normal=150, n_outliers=15)
    
    parameter_sensitivity_analysis(X_sens, true_anomalies_sens, detector)
    plt.suptitle('Parameter Sensitivity Analysis', fontsize=16)
    plt.savefig('plots/parameter_sensitivity_distance_methods.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    demonstrate_distance_methods() 