"""
Statistical Anomaly Detection Methods

This module implements statistical methods for anomaly detection from scratch,
including Z-score and Interquartile Range (IQR) methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, List
import os
from synthetic_anomaly_data import SyntheticAnomalyData
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


class StatisticalAnomalyDetector:
    """Statistical methods for anomaly detection."""
    
    def __init__(self):
        """Initialize the detector."""
        self.fitted_ = False
        self.mean_ = None
        self.std_ = None
        self.q1_ = None
        self.q3_ = None
        self.iqr_ = None
        self.median_ = None
    
    def z_score_detection(self, 
                         X: np.ndarray, 
                         threshold: float = 3.0,
                         axis: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using Z-score method.
        
        Args:
            X: Input data (1D or 2D array)
            threshold: Z-score threshold for anomaly detection
            axis: Axis along which to compute statistics (None for all data)
            
        Returns:
            z_scores: Z-scores for each data point
            anomalies: Boolean array indicating anomalies
        """
        # Calculate mean and standard deviation
        if axis is None:
            mean = np.mean(X)
            std = np.std(X, ddof=1)  # Use sample standard deviation
        else:
            mean = np.mean(X, axis=axis, keepdims=True)
            std = np.std(X, axis=axis, ddof=1, keepdims=True)
        
        # Store parameters for later use
        self.mean_ = mean
        self.std_ = std
        
        # Calculate Z-scores
        z_scores = np.abs((X - mean) / std)
        
        # Detect anomalies
        if X.ndim == 1:
            anomalies = z_scores > threshold
        else:
            # For 2D data, consider a point anomalous if any feature exceeds threshold
            anomalies = np.any(z_scores > threshold, axis=1)
        
        self.fitted_ = True
        return z_scores, anomalies
    
    def iqr_detection(self, 
                     X: np.ndarray, 
                     multiplier: float = 1.5,
                     axis: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect anomalies using Interquartile Range (IQR) method.
        
        Args:
            X: Input data (1D or 2D array)
            multiplier: IQR multiplier for outlier detection (typically 1.5)
            axis: Axis along which to compute statistics (None for all data)
            
        Returns:
            outlier_scores: Distance from normal range for each point
            anomalies: Boolean array indicating anomalies
            bounds: Tuple of (lower_bound, upper_bound)
        """
        if axis is None:
            q1 = np.percentile(X, 25)
            q3 = np.percentile(X, 75)
            median = np.median(X)
        else:
            q1 = np.percentile(X, 25, axis=axis, keepdims=True)
            q3 = np.percentile(X, 75, axis=axis, keepdims=True)
            median = np.median(X, axis=axis, keepdims=True)
        
        # Store parameters
        self.q1_ = q1
        self.q3_ = q3
        self.iqr_ = q3 - q1
        self.median_ = median
        
        # Calculate bounds
        lower_bound = q1 - multiplier * self.iqr_
        upper_bound = q3 + multiplier * self.iqr_
        
        # Calculate outlier scores (distance from normal range)
        outlier_scores = np.maximum(
            lower_bound - X,  # How far below lower bound
            X - upper_bound   # How far above upper bound
        )
        outlier_scores = np.maximum(outlier_scores, 0)  # Only positive distances
        
        # Detect anomalies
        if X.ndim == 1:
            anomalies = (X < lower_bound) | (X > upper_bound)
        else:
            # For 2D data, consider a point anomalous if any feature is outside bounds
            anomalies = np.any((X < lower_bound) | (X > upper_bound), axis=1)
        
        self.fitted_ = True
        return outlier_scores, anomalies, (lower_bound, upper_bound)
    
    def modified_z_score_detection(self, 
                                  X: np.ndarray, 
                                  threshold: float = 3.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using modified Z-score (based on median absolute deviation).
        More robust to outliers than standard Z-score.
        
        Args:
            X: Input data (1D array)
            threshold: Modified Z-score threshold
            
        Returns:
            modified_z_scores: Modified Z-scores for each data point
            anomalies: Boolean array indicating anomalies
        """
        # Calculate median and median absolute deviation (MAD)
        median = np.median(X, axis=0)
        mad = np.median(np.abs(X - median), axis=0)
        
        # Calculate modified Z-scores
        # 0.6745 is the 75th percentile of the standard normal distribution
        modified_z_scores = 0.6745 * (X - median) / mad
        modified_z_scores = np.abs(modified_z_scores)
        
        # Detect anomalies
        if X.ndim == 1:
            anomalies = modified_z_scores > threshold
        else:
            anomalies = np.any(modified_z_scores > threshold, axis=1)
        
        return modified_z_scores, anomalies


def visualize_1d_detection(data: np.ndarray, 
                          method_results: Dict[str, Tuple], 
                          true_anomalies: np.ndarray = None):
    """Visualize 1D anomaly detection results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original data
    ax = axes[0, 0]
    ax.scatter(range(len(data)), data, c='blue', alpha=0.6, s=30)
    if true_anomalies is not None:
        anomaly_indices = np.where(true_anomalies)[0]
        ax.scatter(anomaly_indices, data[anomaly_indices], 
                  c='red', s=100, marker='x', linewidths=2, label='True Anomalies')
        ax.legend()
    ax.set_title('Original Data')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)
    
    # Z-score method
    ax = axes[0, 1]
    z_scores, z_anomalies = method_results['z_score']
    ax.scatter(range(len(data)), z_scores, c='blue', alpha=0.6, s=30)
    ax.axhline(y=3.0, color='red', linestyle='--', label='Threshold (3.0)')
    anomaly_indices = np.where(z_anomalies)[0]
    if len(anomaly_indices) > 0:
        ax.scatter(anomaly_indices, z_scores[anomaly_indices], 
                  c='red', s=100, marker='o', alpha=0.8, label='Detected Anomalies')
    ax.set_title('Z-Score Method')
    ax.set_xlabel('Index')
    ax.set_ylabel('|Z-Score|')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # IQR method
    ax = axes[1, 0]
    outlier_scores, iqr_anomalies, bounds = method_results['iqr']
    ax.scatter(range(len(data)), data, c='blue', alpha=0.6, s=30)
    ax.axhline(y=bounds[0], color='red', linestyle='--', alpha=0.7, label='IQR Bounds')
    ax.axhline(y=bounds[1], color='red', linestyle='--', alpha=0.7)
    anomaly_indices = np.where(iqr_anomalies)[0]
    if len(anomaly_indices) > 0:
        ax.scatter(anomaly_indices, data[anomaly_indices], 
                  c='red', s=100, marker='o', alpha=0.8, label='Detected Anomalies')
    ax.set_title('IQR Method')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Modified Z-score method
    ax = axes[1, 1]
    mod_z_scores, mod_z_anomalies = method_results['modified_z_score']
    ax.scatter(range(len(data)), mod_z_scores, c='blue', alpha=0.6, s=30)
    ax.axhline(y=3.5, color='red', linestyle='--', label='Threshold (3.5)')
    anomaly_indices = np.where(mod_z_anomalies)[0]
    if len(anomaly_indices) > 0:
        ax.scatter(anomaly_indices, mod_z_scores[anomaly_indices], 
                  c='red', s=100, marker='o', alpha=0.8, label='Detected Anomalies')
    ax.set_title('Modified Z-Score Method')
    ax.set_xlabel('Index')
    ax.set_ylabel('Modified |Z-Score|')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()


def visualize_2d_detection(X: np.ndarray, 
                          method_results: Dict[str, Tuple], 
                          true_anomalies: np.ndarray = None):
    """Visualize 2D anomaly detection results."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    methods = ['z_score', 'iqr', 'modified_z_score']
    titles = ['Z-Score Method', 'IQR Method', 'Modified Z-Score Method']
    
    for i, (method, title) in enumerate(zip(methods, titles)):
        ax = axes[i]
        
        if method == 'iqr':
            _, anomalies, _ = method_results[method]
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
                      c='orange', alpha=0.8, s=150, marker='x', 
                      linewidths=3, label='True Anomalies')
        
        ax.set_title(title)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()


def evaluate_detection_methods(true_anomalies: np.ndarray, 
                             method_results: Dict[str, Tuple]) -> Dict[str, Dict[str, float]]:
    """Evaluate anomaly detection methods."""
    results = {}
    
    for method_name, method_result in method_results.items():
        if method_name == 'iqr':
            _, predicted_anomalies, _ = method_result
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
            if method_name == 'z_score' or method_name == 'modified_z_score':
                scores, _ = method_result
                if scores.ndim > 1:
                    scores = np.max(scores, axis=1)  # Take max score across features
                roc_auc = roc_auc_score(true_anomalies, scores)
            elif method_name == 'iqr':
                scores, _, _ = method_result
                if scores.ndim > 1:
                    scores = np.max(scores, axis=1)  # Take max score across features
                roc_auc = roc_auc_score(true_anomalies, scores)
            else:
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


def demonstrate_statistical_methods():
    """Demonstrate statistical anomaly detection methods."""
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Initialize data generator and detector
    data_generator = SyntheticAnomalyData()
    detector = StatisticalAnomalyDetector()
    
    print("Statistical Anomaly Detection Demonstration")
    print("=" * 50)
    
    # Test on 1D data (time series)
    print("\n1. Time Series Anomaly Detection")
    print("-" * 30)
    
    t, y, true_anomalies_1d = data_generator.time_series_with_spikes(length=200, 
                                                                   spike_probability=0.05)
    
    # Apply statistical methods
    method_results_1d = {}
    
    # Z-score method
    z_scores, z_anomalies = detector.z_score_detection(y, threshold=3.0)
    method_results_1d['z_score'] = (z_scores, z_anomalies)
    
    # IQR method
    outlier_scores, iqr_anomalies, bounds = detector.iqr_detection(y, multiplier=1.5)
    method_results_1d['iqr'] = (outlier_scores, iqr_anomalies, bounds)
    
    # Modified Z-score method
    mod_z_scores, mod_z_anomalies = detector.modified_z_score_detection(y, threshold=3.5)
    method_results_1d['modified_z_score'] = (mod_z_scores, mod_z_anomalies)
    
    # Visualize 1D results
    plt.figure(figsize=(15, 10))
    visualize_1d_detection(y, method_results_1d, true_anomalies_1d)
    plt.savefig('plots/statistical_methods_1d.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Evaluate 1D methods
    eval_results_1d = evaluate_detection_methods(true_anomalies_1d, method_results_1d)
    
    print("\n1D Anomaly Detection Results:")
    for method, metrics in eval_results_1d.items():
        print(f"{method:15}: Precision={metrics['precision']:.3f}, "
              f"Recall={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}, "
              f"ROC-AUC={metrics['roc_auc']:.3f}")
    
    # Test on 2D data
    print("\n2. 2D Anomaly Detection")
    print("-" * 25)
    
    X_2d, true_anomalies_2d = data_generator.gaussian_clusters_with_outliers(
        n_normal=200, n_outliers=20)
    
    # Apply statistical methods to 2D data
    method_results_2d = {}
    
    # Z-score method (per feature)
    z_scores_2d, z_anomalies_2d = detector.z_score_detection(X_2d, threshold=2.5, axis=0)
    method_results_2d['z_score'] = (z_scores_2d, z_anomalies_2d)
    
    # IQR method (per feature)
    outlier_scores_2d, iqr_anomalies_2d, bounds_2d = detector.iqr_detection(X_2d, multiplier=1.5, axis=0)
    method_results_2d['iqr'] = (outlier_scores_2d, iqr_anomalies_2d, bounds_2d)
    
    # Modified Z-score method (per feature)
    mod_z_scores_2d, mod_z_anomalies_2d = detector.modified_z_score_detection(X_2d, threshold=3.0)
    method_results_2d['modified_z_score'] = (mod_z_scores_2d, mod_z_anomalies_2d)
    
    # Visualize 2D results
    plt.figure(figsize=(18, 5))
    visualize_2d_detection(X_2d, method_results_2d, true_anomalies_2d)
    plt.savefig('plots/statistical_methods_2d.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Evaluate 2D methods
    eval_results_2d = evaluate_detection_methods(true_anomalies_2d, method_results_2d)
    
    print("\n2D Anomaly Detection Results:")
    for method, metrics in eval_results_2d.items():
        print(f"{method:15}: Precision={metrics['precision']:.3f}, "
              f"Recall={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}, "
              f"ROC-AUC={metrics['roc_auc']:.3f}")
    
    # Threshold sensitivity analysis
    print("\n3. Threshold Sensitivity Analysis")
    print("-" * 35)
    
    # Test different Z-score thresholds
    thresholds = np.arange(1.0, 4.5, 0.5)
    threshold_results = []
    
    for threshold in thresholds:
        _, z_anomalies_thresh = detector.z_score_detection(y, threshold=threshold)
        metrics = evaluate_detection_methods(true_anomalies_1d, 
                                           {'z_score': (z_scores, z_anomalies_thresh)})
        threshold_results.append({
            'threshold': threshold,
            'precision': metrics['z_score']['precision'],
            'recall': metrics['z_score']['recall'],
            'f1_score': metrics['z_score']['f1_score']
        })
    
    # Plot threshold sensitivity
    plt.figure(figsize=(12, 8))
    
    thresholds = [r['threshold'] for r in threshold_results]
    precisions = [r['precision'] for r in threshold_results]
    recalls = [r['recall'] for r in threshold_results]
    f1_scores = [r['f1_score'] for r in threshold_results]
    
    plt.subplot(2, 1, 1)
    plt.plot(thresholds, precisions, 'b-o', label='Precision', linewidth=2)
    plt.plot(thresholds, recalls, 'r-s', label='Recall', linewidth=2)
    plt.plot(thresholds, f1_scores, 'g-^', label='F1-Score', linewidth=2)
    plt.xlabel('Z-Score Threshold')
    plt.ylabel('Metric Value')
    plt.title('Z-Score Threshold Sensitivity Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Test different IQR multipliers
    multipliers = np.arange(0.5, 3.5, 0.5)
    multiplier_results = []
    
    for multiplier in multipliers:
        _, iqr_anomalies_mult, _ = detector.iqr_detection(y, multiplier=multiplier)
        metrics = evaluate_detection_methods(true_anomalies_1d, 
                                           {'iqr': (outlier_scores, iqr_anomalies_mult, bounds)})
        multiplier_results.append({
            'multiplier': multiplier,
            'precision': metrics['iqr']['precision'],
            'recall': metrics['iqr']['recall'],
            'f1_score': metrics['iqr']['f1_score']
        })
    
    plt.subplot(2, 1, 2)
    multipliers = [r['multiplier'] for r in multiplier_results]
    precisions = [r['precision'] for r in multiplier_results]
    recalls = [r['recall'] for r in multiplier_results]
    f1_scores = [r['f1_score'] for r in multiplier_results]
    
    plt.plot(multipliers, precisions, 'b-o', label='Precision', linewidth=2)
    plt.plot(multipliers, recalls, 'r-s', label='Recall', linewidth=2)
    plt.plot(multipliers, f1_scores, 'g-^', label='F1-Score', linewidth=2)
    plt.xlabel('IQR Multiplier')
    plt.ylabel('Metric Value')
    plt.title('IQR Multiplier Sensitivity Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/threshold_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    demonstrate_statistical_methods() 