"""
Comprehensive Anomaly Detection Visualization

This module provides unified visualization and evaluation for all
anomaly detection methods implemented in this package.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                           roc_auc_score, roc_curve, precision_recall_curve)
from typing import Dict, List, Tuple, Any
import os
import pandas as pd

# Import our custom modules
from synthetic_anomaly_data import SyntheticAnomalyData
from stat_methods import StatisticalAnomalyDetector
from distance_methods import DistanceAnomalyDetector
from model_based_methods import ModelBasedAnomalyDetector


class AnomalyVisualizationSuite:
    """Comprehensive visualization suite for anomaly detection methods."""
    
    def __init__(self):
        """Initialize the visualization suite."""
        self.data_generator = SyntheticAnomalyData()
        self.stat_detector = StatisticalAnomalyDetector()
        self.distance_detector = DistanceAnomalyDetector()
        self.model_detector = ModelBasedAnomalyDetector()
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def run_all_methods(self, X: np.ndarray, 
                       include_autoencoder: bool = True) -> Dict[str, Tuple]:
        """
        Run all anomaly detection methods on the given data.
        
        Args:
            X: Input data
            include_autoencoder: Whether to include autoencoder (slower)
            
        Returns:
            Dictionary with method results
        """
        results = {}
        
        print("Running statistical methods...")
        # Statistical methods
        z_scores, z_anomalies = self.stat_detector.z_score_detection(X, threshold=2.5)
        results['z_score'] = (z_scores, z_anomalies)
        
        outlier_scores, iqr_anomalies, bounds = self.stat_detector.iqr_detection(X, multiplier=1.5)
        results['iqr'] = (outlier_scores, iqr_anomalies)
        
        print("Running distance-based methods...")
        # Distance-based methods
        knn_distances, knn_anomalies = self.distance_detector.knn_distance_detection(X, k=5)
        results['knn_distance'] = (knn_distances, knn_anomalies)
        
        cluster_labels, dbscan_anomalies = self.distance_detector.dbscan_outlier_detection(X, eps=0.8)
        results['dbscan'] = (cluster_labels, dbscan_anomalies)
        
        lof_scores, lof_anomalies = self.distance_detector.local_outlier_factor_scratch(X, k=5)
        results['lof'] = (lof_scores, lof_anomalies)
        
        print("Running model-based methods...")
        # Model-based methods
        iso_scores, iso_anomalies = self.model_detector.isolation_forest_detection(X)
        results['isolation_forest'] = (iso_scores, iso_anomalies)
        
        svm_scores, svm_anomalies = self.model_detector.one_class_svm_detection(X)
        results['one_class_svm'] = (svm_scores, svm_anomalies)
        
        if include_autoencoder:
            print("Running autoencoder (this may take a moment)...")
            ae_scores, ae_anomalies = self.model_detector.autoencoder_detection(X, epochs=50)
            results['autoencoder'] = (ae_scores, ae_anomalies)
        
        return results
    
    def create_comprehensive_2d_plot(self, X: np.ndarray, 
                                   true_anomalies: np.ndarray,
                                   method_results: Dict[str, Tuple],
                                   save_path: str = None):
        """Create comprehensive 2D visualization of all methods."""
        # Determine subplot layout
        n_methods = len(method_results)
        n_cols = 3
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        method_names = list(method_results.keys())
        title_mapping = {
            'z_score': 'Z-Score',
            'iqr': 'IQR Method',
            'knn_distance': 'k-NN Distance',
            'dbscan': 'DBSCAN',
            'lof': 'Local Outlier Factor',
            'isolation_forest': 'Isolation Forest',
            'one_class_svm': 'One-Class SVM',
            'autoencoder': 'Autoencoder'
        }
        
        for i, method in enumerate(method_names):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]
            
            # Get results
            if method == 'iqr':
                _, anomalies = method_results[method][:2]
            elif method == 'dbscan':
                _, anomalies = method_results[method]
            else:
                _, anomalies = method_results[method]
            
            # Plot normal points
            normal_mask = ~anomalies
            ax.scatter(X[normal_mask, 0], X[normal_mask, 1], 
                      c='lightblue', alpha=0.6, s=30, label='Normal')
            
            # Plot detected anomalies
            if np.any(anomalies):
                ax.scatter(X[anomalies, 0], X[anomalies, 1], 
                          c='red', alpha=0.8, s=80, marker='o',
                          edgecolors='darkred', linewidths=1,
                          label='Detected')
            
            # Plot true anomalies
            true_anomaly_mask = true_anomalies.astype(bool)
            ax.scatter(X[true_anomaly_mask, 0], X[true_anomaly_mask, 1], 
                      c='orange', alpha=0.9, s=120, marker='+', 
                      linewidths=3, label='True Anomalies')
            
            ax.set_title(title_mapping.get(method, method.replace('_', ' ').title()))
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.legend(fontsize='small')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_methods, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_performance_comparison(self, true_anomalies: np.ndarray,
                                    method_results: Dict[str, Tuple],
                                    save_path: str = None):
        """Create performance comparison visualization."""
        # Calculate metrics for all methods
        metrics_data = []
        
        for method, result in method_results.items():
            # Extract anomaly predictions
            if method == 'iqr':
                _, anomalies = result[:2]
            elif method == 'dbscan':
                _, anomalies = result
            else:
                _, anomalies = result
            
            # Convert to binary
            anomalies_binary = anomalies.astype(int)
            
            # Calculate metrics
            precision = precision_score(true_anomalies, anomalies_binary, zero_division=0)
            recall = recall_score(true_anomalies, anomalies_binary, zero_division=0)
            f1 = f1_score(true_anomalies, anomalies_binary, zero_division=0)
            
            # Calculate ROC AUC if possible
            try:
                if method in ['z_score', 'knn_distance', 'lof', 'autoencoder']:
                    scores, _ = result
                    if scores.ndim > 1:
                        scores = np.max(scores, axis=1)
                    roc_auc = roc_auc_score(true_anomalies, scores)
                elif method in ['isolation_forest', 'one_class_svm']:
                    scores, _ = result
                    roc_auc = roc_auc_score(true_anomalies, -scores)  # Convert to positive
                elif method == 'iqr':
                    scores, _, _ = result
                    if scores.ndim > 1:
                        scores = np.max(scores, axis=1)
                    roc_auc = roc_auc_score(true_anomalies, scores)
                else:
                    roc_auc = 0.0
            except:
                roc_auc = 0.0
            
            metrics_data.append({
                'Method': method.replace('_', ' ').title(),
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC-AUC': roc_auc
            })
        
        # Create DataFrame
        df_metrics = pd.DataFrame(metrics_data)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Precision comparison
        ax = axes[0, 0]
        bars = ax.bar(df_metrics['Method'], df_metrics['Precision'], 
                     color='skyblue', alpha=0.7, edgecolor='navy')
        ax.set_title('Precision Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Precision')
        ax.set_ylim(0, 1)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, df_metrics['Precision']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Recall comparison
        ax = axes[0, 1]
        bars = ax.bar(df_metrics['Method'], df_metrics['Recall'], 
                     color='lightgreen', alpha=0.7, edgecolor='darkgreen')
        ax.set_title('Recall Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Recall')
        ax.set_ylim(0, 1)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        for bar, value in zip(bars, df_metrics['Recall']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # F1-Score comparison
        ax = axes[1, 0]
        bars = ax.bar(df_metrics['Method'], df_metrics['F1-Score'], 
                     color='salmon', alpha=0.7, edgecolor='darkred')
        ax.set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('F1-Score')
        ax.set_ylim(0, 1)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        for bar, value in zip(bars, df_metrics['F1-Score']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # ROC-AUC comparison
        ax = axes[1, 1]
        bars = ax.bar(df_metrics['Method'], df_metrics['ROC-AUC'], 
                     color='gold', alpha=0.7, edgecolor='orange')
        ax.set_title('ROC-AUC Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('ROC-AUC')
        ax.set_ylim(0, 1)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        for bar, value in zip(bars, df_metrics['ROC-AUC']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, df_metrics
    
    def create_roc_comparison(self, true_anomalies: np.ndarray,
                            method_results: Dict[str, Tuple],
                            save_path: str = None):
        """Create ROC curve comparison."""
        plt.figure(figsize=(12, 10))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(method_results)))
        
        for i, (method, result) in enumerate(method_results.items()):
            try:
                # Get scores
                if method in ['z_score', 'knn_distance', 'lof', 'autoencoder']:
                    scores, _ = result
                    if scores.ndim > 1:
                        scores = np.max(scores, axis=1)
                elif method in ['isolation_forest', 'one_class_svm']:
                    scores, _ = result
                    scores = -scores  # Convert to positive
                elif method == 'iqr':
                    scores, _, _ = result
                    if scores.ndim > 1:
                        scores = np.max(scores, axis=1)
                else:
                    continue
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(true_anomalies, scores)
                auc = roc_auc_score(true_anomalies, scores)
                
                plt.plot(fpr, tpr, color=colors[i], linewidth=2, 
                        label=f'{method.replace("_", " ").title()} (AUC = {auc:.3f})')
                
            except Exception as e:
                print(f"Could not plot ROC for {method}: {e}")
                continue
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - All Anomaly Detection Methods', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    def create_precision_at_k_analysis(self, true_anomalies: np.ndarray,
                                     method_results: Dict[str, Tuple],
                                     max_k: int = 50,
                                     save_path: str = None):
        """Create Precision@k analysis."""
        plt.figure(figsize=(12, 8))
        
        n_samples = len(true_anomalies)
        k_values = range(1, min(max_k + 1, n_samples + 1))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(method_results)))
        
        for i, (method, result) in enumerate(method_results.items()):
            try:
                # Get scores
                if method in ['z_score', 'knn_distance', 'lof', 'autoencoder']:
                    scores, _ = result
                    if scores.ndim > 1:
                        scores = np.max(scores, axis=1)
                elif method in ['isolation_forest', 'one_class_svm']:
                    scores, _ = result
                    scores = -scores  # Convert to positive
                elif method == 'iqr':
                    scores, _, _ = result
                    if scores.ndim > 1:
                        scores = np.max(scores, axis=1)
                else:
                    continue
                
                precisions_at_k = []
                
                for k in k_values:
                    # Get top k anomalous points
                    top_k_indices = np.argsort(scores)[-k:]
                    
                    # Calculate precision@k
                    true_positives = np.sum(true_anomalies[top_k_indices])
                    precision_at_k = true_positives / k if k > 0 else 0
                    precisions_at_k.append(precision_at_k)
                
                plt.plot(k_values, precisions_at_k, color=colors[i], linewidth=2, 
                        marker='o', markersize=3, alpha=0.8,
                        label=method.replace('_', ' ').title())
                
            except Exception as e:
                print(f"Could not plot Precision@k for {method}: {e}")
                continue
        
        plt.xlabel('k (Top k predictions)', fontsize=12)
        plt.ylabel('Precision@k', fontsize=12)
        plt.title('Precision@k Analysis - All Methods', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim([1, max(k_values)])
        plt.ylim([0, 1.05])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    def visualize_time_series_anomalies(self, t: np.ndarray, 
                                      y: np.ndarray, 
                                      true_anomalies: np.ndarray,
                                      save_path: str = None):
        """Visualize time series anomaly detection."""
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Apply statistical methods to time series
        z_scores, z_anomalies = self.stat_detector.z_score_detection(y, threshold=3.0)
        outlier_scores, iqr_anomalies, bounds = self.stat_detector.iqr_detection(y, multiplier=1.5)
        
        # Original time series with true anomalies
        ax = axes[0]
        ax.plot(t, y, 'b-', alpha=0.7, linewidth=1, label='Time Series')
        
        # Highlight true anomalies
        true_anomaly_indices = np.where(true_anomalies == 1)[0]
        if len(true_anomaly_indices) > 0:
            ax.scatter(t[true_anomaly_indices], y[true_anomaly_indices], 
                      c='red', s=100, marker='o', alpha=0.8,
                      label='True Anomalies', zorder=5)
        
        ax.set_title('Time Series with True Anomalies', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Detected anomalies comparison
        ax = axes[1]
        ax.plot(t, y, 'b-', alpha=0.5, linewidth=1, label='Time Series')
        
        # Z-score anomalies
        z_anomaly_indices = np.where(z_anomalies)[0]
        if len(z_anomaly_indices) > 0:
            ax.scatter(t[z_anomaly_indices], y[z_anomaly_indices], 
                      c='orange', s=80, marker='^', alpha=0.8,
                      label='Z-Score Anomalies', zorder=4)
        
        # IQR anomalies
        iqr_anomaly_indices = np.where(iqr_anomalies)[0]
        if len(iqr_anomaly_indices) > 0:
            ax.scatter(t[iqr_anomaly_indices], y[iqr_anomaly_indices], 
                      c='green', s=80, marker='s', alpha=0.8,
                      label='IQR Anomalies', zorder=4)
        
        # True anomalies for comparison
        if len(true_anomaly_indices) > 0:
            ax.scatter(t[true_anomaly_indices], y[true_anomaly_indices], 
                      c='red', s=100, marker='o', alpha=0.8,
                      label='True Anomalies', zorder=5)
        
        # Add IQR bounds
        if bounds[0].ndim == 0:  # scalar bounds
            ax.axhline(y=bounds[0], color='gray', linestyle='--', alpha=0.5, label='IQR Bounds')
            ax.axhline(y=bounds[1], color='gray', linestyle='--', alpha=0.5)
        
        ax.set_title('Detected Anomalies (Statistical Methods)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def run_comprehensive_analysis():
    """Run comprehensive anomaly detection analysis and visualization."""
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Initialize visualization suite
    viz_suite = AnomalyVisualizationSuite()
    
    print("Comprehensive Anomaly Detection Analysis")
    print("=" * 45)
    
    # Test on different datasets
    datasets = {
        'gaussian_clusters': viz_suite.data_generator.gaussian_clusters_with_outliers(n_normal=200, n_outliers=20),
        'mixed_distributions': viz_suite.data_generator.mixed_distributions(n_normal=200, n_outliers=20),
        'moons': viz_suite.data_generator.moons_with_outliers(n_normal=200, n_outliers=20)
    }
    
    for dataset_name, (X, true_anomalies) in datasets.items():
        print(f"\nAnalyzing {dataset_name.replace('_', ' ').title()} Dataset")
        print("-" * 50)
        
        # Run all methods
        method_results = viz_suite.run_all_methods(X, include_autoencoder=False)  # Skip autoencoder for speed
        
        # Create comprehensive 2D plot
        viz_suite.create_comprehensive_2d_plot(
            X, true_anomalies, method_results,
            save_path=f'plots/comprehensive_2d_{dataset_name}.png'
        )
        plt.suptitle(f'All Methods - {dataset_name.replace("_", " ").title()}', fontsize=16)
        plt.show()
        
        # Create performance comparison
        fig, df_metrics = viz_suite.create_performance_comparison(
            true_anomalies, method_results,
            save_path=f'plots/performance_comparison_{dataset_name}.png'
        )
        plt.suptitle(f'Performance Comparison - {dataset_name.replace("_", " ").title()}', fontsize=16)
        plt.show()
        
        # Print performance table
        print(f"\nPerformance Results - {dataset_name.replace('_', ' ').title()}:")
        print(df_metrics.to_string(index=False, float_format='%.3f'))
        
        # Create ROC comparison
        viz_suite.create_roc_comparison(
            true_anomalies, method_results,
            save_path=f'plots/roc_comparison_{dataset_name}.png'
        )
        plt.show()
        
        # Create Precision@k analysis
        viz_suite.create_precision_at_k_analysis(
            true_anomalies, method_results,
            save_path=f'plots/precision_at_k_{dataset_name}.png'
        )
        plt.show()
    
    # Time series analysis
    print("\nTime Series Anomaly Analysis")
    print("-" * 30)
    
    t, y, true_anomalies_ts = viz_suite.data_generator.time_series_with_spikes(length=300, spike_probability=0.03)
    
    viz_suite.visualize_time_series_anomalies(
        t, y, true_anomalies_ts,
        save_path='plots/time_series_anomaly_detection.png'
    )
    plt.show()
    
    print("\nAnalysis complete! Check the 'plots' directory for all visualizations.")


if __name__ == "__main__":
    run_comprehensive_analysis() 