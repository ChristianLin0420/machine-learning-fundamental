"""
DBSCAN Comparison with Scikit-learn
===================================

This module provides comprehensive comparison between our DBSCAN implementation
and scikit-learn's DBSCAN implementation, including:

- Performance comparison (clustering quality, runtime)
- Clustering result visualization side-by-side
- Statistical analysis across multiple datasets
- Parameter optimization comparison
- Edge case testing and robustness analysis

Features:
- Side-by-side clustering visualizations
- Performance metrics comparison
- Parameter sensitivity analysis
- Multiple dataset evaluation
- Algorithm behavior analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import DBSCAN as SklearnDBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import time
from dbscan_from_scratch import DBSCANScratch, DBSCANAnalyzer, generate_spiral_dataset
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DBSCANComparator:
    """
    Comprehensive comparator between our DBSCAN implementation and sklearn.
    
    This class provides tools for comparing performance, quality, and
    behavior between different DBSCAN implementations.
    """
    
    def __init__(self):
        """Initialize the comparator."""
        self.results = {}
        self.datasets = {}
    
    def generate_comparison_datasets(self):
        """
        Generate various datasets for comprehensive comparison.
        
        Returns:
        --------
        dict : Dictionary containing different test datasets
        """
        datasets = {}
        
        # Dataset 1: Two moons (non-convex clusters)
        X_moons, y_moons = make_moons(n_samples=300, noise=0.05, random_state=42)
        datasets['moons'] = {
            'data': X_moons, 
            'labels': y_moons, 
            'name': 'Two Moons',
            'optimal_eps': 0.15,
            'optimal_min_samples': 5
        }
        
        # Dataset 2: Well-separated blobs
        X_blobs, y_blobs = make_blobs(n_samples=300, centers=4, n_features=2, 
                                     random_state=42, cluster_std=0.8)
        datasets['blobs'] = {
            'data': X_blobs, 
            'labels': y_blobs, 
            'name': 'Well-separated Blobs',
            'optimal_eps': 0.8,
            'optimal_min_samples': 10
        }
        
        # Dataset 3: Concentric circles
        X_circles, y_circles = make_circles(n_samples=300, noise=0.05, 
                                           factor=0.3, random_state=42)
        datasets['circles'] = {
            'data': X_circles, 
            'labels': y_circles, 
            'name': 'Concentric Circles',
            'optimal_eps': 0.15,
            'optimal_min_samples': 5
        }
        
        # Dataset 4: Spiral dataset
        X_spiral, y_spiral = generate_spiral_dataset(n_samples=300, noise=0.05, random_state=42)
        datasets['spiral'] = {
            'data': X_spiral, 
            'labels': y_spiral, 
            'name': 'Spiral Dataset',
            'optimal_eps': 0.1,
            'optimal_min_samples': 5
        }
        
        # Dataset 5: Overlapping blobs (challenging case)
        X_overlap, y_overlap = make_blobs(n_samples=300, centers=3, n_features=2, 
                                         random_state=42, cluster_std=2.0)
        datasets['overlap'] = {
            'data': X_overlap, 
            'labels': y_overlap, 
            'name': 'Overlapping Clusters',
            'optimal_eps': 1.2,
            'optimal_min_samples': 8
        }
        
        # Dataset 6: High-dimensional data
        X_high, y_high = make_blobs(n_samples=200, centers=3, n_features=8, 
                                   random_state=42, cluster_std=1.5)
        datasets['high_dim'] = {
            'data': X_high, 
            'labels': y_high, 
            'name': 'High-Dimensional Data',
            'optimal_eps': 2.0,
            'optimal_min_samples': 10
        }
        
        self.datasets = datasets
        return datasets
    
    def compare_implementations(self, dataset_name, eps=None, min_samples=None, 
                              random_state=42, **kwargs):
        """
        Compare our implementation with sklearn on a specific dataset.
        
        Parameters:
        -----------
        dataset_name : str
            Name of dataset to use
        eps : float, optional
            Epsilon parameter (uses optimal from dataset if None)
        min_samples : int, optional
            Min samples parameter (uses optimal from dataset if None)
        random_state : int
            Random seed
        **kwargs : dict
            Additional arguments for DBSCAN
            
        Returns:
        --------
        dict : Comparison results
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found. "
                           f"Available: {list(self.datasets.keys())}")
        
        dataset = self.datasets[dataset_name]
        X = dataset['data']
        true_labels = dataset['labels']
        
        if eps is None:
            eps = dataset['optimal_eps']
        if min_samples is None:
            min_samples = dataset['optimal_min_samples']
        
        print(f"Comparing implementations on {dataset['name']}")
        print(f"  Data shape: {X.shape}")
        print(f"  Epsilon: {eps}")
        print(f"  Min samples: {min_samples}")
        
        # Our implementation
        print("\n🔧 Our DBSCAN Implementation:")
        start_time = time.time()
        dbscan_scratch = DBSCANScratch(eps=eps, min_samples=min_samples, **kwargs)
        labels_scratch = dbscan_scratch.fit_predict(X)
        time_scratch = time.time() - start_time
        
        # Sklearn implementation
        print("\n🔍 Scikit-learn DBSCAN:")
        start_time = time.time()
        dbscan_sklearn = SklearnDBSCAN(eps=eps, min_samples=min_samples)
        labels_sklearn = dbscan_sklearn.fit_predict(X)
        time_sklearn = time.time() - start_time
        
        # Compute metrics
        results = {
            'dataset': dataset_name,
            'data_shape': X.shape,
            'eps': eps,
            'min_samples': min_samples,
            'scratch': {
                'labels': labels_scratch,
                'n_clusters': len(set(labels_scratch)) - (1 if -1 in labels_scratch else 0),
                'n_noise': list(labels_scratch).count(-1),
                'n_core_samples': len(dbscan_scratch.core_sample_indices_),
                'time': time_scratch
            },
            'sklearn': {
                'labels': labels_sklearn,
                'n_clusters': len(set(labels_sklearn)) - (1 if -1 in labels_sklearn else 0),
                'n_noise': list(labels_sklearn).count(-1),
                'n_core_samples': len(dbscan_sklearn.core_sample_indices_),
                'time': time_sklearn
            }
        }
        
        # Clustering quality metrics
        for impl_name, labels in [('scratch', labels_scratch), ('sklearn', labels_sklearn)]:
            n_clusters = results[impl_name]['n_clusters']
            n_noise = results[impl_name]['n_noise']
            
            if n_clusters > 1 and n_noise < len(labels):
                try:
                    results[impl_name]['silhouette'] = silhouette_score(X, labels)
                except:
                    results[impl_name]['silhouette'] = -1
            else:
                results[impl_name]['silhouette'] = -1
        
        # If true labels available
        if true_labels is not None:
            for impl_name, labels in [('scratch', labels_scratch), ('sklearn', labels_sklearn)]:
                results[impl_name]['ari'] = adjusted_rand_score(true_labels, labels)
                results[impl_name]['nmi'] = normalized_mutual_info_score(true_labels, labels)
        
        # Store results
        self.results[dataset_name] = results
        
        # Print comparison
        print(f"\n📊 Comparison Results:")
        print(f"  {'Metric':<20} {'Our Impl':<15} {'Sklearn':<15} {'Difference':<15}")
        print(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*15}")
        print(f"  {'N Clusters':<20} {results['scratch']['n_clusters']:<15} "
              f"{results['sklearn']['n_clusters']:<15} "
              f"{abs(results['scratch']['n_clusters'] - results['sklearn']['n_clusters']):<15}")
        print(f"  {'N Noise':<20} {results['scratch']['n_noise']:<15} "
              f"{results['sklearn']['n_noise']:<15} "
              f"{abs(results['scratch']['n_noise'] - results['sklearn']['n_noise']):<15}")
        print(f"  {'N Core Samples':<20} {results['scratch']['n_core_samples']:<15} "
              f"{results['sklearn']['n_core_samples']:<15} "
              f"{abs(results['scratch']['n_core_samples'] - results['sklearn']['n_core_samples']):<15}")
        print(f"  {'Time (s)':<20} {results['scratch']['time']:<15.4f} "
              f"{results['sklearn']['time']:<15.4f} "
              f"{abs(results['scratch']['time'] - results['sklearn']['time']):<15.4f}")
        
        if 'silhouette' in results['scratch']:
            print(f"  {'Silhouette':<20} {results['scratch']['silhouette']:<15.4f} "
                  f"{results['sklearn']['silhouette']:<15.4f} "
                  f"{abs(results['scratch']['silhouette'] - results['sklearn']['silhouette']):<15.4f}")
        
        if 'ari' in results['scratch']:
            print(f"  {'ARI':<20} {results['scratch']['ari']:<15.4f} "
                  f"{results['sklearn']['ari']:<15.4f} "
                  f"{abs(results['scratch']['ari'] - results['sklearn']['ari']):<15.4f}")
        
        return results
    
    def visualize_comparison(self, dataset_name, save_path=None, figsize=(16, 12)):
        """
        Visualize clustering results comparison.
        
        Parameters:
        -----------
        dataset_name : str
            Name of dataset to visualize
        save_path : str, optional
            Path to save the plot
        figsize : tuple
            Figure size
        """
        if dataset_name not in self.results:
            raise ValueError(f"No results for dataset '{dataset_name}'. Run comparison first.")
        
        dataset = self.datasets[dataset_name]
        results = self.results[dataset_name]
        X = dataset['data']
        true_labels = dataset['labels']
        
        # For high-dimensional data, use PCA for visualization
        if X.shape[1] > 2:
            pca = PCA(n_components=2, random_state=42)
            X_vis = pca.fit_transform(X)
            title_suffix = " (PCA projection)"
        else:
            X_vis = X
            title_suffix = ""
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Plot 1: True labels (if available)
        ax1 = axes[0, 0]
        if true_labels is not None:
            unique_true = np.unique(true_labels)
            colors_true = plt.cm.Spectral(np.linspace(0, 1, len(unique_true)))
            for label, color in zip(unique_true, colors_true):
                mask = true_labels == label
                ax1.scatter(X_vis[mask, 0], X_vis[mask, 1], c=[color], 
                           alpha=0.7, s=50, label=f'True {label}')
            ax1.set_title(f'True Labels{title_suffix}')
        else:
            ax1.scatter(X_vis[:, 0], X_vis[:, 1], alpha=0.7, s=50)
            ax1.set_title(f'Data Points{title_suffix}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Our implementation
        ax2 = axes[0, 1]
        labels_scratch = results['scratch']['labels']
        unique_scratch = np.unique(labels_scratch)
        colors_scratch = plt.cm.Spectral(np.linspace(0, 1, len(unique_scratch)))
        
        for label, color in zip(unique_scratch, colors_scratch):
            mask = labels_scratch == label
            if label == -1:
                ax2.scatter(X_vis[mask, 0], X_vis[mask, 1], c='black', 
                           marker='x', s=50, alpha=0.6, label='Noise')
            else:
                ax2.scatter(X_vis[mask, 0], X_vis[mask, 1], c=[color], 
                           alpha=0.7, s=50, label=f'Cluster {label}')
        
        ax2.set_title(f'Our DBSCAN{title_suffix}\n'
                     f'Clusters: {results["scratch"]["n_clusters"]}, '
                     f'Noise: {results["scratch"]["n_noise"]}')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Sklearn implementation
        ax3 = axes[0, 2]
        labels_sklearn = results['sklearn']['labels']
        unique_sklearn = np.unique(labels_sklearn)
        colors_sklearn = plt.cm.Spectral(np.linspace(0, 1, len(unique_sklearn)))
        
        for label, color in zip(unique_sklearn, colors_sklearn):
            mask = labels_sklearn == label
            if label == -1:
                ax3.scatter(X_vis[mask, 0], X_vis[mask, 1], c='black', 
                           marker='x', s=50, alpha=0.6, label='Noise')
            else:
                ax3.scatter(X_vis[mask, 0], X_vis[mask, 1], c=[color], 
                           alpha=0.7, s=50, label=f'Cluster {label}')
        
        ax3.set_title(f'Sklearn DBSCAN{title_suffix}\n'
                     f'Clusters: {results["sklearn"]["n_clusters"]}, '
                     f'Noise: {results["sklearn"]["n_noise"]}')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Performance comparison
        ax4 = axes[1, 0]
        metrics = ['N Clusters', 'N Noise', 'N Core', 'Time (s)']
        our_values = [results['scratch']['n_clusters'], 
                     results['scratch']['n_noise'], 
                     results['scratch']['n_core_samples'],
                     results['scratch']['time']]
        sklearn_values = [results['sklearn']['n_clusters'], 
                         results['sklearn']['n_noise'], 
                         results['sklearn']['n_core_samples'],
                         results['sklearn']['time']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, our_values, width, label='Our Implementation', alpha=0.7)
        bars2 = ax4.bar(x + width/2, sklearn_values, width, label='Scikit-learn', alpha=0.7)
        
        ax4.set_xlabel('Metrics')
        ax4.set_ylabel('Values')
        ax4.set_title('Performance Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add value annotations
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.annotate(f'{height:.3f}', 
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
        
        # Plot 5: Label agreement analysis
        ax5 = axes[1, 1]
        
        # Compute label agreement (ignoring label permutation)
        agreement_matrix = np.zeros((len(unique_scratch), len(unique_sklearn)))
        
        for i, label_s in enumerate(unique_scratch):
            for j, label_k in enumerate(unique_sklearn):
                mask_s = labels_scratch == label_s
                mask_k = labels_sklearn == label_k
                agreement = np.sum(mask_s & mask_k)
                agreement_matrix[i, j] = agreement
        
        im = ax5.imshow(agreement_matrix, aspect='auto', cmap='Blues')
        ax5.set_xticks(range(len(unique_sklearn)))
        ax5.set_xticklabels([f'SK-{l}' for l in unique_sklearn])
        ax5.set_yticks(range(len(unique_scratch)))
        ax5.set_yticklabels([f'Our-{l}' for l in unique_scratch])
        ax5.set_xlabel('Sklearn Labels')
        ax5.set_ylabel('Our Labels')
        ax5.set_title('Label Agreement Matrix')
        
        # Add text annotations
        for i in range(len(unique_scratch)):
            for j in range(len(unique_sklearn)):
                text = ax5.text(j, i, int(agreement_matrix[i, j]),
                               ha="center", va="center", color="white" if agreement_matrix[i, j] > agreement_matrix.max()/2 else "black")
        
        plt.colorbar(im, ax=ax5)
        
        # Plot 6: Quality metrics comparison
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        info_text = f"""Comparison Summary:
        
Dataset: {dataset['name']}
Parameters:
  ε (epsilon): {results['eps']}
  min_samples: {results['min_samples']}

Results Comparison:
  Clusters: {results['scratch']['n_clusters']} vs {results['sklearn']['n_clusters']}
  Noise Points: {results['scratch']['n_noise']} vs {results['sklearn']['n_noise']}
  Core Samples: {results['scratch']['n_core_samples']} vs {results['sklearn']['n_core_samples']}
  Runtime: {results['scratch']['time']:.4f}s vs {results['sklearn']['time']:.4f}s
"""
        
        if 'silhouette' in results['scratch']:
            info_text += f"\nQuality Metrics:\n"
            info_text += f"  Silhouette: {results['scratch']['silhouette']:.4f} vs {results['sklearn']['silhouette']:.4f}\n"
        
        if 'ari' in results['scratch']:
            info_text += f"  ARI: {results['scratch']['ari']:.4f} vs {results['sklearn']['ari']:.4f}\n"
            info_text += f"  NMI: {results['scratch']['nmi']:.4f} vs {results['sklearn']['nmi']:.4f}\n"
        
        # Agreement analysis
        total_points = len(labels_scratch)
        agreement_points = np.sum(labels_scratch == labels_sklearn)
        agreement_percent = (agreement_points / total_points) * 100
        
        info_text += f"\nLabel Agreement:\n"
        info_text += f"  Exact Match: {agreement_points}/{total_points} ({agreement_percent:.1f}%)\n"
        
        ax6.text(0.1, 0.9, info_text, transform=ax6.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison visualization saved to {save_path}")
        
        plt.show()
    
    def distance_metric_comparison(self, dataset_name, save_path=None):
        """
        Compare different distance metrics.
        
        Parameters:
        -----------
        dataset_name : str
            Name of dataset to use
        save_path : str, optional
            Path to save the plot
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        dataset = self.datasets[dataset_name]
        X = dataset['data']
        eps = dataset['optimal_eps']
        min_samples = dataset['optimal_min_samples']
        
        print(f"Comparing distance metrics on {dataset['name']}")
        
        # Test different distance metrics
        metrics = ['euclidean', 'manhattan', 'cosine']
        results = {}
        
        for metric in metrics:
            print(f"\nTesting {metric} distance...")
            
            try:
                dbscan = DBSCANScratch(eps=eps, min_samples=min_samples, metric=metric)
                labels = dbscan.fit_predict(X)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                # Compute silhouette score if valid clustering
                if n_clusters > 1 and n_noise < len(labels):
                    try:
                        sil_score = silhouette_score(X, labels)
                    except:
                        sil_score = -1
                else:
                    sil_score = -1
                
                results[metric] = {
                    'labels': labels,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'silhouette': sil_score,
                    'dbscan': dbscan
                }
                
                print(f"  Clusters: {n_clusters}, Noise: {n_noise}, Silhouette: {sil_score:.3f}")
                
            except Exception as e:
                print(f"  Error with {metric}: {e}")
                results[metric] = None
        
        # Visualize results
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # For high-dimensional data, use PCA
        if X.shape[1] > 2:
            pca = PCA(n_components=2, random_state=42)
            X_vis = pca.fit_transform(X)
            title_suffix = " (PCA projection)"
        else:
            X_vis = X
            title_suffix = ""
        
        # Plot clustering results for each metric
        valid_metrics = [m for m in metrics if results[m] is not None]
        
        for i, metric in enumerate(valid_metrics[:3]):  # Plot first 3 valid metrics
            if i >= 3:
                break
            
            ax = axes[i//2, i%2] if i < 3 else None
            if ax is None:
                continue
            
            labels = results[metric]['labels']
            unique_labels = np.unique(labels)
            colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(unique_labels, colors):
                mask = labels == label
                if label == -1:
                    ax.scatter(X_vis[mask, 0], X_vis[mask, 1], c='black', 
                             marker='x', s=50, alpha=0.6, label='Noise')
                else:
                    ax.scatter(X_vis[mask, 0], X_vis[mask, 1], c=[color], 
                             alpha=0.7, s=50, label=f'Cluster {label}')
            
            ax.set_title(f'{metric.capitalize()} Distance{title_suffix}\n'
                        f'Clusters: {results[metric]["n_clusters"]}, '
                        f'Noise: {results[metric]["n_noise"]}, '
                        f'Silhouette: {results[metric]["silhouette"]:.3f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Comparison summary
        ax4 = axes[1, 1] if len(valid_metrics) <= 3 else axes[-1, -1]
        ax4.axis('off')
        
        summary_text = f"""Distance Metric Comparison
        
Dataset: {dataset['name']}
Parameters: ε={eps}, min_samples={min_samples}

Results:
"""
        
        for metric in valid_metrics:
            if results[metric] is not None:
                r = results[metric]
                summary_text += f"  {metric.capitalize()}:\n"
                summary_text += f"    Clusters: {r['n_clusters']}\n"
                summary_text += f"    Noise: {r['n_noise']}\n"
                summary_text += f"    Silhouette: {r['silhouette']:.3f}\n\n"
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distance metric comparison saved to {save_path}")
        
        plt.show()
        
        return results
    
    def algorithm_comparison(self, dataset_name, save_path=None):
        """
        Compare BFS vs DFS cluster expansion algorithms.
        
        Parameters:
        -----------
        dataset_name : str
            Name of dataset to use
        save_path : str, optional
            Path to save the plot
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        dataset = self.datasets[dataset_name]
        X = dataset['data']
        eps = dataset['optimal_eps']
        min_samples = dataset['optimal_min_samples']
        
        print(f"Comparing BFS vs DFS algorithms on {dataset['name']}")
        
        # Test different algorithms
        algorithms = ['bfs', 'dfs']
        results = {}
        
        for algorithm in algorithms:
            print(f"\nTesting {algorithm.upper()} algorithm...")
            
            start_time = time.time()
            dbscan = DBSCANScratch(eps=eps, min_samples=min_samples, algorithm=algorithm)
            labels = dbscan.fit_predict(X)
            runtime = time.time() - start_time
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            results[algorithm] = {
                'labels': labels,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'runtime': runtime,
                'dbscan': dbscan
            }
            
            print(f"  Clusters: {n_clusters}, Noise: {n_noise}, Time: {runtime:.4f}s")
        
        # Check if results are identical
        labels_identical = np.array_equal(results['bfs']['labels'], results['dfs']['labels'])
        
        # Visualize results
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # For high-dimensional data, use PCA
        if X.shape[1] > 2:
            pca = PCA(n_components=2, random_state=42)
            X_vis = pca.fit_transform(X)
            title_suffix = " (PCA projection)"
        else:
            X_vis = X
            title_suffix = ""
        
        # Plot BFS results
        ax1 = axes[0, 0]
        labels_bfs = results['bfs']['labels']
        unique_bfs = np.unique(labels_bfs)
        colors_bfs = plt.cm.Spectral(np.linspace(0, 1, len(unique_bfs)))
        
        for label, color in zip(unique_bfs, colors_bfs):
            mask = labels_bfs == label
            if label == -1:
                ax1.scatter(X_vis[mask, 0], X_vis[mask, 1], c='black', 
                           marker='x', s=50, alpha=0.6, label='Noise')
            else:
                ax1.scatter(X_vis[mask, 0], X_vis[mask, 1], c=[color], 
                           alpha=0.7, s=50, label=f'Cluster {label}')
        
        ax1.set_title(f'BFS Algorithm{title_suffix}\n'
                     f'Time: {results["bfs"]["runtime"]:.4f}s')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot DFS results
        ax2 = axes[0, 1]
        labels_dfs = results['dfs']['labels']
        unique_dfs = np.unique(labels_dfs)
        colors_dfs = plt.cm.Spectral(np.linspace(0, 1, len(unique_dfs)))
        
        for label, color in zip(unique_dfs, colors_dfs):
            mask = labels_dfs == label
            if label == -1:
                ax2.scatter(X_vis[mask, 0], X_vis[mask, 1], c='black', 
                           marker='x', s=50, alpha=0.6, label='Noise')
            else:
                ax2.scatter(X_vis[mask, 0], X_vis[mask, 1], c=[color], 
                           alpha=0.7, s=50, label=f'Cluster {label}')
        
        ax2.set_title(f'DFS Algorithm{title_suffix}\n'
                     f'Time: {results["dfs"]["runtime"]:.4f}s')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot runtime comparison
        ax3 = axes[1, 0]
        algorithms_list = list(results.keys())
        runtimes = [results[alg]['runtime'] for alg in algorithms_list]
        
        bars = ax3.bar(algorithms_list, runtimes, alpha=0.7, color=['skyblue', 'lightcoral'])
        ax3.set_ylabel('Runtime (seconds)')
        ax3.set_title('Algorithm Runtime Comparison')
        ax3.grid(True, alpha=0.3)
        
        # Add value annotations
        for bar, runtime in zip(bars, runtimes):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{runtime:.4f}s', ha='center', va='bottom')
        
        # Plot comparison summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = f"""Algorithm Comparison Summary
        
Dataset: {dataset['name']}
Parameters: ε={eps}, min_samples={min_samples}

BFS Results:
  Clusters: {results['bfs']['n_clusters']}
  Noise Points: {results['bfs']['n_noise']}
  Runtime: {results['bfs']['runtime']:.4f}s

DFS Results:
  Clusters: {results['dfs']['n_clusters']}
  Noise Points: {results['dfs']['n_noise']}
  Runtime: {results['dfs']['runtime']:.4f}s

Analysis:
  Results Identical: {labels_identical}
  Speed Difference: {abs(results['bfs']['runtime'] - results['dfs']['runtime']):.4f}s
  
Note: Both algorithms should produce 
identical clustering results, differing 
only in the order of cluster expansion.
        """
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Algorithm comparison saved to {save_path}")
        
        plt.show()
        
        return results
    
    def comprehensive_analysis(self, save_plots=True):
        """
        Run comprehensive analysis across all datasets.
        
        Parameters:
        -----------
        save_plots : bool
            Whether to save generated plots
        """
        print("🔍 COMPREHENSIVE DBSCAN ANALYSIS")
        print("=" * 50)
        
        # Generate datasets
        self.generate_comparison_datasets()
        
        # Analyze each dataset
        summary_results = []
        
        for dataset_name, dataset in self.datasets.items():
            print(f"\n📊 Analyzing {dataset['name']}")
            print("-" * 40)
            
            # Compare implementations
            results = self.compare_implementations(dataset_name)
            
            # Visualize comparison
            if save_plots:
                save_path = f"plots/comparison_{dataset_name}.png"
            else:
                save_path = None
            self.visualize_comparison(dataset_name, save_path)
            
            # Add to summary
            summary_results.append({
                'Dataset': dataset['name'],
                'Our Clusters': results['scratch']['n_clusters'],
                'Sklearn Clusters': results['sklearn']['n_clusters'],
                'Our Noise': results['scratch']['n_noise'],
                'Sklearn Noise': results['sklearn']['n_noise'],
                'Our Time': results['scratch']['time'],
                'Sklearn Time': results['sklearn']['time'],
                'Cluster Diff': abs(results['scratch']['n_clusters'] - results['sklearn']['n_clusters']),
                'Noise Diff': abs(results['scratch']['n_noise'] - results['sklearn']['n_noise'])
            })
        
        # Create summary table
        summary_df = pd.DataFrame(summary_results)
        print(f"\n📋 SUMMARY TABLE:")
        print("=" * 100)
        print(summary_df.to_string(index=False, float_format='%.4f'))
        
        # Overall performance analysis
        print(f"\n🎯 OVERALL PERFORMANCE:")
        print("=" * 30)
        
        cluster_diff = summary_df['Cluster Diff']
        noise_diff = summary_df['Noise Diff']
        time_diff = summary_df['Our Time'] - summary_df['Sklearn Time']
        
        print(f"Average cluster difference: {np.mean(cluster_diff):.2f} ± {np.std(cluster_diff):.2f}")
        print(f"Average noise difference: {np.mean(noise_diff):.2f} ± {np.std(noise_diff):.2f}")
        print(f"Average time difference: {np.mean(time_diff):.4f} ± {np.std(time_diff):.4f} seconds")
        
        # Perfect agreement analysis
        perfect_cluster_agreement = np.sum(cluster_diff == 0)
        perfect_noise_agreement = np.sum(noise_diff == 0)
        
        print(f"Perfect cluster agreement: {perfect_cluster_agreement}/{len(summary_df)} datasets")
        print(f"Perfect noise agreement: {perfect_noise_agreement}/{len(summary_df)} datasets")
        
        return summary_df

def main():
    """
    Main function to run comprehensive DBSCAN comparison.
    """
    print("🎯 DBSCAN COMPARISON WITH SKLEARN")
    print("=" * 50)
    
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Create comparator
    comparator = DBSCANComparator()
    
    # Run comprehensive analysis
    summary = comparator.comprehensive_analysis(save_plots=True)
    
    # Test distance metric comparison
    print(f"\n🔧 Testing Distance Metrics:")
    print("-" * 40)
    distance_results = comparator.distance_metric_comparison(
        'moons', save_path='plots/distance_metric_comparison.png'
    )
    
    # Test algorithm comparison (BFS vs DFS)
    print(f"\n🔄 Testing Algorithm Comparison:")
    print("-" * 40)
    algorithm_results = comparator.algorithm_comparison(
        'moons', save_path='plots/algorithm_comparison.png'
    )
    
    # Test parameter sensitivity using our analyzer
    print(f"\n📈 Testing Parameter Sensitivity:")
    print("-" * 40)
    analyzer = DBSCANAnalyzer()
    
    # Use moons dataset for parameter analysis
    moons_data = comparator.datasets['moons']['data']
    analyzer.k_distance_graph(moons_data, k=4, 
                             save_path='plots/k_distance_graph_comparison.png')
    
    print("\n✅ COMPREHENSIVE DBSCAN COMPARISON COMPLETE!")
    print("📁 Check the 'plots' folder for generated visualizations.")
    print("🔧 Key findings:")
    print("   - Implementation accuracy compared to sklearn")
    print("   - Performance characteristics analysis")
    print("   - Distance metric effects on clustering")
    print("   - BFS vs DFS algorithm comparison")
    print("   - Parameter optimization guidance")
    
    return comparator, summary

if __name__ == "__main__":
    main() 