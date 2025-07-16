"""
K-Means Comparison with Scikit-learn
====================================

This module provides comprehensive comparison between our K-Means implementation
and scikit-learn's implementation, including:

- Performance comparison (inertia, convergence speed)
- Clustering quality metrics (silhouette, ARI, NMI)
- Visualization of clustering results
- Decision boundary analysis
- Initialization method comparison
- Scalability analysis

Features:
- Side-by-side clustering visualizations
- Statistical performance comparison
- Convergence analysis
- Multiple dataset evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, load_iris, make_circles, make_moons
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import time
from kmeans_from_scratch import KMeansScratch, KMeansAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class KMeansComparator:
    """
    Comprehensive comparator between our K-Means implementation and sklearn.
    
    This class provides tools for comparing performance, quality, and
    behavior between different K-Means implementations.
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
        
        # Dataset 1: Well-separated blobs
        X_blobs, y_blobs = make_blobs(n_samples=300, centers=4, n_features=2, 
                                      random_state=42, cluster_std=1.2)
        datasets['blobs'] = {
            'data': X_blobs, 
            'labels': y_blobs, 
            'name': 'Well-separated Blobs',
            'optimal_k': 4
        }
        
        # Dataset 2: Overlapping clusters
        X_overlap, y_overlap = make_blobs(n_samples=300, centers=3, n_features=2, 
                                         random_state=42, cluster_std=2.5)
        datasets['overlap'] = {
            'data': X_overlap, 
            'labels': y_overlap, 
            'name': 'Overlapping Clusters',
            'optimal_k': 3
        }
        
        # Dataset 3: Iris dataset (real-world data)
        iris = load_iris()
        datasets['iris'] = {
            'data': iris.data, 
            'labels': iris.target, 
            'name': 'Iris Dataset',
            'optimal_k': 3
        }
        
        # Dataset 4: Circles (challenging for K-Means)
        X_circles, y_circles = make_circles(n_samples=300, noise=0.1, 
                                           factor=0.3, random_state=42)
        datasets['circles'] = {
            'data': X_circles, 
            'labels': y_circles, 
            'name': 'Concentric Circles',
            'optimal_k': 2
        }
        
        # Dataset 5: Moons (challenging for K-Means)
        X_moons, y_moons = make_moons(n_samples=300, noise=0.1, random_state=42)
        datasets['moons'] = {
            'data': X_moons, 
            'labels': y_moons, 
            'name': 'Two Moons',
            'optimal_k': 2
        }
        
        # Dataset 6: High-dimensional data
        X_high, y_high = make_blobs(n_samples=200, centers=3, n_features=8, 
                                   random_state=42, cluster_std=1.5)
        datasets['high_dim'] = {
            'data': X_high, 
            'labels': y_high, 
            'name': 'High-Dimensional Data',
            'optimal_k': 3
        }
        
        self.datasets = datasets
        return datasets
    
    def compare_implementations(self, dataset_name, n_clusters=None, 
                              random_state=42, **kwargs):
        """
        Compare our implementation with sklearn on a specific dataset.
        
        Parameters:
        -----------
        dataset_name : str
            Name of dataset to use
        n_clusters : int, optional
            Number of clusters (uses optimal_k from dataset if None)
        random_state : int
            Random seed
        **kwargs : dict
            Additional arguments for K-Means
            
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
        
        if n_clusters is None:
            n_clusters = dataset['optimal_k']
        
        print(f"Comparing implementations on {dataset['name']}")
        print(f"  Data shape: {X.shape}")
        print(f"  Number of clusters: {n_clusters}")
        
        # Our implementation
        print("\n🔧 Our K-Means Implementation:")
        start_time = time.time()
        kmeans_scratch = KMeansScratch(n_clusters=n_clusters, 
                                      random_state=random_state, **kwargs)
        labels_scratch = kmeans_scratch.fit_predict(X)
        time_scratch = time.time() - start_time
        
        # Sklearn implementation
        print("\n🔍 Scikit-learn K-Means:")
        start_time = time.time()
        kmeans_sklearn = SklearnKMeans(n_clusters=n_clusters, 
                                      random_state=random_state, 
                                      n_init=10, max_iter=300)
        labels_sklearn = kmeans_sklearn.fit_predict(X)
        time_sklearn = time.time() - start_time
        
        # Compute metrics
        results = {
            'dataset': dataset_name,
            'data_shape': X.shape,
            'n_clusters': n_clusters,
            'scratch': {
                'labels': labels_scratch,
                'centroids': kmeans_scratch.cluster_centers_,
                'inertia': kmeans_scratch.inertia_,
                'n_iter': kmeans_scratch.n_iter_,
                'converged': kmeans_scratch.converged_,
                'time': time_scratch
            },
            'sklearn': {
                'labels': labels_sklearn,
                'centroids': kmeans_sklearn.cluster_centers_,
                'inertia': kmeans_sklearn.inertia_,
                'n_iter': kmeans_sklearn.n_iter_,
                'time': time_sklearn
            }
        }
        
        # Clustering quality metrics
        if len(np.unique(labels_scratch)) > 1:
            results['scratch']['silhouette'] = silhouette_score(X, labels_scratch)
            results['sklearn']['silhouette'] = silhouette_score(X, labels_sklearn)
        
        # If true labels available
        if true_labels is not None:
            results['scratch']['ari'] = adjusted_rand_score(true_labels, labels_scratch)
            results['scratch']['nmi'] = normalized_mutual_info_score(true_labels, labels_scratch)
            results['sklearn']['ari'] = adjusted_rand_score(true_labels, labels_sklearn)
            results['sklearn']['nmi'] = normalized_mutual_info_score(true_labels, labels_sklearn)
        
        # Store results
        self.results[dataset_name] = results
        
        # Print comparison
        print(f"\n📊 Comparison Results:")
        print(f"  {'Metric':<15} {'Our Impl':<12} {'Sklearn':<12} {'Difference':<12}")
        print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*12}")
        print(f"  {'Inertia':<15} {results['scratch']['inertia']:<12.4f} "
              f"{results['sklearn']['inertia']:<12.4f} "
              f"{abs(results['scratch']['inertia'] - results['sklearn']['inertia']):<12.4f}")
        print(f"  {'Iterations':<15} {results['scratch']['n_iter']:<12} "
              f"{results['sklearn']['n_iter']:<12} "
              f"{abs(results['scratch']['n_iter'] - results['sklearn']['n_iter']):<12}")
        print(f"  {'Time (s)':<15} {results['scratch']['time']:<12.4f} "
              f"{results['sklearn']['time']:<12.4f} "
              f"{abs(results['scratch']['time'] - results['sklearn']['time']):<12.4f}")
        
        if 'silhouette' in results['scratch']:
            print(f"  {'Silhouette':<15} {results['scratch']['silhouette']:<12.4f} "
                  f"{results['sklearn']['silhouette']:<12.4f} "
                  f"{abs(results['scratch']['silhouette'] - results['sklearn']['silhouette']):<12.4f}")
        
        if 'ari' in results['scratch']:
            print(f"  {'ARI':<15} {results['scratch']['ari']:<12.4f} "
                  f"{results['sklearn']['ari']:<12.4f} "
                  f"{abs(results['scratch']['ari'] - results['sklearn']['ari']):<12.4f}")
        
        return results
    
    def visualize_comparison(self, dataset_name, save_path=None):
        """
        Visualize clustering results comparison.
        
        Parameters:
        -----------
        dataset_name : str
            Name of dataset to visualize
        save_path : str, optional
            Path to save the plot
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
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: True labels (if available)
        ax1 = axes[0, 0]
        if true_labels is not None:
            scatter = ax1.scatter(X_vis[:, 0], X_vis[:, 1], c=true_labels, 
                                 cmap='viridis', alpha=0.7, s=50)
            ax1.set_title(f'True Labels{title_suffix}')
            plt.colorbar(scatter, ax=ax1)
        else:
            ax1.scatter(X_vis[:, 0], X_vis[:, 1], alpha=0.7, s=50)
            ax1.set_title(f'Data Points{title_suffix}')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Our implementation
        ax2 = axes[0, 1]
        scatter2 = ax2.scatter(X_vis[:, 0], X_vis[:, 1], 
                              c=results['scratch']['labels'], 
                              cmap='viridis', alpha=0.7, s=50)
        
        # Plot centroids (project if needed)
        centroids = results['scratch']['centroids']
        if X.shape[1] > 2:
            centroids_vis = pca.transform(centroids)
        else:
            centroids_vis = centroids
        
        ax2.scatter(centroids_vis[:, 0], centroids_vis[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, label='Centroids')
        ax2.set_title(f'Our Implementation{title_suffix}\n'
                     f'Inertia: {results["scratch"]["inertia"]:.2f}, '
                     f'Iter: {results["scratch"]["n_iter"]}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2)
        
        # Plot 3: Sklearn implementation
        ax3 = axes[1, 0]
        scatter3 = ax3.scatter(X_vis[:, 0], X_vis[:, 1], 
                              c=results['sklearn']['labels'], 
                              cmap='viridis', alpha=0.7, s=50)
        
        centroids_sklearn = results['sklearn']['centroids']
        if X.shape[1] > 2:
            centroids_sklearn_vis = pca.transform(centroids_sklearn)
        else:
            centroids_sklearn_vis = centroids_sklearn
        
        ax3.scatter(centroids_sklearn_vis[:, 0], centroids_sklearn_vis[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, label='Centroids')
        ax3.set_title(f'Scikit-learn Implementation{title_suffix}\n'
                     f'Inertia: {results["sklearn"]["inertia"]:.2f}, '
                     f'Iter: {results["sklearn"]["n_iter"]}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter3, ax=ax3)
        
        # Plot 4: Performance comparison
        ax4 = axes[1, 1]
        metrics = ['Inertia', 'Iterations', 'Time (s)']
        our_values = [results['scratch']['inertia'], 
                     results['scratch']['n_iter'], 
                     results['scratch']['time']]
        sklearn_values = [results['sklearn']['inertia'], 
                         results['sklearn']['n_iter'], 
                         results['sklearn']['time']]
        
        # Normalize values for better visualization
        our_norm = np.array(our_values) / np.array(sklearn_values)
        sklearn_norm = np.ones(len(metrics))
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax4.bar(x - width/2, our_norm, width, label='Our Implementation', alpha=0.7)
        ax4.bar(x + width/2, sklearn_norm, width, label='Scikit-learn', alpha=0.7)
        
        ax4.set_xlabel('Metrics')
        ax4.set_ylabel('Normalized Values (relative to sklearn)')
        ax4.set_title('Performance Comparison\n(Values relative to sklearn)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        
        # Add value annotations
        for i, (our_val, sk_val) in enumerate(zip(our_values, sklearn_values)):
            ax4.annotate(f'{our_val:.3f}', 
                        xy=(i - width/2, our_norm[i]), 
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
            ax4.annotate(f'{sk_val:.3f}', 
                        xy=(i + width/2, sklearn_norm[i]), 
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison visualization saved to {save_path}")
        
        plt.show()
    
    def initialization_comparison(self, dataset_name, n_trials=10, save_path=None):
        """
        Compare different initialization methods.
        
        Parameters:
        -----------
        dataset_name : str
            Name of dataset to use
        n_trials : int
            Number of trials for each initialization method
        save_path : str, optional
            Path to save the plot
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        dataset = self.datasets[dataset_name]
        X = dataset['data']
        n_clusters = dataset['optimal_k']
        
        print(f"Comparing initialization methods on {dataset['name']}")
        
        # Test different initialization methods
        init_methods = ['random', 'k-means++']
        results = {method: {'inertias': [], 'iterations': [], 'times': []} 
                  for method in init_methods}
        
        for init_method in init_methods:
            print(f"\nTesting {init_method} initialization...")
            
            for trial in range(n_trials):
                start_time = time.time()
                kmeans = KMeansScratch(n_clusters=n_clusters, 
                                     init=init_method, 
                                     random_state=trial)
                kmeans.fit(X)
                elapsed_time = time.time() - start_time
                
                results[init_method]['inertias'].append(kmeans.inertia_)
                results[init_method]['iterations'].append(kmeans.n_iter_)
                results[init_method]['times'].append(elapsed_time)
        
        # Visualize results
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Inertia comparison
        ax1 = axes[0]
        inertia_data = [results[method]['inertias'] for method in init_methods]
        bp1 = ax1.boxplot(inertia_data, labels=init_methods, patch_artist=True)
        ax1.set_ylabel('Final Inertia')
        ax1.set_title('Inertia Distribution by Initialization')
        ax1.grid(True, alpha=0.3)
        
        # Color the boxplots
        colors = ['lightblue', 'lightgreen']
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
        
        # Plot 2: Iterations comparison
        ax2 = axes[1]
        iter_data = [results[method]['iterations'] for method in init_methods]
        bp2 = ax2.boxplot(iter_data, labels=init_methods, patch_artist=True)
        ax2.set_ylabel('Iterations to Convergence')
        ax2.set_title('Convergence Speed by Initialization')
        ax2.grid(True, alpha=0.3)
        
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
        
        # Plot 3: Time comparison
        ax3 = axes[2]
        time_data = [results[method]['times'] for method in init_methods]
        bp3 = ax3.boxplot(time_data, labels=init_methods, patch_artist=True)
        ax3.set_ylabel('Computation Time (seconds)')
        ax3.set_title('Runtime by Initialization')
        ax3.grid(True, alpha=0.3)
        
        for patch, color in zip(bp3['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Initialization comparison saved to {save_path}")
        
        plt.show()
        
        # Print statistics
        print(f"\n📊 Initialization Comparison Results (over {n_trials} trials):")
        for method in init_methods:
            print(f"\n{method.upper()} Initialization:")
            print(f"  Inertia: {np.mean(results[method]['inertias']):.4f} ± "
                  f"{np.std(results[method]['inertias']):.4f}")
            print(f"  Iterations: {np.mean(results[method]['iterations']):.1f} ± "
                  f"{np.std(results[method]['iterations']):.1f}")
            print(f"  Time: {np.mean(results[method]['times']):.4f} ± "
                  f"{np.std(results[method]['times']):.4f}")
        
        return results
    
    def comprehensive_analysis(self, save_plots=True):
        """
        Run comprehensive analysis across all datasets.
        
        Parameters:
        -----------
        save_plots : bool
            Whether to save generated plots
        """
        print("🔍 COMPREHENSIVE K-MEANS ANALYSIS")
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
            
            # Visualize if 2D or can be projected to 2D
            if dataset['data'].shape[1] <= 8:  # Reasonable for PCA
                if save_plots:
                    save_path = f"plots/comparison_{dataset_name}.png"
                else:
                    save_path = None
                self.visualize_comparison(dataset_name, save_path)
            
            # Add to summary
            summary_results.append({
                'Dataset': dataset['name'],
                'Our Inertia': results['scratch']['inertia'],
                'Sklearn Inertia': results['sklearn']['inertia'],
                'Our Iterations': results['scratch']['n_iter'],
                'Sklearn Iterations': results['sklearn']['n_iter'],
                'Our Time': results['scratch']['time'],
                'Sklearn Time': results['sklearn']['time']
            })
        
        # Create summary table
        summary_df = pd.DataFrame(summary_results)
        print(f"\n📋 SUMMARY TABLE:")
        print("=" * 80)
        print(summary_df.to_string(index=False, float_format='%.4f'))
        
        # Overall performance analysis
        print(f"\n🎯 OVERALL PERFORMANCE:")
        print("=" * 30)
        
        inertia_diff = summary_df['Our Inertia'] - summary_df['Sklearn Inertia']
        iter_diff = summary_df['Our Iterations'] - summary_df['Sklearn Iterations']
        time_diff = summary_df['Our Time'] - summary_df['Sklearn Time']
        
        print(f"Average inertia difference: {np.mean(inertia_diff):.4f} ± {np.std(inertia_diff):.4f}")
        print(f"Average iteration difference: {np.mean(iter_diff):.1f} ± {np.std(iter_diff):.1f}")
        print(f"Average time difference: {np.mean(time_diff):.4f} ± {np.std(time_diff):.4f}")
        
        return summary_df

def main():
    """
    Main function to run comprehensive K-Means comparison.
    """
    print("🎯 K-MEANS COMPARISON WITH SKLEARN")
    print("=" * 50)
    
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Create comparator
    comparator = KMeansComparator()
    
    # Run comprehensive analysis
    summary = comparator.comprehensive_analysis(save_plots=True)
    
    # Test initialization comparison on blobs dataset
    print(f"\n🔧 Testing Initialization Methods:")
    print("-" * 40)
    comparator.initialization_comparison('blobs', n_trials=10, 
                                       save_path='plots/initialization_comparison.png')
    
    # Test elbow method
    print(f"\n📈 Testing Elbow Method:")
    print("-" * 40)
    analyzer = KMeansAnalyzer()
    
    # Use blobs dataset for elbow method
    blobs_data = comparator.datasets['blobs']['data']
    analyzer.elbow_method(blobs_data, k_range=range(1, 11), random_state=42)
    analyzer.silhouette_analysis(blobs_data, k_range=range(2, 11), random_state=42)
    analyzer.plot_elbow_and_silhouette(save_path='plots/elbow_silhouette_analysis.png')
    
    print("\n✅ COMPREHENSIVE K-MEANS COMPARISON COMPLETE!")
    print("📁 Check the 'plots' folder for generated visualizations.")
    
    return comparator, summary

if __name__ == "__main__":
    main() 