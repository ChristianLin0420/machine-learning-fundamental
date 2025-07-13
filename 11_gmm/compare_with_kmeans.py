"""
GMM vs K-means Comparison - Advanced Analysis
============================================

This module provides comprehensive comparison between Gaussian Mixture Models
and K-means clustering including:
- Side-by-side clustering visualizations
- Decision boundary comparisons
- Clustering quality metrics analysis
- Soft vs hard clustering demonstrations
- Convergence behavior comparison
- Strengths and weaknesses analysis

Mathematical Foundation:
- K-means: Hard assignments, spherical clusters, minimize within-cluster sum of squares
- GMM: Soft assignments, elliptical clusters, maximize likelihood
- Comparison metrics: ARI, Silhouette Score, Calinski-Harabasz Index
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.metrics import adjusted_rand_score, silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse
import warnings
warnings.filterwarnings('ignore')

# Import our custom GMM implementation
from gmm_from_scratch import GaussianMixtureScratch

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ClusteringComparator:
    """
    Comprehensive comparison tool for GMM and K-means clustering.
    
    Features:
    - Multiple clustering algorithms comparison
    - Visual analysis with decision boundaries
    - Quantitative metrics evaluation
    - Convergence behavior analysis
    - Dataset-specific performance insights
    """
    
    def __init__(self, random_state=42):
        """
        Initialize clustering comparator.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.results = {}
        
    def generate_test_datasets(self):
        """
        Generate diverse datasets for clustering comparison.
        
        Returns:
        --------
        dict : Dictionary containing test datasets
        """
        datasets = {}
        
        # 1. Well-separated spherical clusters (K-means friendly)
        print("Generating well-separated spherical clusters...")
        X_spherical, y_spherical = make_blobs(
            n_samples=300, centers=4, cluster_std=1.0, 
            center_box=(-8.0, 8.0), random_state=self.random_state
        )
        datasets['spherical'] = {
            'data': X_spherical, 
            'labels': y_spherical, 
            'name': 'Spherical Clusters',
            'description': 'Well-separated, spherical clusters (K-means friendly)'
        }
        
        # 2. Overlapping elliptical clusters (GMM friendly)
        print("Generating overlapping elliptical clusters...")
        X_elliptical, y_elliptical = make_blobs(
            n_samples=400, centers=3, cluster_std=2.5, 
            center_box=(-6.0, 6.0), random_state=self.random_state
        )
        # Add correlation to make clusters elliptical
        transformation_matrix = np.array([[1.5, 0.5], [0.3, 0.8]])
        X_elliptical = X_elliptical @ transformation_matrix
        datasets['elliptical'] = {
            'data': X_elliptical, 
            'labels': y_elliptical, 
            'name': 'Elliptical Clusters',
            'description': 'Overlapping, elliptical clusters (GMM friendly)'
        }
        
        # 3. Different cluster sizes
        print("Generating clusters with different sizes...")
        # Create clusters with different sample sizes
        centers = np.array([[-3, -3], [0, 0], [3, 3]])
        cluster_sizes = [100, 200, 150]
        X_sizes = []
        y_sizes = []
        
        for i, (center, size) in enumerate(zip(centers, cluster_sizes)):
            cluster_data = np.random.multivariate_normal(
                center, [[1.5, 0.3], [0.3, 1.5]], size
            )
            X_sizes.append(cluster_data)
            y_sizes.extend([i] * size)
        
        X_sizes = np.vstack(X_sizes)
        y_sizes = np.array(y_sizes)
        
        datasets['different_sizes'] = {
            'data': X_sizes, 
            'labels': y_sizes, 
            'name': 'Different Cluster Sizes',
            'description': 'Clusters with varying sample sizes'
        }
        
        # 4. Non-Gaussian clusters (challenging for both)
        print("Generating non-Gaussian clusters...")
        X_moons, y_moons = make_moons(n_samples=300, noise=0.1, random_state=self.random_state)
        datasets['moons'] = {
            'data': X_moons, 
            'labels': y_moons, 
            'name': 'Moon Shapes',
            'description': 'Non-Gaussian, crescent-shaped clusters'
        }
        
        # 5. Concentric circles
        print("Generating concentric circles...")
        X_circles, y_circles = make_circles(n_samples=300, noise=0.05, factor=0.6, random_state=self.random_state)
        datasets['circles'] = {
            'data': X_circles, 
            'labels': y_circles, 
            'name': 'Concentric Circles',
            'description': 'Nested circular clusters'
        }
        
        return datasets
    
    def fit_clustering_algorithms(self, X, n_clusters, dataset_name):
        """
        Fit multiple clustering algorithms to the data.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        n_clusters : int
            Number of clusters
        dataset_name : str
            Name of the dataset for identification
            
        Returns:
        --------
        dict : Results from different algorithms
        """
        results = {}
        
        print(f"Fitting clustering algorithms for {dataset_name}...")
        
        # 1. K-means
        print("  Fitting K-means...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        kmeans_labels = kmeans.fit_predict(X)
        
        results['kmeans'] = {
            'model': kmeans,
            'labels': kmeans_labels,
            'centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'n_iter': kmeans.n_iter_
        }
        
        # 2. Our custom GMM
        print("  Fitting custom GMM...")
        gmm_custom = GaussianMixtureScratch(
            n_components=n_clusters, 
            max_iter=100, 
            tol=1e-6,
            init_method='kmeans',
            random_state=self.random_state
        )
        gmm_custom.fit(X)
        gmm_custom_labels = gmm_custom.predict(X)
        gmm_custom_proba = gmm_custom.predict_proba(X)
        
        results['gmm_custom'] = {
            'model': gmm_custom,
            'labels': gmm_custom_labels,
            'probabilities': gmm_custom_proba,
            'means': gmm_custom.means_,
            'covariances': gmm_custom.covariances_,
            'weights': gmm_custom.weights_,
            'log_likelihood': gmm_custom.log_likelihood_history_[-1],
            'n_iter': gmm_custom.n_iter_,
            'converged': gmm_custom.converged_
        }
        
        # 3. Scikit-learn GMM for comparison
        print("  Fitting sklearn GMM...")
        gmm_sklearn = GaussianMixture(
            n_components=n_clusters, 
            random_state=self.random_state,
            max_iter=100,
            tol=1e-6
        )
        gmm_sklearn.fit(X)
        gmm_sklearn_labels = gmm_sklearn.predict(X)
        gmm_sklearn_proba = gmm_sklearn.predict_proba(X)
        
        results['gmm_sklearn'] = {
            'model': gmm_sklearn,
            'labels': gmm_sklearn_labels,
            'probabilities': gmm_sklearn_proba,
            'means': gmm_sklearn.means_,
            'covariances': gmm_sklearn.covariances_,
            'weights': gmm_sklearn.weights_,
            'log_likelihood': gmm_sklearn.score(X) * len(X),  # Convert to total log-likelihood
            'converged': gmm_sklearn.converged_
        }
        
        return results
    
    def compute_clustering_metrics(self, X, true_labels, predicted_labels):
        """
        Compute comprehensive clustering evaluation metrics.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        true_labels : np.ndarray
            Ground truth labels
        predicted_labels : np.ndarray
            Predicted cluster labels
            
        Returns:
        --------
        dict : Dictionary of metrics
        """
        metrics = {}
        
        # Adjusted Rand Index (measures similarity to ground truth)
        metrics['ari'] = adjusted_rand_score(true_labels, predicted_labels)
        
        # Silhouette Score (measures cluster cohesion and separation)
        metrics['silhouette'] = silhouette_score(X, predicted_labels)
        
        # Calinski-Harabasz Index (ratio of between-cluster to within-cluster dispersion)
        metrics['calinski_harabasz'] = calinski_harabasz_score(X, predicted_labels)
        
        return metrics
    
    def plot_clustering_comparison(self, X, true_labels, results, dataset_name, save_path=None):
        """
        Create comprehensive clustering comparison visualization.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data (2D)
        true_labels : np.ndarray
            Ground truth labels
        results : dict
            Results from different clustering algorithms
        dataset_name : str
            Name of the dataset
        save_path : str, optional
            Path to save the plot
        """
        if X.shape[1] != 2:
            print(f"Skipping visualization for {dataset_name} (not 2D)")
            return
        
        # Create subplot layout
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Original data with true labels
        ax = axes[0, 0]
        scatter = ax.scatter(X[:, 0], X[:, 1], c=true_labels, cmap='Set1', alpha=0.7, s=50)
        ax.set_title('True Labels')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax)
        
        # Plot 2: K-means results
        ax = axes[0, 1]
        kmeans_labels = results['kmeans']['labels']
        scatter = ax.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.7, s=50)
        centers = results['kmeans']['centers']
        ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidths=3)
        ax.set_title('K-means Clustering')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax)
        
        # Plot 3: Custom GMM results
        ax = axes[0, 2]
        gmm_labels = results['gmm_custom']['labels']
        scatter = ax.scatter(X[:, 0], X[:, 1], c=gmm_labels, cmap='viridis', alpha=0.7, s=50)
        means = results['gmm_custom']['means']
        ax.scatter(means[:, 0], means[:, 1], c='red', marker='x', s=200, linewidths=3)
        
        # Add Gaussian contours
        self._plot_gmm_contours(ax, results['gmm_custom'])
        
        ax.set_title('Custom GMM Clustering')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax)
        
        # Plot 4: GMM uncertainty (entropy)
        ax = axes[1, 0]
        probabilities = results['gmm_custom']['probabilities']
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)
        scatter = ax.scatter(X[:, 0], X[:, 1], c=entropy, cmap='plasma', alpha=0.7, s=50)
        ax.scatter(means[:, 0], means[:, 1], c='red', marker='x', s=200, linewidths=3)
        ax.set_title('GMM Uncertainty (Entropy)')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Entropy')
        
        # Plot 5: Sklearn GMM comparison
        ax = axes[1, 1]
        sklearn_labels = results['gmm_sklearn']['labels']
        scatter = ax.scatter(X[:, 0], X[:, 1], c=sklearn_labels, cmap='viridis', alpha=0.7, s=50)
        sklearn_means = results['gmm_sklearn']['means']
        ax.scatter(sklearn_means[:, 0], sklearn_means[:, 1], c='red', marker='x', s=200, linewidths=3)
        ax.set_title('Sklearn GMM Clustering')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax)
        
        # Plot 6: Metrics comparison
        ax = axes[1, 2]
        algorithms = ['K-means', 'Custom GMM', 'Sklearn GMM']
        
        # Compute metrics for each algorithm
        metrics_data = {}
        for alg_name, result_key in zip(algorithms, ['kmeans', 'gmm_custom', 'gmm_sklearn']):
            labels = results[result_key]['labels']
            metrics = self.compute_clustering_metrics(X, true_labels, labels)
            for metric_name, value in metrics.items():
                if metric_name not in metrics_data:
                    metrics_data[metric_name] = []
                metrics_data[metric_name].append(value)
        
        # Plot metrics
        x_pos = np.arange(len(algorithms))
        width = 0.25
        
        for i, (metric_name, values) in enumerate(metrics_data.items()):
            ax.bar(x_pos + i*width, values, width, label=metric_name.upper(), alpha=0.7)
        
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Score')
        ax.set_title('Clustering Metrics Comparison')
        ax.set_xticks(x_pos + width)
        ax.set_xticklabels(algorithms)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Clustering Comparison: {dataset_name}', fontsize=16, y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        plt.show()
    
    def _plot_gmm_contours(self, ax, gmm_result, n_std=2):
        """
        Plot Gaussian mixture contours.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Axes to plot on
        gmm_result : dict
            GMM results containing means and covariances
        n_std : float
            Number of standard deviations for contours
        """
        means = gmm_result['means']
        covariances = gmm_result['covariances']
        n_components = len(means)
        
        colors = plt.cm.Set1(np.linspace(0, 1, n_components))
        
        for k in range(n_components):
            mean = means[k]
            cov = covariances[k]
            
            # Compute eigenvalues and eigenvectors
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            
            # Calculate ellipse parameters
            angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
            width = 2 * n_std * np.sqrt(eigenvals[0])
            height = 2 * n_std * np.sqrt(eigenvals[1])
            
            # Create ellipse
            ellipse = Ellipse(mean, width, height, angle=angle, 
                            facecolor='none', edgecolor=colors[k], 
                            linewidth=2, alpha=0.7)
            ax.add_patch(ellipse)
    
    def analyze_convergence_behavior(self, results_dict, save_path=None):
        """
        Analyze and visualize convergence behavior of different algorithms.
        
        Parameters:
        -----------
        results_dict : dict
            Results from multiple datasets
        save_path : str, optional
            Path to save the plot
        """
        print("Analyzing convergence behavior...")
        
        # Extract convergence data
        dataset_names = []
        kmeans_iters = []
        gmm_custom_iters = []
        gmm_custom_converged = []
        
        for dataset_name, results in results_dict.items():
            dataset_names.append(dataset_name)
            kmeans_iters.append(results['kmeans']['n_iter'])
            gmm_custom_iters.append(results['gmm_custom']['n_iter'])
            gmm_custom_converged.append(results['gmm_custom']['converged'])
        
        # Create convergence comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Iteration counts
        x_pos = np.arange(len(dataset_names))
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, kmeans_iters, width, label='K-means', alpha=0.7)
        bars2 = ax1.bar(x_pos + width/2, gmm_custom_iters, width, label='Custom GMM', alpha=0.7)
        
        ax1.set_xlabel('Dataset')
        ax1.set_ylabel('Iterations to Convergence')
        ax1.set_title('Convergence Speed Comparison')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([name.replace('_', ' ').title() for name in dataset_names], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom')
        
        # Plot 2: Convergence success rate
        convergence_rates = {
            'K-means': 1.0,  # K-means always converges (or reaches max_iter)
            'Custom GMM': np.mean(gmm_custom_converged)
        }
        
        algorithms = list(convergence_rates.keys())
        rates = list(convergence_rates.values())
        
        bars = ax2.bar(algorithms, rates, alpha=0.7, color=['skyblue', 'lightcoral'])
        ax2.set_ylabel('Convergence Rate')
        ax2.set_title('Algorithm Convergence Success Rate')
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3)
        
        # Add percentage labels
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax2.annotate(f'{rate:.1%}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Convergence analysis plot saved to {save_path}")
        
        plt.show()
    
    def create_metrics_summary(self, results_dict, save_path=None):
        """
        Create comprehensive metrics summary across all datasets.
        
        Parameters:
        -----------
        results_dict : dict
            Results from multiple datasets
        save_path : str, optional
            Path to save the plot
        """
        print("Creating comprehensive metrics summary...")
        
        # Collect all metrics
        datasets = list(results_dict.keys())
        algorithms = ['K-means', 'Custom GMM', 'Sklearn GMM']
        algorithm_keys = ['kmeans', 'gmm_custom', 'gmm_sklearn']
        metrics = ['ARI', 'Silhouette', 'Calinski-Harabasz']
        
        # Initialize data structure
        metrics_data = {metric: {alg: [] for alg in algorithms} for metric in metrics}
        
        # Collect metrics for each dataset and algorithm
        for dataset_name, dataset_results in results_dict.items():
            X = dataset_results['data']
            true_labels = dataset_results['true_labels']
            
            for alg_name, alg_key in zip(algorithms, algorithm_keys):
                predicted_labels = dataset_results[alg_key]['labels']
                computed_metrics = self.compute_clustering_metrics(X, true_labels, predicted_labels)
                
                metrics_data['ARI'][alg_name].append(computed_metrics['ari'])
                metrics_data['Silhouette'][alg_name].append(computed_metrics['silhouette'])
                metrics_data['Calinski-Harabasz'][alg_name].append(computed_metrics['calinski_harabasz'])
        
        # Create comprehensive metrics plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Prepare data for grouped bar chart
            x_pos = np.arange(len(datasets))
            width = 0.25
            
            for j, alg in enumerate(algorithms):
                values = metrics_data[metric][alg]
                ax.bar(x_pos + j*width, values, width, label=alg, alpha=0.7)
            
            ax.set_xlabel('Dataset')
            ax.set_ylabel(f'{metric} Score')
            ax.set_title(f'{metric} Comparison Across Datasets')
            ax.set_xticks(x_pos + width)
            ax.set_xticklabels([name.replace('_', ' ').title() for name in datasets], rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics summary plot saved to {save_path}")
        
        plt.show()
        
        # Print summary statistics
        print("\n" + "="*70)
        print("CLUSTERING PERFORMANCE SUMMARY")
        print("="*70)
        
        for metric in metrics:
            print(f"\n{metric} Scores:")
            print("-" * 40)
            for alg in algorithms:
                values = metrics_data[metric][alg]
                mean_score = np.mean(values)
                std_score = np.std(values)
                print(f"{alg:15}: {mean_score:.4f} ± {std_score:.4f}")
    
    def run_comprehensive_analysis(self):
        """
        Run comprehensive clustering comparison analysis.
        """
        print("🔄 STARTING COMPREHENSIVE GMM vs K-MEANS ANALYSIS")
        print("=" * 80)
        
        # Create plots directory
        import os
        os.makedirs('plots', exist_ok=True)
        
        # Generate test datasets
        datasets = self.generate_test_datasets()
        
        # Store all results
        all_results = {}
        
        # Analyze each dataset
        for dataset_name, dataset_info in datasets.items():
            print(f"\n{'='*50}")
            print(f"ANALYZING DATASET: {dataset_info['name'].upper()}")
            print(f"Description: {dataset_info['description']}")
            print(f"{'='*50}")
            
            X = dataset_info['data']
            true_labels = dataset_info['labels']
            n_clusters = len(np.unique(true_labels))
            
            print(f"Dataset shape: {X.shape}")
            print(f"Number of clusters: {n_clusters}")
            
            # Fit clustering algorithms
            results = self.fit_clustering_algorithms(X, n_clusters, dataset_name)
            
            # Add dataset info to results
            results['data'] = X
            results['true_labels'] = true_labels
            results['dataset_info'] = dataset_info
            
            # Store results
            all_results[dataset_name] = results
            
            # Create comparison visualization
            self.plot_clustering_comparison(
                X, true_labels, results, dataset_info['name'],
                f'plots/clustering_comparison_{dataset_name}.png'
            )
            
            # Print performance summary for this dataset
            print(f"\nPerformance Summary for {dataset_info['name']}:")
            print("-" * 50)
            
            for alg_name, alg_key in zip(['K-means', 'Custom GMM', 'Sklearn GMM'], 
                                       ['kmeans', 'gmm_custom', 'gmm_sklearn']):
                labels = results[alg_key]['labels']
                metrics = self.compute_clustering_metrics(X, true_labels, labels)
                
                print(f"{alg_name:12}: ARI={metrics['ari']:.4f}, "
                      f"Silhouette={metrics['silhouette']:.4f}, "
                      f"Calinski-Harabasz={metrics['calinski_harabasz']:.2f}")
        
        # Generate comprehensive analysis
        print(f"\n{'='*70}")
        print("GENERATING COMPREHENSIVE ANALYSIS")
        print(f"{'='*70}")
        
        # Convergence analysis
        self.analyze_convergence_behavior(all_results, 'plots/convergence_analysis.png')
        
        # Metrics summary
        self.create_metrics_summary(all_results, 'plots/metrics_summary.png')
        
        print("\n✅ COMPREHENSIVE CLUSTERING ANALYSIS COMPLETE!")
        print("📁 Check the 'plots' folder for generated visualizations.")
        print(f"\n📋 SUMMARY:")
        print(f"• Analyzed {len(datasets)} different dataset types")
        print(f"• Compared 3 clustering algorithms")
        print(f"• Generated {len(datasets) + 2} comprehensive visualization files")
        print("• Demonstrated strengths and weaknesses of each approach")
        
        return all_results

def main():
    """
    Main function to run GMM vs K-means comparison.
    """
    print("⚖️ GMM vs K-MEANS CLUSTERING COMPARISON")
    print("=" * 80)
    
    # Create comparator and run analysis
    comparator = ClusteringComparator(random_state=42)
    results = comparator.run_comprehensive_analysis()
    
    # Additional insights
    print("\n🔍 KEY INSIGHTS:")
    print("-" * 30)
    print("• K-means: Best for spherical, well-separated clusters")
    print("• GMM: Better for overlapping, elliptical clusters")
    print("• GMM provides uncertainty estimates (soft clustering)")
    print("• K-means is computationally faster and simpler")
    print("• GMM handles clusters of different sizes better")
    print("• Both struggle with non-Gaussian cluster shapes")

if __name__ == "__main__":
    main() 