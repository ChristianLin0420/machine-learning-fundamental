"""
K-Means Clustering from Scratch - Advanced Implementation
=======================================================

This module implements a comprehensive K-Means clustering algorithm from scratch,
covering all fundamental concepts and advanced techniques:

Core Algorithm:
- Assignment Step: Assign points to closest centroids
- Update Step: Move centroids to cluster means
- Convergence: Monitor centroid movement and iterations

Advanced Features:
- K-Means++ initialization for better convergence
- Multiple distance metrics (Euclidean, Manhattan, Cosine)
- Convergence monitoring and visualization
- Comprehensive evaluation metrics
- Comparison with scikit-learn implementation

Mathematical Foundation:
- Objective: Minimize within-cluster sum of squares (WCSS)
- WCSS = Σ_i Σ_{x∈C_i} ||x - μ_i||²
- EM-like algorithm: E-step (assignment) + M-step (update)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, load_iris
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import List, Tuple, Dict, Union, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class KMeansScratch:
    """
    K-Means clustering implementation from scratch.
    
    This class provides a complete K-Means implementation with:
    - Multiple initialization methods (random, K-Means++)
    - Various distance metrics
    - Convergence monitoring
    - Comprehensive evaluation metrics
    - Visualization capabilities
    
    Parameters:
    -----------
    n_clusters : int
        Number of clusters to form
    max_iter : int
        Maximum number of iterations
    tol : float
        Tolerance for convergence (centroid movement threshold)
    init : str
        Initialization method ('random', 'k-means++')
    distance_metric : str
        Distance metric ('euclidean', 'manhattan', 'cosine')
    random_state : int
        Random seed for reproducibility
    """
    
    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, 
                 init='k-means++', distance_metric='euclidean', random_state=42):
        """
        Initialize K-Means clustering algorithm.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
        init : str
            Initialization method
        distance_metric : str
            Distance metric to use
        random_state : int
            Random seed
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.distance_metric = distance_metric
        self.random_state = random_state
        
        # Results
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
        self.converged_ = False
        
        # Training history
        self.centroid_history_ = []
        self.inertia_history_ = []
        self.assignment_history_ = []
        
        np.random.seed(random_state)
    
    def _compute_distance(self, X, centroids):
        """
        Compute distances between points and centroids.
        
        Parameters:
        -----------
        X : np.ndarray
            Data points (n_samples, n_features)
        centroids : np.ndarray
            Cluster centroids (n_clusters, n_features)
            
        Returns:
        --------
        np.ndarray : Distance matrix (n_samples, n_clusters)
        """
        n_samples, n_features = X.shape
        n_clusters = centroids.shape[0]
        distances = np.zeros((n_samples, n_clusters))
        
        for i, centroid in enumerate(centroids):
            if self.distance_metric == 'euclidean':
                distances[:, i] = np.sqrt(np.sum((X - centroid) ** 2, axis=1))
            elif self.distance_metric == 'manhattan':
                distances[:, i] = np.sum(np.abs(X - centroid), axis=1)
            elif self.distance_metric == 'cosine':
                # Cosine distance = 1 - cosine similarity
                dot_product = np.dot(X, centroid)
                norms = np.linalg.norm(X, axis=1) * np.linalg.norm(centroid)
                # Handle zero norm case
                norms = np.where(norms == 0, 1e-10, norms)
                cosine_sim = dot_product / norms
                distances[:, i] = 1 - cosine_sim
            else:
                raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
        
        return distances
    
    def _initialize_centroids(self, X):
        """
        Initialize cluster centroids using specified method.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data (n_samples, n_features)
            
        Returns:
        --------
        np.ndarray : Initial centroids (n_clusters, n_features)
        """
        n_samples, n_features = X.shape
        
        if self.init == 'random':
            # Random initialization within data bounds
            min_vals = np.min(X, axis=0)
            max_vals = np.max(X, axis=0)
            centroids = np.random.uniform(min_vals, max_vals, (self.n_clusters, n_features))
            
        elif self.init == 'k-means++':
            # K-Means++ initialization
            centroids = self._kmeans_plus_plus_init(X)
            
        else:
            raise ValueError(f"Unsupported initialization method: {self.init}")
        
        return centroids
    
    def _kmeans_plus_plus_init(self, X):
        """
        K-Means++ initialization for better initial centroids.
        
        Algorithm:
        1. Choose first centroid uniformly at random
        2. For each subsequent centroid:
           - Compute distances to nearest existing centroid
           - Choose next centroid with probability proportional to squared distance
        
        Parameters:
        -----------
        X : np.ndarray
            Input data (n_samples, n_features)
            
        Returns:
        --------
        np.ndarray : K-Means++ initialized centroids
        """
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))
        
        # Choose first centroid randomly
        centroids[0] = X[np.random.randint(n_samples)]
        
        for c in range(1, self.n_clusters):
            # Compute distances to nearest centroid for each point
            distances = np.array([min([np.sum((x - centroid)**2) for centroid in centroids[:c]]) 
                                for x in X])
            
            # Choose next centroid with probability proportional to squared distance
            probabilities = distances / distances.sum()
            cumulative_prob = probabilities.cumsum()
            r = np.random.rand()
            
            # Find the first point where cumulative probability exceeds r
            for i, prob in enumerate(cumulative_prob):
                if r < prob:
                    centroids[c] = X[i]
                    break
        
        return centroids
    
    def _assign_clusters(self, X, centroids):
        """
        Assign each point to the closest centroid.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data (n_samples, n_features)
        centroids : np.ndarray
            Current centroids (n_clusters, n_features)
            
        Returns:
        --------
        np.ndarray : Cluster assignments (n_samples,)
        """
        distances = self._compute_distance(X, centroids)
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X, labels):
        """
        Update centroids to the mean of assigned points.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data (n_samples, n_features)
        labels : np.ndarray
            Current cluster assignments (n_samples,)
            
        Returns:
        --------
        np.ndarray : Updated centroids (n_clusters, n_features)
        """
        n_features = X.shape[1]
        centroids = np.zeros((self.n_clusters, n_features))
        
        for k in range(self.n_clusters):
            # Find points assigned to cluster k
            cluster_points = X[labels == k]
            
            if len(cluster_points) > 0:
                # Update centroid to mean of assigned points
                centroids[k] = np.mean(cluster_points, axis=0)
            else:
                # Handle empty cluster - reinitialize randomly
                centroids[k] = X[np.random.randint(len(X))]
                print(f"Warning: Empty cluster {k} detected. Reinitializing randomly.")
        
        return centroids
    
    def _compute_inertia(self, X, labels, centroids):
        """
        Compute within-cluster sum of squares (inertia/WCSS).
        
        Parameters:
        -----------
        X : np.ndarray
            Input data (n_samples, n_features)
        labels : np.ndarray
            Cluster assignments (n_samples,)
        centroids : np.ndarray
            Cluster centroids (n_clusters, n_features)
            
        Returns:
        --------
        float : Total within-cluster sum of squares
        """
        inertia = 0.0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centroids[k]) ** 2)
        return inertia
    
    def _has_converged(self, old_centroids, new_centroids):
        """
        Check if algorithm has converged based on centroid movement.
        
        Parameters:
        -----------
        old_centroids : np.ndarray
            Previous centroids
        new_centroids : np.ndarray
            Current centroids
            
        Returns:
        --------
        bool : True if converged, False otherwise
        """
        centroid_movement = np.sqrt(np.sum((old_centroids - new_centroids) ** 2))
        return centroid_movement < self.tol
    
    def fit(self, X):
        """
        Fit K-Means clustering to data.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data (n_samples, n_features)
            
        Returns:
        --------
        self : Returns the instance itself
        """
        n_samples, n_features = X.shape
        
        if self.n_clusters > n_samples:
            raise ValueError(f"n_clusters ({self.n_clusters}) cannot be larger than "
                           f"n_samples ({n_samples})")
        
        print(f"Fitting K-Means with {self.n_clusters} clusters...")
        print(f"  Data shape: {X.shape}")
        print(f"  Initialization: {self.init}")
        print(f"  Distance metric: {self.distance_metric}")
        print(f"  Max iterations: {self.max_iter}")
        print(f"  Tolerance: {self.tol}")
        
        # Initialize centroids
        centroids = self._initialize_centroids(X)
        
        # Initialize tracking
        self.centroid_history_ = [centroids.copy()]
        self.inertia_history_ = []
        self.assignment_history_ = []
        
        # Main K-Means loop
        for iteration in range(self.max_iter):
            # Assignment step (E-step-like)
            labels = self._assign_clusters(X, centroids)
            
            # Compute current inertia
            inertia = self._compute_inertia(X, labels, centroids)
            
            # Store history
            self.assignment_history_.append(labels.copy())
            self.inertia_history_.append(inertia)
            
            # Update step (M-step-like)
            new_centroids = self._update_centroids(X, labels)
            
            # Check for convergence
            if self._has_converged(centroids, new_centroids):
                print(f"Converged after {iteration + 1} iterations")
                self.converged_ = True
                break
            
            # Update centroids
            centroids = new_centroids
            self.centroid_history_.append(centroids.copy())
            
            # Progress reporting
            if iteration % 50 == 0:
                print(f"  Iteration {iteration}: Inertia = {inertia:.4f}")
        
        # Store final results
        self.n_iter_ = iteration + 1
        self.cluster_centers_ = centroids
        self.labels_ = labels
        self.inertia_ = inertia
        
        if not self.converged_:
            print(f"Maximum iterations ({self.max_iter}) reached without convergence")
        
        print(f"Final inertia: {self.inertia_:.4f}")
        
        return self
    
    def predict(self, X):
        """
        Predict cluster labels for new data.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data (n_samples, n_features)
            
        Returns:
        --------
        np.ndarray : Predicted cluster labels (n_samples,)
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        return self._assign_clusters(X, self.cluster_centers_)
    
    def fit_predict(self, X):
        """
        Fit the model and predict cluster labels.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data (n_samples, n_features)
            
        Returns:
        --------
        np.ndarray : Cluster labels (n_samples,)
        """
        return self.fit(X).labels_
    
    def score(self, X):
        """
        Compute negative inertia (for consistency with sklearn).
        
        Parameters:
        -----------
        X : np.ndarray
            Input data (n_samples, n_features)
            
        Returns:
        --------
        float : Negative inertia
        """
        labels = self.predict(X)
        inertia = self._compute_inertia(X, labels, self.cluster_centers_)
        return -inertia
    
    def plot_convergence(self, save_path=None):
        """
        Plot convergence history.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        """
        if not self.inertia_history_:
            raise ValueError("No training history available. Fit the model first.")
        
        plt.figure(figsize=(12, 4))
        
        # Plot 1: Inertia convergence
        plt.subplot(1, 2, 1)
        plt.plot(self.inertia_history_, 'b-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('Iteration')
        plt.ylabel('Inertia (WCSS)')
        plt.title('K-Means Convergence')
        plt.grid(True, alpha=0.3)
        
        # Add convergence indicator
        if self.converged_:
            plt.axvline(x=len(self.inertia_history_)-1, color='red', linestyle='--', 
                       label=f'Converged at iteration {self.n_iter_}')
            plt.legend()
        
        # Plot 2: Centroid movement
        plt.subplot(1, 2, 2)
        if len(self.centroid_history_) > 1:
            movements = []
            for i in range(1, len(self.centroid_history_)):
                movement = np.sqrt(np.sum((self.centroid_history_[i] - 
                                         self.centroid_history_[i-1]) ** 2))
                movements.append(movement)
            
            plt.plot(movements, 'g-', linewidth=2, marker='s', markersize=4)
            plt.axhline(y=self.tol, color='red', linestyle='--', 
                       label=f'Tolerance = {self.tol}')
            plt.xlabel('Iteration')
            plt.ylabel('Centroid Movement')
            plt.title('Centroid Movement Over Time')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Convergence plot saved to {save_path}")
        
        plt.show()

class KMeansAnalyzer:
    """
    Comprehensive analyzer for K-Means clustering results.
    
    This class provides tools for analyzing K-Means performance,
    including elbow method, silhouette analysis, and comparison studies.
    """
    
    def __init__(self):
        """Initialize the K-Means analyzer."""
        self.results = {}
    
    def elbow_method(self, X, k_range=range(1, 11), **kmeans_kwargs):
        """
        Perform elbow method analysis to find optimal number of clusters.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        k_range : range
            Range of k values to test
        **kmeans_kwargs : dict
            Additional arguments for KMeansScratch
            
        Returns:
        --------
        dict : Results containing k values and inertias
        """
        print("Performing Elbow Method Analysis...")
        
        inertias = []
        k_values = list(k_range)
        
        for k in k_values:
            print(f"  Testing k = {k}")
            kmeans = KMeansScratch(n_clusters=k, **kmeans_kwargs)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        results = {
            'k_values': k_values,
            'inertias': inertias
        }
        
        self.results['elbow'] = results
        return results
    
    def silhouette_analysis(self, X, k_range=range(2, 11), **kmeans_kwargs):
        """
        Perform silhouette analysis to evaluate cluster quality.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        k_range : range
            Range of k values to test
        **kmeans_kwargs : dict
            Additional arguments for KMeansScratch
            
        Returns:
        --------
        dict : Results containing k values and silhouette scores
        """
        print("Performing Silhouette Analysis...")
        
        silhouette_scores = []
        k_values = list(k_range)
        
        for k in k_values:
            print(f"  Testing k = {k}")
            kmeans = KMeansScratch(n_clusters=k, **kmeans_kwargs)
            labels = kmeans.fit_predict(X)
            
            if len(np.unique(labels)) > 1:  # Need at least 2 clusters for silhouette
                score = silhouette_score(X, labels)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(0)
        
        results = {
            'k_values': k_values,
            'silhouette_scores': silhouette_scores
        }
        
        self.results['silhouette'] = results
        return results
    
    def plot_elbow_and_silhouette(self, save_path=None):
        """
        Plot elbow method and silhouette analysis results.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        """
        if 'elbow' not in self.results or 'silhouette' not in self.results:
            raise ValueError("Run elbow_method and silhouette_analysis first")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Elbow plot
        ax1 = axes[0]
        elbow_data = self.results['elbow']
        ax1.plot(elbow_data['k_values'], elbow_data['inertias'], 
                'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia (WCSS)')
        ax1.set_title('Elbow Method for Optimal k')
        ax1.grid(True, alpha=0.3)
        
        # Add elbow detection (simple method)
        inertias = np.array(elbow_data['inertias'])
        if len(inertias) > 2:
            # Compute second derivative to find elbow
            second_deriv = np.gradient(np.gradient(inertias))
            elbow_idx = np.argmax(second_deriv[1:-1]) + 1  # Avoid endpoints
            elbow_k = elbow_data['k_values'][elbow_idx]
            ax1.axvline(x=elbow_k, color='red', linestyle='--', 
                       label=f'Suggested k = {elbow_k}')
            ax1.legend()
        
        # Silhouette plot
        ax2 = axes[1]
        silhouette_data = self.results['silhouette']
        ax2.plot(silhouette_data['k_values'], silhouette_data['silhouette_scores'], 
                'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Analysis for Optimal k')
        ax2.grid(True, alpha=0.3)
        
        # Find best silhouette score
        best_idx = np.argmax(silhouette_data['silhouette_scores'])
        best_k = silhouette_data['k_values'][best_idx]
        best_score = silhouette_data['silhouette_scores'][best_idx]
        ax2.axvline(x=best_k, color='red', linestyle='--', 
                   label=f'Best k = {best_k} (score = {best_score:.3f})')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Elbow and silhouette analysis saved to {save_path}")
        
        plt.show()
        
        return {
            'elbow_suggested_k': elbow_k if 'elbow_k' in locals() else None,
            'silhouette_best_k': best_k,
            'silhouette_best_score': best_score
        }

def generate_sample_datasets():
    """
    Generate sample datasets for K-Means testing.
    
    Returns:
    --------
    dict : Dictionary containing different datasets
    """
    datasets = {}
    
    # Dataset 1: Well-separated blobs
    X_blobs, y_blobs = make_blobs(n_samples=300, centers=4, n_features=2, 
                                  random_state=42, cluster_std=1.0)
    datasets['blobs'] = {'data': X_blobs, 'labels': y_blobs, 'name': 'Well-separated Blobs'}
    
    # Dataset 2: Closer blobs
    X_close, y_close = make_blobs(n_samples=300, centers=3, n_features=2, 
                                  random_state=42, cluster_std=2.0)
    datasets['close_blobs'] = {'data': X_close, 'labels': y_close, 'name': 'Close Blobs'}
    
    # Dataset 3: Iris dataset
    iris = load_iris()
    datasets['iris'] = {'data': iris.data, 'labels': iris.target, 'name': 'Iris Dataset'}
    
    # Dataset 4: High-dimensional data (for testing)
    X_high, y_high = make_blobs(n_samples=200, centers=3, n_features=10, 
                                random_state=42, cluster_std=1.5)
    datasets['high_dim'] = {'data': X_high, 'labels': y_high, 'name': 'High-Dimensional Data'}
    
    return datasets

def main():
    """
    Main function to demonstrate K-Means implementation.
    """
    print("🎯 K-MEANS CLUSTERING FROM SCRATCH")
    print("=" * 50)
    
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Generate sample datasets
    datasets = generate_sample_datasets()
    
    # Test with different datasets
    for name, dataset in datasets.items():
        if name == 'high_dim':  # Skip high-dimensional for basic demo
            continue
            
        print(f"\n📊 Testing with {dataset['name']}")
        print("-" * 40)
        
        X = dataset['data']
        true_labels = dataset['labels']
        
        # For 2D data, determine appropriate number of clusters
        if X.shape[1] == 2:
            n_clusters = len(np.unique(true_labels))
        else:
            n_clusters = 3  # Default for higher dimensions
        
        # Test our implementation
        print(f"\n🔧 Our K-Means Implementation:")
        kmeans_scratch = KMeansScratch(n_clusters=n_clusters, random_state=42)
        labels_scratch = kmeans_scratch.fit_predict(X)
        
        # Test sklearn implementation for comparison
        print(f"\n🔍 Scikit-learn K-Means:")
        kmeans_sklearn = SklearnKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels_sklearn = kmeans_sklearn.fit_predict(X)
        
        # Compare results
        print(f"\n📈 Comparison Results:")
        print(f"  Our inertia: {kmeans_scratch.inertia_:.4f}")
        print(f"  Sklearn inertia: {kmeans_sklearn.inertia_:.4f}")
        print(f"  Our iterations: {kmeans_scratch.n_iter_}")
        print(f"  Sklearn iterations: {kmeans_sklearn.n_iter_}")
        
        # Evaluation metrics
        if len(np.unique(labels_scratch)) > 1:
            sil_scratch = silhouette_score(X, labels_scratch)
            sil_sklearn = silhouette_score(X, labels_sklearn)
            print(f"  Our silhouette: {sil_scratch:.4f}")
            print(f"  Sklearn silhouette: {sil_sklearn:.4f}")
        
        # If we have true labels, compute ARI
        if true_labels is not None:
            ari_scratch = adjusted_rand_score(true_labels, labels_scratch)
            ari_sklearn = adjusted_rand_score(true_labels, labels_sklearn)
            print(f"  Our ARI: {ari_scratch:.4f}")
            print(f"  Sklearn ARI: {ari_sklearn:.4f}")
    
    print("\n✅ K-MEANS IMPLEMENTATION COMPLETE!")
    print("📁 Check the implementation for comprehensive K-Means algorithms.")
    
    return datasets

if __name__ == "__main__":
    main() 