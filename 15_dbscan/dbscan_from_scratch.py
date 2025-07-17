"""
DBSCAN Clustering from Scratch - Advanced Implementation
=======================================================

This module implements a comprehensive DBSCAN (Density-Based Spatial Clustering 
of Applications with Noise) algorithm from scratch, covering all fundamental concepts:

Core Algorithm:
- Density reachability and core/border/noise points
- Epsilon neighborhood (ε) and minimum points (minPts)
- BFS/DFS for expanding clusters
- Clustering without predefined k
- Handling arbitrary cluster shapes and noise

Mathematical Foundation:
- Core Point: |N_ε(p)| ≥ minPts
- Border Point: Not core but in neighborhood of core point
- Noise Point: Neither core nor border
- Density Reachable: Point q reachable from p through core points
- Density Connected: Points mutually density reachable from same core point

Advanced Features:
- Multiple distance metrics
- Parameter optimization tools
- K-distance graph for epsilon selection
- Comprehensive visualization and analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from typing import List, Tuple, Dict, Union, Optional, Set
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DBSCANScratch:
    """
    DBSCAN clustering implementation from scratch.
    
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) 
    groups together points that are closely packed while marking points 
    in low-density regions as outliers.
    
    Parameters:
    -----------
    eps : float
        Maximum distance between two samples for one to be considered 
        as in the neighborhood of the other
    min_samples : int
        Number of samples in a neighborhood for a point to be considered 
        as a core point
    metric : str
        Distance metric to use ('euclidean', 'manhattan', 'cosine')
    algorithm : str
        Algorithm to use for cluster expansion ('bfs', 'dfs')
    """
    
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean', algorithm='bfs'):
        """
        Initialize DBSCAN clustering algorithm.
        
        Parameters:
        -----------
        eps : float
            Epsilon neighborhood radius
        min_samples : int
            Minimum number of points to form core point
        metric : str
            Distance metric
        algorithm : str
            Cluster expansion algorithm
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.algorithm = algorithm
        
        # Results
        self.labels_ = None
        self.core_sample_indices_ = None
        self.components_ = None
        self.n_features_in_ = None
        
        # Analysis results
        self.point_types_ = None  # 'core', 'border', 'noise'
        self.cluster_info_ = {}
        self.neighborhood_cache_ = {}
        
        # Constants for labeling
        self.NOISE = -1
        self.UNCLASSIFIED = -2
    
    def _compute_distance(self, point1, point2):
        """
        Compute distance between two points.
        
        Parameters:
        -----------
        point1, point2 : np.ndarray
            Points to compute distance between
            
        Returns:
        --------
        float : Distance between points
        """
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((point1 - point2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(point1 - point2))
        elif self.metric == 'cosine':
            dot_product = np.dot(point1, point2)
            norms = np.linalg.norm(point1) * np.linalg.norm(point2)
            if norms == 0:
                return 0
            return 1 - (dot_product / norms)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
    
    def _region_query(self, X, point_idx):
        """
        Find all points within eps distance of given point.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data (n_samples, n_features)
        point_idx : int
            Index of query point
            
        Returns:
        --------
        list : Indices of points in epsilon neighborhood
        """
        # Check cache first
        if point_idx in self.neighborhood_cache_:
            return self.neighborhood_cache_[point_idx]
        
        neighbors = []
        query_point = X[point_idx]
        
        for i, point in enumerate(X):
            if self._compute_distance(query_point, point) <= self.eps:
                neighbors.append(i)
        
        # Cache the result
        self.neighborhood_cache_[point_idx] = neighbors
        return neighbors
    
    def _expand_cluster_bfs(self, X, point_idx, neighbors, cluster_id, labels):
        """
        Expand cluster using breadth-first search.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        point_idx : int
            Starting point index
        neighbors : list
            Initial neighborhood
        cluster_id : int
            Current cluster ID
        labels : np.ndarray
            Point labels array
            
        Returns:
        --------
        bool : True if cluster expanded successfully
        """
        # Assign initial point to cluster
        labels[point_idx] = cluster_id
        
        # Use queue for BFS
        queue = deque(neighbors)
        
        while queue:
            current_point = queue.popleft()
            
            # Skip if already processed
            if labels[current_point] != self.UNCLASSIFIED:
                continue
            
            # Assign to cluster
            labels[current_point] = cluster_id
            
            # Find neighbors of current point
            current_neighbors = self._region_query(X, current_point)
            
            # If current point is core point, add its unprocessed neighbors to queue
            if len(current_neighbors) >= self.min_samples:
                for neighbor in current_neighbors:
                    if labels[neighbor] == self.UNCLASSIFIED:
                        queue.append(neighbor)
        
        return True
    
    def _expand_cluster_dfs(self, X, point_idx, neighbors, cluster_id, labels):
        """
        Expand cluster using depth-first search.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        point_idx : int
            Starting point index
        neighbors : list
            Initial neighborhood
        cluster_id : int
            Current cluster ID
        labels : np.ndarray
            Point labels array
            
        Returns:
        --------
        bool : True if cluster expanded successfully
        """
        # Assign initial point to cluster
        labels[point_idx] = cluster_id
        
        # Use stack for DFS
        stack = list(neighbors)
        
        while stack:
            current_point = stack.pop()
            
            # Skip if already processed
            if labels[current_point] != self.UNCLASSIFIED:
                continue
            
            # Assign to cluster
            labels[current_point] = cluster_id
            
            # Find neighbors of current point
            current_neighbors = self._region_query(X, current_point)
            
            # If current point is core point, add its unprocessed neighbors to stack
            if len(current_neighbors) >= self.min_samples:
                for neighbor in current_neighbors:
                    if labels[neighbor] == self.UNCLASSIFIED:
                        stack.append(neighbor)
        
        return True
    
    def _expand_cluster(self, X, point_idx, neighbors, cluster_id, labels):
        """
        Expand cluster using specified algorithm.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        point_idx : int
            Starting point index
        neighbors : list
            Initial neighborhood
        cluster_id : int
            Current cluster ID
        labels : np.ndarray
            Point labels array
            
        Returns:
        --------
        bool : True if cluster expanded successfully
        """
        if self.algorithm == 'bfs':
            return self._expand_cluster_bfs(X, point_idx, neighbors, cluster_id, labels)
        elif self.algorithm == 'dfs':
            return self._expand_cluster_dfs(X, point_idx, neighbors, cluster_id, labels)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def _classify_points(self, X, labels):
        """
        Classify points as core, border, or noise.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        labels : np.ndarray
            Point labels
            
        Returns:
        --------
        np.ndarray : Point type classifications
        """
        n_samples = len(X)
        point_types = np.full(n_samples, 'noise', dtype=object)
        core_indices = []
        
        for i in range(n_samples):
            neighbors = self._region_query(X, i)
            
            if len(neighbors) >= self.min_samples:
                point_types[i] = 'core'
                core_indices.append(i)
            elif labels[i] != self.NOISE:
                point_types[i] = 'border'
        
        self.core_sample_indices_ = np.array(core_indices)
        return point_types
    
    def _analyze_clusters(self, X, labels):
        """
        Analyze cluster properties.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        labels : np.ndarray
            Point labels
        """
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if self.NOISE in unique_labels else 0)
        
        self.cluster_info_ = {
            'n_clusters': n_clusters,
            'n_noise': np.sum(labels == self.NOISE),
            'cluster_sizes': {},
            'cluster_centers': {},
            'cluster_densities': {}
        }
        
        for label in unique_labels:
            if label == self.NOISE:
                continue
            
            cluster_points = X[labels == label]
            cluster_size = len(cluster_points)
            cluster_center = np.mean(cluster_points, axis=0)
            
            # Calculate cluster density (average distance to centroid)
            distances = [self._compute_distance(point, cluster_center) 
                        for point in cluster_points]
            cluster_density = np.mean(distances)
            
            self.cluster_info_['cluster_sizes'][label] = cluster_size
            self.cluster_info_['cluster_centers'][label] = cluster_center
            self.cluster_info_['cluster_densities'][label] = cluster_density
    
    def fit(self, X):
        """
        Fit DBSCAN clustering to data.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data (n_samples, n_features)
            
        Returns:
        --------
        self : Returns the instance itself
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        
        print(f"Fitting DBSCAN clustering...")
        print(f"  Data shape: {X.shape}")
        print(f"  Epsilon (eps): {self.eps}")
        print(f"  Min samples: {self.min_samples}")
        print(f"  Distance metric: {self.metric}")
        print(f"  Expansion algorithm: {self.algorithm}")
        
        # Initialize labels
        labels = np.full(n_samples, self.UNCLASSIFIED, dtype=int)
        cluster_id = 0
        
        # Clear cache
        self.neighborhood_cache_ = {}
        
        # Main DBSCAN algorithm
        for point_idx in range(n_samples):
            # Skip if already processed
            if labels[point_idx] != self.UNCLASSIFIED:
                continue
            
            # Find neighbors
            neighbors = self._region_query(X, point_idx)
            
            # Check if core point
            if len(neighbors) < self.min_samples:
                # Mark as noise (will be corrected if later found to be border point)
                labels[point_idx] = self.NOISE
            else:
                # Expand cluster from this core point
                self._expand_cluster(X, point_idx, neighbors, cluster_id, labels)
                cluster_id += 1
        
        # Store results
        self.labels_ = labels
        
        # Classify point types
        self.point_types_ = self._classify_points(X, labels)
        
        # Analyze clusters
        self._analyze_clusters(X, labels)
        
        # Extract core samples for sklearn compatibility
        if len(self.core_sample_indices_) > 0:
            self.components_ = X[self.core_sample_indices_]
        else:
            self.components_ = np.empty((0, n_features))
        
        print(f"  Found {self.cluster_info_['n_clusters']} clusters")
        print(f"  Noise points: {self.cluster_info_['n_noise']}")
        print(f"  Core points: {len(self.core_sample_indices_)}")
        
        return self
    
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
    
    def predict(self, X):
        """
        Predict cluster labels for new data points.
        
        Note: DBSCAN doesn't naturally support prediction on new data.
        This method assigns new points to existing clusters based on 
        nearest core points.
        
        Parameters:
        -----------
        X : np.ndarray
            New data points (n_samples, n_features)
            
        Returns:
        --------
        np.ndarray : Predicted cluster labels
        """
        if self.components_ is None or len(self.components_) == 0:
            raise ValueError("No core samples found. Fit the model first.")
        
        X = np.asarray(X)
        n_samples = len(X)
        predictions = np.full(n_samples, self.NOISE, dtype=int)
        
        # For each new point, find nearest core point
        for i, point in enumerate(X):
            min_distance = float('inf')
            closest_cluster = self.NOISE
            
            # Check distance to all core points
            for core_idx in self.core_sample_indices_:
                core_point = self.components_[core_idx - 
                           (0 if hasattr(self, '_fitted_data') else 
                            core_idx - len(self.core_sample_indices_))]
                distance = self._compute_distance(point, core_point)
                
                if distance <= self.eps and distance < min_distance:
                    min_distance = distance
                    closest_cluster = self.labels_[core_idx]
            
            predictions[i] = closest_cluster
        
        return predictions
    
    def plot_clusters(self, X, save_path=None, figsize=(12, 8)):
        """
        Visualize clustering results.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        save_path : str, optional
            Path to save the plot
        figsize : tuple
            Figure size
        """
        if self.labels_ is None:
            raise ValueError("Model must be fitted before plotting")
        
        X = np.asarray(X)
        
        # For high-dimensional data, use first 2 dimensions or PCA
        if X.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42)
            X_vis = pca.fit_transform(X)
            title_suffix = " (PCA projection)"
        else:
            X_vis = X
            title_suffix = ""
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: All points with cluster labels
        ax1 = axes[0, 0]
        unique_labels = np.unique(self.labels_)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == self.NOISE:
                # Noise points in black
                mask = self.labels_ == label
                ax1.scatter(X_vis[mask, 0], X_vis[mask, 1], 
                           c='black', marker='x', s=50, alpha=0.6, label='Noise')
            else:
                mask = self.labels_ == label
                ax1.scatter(X_vis[mask, 0], X_vis[mask, 1], 
                           c=[color], s=50, alpha=0.7, label=f'Cluster {label}')
        
        ax1.set_title(f'DBSCAN Clustering Results{title_suffix}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Point types (core, border, noise)
        ax2 = axes[0, 1]
        type_colors = {'core': 'red', 'border': 'blue', 'noise': 'black'}
        type_markers = {'core': 'o', 'border': 's', 'noise': 'x'}
        
        for point_type in ['core', 'border', 'noise']:
            mask = self.point_types_ == point_type
            if np.any(mask):
                ax2.scatter(X_vis[mask, 0], X_vis[mask, 1],
                           c=type_colors[point_type], marker=type_markers[point_type],
                           s=50, alpha=0.7, label=f'{point_type.capitalize()} points')
        
        ax2.set_title(f'Point Types{title_suffix}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Cluster statistics
        ax3 = axes[1, 0]
        if self.cluster_info_['n_clusters'] > 0:
            cluster_labels = list(self.cluster_info_['cluster_sizes'].keys())
            cluster_sizes = list(self.cluster_info_['cluster_sizes'].values())
            
            bars = ax3.bar(range(len(cluster_labels)), cluster_sizes, alpha=0.7)
            ax3.set_xlabel('Cluster ID')
            ax3.set_ylabel('Number of Points')
            ax3.set_title('Cluster Sizes')
            ax3.set_xticks(range(len(cluster_labels)))
            ax3.set_xticklabels(cluster_labels)
            ax3.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, size in zip(bars, cluster_sizes):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{size}', ha='center', va='bottom')
        else:
            ax3.text(0.5, 0.5, 'No clusters found', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=14)
            ax3.set_title('Cluster Sizes')
        
        # Plot 4: Algorithm information
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        info_text = f"""DBSCAN Parameters:
        
Epsilon (ε): {self.eps}
Min Samples: {self.min_samples}
Distance Metric: {self.metric}
Algorithm: {self.algorithm}

Results:
Clusters Found: {self.cluster_info_['n_clusters']}
Noise Points: {self.cluster_info_['n_noise']}
Core Points: {len(self.core_sample_indices_)}
Border Points: {np.sum(self.point_types_ == 'border')}
        """
        
        ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cluster plot saved to {save_path}")
        
        plt.show()

class DBSCANAnalyzer:
    """
    Comprehensive analyzer for DBSCAN clustering results.
    
    This class provides tools for parameter optimization, 
    performance evaluation, and comparative analysis.
    """
    
    def __init__(self):
        """Initialize the DBSCAN analyzer."""
        self.results = {}
    
    def k_distance_graph(self, X, k=None, save_path=None):
        """
        Generate k-distance graph for epsilon selection.
        
        The k-distance graph helps identify the optimal epsilon value
        by finding the "elbow" in the k-distance plot.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        k : int, optional
            Number of nearest neighbors (default: min_samples - 1)
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        np.ndarray : k-distances for each point
        """
        if k is None:
            k = 4  # Default k for DBSCAN parameter estimation
        
        print(f"Generating {k}-distance graph for epsilon selection...")
        
        # Compute k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
        nbrs.fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        # Get k-distances (excluding the point itself)
        k_distances = distances[:, k]
        k_distances_sorted = np.sort(k_distances)[::-1]
        
        # Plot k-distance graph
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(k_distances_sorted)), k_distances_sorted, 'b-', linewidth=2)
        plt.xlabel('Points sorted by distance')
        plt.ylabel(f'{k}-distance')
        plt.title(f'{k}-Distance Graph for Epsilon Selection')
        plt.grid(True, alpha=0.3)
        
        # Find potential elbow point
        # Simple method: point with maximum second derivative
        if len(k_distances_sorted) > 2:
            second_deriv = np.gradient(np.gradient(k_distances_sorted))
            elbow_idx = np.argmax(second_deriv[:len(second_deriv)//2])  # Look in first half
            suggested_eps = k_distances_sorted[elbow_idx]
            
            plt.axhline(y=suggested_eps, color='red', linestyle='--', 
                       label=f'Suggested ε = {suggested_eps:.3f}')
            plt.axvline(x=elbow_idx, color='red', linestyle='--', alpha=0.7)
            plt.legend()
            
            print(f"Suggested epsilon: {suggested_eps:.3f}")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"K-distance graph saved to {save_path}")
        
        plt.show()
        
        return k_distances_sorted
    
    def parameter_sensitivity_analysis(self, X, eps_range=None, min_samples_range=None, 
                                     save_path=None):
        """
        Analyze sensitivity to DBSCAN parameters.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
        eps_range : list, optional
            Range of epsilon values to test
        min_samples_range : list, optional
            Range of min_samples values to test
        save_path : str, optional
            Path to save the plot
            
        Returns:
        --------
        dict : Parameter sensitivity results
        """
        if eps_range is None:
            # Auto-generate eps range based on data
            eps_range = np.linspace(0.1, 2.0, 20)
        
        if min_samples_range is None:
            min_samples_range = range(3, 11)
        
        print("Performing parameter sensitivity analysis...")
        
        results = {
            'eps_values': [],
            'min_samples_values': [],
            'n_clusters': [],
            'n_noise': [],
            'silhouette_scores': []
        }
        
        # Test different parameter combinations
        for eps in eps_range:
            for min_samples in min_samples_range:
                print(f"  Testing eps={eps:.3f}, min_samples={min_samples}")
                
                dbscan = DBSCANScratch(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(X)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                # Compute silhouette score if we have clusters
                if n_clusters > 1 and n_noise < len(labels):
                    try:
                        sil_score = silhouette_score(X, labels)
                    except:
                        sil_score = -1
                else:
                    sil_score = -1
                
                results['eps_values'].append(eps)
                results['min_samples_values'].append(min_samples)
                results['n_clusters'].append(n_clusters)
                results['n_noise'].append(n_noise)
                results['silhouette_scores'].append(sil_score)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Convert to arrays for easier plotting
        eps_vals = np.array(results['eps_values'])
        min_samples_vals = np.array(results['min_samples_values'])
        n_clusters = np.array(results['n_clusters'])
        n_noise = np.array(results['n_noise'])
        sil_scores = np.array(results['silhouette_scores'])
        
        # Plot 1: Number of clusters heatmap
        ax1 = axes[0, 0]
        eps_unique = np.unique(eps_vals)
        min_samples_unique = np.unique(min_samples_vals)
        cluster_matrix = np.zeros((len(min_samples_unique), len(eps_unique)))
        
        for i, min_samp in enumerate(min_samples_unique):
            for j, eps_val in enumerate(eps_unique):
                mask = (eps_vals == eps_val) & (min_samples_vals == min_samp)
                if np.any(mask):
                    cluster_matrix[i, j] = n_clusters[mask][0]
        
        im1 = ax1.imshow(cluster_matrix, aspect='auto', cmap='viridis')
        ax1.set_xticks(range(len(eps_unique)))
        ax1.set_xticklabels([f'{eps:.2f}' for eps in eps_unique])
        ax1.set_yticks(range(len(min_samples_unique)))
        ax1.set_yticklabels(min_samples_unique)
        ax1.set_xlabel('Epsilon')
        ax1.set_ylabel('Min Samples')
        ax1.set_title('Number of Clusters')
        plt.colorbar(im1, ax=ax1)
        
        # Plot 2: Noise points heatmap
        ax2 = axes[0, 1]
        noise_matrix = np.zeros((len(min_samples_unique), len(eps_unique)))
        
        for i, min_samp in enumerate(min_samples_unique):
            for j, eps_val in enumerate(eps_unique):
                mask = (eps_vals == eps_val) & (min_samples_vals == min_samp)
                if np.any(mask):
                    noise_matrix[i, j] = n_noise[mask][0]
        
        im2 = ax2.imshow(noise_matrix, aspect='auto', cmap='Reds')
        ax2.set_xticks(range(len(eps_unique)))
        ax2.set_xticklabels([f'{eps:.2f}' for eps in eps_unique])
        ax2.set_yticks(range(len(min_samples_unique)))
        ax2.set_yticklabels(min_samples_unique)
        ax2.set_xlabel('Epsilon')
        ax2.set_ylabel('Min Samples')
        ax2.set_title('Number of Noise Points')
        plt.colorbar(im2, ax=ax2)
        
        # Plot 3: Silhouette score heatmap
        ax3 = axes[1, 0]
        sil_matrix = np.full((len(min_samples_unique), len(eps_unique)), -1.0)
        
        for i, min_samp in enumerate(min_samples_unique):
            for j, eps_val in enumerate(eps_unique):
                mask = (eps_vals == eps_val) & (min_samples_vals == min_samp)
                if np.any(mask):
                    sil_matrix[i, j] = sil_scores[mask][0]
        
        # Mask invalid silhouette scores
        sil_matrix_masked = np.ma.masked_where(sil_matrix == -1, sil_matrix)
        
        im3 = ax3.imshow(sil_matrix_masked, aspect='auto', cmap='RdYlBu')
        ax3.set_xticks(range(len(eps_unique)))
        ax3.set_xticklabels([f'{eps:.2f}' for eps in eps_unique])
        ax3.set_yticks(range(len(min_samples_unique)))
        ax3.set_yticklabels(min_samples_unique)
        ax3.set_xlabel('Epsilon')
        ax3.set_ylabel('Min Samples')
        ax3.set_title('Silhouette Score')
        plt.colorbar(im3, ax=ax3)
        
        # Plot 4: Parameter recommendations
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Find best parameters based on silhouette score
        valid_sil_mask = sil_scores > -1
        if np.any(valid_sil_mask):
            best_idx = np.argmax(sil_scores[valid_sil_mask])
            valid_indices = np.where(valid_sil_mask)[0]
            best_global_idx = valid_indices[best_idx]
            
            best_eps = eps_vals[best_global_idx]
            best_min_samples = min_samples_vals[best_global_idx]
            best_sil_score = sil_scores[best_global_idx]
            best_n_clusters = n_clusters[best_global_idx]
            best_n_noise = n_noise[best_global_idx]
            
            recommendation_text = f"""Parameter Recommendations:

Best Parameters (by Silhouette Score):
  Epsilon: {best_eps:.3f}
  Min Samples: {best_min_samples}
  
Results:
  Silhouette Score: {best_sil_score:.3f}
  Number of Clusters: {best_n_clusters}
  Noise Points: {best_n_noise}

Parameter Ranges Tested:
  Epsilon: {min(eps_range):.3f} - {max(eps_range):.3f}
  Min Samples: {min(min_samples_range)} - {max(min_samples_range)}
  Total Combinations: {len(eps_range) * len(min_samples_range)}
            """
        else:
            recommendation_text = """Parameter Recommendations:

No valid clustering found in the 
tested parameter ranges.

Suggestions:
- Try larger epsilon values
- Try smaller min_samples values
- Check data preprocessing
- Consider different distance metrics
            """
        
        ax4.text(0.1, 0.9, recommendation_text, transform=ax4.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Parameter sensitivity analysis saved to {save_path}")
        
        plt.show()
        
        self.results['parameter_sensitivity'] = results
        return results

def generate_spiral_dataset(n_samples=300, noise=0.1, random_state=42):
    """
    Generate a synthetic spiral dataset.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    noise : float
        Amount of noise to add
    random_state : int
        Random seed
        
    Returns:
    --------
    np.ndarray : Spiral data points
    np.ndarray : True labels (for evaluation)
    """
    np.random.seed(random_state)
    
    n_per_spiral = n_samples // 2
    
    # Generate two spirals
    spiral1_t = np.linspace(0, 4*np.pi, n_per_spiral)
    spiral1_x = spiral1_t * np.cos(spiral1_t) / (4*np.pi)
    spiral1_y = spiral1_t * np.sin(spiral1_t) / (4*np.pi)
    
    spiral2_t = np.linspace(0, 4*np.pi, n_per_spiral)
    spiral2_x = -spiral2_t * np.cos(spiral2_t + np.pi) / (4*np.pi)
    spiral2_y = -spiral2_t * np.sin(spiral2_t + np.pi) / (4*np.pi)
    
    # Combine spirals
    X = np.vstack([
        np.column_stack([spiral1_x, spiral1_y]),
        np.column_stack([spiral2_x, spiral2_y])
    ])
    
    # Add noise
    X += np.random.normal(0, noise, X.shape)
    
    # Create labels
    y = np.hstack([np.zeros(n_per_spiral), np.ones(n_per_spiral)])
    
    return X, y

def generate_sample_datasets():
    """
    Generate sample datasets for DBSCAN testing.
    
    Returns:
    --------
    dict : Dictionary containing different datasets
    """
    datasets = {}
    
    # Dataset 1: Two moons
    X_moons, y_moons = make_moons(n_samples=300, noise=0.05, random_state=42)
    datasets['moons'] = {
        'data': X_moons, 
        'labels': y_moons, 
        'name': 'Two Moons',
        'suggested_eps': 0.15,
        'suggested_min_samples': 5
    }
    
    # Dataset 2: Blobs
    X_blobs, y_blobs = make_blobs(n_samples=300, centers=4, n_features=2, 
                                  random_state=42, cluster_std=0.8)
    datasets['blobs'] = {
        'data': X_blobs, 
        'labels': y_blobs, 
        'name': 'Well-separated Blobs',
        'suggested_eps': 0.8,
        'suggested_min_samples': 10
    }
    
    # Dataset 3: Spirals
    X_spiral, y_spiral = generate_spiral_dataset(n_samples=300, noise=0.05, random_state=42)
    datasets['spiral'] = {
        'data': X_spiral, 
        'labels': y_spiral, 
        'name': 'Spiral Dataset',
        'suggested_eps': 0.1,
        'suggested_min_samples': 5
    }
    
    # Dataset 4: Circles
    X_circles, y_circles = make_circles(n_samples=300, noise=0.05, factor=0.3, random_state=42)
    datasets['circles'] = {
        'data': X_circles, 
        'labels': y_circles, 
        'name': 'Concentric Circles',
        'suggested_eps': 0.15,
        'suggested_min_samples': 5
    }
    
    return datasets

def main():
    """
    Main function to demonstrate DBSCAN implementation.
    """
    print("🎯 DBSCAN CLUSTERING FROM SCRATCH")
    print("=" * 50)
    
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Generate sample datasets
    datasets = generate_sample_datasets()
    
    # Test with different datasets
    for name, dataset in datasets.items():
        print(f"\n📊 Testing with {dataset['name']}")
        print("-" * 40)
        
        X = dataset['data']
        true_labels = dataset['labels']
        
        # Use suggested parameters
        eps = dataset['suggested_eps']
        min_samples = dataset['suggested_min_samples']
        
        # Test our implementation
        print(f"\n🔧 Our DBSCAN Implementation:")
        dbscan_scratch = DBSCANScratch(eps=eps, min_samples=min_samples)
        labels_scratch = dbscan_scratch.fit_predict(X)
        
        # Visualize results
        dbscan_scratch.plot_clusters(X, save_path=f'plots/dbscan_{name}.png')
        
        # Evaluation metrics
        n_clusters = len(set(labels_scratch)) - (1 if -1 in labels_scratch else 0)
        n_noise = list(labels_scratch).count(-1)
        
        print(f"\n📈 Results:")
        print(f"  Clusters found: {n_clusters}")
        print(f"  Noise points: {n_noise}")
        print(f"  Core points: {len(dbscan_scratch.core_sample_indices_)}")
        
        # Compute metrics if we have valid clustering
        if n_clusters > 1 and n_noise < len(labels_scratch):
            try:
                sil_score = silhouette_score(X, labels_scratch)
                print(f"  Silhouette score: {sil_score:.4f}")
            except:
                print(f"  Silhouette score: Could not compute")
        
        # If we have true labels, compute ARI
        if true_labels is not None and n_clusters > 0:
            ari_score = adjusted_rand_score(true_labels, labels_scratch)
            nmi_score = normalized_mutual_info_score(true_labels, labels_scratch)
            print(f"  Adjusted Rand Index: {ari_score:.4f}")
            print(f"  Normalized Mutual Info: {nmi_score:.4f}")
    
    # Demonstrate parameter analysis tools
    print(f"\n🔍 Parameter Analysis Tools:")
    print("-" * 40)
    
    # Use moons dataset for analysis
    X_moons = datasets['moons']['data']
    
    # Create analyzer
    analyzer = DBSCANAnalyzer()
    
    # K-distance graph
    print(f"\nGenerating k-distance graph...")
    k_distances = analyzer.k_distance_graph(X_moons, k=4, 
                                           save_path='plots/k_distance_graph.png')
    
    # Parameter sensitivity analysis
    print(f"\nPerforming parameter sensitivity analysis...")
    eps_range = np.linspace(0.05, 0.5, 10)
    min_samples_range = range(3, 8)
    
    sensitivity_results = analyzer.parameter_sensitivity_analysis(
        X_moons, eps_range=eps_range, min_samples_range=min_samples_range,
        save_path='plots/parameter_sensitivity.png'
    )
    
    print("\n✅ DBSCAN IMPLEMENTATION COMPLETE!")
    print("📁 Check the 'plots' folder for generated visualizations.")
    print("🔧 The implementation covers all key DBSCAN concepts:")
    print("   - Core, border, and noise point classification")
    print("   - Epsilon neighborhood queries")
    print("   - BFS/DFS cluster expansion")
    print("   - Parameter optimization tools")
    print("   - Comprehensive evaluation and visualization")
    
    return datasets, analyzer

if __name__ == "__main__":
    main() 