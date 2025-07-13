"""
Gaussian Mixture Models (GMM) from Scratch - Advanced Implementation
===================================================================

This module implements comprehensive Gaussian Mixture Models including:
- Expectation-Maximization (EM) algorithm
- Multivariate Gaussian distributions
- Soft clustering with posterior probabilities
- Parameter estimation (means, covariances, mixing coefficients)
- Log-likelihood convergence analysis
- Comparison with K-means clustering
- Density estimation and contour visualization

Mathematical Foundation:
- GMM: p(x) = Σ(k=1 to K) π_k * N(x|μ_k, Σ_k)
- E-step: γ(z_nk) = π_k * N(x_n|μ_k, Σ_k) / Σ_j π_j * N(x_n|μ_j, Σ_j)
- M-step: μ_k = Σ_n γ(z_nk) * x_n / N_k
- Covariance: Σ_k = Σ_n γ(z_nk) * (x_n - μ_k)(x_n - μ_k)^T / N_k
- Mixing weights: π_k = N_k / N
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, load_iris
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class GaussianMixtureScratch:
    """
    Gaussian Mixture Model implementation from scratch using EM algorithm.
    
    Features:
    - Full EM algorithm with E-step and M-step
    - Multiple initialization strategies
    - Convergence monitoring and log-likelihood tracking
    - Soft and hard clustering predictions
    - Density estimation and probability computation
    - Comprehensive visualization capabilities
    """
    
    def __init__(self, n_components=3, max_iter=100, tol=1e-6, 
                 init_method='kmeans', random_state=42):
        """
        Initialize Gaussian Mixture Model.
        
        Parameters:
        -----------
        n_components : int
            Number of Gaussian components
        max_iter : int
            Maximum number of EM iterations
        tol : float
            Convergence tolerance for log-likelihood
        init_method : str
            Initialization method ('kmeans', 'random', 'kmeans++')
        random_state : int
            Random seed for reproducibility
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.init_method = init_method
        self.random_state = random_state
        
        # Model parameters
        self.means_ = None
        self.covariances_ = None
        self.weights_ = None
        
        # Training history
        self.log_likelihood_history_ = []
        self.n_iter_ = 0
        self.converged_ = False
        
        # Data properties
        self.n_samples_ = None
        self.n_features_ = None
        
    def _initialize_parameters(self, X):
        """
        Initialize GMM parameters using specified method.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data
        """
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        if self.init_method == 'kmeans':
            # Initialize using K-means clustering
            kmeans = KMeans(n_clusters=self.n_components, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            
            self.means_ = np.zeros((self.n_components, n_features))
            self.covariances_ = np.zeros((self.n_components, n_features, n_features))
            
            for k in range(self.n_components):
                mask = labels == k
                if np.sum(mask) > 0:
                    self.means_[k] = np.mean(X[mask], axis=0)
                    self.covariances_[k] = np.cov(X[mask].T) + 1e-6 * np.eye(n_features)
                else:
                    # Handle empty clusters
                    self.means_[k] = X[np.random.randint(n_samples)]
                    self.covariances_[k] = np.eye(n_features)
                    
        elif self.init_method == 'random':
            # Random initialization
            self.means_ = X[np.random.choice(n_samples, self.n_components, replace=False)]
            self.covariances_ = np.array([np.eye(n_features) for _ in range(self.n_components)])
            
        elif self.init_method == 'kmeans++':
            # K-means++ initialization
            self.means_ = self._kmeans_plus_plus_init(X)
            self.covariances_ = np.array([np.cov(X.T) + 1e-6 * np.eye(n_features) 
                                        for _ in range(self.n_components)])
        
        # Initialize mixing weights uniformly
        self.weights_ = np.ones(self.n_components) / self.n_components
        
        print(f"Initialized GMM with {self.n_components} components using {self.init_method}")
        print(f"Initial means shape: {self.means_.shape}")
        print(f"Initial covariances shape: {self.covariances_.shape}")
        print(f"Initial weights: {self.weights_}")
        
    def _kmeans_plus_plus_init(self, X):
        """
        K-means++ initialization for better initial cluster centers.
        
        Parameters:
        -----------
        X : np.ndarray
            Training data
            
        Returns:
        --------
        np.ndarray : Initial means
        """
        n_samples, n_features = X.shape
        means = np.zeros((self.n_components, n_features))
        
        # Choose first center randomly
        means[0] = X[np.random.randint(n_samples)]
        
        for k in range(1, self.n_components):
            # Calculate distances to nearest center
            distances = np.array([min([np.linalg.norm(x - c)**2 for c in means[:k]]) for x in X])
            
            # Choose next center with probability proportional to squared distance
            probabilities = distances / distances.sum()
            cumulative_probabilities = probabilities.cumsum()
            r = np.random.rand()
            
            for i, p in enumerate(cumulative_probabilities):
                if r < p:
                    means[k] = X[i]
                    break
                    
        return means
    
    def _multivariate_gaussian_pdf(self, X, mean, cov):
        """
        Compute multivariate Gaussian probability density function.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Data points
        mean : np.ndarray, shape (n_features,)
            Mean vector
        cov : np.ndarray, shape (n_features, n_features)
            Covariance matrix
            
        Returns:
        --------
        np.ndarray : PDF values
        """
        try:
            return multivariate_normal.pdf(X, mean=mean, cov=cov)
        except np.linalg.LinAlgError:
            # Handle singular covariance matrix
            cov_reg = cov + 1e-6 * np.eye(cov.shape[0])
            return multivariate_normal.pdf(X, mean=mean, cov=cov_reg)
    
    def _e_step(self, X):
        """
        Expectation step: compute posterior probabilities (responsibilities).
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        np.ndarray : Responsibilities γ(z_nk), shape (n_samples, n_components)
        """
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        
        # Compute likelihood for each component
        for k in range(self.n_components):
            responsibilities[:, k] = (self.weights_[k] * 
                                    self._multivariate_gaussian_pdf(X, self.means_[k], self.covariances_[k]))
        
        # Normalize to get posterior probabilities
        responsibilities_sum = responsibilities.sum(axis=1, keepdims=True)
        responsibilities_sum[responsibilities_sum == 0] = 1e-8  # Avoid division by zero
        responsibilities /= responsibilities_sum
        
        return responsibilities
    
    def _m_step(self, X, responsibilities):
        """
        Maximization step: update parameters using responsibilities.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data
        responsibilities : np.ndarray, shape (n_samples, n_components)
            Posterior probabilities from E-step
        """
        n_samples, n_features = X.shape
        
        # Effective number of points assigned to each component
        N_k = responsibilities.sum(axis=0)
        
        # Update mixing weights
        self.weights_ = N_k / n_samples
        
        # Update means
        for k in range(self.n_components):
            if N_k[k] > 0:
                self.means_[k] = np.sum(responsibilities[:, k:k+1] * X, axis=0) / N_k[k]
            else:
                # Handle empty components
                self.means_[k] = X[np.random.randint(n_samples)]
        
        # Update covariances
        for k in range(self.n_components):
            if N_k[k] > 0:
                diff = X - self.means_[k]
                
                # Compute weighted covariance matrix
                weighted_diff = responsibilities[:, k:k+1] * diff
                self.covariances_[k] = (weighted_diff.T @ diff) / N_k[k]
                
                # Add regularization to prevent singular matrices
                self.covariances_[k] += 1e-6 * np.eye(n_features)
            else:
                # Handle empty components
                self.covariances_[k] = np.eye(n_features)
    
    def _compute_log_likelihood(self, X):
        """
        Compute log-likelihood of the data given current parameters.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        float : Log-likelihood
        """
        n_samples = X.shape[0]
        log_likelihood = 0
        
        for n in range(n_samples):
            sample_likelihood = 0
            for k in range(self.n_components):
                sample_likelihood += (self.weights_[k] * 
                                    self._multivariate_gaussian_pdf(X[n:n+1], self.means_[k], self.covariances_[k]))
            
            if sample_likelihood > 0:
                log_likelihood += np.log(sample_likelihood)
            else:
                log_likelihood += -np.inf
        
        return log_likelihood
    
    def fit(self, X):
        """
        Fit Gaussian Mixture Model using EM algorithm.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        self : GaussianMixtureScratch
            Fitted model
        """
        self.n_samples_, self.n_features_ = X.shape
        
        # Initialize parameters
        self._initialize_parameters(X)
        
        # Initialize log-likelihood history
        self.log_likelihood_history_ = []
        prev_log_likelihood = -np.inf
        
        print(f"\nStarting EM algorithm...")
        print(f"Convergence tolerance: {self.tol}")
        print(f"Maximum iterations: {self.max_iter}")
        print("-" * 50)
        
        for iteration in range(self.max_iter):
            # E-step: compute responsibilities
            responsibilities = self._e_step(X)
            
            # M-step: update parameters
            self._m_step(X, responsibilities)
            
            # Compute log-likelihood
            current_log_likelihood = self._compute_log_likelihood(X)
            self.log_likelihood_history_.append(current_log_likelihood)
            
            # Check convergence
            log_likelihood_change = current_log_likelihood - prev_log_likelihood
            
            if iteration % 10 == 0 or iteration < 10:
                print(f"Iteration {iteration:3d}: Log-likelihood = {current_log_likelihood:.4f}, "
                      f"Change = {log_likelihood_change:.6f}")
            
            if abs(log_likelihood_change) < self.tol:
                print(f"\nConverged after {iteration + 1} iterations!")
                print(f"Final log-likelihood: {current_log_likelihood:.4f}")
                self.converged_ = True
                break
                
            prev_log_likelihood = current_log_likelihood
            
        self.n_iter_ = iteration + 1
        
        if not self.converged_:
            print(f"\nMaximum iterations ({self.max_iter}) reached without convergence.")
            print(f"Final log-likelihood: {current_log_likelihood:.4f}")
        
        print(f"\nFinal parameters:")
        print(f"Mixing weights: {self.weights_}")
        print(f"Means shape: {self.means_.shape}")
        print(f"Covariances shape: {self.covariances_.shape}")
        
        return self
    
    def predict_proba(self, X):
        """
        Predict posterior probabilities (soft assignments) for new data.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to predict
            
        Returns:
        --------
        np.ndarray : Posterior probabilities, shape (n_samples, n_components)
        """
        if self.means_ is None:
            raise ValueError("Model must be fitted before making predictions")
        
        return self._e_step(X)
    
    def predict(self, X):
        """
        Predict hard cluster assignments for new data.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to predict
            
        Returns:
        --------
        np.ndarray : Cluster labels, shape (n_samples,)
        """
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)
    
    def score_samples(self, X):
        """
        Compute log-likelihood of samples under the model.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to score
            
        Returns:
        --------
        np.ndarray : Log-likelihood per sample
        """
        n_samples = X.shape[0]
        log_likelihoods = np.zeros(n_samples)
        
        for n in range(n_samples):
            sample_likelihood = 0
            for k in range(self.n_components):
                sample_likelihood += (self.weights_[k] * 
                                    self._multivariate_gaussian_pdf(X[n:n+1], self.means_[k], self.covariances_[k]))
            
            log_likelihoods[n] = np.log(sample_likelihood) if sample_likelihood > 0 else -np.inf
        
        return log_likelihoods
    
    def score(self, X):
        """
        Compute average log-likelihood of the data.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to score
            
        Returns:
        --------
        float : Average log-likelihood
        """
        return np.mean(self.score_samples(X))
    
    def plot_convergence(self, save_path=None):
        """
        Plot log-likelihood convergence during EM algorithm.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        """
        if not self.log_likelihood_history_:
            raise ValueError("No convergence history available. Fit the model first.")
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.log_likelihood_history_, 'b-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('EM Iteration')
        plt.ylabel('Log-Likelihood')
        plt.title(f'GMM Convergence (K={self.n_components})')
        plt.grid(True, alpha=0.3)
        
        # Add convergence info
        if self.converged_:
            plt.axvline(x=len(self.log_likelihood_history_)-1, color='r', linestyle='--', 
                       alpha=0.7, label=f'Converged at iteration {self.n_iter_}')
            plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Convergence plot saved to {save_path}")
        
        plt.show()
    
    def plot_clusters_2d(self, X, true_labels=None, save_path=None):
        """
        Plot 2D clustering results with Gaussian contours.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, 2)
            2D data to plot
        true_labels : np.ndarray, optional
            True cluster labels for comparison
        save_path : str, optional
            Path to save the plot
        """
        if X.shape[1] != 2:
            raise ValueError("This method only works for 2D data")
        
        if self.means_ is None:
            raise ValueError("Model must be fitted before plotting")
        
        # Get predictions
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        # Create subplot layout
        n_plots = 3 if true_labels is not None else 2
        fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))
        if n_plots == 2:
            axes = [axes[0], axes[1]]
        
        # Plot 1: Hard clustering
        ax1 = axes[0]
        scatter = ax1.scatter(X[:, 0], X[:, 1], c=predictions, cmap='viridis', alpha=0.7, s=50)
        ax1.scatter(self.means_[:, 0], self.means_[:, 1], c='red', marker='x', s=200, linewidths=3)
        
        # Add Gaussian contours
        self._plot_gaussian_contours(ax1, X)
        
        ax1.set_title(f'GMM Hard Clustering (K={self.n_components})')
        ax1.set_xlabel('Feature 1')
        ax1.set_ylabel('Feature 2')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1)
        
        # Plot 2: Soft clustering (uncertainty)
        ax2 = axes[1]
        # Use entropy as measure of uncertainty
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-8), axis=1)
        scatter2 = ax2.scatter(X[:, 0], X[:, 1], c=entropy, cmap='plasma', alpha=0.7, s=50)
        ax2.scatter(self.means_[:, 0], self.means_[:, 1], c='red', marker='x', s=200, linewidths=3)
        
        ax2.set_title('GMM Uncertainty (Entropy)')
        ax2.set_xlabel('Feature 1')
        ax2.set_ylabel('Feature 2')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Entropy')
        
        # Plot 3: True labels (if available)
        if true_labels is not None:
            ax3 = axes[2]
            scatter3 = ax3.scatter(X[:, 0], X[:, 1], c=true_labels, cmap='Set1', alpha=0.7, s=50)
            ax3.set_title('True Labels')
            ax3.set_xlabel('Feature 1')
            ax3.set_ylabel('Feature 2')
            ax3.grid(True, alpha=0.3)
            plt.colorbar(scatter3, ax=ax3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Clustering plot saved to {save_path}")
        
        plt.show()
    
    def _plot_gaussian_contours(self, ax, X, n_std=2):
        """
        Plot Gaussian contour ellipses for each component.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Axes to plot on
        X : np.ndarray
            Data for determining plot limits
        n_std : float
            Number of standard deviations for contours
        """
        colors = plt.cm.Set1(np.linspace(0, 1, self.n_components))
        
        for k in range(self.n_components):
            mean = self.means_[k]
            cov = self.covariances_[k]
            
            # Compute eigenvalues and eigenvectors for ellipse
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

def load_datasets():
    """
    Load and prepare datasets for GMM analysis.
    
    Returns:
    --------
    dict : Dictionary containing different datasets
    """
    datasets = {}
    
    # 1. Synthetic blobs
    print("Loading synthetic blob dataset...")
    X_blobs, y_blobs = make_blobs(n_samples=500, centers=3, cluster_std=1.5, 
                                 center_box=(-10.0, 10.0), random_state=42)
    datasets['blobs'] = {'data': X_blobs, 'labels': y_blobs, 'name': 'Synthetic Blobs'}
    
    # 2. Iris dataset (first two features)
    print("Loading Iris dataset...")
    iris = load_iris()
    X_iris = iris.data[:, :2]  # Use only first two features for 2D visualization
    y_iris = iris.target
    datasets['iris'] = {'data': X_iris, 'labels': y_iris, 'name': 'Iris (2D)'}
    
    # 3. Overlapping clusters
    print("Loading overlapping clusters dataset...")
    X_overlap, y_overlap = make_blobs(n_samples=400, centers=3, cluster_std=2.0, 
                                     center_box=(-5.0, 5.0), random_state=123)
    datasets['overlap'] = {'data': X_overlap, 'labels': y_overlap, 'name': 'Overlapping Clusters'}
    
    return datasets

def demonstrate_gmm_properties():
    """
    Demonstrate key GMM properties and EM algorithm convergence.
    """
    print("=" * 70)
    print("GMM MATHEMATICAL PROPERTIES DEMONSTRATION")
    print("=" * 70)
    
    # Load datasets
    datasets = load_datasets()
    
    results = {}
    
    for dataset_name, dataset_info in datasets.items():
        print(f"\n{dataset_info['name'].upper()} ANALYSIS")
        print("-" * 50)
        
        X = dataset_info['data']
        y_true = dataset_info['labels']
        n_true_clusters = len(np.unique(y_true))
        
        print(f"Dataset shape: {X.shape}")
        print(f"True number of clusters: {n_true_clusters}")
        
        # Fit GMM with different initialization methods
        init_methods = ['kmeans', 'random', 'kmeans++']
        dataset_results = {}
        
        for init_method in init_methods:
            print(f"\nTesting initialization: {init_method}")
            
            gmm = GaussianMixtureScratch(
                n_components=n_true_clusters,
                max_iter=100,
                tol=1e-6,
                init_method=init_method,
                random_state=42
            )
            
            gmm.fit(X)
            
            # Store results
            dataset_results[init_method] = {
                'model': gmm,
                'final_log_likelihood': gmm.log_likelihood_history_[-1],
                'n_iterations': gmm.n_iter_,
                'converged': gmm.converged_
            }
            
            print(f"Final log-likelihood: {gmm.log_likelihood_history_[-1]:.4f}")
            print(f"Iterations: {gmm.n_iter_}")
            print(f"Converged: {gmm.converged_}")
        
        results[dataset_name] = dataset_results
    
    return results, datasets

def main():
    """
    Main function to run comprehensive GMM analysis.
    """
    print("🔄 GAUSSIAN MIXTURE MODELS (GMM) - ADVANCED IMPLEMENTATION")
    print("=" * 80)
    
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    # 1. Demonstrate GMM properties
    results, datasets = demonstrate_gmm_properties()
    
    # 2. Generate detailed analysis for best performing models
    print("\n📊 GENERATING DETAILED ANALYSIS...")
    print("-" * 50)
    
    for dataset_name, dataset_info in datasets.items():
        X = dataset_info['data']
        y_true = dataset_info['labels']
        
        # Find best initialization method based on log-likelihood
        best_method = max(results[dataset_name].keys(), 
                         key=lambda k: results[dataset_name][k]['final_log_likelihood'])
        
        best_gmm = results[dataset_name][best_method]['model']
        
        print(f"\nAnalyzing {dataset_info['name']} with best method: {best_method}")
        
        # Plot convergence
        best_gmm.plot_convergence(f'plots/gmm_{dataset_name}_convergence.png')
        
        # Plot clustering results
        best_gmm.plot_clusters_2d(X, y_true, f'plots/gmm_{dataset_name}_clusters.png')
        
        # Compute clustering metrics
        y_pred = best_gmm.predict(X)
        ari = adjusted_rand_score(y_true, y_pred)
        silhouette = silhouette_score(X, y_pred)
        
        print(f"Adjusted Rand Index: {ari:.4f}")
        print(f"Silhouette Score: {silhouette:.4f}")
        print(f"Average Log-Likelihood: {best_gmm.score(X):.4f}")
    
    # 3. Compare initialization methods
    print("\n📈 INITIALIZATION METHOD COMPARISON...")
    print("-" * 45)
    
    # Define initialization methods for summary
    init_methods = ['kmeans', 'random', 'kmeans++']
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, (dataset_name, dataset_info) in enumerate(datasets.items()):
        ax = axes[i]
        
        methods = []
        log_likelihoods = []
        iterations = []
        
        for method, result in results[dataset_name].items():
            methods.append(method)
            log_likelihoods.append(result['final_log_likelihood'])
            iterations.append(result['n_iterations'])
        
        # Plot log-likelihood comparison
        bars = ax.bar(methods, log_likelihoods, alpha=0.7)
        ax.set_title(f'{dataset_info["name"]}\nFinal Log-Likelihood')
        ax.set_ylabel('Log-Likelihood')
        ax.tick_params(axis='x', rotation=45)
        
        # Add iteration count annotations
        for bar, iters in zip(bars, iterations):
            height = bar.get_height()
            ax.annotate(f'{iters} iter', 
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/gmm_initialization_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✅ GMM FROM SCRATCH ANALYSIS COMPLETE!")
    print("📁 Check the 'plots' folder for generated visualizations.")
    print("\n📋 SUMMARY:")
    print(f"• Analyzed {len(datasets)} different datasets")
    print(f"• Tested {len(init_methods)} initialization methods")
    print(f"• Generated {len(datasets) * 2 + 1} visualization files")
    print("• Demonstrated EM algorithm convergence")
    print("• Compared soft vs hard clustering approaches")

if __name__ == "__main__":
    main() 