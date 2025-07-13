"""
EM Algorithm for GMM - Generalized Framework Implementation
==========================================================

This module refactors the GMM implementation into a general EM algorithm
framework, demonstrating how EM applies broadly to latent variable models:

Key Concepts:
- Abstract EMAlgorithm base class with E-step and M-step methods
- Concrete GMMEMAlgorithm implementation inheriting from base class
- Demonstrates the generality of the EM framework
- Shows how different models can share the same algorithmic structure
- Compares with previous GMM implementation for validation

Mathematical Framework:
- General EM: Maximize Q(θ|θ^(t)) = E[log p(X,Z|θ) | X, θ^(t)]
- GMM E-step: Compute γ(z_nk) = π_k N(x_n|μ_k,Σ_k) / Σ_j π_j N(x_n|μ_j,Σ_j)
- GMM M-step: Update π_k, μ_k, Σ_k using weighted maximum likelihood
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, load_iris
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class EMAlgorithm(ABC):
    """
    Abstract base class for Expectation-Maximization algorithms.
    
    This class provides the general EM framework that can be specialized
    for different latent variable models (GMM, HMM, Factor Analysis, etc.).
    
    Features:
    - Abstract E-step and M-step methods for specialization
    - Common convergence monitoring and logging
    - General parameter initialization framework
    - Standardized fit() method for all EM algorithms
    """
    
    def __init__(self, max_iter=100, tol=1e-6, random_state=42, verbose=True):
        """
        Initialize general EM algorithm.
        
        Parameters:
        -----------
        max_iter : int
            Maximum number of EM iterations
        tol : float
            Convergence tolerance for log-likelihood
        random_state : int
            Random seed for reproducibility
        verbose : bool
            Whether to print progress information
        """
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        
        # Common tracking variables
        self.log_likelihood_history = []
        self.n_iter = 0
        self.converged = False
        
        # Data storage
        self.X = None
        self.n_samples = None
        self.n_features = None
    
    @abstractmethod
    def initialize_parameters(self, X):
        """
        Initialize model parameters.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data
        """
        pass
    
    @abstractmethod
    def e_step(self, X):
        """
        Expectation step: Compute expected values of latent variables.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        responsibilities : np.ndarray
            Expected values of latent variables
        """
        pass
    
    @abstractmethod
    def m_step(self, X, responsibilities):
        """
        Maximization step: Update parameters using expected latent variables.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data
        responsibilities : np.ndarray
            Expected values from E-step
        """
        pass
    
    @abstractmethod
    def compute_log_likelihood(self, X):
        """
        Compute log-likelihood of observed data.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        float : Log-likelihood
        """
        pass
    
    def fit(self, X):
        """
        Fit the EM algorithm to data.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data
            
        Returns:
        --------
        self : EMAlgorithm
            Fitted model
        """
        self.X = X
        self.n_samples, self.n_features = X.shape
        
        if self.verbose:
            print(f"Starting EM algorithm...")
            print(f"Data shape: {X.shape}")
            print(f"Convergence tolerance: {self.tol}")
            print(f"Maximum iterations: {self.max_iter}")
            print("-" * 60)
        
        # Initialize parameters
        self.initialize_parameters(X)
        
        # Initialize tracking
        self.log_likelihood_history = []
        prev_log_likelihood = -np.inf
        
        for iteration in range(self.max_iter):
            # E-step: Compute expected latent variables
            responsibilities = self.e_step(X)
            
            # M-step: Update parameters
            self.m_step(X, responsibilities)
            
            # Compute log-likelihood
            current_log_likelihood = self.compute_log_likelihood(X)
            self.log_likelihood_history.append(current_log_likelihood)
            
            # Check convergence
            log_likelihood_change = current_log_likelihood - prev_log_likelihood
            
            if self.verbose and (iteration % 10 == 0 or iteration < 10):
                print(f"Iteration {iteration:3d}: Log-likelihood = {current_log_likelihood:.4f}, "
                      f"Change = {log_likelihood_change:.6f}")
            
            if abs(log_likelihood_change) < self.tol:
                if self.verbose:
                    print(f"\nConverged after {iteration + 1} iterations!")
                    print(f"Final log-likelihood: {current_log_likelihood:.4f}")
                self.converged = True
                break
                
            prev_log_likelihood = current_log_likelihood
            
        self.n_iter = iteration + 1
        
        if not self.converged and self.verbose:
            print(f"\nMaximum iterations ({self.max_iter}) reached without convergence.")
            print(f"Final log-likelihood: {current_log_likelihood:.4f}")
        
        return self
    
    def plot_convergence(self, title="EM Algorithm Convergence", save_path=None):
        """
        Plot log-likelihood convergence.
        
        Parameters:
        -----------
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
        """
        if not self.log_likelihood_history:
            raise ValueError("No convergence history available. Fit the model first.")
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.log_likelihood_history, 'b-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('EM Iteration')
        plt.ylabel('Log-Likelihood')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
        if self.converged:
            plt.axvline(x=len(self.log_likelihood_history)-1, color='r', linestyle='--', 
                       alpha=0.7, label=f'Converged at iteration {self.n_iter}')
            plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Convergence plot saved to {save_path}")
        
        plt.show()

class GMMEMAlgorithm(EMAlgorithm):
    """
    Gaussian Mixture Model implementation using the general EM framework.
    
    This class demonstrates how GMM fits into the general EM algorithm pattern
    by implementing the abstract methods from EMAlgorithm.
    
    Features:
    - Inherits convergence monitoring from EMAlgorithm
    - Implements GMM-specific E-step and M-step
    - Provides clustering and density estimation capabilities
    - Compatible with scikit-learn interface
    """
    
    def __init__(self, n_components=3, covariance_type='full', **kwargs):
        """
        Initialize GMM using EM framework.
        
        Parameters:
        -----------
        n_components : int
            Number of Gaussian components
        covariance_type : str
            Type of covariance matrix ('full', 'diag', 'spherical')
        **kwargs : dict
            Additional arguments passed to EMAlgorithm
        """
        super().__init__(**kwargs)
        self.n_components = n_components
        self.covariance_type = covariance_type
        
        # Model parameters
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
    
    def initialize_parameters(self, X):
        """
        Initialize GMM parameters using K-means.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data
        """
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # Initialize using K-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_components, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Initialize means from K-means centers
        self.means_ = kmeans.cluster_centers_.copy()
        
        # Initialize covariances from cluster scatter matrices
        self.covariances_ = np.zeros((self.n_components, n_features, n_features))
        for k in range(self.n_components):
            mask = labels == k
            if np.sum(mask) > 0:
                if self.covariance_type == 'full':
                    self.covariances_[k] = np.cov(X[mask].T) + 1e-6 * np.eye(n_features)
                elif self.covariance_type == 'diag':
                    self.covariances_[k] = np.diag(np.var(X[mask], axis=0) + 1e-6)
                elif self.covariance_type == 'spherical':
                    var = np.mean(np.var(X[mask], axis=0))
                    self.covariances_[k] = (var + 1e-6) * np.eye(n_features)
            else:
                # Handle empty clusters
                if self.covariance_type == 'full':
                    self.covariances_[k] = np.eye(n_features)
                elif self.covariance_type == 'diag':
                    self.covariances_[k] = np.eye(n_features)
                elif self.covariance_type == 'spherical':
                    self.covariances_[k] = np.eye(n_features)
        
        # Initialize mixing weights uniformly
        self.weights_ = np.ones(self.n_components) / self.n_components
        
        if self.verbose:
            print(f"Initialized GMM with {self.n_components} components")
            print(f"Covariance type: {self.covariance_type}")
            print(f"Initial mixing weights: {self.weights_}")
    
    def e_step(self, X):
        """
        E-step: Compute responsibilities (posterior probabilities).
        
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
            try:
                responsibilities[:, k] = (self.weights_[k] * 
                                        multivariate_normal.pdf(X, self.means_[k], self.covariances_[k]))
            except np.linalg.LinAlgError:
                # Handle singular covariance matrix
                cov_reg = self.covariances_[k] + 1e-6 * np.eye(self.covariances_[k].shape[0])
                responsibilities[:, k] = (self.weights_[k] * 
                                        multivariate_normal.pdf(X, self.means_[k], cov_reg))
        
        # Normalize to get posterior probabilities
        responsibilities_sum = responsibilities.sum(axis=1, keepdims=True)
        responsibilities_sum[responsibilities_sum == 0] = 1e-8  # Avoid division by zero
        responsibilities /= responsibilities_sum
        
        return responsibilities
    
    def m_step(self, X, responsibilities):
        """
        M-step: Update parameters using responsibilities.
        
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
                weighted_diff = responsibilities[:, k:k+1] * diff
                
                if self.covariance_type == 'full':
                    self.covariances_[k] = (weighted_diff.T @ diff) / N_k[k]
                    self.covariances_[k] += 1e-6 * np.eye(n_features)
                elif self.covariance_type == 'diag':
                    self.covariances_[k] = np.diag(np.sum(weighted_diff * diff, axis=0) / N_k[k] + 1e-6)
                elif self.covariance_type == 'spherical':
                    var = np.sum(weighted_diff * diff) / (N_k[k] * n_features)
                    self.covariances_[k] = (var + 1e-6) * np.eye(n_features)
            else:
                # Handle empty components
                self.covariances_[k] = np.eye(n_features)
    
    def compute_log_likelihood(self, X):
        """
        Compute log-likelihood of observed data.
        
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
                try:
                    sample_likelihood += (self.weights_[k] * 
                                        multivariate_normal.pdf(X[n:n+1], self.means_[k], self.covariances_[k]))
                except np.linalg.LinAlgError:
                    # Handle singular covariance matrix
                    cov_reg = self.covariances_[k] + 1e-6 * np.eye(self.covariances_[k].shape[0])
                    sample_likelihood += (self.weights_[k] * 
                                        multivariate_normal.pdf(X[n:n+1], self.means_[k], cov_reg))
            
            if sample_likelihood > 0:
                log_likelihood += np.log(sample_likelihood)
            else:
                log_likelihood += -np.inf
        
        return log_likelihood
    
    def predict(self, X):
        """
        Predict cluster labels for new data.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to predict
            
        Returns:
        --------
        np.ndarray : Cluster labels
        """
        responsibilities = self.e_step(X)
        return np.argmax(responsibilities, axis=1)
    
    def predict_proba(self, X):
        """
        Predict posterior probabilities for new data.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Data to predict
            
        Returns:
        --------
        np.ndarray : Posterior probabilities
        """
        return self.e_step(X)
    
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
        return self.compute_log_likelihood(X) / len(X)

def compare_implementations():
    """
    Compare the generalized EM implementation with scikit-learn GMM.
    """
    print("🔄 COMPARING EM IMPLEMENTATIONS")
    print("=" * 50)
    
    # Generate test data
    np.random.seed(42)
    X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=1.5, random_state=42)
    
    print(f"Test data shape: {X.shape}")
    print(f"True number of clusters: {len(np.unique(y_true))}")
    
    # 1. Our generalized EM implementation
    print("\n1. Training Generalized EM GMM...")
    gmm_em = GMMEMAlgorithm(n_components=3, max_iter=100, tol=1e-6, random_state=42)
    gmm_em.fit(X)
    
    # 2. Scikit-learn implementation
    print("\n2. Training Scikit-learn GMM...")
    gmm_sklearn = GaussianMixture(n_components=3, max_iter=100, tol=1e-6, random_state=42)
    gmm_sklearn.fit(X)
    
    # Compare results
    print("\n📊 COMPARISON RESULTS")
    print("-" * 30)
    
    # Predictions
    labels_em = gmm_em.predict(X)
    labels_sklearn = gmm_sklearn.predict(X)
    
    # Metrics
    ari_em = adjusted_rand_score(y_true, labels_em)
    ari_sklearn = adjusted_rand_score(y_true, labels_sklearn)
    
    silhouette_em = silhouette_score(X, labels_em)
    silhouette_sklearn = silhouette_score(X, labels_sklearn)
    
    print(f"Generalized EM:")
    print(f"  ARI: {ari_em:.4f}")
    print(f"  Silhouette: {silhouette_em:.4f}")
    print(f"  Log-likelihood: {gmm_em.compute_log_likelihood(X):.4f}")
    print(f"  Iterations: {gmm_em.n_iter}")
    print(f"  Converged: {gmm_em.converged}")
    
    print(f"\nScikit-learn GMM:")
    print(f"  ARI: {ari_sklearn:.4f}")
    print(f"  Silhouette: {silhouette_sklearn:.4f}")
    print(f"  Log-likelihood: {gmm_sklearn.score(X) * len(X):.4f}")
    print(f"  Iterations: {gmm_sklearn.n_iter_}")
    print(f"  Converged: {gmm_sklearn.converged_}")
    
    # Parameter comparison
    print(f"\nParameter Comparison:")
    print(f"  Mean difference: {np.mean(np.abs(gmm_em.means_ - gmm_sklearn.means_)):.6f}")
    print(f"  Weight difference: {np.mean(np.abs(gmm_em.weights_ - gmm_sklearn.weights_)):.6f}")
    
    return gmm_em, gmm_sklearn, X, y_true

def demonstrate_covariance_types():
    """
    Demonstrate different covariance types in the generalized EM framework.
    """
    print("\n🔧 DEMONSTRATING COVARIANCE TYPES")
    print("=" * 45)
    
    # Generate elliptical clusters
    np.random.seed(42)
    X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=2.0, random_state=42)
    
    # Apply transformation to make clusters elliptical
    transformation = np.array([[1.5, 0.5], [0.3, 0.8]])
    X = X @ transformation
    
    covariance_types = ['full', 'diag', 'spherical']
    results = {}
    
    for cov_type in covariance_types:
        print(f"\nTesting covariance type: {cov_type}")
        
        gmm = GMMEMAlgorithm(
            n_components=3, 
            covariance_type=cov_type,
            max_iter=100, 
            tol=1e-6, 
            random_state=42,
            verbose=False
        )
        
        gmm.fit(X)
        labels = gmm.predict(X)
        
        ari = adjusted_rand_score(y_true, labels)
        silhouette = silhouette_score(X, labels)
        
        results[cov_type] = {
            'model': gmm,
            'labels': labels,
            'ari': ari,
            'silhouette': silhouette,
            'log_likelihood': gmm.compute_log_likelihood(X),
            'iterations': gmm.n_iter
        }
        
        print(f"  ARI: {ari:.4f}")
        print(f"  Silhouette: {silhouette:.4f}")
        print(f"  Log-likelihood: {gmm.compute_log_likelihood(X):.4f}")
        print(f"  Iterations: {gmm.n_iter}")
    
    # Visualization
    plot_covariance_comparison(X, y_true, results)
    
    return results

def plot_covariance_comparison(X, y_true, results):
    """
    Plot comparison of different covariance types.
    
    Parameters:
    -----------
    X : np.ndarray
        Input data
    y_true : np.ndarray
        True labels
    results : dict
        Results from different covariance types
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: True labels
    ax1 = axes[0, 0]
    scatter = ax1.scatter(X[:, 0], X[:, 1], c=y_true, cmap='Set1', alpha=0.7, s=50)
    ax1.set_title('True Labels')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1)
    
    # Plot 2-4: Different covariance types
    covariance_types = ['full', 'diag', 'spherical']
    axes_flat = [axes[0, 1], axes[1, 0], axes[1, 1]]
    
    for i, cov_type in enumerate(covariance_types):
        ax = axes_flat[i]
        labels = results[cov_type]['labels']
        
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7, s=50)
        
        # Plot cluster centers
        means = results[cov_type]['model'].means_
        ax.scatter(means[:, 0], means[:, 1], c='red', marker='x', s=200, linewidths=3)
        
        ax.set_title(f'{cov_type.title()} Covariance\nARI: {results[cov_type]["ari"]:.3f}')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax)
    
    plt.tight_layout()
    plt.savefig('plots/covariance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_convergence_behavior():
    """
    Analyze convergence behavior across different datasets.
    """
    print("\n📈 ANALYZING CONVERGENCE BEHAVIOR")
    print("=" * 40)
    
    datasets = {
        'blobs': make_blobs(n_samples=300, centers=3, cluster_std=1.5, random_state=42),
        'iris': (load_iris().data[:, :2], load_iris().target)
    }
    
    convergence_results = {}
    
    for name, (X, y_true) in datasets.items():
        print(f"\nDataset: {name}")
        print(f"Shape: {X.shape}")
        
        gmm = GMMEMAlgorithm(
            n_components=len(np.unique(y_true)),
            max_iter=200,
            tol=1e-8,
            random_state=42,
            verbose=False
        )
        
        gmm.fit(X)
        
        convergence_results[name] = {
            'model': gmm,
            'log_likelihood_history': gmm.log_likelihood_history,
            'n_iter': gmm.n_iter,
            'converged': gmm.converged,
            'final_ll': gmm.log_likelihood_history[-1]
        }
        
        print(f"  Iterations: {gmm.n_iter}")
        print(f"  Converged: {gmm.converged}")
        print(f"  Final log-likelihood: {gmm.log_likelihood_history[-1]:.4f}")
    
    # Plot convergence comparison
    plot_convergence_comparison(convergence_results)
    
    return convergence_results

def plot_convergence_comparison(results):
    """
    Plot convergence comparison across datasets.
    
    Parameters:
    -----------
    results : dict
        Convergence results for different datasets
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Log-likelihood evolution
    for name, result in results.items():
        ax1.plot(result['log_likelihood_history'], linewidth=2, marker='o', 
                markersize=4, label=f'{name.title()} (iter: {result["n_iter"]})')
    
    ax1.set_xlabel('EM Iteration')
    ax1.set_ylabel('Log-Likelihood')
    ax1.set_title('Convergence Comparison Across Datasets')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Convergence speed
    datasets = list(results.keys())
    iterations = [results[name]['n_iter'] for name in datasets]
    converged = [results[name]['converged'] for name in datasets]
    
    colors = ['green' if c else 'red' for c in converged]
    bars = ax2.bar(datasets, iterations, color=colors, alpha=0.7)
    
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Iterations to Convergence')
    ax2.set_title('Convergence Speed Comparison')
    ax2.grid(True, alpha=0.3)
    
    # Add convergence status annotations
    for bar, conv in zip(bars, converged):
        height = bar.get_height()
        status = 'Converged' if conv else 'Max iter'
        ax2.annotate(status, xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('plots/gmm_convergence_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function to run comprehensive GMM EM analysis.
    """
    print("🔄 EXPECTATION-MAXIMIZATION: GMM GENERALIZED FRAMEWORK")
    print("=" * 80)
    
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    # 1. Compare implementations
    gmm_em, gmm_sklearn, X, y_true = compare_implementations()
    
    # 2. Demonstrate covariance types
    covariance_results = demonstrate_covariance_types()
    
    # 3. Analyze convergence behavior
    convergence_results = analyze_convergence_behavior()
    
    # 4. Plot convergence for main model
    gmm_em.plot_convergence("Generalized EM GMM Convergence", 'plots/gmm_em_convergence.png')
    
    print("\n✅ GMM GENERALIZED EM ANALYSIS COMPLETE!")
    print("📁 Check the 'plots' folder for generated visualizations.")
    print("\n📋 SUMMARY:")
    print(f"• Demonstrated generalized EM framework with abstract base class")
    print(f"• Implemented GMM as concrete specialization of EM algorithm")
    print(f"• Compared with scikit-learn implementation for validation")
    print(f"• Analyzed different covariance types: full, diagonal, spherical")
    print(f"• Studied convergence behavior across multiple datasets")
    
    return gmm_em, gmm_sklearn, covariance_results, convergence_results

if __name__ == "__main__":
    main() 