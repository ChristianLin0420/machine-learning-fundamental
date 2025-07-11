"""
Principal Component Analysis (PCA) Implementation from Scratch

This module implements PCA and Kernel PCA from scratch with:
- Standard PCA using eigendecomposition and SVD
- Data standardization and centering
- Dimensionality reduction and reconstruction
- Explained variance analysis
- Kernel PCA with RBF kernel
- Comprehensive visualization and evaluation
- Comparison with scikit-learn
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_digits, make_moons, make_circles, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class PCAScratch:
    """
    Principal Component Analysis implementation from scratch.
    
    Implements both eigendecomposition and SVD approaches for PCA with
    comprehensive analysis of variance explained and reconstruction capabilities.
    """
    
    def __init__(self, n_components=None, method='eigen'):
        """
        Initialize PCA.
        
        Args:
            n_components (int): Number of principal components to keep
            method (str): Method to use ('eigen' for eigendecomposition, 'svd' for SVD)
        """
        self.n_components = n_components
        self.method = method
        
        # Fitted parameters
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.n_features_ = None
        self.n_samples_ = None
    
    def _center_data(self, X):
        """Center the data by subtracting the mean."""
        self.mean_ = np.mean(X, axis=0)
        return X - self.mean_
    
    def _compute_covariance_matrix(self, X_centered):
        """Compute the covariance matrix."""
        n_samples = X_centered.shape[0]
        return (X_centered.T @ X_centered) / (n_samples - 1)
    
    def _eigendecomposition_method(self, X_centered):
        """Perform PCA using eigendecomposition of covariance matrix."""
        # Compute covariance matrix
        cov_matrix = self._compute_covariance_matrix(X_centered)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Store results
        self.explained_variance_ = eigenvalues
        self.components_ = eigenvectors.T  # Each row is a principal component
        
        return eigenvalues, eigenvectors
    
    def _svd_method(self, X_centered):
        """Perform PCA using Singular Value Decomposition."""
        # SVD: X = U @ S @ V^T
        U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Eigenvalues are squares of singular values divided by (n-1)
        eigenvalues = (s ** 2) / (self.n_samples_ - 1)
        
        # Principal components are rows of V^T
        eigenvectors = Vt
        
        # Store results
        self.explained_variance_ = eigenvalues
        self.singular_values_ = s
        self.components_ = eigenvectors  # Each row is a principal component
        
        return eigenvalues, eigenvectors
    
    def fit(self, X):
        """
        Fit PCA to the data.
        
        Args:
            X (np.array): Input data of shape (n_samples, n_features)
        """
        X = np.array(X)
        self.n_samples_, self.n_features_ = X.shape
        
        # Center the data
        X_centered = self._center_data(X)
        
        # Choose method for PCA
        if self.method == 'eigen':
            eigenvalues, eigenvectors = self._eigendecomposition_method(X_centered)
        elif self.method == 'svd':
            eigenvalues, eigenvectors = self._svd_method(X_centered)
        else:
            raise ValueError("Method must be 'eigen' or 'svd'")
        
        # Set number of components if not specified
        if self.n_components is None:
            self.n_components = min(self.n_samples_, self.n_features_)
        
        # Keep only the requested number of components
        self.components_ = self.components_[:self.n_components]
        self.explained_variance_ = self.explained_variance_[:self.n_components]
        
        # Calculate explained variance ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        
        return self
    
    def transform(self, X):
        """
        Transform data to lower dimensional space.
        
        Args:
            X (np.array): Data to transform
            
        Returns:
            np.array: Transformed data
        """
        if self.components_ is None:
            raise ValueError("PCA must be fitted before transforming data")
        
        X = np.array(X)
        
        # Center the data using the training mean
        X_centered = X - self.mean_
        
        # Project onto principal components
        return X_centered @ self.components_.T
    
    def fit_transform(self, X):
        """Fit PCA and transform the data."""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_transformed):
        """
        Reconstruct data from lower dimensional representation.
        
        Args:
            X_transformed (np.array): Transformed data
            
        Returns:
            np.array: Reconstructed data
        """
        if self.components_ is None:
            raise ValueError("PCA must be fitted before inverse transforming data")
        
        X_transformed = np.array(X_transformed)
        
        # Reconstruct in original space
        X_reconstructed = X_transformed @ self.components_
        
        # Add back the mean
        return X_reconstructed + self.mean_
    
    def reconstruction_error(self, X):
        """
        Calculate reconstruction error.
        
        Args:
            X (np.array): Original data
            
        Returns:
            float: Mean squared reconstruction error
        """
        X_transformed = self.transform(X)
        X_reconstructed = self.inverse_transform(X_transformed)
        return mean_squared_error(X, X_reconstructed)
    
    def explained_variance_cumsum(self):
        """Calculate cumulative explained variance ratio."""
        return np.cumsum(self.explained_variance_ratio_)

class KernelPCAScratch:
    """
    Kernel PCA implementation from scratch.
    
    Implements nonlinear dimensionality reduction using the kernel trick
    with RBF (Gaussian) kernel.
    """
    
    def __init__(self, n_components=2, kernel='rbf', gamma=1.0):
        """
        Initialize Kernel PCA.
        
        Args:
            n_components (int): Number of principal components
            kernel (str): Kernel type ('rbf' for Gaussian)
            gamma (float): Kernel parameter for RBF
        """
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        
        # Fitted parameters
        self.alphas_ = None
        self.lambdas_ = None
        self.X_fit_ = None
        self.K_fit_ = None
        self.n_samples_ = None
    
    def _rbf_kernel(self, X1, X2):
        """
        Compute RBF (Gaussian) kernel matrix.
        
        K(x, y) = exp(-gamma * ||x - y||^2)
        """
        # Compute squared distances efficiently
        X1_norm = np.sum(X1**2, axis=1, keepdims=True)
        X2_norm = np.sum(X2**2, axis=1, keepdims=True)
        squared_distances = X1_norm + X2_norm.T - 2 * X1 @ X2.T
        
        return np.exp(-self.gamma * squared_distances)
    
    def _center_kernel_matrix(self, K):
        """
        Center the kernel matrix using the double-centering trick.
        
        K_centered = K - 1_n K - K 1_n + 1_n K 1_n
        where 1_n is a matrix of ones divided by n.
        """
        n = K.shape[0]
        one_n = np.ones((n, n)) / n
        
        K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
        
        return K_centered
    
    def fit(self, X):
        """
        Fit Kernel PCA to the data.
        
        Args:
            X (np.array): Input data
        """
        X = np.array(X)
        self.n_samples_ = X.shape[0]
        self.X_fit_ = X.copy()
        
        # Compute kernel matrix
        K = self._rbf_kernel(X, X)
        
        # Center the kernel matrix
        K_centered = self._center_kernel_matrix(K)
        self.K_fit_ = K_centered
        
        # Eigendecomposition of centered kernel matrix
        eigenvalues, eigenvectors = np.linalg.eigh(K_centered)
        
        # Sort by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Keep only positive eigenvalues and corresponding eigenvectors
        positive_idx = eigenvalues > 1e-12
        eigenvalues = eigenvalues[positive_idx]
        eigenvectors = eigenvectors[:, positive_idx]
        
        # Normalize eigenvectors
        self.lambdas_ = eigenvalues[:self.n_components]
        self.alphas_ = eigenvectors[:, :self.n_components]
        
        # Normalize: alpha_i = v_i / sqrt(lambda_i)
        for i in range(self.n_components):
            if self.lambdas_[i] > 0:
                self.alphas_[:, i] /= np.sqrt(self.lambdas_[i])
        
        return self
    
    def transform(self, X):
        """
        Transform data to kernel PCA space.
        
        Args:
            X (np.array): Data to transform
            
        Returns:
            np.array: Transformed data
        """
        if self.alphas_ is None:
            raise ValueError("Kernel PCA must be fitted before transforming data")
        
        X = np.array(X)
        
        # Compute kernel matrix between X and training data
        K = self._rbf_kernel(X, self.X_fit_)
        
        # Center the kernel matrix (approximate centering for new data)
        # This is a simplified centering - full centering would require 
        # storing more information from training
        n_train = self.X_fit_.shape[0]
        K_mean_train = np.mean(self.K_fit_, axis=0)
        K_mean_new = np.mean(K, axis=1, keepdims=True)
        K_mean_total = np.mean(self.K_fit_)
        
        K_centered = K - K_mean_new - K_mean_train + K_mean_total
        
        # Transform to principal component space
        return K_centered @ self.alphas_
    
    def fit_transform(self, X):
        """Fit Kernel PCA and transform the data."""
        self.fit(X)
        
        # For training data, use the centered kernel matrix directly
        return self.K_fit_ @ self.alphas_

def plot_pca_2d(X_transformed, y, title="PCA 2D Projection", filename=None):
    """
    Plot 2D PCA projection with class labels.
    
    Args:
        X_transformed (np.array): 2D projected data
        y (np.array): Class labels
        title (str): Plot title
        filename (str): Save filename
    """
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot with different colors for each class
    unique_classes = np.unique(y)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))
    
    for i, class_label in enumerate(unique_classes):
        mask = y == class_label
        plt.scatter(X_transformed[mask, 0], X_transformed[mask, 1], 
                   c=[colors[i]], label=f'Class {class_label}', 
                   alpha=0.7, s=50)
    
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if filename:
        plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_scree_plot(explained_variance_ratio, title="Scree Plot", filename=None):
    """
    Plot scree plot showing explained variance by component.
    
    Args:
        explained_variance_ratio (np.array): Explained variance ratios
        title (str): Plot title
        filename (str): Save filename
    """
    n_components = len(explained_variance_ratio)
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Individual explained variance
    ax1.bar(range(1, n_components + 1), explained_variance_ratio, 
            alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Individual Explained Variance')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative explained variance
    ax2.plot(range(1, n_components + 1), cumulative_variance, 
             'bo-', linewidth=2, markersize=8)
    ax2.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% Variance')
    ax2.axhline(y=0.90, color='orange', linestyle='--', alpha=0.7, label='90% Variance')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance Ratio')
    ax2.set_title('Cumulative Explained Variance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if filename:
        plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_reconstruction_error(X, pca_list, component_range, title="Reconstruction Error", filename=None):
    """
    Plot reconstruction error vs number of components.
    
    Args:
        X (np.array): Original data
        pca_list (list): List of fitted PCA objects
        component_range (list): Range of component numbers
        title (str): Plot title
        filename (str): Save filename
    """
    errors = []
    
    for pca in pca_list:
        error = pca.reconstruction_error(X)
        errors.append(error)
    
    plt.figure(figsize=(10, 6))
    plt.plot(component_range, errors, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Components')
    plt.ylabel('Mean Squared Reconstruction Error')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    
    if filename:
        plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_digit_reconstruction(X_original, X_reconstructed, indices=None, title="Digit Reconstruction", filename=None):
    """
    Plot original vs reconstructed digit images.
    
    Args:
        X_original (np.array): Original digit data
        X_reconstructed (np.array): Reconstructed digit data
        indices (list): Indices of digits to show
        title (str): Plot title
        filename (str): Save filename
    """
    if indices is None:
        indices = np.random.choice(len(X_original), 5, replace=False)
    
    fig, axes = plt.subplots(2, len(indices), figsize=(15, 6))
    
    for i, idx in enumerate(indices):
        # Original image
        axes[0, i].imshow(X_original[idx].reshape(8, 8), cmap='gray')
        axes[0, i].set_title(f'Original {idx}')
        axes[0, i].axis('off')
        
        # Reconstructed image
        axes[1, i].imshow(X_reconstructed[idx].reshape(8, 8), cmap='gray')
        axes[1, i].set_title(f'Reconstructed {idx}')
        axes[1, i].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if filename:
        plt.savefig(f'plots/{filename}', dpi=300, bbox_inches='tight')
    
    plt.show()

def compare_with_sklearn(X, y, n_components=2):
    """Compare custom PCA implementation with scikit-learn."""
    print(f"\nComparison with sklearn PCA (n_components={n_components}):")
    print("=" * 60)
    
    # Custom implementation
    pca_custom = PCAScratch(n_components=n_components, method='svd')
    X_custom = pca_custom.fit_transform(X)
    
    # Sklearn implementation
    pca_sklearn = PCA(n_components=n_components)
    X_sklearn = pca_sklearn.fit_transform(X)
    
    print(f"Custom PCA explained variance ratio: {pca_custom.explained_variance_ratio_}")
    print(f"Sklearn PCA explained variance ratio: {pca_sklearn.explained_variance_ratio_}")
    
    print(f"\nCustom PCA cumulative variance: {np.sum(pca_custom.explained_variance_ratio_):.6f}")
    print(f"Sklearn PCA cumulative variance: {np.sum(pca_sklearn.explained_variance_ratio_):.6f}")
    
    # Compare reconstruction error
    custom_error = pca_custom.reconstruction_error(X)
    
    X_sklearn_reconstructed = pca_sklearn.inverse_transform(X_sklearn)
    sklearn_error = mean_squared_error(X, X_sklearn_reconstructed)
    
    print(f"\nCustom PCA reconstruction error: {custom_error:.8f}")
    print(f"Sklearn PCA reconstruction error: {sklearn_error:.8f}")
    print(f"Reconstruction error difference: {abs(custom_error - sklearn_error):.10f}")
    
    return {
        'custom_explained_variance': pca_custom.explained_variance_ratio_,
        'sklearn_explained_variance': pca_sklearn.explained_variance_ratio_,
        'custom_error': custom_error,
        'sklearn_error': sklearn_error
    }

def main():
    """Main function to run PCA experiments."""
    print("="*70)
    print("PRINCIPAL COMPONENT ANALYSIS IMPLEMENTATION FROM SCRATCH")
    print("="*70)
    
    # 1. PCA on Iris Dataset
    print("\n1. PCA ON IRIS DATASET (4D → 2D)")
    print("-" * 50)
    
    # Load Iris dataset
    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target
    
    # Standardize the data
    scaler_iris = StandardScaler()
    X_iris_scaled = scaler_iris.fit_transform(X_iris)
    
    # Apply PCA
    pca_iris = PCAScratch(n_components=2, method='svd')
    X_iris_pca = pca_iris.fit_transform(X_iris_scaled)
    
    print(f"Original Iris shape: {X_iris.shape}")
    print(f"PCA Iris shape: {X_iris_pca.shape}")
    print(f"Explained variance ratio: {pca_iris.explained_variance_ratio_}")
    print(f"Cumulative explained variance: {np.sum(pca_iris.explained_variance_ratio_):.4f}")
    
    # Plot 2D projection
    plot_pca_2d(X_iris_pca, y_iris, 
                "PCA 2D Projection - Iris Dataset", 
                "pca_iris_2d.png")
    
    # Plot scree plot for all components
    pca_iris_full = PCAScratch(n_components=4, method='svd')
    pca_iris_full.fit(X_iris_scaled)
    plot_scree_plot(pca_iris_full.explained_variance_ratio_, 
                   "Scree Plot - Iris Dataset", 
                   "pca_iris_scree.png")
    
    # 2. PCA on Digits Dataset
    print("\n2. PCA ON DIGITS DATASET (64D → 2D)")
    print("-" * 50)
    
    # Load Digits dataset
    digits = load_digits()
    X_digits, y_digits = digits.data, digits.target
    
    # Standardize the data
    scaler_digits = StandardScaler()
    X_digits_scaled = scaler_digits.fit_transform(X_digits)
    
    # Apply PCA
    pca_digits = PCAScratch(n_components=2, method='svd')
    X_digits_pca = pca_digits.fit_transform(X_digits_scaled)
    
    print(f"Original Digits shape: {X_digits.shape}")
    print(f"PCA Digits shape: {X_digits_pca.shape}")
    print(f"Explained variance ratio: {pca_digits.explained_variance_ratio_}")
    print(f"Cumulative explained variance: {np.sum(pca_digits.explained_variance_ratio_):.4f}")
    
    # Plot 2D projection
    plot_pca_2d(X_digits_pca, y_digits, 
                "PCA 2D Projection - Digits Dataset", 
                "pca_digits_2d.png")
    
    # 3. Explained Variance Analysis
    print("\n3. EXPLAINED VARIANCE ANALYSIS")
    print("-" * 50)
    
    # Analyze different numbers of components for digits
    component_range = [1, 2, 5, 10, 20, 30, 40, 50]
    pca_list = []
    
    print("Component analysis for Digits dataset:")
    print("Components\tExpl. Var.\tCumulative\tReconstruction Error")
    print("-" * 65)
    
    for n_comp in component_range:
        pca = PCAScratch(n_components=n_comp, method='svd')
        pca.fit(X_digits_scaled)
        pca_list.append(pca)
        
        cum_var = np.sum(pca.explained_variance_ratio_)
        recon_error = pca.reconstruction_error(X_digits_scaled)
        
        print(f"{n_comp:^10}\t{pca.explained_variance_ratio_[0]:.4f}\t\t{cum_var:.4f}\t\t{recon_error:.6f}")
    
    # Plot reconstruction error
    plot_reconstruction_error(X_digits_scaled, pca_list, component_range,
                            "Reconstruction Error vs Components - Digits",
                            "pca_digits_reconstruction_error.png")
    
    # Full scree plot for digits (first 20 components)
    pca_digits_full = PCAScratch(n_components=20, method='svd')
    pca_digits_full.fit(X_digits_scaled)
    plot_scree_plot(pca_digits_full.explained_variance_ratio_, 
                   "Scree Plot - Digits Dataset (20 components)", 
                   "pca_digits_scree.png")
    
    # 4. Digit Reconstruction Visualization
    print("\n4. DIGIT RECONSTRUCTION VISUALIZATION")
    print("-" * 50)
    
    # Reconstruct digits with different numbers of components
    for n_comp in [2, 5, 10, 20]:
        pca_recon = PCAScratch(n_components=n_comp, method='svd')
        X_transformed = pca_recon.fit_transform(X_digits_scaled)
        X_reconstructed = pca_recon.inverse_transform(X_transformed)
        
        # Convert back to original scale
        X_reconstructed_original = scaler_digits.inverse_transform(X_reconstructed)
        
        plot_digit_reconstruction(X_digits, X_reconstructed_original,
                                indices=[0, 10, 50, 100, 200],
                                title=f"Digit Reconstruction with {n_comp} Components",
                                filename=f"pca_digits_reconstruction_{n_comp}.png")
        
        recon_error = mean_squared_error(X_digits, X_reconstructed_original)
        print(f"Reconstruction error with {n_comp} components: {recon_error:.6f}")
    
    # 5. Kernel PCA Analysis
    print("\n5. KERNEL PCA ANALYSIS")
    print("-" * 50)
    
    # Generate nonlinear data
    print("Generating nonlinear datasets for Kernel PCA...")
    
    # Moon dataset
    X_moons, y_moons = make_moons(n_samples=300, noise=0.1, random_state=42)
    
    # Circular dataset
    X_circles, y_circles = make_circles(n_samples=300, noise=0.1, factor=0.6, random_state=42)
    
    # Apply standard PCA and Kernel PCA to moons
    print("\nMoon dataset analysis:")
    
    # Standard PCA
    pca_moons = PCAScratch(n_components=2, method='svd')
    X_moons_pca = pca_moons.fit_transform(X_moons)
    
    # Kernel PCA
    kpca_moons = KernelPCAScratch(n_components=2, gamma=1.0)
    X_moons_kpca = kpca_moons.fit_transform(X_moons)
    
    # Plot comparisons
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Original data
    scatter1 = axes[0].scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap='viridis')
    axes[0].set_title('Original Moon Data')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    
    # Standard PCA
    scatter2 = axes[1].scatter(X_moons_pca[:, 0], X_moons_pca[:, 1], c=y_moons, cmap='viridis')
    axes[1].set_title('Standard PCA')
    axes[1].set_xlabel('PC 1')
    axes[1].set_ylabel('PC 2')
    
    # Kernel PCA
    scatter3 = axes[2].scatter(X_moons_kpca[:, 0], X_moons_kpca[:, 1], c=y_moons, cmap='viridis')
    axes[2].set_title('Kernel PCA (RBF)')
    axes[2].set_xlabel('KPC 1')
    axes[2].set_ylabel('KPC 2')
    
    plt.tight_layout()
    plt.savefig('plots/pca_vs_kernel_pca_moons.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Apply to circles dataset
    print("\nCircular dataset analysis:")
    
    # Standard PCA
    pca_circles = PCAScratch(n_components=2, method='svd')
    X_circles_pca = pca_circles.fit_transform(X_circles)
    
    # Kernel PCA
    kpca_circles = KernelPCAScratch(n_components=2, gamma=1.0)
    X_circles_kpca = kpca_circles.fit_transform(X_circles)
    
    # Plot comparisons
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Original data
    scatter1 = axes[0].scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, cmap='viridis')
    axes[0].set_title('Original Circular Data')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    
    # Standard PCA
    scatter2 = axes[1].scatter(X_circles_pca[:, 0], X_circles_pca[:, 1], c=y_circles, cmap='viridis')
    axes[1].set_title('Standard PCA')
    axes[1].set_xlabel('PC 1')
    axes[1].set_ylabel('PC 2')
    
    # Kernel PCA
    scatter3 = axes[2].scatter(X_circles_kpca[:, 0], X_circles_kpca[:, 1], c=y_circles, cmap='viridis')
    axes[2].set_title('Kernel PCA (RBF)')
    axes[2].set_xlabel('KPC 1')
    axes[2].set_ylabel('KPC 2')
    
    plt.tight_layout()
    plt.savefig('plots/pca_vs_kernel_pca_circles.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. Method Comparison: Eigendecomposition vs SVD
    print("\n6. METHOD COMPARISON: EIGENDECOMPOSITION VS SVD")
    print("-" * 50)
    
    # Compare eigen vs SVD methods on digits
    pca_eigen = PCAScratch(n_components=10, method='eigen')
    pca_svd = PCAScratch(n_components=10, method='svd')
    
    X_eigen = pca_eigen.fit_transform(X_digits_scaled)
    X_svd = pca_svd.fit_transform(X_digits_scaled)
    
    print("Eigendecomposition vs SVD comparison:")
    print(f"Eigendecomposition explained variance: {pca_eigen.explained_variance_ratio_[:3]}")
    print(f"SVD explained variance: {pca_svd.explained_variance_ratio_[:3]}")
    print(f"Difference in explained variance: {np.abs(pca_eigen.explained_variance_ratio_ - pca_svd.explained_variance_ratio_)[:3]}")
    
    # 7. Comparison with Scikit-learn
    print("\n7. COMPARISON WITH SCIKIT-LEARN")
    print("-" * 50)
    
    # Compare on Iris dataset
    iris_comparison = compare_with_sklearn(X_iris_scaled, y_iris, n_components=2)
    
    # Compare on Digits dataset
    digits_comparison = compare_with_sklearn(X_digits_scaled, y_digits, n_components=10)
    
    print("\n" + "="*70)
    print("PCA EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("Generated visualizations:")
    print("- pca_iris_2d.png")
    print("- pca_iris_scree.png")
    print("- pca_digits_2d.png")
    print("- pca_digits_scree.png")
    print("- pca_digits_reconstruction_error.png")
    print("- pca_digits_reconstruction_[2,5,10,20].png")
    print("- pca_vs_kernel_pca_moons.png")
    print("- pca_vs_kernel_pca_circles.png")

if __name__ == "__main__":
    main() 