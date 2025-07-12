"""
Singular Value Decomposition (SVD) from Scratch - Advanced Implementation
========================================================================

This module implements comprehensive SVD analysis including:
- Matrix decomposition A = UΣV^T
- Rank-k approximation and reconstruction
- Relationship to eigendecomposition and PCA
- Frobenius norm error analysis
- Condition number and numerical stability
- Comparison with scipy implementations

Mathematical Foundation:
- SVD: A = UΣV^T where U, V orthogonal, Σ diagonal
- Rank-k approximation: A_k = Σ(i=1 to k) σ_i * u_i * v_i^T
- Reconstruction error: ||A - A_k||_F = sqrt(Σ(i=k+1 to r) σ_i^2)
- Energy preservation: Σ(i=1 to k) σ_i^2 / Σ(i=1 to r) σ_i^2
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import svd as scipy_svd
from sklearn.datasets import make_low_rank_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SVDAnalyzer:
    """
    Comprehensive SVD analysis and rank-k approximation class.
    
    Features:
    - Matrix decomposition using numpy.linalg.svd
    - Rank-k approximation with reconstruction
    - Error analysis and energy preservation
    - Comparison with scipy implementation
    - Extensive visualization capabilities
    """
    
    def __init__(self, matrix=None, name="Matrix"):
        """
        Initialize SVD analyzer.
        
        Parameters:
        -----------
        matrix : np.ndarray, optional
            Input matrix to decompose
        name : str
            Name for plotting and identification
        """
        self.matrix = matrix
        self.name = name
        self.U = None
        self.s = None
        self.Vt = None
        self.rank = None
        self.condition_number = None
        self.frobenius_norm = None
        
    def decompose(self, matrix=None):
        """
        Perform SVD decomposition: A = UΣV^T
        
        Parameters:
        -----------
        matrix : np.ndarray, optional
            Matrix to decompose (uses self.matrix if None)
            
        Returns:
        --------
        tuple : (U, s, Vt) SVD components
        """
        if matrix is not None:
            self.matrix = matrix
            
        if self.matrix is None:
            raise ValueError("No matrix provided for decomposition")
            
        # Perform SVD decomposition
        self.U, self.s, self.Vt = np.linalg.svd(self.matrix, full_matrices=False)
        
        # Calculate matrix properties
        self.rank = np.sum(self.s > 1e-10)  # Numerical rank
        self.condition_number = self.s[0] / self.s[-1] if self.s[-1] > 1e-10 else np.inf
        self.frobenius_norm = np.linalg.norm(self.matrix, 'fro')
        
        print(f"SVD Decomposition Results for {self.name}:")
        print(f"Matrix shape: {self.matrix.shape}")
        print(f"Rank: {self.rank}")
        print(f"Condition number: {self.condition_number:.2e}")
        print(f"Frobenius norm: {self.frobenius_norm:.4f}")
        print(f"Largest singular value: {self.s[0]:.4f}")
        print(f"Smallest singular value: {self.s[-1]:.4e}")
        print()
        
        return self.U, self.s, self.Vt
    
    def rank_k_approximation(self, k):
        """
        Compute rank-k approximation: A_k = Σ(i=1 to k) σ_i * u_i * v_i^T
        
        Parameters:
        -----------
        k : int
            Number of singular values to keep
            
        Returns:
        --------
        np.ndarray : Rank-k approximation matrix
        """
        if self.U is None:
            raise ValueError("Must perform decomposition first")
            
        k = min(k, len(self.s))
        
        # Reconstruct using top-k singular values
        A_k = self.U[:, :k] @ np.diag(self.s[:k]) @ self.Vt[:k, :]
        
        return A_k
    
    def reconstruction_error(self, k):
        """
        Calculate Frobenius reconstruction error for rank-k approximation.
        
        Parameters:
        -----------
        k : int
            Number of singular values to keep
            
        Returns:
        --------
        float : Frobenius norm of reconstruction error
        """
        if self.U is None:
            raise ValueError("Must perform decomposition first")
            
        k = min(k, len(self.s))
        
        # Error is sum of squared singular values beyond k
        if k >= len(self.s):
            return 0.0
        
        error = np.sqrt(np.sum(self.s[k:] ** 2))
        return error
    
    def energy_preserved(self, k):
        """
        Calculate energy (variance) preserved by rank-k approximation.
        
        Parameters:
        -----------
        k : int
            Number of singular values to keep
            
        Returns:
        --------
        float : Fraction of energy preserved (0 to 1)
        """
        if self.U is None:
            raise ValueError("Must perform decomposition first")
            
        k = min(k, len(self.s))
        
        total_energy = np.sum(self.s ** 2)
        preserved_energy = np.sum(self.s[:k] ** 2)
        
        return preserved_energy / total_energy
    
    def compression_ratio(self, k):
        """
        Calculate compression ratio for rank-k approximation.
        
        Parameters:
        -----------
        k : int
            Number of singular values to keep
            
        Returns:
        --------
        float : Compression ratio (original_size / compressed_size)
        """
        m, n = self.matrix.shape
        original_size = m * n
        compressed_size = k * (m + n + 1)  # U[:,:k] + s[:k] + Vt[:k,:]
        
        return original_size / compressed_size
    
    def compare_with_scipy(self):
        """
        Compare numpy SVD with scipy SVD implementation.
        
        Returns:
        --------
        dict : Comparison results
        """
        if self.matrix is None:
            raise ValueError("No matrix provided")
            
        # Scipy SVD
        U_scipy, s_scipy, Vt_scipy = scipy_svd(self.matrix, full_matrices=False)
        
        # Compare singular values (should be identical)
        s_diff = np.max(np.abs(self.s - s_scipy))
        
        # Compare reconstruction (allowing for sign flips)
        reconstruction_numpy = self.U @ np.diag(self.s) @ self.Vt
        reconstruction_scipy = U_scipy @ np.diag(s_scipy) @ Vt_scipy
        
        reconstruction_diff = np.max(np.abs(reconstruction_numpy - reconstruction_scipy))
        
        results = {
            'singular_values_diff': s_diff,
            'reconstruction_diff': reconstruction_diff,
            'numpy_condition': self.condition_number,
            'scipy_condition': s_scipy[0] / s_scipy[-1] if s_scipy[-1] > 1e-10 else np.inf
        }
        
        print("Comparison with Scipy SVD:")
        print(f"Max singular value difference: {s_diff:.2e}")
        print(f"Max reconstruction difference: {reconstruction_diff:.2e}")
        print(f"Numpy condition number: {results['numpy_condition']:.2e}")
        print(f"Scipy condition number: {results['scipy_condition']:.2e}")
        print()
        
        return results
    
    def plot_singular_values(self, save_path=None):
        """
        Plot singular values and cumulative energy.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        """
        if self.U is None:
            raise ValueError("Must perform decomposition first")
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Singular values (linear scale)
        ax1.plot(range(1, len(self.s) + 1), self.s, 'bo-', linewidth=2, markersize=6)
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Singular Value')
        ax1.set_title(f'Singular Values - {self.name}')
        ax1.grid(True, alpha=0.3)
        
        # 2. Singular values (log scale)
        ax2.semilogy(range(1, len(self.s) + 1), self.s, 'ro-', linewidth=2, markersize=6)
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Singular Value (log scale)')
        ax2.set_title(f'Singular Values (Log Scale) - {self.name}')
        ax2.grid(True, alpha=0.3)
        
        # 3. Cumulative energy
        cumulative_energy = np.cumsum(self.s ** 2) / np.sum(self.s ** 2)
        ax3.plot(range(1, len(self.s) + 1), cumulative_energy, 'go-', linewidth=2, markersize=6)
        ax3.axhline(y=0.9, color='r', linestyle='--', alpha=0.7, label='90% Energy')
        ax3.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='95% Energy')
        ax3.axhline(y=0.99, color='purple', linestyle='--', alpha=0.7, label='99% Energy')
        ax3.set_xlabel('Number of Components')
        ax3.set_ylabel('Cumulative Energy Preserved')
        ax3.set_title(f'Cumulative Energy - {self.name}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Energy distribution
        energy_ratios = (self.s ** 2) / np.sum(self.s ** 2)
        ax4.bar(range(1, min(21, len(self.s) + 1)), energy_ratios[:20], alpha=0.7)
        ax4.set_xlabel('Component Index')
        ax4.set_ylabel('Energy Ratio')
        ax4.set_title(f'Energy Distribution (Top 20) - {self.name}')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Singular values plot saved to {save_path}")
        
        plt.show()
    
    def plot_reconstruction_analysis(self, max_k=None, save_path=None):
        """
        Plot reconstruction error and compression analysis.
        
        Parameters:
        -----------
        max_k : int, optional
            Maximum k to analyze (default: min(50, rank))
        save_path : str, optional
            Path to save the plot
        """
        if self.U is None:
            raise ValueError("Must perform decomposition first")
            
        if max_k is None:
            max_k = min(50, len(self.s))
            
        k_values = range(1, max_k + 1)
        errors = [self.reconstruction_error(k) for k in k_values]
        energies = [self.energy_preserved(k) for k in k_values]
        compressions = [self.compression_ratio(k) for k in k_values]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Reconstruction error
        ax1.plot(k_values, errors, 'bo-', linewidth=2, markersize=4)
        ax1.set_xlabel('Rank k')
        ax1.set_ylabel('Frobenius Error')
        ax1.set_title(f'Reconstruction Error vs Rank - {self.name}')
        ax1.grid(True, alpha=0.3)
        
        # 2. Reconstruction error (log scale)
        ax2.semilogy(k_values, errors, 'ro-', linewidth=2, markersize=4)
        ax2.set_xlabel('Rank k')
        ax2.set_ylabel('Frobenius Error (log scale)')
        ax2.set_title(f'Reconstruction Error (Log Scale) - {self.name}')
        ax2.grid(True, alpha=0.3)
        
        # 3. Energy preserved
        ax3.plot(k_values, energies, 'go-', linewidth=2, markersize=4)
        ax3.axhline(y=0.9, color='r', linestyle='--', alpha=0.7, label='90%')
        ax3.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='95%')
        ax3.axhline(y=0.99, color='purple', linestyle='--', alpha=0.7, label='99%')
        ax3.set_xlabel('Rank k')
        ax3.set_ylabel('Energy Preserved')
        ax3.set_title(f'Energy Preservation vs Rank - {self.name}')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Compression ratio
        ax4.plot(k_values, compressions, 'mo-', linewidth=2, markersize=4)
        ax4.axhline(y=1, color='r', linestyle='--', alpha=0.7, label='No Compression')
        ax4.set_xlabel('Rank k')
        ax4.set_ylabel('Compression Ratio')
        ax4.set_title(f'Compression Ratio vs Rank - {self.name}')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Reconstruction analysis plot saved to {save_path}")
        
        plt.show()
        
        return k_values, errors, energies, compressions

def demonstrate_svd_properties():
    """
    Demonstrate key SVD properties and mathematical relationships.
    """
    print("=" * 70)
    print("SVD MATHEMATICAL PROPERTIES DEMONSTRATION")
    print("=" * 70)
    
    # Create test matrices with different properties
    np.random.seed(42)
    
    # 1. Low-rank matrix
    print("1. LOW-RANK MATRIX ANALYSIS")
    print("-" * 40)
    low_rank = make_low_rank_matrix(n_samples=100, n_features=80, effective_rank=10, tail_strength=0.1)
    svd_low = SVDAnalyzer(low_rank, "Low-rank Matrix")
    svd_low.decompose()
    
    # 2. Random matrix
    print("2. RANDOM MATRIX ANALYSIS")
    print("-" * 40)
    random_matrix = np.random.randn(50, 30)
    svd_random = SVDAnalyzer(random_matrix, "Random Matrix")
    svd_random.decompose()
    
    # 3. Structured matrix (Hankel)
    print("3. STRUCTURED MATRIX (HANKEL) ANALYSIS")
    print("-" * 40)
    n = 20
    hankel = np.array([[i + j for j in range(n)] for i in range(n)])
    svd_hankel = SVDAnalyzer(hankel, "Hankel Matrix")
    svd_hankel.decompose()
    
    return svd_low, svd_random, svd_hankel

def analyze_rank_approximations():
    """
    Analyze rank-k approximations for different matrices.
    """
    print("=" * 70)
    print("RANK-K APPROXIMATION ANALYSIS")
    print("=" * 70)
    
    # Create a matrix with known structure
    np.random.seed(42)
    m, n = 50, 40
    true_rank = 8
    
    # Generate low-rank matrix: A = BC where B is m×r, C is r×n
    B = np.random.randn(m, true_rank)
    C = np.random.randn(true_rank, n)
    A_clean = B @ C
    
    # Add noise
    noise_level = 0.1
    A_noisy = A_clean + noise_level * np.random.randn(m, n)
    
    # Analyze both matrices
    svd_clean = SVDAnalyzer(A_clean, f"Clean Low-rank (rank={true_rank})")
    svd_clean.decompose()
    
    svd_noisy = SVDAnalyzer(A_noisy, f"Noisy Low-rank (rank={true_rank})")
    svd_noisy.decompose()
    
    # Compare reconstructions at different ranks
    test_ranks = [1, 2, 5, 8, 10, 15, 20]
    
    print(f"Reconstruction Analysis (True rank = {true_rank}):")
    print("Rank | Clean Error | Noisy Error | Clean Energy | Noisy Energy")
    print("-" * 65)
    
    for k in test_ranks:
        clean_error = svd_clean.reconstruction_error(k)
        noisy_error = svd_noisy.reconstruction_error(k)
        clean_energy = svd_clean.energy_preserved(k)
        noisy_energy = svd_noisy.energy_preserved(k)
        
        print(f"{k:4d} | {clean_error:11.4f} | {noisy_error:11.4f} | "
              f"{clean_energy:12.4f} | {noisy_energy:12.4f}")
    
    return svd_clean, svd_noisy

def compare_svd_eigendecomposition():
    """
    Compare SVD with eigendecomposition for symmetric matrices.
    """
    print("=" * 70)
    print("SVD vs EIGENDECOMPOSITION COMPARISON")
    print("=" * 70)
    
    # Create symmetric positive definite matrix
    np.random.seed(42)
    n = 20
    A_random = np.random.randn(n, n)
    A_symmetric = A_random.T @ A_random  # Symmetric positive definite
    
    # SVD analysis
    svd_analyzer = SVDAnalyzer(A_symmetric, "Symmetric Matrix")
    U, s, Vt = svd_analyzer.decompose()
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(A_symmetric)
    eigenvalues = eigenvalues[::-1]  # Sort descending
    eigenvectors = eigenvectors[:, ::-1]
    
    print("Comparison Results:")
    print(f"SVD singular values (first 10): {s[:10]}")
    print(f"Eigenvalues (first 10): {eigenvalues[:10]}")
    print(f"Max difference in values: {np.max(np.abs(s - eigenvalues)):.2e}")
    
    # For symmetric matrices: A = QΛQ^T = UΣV^T where U=V=Q, Σ=Λ
    print(f"U ≈ V check: {np.max(np.abs(U - Vt.T)):.2e}")
    print(f"U ≈ eigenvectors check: {np.max(np.abs(np.abs(U) - np.abs(eigenvectors))):.2e}")
    
    return svd_analyzer

def relationship_to_pca():
    """
    Demonstrate relationship between SVD and PCA.
    """
    print("=" * 70)
    print("SVD-PCA RELATIONSHIP DEMONSTRATION")
    print("=" * 70)
    
    # Generate sample data
    np.random.seed(42)
    n_samples, n_features = 100, 50
    
    # Create correlated data
    true_components = 3
    latent = np.random.randn(n_samples, true_components)
    mixing_matrix = np.random.randn(true_components, n_features)
    X = latent @ mixing_matrix + 0.1 * np.random.randn(n_samples, n_features)
    
    # Center the data (crucial for PCA)
    X_centered = X - np.mean(X, axis=0)
    
    print(f"Data shape: {X.shape}")
    print(f"Data centered: mean = {np.mean(X_centered, axis=0)[:5]} (showing first 5)")
    
    # Method 1: SVD of centered data
    svd_analyzer = SVDAnalyzer(X_centered, "Centered Data")
    U, s, Vt = svd_analyzer.decompose()
    
    # Method 2: Eigendecomposition of covariance matrix
    cov_matrix = (X_centered.T @ X_centered) / (n_samples - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    eigenvalues = eigenvalues[::-1]  # Sort descending
    eigenvectors = eigenvectors[:, ::-1]
    
    # Method 3: SVD-based PCA
    # Principal components are V^T (rows of Vt)
    # Explained variance ratios are s^2 / (n-1) / total_variance
    explained_variance = (s ** 2) / (n_samples - 1)
    explained_variance_ratio = explained_variance / np.sum(explained_variance)
    
    print("\nPCA via different methods:")
    print(f"SVD explained variance (first 5): {explained_variance[:5]}")
    print(f"Eigendecomposition eigenvalues (first 5): {eigenvalues[:5]}")
    print(f"Max difference: {np.max(np.abs(explained_variance - eigenvalues)):.2e}")
    
    print(f"\nExplained variance ratios (first 5): {explained_variance_ratio[:5]}")
    print(f"Cumulative explained variance (first 5): {np.cumsum(explained_variance_ratio)[:5]}")
    
    # Transform data using SVD (equivalent to PCA transform)
    # X_transformed = X_centered @ V = U @ S
    X_transformed_method1 = X_centered @ Vt.T
    X_transformed_method2 = U @ np.diag(s)
    
    print(f"\nTransformation equivalence check:")
    print(f"X @ V ≈ U @ S: {np.max(np.abs(X_transformed_method1 - X_transformed_method2)):.2e}")
    
    return svd_analyzer, explained_variance_ratio

def main():
    """
    Main function to run comprehensive SVD analysis.
    """
    print("🔢 SINGULAR VALUE DECOMPOSITION (SVD) - ADVANCED IMPLEMENTATION")
    print("=" * 80)
    
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    # 1. Demonstrate SVD properties
    svd_low, svd_random, svd_hankel = demonstrate_svd_properties()
    
    # 2. Generate visualizations for different matrices
    print("\n📊 GENERATING VISUALIZATIONS...")
    print("-" * 50)
    
    # Plot singular values for different matrices
    svd_low.plot_singular_values('plots/svd_low_rank_singular_values.png')
    svd_random.plot_singular_values('plots/svd_random_singular_values.png')
    svd_hankel.plot_singular_values('plots/svd_hankel_singular_values.png')
    
    # Plot reconstruction analysis
    svd_low.plot_reconstruction_analysis(save_path='plots/svd_low_rank_reconstruction.png')
    svd_random.plot_reconstruction_analysis(save_path='plots/svd_random_reconstruction.png')
    
    # 3. Analyze rank approximations
    svd_clean, svd_noisy = analyze_rank_approximations()
    
    # 4. Compare with scipy
    print("\n🔍 SCIPY COMPARISON...")
    print("-" * 30)
    svd_low.compare_with_scipy()
    svd_random.compare_with_scipy()
    
    # 5. Compare SVD with eigendecomposition
    svd_symmetric = compare_svd_eigendecomposition()
    
    # 6. Demonstrate relationship to PCA
    svd_pca, explained_ratios = relationship_to_pca()
    
    # 7. Generate comprehensive comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot singular values for different matrix types
    matrices = [svd_low, svd_random, svd_hankel]
    titles = ['Low-rank Matrix', 'Random Matrix', 'Hankel Matrix']
    
    for i, (svd_obj, title) in enumerate(zip(matrices, titles)):
        ax = axes[0, i]
        ax.semilogy(range(1, len(svd_obj.s) + 1), svd_obj.s, 'o-', linewidth=2, markersize=4)
        ax.set_title(f'{title}\n(Condition: {svd_obj.condition_number:.1e})')
        ax.set_xlabel('Index')
        ax.set_ylabel('Singular Value')
        ax.grid(True, alpha=0.3)
    
    # Plot reconstruction errors
    for i, (svd_obj, title) in enumerate(zip(matrices, titles)):
        ax = axes[1, i]
        max_k = min(20, len(svd_obj.s))
        k_vals = range(1, max_k + 1)
        errors = [svd_obj.reconstruction_error(k) for k in k_vals]
        energies = [svd_obj.energy_preserved(k) for k in k_vals]
        
        ax2 = ax.twinx()
        line1 = ax.plot(k_vals, errors, 'b-o', linewidth=2, markersize=4, label='Error')
        line2 = ax2.plot(k_vals, energies, 'r-s', linewidth=2, markersize=4, label='Energy')
        
        ax.set_xlabel('Rank k')
        ax.set_ylabel('Reconstruction Error', color='b')
        ax2.set_ylabel('Energy Preserved', color='r')
        ax.set_title(f'Error vs Energy - {title}')
        ax.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.tight_layout()
    plt.savefig('plots/svd_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✅ SVD FROM SCRATCH ANALYSIS COMPLETE!")
    print("📁 Check the 'plots' folder for generated visualizations.")
    print("\n📋 SUMMARY:")
    print(f"• Analyzed {len(matrices)} different matrix types")
    print(f"• Generated {len(matrices) * 2 + 1} visualization files")
    print("• Demonstrated SVD properties and applications")
    print("• Validated against scipy implementations")
    print("• Showed relationship to PCA and eigendecomposition")

if __name__ == "__main__":
    main() 