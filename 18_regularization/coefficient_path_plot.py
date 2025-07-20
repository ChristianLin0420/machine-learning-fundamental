"""
Coefficient Path Plotting
=========================

This module demonstrates how coefficient values change with regularization strength
using sklearn's lasso_path and enet_path functions. Shows the shrinkage and
feature selection effects of L1 and L2 regularization.

Key Visualizations:
- Lasso path: How coefficients shrink to zero (feature selection)
- Ridge path: How coefficients shrink uniformly (no feature selection)
- Comparative analysis of shrinkage patterns

Author: Machine Learning Fundamentals Course
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_diabetes, make_regression
from sklearn.linear_model import lasso_path, enet_path, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CoefficientPathAnalyzer:
    """
    Analyzes and visualizes coefficient paths for different regularization methods.
    
    This class provides tools for understanding how regularization affects
    coefficient values across different regularization strengths.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the coefficient path analyzer.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.X = None
        self.y = None
        self.feature_names = None
        
    def load_data(self, dataset='diabetes', n_features=None):
        """
        Load and prepare dataset for coefficient path analysis.
        
        Parameters:
        -----------
        dataset : str
            Dataset to use ('diabetes' or 'synthetic')
        n_features : int, optional
            Number of features for synthetic data
            
        Returns:
        --------
        dict : Dataset information
        """
        if dataset == 'diabetes':
            data = load_diabetes()
            X, y = data.data, data.target
            feature_names = data.feature_names
            print("🏥 Loading Diabetes Dataset for Coefficient Path Analysis")
            print("=" * 60)
        else:
            # Create synthetic dataset
            if n_features is None:
                n_features = 20
            
            X, y = make_regression(
                n_samples=200, n_features=n_features, n_informative=n_features//2,
                noise=0.1, random_state=self.random_state
            )
            feature_names = [f'Feature_{i+1}' for i in range(n_features)]
            print(f"🔬 Creating Synthetic Dataset ({n_features} features)")
            print("=" * 50)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        self.X = X_scaled
        self.y = y
        self.feature_names = feature_names
        
        print(f"Dataset shape: {X.shape}")
        print(f"Features: {len(feature_names)} total")
        print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
        
        return {
            'X': X_scaled,
            'y': y,
            'feature_names': feature_names,
            'n_features': len(feature_names)
        }
    
    def compute_lasso_path(self, n_alphas=100, alpha_min_ratio=1e-4):
        """
        Compute Lasso regularization path.
        
        Parameters:
        -----------
        n_alphas : int
            Number of alpha values along the path
        alpha_min_ratio : float
            Ratio of minimum to maximum alpha
            
        Returns:
        --------
        tuple : (alphas, coefficients)
        """
        print("\n🔍 Computing Lasso Path...")
        
        # Create alpha range manually
        alpha_max = np.linalg.norm(self.X.T @ self.y, ord=np.inf) / len(self.X)
        alphas = np.logspace(
            np.log10(alpha_max * alpha_min_ratio),
            np.log10(alpha_max),
            n_alphas
        )
        
        # Compute Lasso path
        alphas, coefs, _ = lasso_path(
            self.X, self.y, 
            alphas=alphas,
            max_iter=2000
        )
        
        print(f"Alpha range: [{alphas.min():.6f}, {alphas.max():.2f}]")
        print(f"Coefficient matrix shape: {coefs.shape}")
        
        return alphas, coefs
    
    def compute_ridge_path(self, n_alphas=100, alpha_min_ratio=1e-4):
        """
        Compute Ridge regularization path manually.
        
        Parameters:
        -----------
        n_alphas : int
            Number of alpha values along the path
        alpha_min_ratio : float
            Ratio of minimum to maximum alpha
            
        Returns:
        --------
        tuple : (alphas, coefficients)
        """
        print("\n🔍 Computing Ridge Path...")
        
        # Create alpha range for Ridge
        alpha_max = np.linalg.norm(self.X.T @ self.y, ord=np.inf) / len(self.X)
        alphas = np.logspace(
            np.log10(alpha_max * alpha_min_ratio),
            np.log10(alpha_max),
            n_alphas
        )
        
        # Compute Ridge path by fitting Ridge models
        coefs = []
        for alpha in alphas:
            ridge = Ridge(alpha=alpha, fit_intercept=False)
            ridge.fit(self.X, self.y)
            coefs.append(ridge.coef_)
        
        coefs = np.array(coefs).T  # Shape: (n_features, n_alphas)
        
        print(f"Alpha range: [{alphas.min():.6f}, {alphas.max():.2f}]")
        print(f"Coefficient matrix shape: {coefs.shape}")
        
        return alphas, coefs
    
    def compute_elasticnet_path(self, l1_ratio=0.5, n_alphas=100, alpha_min_ratio=1e-4):
        """
        Compute ElasticNet regularization path.
        
        Parameters:
        -----------
        l1_ratio : float
            Mixing parameter between L1 and L2 regularization
        n_alphas : int
            Number of alpha values along the path
        alpha_min_ratio : float
            Ratio of minimum to maximum alpha
            
        Returns:
        --------
        tuple : (alphas, coefficients)
        """
        print(f"\n🔍 Computing ElasticNet Path (L1 ratio: {l1_ratio})...")
        
        # Create alpha range manually
        alpha_max = np.linalg.norm(self.X.T @ self.y, ord=np.inf) / len(self.X)
        alphas = np.logspace(
            np.log10(alpha_max * alpha_min_ratio),
            np.log10(alpha_max),
            n_alphas
        )
        
        # Compute ElasticNet path
        alphas, coefs, _ = enet_path(
            self.X, self.y,
            l1_ratio=l1_ratio,
            alphas=alphas,
            max_iter=2000
        )
        
        print(f"Alpha range: [{alphas.min():.6f}, {alphas.max():.2f}]")
        print(f"Coefficient matrix shape: {coefs.shape}")
        
        return alphas, coefs
    
    def plot_coefficient_paths(self, save_path=None, figsize=(18, 12)):
        """
        Plot coefficient paths for Lasso, Ridge, and ElasticNet.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        figsize : tuple
            Figure size
        """
        if self.X is None:
            print("No data loaded. Run load_data first.")
            return
        
        # Compute paths
        lasso_alphas, lasso_coefs = self.compute_lasso_path()
        ridge_alphas, ridge_coefs = self.compute_ridge_path()
        enet_alphas, enet_coefs = self.compute_elasticnet_path()
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Coefficient Paths: Regularization Effects', fontsize=16, fontweight='bold')
        
        # Color map for features
        n_features = len(self.feature_names)
        colors = plt.cm.tab20(np.linspace(0, 1, n_features))
        
        # Plot 1: Lasso Path
        ax1 = axes[0, 0]
        for i, feature in enumerate(self.feature_names):
            ax1.semilogx(lasso_alphas, lasso_coefs[i, :], 
                        color=colors[i], linewidth=2, label=feature)
        
        ax1.set_xlabel('Alpha (log scale)')
        ax1.set_ylabel('Coefficient Value')
        ax1.set_title('Lasso Path (L1 Regularization)')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add vertical line at a specific alpha
        alpha_highlight = lasso_alphas[len(lasso_alphas)//3]
        ax1.axvline(x=alpha_highlight, color='red', linestyle=':', alpha=0.7, 
                   label=f'α = {alpha_highlight:.3f}')
        
        if n_features <= 10:  # Only show legend for small number of features
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        # Plot 2: Ridge Path
        ax2 = axes[0, 1]
        for i, feature in enumerate(self.feature_names):
            ax2.semilogx(ridge_alphas, ridge_coefs[i, :], 
                        color=colors[i], linewidth=2, label=feature)
        
        ax2.set_xlabel('Alpha (log scale)')
        ax2.set_ylabel('Coefficient Value')
        ax2.set_title('Ridge Path (L2 Regularization)')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add vertical line at a specific alpha
        alpha_highlight = ridge_alphas[len(ridge_alphas)//3]
        ax2.axvline(x=alpha_highlight, color='red', linestyle=':', alpha=0.7, 
                   label=f'α = {alpha_highlight:.3f}')
        
        # Plot 3: ElasticNet Path
        ax3 = axes[0, 2]
        for i, feature in enumerate(self.feature_names):
            ax3.semilogx(enet_alphas, enet_coefs[i, :], 
                        color=colors[i], linewidth=2, label=feature)
        
        ax3.set_xlabel('Alpha (log scale)')
        ax3.set_ylabel('Coefficient Value')
        ax3.set_title('ElasticNet Path (L1 + L2)')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Plot 4: Number of non-zero coefficients
        ax4 = axes[1, 0]
        
        # Calculate number of non-zero coefficients for each method
        lasso_nonzero = np.sum(np.abs(lasso_coefs) > 1e-5, axis=0)
        ridge_nonzero = np.sum(np.abs(ridge_coefs) > 1e-5, axis=0)
        enet_nonzero = np.sum(np.abs(enet_coefs) > 1e-5, axis=0)
        
        ax4.semilogx(lasso_alphas, lasso_nonzero, 'o-', label='Lasso', linewidth=2)
        ax4.semilogx(ridge_alphas, ridge_nonzero, 's-', label='Ridge', linewidth=2)
        ax4.semilogx(enet_alphas, enet_nonzero, '^-', label='ElasticNet', linewidth=2)
        
        ax4.set_xlabel('Alpha (log scale)')
        ax4.set_ylabel('Number of Non-zero Coefficients')
        ax4.set_title('Feature Selection Effect')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Coefficient magnitude comparison at specific alpha
        ax5 = axes[1, 1]
        
        # Find common alpha value
        common_alpha = 0.1
        
        # Get coefficients at this alpha (closest value)
        lasso_idx = np.argmin(np.abs(lasso_alphas - common_alpha))
        ridge_idx = np.argmin(np.abs(ridge_alphas - common_alpha))
        enet_idx = np.argmin(np.abs(enet_alphas - common_alpha))
        
        lasso_coef_at_alpha = lasso_coefs[:, lasso_idx]
        ridge_coef_at_alpha = ridge_coefs[:, ridge_idx]
        enet_coef_at_alpha = enet_coefs[:, enet_idx]
        
        x_pos = np.arange(len(self.feature_names))
        width = 0.25
        
        ax5.bar(x_pos - width, lasso_coef_at_alpha, width, label='Lasso', alpha=0.8)
        ax5.bar(x_pos, ridge_coef_at_alpha, width, label='Ridge', alpha=0.8)
        ax5.bar(x_pos + width, enet_coef_at_alpha, width, label='ElasticNet', alpha=0.8)
        
        ax5.set_xlabel('Features')
        ax5.set_ylabel('Coefficient Value')
        ax5.set_title(f'Coefficient Comparison at α ≈ {common_alpha}')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels([f'F{i+1}' for i in range(len(self.feature_names))], rotation=45)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Plot 6: L2 norm of coefficients
        ax6 = axes[1, 2]
        
        lasso_l2_norm = np.sqrt(np.sum(lasso_coefs**2, axis=0))
        ridge_l2_norm = np.sqrt(np.sum(ridge_coefs**2, axis=0))
        enet_l2_norm = np.sqrt(np.sum(enet_coefs**2, axis=0))
        
        ax6.loglog(lasso_alphas, lasso_l2_norm, 'o-', label='Lasso', linewidth=2)
        ax6.loglog(ridge_alphas, ridge_l2_norm, 's-', label='Ridge', linewidth=2)
        ax6.loglog(enet_alphas, enet_l2_norm, '^-', label='ElasticNet', linewidth=2)
        
        ax6.set_xlabel('Alpha (log scale)')
        ax6.set_ylabel('L2 Norm of Coefficients (log scale)')
        ax6.set_title('Coefficient Shrinkage (L2 Norm)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Coefficient paths plot saved to {save_path}")
        
        plt.show()
        
        return {
            'lasso': (lasso_alphas, lasso_coefs),
            'ridge': (ridge_alphas, ridge_coefs),
            'elasticnet': (enet_alphas, enet_coefs)
        }
    
    def plot_individual_feature_paths(self, top_n=5, save_path=None, figsize=(15, 10)):
        """
        Plot paths for individual features to show detailed behavior.
        
        Parameters:
        -----------
        top_n : int
            Number of top features to highlight
        save_path : str, optional
            Path to save the plot
        figsize : tuple
            Figure size
        """
        if self.X is None:
            print("No data loaded. Run load_data first.")
            return
        
        # Compute paths
        lasso_alphas, lasso_coefs = self.compute_lasso_path()
        ridge_alphas, ridge_coefs = self.compute_ridge_path()
        
        # Find top features based on maximum absolute coefficient in Lasso
        max_lasso_coefs = np.max(np.abs(lasso_coefs), axis=1)
        top_feature_indices = np.argsort(max_lasso_coefs)[-top_n:][::-1]
        
        fig, axes = plt.subplots(2, top_n, figsize=figsize)
        if top_n == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle(f'Individual Feature Paths: Top {top_n} Features', 
                    fontsize=16, fontweight='bold')
        
        colors = ['blue', 'red']
        methods = ['Lasso', 'Ridge']
        alphas_list = [lasso_alphas, ridge_alphas]
        coefs_list = [lasso_coefs, ridge_coefs]
        
        for i, feature_idx in enumerate(top_feature_indices):
            feature_name = self.feature_names[feature_idx]
            
            for method_idx, (method, alphas, coefs) in enumerate(zip(methods, alphas_list, coefs_list)):
                ax = axes[method_idx, i]
                
                # Plot coefficient path for this feature
                ax.semilogx(alphas, coefs[feature_idx, :], 
                           color=colors[method_idx], linewidth=3, label=feature_name)
                
                # Add zero line
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                
                # Highlight where coefficient becomes zero (for Lasso)
                if method == 'Lasso':
                    # Find where coefficient first becomes zero
                    zero_indices = np.where(np.abs(coefs[feature_idx, :]) < 1e-5)[0]
                    if len(zero_indices) > 0:
                        zero_alpha = alphas[zero_indices[0]]
                        ax.axvline(x=zero_alpha, color='red', linestyle=':', alpha=0.7,
                                  label=f'Zero at α={zero_alpha:.3f}')
                
                ax.set_xlabel('Alpha (log scale)')
                ax.set_ylabel('Coefficient Value')
                ax.set_title(f'{method}: {feature_name}')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Individual feature paths plot saved to {save_path}")
        
        plt.show()
    
    def analyze_shrinkage_patterns(self):
        """
        Analyze and compare shrinkage patterns between methods.
        
        Returns:
        --------
        dict : Analysis results
        """
        print("\n📊 COEFFICIENT SHRINKAGE ANALYSIS")
        print("=" * 45)
        
        # Compute paths
        lasso_alphas, lasso_coefs = self.compute_lasso_path()
        ridge_alphas, ridge_coefs = self.compute_ridge_path()
        
        # Analysis at different regularization strengths
        alpha_points = [0.001, 0.01, 0.1, 1.0]
        
        analysis_results = {}
        
        for alpha in alpha_points:
            # Find closest alpha values
            lasso_idx = np.argmin(np.abs(lasso_alphas - alpha))
            ridge_idx = np.argmin(np.abs(ridge_alphas - alpha))
            
            lasso_coef = lasso_coefs[:, lasso_idx]
            ridge_coef = ridge_coefs[:, ridge_idx]
            
            # Calculate statistics
            lasso_nonzero = np.sum(np.abs(lasso_coef) > 1e-5)
            ridge_nonzero = np.sum(np.abs(ridge_coef) > 1e-5)
            
            lasso_l2_norm = np.sqrt(np.sum(lasso_coef**2))
            ridge_l2_norm = np.sqrt(np.sum(ridge_coef**2))
            
            lasso_l1_norm = np.sum(np.abs(lasso_coef))
            ridge_l1_norm = np.sum(np.abs(ridge_coef))
            
            analysis_results[alpha] = {
                'lasso_nonzero': lasso_nonzero,
                'ridge_nonzero': ridge_nonzero,
                'lasso_l2_norm': lasso_l2_norm,
                'ridge_l2_norm': ridge_l2_norm,
                'lasso_l1_norm': lasso_l1_norm,
                'ridge_l1_norm': ridge_l1_norm
            }
            
            print(f"\nAlpha = {alpha}:")
            print(f"  Lasso: {lasso_nonzero}/{len(self.feature_names)} features, "
                  f"L2 norm: {lasso_l2_norm:.3f}")
            print(f"  Ridge: {ridge_nonzero}/{len(self.feature_names)} features, "
                  f"L2 norm: {ridge_l2_norm:.3f}")
        
        # Summary insights
        print("\n💡 Key Insights:")
        print("• Lasso progressively sets coefficients to exactly zero")
        print("• Ridge shrinks all coefficients but never sets them to zero")
        print("• Lasso provides automatic feature selection")
        print("• Ridge provides more stable coefficient estimates")
        
        return analysis_results


def main():
    """
    Main function to demonstrate coefficient path analysis.
    """
    print("🎯 COEFFICIENT PATH ANALYSIS")
    print("=" * 40)
    
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Initialize analyzer
    analyzer = CoefficientPathAnalyzer(random_state=42)
    
    # Test with diabetes dataset
    print("\n📊 PHASE 1: Diabetes Dataset Analysis")
    print("=" * 40)
    
    dataset_info = analyzer.load_data('diabetes')
    
    # Plot coefficient paths
    paths = analyzer.plot_coefficient_paths(save_path='plots/coefficient_paths_diabetes.png')
    
    # Plot individual feature paths
    analyzer.plot_individual_feature_paths(
        top_n=5, 
        save_path='plots/individual_paths_diabetes.png'
    )
    
    # Analyze shrinkage patterns
    shrinkage_analysis = analyzer.analyze_shrinkage_patterns()
    
    # Test with synthetic dataset
    print("\n\n🔬 PHASE 2: Synthetic Dataset Analysis")
    print("=" * 40)
    
    synthetic_info = analyzer.load_data('synthetic', n_features=15)
    
    # Plot coefficient paths for synthetic data
    analyzer.plot_coefficient_paths(save_path='plots/coefficient_paths_synthetic.png')
    
    # Individual paths for synthetic data
    analyzer.plot_individual_feature_paths(
        top_n=6, 
        save_path='plots/individual_paths_synthetic.png'
    )
    
    print("\n✅ COEFFICIENT PATH ANALYSIS COMPLETE!")
    print("📁 Check the 'plots' folder for visualizations.")
    print("🔧 Key demonstrations include:")
    print("   - Lasso path showing feature selection (coefficients → 0)")
    print("   - Ridge path showing uniform shrinkage")
    print("   - ElasticNet path combining both effects")
    print("   - Individual feature behavior analysis")
    print("   - Quantitative shrinkage pattern comparison")

if __name__ == "__main__":
    main() 