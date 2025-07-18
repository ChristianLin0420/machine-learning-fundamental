"""
Feature Scaling Visualization Module
===================================

This module provides comprehensive visualization tools for analyzing
the effects of different feature scaling methods on data distributions
and statistical properties.

Visualization Types:
- Distribution histograms before/after scaling
- Boxplot comparisons across scaling methods
- Statistical summary plots
- Correlation matrix changes
- Feature importance effects
- Outlier detection analysis

Features:
- Side-by-side comparisons
- Statistical annotations
- Interactive analysis
- Publication-ready plots
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_wine, load_diabetes
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scaling_from_scratch import (
    StandardScaler, MinMaxScaler, RobustScaler, L2Normalizer, 
    ScalingAnalyzer, load_sample_datasets
)
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ScalingVisualizer:
    """
    Comprehensive visualizer for feature scaling effects.
    
    This class provides advanced visualization tools for understanding
    the impact of different scaling methods on data characteristics.
    """
    
    def __init__(self):
        """Initialize the scaling visualizer."""
        self.scalers = {
            'Original': None,
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler(),
            'L2Normalizer': L2Normalizer()
        }
        self.scaled_data = {}
        self.feature_names = None
    
    def prepare_data(self, X, feature_names=None):
        """
        Prepare data by applying all scaling methods.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        feature_names : list, optional
            Names of features
        """
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        self.feature_names = feature_names
        
        # Store original data
        self.scaled_data['Original'] = X
        
        # Apply each scaler
        for name, scaler in self.scalers.items():
            if scaler is not None:
                self.scaled_data[name] = scaler.fit_transform(X)
    
    def plot_distribution_comparison(self, save_path=None, figsize=(20, 12)):
        """
        Create comprehensive distribution comparison plots.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        figsize : tuple
            Figure size
        """
        if not self.scaled_data:
            raise ValueError("No data prepared. Call prepare_data() first.")
        
        n_features = min(6, len(self.feature_names))  # Limit for readability
        n_methods = len(self.scaled_data)
        
        fig, axes = plt.subplots(n_features, n_methods, figsize=figsize)
        
        if n_features == 1:
            axes = axes.reshape(1, -1)
        if n_methods == 1:
            axes = axes.reshape(-1, 1)
        
        # Color palette for consistency
        colors = plt.cm.Set3(np.linspace(0, 1, n_methods))
        
        for i, feature_idx in enumerate(range(n_features)):
            feature_name = self.feature_names[feature_idx]
            
            for j, (method_name, X_scaled) in enumerate(self.scaled_data.items()):
                ax = axes[i, j]
                
                data = X_scaled[:, feature_idx]
                
                # Create histogram with KDE
                ax.hist(data, bins=30, alpha=0.7, density=True, 
                       color=colors[j], edgecolor='black', linewidth=0.5)
                
                # Add KDE curve
                try:
                    from scipy import stats
                    kde = stats.gaussian_kde(data)
                    x_range = np.linspace(data.min(), data.max(), 100)
                    ax.plot(x_range, kde(x_range), 'k-', linewidth=2, alpha=0.8)
                except ImportError:
                    pass
                
                # Calculate and display statistics
                mean_val = np.mean(data)
                std_val = np.std(data)
                median_val = np.median(data)
                skew_val = stats.skew(data) if 'stats' in locals() else 0
                
                # Add vertical lines for statistics
                ax.axvline(mean_val, color='red', linestyle='--', 
                          linewidth=2, alpha=0.8, label=f'μ={mean_val:.2f}')
                ax.axvline(median_val, color='green', linestyle='--', 
                          linewidth=2, alpha=0.8, label=f'Med={median_val:.2f}')
                
                # Title with statistics
                ax.set_title(f'{method_name}\n{feature_name}\n'
                           f'μ={mean_val:.2f}, σ={std_val:.2f}, skew={skew_val:.2f}',
                           fontsize=10)
                
                # Formatting
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
                
                # Set y-label for first column
                if j == 0:
                    ax.set_ylabel('Density')
                
                # Set x-label for last row
                if i == n_features - 1:
                    ax.set_xlabel('Value')
        
        plt.suptitle('Distribution Comparison Across Scaling Methods', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distribution comparison saved to {save_path}")
        
        plt.show()
    
    def plot_boxplot_comparison(self, save_path=None, figsize=(18, 10)):
        """
        Create boxplot comparison across all scaling methods.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        figsize : tuple
            Figure size
        """
        if not self.scaled_data:
            raise ValueError("No data prepared. Call prepare_data() first.")
        
        n_methods = len(self.scaled_data)
        n_features = min(8, len(self.feature_names))  # Limit for readability
        
        fig, axes = plt.subplots(2, (n_methods + 1) // 2, figsize=figsize)
        axes = axes.flatten() if n_methods > 1 else [axes]
        
        for i, (method_name, X_scaled) in enumerate(self.scaled_data.items()):
            ax = axes[i]
            
            # Prepare data for boxplot
            box_data = []
            labels = []
            for j in range(n_features):
                box_data.append(X_scaled[:, j])
                labels.append(self.feature_names[j][:10])  # Truncate long names
            
            # Create boxplot
            bp = ax.boxplot(box_data, labels=labels, patch_artist=True, 
                           showmeans=True, meanline=True)
            
            # Color the boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(box_data)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Customize plot
            ax.set_title(f'{method_name}', fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add statistics annotation
            stats_text = f"Range: [{np.min(X_scaled):.2f}, {np.max(X_scaled):.2f}]"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   verticalalignment='top', fontsize=8)
        
        # Hide unused subplots
        for i in range(len(self.scaled_data), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Boxplot Comparison Across Scaling Methods', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Boxplot comparison saved to {save_path}")
        
        plt.show()
    
    def plot_correlation_analysis(self, save_path=None, figsize=(16, 12)):
        """
        Analyze how scaling affects feature correlations.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        figsize : tuple
            Figure size
        """
        if not self.scaled_data:
            raise ValueError("No data prepared. Call prepare_data() first.")
        
        n_methods = len(self.scaled_data)
        cols = 3
        rows = (n_methods + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if n_methods > 1 else [axes]
        
        for i, (method_name, X_scaled) in enumerate(self.scaled_data.items()):
            ax = axes[i]
            
            # Compute correlation matrix
            n_features = min(10, X_scaled.shape[1])  # Limit for readability
            corr_matrix = np.corrcoef(X_scaled[:, :n_features].T)
            
            # Create heatmap
            im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Correlation', rotation=270, labelpad=15)
            
            # Customize
            ax.set_title(f'{method_name}\nCorrelation Matrix', fontweight='bold')
            ax.set_xticks(range(n_features))
            ax.set_yticks(range(n_features))
            ax.set_xticklabels([name[:8] for name in self.feature_names[:n_features]], 
                              rotation=45)
            ax.set_yticklabels([name[:8] for name in self.feature_names[:n_features]])
            
            # Add correlation values
            for row in range(n_features):
                for col in range(n_features):
                    text = ax.text(col, row, f'{corr_matrix[row, col]:.2f}',
                                 ha="center", va="center", color="white" if abs(corr_matrix[row, col]) > 0.5 else "black",
                                 fontsize=8)
        
        # Hide unused subplots
        for i in range(len(self.scaled_data), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Feature Correlation Analysis Across Scaling Methods', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Correlation analysis saved to {save_path}")
        
        plt.show()
    
    def plot_outlier_analysis(self, save_path=None, figsize=(16, 10)):
        """
        Analyze outlier detection across different scaling methods.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        figsize : tuple
            Figure size
        """
        if not self.scaled_data:
            raise ValueError("No data prepared. Call prepare_data() first.")
        
        n_methods = len(self.scaled_data)
        
        fig, axes = plt.subplots(2, (n_methods + 1) // 2, figsize=figsize)
        axes = axes.flatten() if n_methods > 1 else [axes]
        
        for i, (method_name, X_scaled) in enumerate(self.scaled_data.items()):
            ax = axes[i]
            
            # Use first two features for 2D visualization
            if X_scaled.shape[1] >= 2:
                x_data = X_scaled[:, 0]
                y_data = X_scaled[:, 1]
                
                # Compute Mahalanobis distance for outlier detection
                try:
                    from scipy.spatial.distance import mahalanobis
                    from scipy.linalg import inv
                    
                    # Covariance matrix
                    cov_matrix = np.cov(X_scaled[:, :2].T)
                    inv_cov = inv(cov_matrix)
                    
                    # Compute distances
                    mean_vec = np.mean(X_scaled[:, :2], axis=0)
                    distances = []
                    for point in X_scaled[:, :2]:
                        dist = mahalanobis(point, mean_vec, inv_cov)
                        distances.append(dist)
                    
                    distances = np.array(distances)
                    
                    # Define outliers (top 5%)
                    threshold = np.percentile(distances, 95)
                    outliers = distances > threshold
                    
                    # Plot
                    ax.scatter(x_data[~outliers], y_data[~outliers], 
                             alpha=0.6, s=30, label='Normal', color='blue')
                    ax.scatter(x_data[outliers], y_data[outliers], 
                             alpha=0.8, s=50, label='Outliers', color='red', marker='x')
                    
                    ax.set_xlabel(f'{self.feature_names[0]} (scaled)')
                    ax.set_ylabel(f'{self.feature_names[1]} (scaled)')
                    ax.set_title(f'{method_name}\nOutlier Detection\n({np.sum(outliers)} outliers)')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                except ImportError:
                    # Fallback to simple z-score based outlier detection
                    z_scores = np.abs((X_scaled[:, :2] - np.mean(X_scaled[:, :2], axis=0)) / 
                                    np.std(X_scaled[:, :2], axis=0))
                    outliers = np.any(z_scores > 3, axis=1)
                    
                    ax.scatter(x_data[~outliers], y_data[~outliers], 
                             alpha=0.6, s=30, label='Normal', color='blue')
                    ax.scatter(x_data[outliers], y_data[outliers], 
                             alpha=0.8, s=50, label='Outliers', color='red', marker='x')
                    
                    ax.set_xlabel(f'{self.feature_names[0]} (scaled)')
                    ax.set_ylabel(f'{self.feature_names[1]} (scaled)')
                    ax.set_title(f'{method_name}\nOutlier Detection (Z-score)\n({np.sum(outliers)} outliers)')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Need ≥2 features', ha='center', va='center', 
                       transform=ax.transAxes)
                ax.set_title(f'{method_name}')
        
        # Hide unused subplots
        for i in range(len(self.scaled_data), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Outlier Detection Across Scaling Methods', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Outlier analysis saved to {save_path}")
        
        plt.show()
    
    def plot_pca_analysis(self, save_path=None, figsize=(16, 10)):
        """
        Analyze PCA results across different scaling methods.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        figsize : tuple
            Figure size
        """
        if not self.scaled_data:
            raise ValueError("No data prepared. Call prepare_data() first.")
        
        n_methods = len(self.scaled_data)
        
        fig, axes = plt.subplots(2, (n_methods + 1) // 2, figsize=figsize)
        axes = axes.flatten() if n_methods > 1 else [axes]
        
        for i, (method_name, X_scaled) in enumerate(self.scaled_data.items()):
            ax = axes[i]
            
            # Apply PCA
            pca = PCA(n_components=min(2, X_scaled.shape[1]))
            X_pca = pca.fit_transform(X_scaled)
            
            # Plot PCA results
            if X_pca.shape[1] >= 2:
                ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=30)
                ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
                ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
                
                # Add explained variance info
                total_var = np.sum(pca.explained_variance_ratio_)
                ax.set_title(f'{method_name}\nPCA Analysis\nTotal Var: {total_var:.2%}')
            else:
                ax.scatter(X_pca[:, 0], np.zeros_like(X_pca[:, 0]), alpha=0.6, s=30)
                ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
                ax.set_ylabel('0')
                ax.set_title(f'{method_name}\nPCA Analysis (1D)')
            
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(self.scaled_data), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('PCA Analysis Across Scaling Methods', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PCA analysis saved to {save_path}")
        
        plt.show()
    
    def plot_statistical_summary(self, save_path=None, figsize=(16, 12)):
        """
        Create comprehensive statistical summary visualization.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        figsize : tuple
            Figure size
        """
        if not self.scaled_data:
            raise ValueError("No data prepared. Call prepare_data() first.")
        
        # Prepare statistical data
        stats_data = []
        
        for method_name, X_scaled in self.scaled_data.items():
            for i, feature_name in enumerate(self.feature_names[:min(8, len(self.feature_names))]):
                data = X_scaled[:, i]
                
                stats_data.append({
                    'Method': method_name,
                    'Feature': feature_name,
                    'Mean': np.mean(data),
                    'Std': np.std(data),
                    'Min': np.min(data),
                    'Max': np.max(data),
                    'Median': np.median(data),
                    'Q1': np.percentile(data, 25),
                    'Q3': np.percentile(data, 75),
                    'IQR': np.percentile(data, 75) - np.percentile(data, 25),
                    'Range': np.max(data) - np.min(data)
                })
        
        df = pd.DataFrame(stats_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        # Plot 1: Mean comparison
        ax = axes[0]
        pivot_data = df.pivot(index='Feature', columns='Method', values='Mean')
        pivot_data.plot(kind='bar', ax=ax, alpha=0.8)
        ax.set_title('Mean Values Across Methods')
        ax.set_ylabel('Mean')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Standard deviation comparison
        ax = axes[1]
        pivot_data = df.pivot(index='Feature', columns='Method', values='Std')
        pivot_data.plot(kind='bar', ax=ax, alpha=0.8)
        ax.set_title('Standard Deviation Across Methods')
        ax.set_ylabel('Std Dev')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Range comparison
        ax = axes[2]
        pivot_data = df.pivot(index='Feature', columns='Method', values='Range')
        pivot_data.plot(kind='bar', ax=ax, alpha=0.8)
        ax.set_title('Range Across Methods')
        ax.set_ylabel('Range')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Distribution of means
        ax = axes[3]
        for method in df['Method'].unique():
            method_data = df[df['Method'] == method]['Mean']
            ax.hist(method_data, alpha=0.6, label=method, bins=15)
        ax.set_title('Distribution of Feature Means')
        ax.set_xlabel('Mean Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Min-Max ranges
        ax = axes[4]
        methods = df['Method'].unique()
        for i, method in enumerate(methods):
            method_data = df[df['Method'] == method]
            mins = method_data['Min'].values
            maxs = method_data['Max'].values
            features = range(len(mins))
            
            ax.scatter([i] * len(features), mins, alpha=0.6, s=50, marker='v', label=f'{method} Min')
            ax.scatter([i] * len(features), maxs, alpha=0.6, s=50, marker='^', label=f'{method} Max')
        
        ax.set_title('Min-Max Values by Method')
        ax.set_xlabel('Method')
        ax.set_ylabel('Value')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Statistical summary table
        ax = axes[5]
        ax.axis('tight')
        ax.axis('off')
        
        # Create summary table
        summary_stats = df.groupby('Method').agg({
            'Mean': ['mean', 'std'],
            'Std': ['mean', 'std'],
            'Range': ['mean', 'std']
        }).round(3)
        
        table_data = []
        for method in summary_stats.index:
            row = [
                method,
                f"{summary_stats.loc[method, ('Mean', 'mean')]:.3f} ± {summary_stats.loc[method, ('Mean', 'std')]:.3f}",
                f"{summary_stats.loc[method, ('Std', 'mean')]:.3f} ± {summary_stats.loc[method, ('Std', 'std')]:.3f}",
                f"{summary_stats.loc[method, ('Range', 'mean')]:.3f} ± {summary_stats.loc[method, ('Range', 'std')]:.3f}"
            ]
            table_data.append(row)
        
        table = ax.table(cellText=table_data,
                        colLabels=['Method', 'Mean ± Std', 'Std ± Std', 'Range ± Std'],
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax.set_title('Statistical Summary Table')
        
        plt.suptitle('Comprehensive Statistical Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Statistical summary saved to {save_path}")
        
        plt.show()
    
    def create_comprehensive_report(self, X, feature_names=None, save_dir='plots'):
        """
        Create a comprehensive scaling analysis report.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        feature_names : list, optional
            Names of features
        save_dir : str
            Directory to save plots
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("Creating comprehensive scaling analysis report...")
        
        # Prepare data
        self.prepare_data(X, feature_names)
        
        # Generate all visualizations
        print("  Generating distribution comparison...")
        self.plot_distribution_comparison(save_path=f'{save_dir}/distribution_comparison.png')
        
        print("  Generating boxplot comparison...")
        self.plot_boxplot_comparison(save_path=f'{save_dir}/boxplot_comparison.png')
        
        print("  Generating correlation analysis...")
        self.plot_correlation_analysis(save_path=f'{save_dir}/correlation_analysis.png')
        
        print("  Generating outlier analysis...")
        self.plot_outlier_analysis(save_path=f'{save_dir}/outlier_analysis.png')
        
        print("  Generating PCA analysis...")
        self.plot_pca_analysis(save_path=f'{save_dir}/pca_analysis.png')
        
        print("  Generating statistical summary...")
        self.plot_statistical_summary(save_path=f'{save_dir}/statistical_summary.png')
        
        print(f"Comprehensive report saved to {save_dir}/")


def main():
    """
    Main function to demonstrate scaling visualization.
    """
    print("🎯 FEATURE SCALING VISUALIZATION")
    print("=" * 40)
    
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Load sample datasets
    datasets = load_sample_datasets()
    
    # Create visualizer
    visualizer = ScalingVisualizer()
    
    # Test with different datasets
    for dataset_name, dataset in datasets.items():
        if dataset_name == 'synthetic':  # Focus on synthetic for detailed analysis
            print(f"\n📊 Creating comprehensive analysis for {dataset['name']}")
            print("-" * 50)
            
            X = dataset['data']
            feature_names = dataset['feature_names']
            
            # Create comprehensive report
            visualizer.create_comprehensive_report(
                X, feature_names, save_dir=f'plots/{dataset_name}_analysis'
            )
    
    # Create a quick comparison for wine dataset
    print(f"\n🍷 Quick analysis for Wine dataset")
    print("-" * 30)
    
    wine_data = datasets['wine']['data'][:200]  # Subset for faster processing
    wine_features = datasets['wine']['feature_names'][:8]  # First 8 features
    
    visualizer.prepare_data(wine_data[:, :8], wine_features)
    visualizer.plot_distribution_comparison(save_path='plots/wine_distributions.png', figsize=(16, 8))
    visualizer.plot_boxplot_comparison(save_path='plots/wine_boxplots.png')
    
    print("\n✅ SCALING VISUALIZATION COMPLETE!")
    print("📁 Check the 'plots' folder for all visualizations.")
    print("🔧 Generated visualizations include:")
    print("   - Distribution comparisons")
    print("   - Boxplot comparisons") 
    print("   - Correlation analysis")
    print("   - Outlier detection")
    print("   - PCA analysis")
    print("   - Statistical summaries")
    
    return visualizer

if __name__ == "__main__":
    main() 