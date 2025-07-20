"""
Ridge vs Lasso Comparison Plots
===============================

This module creates detailed comparison plots between Ridge and Lasso regression,
showing coefficient differences, sparsity effects, and performance trade-offs.

Key Visualizations:
- Side-by-side coefficient bar plots
- Sparsity comparison (number of non-zero coefficients)
- Performance metrics comparison
- Feature selection patterns
- Coefficient magnitude distributions

Author: Machine Learning Fundamentals Course
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_diabetes, make_regression
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RidgeLassoComparator:
    """
    Comprehensive comparison tool for Ridge vs Lasso regression.
    
    This class provides detailed visualizations and analysis to understand
    the differences between Ridge and Lasso regularization techniques.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the Ridge vs Lasso comparator.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.results = {}
        
    def load_data(self, dataset='diabetes', noise_level=0.1):
        """
        Load and prepare dataset for comparison.
        
        Parameters:
        -----------
        dataset : str
            Dataset to use ('diabetes', 'synthetic', 'high_dim')
        noise_level : float
            Noise level for synthetic data
            
        Returns:
        --------
        dict : Dataset information
        """
        if dataset == 'diabetes':
            data = load_diabetes()
            X, y = data.data, data.target
            feature_names = data.feature_names
            print("🏥 Loading Diabetes Dataset")
            print("=" * 35)
            
        elif dataset == 'high_dim':
            # High-dimensional synthetic dataset
            X, y = make_regression(
                n_samples=200, n_features=50, n_informative=10,
                noise=noise_level * 100, random_state=self.random_state
            )
            feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]
            print("🔬 Creating High-Dimensional Synthetic Dataset")
            print("=" * 50)
            
        else:  # synthetic
            X, y = make_regression(
                n_samples=300, n_features=20, n_informative=10,
                noise=noise_level * 100, random_state=self.random_state
            )
            feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]
            print("🔬 Creating Synthetic Dataset")
            print("=" * 35)
        
        print(f"Dataset shape: {X.shape}")
        print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
        print(f"Features: {len(feature_names)} total")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names,
            'scaler': scaler
        }
    
    def find_optimal_alphas(self, alpha_range=None, cv_folds=5):
        """
        Find optimal alpha values for Ridge and Lasso using cross-validation.
        
        Parameters:
        -----------
        alpha_range : array-like, optional
            Range of alpha values to test
        cv_folds : int
            Number of cross-validation folds
            
        Returns:
        --------
        dict : Optimal alpha values and CV scores
        """
        if alpha_range is None:
            alpha_range = np.logspace(-3, 2, 50)
        
        print(f"\n🔍 Finding Optimal Alphas (CV={cv_folds} folds)")
        print("=" * 45)
        
        # Test Ridge
        ridge_scores = []
        for alpha in alpha_range:
            ridge = Ridge(alpha=alpha, random_state=self.random_state)
            scores = cross_val_score(ridge, self.X_train, self.y_train, 
                                   cv=cv_folds, scoring='r2')
            ridge_scores.append(np.mean(scores))
        
        # Test Lasso
        lasso_scores = []
        for alpha in alpha_range:
            lasso = Lasso(alpha=alpha, max_iter=2000, random_state=self.random_state)
            scores = cross_val_score(lasso, self.X_train, self.y_train, 
                                   cv=cv_folds, scoring='r2')
            lasso_scores.append(np.mean(scores))
        
        # Find best alphas
        ridge_scores = np.array(ridge_scores)
        lasso_scores = np.array(lasso_scores)
        
        best_ridge_idx = np.argmax(ridge_scores)
        best_lasso_idx = np.argmax(lasso_scores)
        
        best_ridge_alpha = alpha_range[best_ridge_idx]
        best_lasso_alpha = alpha_range[best_lasso_idx]
        
        best_ridge_score = ridge_scores[best_ridge_idx]
        best_lasso_score = lasso_scores[best_lasso_idx]
        
        print(f"Best Ridge α: {best_ridge_alpha:.4f}, CV R²: {best_ridge_score:.4f}")
        print(f"Best Lasso α: {best_lasso_alpha:.4f}, CV R²: {best_lasso_score:.4f}")
        
        optimal_alphas = {
            'ridge_alpha': best_ridge_alpha,
            'lasso_alpha': best_lasso_alpha,
            'ridge_score': best_ridge_score,
            'lasso_score': best_lasso_score,
            'alpha_range': alpha_range,
            'ridge_scores': ridge_scores,
            'lasso_scores': lasso_scores
        }
        
        return optimal_alphas
    
    def compare_models(self, alpha_ridge=None, alpha_lasso=None):
        """
        Compare Ridge, Lasso, and baseline models.
        
        Parameters:
        -----------
        alpha_ridge : float, optional
            Alpha for Ridge regression
        alpha_lasso : float, optional
            Alpha for Lasso regression
            
        Returns:
        --------
        dict : Model comparison results
        """
        # Find optimal alphas if not provided
        if alpha_ridge is None or alpha_lasso is None:
            optimal_alphas = self.find_optimal_alphas()
            if alpha_ridge is None:
                alpha_ridge = optimal_alphas['ridge_alpha']
            if alpha_lasso is None:
                alpha_lasso = optimal_alphas['lasso_alpha']
        
        print(f"\n🔍 Comparing Models (Ridge α={alpha_ridge:.4f}, Lasso α={alpha_lasso:.4f})")
        print("=" * 65)
        
        # Initialize models
        models = {
            'Linear': LinearRegression(),
            'Ridge': Ridge(alpha=alpha_ridge, random_state=self.random_state),
            'Lasso': Lasso(alpha=alpha_lasso, max_iter=2000, random_state=self.random_state),
            'ElasticNet': ElasticNet(alpha=(alpha_ridge + alpha_lasso)/2, l1_ratio=0.5, 
                                   max_iter=2000, random_state=self.random_state)
        }
        
        results = {}
        
        for name, model in models.items():
            # Fit model
            model.fit(self.X_train, self.y_train)
            
            # Predictions
            y_train_pred = model.predict(self.X_train)
            y_test_pred = model.predict(self.X_test)
            
            # Metrics
            train_mse = mean_squared_error(self.y_train, y_train_pred)
            test_mse = mean_squared_error(self.y_test, y_test_pred)
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            test_mae = mean_absolute_error(self.y_test, y_test_pred)
            
            # Coefficients analysis
            if hasattr(model, 'coef_'):
                coefs = model.coef_
                non_zero_coefs = np.sum(np.abs(coefs) > 1e-5)
                max_coef = np.max(np.abs(coefs))
                coef_std = np.std(coefs)
            else:
                coefs = None
                non_zero_coefs = 0
                max_coef = 0
                coef_std = 0
            
            results[name] = {
                'model': model,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'test_mae': test_mae,
                'coefficients': coefs,
                'non_zero_coefs': non_zero_coefs,
                'max_coef': max_coef,
                'coef_std': coef_std,
                'alpha': alpha_ridge if name == 'Ridge' else (alpha_lasso if name == 'Lasso' else None)
            }
            
            print(f"{name:>12}: R² = {test_r2:.4f}, MSE = {test_mse:.2f}, Non-zero coefs = {non_zero_coefs}")
        
        self.results = results
        return results
    
    def plot_ridge_vs_lasso_comparison(self, save_path=None, figsize=(18, 12)):
        """
        Create comprehensive Ridge vs Lasso comparison plots.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        figsize : tuple
            Figure size
        """
        if not self.results:
            print("No results available. Run compare_models first.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Ridge vs Lasso Regression Comparison', fontsize=16, fontweight='bold')
        
        # Colors for different models
        colors = {
            'Linear': 'gray',
            'Ridge': 'blue', 
            'Lasso': 'red',
            'ElasticNet': 'green'
        }
        
        # Plot 1: Coefficient Values Comparison (Ridge vs Lasso)
        ax1 = axes[0, 0]
        
        ridge_coefs = self.results['Ridge']['coefficients']
        lasso_coefs = self.results['Lasso']['coefficients']
        
        # Only show features where at least one method has non-zero coefficient
        important_features = np.where((np.abs(ridge_coefs) > 1e-5) | (np.abs(lasso_coefs) > 1e-5))[0]
        
        if len(important_features) > 20:  # Limit to top 20 features for readability
            # Select top features based on combined importance
            combined_importance = np.abs(ridge_coefs) + np.abs(lasso_coefs)
            important_features = np.argsort(combined_importance)[-20:][::-1]
        
        x_pos = np.arange(len(important_features))
        width = 0.35
        
        ridge_vals = ridge_coefs[important_features]
        lasso_vals = lasso_coefs[important_features]
        
        ax1.bar(x_pos - width/2, ridge_vals, width, label='Ridge', 
               color=colors['Ridge'], alpha=0.8)
        ax1.bar(x_pos + width/2, lasso_vals, width, label='Lasso', 
               color=colors['Lasso'], alpha=0.8)
        
        ax1.set_xlabel('Feature Index')
        ax1.set_ylabel('Coefficient Value')
        ax1.set_title('Coefficient Comparison: Ridge vs Lasso')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f'F{i}' for i in important_features], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Plot 2: Performance Metrics Comparison
        ax2 = axes[0, 1]
        
        metrics = ['test_r2', 'test_mse', 'test_mae']
        metric_labels = ['R²', 'MSE', 'MAE']
        
        # Normalize metrics for comparison (higher is better)
        normalized_metrics = {}
        for model_name, results in self.results.items():
            normalized_metrics[model_name] = [
                results['test_r2'],  # R² (higher is better)
                1 / (1 + results['test_mse']),  # Normalized MSE (higher is better)
                1 / (1 + results['test_mae'])   # Normalized MAE (higher is better)
            ]
        
        x_pos = np.arange(len(metric_labels))
        width = 0.2
        
        for i, (model_name, values) in enumerate(normalized_metrics.items()):
            if model_name in ['Ridge', 'Lasso', 'Linear']:  # Focus on main models
                ax2.bar(x_pos + i * width, values, width, 
                       label=model_name, color=colors[model_name], alpha=0.8)
        
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Normalized Score (Higher = Better)')
        ax2.set_title('Performance Metrics Comparison')
        ax2.set_xticks(x_pos + width)
        ax2.set_xticklabels(metric_labels)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Model Sparsity (Number of Non-zero Coefficients)
        ax3 = axes[0, 2]
        
        models = ['Linear', 'Ridge', 'Lasso', 'ElasticNet']
        non_zero_counts = [self.results[model]['non_zero_coefs'] for model in models]
        model_colors = [colors[model] for model in models]
        
        bars = ax3.bar(models, non_zero_counts, color=model_colors, alpha=0.8)
        ax3.set_ylabel('Number of Non-zero Coefficients')
        ax3.set_title('Model Sparsity Comparison')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, non_zero_counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count}', ha='center', va='bottom')
        
        # Total features line
        ax3.axhline(y=len(self.feature_names), color='black', linestyle='--', 
                   alpha=0.7, label=f'Total features: {len(self.feature_names)}')
        ax3.legend()
        
        # Plot 4: Coefficient Magnitude Distribution
        ax4 = axes[1, 0]
        
        ridge_coef_abs = np.abs(ridge_coefs[np.abs(ridge_coefs) > 1e-10])
        lasso_coef_abs = np.abs(lasso_coefs[np.abs(lasso_coefs) > 1e-10])
        
        # Create histograms
        ax4.hist(ridge_coef_abs, bins=20, alpha=0.7, label='Ridge', 
                color=colors['Ridge'], density=True)
        ax4.hist(lasso_coef_abs, bins=20, alpha=0.7, label='Lasso', 
                color=colors['Lasso'], density=True)
        
        ax4.set_xlabel('Absolute Coefficient Value')
        ax4.set_ylabel('Density')
        ax4.set_title('Coefficient Magnitude Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Overfitting Analysis (Train vs Test R²)
        ax5 = axes[1, 1]
        
        train_r2_values = [self.results[model]['train_r2'] for model in models]
        test_r2_values = [self.results[model]['test_r2'] for model in models]
        
        # Scatter plot
        for i, model in enumerate(models):
            ax5.scatter(train_r2_values[i], test_r2_values[i], 
                       s=100, color=colors[model], label=model, alpha=0.8)
        
        # Perfect generalization line
        min_r2 = min(min(train_r2_values), min(test_r2_values))
        max_r2 = max(max(train_r2_values), max(test_r2_values))
        ax5.plot([min_r2, max_r2], [min_r2, max_r2], 'k--', alpha=0.5, 
                label='Perfect Generalization')
        
        ax5.set_xlabel('Training R²')
        ax5.set_ylabel('Test R²')
        ax5.set_title('Overfitting Analysis')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Add annotations for each point
        for i, model in enumerate(models):
            ax5.annotate(model, (train_r2_values[i], test_r2_values[i]),
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=9, alpha=0.8)
        
        # Plot 6: Feature Selection Pattern Analysis
        ax6 = axes[1, 2]
        
        # Show feature selection overlap
        ridge_selected = set(np.where(np.abs(ridge_coefs) > 1e-5)[0])
        lasso_selected = set(np.where(np.abs(lasso_coefs) > 1e-5)[0])
        
        ridge_only = ridge_selected - lasso_selected
        lasso_only = lasso_selected - ridge_selected
        common = ridge_selected.intersection(lasso_selected)
        
        categories = ['Ridge Only', 'Common', 'Lasso Only']
        counts = [len(ridge_only), len(common), len(lasso_only)]
        colors_venn = ['lightblue', 'purple', 'lightcoral']
        
        bars = ax6.bar(categories, counts, color=colors_venn, alpha=0.8)
        ax6.set_ylabel('Number of Selected Features')
        ax6.set_title('Feature Selection Overlap')
        ax6.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Ridge vs Lasso comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_alpha_sensitivity(self, save_path=None, figsize=(15, 5)):
        """
        Plot sensitivity to alpha parameter for Ridge and Lasso.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        figsize : tuple
            Figure size
        """
        print("\n📊 Analyzing Alpha Sensitivity")
        print("=" * 35)
        
        alpha_range = np.logspace(-4, 2, 50)
        
        ridge_scores = []
        lasso_scores = []
        ridge_sparsity = []
        lasso_sparsity = []
        
        for alpha in alpha_range:
            # Ridge
            ridge = Ridge(alpha=alpha, random_state=self.random_state)
            ridge.fit(self.X_train, self.y_train)
            ridge_pred = ridge.predict(self.X_test)
            ridge_scores.append(r2_score(self.y_test, ridge_pred))
            ridge_sparsity.append(np.sum(np.abs(ridge.coef_) > 1e-5))
            
            # Lasso
            lasso = Lasso(alpha=alpha, max_iter=2000, random_state=self.random_state)
            lasso.fit(self.X_train, self.y_train)
            lasso_pred = lasso.predict(self.X_test)
            lasso_scores.append(r2_score(self.y_test, lasso_pred))
            lasso_sparsity.append(np.sum(np.abs(lasso.coef_) > 1e-5))
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle('Alpha Sensitivity Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: R² vs Alpha
        ax1 = axes[0]
        ax1.semilogx(alpha_range, ridge_scores, 'o-', label='Ridge', color='blue')
        ax1.semilogx(alpha_range, lasso_scores, 's-', label='Lasso', color='red')
        ax1.set_xlabel('Alpha (Regularization Strength)')
        ax1.set_ylabel('Test R²')
        ax1.set_title('Performance vs Alpha')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Sparsity vs Alpha
        ax2 = axes[1]
        ax2.semilogx(alpha_range, ridge_sparsity, 'o-', label='Ridge', color='blue')
        ax2.semilogx(alpha_range, lasso_sparsity, 's-', label='Lasso', color='red')
        ax2.set_xlabel('Alpha (Regularization Strength)')
        ax2.set_ylabel('Number of Non-zero Coefficients')
        ax2.set_title('Sparsity vs Alpha')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Performance-Sparsity Trade-off
        ax3 = axes[2]
        ax3.scatter(ridge_sparsity, ridge_scores, label='Ridge', color='blue', alpha=0.7)
        ax3.scatter(lasso_sparsity, lasso_scores, label='Lasso', color='red', alpha=0.7)
        ax3.set_xlabel('Number of Non-zero Coefficients')
        ax3.set_ylabel('Test R²')
        ax3.set_title('Performance-Sparsity Trade-off')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Alpha sensitivity plot saved to {save_path}")
        
        plt.show()
    
    def print_detailed_comparison(self):
        """Print detailed numerical comparison between Ridge and Lasso."""
        if not self.results:
            print("No results available. Run compare_models first.")
            return
        
        print("\n📊 DETAILED RIDGE VS LASSO COMPARISON")
        print("=" * 50)
        
        # Performance comparison
        print("\nPerformance Metrics:")
        print("-" * 20)
        for model_name in ['Ridge', 'Lasso']:
            results = self.results[model_name]
            print(f"{model_name}:")
            print(f"  Test R²: {results['test_r2']:.4f}")
            print(f"  Test MSE: {results['test_mse']:.4f}")
            print(f"  Test MAE: {results['test_mae']:.4f}")
            print(f"  Non-zero coefficients: {results['non_zero_coefs']}/{len(self.feature_names)}")
            if results['alpha']:
                print(f"  Alpha: {results['alpha']:.4f}")
            print()
        
        # Feature selection analysis
        ridge_coefs = self.results['Ridge']['coefficients']
        lasso_coefs = self.results['Lasso']['coefficients']
        
        ridge_selected = np.where(np.abs(ridge_coefs) > 1e-5)[0]
        lasso_selected = np.where(np.abs(lasso_coefs) > 1e-5)[0]
        
        common_features = set(ridge_selected).intersection(set(lasso_selected))
        ridge_only = set(ridge_selected) - set(lasso_selected)
        lasso_only = set(lasso_selected) - set(ridge_selected)
        
        print("Feature Selection Analysis:")
        print("-" * 30)
        print(f"Ridge selected: {len(ridge_selected)} features")
        print(f"Lasso selected: {len(lasso_selected)} features")
        print(f"Common features: {len(common_features)}")
        print(f"Ridge only: {len(ridge_only)}")
        print(f"Lasso only: {len(lasso_only)}")
        
        # Top features
        print(f"\nTop 10 Features by Absolute Coefficient:")
        print("-" * 40)
        
        for model_name, coefs in [('Ridge', ridge_coefs), ('Lasso', lasso_coefs)]:
            top_indices = np.argsort(np.abs(coefs))[-10:][::-1]
            print(f"\n{model_name}:")
            for i, idx in enumerate(top_indices):
                feature_name = self.feature_names[idx] if hasattr(self, 'feature_names') else f'Feature_{idx}'
                print(f"  {i+1:2d}. {feature_name}: {coefs[idx]:8.4f}")
        
        # Statistical summary
        print(f"\nCoefficient Statistics:")
        print("-" * 25)
        print(f"Ridge - Mean: {np.mean(ridge_coefs):.4f}, Std: {np.std(ridge_coefs):.4f}")
        print(f"Lasso - Mean: {np.mean(lasso_coefs):.4f}, Std: {np.std(lasso_coefs):.4f}")


def main():
    """
    Main function to demonstrate Ridge vs Lasso comparison.
    """
    print("🎯 RIDGE VS LASSO COMPARISON ANALYSIS")
    print("=" * 45)
    
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Initialize comparator
    comparator = RidgeLassoComparator(random_state=42)
    
    # Test with diabetes dataset
    print("\n🏥 PHASE 1: Diabetes Dataset Analysis")
    print("=" * 40)
    
    dataset_info = comparator.load_data('diabetes')
    
    # Compare models
    results = comparator.compare_models()
    
    # Create comprehensive comparison plot
    comparator.plot_ridge_vs_lasso_comparison(save_path='plots/ridge_vs_lasso_comparison.png')
    
    # Alpha sensitivity analysis
    comparator.plot_alpha_sensitivity(save_path='plots/alpha_sensitivity.png')
    
    # Print detailed comparison
    comparator.print_detailed_comparison()
    
    # Test with high-dimensional synthetic data
    print("\n\n🔬 PHASE 2: High-Dimensional Synthetic Data")
    print("=" * 45)
    
    high_dim_info = comparator.load_data('high_dim')
    
    # Compare models on high-dimensional data
    results_high_dim = comparator.compare_models()
    
    # Plot comparison for high-dimensional case
    comparator.plot_ridge_vs_lasso_comparison(save_path='plots/ridge_vs_lasso_high_dim.png')
    
    # Alpha sensitivity for high-dimensional case
    comparator.plot_alpha_sensitivity(save_path='plots/alpha_sensitivity_high_dim.png')
    
    print("\n✅ RIDGE VS LASSO COMPARISON COMPLETE!")
    print("📁 Check the 'plots' folder for visualizations.")
    print("🔧 Key insights demonstrated:")
    print("   - Ridge: Shrinks coefficients uniformly, no feature selection")
    print("   - Lasso: Sets coefficients to zero, automatic feature selection")
    print("   - Performance vs sparsity trade-offs")
    print("   - Alpha parameter sensitivity analysis")
    print("   - Feature selection pattern differences")

if __name__ == "__main__":
    main() 