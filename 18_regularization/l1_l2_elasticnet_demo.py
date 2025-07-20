"""
L1, L2, and ElasticNet Regularization Demo
==========================================

This module demonstrates the effects of different regularization techniques
on linear regression models, showing how they handle overfitting and feature selection.

Regularization Techniques:
- L1 (Lasso): Promotes sparsity and feature selection
- L2 (Ridge): Penalizes large weights uniformly
- ElasticNet: Combines L1 and L2 penalties

Key Concepts:
- Bias-variance tradeoff
- Feature selection vs. feature shrinkage
- Regularization strength effects
- Model performance comparison

Author: Machine Learning Fundamentals Course
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RegularizationDemo:
    """
    Comprehensive demonstration of regularization techniques.
    
    This class provides tools for comparing Ridge, Lasso, and ElasticNet
    regression with detailed analysis and visualization.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the regularization demo.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_and_prepare_data(self, dataset='diabetes'):
        """
        Load and prepare dataset for regularization analysis.
        
        Parameters:
        -----------
        dataset : str
            Dataset to use ('diabetes' or 'synthetic')
            
        Returns:
        --------
        dict : Dataset information
        """
        if dataset == 'diabetes':
            data = load_diabetes()
            dataset_name = "Diabetes Dataset"
            print("🏥 Loading Diabetes Dataset")
            print("=" * 40)
            print("Target: Diabetes progression measure")
            print("Features: Age, sex, BMI, blood pressure, etc.")
        else:
            # Use diabetes as fallback for consistency
            data = load_diabetes()
            dataset_name = "Diabetes Dataset"
            print("🏥 Loading Diabetes Dataset")
            print("=" * 40)
            print("Target: Diabetes progression measure")
            print("Features: Age, sex, BMI, blood pressure, etc.")
        
        X, y = data.data, data.target
        feature_names = data.feature_names
        
        print(f"Dataset shape: {X.shape}")
        print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")
        print(f"Features: {', '.join(feature_names[:5])}...")
        
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
        
        dataset_info = {
            'name': dataset_name,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'feature_names': feature_names,
            'target_name': 'progression',
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler
        }
        
        return dataset_info
    
    def compare_regularization_methods(self, alpha_range=None):
        """
        Compare Ridge, Lasso, and ElasticNet across different alpha values.
        
        Parameters:
        -----------
        alpha_range : array-like, optional
            Range of alpha values to test
            
        Returns:
        --------
        dict : Comparison results
        """
        if alpha_range is None:
            alpha_range = np.logspace(-3, 3, 20)  # 0.001 to 1000
        
        print("\n🔍 Comparing Regularization Methods")
        print("=" * 40)
        
        # Models to compare
        models = {
            'Linear (No Regularization)': LinearRegression(),
            'Ridge (L2)': Ridge(),
            'Lasso (L1)': Lasso(max_iter=2000),
            'ElasticNet (L1+L2)': ElasticNet(max_iter=2000, l1_ratio=0.5)
        }
        
        results = {}
        
        for model_name, model in models.items():
            print(f"\nTesting {model_name}...")
            
            if model_name == 'Linear (No Regularization)':
                # Fit linear regression without regularization
                model.fit(self.X_train, self.y_train)
                y_pred_train = model.predict(self.X_train)
                y_pred_test = model.predict(self.X_test)
                
                # Calculate metrics
                train_mse = mean_squared_error(self.y_train, y_pred_train)
                test_mse = mean_squared_error(self.y_test, y_pred_test)
                train_r2 = r2_score(self.y_train, y_pred_train)
                test_r2 = r2_score(self.y_test, y_pred_test)
                non_zero_coef = np.sum(np.abs(model.coef_) > 1e-5)
                
                results[model_name] = {
                    'alpha': [0],
                    'train_mse': [train_mse],
                    'test_mse': [test_mse],
                    'train_r2': [train_r2],
                    'test_r2': [test_r2],
                    'non_zero_coef': [non_zero_coef],
                    'coefficients': [model.coef_.copy()]
                }
                
                print(f"  MSE: {test_mse:.3f}, R²: {test_r2:.3f}, Non-zero coef: {non_zero_coef}")
                
            else:
                # Test different alpha values
                alpha_results = {
                    'alpha': [],
                    'train_mse': [],
                    'test_mse': [],
                    'train_r2': [],
                    'test_r2': [],
                    'non_zero_coef': [],
                    'coefficients': []
                }
                
                for alpha in alpha_range:
                    model.set_params(alpha=alpha)
                    model.fit(self.X_train, self.y_train)
                    
                    y_pred_train = model.predict(self.X_train)
                    y_pred_test = model.predict(self.X_test)
                    
                    # Calculate metrics
                    train_mse = mean_squared_error(self.y_train, y_pred_train)
                    test_mse = mean_squared_error(self.y_test, y_pred_test)
                    train_r2 = r2_score(self.y_train, y_pred_train)
                    test_r2 = r2_score(self.y_test, y_pred_test)
                    non_zero_coef = np.sum(np.abs(model.coef_) > 1e-5)
                    
                    alpha_results['alpha'].append(alpha)
                    alpha_results['train_mse'].append(train_mse)
                    alpha_results['test_mse'].append(test_mse)
                    alpha_results['train_r2'].append(train_r2)
                    alpha_results['test_r2'].append(test_r2)
                    alpha_results['non_zero_coef'].append(non_zero_coef)
                    alpha_results['coefficients'].append(model.coef_.copy())
                
                results[model_name] = alpha_results
                
                # Find best alpha based on test R²
                best_idx = np.argmax(alpha_results['test_r2'])
                best_alpha = alpha_results['alpha'][best_idx]
                best_mse = alpha_results['test_mse'][best_idx]
                best_r2 = alpha_results['test_r2'][best_idx]
                best_non_zero = alpha_results['non_zero_coef'][best_idx]
                
                print(f"  Best α: {best_alpha:.3f}, MSE: {best_mse:.3f}, R²: {best_r2:.3f}, Non-zero coef: {best_non_zero}")
        
        self.results = results
        return results
    
    def plot_regularization_comparison(self, save_path=None, figsize=(16, 12)):
        """
        Create comprehensive plots comparing regularization methods.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        figsize : tuple
            Figure size
        """
        if not self.results:
            print("No results to plot. Run compare_regularization_methods first.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Regularization Methods Comparison', fontsize=16, fontweight='bold')
        
        # Colors for different methods
        colors = {
            'Linear (No Regularization)': 'red',
            'Ridge (L2)': 'blue',
            'Lasso (L1)': 'green',
            'ElasticNet (L1+L2)': 'orange'
        }
        
        # Plot 1: MSE vs Alpha
        ax1 = axes[0, 0]
        for model_name, results in self.results.items():
            if model_name == 'Linear (No Regularization)':
                ax1.axhline(y=results['test_mse'][0], color=colors[model_name], 
                           linestyle='--', label=model_name, alpha=0.7)
            else:
                ax1.semilogx(results['alpha'], results['test_mse'], 
                           'o-', color=colors[model_name], label=model_name)
        
        ax1.set_xlabel('Alpha (Regularization Strength)')
        ax1.set_ylabel('Test MSE')
        ax1.set_title('Test MSE vs Alpha')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: R² vs Alpha
        ax2 = axes[0, 1]
        for model_name, results in self.results.items():
            if model_name == 'Linear (No Regularization)':
                ax2.axhline(y=results['test_r2'][0], color=colors[model_name], 
                           linestyle='--', label=model_name, alpha=0.7)
            else:
                ax2.semilogx(results['alpha'], results['test_r2'], 
                           'o-', color=colors[model_name], label=model_name)
        
        ax2.set_xlabel('Alpha (Regularization Strength)')
        ax2.set_ylabel('Test R²')
        ax2.set_title('Test R² vs Alpha')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Number of Non-zero Coefficients
        ax3 = axes[0, 2]
        for model_name, results in self.results.items():
            if model_name == 'Linear (No Regularization)':
                ax3.axhline(y=results['non_zero_coef'][0], color=colors[model_name], 
                           linestyle='--', label=model_name, alpha=0.7)
            else:
                ax3.semilogx(results['alpha'], results['non_zero_coef'], 
                           'o-', color=colors[model_name], label=model_name)
        
        ax3.set_xlabel('Alpha (Regularization Strength)')
        ax3.set_ylabel('Number of Non-zero Coefficients')
        ax3.set_title('Feature Selection Effect')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Training vs Test MSE (Bias-Variance)
        ax4 = axes[1, 0]
        for model_name, results in self.results.items():
            if model_name != 'Linear (No Regularization)':
                ax4.semilogx(results['alpha'], results['train_mse'], 
                           '--', color=colors[model_name], alpha=0.7, label=f'{model_name} (Train)')
                ax4.semilogx(results['alpha'], results['test_mse'], 
                           '-', color=colors[model_name], label=f'{model_name} (Test)')
        
        ax4.set_xlabel('Alpha (Regularization Strength)')
        ax4.set_ylabel('MSE')
        ax4.set_title('Bias-Variance Tradeoff')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Best Model Comparison (Bar plot)
        ax5 = axes[1, 1]
        best_results = {}
        for model_name, results in self.results.items():
            if model_name == 'Linear (No Regularization)':
                best_results[model_name] = {
                    'r2': results['test_r2'][0],
                    'mse': results['test_mse'][0],
                    'non_zero': results['non_zero_coef'][0]
                }
            else:
                best_idx = np.argmax(results['test_r2'])
                best_results[model_name] = {
                    'r2': results['test_r2'][best_idx],
                    'mse': results['test_mse'][best_idx],
                    'non_zero': results['non_zero_coef'][best_idx]
                }
        
        models = list(best_results.keys())
        r2_scores = [best_results[model]['r2'] for model in models]
        model_colors = [colors[model] for model in models]
        
        bars = ax5.bar(models, r2_scores, color=model_colors, alpha=0.7)
        ax5.set_ylabel('Best Test R²')
        ax5.set_title('Best Performance Comparison')
        ax5.set_xticklabels(models, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, r2 in zip(bars, r2_scores):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{r2:.3f}', ha='center', va='bottom')
        
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Coefficient Comparison (Best models)
        ax6 = axes[1, 2]
        feature_names = [f'F{i}' for i in range(len(self.X_train[0]))]
        
        # Get best coefficients for each method
        best_coefs = {}
        for model_name, results in self.results.items():
            if model_name == 'Linear (No Regularization)':
                best_coefs[model_name] = results['coefficients'][0]
            else:
                best_idx = np.argmax(results['test_r2'])
                best_coefs[model_name] = results['coefficients'][best_idx]
        
        # Plot coefficients
        x_pos = np.arange(len(feature_names))
        width = 0.2
        
        for i, (model_name, coefs) in enumerate(best_coefs.items()):
            if model_name != 'Linear (No Regularization)':  # Skip linear for clarity
                ax6.bar(x_pos + i * width, coefs, width, 
                       label=model_name, color=colors[model_name], alpha=0.7)
        
        ax6.set_xlabel('Features')
        ax6.set_ylabel('Coefficient Value')
        ax6.set_title('Coefficient Comparison (Best Models)')
        ax6.set_xticks(x_pos + width)
        ax6.set_xticklabels(feature_names, rotation=45)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Regularization comparison plot saved to {save_path}")
        
        plt.show()
    
    def elasticnet_analysis(self, l1_ratios=None, alpha_range=None):
        """
        Detailed analysis of ElasticNet with different L1 ratios.
        
        Parameters:
        -----------
        l1_ratios : array-like, optional
            Range of L1 ratios to test
        alpha_range : array-like, optional
            Range of alpha values to test
            
        Returns:
        --------
        dict : ElasticNet analysis results
        """
        if l1_ratios is None:
            l1_ratios = np.linspace(0.1, 0.9, 9)
        
        if alpha_range is None:
            alpha_range = np.logspace(-2, 2, 10)
        
        print("\n🔄 ElasticNet L1 Ratio Analysis")
        print("=" * 35)
        
        elasticnet_results = {}
        
        for l1_ratio in l1_ratios:
            print(f"Testing L1 ratio: {l1_ratio:.1f}")
            
            results = {
                'alpha': [],
                'test_mse': [],
                'test_r2': [],
                'non_zero_coef': []
            }
            
            for alpha in alpha_range:
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=2000)
                model.fit(self.X_train, self.y_train)
                
                y_pred = model.predict(self.X_test)
                test_mse = mean_squared_error(self.y_test, y_pred)
                test_r2 = r2_score(self.y_test, y_pred)
                non_zero_coef = np.sum(np.abs(model.coef_) > 1e-5)
                
                results['alpha'].append(alpha)
                results['test_mse'].append(test_mse)
                results['test_r2'].append(test_r2)
                results['non_zero_coef'].append(non_zero_coef)
            
            elasticnet_results[l1_ratio] = results
            
            # Find best performance for this L1 ratio
            best_idx = np.argmax(results['test_r2'])
            best_alpha = results['alpha'][best_idx]
            best_r2 = results['test_r2'][best_idx]
            
            print(f"  Best α: {best_alpha:.3f}, R²: {best_r2:.3f}")
        
        return elasticnet_results
    
    def plot_elasticnet_analysis(self, elasticnet_results, save_path=None, figsize=(15, 10)):
        """
        Plot ElasticNet analysis results.
        
        Parameters:
        -----------
        elasticnet_results : dict
            Results from elasticnet_analysis
        save_path : str, optional
            Path to save the plot
        figsize : tuple
            Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('ElasticNet L1 Ratio Analysis', fontsize=16, fontweight='bold')
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(elasticnet_results)))
        
        # Plot 1: R² vs Alpha for different L1 ratios
        ax1 = axes[0, 0]
        for i, (l1_ratio, results) in enumerate(elasticnet_results.items()):
            ax1.semilogx(results['alpha'], results['test_r2'], 
                        'o-', color=colors[i], label=f'L1 ratio: {l1_ratio:.1f}')
        
        ax1.set_xlabel('Alpha')
        ax1.set_ylabel('Test R²')
        ax1.set_title('R² vs Alpha for Different L1 Ratios')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Number of features vs Alpha
        ax2 = axes[0, 1]
        for i, (l1_ratio, results) in enumerate(elasticnet_results.items()):
            ax2.semilogx(results['alpha'], results['non_zero_coef'], 
                        'o-', color=colors[i], label=f'L1 ratio: {l1_ratio:.1f}')
        
        ax2.set_xlabel('Alpha')
        ax2.set_ylabel('Number of Non-zero Coefficients')
        ax2.set_title('Feature Selection vs Alpha')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Best performance for each L1 ratio
        ax3 = axes[1, 0]
        l1_ratios = list(elasticnet_results.keys())
        best_r2s = []
        best_alphas = []
        best_features = []
        
        for l1_ratio, results in elasticnet_results.items():
            best_idx = np.argmax(results['test_r2'])
            best_r2s.append(results['test_r2'][best_idx])
            best_alphas.append(results['alpha'][best_idx])
            best_features.append(results['non_zero_coef'][best_idx])
        
        ax3.plot(l1_ratios, best_r2s, 'o-', linewidth=2, markersize=8)
        ax3.set_xlabel('L1 Ratio')
        ax3.set_ylabel('Best Test R²')
        ax3.set_title('Best Performance vs L1 Ratio')
        ax3.grid(True, alpha=0.3)
        
        # Add annotations for best performance
        best_l1_idx = np.argmax(best_r2s)
        best_l1_ratio = l1_ratios[best_l1_idx]
        best_overall_r2 = best_r2s[best_l1_idx]
        
        ax3.annotate(f'Best: L1={best_l1_ratio:.1f}\nR²={best_overall_r2:.3f}',
                    xy=(best_l1_ratio, best_overall_r2),
                    xytext=(best_l1_ratio + 0.1, best_overall_r2 - 0.01),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # Plot 4: Features vs L1 ratio (at best alpha for each)
        ax4 = axes[1, 1]
        ax4.plot(l1_ratios, best_features, 'o-', linewidth=2, markersize=8, color='green')
        ax4.set_xlabel('L1 Ratio')
        ax4.set_ylabel('Number of Selected Features')
        ax4.set_title('Feature Selection vs L1 Ratio')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ElasticNet analysis plot saved to {save_path}")
        
        plt.show()
    
    def print_summary(self):
        """Print comprehensive summary of regularization analysis."""
        if not self.results:
            print("No results available. Run analysis first.")
            return
        
        print("\n📊 REGULARIZATION ANALYSIS SUMMARY")
        print("=" * 50)
        
        # Find best model overall
        best_model = None
        best_r2 = -np.inf
        
        summary_data = []
        
        for model_name, results in self.results.items():
            if model_name == 'Linear (No Regularization)':
                r2 = results['test_r2'][0]
                mse = results['test_mse'][0]
                features = results['non_zero_coef'][0]
                alpha = 'N/A'
            else:
                best_idx = np.argmax(results['test_r2'])
                r2 = results['test_r2'][best_idx]
                mse = results['test_mse'][best_idx]
                features = results['non_zero_coef'][best_idx]
                alpha = results['alpha'][best_idx]
            
            summary_data.append({
                'Model': model_name,
                'Best Alpha': alpha if alpha != 'N/A' else 'N/A',
                'Test R²': f"{r2:.4f}",
                'Test MSE': f"{mse:.3f}",
                'Features': features
            })
            
            if r2 > best_r2:
                best_r2 = r2
                best_model = model_name
        
        # Create summary table
        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False))
        
        print(f"\n🏆 Best Model: {best_model}")
        print(f"📈 Best R²: {best_r2:.4f}")
        
        print("\n💡 Key Insights:")
        print("• L1 (Lasso) promotes sparsity - automatically selects features")
        print("• L2 (Ridge) shrinks coefficients uniformly - prevents overfitting")
        print("• ElasticNet combines both - balanced feature selection and shrinkage")
        print("• Higher alpha = stronger regularization = simpler models")
        print("• Choose method based on: interpretability needs vs. prediction accuracy")


def main():
    """
    Main function to demonstrate regularization techniques.
    """
    print("🎯 L1, L2, AND ELASTICNET REGULARIZATION DEMO")
    print("=" * 55)
    
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Initialize demo
    demo = RegularizationDemo(random_state=42)
    
    # Load and prepare data
    dataset_info = demo.load_and_prepare_data('diabetes')
    
    # Compare regularization methods
    print("\n🔍 PHASE 1: Regularization Methods Comparison")
    print("=" * 50)
    
    alpha_range = np.logspace(-3, 3, 25)  # More points for smoother curves
    results = demo.compare_regularization_methods(alpha_range)
    
    # Plot comparison
    demo.plot_regularization_comparison(save_path='plots/regularization_comparison.png')
    
    # ElasticNet detailed analysis
    print("\n🔄 PHASE 2: ElasticNet L1 Ratio Analysis")
    print("=" * 45)
    
    l1_ratios = np.linspace(0.1, 0.9, 9)
    alpha_range_elastic = np.logspace(-2, 2, 15)
    elasticnet_results = demo.elasticnet_analysis(l1_ratios, alpha_range_elastic)
    
    # Plot ElasticNet analysis
    demo.plot_elasticnet_analysis(elasticnet_results, save_path='plots/elasticnet_analysis.png')
    
    # Print comprehensive summary
    demo.print_summary()
    
    print("\n✅ REGULARIZATION DEMO COMPLETE!")
    print("📁 Check the 'plots' folder for visualizations.")
    print("🔧 Key takeaways:")
    print("   - L1 (Lasso): Feature selection via sparsity")
    print("   - L2 (Ridge): Coefficient shrinkage without feature removal")
    print("   - ElasticNet: Best of both worlds with tunable balance")
    print("   - Regularization strength controls bias-variance tradeoff")

if __name__ == "__main__":
    main() 