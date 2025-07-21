"""
Polynomial Regression Demo: Bias-Variance Tradeoff Analysis

This module demonstrates the bias-variance tradeoff using polynomial regression
with degrees ranging from 1 to 20 on synthetic nonlinear data.

Key Concepts:
- Low degree polynomials: High bias, low variance (underfitting)
- High degree polynomials: Low bias, high variance (overfitting)
- Optimal degree: Balance between bias and variance

Author: ML Learning Series
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class PolynomialBiasVarianceDemo:
    """
    Comprehensive demonstration of bias-variance tradeoff using polynomial regression.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the demo with random state for reproducibility.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Store results
        self.results = {}
        self.true_function = None
        self.X_test_range = None
        
    def generate_synthetic_data(self, n_samples=100, noise_level=0.3):
        """
        Generate synthetic nonlinear data for bias-variance analysis.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to generate
        noise_level : float
            Standard deviation of noise
            
        Returns:
        --------
        X, y : arrays
            Generated features and target values
        """
        # Generate true function: sinusoidal with polynomial components
        X = np.linspace(0, 1, n_samples).reshape(-1, 1)
        
        # True underlying function (complex nonlinear)
        def true_function(x):
            return 1.5 * x + 0.5 * np.sin(15 * x) + 0.3 * x**2 - 0.2 * x**3
        
        self.true_function = true_function
        
        # Generate clean target values
        y_clean = true_function(X.ravel())
        
        # Add noise
        noise = np.random.normal(0, noise_level, n_samples)
        y = y_clean + noise
        
        # Store test range for plotting
        self.X_test_range = np.linspace(0, 1, 200).reshape(-1, 1)
        
        print(f"Generated {n_samples} samples with noise level {noise_level}")
        print(f"True function: 1.5x + 0.5sin(15x) + 0.3x² - 0.2x³")
        
        return X, y
    
    def fit_polynomial_models(self, X, y, max_degree=20, test_size=0.3):
        """
        Fit polynomial models of varying degrees and evaluate performance.
        
        Parameters:
        -----------
        X, y : arrays
            Training data
        max_degree : int
            Maximum polynomial degree to test
        test_size : float
            Fraction of data to use for testing
            
        Returns:
        --------
        results : dict
            Dictionary containing training and test errors
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        degrees = range(1, max_degree + 1)
        train_errors = []
        test_errors = []
        models = {}
        
        print("Fitting polynomial models...")
        
        for degree in degrees:
            # Create polynomial pipeline
            poly_model = Pipeline([
                ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                ('linear', LinearRegression())
            ])
            
            # Fit model
            poly_model.fit(X_train, y_train)
            
            # Predictions
            y_train_pred = poly_model.predict(X_train)
            y_test_pred = poly_model.predict(X_test)
            
            # Calculate errors
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            
            train_errors.append(train_mse)
            test_errors.append(test_mse)
            models[degree] = poly_model
            
            if degree % 5 == 0:
                print(f"  Degree {degree:2d}: Train MSE = {train_mse:.4f}, Test MSE = {test_mse:.4f}")
        
        # Store results
        self.results = {
            'degrees': list(degrees),
            'train_errors': train_errors,
            'test_errors': test_errors,
            'models': models,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
        
        # Find optimal degree
        optimal_idx = np.argmin(test_errors)
        optimal_degree = degrees[optimal_idx]
        
        print(f"\nOptimal polynomial degree: {optimal_degree}")
        print(f"Minimum test MSE: {test_errors[optimal_idx]:.4f}")
        
        return self.results
    
    def bias_variance_decomposition(self, X, y, degree, n_bootstrap=100):
        """
        Perform bias-variance decomposition for a specific polynomial degree.
        
        Parameters:
        -----------
        X, y : arrays
            Training data
        degree : int
            Polynomial degree to analyze
        n_bootstrap : int
            Number of bootstrap samples
            
        Returns:
        --------
        decomposition : dict
            Bias, variance, and noise estimates
        """
        print(f"\nPerforming bias-variance decomposition for degree {degree}...")
        
        # Create test points
        X_test = self.X_test_range
        y_true = self.true_function(X_test.ravel())
        
        # Store predictions from all bootstrap samples
        predictions = []
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            n_samples = len(X)
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[bootstrap_indices]
            y_boot = y[bootstrap_indices]
            
            # Fit model
            poly_model = Pipeline([
                ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                ('linear', LinearRegression())
            ])
            
            poly_model.fit(X_boot, y_boot)
            
            # Predict on test points
            y_pred = poly_model.predict(X_test)
            predictions.append(y_pred)
        
        predictions = np.array(predictions)
        
        # Calculate bias and variance
        mean_prediction = np.mean(predictions, axis=0)
        bias_squared = np.mean((mean_prediction - y_true)**2)
        variance = np.mean(np.var(predictions, axis=0))
        
        # Estimate noise (irreducible error)
        # Use residuals from the best model fit
        best_model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('linear', LinearRegression())
        ])
        best_model.fit(X, y)
        y_pred_full = best_model.predict(X)
        noise = np.var(y - y_pred_full)
        
        decomposition = {
            'bias_squared': bias_squared,
            'variance': variance,
            'noise': noise,
            'total_error': bias_squared + variance + noise,
            'predictions': predictions,
            'mean_prediction': mean_prediction,
            'true_values': y_true
        }
        
        print(f"  Bias² = {bias_squared:.4f}")
        print(f"  Variance = {variance:.4f}")
        print(f"  Noise = {noise:.4f}")
        print(f"  Total Error = {bias_squared + variance + noise:.4f}")
        
        return decomposition
    
    def plot_bias_variance_tradeoff(self, save_path=None):
        """
        Create comprehensive visualization of bias-variance tradeoff.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        """
        if not self.results:
            raise ValueError("Must fit models first using fit_polynomial_models()")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Polynomial Regression: Bias-Variance Tradeoff Analysis', 
                     fontsize=16, fontweight='bold')
        
        degrees = self.results['degrees']
        train_errors = self.results['train_errors']
        test_errors = self.results['test_errors']
        
        # 1. Training vs Test Error
        ax1 = axes[0, 0]
        ax1.plot(degrees, train_errors, 'o-', label='Training Error', 
                linewidth=2, markersize=6)
        ax1.plot(degrees, test_errors, 's-', label='Test Error', 
                linewidth=2, markersize=6)
        ax1.set_xlabel('Polynomial Degree')
        ax1.set_ylabel('Mean Squared Error')
        ax1.set_title('Training vs Test Error')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Find and mark optimal degree
        optimal_idx = np.argmin(test_errors)
        optimal_degree = degrees[optimal_idx]
        ax1.axvline(optimal_degree, color='red', linestyle='--', alpha=0.7,
                   label=f'Optimal Degree ({optimal_degree})')
        ax1.legend()
        
        # 2. Model Complexity Regions
        ax2 = axes[0, 1]
        ax2.plot(degrees, test_errors, 'o-', linewidth=2, markersize=6)
        ax2.set_xlabel('Polynomial Degree')
        ax2.set_ylabel('Test Error')
        ax2.set_title('Model Complexity Regions')
        ax2.grid(True, alpha=0.3)
        
        # Add regions
        ax2.axvspan(1, 5, alpha=0.2, color='blue', label='Underfitting\n(High Bias)')
        ax2.axvspan(optimal_degree-2, optimal_degree+2, alpha=0.2, color='green', 
                   label='Optimal\n(Balanced)')
        ax2.axvspan(15, 20, alpha=0.2, color='red', label='Overfitting\n(High Variance)')
        ax2.legend()
        
        # 3. Function Approximations
        ax3 = axes[0, 2]
        X_plot = self.X_test_range
        y_true = self.true_function(X_plot.ravel())
        
        ax3.plot(X_plot, y_true, 'k-', linewidth=3, label='True Function', alpha=0.8)
        
        # Show predictions for different degrees
        for degree in [2, optimal_degree, 15]:
            if degree in self.results['models']:
                y_pred = self.results['models'][degree].predict(X_plot)
                if degree == 2:
                    label = f'Degree {degree} (Underfit)'
                    color = 'blue'
                elif degree == optimal_degree:
                    label = f'Degree {degree} (Optimal)'
                    color = 'green'
                else:
                    label = f'Degree {degree} (Overfit)'
                    color = 'red'
                
                ax3.plot(X_plot, y_pred, '--', linewidth=2, 
                        label=label, color=color, alpha=0.8)
        
        # Scatter training points
        ax3.scatter(self.results['X_train'], self.results['y_train'], 
                   alpha=0.6, s=30, color='gray', label='Training Data')
        
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_title('Function Approximations')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Bias-Variance Decomposition
        ax4 = axes[1, 0]
        
        # Perform decomposition for several degrees
        decomp_degrees = [2, 6, 10, 15]
        bias_values = []
        variance_values = []
        
        for deg in decomp_degrees:
            if deg <= max(degrees):
                decomp = self.bias_variance_decomposition(
                    self.results['X_train'], self.results['y_train'], deg, n_bootstrap=50
                )
                bias_values.append(decomp['bias_squared'])
                variance_values.append(decomp['variance'])
        
        x_pos = np.arange(len(decomp_degrees))
        width = 0.35
        
        ax4.bar(x_pos - width/2, bias_values, width, label='Bias²', alpha=0.8)
        ax4.bar(x_pos + width/2, variance_values, width, label='Variance', alpha=0.8)
        
        ax4.set_xlabel('Polynomial Degree')
        ax4.set_ylabel('Error Component')
        ax4.set_title('Bias-Variance Decomposition')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(decomp_degrees)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Prediction Uncertainty
        ax5 = axes[1, 1]
        
        # Show prediction uncertainty for different degrees
        for i, degree in enumerate([2, optimal_degree, 15]):
            if degree in self.results['models']:
                # Generate multiple predictions with noise
                n_samples = 50
                predictions = []
                
                for _ in range(n_samples):
                    # Add small perturbation to training data
                    X_perturb = self.results['X_train'] + np.random.normal(0, 0.01, self.results['X_train'].shape)
                    y_perturb = self.results['y_train'] + np.random.normal(0, 0.05, self.results['y_train'].shape)
                    
                    # Refit model
                    poly_model = Pipeline([
                        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
                        ('linear', LinearRegression())
                    ])
                    poly_model.fit(X_perturb, y_perturb)
                    
                    y_pred = poly_model.predict(X_plot)
                    predictions.append(y_pred)
                
                predictions = np.array(predictions)
                mean_pred = np.mean(predictions, axis=0)
                std_pred = np.std(predictions, axis=0)
                
                color = ['blue', 'green', 'red'][i]
                alpha = 0.3
                
                ax5.plot(X_plot, mean_pred, '-', color=color, linewidth=2,
                        label=f'Degree {degree}')
                ax5.fill_between(X_plot.ravel(), 
                               mean_pred - 2*std_pred, 
                               mean_pred + 2*std_pred,
                               alpha=alpha, color=color)
        
        ax5.plot(X_plot, y_true, 'k-', linewidth=3, label='True Function')
        ax5.set_xlabel('X')
        ax5.set_ylabel('Y')
        ax5.set_title('Prediction Uncertainty (±2σ)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Error Components vs Degree
        ax6 = axes[1, 2]
        
        # Calculate bias and variance for all degrees
        bias_all = []
        variance_all = []
        
        for degree in [1, 3, 5, 7, 10, 15, 20]:
            if degree <= max(degrees):
                decomp = self.bias_variance_decomposition(
                    self.results['X_train'], self.results['y_train'], 
                    degree, n_bootstrap=30
                )
                bias_all.append(decomp['bias_squared'])
                variance_all.append(decomp['variance'])
        
        degree_subset = [1, 3, 5, 7, 10, 15, 20]
        degree_subset = [d for d in degree_subset if d <= max(degrees)]
        
        ax6.plot(degree_subset, bias_all, 'o-', label='Bias²', linewidth=2, markersize=6)
        ax6.plot(degree_subset, variance_all, 's-', label='Variance', linewidth=2, markersize=6)
        ax6.plot(degree_subset, np.array(bias_all) + np.array(variance_all), 
                '^-', label='Bias² + Variance', linewidth=2, markersize=6)
        
        ax6.set_xlabel('Polynomial Degree')
        ax6.set_ylabel('Error Component')
        ax6.set_title('Bias² vs Variance vs Degree')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to: {save_path}")
        
        plt.show()
        
        return fig
    
    def print_summary(self):
        """Print summary of the bias-variance analysis."""
        if not self.results:
            print("No results available. Run fit_polynomial_models() first.")
            return
        
        degrees = self.results['degrees']
        train_errors = self.results['train_errors']
        test_errors = self.results['test_errors']
        
        optimal_idx = np.argmin(test_errors)
        optimal_degree = degrees[optimal_idx]
        
        print("\n" + "="*60)
        print("POLYNOMIAL REGRESSION BIAS-VARIANCE ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Dataset: Synthetic nonlinear function with noise")
        print(f"True function: 1.5x + 0.5sin(15x) + 0.3x² - 0.2x³")
        print(f"Polynomial degrees tested: {min(degrees)} to {max(degrees)}")
        
        print(f"\nOptimal Model:")
        print(f"  Degree: {optimal_degree}")
        print(f"  Training MSE: {train_errors[optimal_idx]:.4f}")
        print(f"  Test MSE: {test_errors[optimal_idx]:.4f}")
        
        # Identify underfitting and overfitting regions
        underfit_threshold = test_errors[optimal_idx] * 1.5
        overfit_start = optimal_degree + 3
        
        underfit_degrees = [d for i, d in enumerate(degrees) 
                          if test_errors[i] > underfit_threshold and d < optimal_degree]
        overfit_degrees = [d for i, d in enumerate(degrees) 
                         if d >= overfit_start and train_errors[i] < test_errors[i] * 0.5]
        
        print(f"\nUnderfitting Region (High Bias):")
        print(f"  Degrees: {underfit_degrees}")
        print(f"  Characteristics: High training and test error")
        
        print(f"\nOverfitting Region (High Variance):")
        print(f"  Degrees: {overfit_degrees}")
        print(f"  Characteristics: Low training error, high test error")
        
        print(f"\nKey Insights:")
        print(f"  • Low-degree polynomials underfit (high bias, low variance)")
        print(f"  • High-degree polynomials overfit (low bias, high variance)")
        print(f"  • Optimal degree balances bias and variance")
        print(f"  • Test error forms characteristic U-shape")
        
        print("="*60)


def main():
    """
    Main execution function demonstrating polynomial regression bias-variance tradeoff.
    """
    print("Polynomial Regression: Bias-Variance Tradeoff Demo")
    print("="*55)
    
    # Initialize demo
    demo = PolynomialBiasVarianceDemo(random_state=42)
    
    # Generate synthetic data
    print("\n1. Generating synthetic nonlinear data...")
    X, y = demo.generate_synthetic_data(n_samples=100, noise_level=0.3)
    
    # Fit polynomial models
    print("\n2. Fitting polynomial models (degree 1-20)...")
    results = demo.fit_polynomial_models(X, y, max_degree=20)
    
    # Create visualizations
    print("\n3. Creating bias-variance tradeoff visualizations...")
    fig = demo.plot_bias_variance_tradeoff(save_path='plots/polynomial_bias_variance.png')
    
    # Print summary
    demo.print_summary()
    
    print("\n4. Individual bias-variance decomposition examples...")
    
    # Detailed analysis for specific degrees
    for degree in [2, 8, 15]:
        print(f"\nDetailed analysis for degree {degree}:")
        decomp = demo.bias_variance_decomposition(X, y, degree, n_bootstrap=100)
        
        total_error = decomp['bias_squared'] + decomp['variance'] + decomp['noise']
        print(f"  Bias² contribution: {decomp['bias_squared']/total_error*100:.1f}%")
        print(f"  Variance contribution: {decomp['variance']/total_error*100:.1f}%")
        print(f"  Noise contribution: {decomp['noise']/total_error*100:.1f}%")


if __name__ == "__main__":
    main() 