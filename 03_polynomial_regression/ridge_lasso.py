"""
Ridge and Lasso Regression Implementation from Scratch

This module implements:
- Ridge Regression (L2 regularization) with closed-form and gradient descent solutions
- Lasso Regression (L1 regularization) with coordinate descent
- Polynomial feature expansion for testing high-dimensional scenarios
- Comprehensive comparison with scikit-learn implementations
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge as SklearnRidge, Lasso as SklearnLasso
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class RidgeRegression:
    """
    Ridge Regression (L2 Regularization) implementation from scratch.
    
    Supports both closed-form solution and gradient descent optimization.
    """
    
    def __init__(self, alpha=1.0, method='closed_form', learning_rate=0.01, 
                 max_iter=1000, tolerance=1e-6):
        """
        Initialize Ridge Regression.
        
        Args:
            alpha (float): Regularization strength (λ)
            method (str): 'closed_form' or 'gradient_descent'
            learning_rate (float): Learning rate for gradient descent
            max_iter (int): Maximum iterations for gradient descent
            tolerance (float): Convergence tolerance for gradient descent
        """
        self.alpha = alpha
        self.method = method
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        
        # Model parameters
        self.weights = None
        self.intercept = None
        
        # Training history (for gradient descent)
        self.loss_history = []
        self.converged = False
    
    def _add_intercept(self, X):
        """Add intercept term to feature matrix."""
        return np.column_stack([np.ones(X.shape[0]), X])
    
    def _compute_loss(self, X, y):
        """Compute Ridge regression loss (MSE + L2 penalty)."""
        predictions = X @ self.weights
        mse = np.mean((y - predictions) ** 2)
        l2_penalty = self.alpha * np.sum(self.weights[1:] ** 2)  # Exclude intercept
        return mse + l2_penalty
    
    def _compute_gradients(self, X, y):
        """Compute gradients for Ridge regression."""
        m = X.shape[0]
        predictions = X @ self.weights
        error = predictions - y
        
        # Gradient of MSE + L2 penalty
        gradients = (2/m) * X.T @ error
        
        # Add L2 regularization (don't regularize intercept)
        gradients[1:] += 2 * self.alpha * self.weights[1:]
        
        return gradients
    
    def fit(self, X, y):
        """
        Fit Ridge regression model.
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Target values
        """
        # Add intercept term
        X_with_intercept = self._add_intercept(X)
        
        if self.method == 'closed_form':
            self._fit_closed_form(X_with_intercept, y)
        elif self.method == 'gradient_descent':
            self._fit_gradient_descent(X_with_intercept, y)
        else:
            raise ValueError("Method must be 'closed_form' or 'gradient_descent'")
        
        return self
    
    def _fit_closed_form(self, X, y):
        """Fit using closed-form solution: w = (X^T X + αI)^{-1} X^T y"""
        n_features = X.shape[1]
        
        # Create regularization matrix (don't regularize intercept)
        reg_matrix = self.alpha * np.eye(n_features)
        reg_matrix[0, 0] = 0  # Don't regularize intercept
        
        # Closed-form solution
        try:
            self.weights = np.linalg.solve(X.T @ X + reg_matrix, X.T @ y)
        except np.linalg.LinAlgError:
            # Use pseudoinverse if matrix is singular
            self.weights = np.linalg.pinv(X.T @ X + reg_matrix) @ X.T @ y
        
        self.intercept = self.weights[0]
        self.converged = True
    
    def _fit_gradient_descent(self, X, y):
        """Fit using gradient descent optimization."""
        # Initialize weights
        np.random.seed(42)
        self.weights = np.random.normal(0, 0.01, X.shape[1])
        
        self.loss_history = []
        prev_loss = float('inf')
        
        for iteration in range(self.max_iter):
            # Compute loss and gradients
            loss = self._compute_loss(X, y)
            gradients = self._compute_gradients(X, y)
            
            # Update weights
            self.weights -= self.learning_rate * gradients
            
            # Store loss
            self.loss_history.append(loss)
            
            # Check convergence
            if abs(prev_loss - loss) < self.tolerance:
                self.converged = True
                break
            
            prev_loss = loss
        
        self.intercept = self.weights[0]
    
    def predict(self, X):
        """Make predictions."""
        if self.weights is None:
            raise ValueError("Model must be fitted before making predictions")
        
        X_with_intercept = self._add_intercept(X)
        return X_with_intercept @ self.weights
    
    def get_params(self):
        """Get model parameters."""
        return {
            'weights': self.weights[1:] if self.weights is not None else None,
            'intercept': self.intercept,
            'alpha': self.alpha
        }


class LassoRegression:
    """
    Lasso Regression (L1 Regularization) implementation using coordinate descent.
    """
    
    def __init__(self, alpha=1.0, max_iter=1000, tolerance=1e-4):
        """
        Initialize Lasso Regression.
        
        Args:
            alpha (float): Regularization strength (λ)
            max_iter (int): Maximum iterations for coordinate descent
            tolerance (float): Convergence tolerance
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.tolerance = tolerance
        
        # Model parameters
        self.weights = None
        self.intercept = None
        
        # Training history
        self.loss_history = []
        self.converged = False
    
    def _soft_threshold(self, x, threshold):
        """Soft thresholding operator for L1 regularization."""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def _compute_loss(self, X, y):
        """Compute Lasso regression loss (MSE + L1 penalty)."""
        predictions = X @ self.weights + self.intercept
        mse = np.mean((y - predictions) ** 2)
        l1_penalty = self.alpha * np.sum(np.abs(self.weights))
        return mse + l1_penalty
    
    def fit(self, X, y):
        """
        Fit Lasso regression using coordinate descent.
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Target values
        """
        n_samples, n_features = X.shape
        
        # Standardize features for coordinate descent
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        X_scaled = (X - self.X_mean) / (self.X_std + 1e-8)
        
        self.y_mean = np.mean(y)
        y_centered = y - self.y_mean
        
        # Initialize weights
        self.weights = np.zeros(n_features)
        self.intercept = 0.0
        
        self.loss_history = []
        
        for iteration in range(self.max_iter):
            weights_old = self.weights.copy()
            
            # Update each weight using coordinate descent
            for j in range(n_features):
                # Compute residual without j-th feature
                residual = y_centered - X_scaled @ self.weights + self.weights[j] * X_scaled[:, j]
                
                # Compute correlation with j-th feature
                rho = X_scaled[:, j] @ residual / n_samples
                
                # Update weight using soft thresholding
                self.weights[j] = self._soft_threshold(rho, self.alpha / n_samples)
            
            # Compute loss
            loss = self._compute_loss(X_scaled, y_centered)
            self.loss_history.append(loss)
            
            # Check convergence
            if np.sum(np.abs(self.weights - weights_old)) < self.tolerance:
                self.converged = True
                break
        
        # Transform weights back to original scale
        self.weights = self.weights / (self.X_std + 1e-8)
        self.intercept = self.y_mean - np.sum(self.weights * self.X_mean)
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        if self.weights is None:
            raise ValueError("Model must be fitted before making predictions")
        
        return X @ self.weights + self.intercept
    
    def get_params(self):
        """Get model parameters."""
        return {
            'weights': self.weights,
            'intercept': self.intercept,
            'alpha': self.alpha,
            'n_nonzero': np.sum(np.abs(self.weights) > 1e-8)
        }


def generate_polynomial_data(n_samples=100, noise_level=0.1, degree=3, random_state=42):
    """
    Generate polynomial regression data for testing.
    
    Args:
        n_samples (int): Number of samples
        noise_level (float): Noise standard deviation
        degree (int): True polynomial degree
        random_state (int): Random seed
    
    Returns:
        tuple: (X, y) where X is features and y is target
    """
    np.random.seed(random_state)
    
    # Generate X uniformly in [-1, 1]
    X = np.random.uniform(-1, 1, (n_samples, 1))
    
    # Generate polynomial target with specific coefficients
    if degree == 1:
        y = 2 * X.ravel() + 1
    elif degree == 2:
        y = 2 * X.ravel()**2 + X.ravel() + 1
    elif degree == 3:
        y = X.ravel()**3 + 2 * X.ravel()**2 - X.ravel() + 1
    else:
        # General polynomial
        coeffs = np.random.normal(0, 1, degree + 1)
        y = sum(coeffs[i] * X.ravel()**i for i in range(degree + 1))
    
    # Add noise
    y += np.random.normal(0, noise_level, n_samples)
    
    return X, y


def create_polynomial_features(X, degree):
    """Create polynomial features up to given degree."""
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    return poly.fit_transform(X)


def plot_regularization_comparison(X, y, X_test, y_test, alphas, max_degree=10):
    """
    Plot comparison of Ridge vs Lasso for different regularization strengths.
    
    Args:
        X, y: Training data
        X_test, y_test: Test data
        alphas: List of regularization strengths
        max_degree: Maximum polynomial degree to test
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Ridge vs Lasso Regularization Comparison', fontsize=16)
    
    # Create polynomial features
    X_poly = create_polynomial_features(X, max_degree)
    X_test_poly = create_polynomial_features(X_test, max_degree)
    
    # Standardize features
    scaler = StandardScaler()
    X_poly_scaled = scaler.fit_transform(X_poly)
    X_test_poly_scaled = scaler.transform(X_test_poly)
    
    ridge_train_scores = []
    ridge_test_scores = []
    lasso_train_scores = []
    lasso_test_scores = []
    lasso_n_features = []
    
    for alpha in alphas:
        # Ridge Regression
        ridge = RidgeRegression(alpha=alpha, method='closed_form')
        ridge.fit(X_poly_scaled, y)
        
        ridge_train_pred = ridge.predict(X_poly_scaled)
        ridge_test_pred = ridge.predict(X_test_poly_scaled)
        
        ridge_train_scores.append(mean_squared_error(y, ridge_train_pred))
        ridge_test_scores.append(mean_squared_error(y_test, ridge_test_pred))
        
        # Lasso Regression
        lasso = LassoRegression(alpha=alpha, max_iter=2000)
        lasso.fit(X_poly_scaled, y)
        
        lasso_train_pred = lasso.predict(X_poly_scaled)
        lasso_test_pred = lasso.predict(X_test_poly_scaled)
        
        lasso_train_scores.append(mean_squared_error(y, lasso_train_pred))
        lasso_test_scores.append(mean_squared_error(y_test, lasso_test_pred))
        lasso_n_features.append(lasso.get_params()['n_nonzero'])
    
    # Plot 1: Ridge Regularization Path
    axes[0, 0].semilogx(alphas, ridge_train_scores, 'b-o', label='Training MSE', markersize=4)
    axes[0, 0].semilogx(alphas, ridge_test_scores, 'r-s', label='Test MSE', markersize=4)
    axes[0, 0].set_xlabel('Regularization Strength (α)')
    axes[0, 0].set_ylabel('Mean Squared Error')
    axes[0, 0].set_title('Ridge Regression: α vs MSE')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Lasso Regularization Path
    axes[0, 1].semilogx(alphas, lasso_train_scores, 'b-o', label='Training MSE', markersize=4)
    axes[0, 1].semilogx(alphas, lasso_test_scores, 'r-s', label='Test MSE', markersize=4)
    axes[0, 1].set_xlabel('Regularization Strength (α)')
    axes[0, 1].set_ylabel('Mean Squared Error')
    axes[0, 1].set_title('Lasso Regression: α vs MSE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Feature Selection (Lasso)
    axes[1, 0].semilogx(alphas, lasso_n_features, 'g-o', markersize=6)
    axes[1, 0].set_xlabel('Regularization Strength (α)')
    axes[1, 0].set_ylabel('Number of Non-zero Features')
    axes[1, 0].set_title('Lasso Feature Selection')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Direct Comparison
    axes[1, 1].semilogx(alphas, ridge_test_scores, 'b-o', label='Ridge Test MSE', markersize=4)
    axes[1, 1].semilogx(alphas, lasso_test_scores, 'r-s', label='Lasso Test MSE', markersize=4)
    axes[1, 1].set_xlabel('Regularization Strength (α)')
    axes[1, 1].set_ylabel('Test MSE')
    axes[1, 1].set_title('Ridge vs Lasso: Test Performance')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/regularization_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_bias_variance_tradeoff(X, y, X_test, y_test, degrees, alpha=0.1):
    """
    Plot bias-variance tradeoff for increasing polynomial degrees.
    
    Args:
        X, y: Training data
        X_test, y_test: Test data
        degrees: List of polynomial degrees to test
        alpha: Regularization strength
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Bias-Variance Tradeoff (α = {alpha})', fontsize=16)
    
    ridge_train_scores = []
    ridge_test_scores = []
    lasso_train_scores = []
    lasso_test_scores = []
    n_features_list = []
    
    for degree in degrees:
        # Create polynomial features
        X_poly = create_polynomial_features(X, degree)
        X_test_poly = create_polynomial_features(X_test, degree)
        
        # Standardize
        scaler = StandardScaler()
        X_poly_scaled = scaler.fit_transform(X_poly)
        X_test_poly_scaled = scaler.transform(X_test_poly)
        
        n_features_list.append(X_poly.shape[1])
        
        # Ridge
        ridge = RidgeRegression(alpha=alpha, method='closed_form')
        ridge.fit(X_poly_scaled, y)
        
        ridge_train_pred = ridge.predict(X_poly_scaled)
        ridge_test_pred = ridge.predict(X_test_poly_scaled)
        
        ridge_train_scores.append(mean_squared_error(y, ridge_train_pred))
        ridge_test_scores.append(mean_squared_error(y_test, ridge_test_pred))
        
        # Lasso
        lasso = LassoRegression(alpha=alpha, max_iter=2000)
        lasso.fit(X_poly_scaled, y)
        
        lasso_train_pred = lasso.predict(X_poly_scaled)
        lasso_test_pred = lasso.predict(X_test_poly_scaled)
        
        lasso_train_scores.append(mean_squared_error(y, lasso_train_pred))
        lasso_test_scores.append(mean_squared_error(y_test, lasso_test_pred))
    
    # Plot 1: Ridge Bias-Variance
    axes[0].plot(degrees, ridge_train_scores, 'b-o', label='Training MSE', markersize=6)
    axes[0].plot(degrees, ridge_test_scores, 'r-s', label='Test MSE', markersize=6)
    axes[0].set_xlabel('Polynomial Degree')
    axes[0].set_ylabel('Mean Squared Error')
    axes[0].set_title('Ridge: Model Complexity vs Error')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Lasso Bias-Variance
    axes[1].plot(degrees, lasso_train_scores, 'b-o', label='Training MSE', markersize=6)
    axes[1].plot(degrees, lasso_test_scores, 'r-s', label='Test MSE', markersize=6)
    axes[1].set_xlabel('Polynomial Degree')
    axes[1].set_ylabel('Mean Squared Error')
    axes[1].set_title('Lasso: Model Complexity vs Error')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Number of Features
    axes[2].plot(degrees, n_features_list, 'g-o', markersize=6)
    axes[2].set_xlabel('Polynomial Degree')
    axes[2].set_ylabel('Number of Features')
    axes[2].set_title('Model Complexity (# Features)')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/bias_variance_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_regression_examples(X, y, X_test, y_test):
    """
    Plot examples of underfit, overfit, and well-fit models.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Regularization Examples: Underfit vs Just Right vs Overfit', fontsize=16)
    
    # Create range for plotting smooth curves
    X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    
    # Example 1: Underfit (high regularization)
    X_poly_1 = create_polynomial_features(X, degree=8)
    X_test_poly_1 = create_polynomial_features(X_test, degree=8)
    X_plot_poly_1 = create_polynomial_features(X_plot, degree=8)
    
    scaler_1 = StandardScaler()
    X_poly_1_scaled = scaler_1.fit_transform(X_poly_1)
    X_test_poly_1_scaled = scaler_1.transform(X_test_poly_1)
    X_plot_poly_1_scaled = scaler_1.transform(X_plot_poly_1)
    
    ridge_underfit = RidgeRegression(alpha=100.0, method='closed_form')
    ridge_underfit.fit(X_poly_1_scaled, y)
    
    y_plot_underfit = ridge_underfit.predict(X_plot_poly_1_scaled)
    test_mse_underfit = mean_squared_error(y_test, ridge_underfit.predict(X_test_poly_1_scaled))
    
    axes[0].scatter(X, y, alpha=0.6, label='Training Data')
    axes[0].scatter(X_test, y_test, alpha=0.6, color='red', label='Test Data')
    axes[0].plot(X_plot, y_plot_underfit, 'g-', linewidth=2, label=f'Ridge (α=100)')
    axes[0].set_title(f'Underfit Model\nTest MSE: {test_mse_underfit:.3f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Example 2: Just Right (moderate regularization)
    ridge_good = RidgeRegression(alpha=1.0, method='closed_form')
    ridge_good.fit(X_poly_1_scaled, y)
    
    y_plot_good = ridge_good.predict(X_plot_poly_1_scaled)
    test_mse_good = mean_squared_error(y_test, ridge_good.predict(X_test_poly_1_scaled))
    
    axes[1].scatter(X, y, alpha=0.6, label='Training Data')
    axes[1].scatter(X_test, y_test, alpha=0.6, color='red', label='Test Data')
    axes[1].plot(X_plot, y_plot_good, 'g-', linewidth=2, label=f'Ridge (α=1.0)')
    axes[1].set_title(f'Well-fit Model\nTest MSE: {test_mse_good:.3f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Example 3: Overfit (low regularization)
    ridge_overfit = RidgeRegression(alpha=0.001, method='closed_form')
    ridge_overfit.fit(X_poly_1_scaled, y)
    
    y_plot_overfit = ridge_overfit.predict(X_plot_poly_1_scaled)
    test_mse_overfit = mean_squared_error(y_test, ridge_overfit.predict(X_test_poly_1_scaled))
    
    axes[2].scatter(X, y, alpha=0.6, label='Training Data')
    axes[2].scatter(X_test, y_test, alpha=0.6, color='red', label='Test Data')
    axes[2].plot(X_plot, y_plot_overfit, 'g-', linewidth=2, label=f'Ridge (α=0.001)')
    axes[2].set_title(f'Overfit Model\nTest MSE: {test_mse_overfit:.3f}')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/regression_examples.png', dpi=300, bbox_inches='tight')
    plt.show()


def compare_with_sklearn(X_train, y_train, X_test, y_test, alpha=1.0, degree=5):
    """
    Compare custom implementations with scikit-learn.
    """
    print("="*60)
    print("COMPARISON WITH SCIKIT-LEARN")
    print("="*60)
    
    # Create polynomial features
    X_train_poly = create_polynomial_features(X_train, degree)
    X_test_poly = create_polynomial_features(X_test, degree)
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_test_scaled = scaler.transform(X_test_poly)
    
    # Custom Ridge
    print(f"\n1. RIDGE REGRESSION (α = {alpha})")
    print("-" * 40)
    
    custom_ridge = RidgeRegression(alpha=alpha, method='closed_form')
    custom_ridge.fit(X_train_scaled, y_train)
    custom_ridge_pred = custom_ridge.predict(X_test_scaled)
    custom_ridge_mse = mean_squared_error(y_test, custom_ridge_pred)
    custom_ridge_r2 = r2_score(y_test, custom_ridge_pred)
    
    sklearn_ridge = SklearnRidge(alpha=alpha)
    sklearn_ridge.fit(X_train_scaled, y_train)
    sklearn_ridge_pred = sklearn_ridge.predict(X_test_scaled)
    sklearn_ridge_mse = mean_squared_error(y_test, sklearn_ridge_pred)
    sklearn_ridge_r2 = r2_score(y_test, sklearn_ridge_pred)
    
    print(f"Custom Ridge  - MSE: {custom_ridge_mse:.6f}, R²: {custom_ridge_r2:.6f}")
    print(f"Sklearn Ridge - MSE: {sklearn_ridge_mse:.6f}, R²: {sklearn_ridge_r2:.6f}")
    print(f"MSE Difference: {abs(custom_ridge_mse - sklearn_ridge_mse):.8f}")
    
    # Custom Lasso
    print(f"\n2. LASSO REGRESSION (α = {alpha})")
    print("-" * 40)
    
    custom_lasso = LassoRegression(alpha=alpha, max_iter=2000)
    custom_lasso.fit(X_train_scaled, y_train)
    custom_lasso_pred = custom_lasso.predict(X_test_scaled)
    custom_lasso_mse = mean_squared_error(y_test, custom_lasso_pred)
    custom_lasso_r2 = r2_score(y_test, custom_lasso_pred)
    
    sklearn_lasso = SklearnLasso(alpha=alpha, max_iter=2000)
    sklearn_lasso.fit(X_train_scaled, y_train)
    sklearn_lasso_pred = sklearn_lasso.predict(X_test_scaled)
    sklearn_lasso_mse = mean_squared_error(y_test, sklearn_lasso_pred)
    sklearn_lasso_r2 = r2_score(y_test, sklearn_lasso_pred)
    
    print(f"Custom Lasso  - MSE: {custom_lasso_mse:.6f}, R²: {custom_lasso_r2:.6f}")
    print(f"Sklearn Lasso - MSE: {sklearn_lasso_mse:.6f}, R²: {sklearn_lasso_r2:.6f}")
    print(f"MSE Difference: {abs(custom_lasso_mse - sklearn_lasso_mse):.8f}")
    
    # Feature selection comparison
    custom_n_features = custom_lasso.get_params()['n_nonzero']
    sklearn_n_features = np.sum(np.abs(sklearn_lasso.coef_) > 1e-8)
    
    print(f"\nFeature Selection:")
    print(f"Custom Lasso: {custom_n_features} non-zero features")
    print(f"Sklearn Lasso: {sklearn_n_features} non-zero features")
    
    return {
        'custom_ridge_mse': custom_ridge_mse,
        'sklearn_ridge_mse': sklearn_ridge_mse,
        'custom_lasso_mse': custom_lasso_mse,
        'sklearn_lasso_mse': sklearn_lasso_mse
    }


def main():
    """
    Main function to run the Ridge and Lasso regression experiments.
    """
    print("="*60)
    print("RIDGE AND LASSO REGRESSION FROM SCRATCH")
    print("="*60)
    
    # Generate synthetic data
    print("\n1. Generating synthetic polynomial data...")
    X, y = generate_polynomial_data(n_samples=200, noise_level=0.2, degree=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Data range: X ∈ [{X.min():.2f}, {X.max():.2f}], y ∈ [{y.min():.2f}, {y.max():.2f}]")
    
    # Test different regularization strengths
    print("\n2. Testing different regularization strengths...")
    alphas = np.logspace(-4, 2, 20)  # From 0.0001 to 100
    plot_regularization_comparison(X_train, y_train, X_test, y_test, alphas, max_degree=8)
    
    # Test bias-variance tradeoff
    print("\n3. Analyzing bias-variance tradeoff...")
    degrees = range(1, 11)
    plot_bias_variance_tradeoff(X_train, y_train, X_test, y_test, degrees, alpha=0.1)
    
    # Show regression examples
    print("\n4. Creating regression examples...")
    plot_regression_examples(X_train, y_train, X_test, y_test)
    
    # Compare with scikit-learn
    print("\n5. Comparing with scikit-learn...")
    comparison_results = compare_with_sklearn(X_train, y_train, X_test, y_test, alpha=1.0, degree=6)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Check the 'plots/' directory for visualizations:")
    print("- regularization_comparison.png")
    print("- bias_variance_tradeoff.png") 
    print("- regression_examples.png")


if __name__ == "__main__":
    main() 