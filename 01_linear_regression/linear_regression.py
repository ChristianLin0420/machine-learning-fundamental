"""
Linear Regression Implementation from Scratch
============================================

Implements multivariate linear regression using:
1. Batch Gradient Descent
2. Normal Equation (closed-form solution)
3. Evaluation metrics (MSE, R²)
4. Visualization capabilities
5. Comparison with scikit-learn
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

class LinearRegressionScratch:
    """
    Linear Regression implementation from scratch
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def fit_gradient_descent(self, X, y):
        """
        Fit the model using batch gradient descent
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
        """
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0
        self.cost_history = []
        
        # Gradient descent
        for i in range(self.max_iterations):
            # Forward pass
            y_pred = self._predict(X)
            
            # Compute cost (MSE)
            cost = self._compute_cost(y, y_pred)
            self.cost_history.append(cost)
            
            # Compute gradients
            dw, db = self._compute_gradients(X, y, y_pred)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Check convergence
            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tolerance:
                print(f"Converged after {i+1} iterations")
                break
                
        return self
    
    def fit_normal_equation(self, X, y):
        """
        Fit the model using normal equation (closed-form solution)
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
        """
        # Add bias term to X
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # Normal equation: θ = (X^T X)^(-1) X^T y
        try:
            theta = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if matrix is singular
            theta = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
            
        self.bias = theta[0]
        self.weights = theta[1:]
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the trained model
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            y_pred: Predictions (n_samples,)
        """
        return self._predict(X)
    
    def _predict(self, X):
        """Internal prediction function"""
        return X @ self.weights + self.bias
    
    def _compute_cost(self, y_true, y_pred):
        """Compute Mean Squared Error cost"""
        return np.mean((y_true - y_pred) ** 2)
    
    def _compute_gradients(self, X, y_true, y_pred):
        """Compute gradients for weights and bias"""
        n_samples = X.shape[0]
        error = y_pred - y_true
        
        dw = (1/n_samples) * X.T @ error
        db = (1/n_samples) * np.sum(error)
        
        return dw, db
    
    def evaluate(self, X, y):
        """
        Evaluate the model performance
        
        Args:
            X: Feature matrix
            y: True target values
            
        Returns:
            dict: Dictionary containing MSE and R² scores
        """
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        return {
            'mse': mse,
            'r2': r2,
            'rmse': np.sqrt(mse)
        }


class RidgeRegressionScratch:
    """
    Ridge Regression (L2 regularization) implementation from scratch
    """
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        """
        Fit Ridge regression using normal equation with regularization
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
        """
        # Add bias term to X
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # Ridge regression: θ = (X^T X + αI)^(-1) X^T y
        n_features = X_with_bias.shape[1]
        identity = np.eye(n_features)
        identity[0, 0] = 0  # Don't regularize bias term
        
        try:
            theta = np.linalg.inv(X_with_bias.T @ X_with_bias + self.alpha * identity) @ X_with_bias.T @ y
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(X_with_bias.T @ X_with_bias + self.alpha * identity) @ X_with_bias.T @ y
            
        self.bias = theta[0]
        self.weights = theta[1:]
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        return X @ self.weights + self.bias


def create_polynomial_features(X, degree=2):
    """
    Create polynomial features up to specified degree
    
    Args:
        X: Input features (n_samples, n_features)
        degree: Maximum polynomial degree
        
    Returns:
        X_poly: Polynomial features
    """
    if X.shape[1] != 1:
        raise ValueError("Polynomial features only supported for single feature")
    
    X_poly = X.copy()
    for d in range(2, degree + 1):
        X_poly = np.column_stack([X_poly, X ** d])
    
    return X_poly


def plot_regression_line_2d(X, y, model, feature_names=None, save_path=None, X_full=None):
    """
    Plot regression line for 2D data (single feature)
    
    Args:
        X: Feature matrix (should be 1D for visualization)
        y: Target values
        model: Trained model
        feature_names: Names of features
        save_path: Path to save the plot
        X_full: Full feature matrix (needed if model was trained on multiple features)
    """
    if X.shape[1] != 1:
        print("Using first feature for 2D visualization")
        X_plot = X[:, 0:1]
        feature_name = feature_names[0] if feature_names else "Feature 1"
    else:
        X_plot = X
        feature_name = feature_names[0] if feature_names else "Feature"
    
    plt.figure(figsize=(10, 6))
    
    # Scatter plot of data points
    plt.scatter(X_plot, y, alpha=0.6, color='blue', label='Data points')
    
    # Plot regression line
    X_line = np.linspace(X_plot.min(), X_plot.max(), 100).reshape(-1, 1)
    
    if hasattr(model, 'predict'):
        # Check if model expects more features than we're providing
        if hasattr(model, 'weights') and len(model.weights) > 1 and X_full is not None:
            # For multivariate model, use mean values for other features
            mean_features = np.mean(X_full[:, 1:], axis=0)
            # Replicate mean features for each point in the line
            mean_features_repeated = np.tile(mean_features, (len(X_line), 1))
            X_line_multi = np.column_stack([X_line.flatten(), mean_features_repeated])
            y_line = model.predict(X_line_multi)
        elif hasattr(model, 'weights') and len(model.weights) > 1:
            # If no X_full provided, train a simple model for this feature only
            from sklearn.linear_model import LinearRegression
            simple_model = LinearRegression()
            simple_model.fit(X_plot, y)
            y_line = simple_model.predict(X_line)
        else:
            # Model was trained on single feature
            y_line = model.predict(X_line)
    else:
        # sklearn model
        if hasattr(model, 'coef_') and len(model.coef_) > 1 and X_full is not None:
            mean_features = np.mean(X_full[:, 1:], axis=0)
            # Replicate mean features for each point in the line
            mean_features_repeated = np.tile(mean_features, (len(X_line), 1))
            X_line_multi = np.column_stack([X_line.flatten(), mean_features_repeated])
            y_line = model.predict(X_line_multi)
        else:
            y_line = X_line.flatten() * model.coef_[0] + model.intercept_
    
    plt.plot(X_line, y_line, color='red', linewidth=2, label='Regression line')
    
    plt.xlabel(feature_name)
    plt.ylabel('Target Value')
    plt.title('Linear Regression Fit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_residuals(y_true, y_pred, save_path=None):
    """
    Plot residual analysis
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        save_path: Path to save the plot
    """
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Residuals vs Predicted values
    ax1.scatter(y_pred, residuals, alpha=0.6)
    ax1.axhline(y=0, color='red', linestyle='--')
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Predicted Values')
    ax1.grid(True, alpha=0.3)
    
    # Histogram of residuals
    ax2.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Residuals')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_learning_curve(cost_history, save_path=None):
    """
    Plot learning curve showing cost vs iterations
    
    Args:
        cost_history: List of cost values during training
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(cost_history, linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('Cost (MSE)')
    plt.title('Learning Curve - Cost vs Iterations')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def compare_with_sklearn(X_train, X_test, y_train, y_test):
    """
    Compare custom implementation with sklearn LinearRegression
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test targets
        
    Returns:
        dict: Comparison results
    """
    # Our implementation
    print("=" * 50)
    print("COMPARISON WITH SCIKIT-LEARN")
    print("=" * 50)
    
    # Gradient Descent
    print("\n1. Custom Implementation (Gradient Descent)")
    model_gd = LinearRegressionScratch(learning_rate=0.01, max_iterations=1000)
    model_gd.fit_gradient_descent(X_train, y_train)
    
    gd_train_metrics = model_gd.evaluate(X_train, y_train)
    gd_test_metrics = model_gd.evaluate(X_test, y_test)
    
    print(f"Training - MSE: {gd_train_metrics['mse']:.4f}, R²: {gd_train_metrics['r2']:.4f}")
    print(f"Testing  - MSE: {gd_test_metrics['mse']:.4f}, R²: {gd_test_metrics['r2']:.4f}")
    
    # Normal Equation
    print("\n2. Custom Implementation (Normal Equation)")
    model_ne = LinearRegressionScratch()
    model_ne.fit_normal_equation(X_train, y_train)
    
    ne_train_metrics = model_ne.evaluate(X_train, y_train)
    ne_test_metrics = model_ne.evaluate(X_test, y_test)
    
    print(f"Training - MSE: {ne_train_metrics['mse']:.4f}, R²: {ne_train_metrics['r2']:.4f}")
    print(f"Testing  - MSE: {ne_test_metrics['mse']:.4f}, R²: {ne_test_metrics['r2']:.4f}")
    
    # Scikit-learn
    print("\n3. Scikit-learn Implementation")
    sklearn_model = LinearRegression()
    sklearn_model.fit(X_train, y_train)
    
    sklearn_train_pred = sklearn_model.predict(X_train)
    sklearn_test_pred = sklearn_model.predict(X_test)
    
    sklearn_train_mse = mean_squared_error(y_train, sklearn_train_pred)
    sklearn_train_r2 = r2_score(y_train, sklearn_train_pred)
    sklearn_test_mse = mean_squared_error(y_test, sklearn_test_pred)
    sklearn_test_r2 = r2_score(y_test, sklearn_test_pred)
    
    print(f"Training - MSE: {sklearn_train_mse:.4f}, R²: {sklearn_train_r2:.4f}")
    print(f"Testing  - MSE: {sklearn_test_mse:.4f}, R²: {sklearn_test_r2:.4f}")
    
    # Comparison summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Gradient Descent vs sklearn MSE difference: {abs(gd_test_metrics['mse'] - sklearn_test_mse):.6f}")
    print(f"Normal Equation vs sklearn MSE difference:  {abs(ne_test_metrics['mse'] - sklearn_test_mse):.6f}")
    
    return {
        'gradient_descent': {'train': gd_train_metrics, 'test': gd_test_metrics, 'model': model_gd},
        'normal_equation': {'train': ne_train_metrics, 'test': ne_test_metrics, 'model': model_ne},
        'sklearn': {
            'train': {'mse': sklearn_train_mse, 'r2': sklearn_train_r2},
            'test': {'mse': sklearn_test_mse, 'r2': sklearn_test_r2},
            'model': sklearn_model
        }
    }


def main():
    """
    Main function to demonstrate linear regression implementation
    """
    print("=" * 60)
    print("LINEAR REGRESSION FROM SCRATCH - DAY 1 CHALLENGE")
    print("=" * 60)
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Load California housing dataset
    print("\n1. Loading California Housing Dataset...")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    feature_names = housing.feature_names
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Target: Median house value in hundreds of thousands of dollars")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features for better convergence
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTraining set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    
    # Train models and compare
    results = compare_with_sklearn(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Visualizations
    print("\n4. Creating Visualizations...")
    
    # Use median income (feature 0) for 2D visualization
    X_train_2d = X_train_scaled[:, 0:1]  # Median income
    X_test_2d = X_test_scaled[:, 0:1]
    
    # Plot regression line
    print("   - Regression line (using median income)")
    plot_regression_line_2d(
        X_train_2d, y_train, 
        results['gradient_descent']['model'],
        ['Median Income (scaled)'],
        'plots/regression_line.png',
        X_train_scaled
    )
    
    # Plot residuals
    print("   - Residual analysis")
    y_pred = results['gradient_descent']['model'].predict(X_test_scaled)
    plot_residuals(y_test, y_pred, 'plots/residuals.png')
    
    # Plot learning curve
    print("   - Learning curve")
    plot_learning_curve(
        results['gradient_descent']['model'].cost_history,
        'plots/learning_curve.png'
    )
    
    # Optional: Polynomial regression with regularization
    print("\n5. Optional: Polynomial Regression with Ridge Regularization")
    print("   (Using median income feature)")
    
    # Create polynomial features
    X_train_poly = create_polynomial_features(X_train_2d, degree=3)
    X_test_poly = create_polynomial_features(X_test_2d, degree=3)
    
    # Ridge regression
    ridge_model = RidgeRegressionScratch(alpha=1.0)
    ridge_model.fit(X_train_poly, y_train)
    
    ridge_train_pred = ridge_model.predict(X_train_poly)
    ridge_test_pred = ridge_model.predict(X_test_poly)
    
    ridge_train_mse = mean_squared_error(y_train, ridge_train_pred)
    ridge_test_mse = mean_squared_error(y_test, ridge_test_pred)
    ridge_train_r2 = r2_score(y_train, ridge_train_pred)
    ridge_test_r2 = r2_score(y_test, ridge_test_pred)
    
    print(f"   Polynomial Ridge - Training MSE: {ridge_train_mse:.4f}, R²: {ridge_train_r2:.4f}")
    print(f"   Polynomial Ridge - Testing MSE:  {ridge_test_mse:.4f}, R²: {ridge_test_r2:.4f}")
    
    print("\n" + "=" * 60)
    print("CHALLENGE COMPLETED! 🎉")
    print("Check the 'plots/' directory for visualizations")
    print("=" * 60)


if __name__ == "__main__":
    main() 