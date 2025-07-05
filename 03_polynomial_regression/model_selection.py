"""
Model Selection and Cross-Validation Implementation from Scratch

This module implements:
- K-fold cross-validation from scratch
- Grid search for hyperparameter tuning
- Bias-variance decomposition analysis
- Model selection strategies for Ridge and Lasso regression
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error
from ridge_lasso import RidgeRegression, LassoRegression, generate_polynomial_data, create_polynomial_features
import warnings
warnings.filterwarnings('ignore')

class CrossValidator:
    """
    K-Fold Cross-Validation implementation from scratch.
    """
    
    def __init__(self, n_splits=5, shuffle=True, random_state=42):
        """
        Initialize cross-validator.
        
        Args:
            n_splits (int): Number of folds
            shuffle (bool): Whether to shuffle data before splitting
            random_state (int): Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X, y=None):
        """
        Generate indices for train/validation splits.
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Target values (optional)
        
        Yields:
            tuple: (train_indices, validation_indices) for each fold
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        
        if self.shuffle:
            np.random.seed(self.random_state)
            indices = np.random.permutation(indices)
        
        fold_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            start = i * fold_size
            if i == self.n_splits - 1:
                # Last fold gets remaining samples
                end = n_samples
            else:
                end = (i + 1) * fold_size
            
            val_indices = indices[start:end]
            train_indices = np.concatenate([indices[:start], indices[end:]])
            
            yield train_indices, val_indices
    
    def cross_val_score(self, model, X, y, scoring='mse'):
        """
        Perform cross-validation and return scores.
        
        Args:
            model: Model instance with fit and predict methods
            X (np.array): Feature matrix
            y (np.array): Target values
            scoring (str): Scoring metric ('mse' or 'r2')
        
        Returns:
            np.array: Cross-validation scores for each fold
        """
        scores = []
        
        for train_idx, val_idx in self.split(X, y):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Create a fresh model instance with only initialization parameters
            if hasattr(model, 'alpha'):
                if isinstance(model, RidgeRegression):
                    model_copy = RidgeRegression(
                        alpha=model.alpha,
                        method=getattr(model, 'method', 'closed_form'),
                        learning_rate=getattr(model, 'learning_rate', 0.01),
                        max_iter=getattr(model, 'max_iter', 1000),
                        tolerance=getattr(model, 'tolerance', 1e-6)
                    )
                elif isinstance(model, LassoRegression):
                    model_copy = LassoRegression(
                        alpha=model.alpha,
                        max_iter=getattr(model, 'max_iter', 1000),
                        tolerance=getattr(model, 'tolerance', 1e-4)
                    )
                else:
                    # Fallback for other models
                    model_copy = type(model)(alpha=model.alpha)
            else:
                # Generic fallback
                model_copy = type(model)()
            
            model_copy.fit(X_train_fold, y_train_fold)
            
            y_pred = model_copy.predict(X_val_fold)
            
            if scoring == 'mse':
                score = mean_squared_error(y_val_fold, y_pred)
            elif scoring == 'r2':
                ss_res = np.sum((y_val_fold - y_pred) ** 2)
                ss_tot = np.sum((y_val_fold - np.mean(y_val_fold)) ** 2)
                score = 1 - (ss_res / ss_tot)
            else:
                raise ValueError("Scoring must be 'mse' or 'r2'")
            
            scores.append(score)
        
        return np.array(scores)


class GridSearchCV:
    """
    Grid Search with Cross-Validation implementation from scratch.
    """
    
    def __init__(self, model, param_grid, cv=5, scoring='mse', n_jobs=1):
        """
        Initialize grid search.
        
        Args:
            model: Model class (RidgeRegression or LassoRegression)
            param_grid (dict): Dictionary of parameter ranges
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric
            n_jobs (int): Number of parallel jobs (not implemented)
        """
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        
        # Results
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None
        self.cv_results_ = {}
    
    def _generate_param_combinations(self):
        """Generate all parameter combinations from param_grid."""
        import itertools
        
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        
        combinations = []
        for values in itertools.product(*param_values):
            combinations.append(dict(zip(param_names, values)))
        
        return combinations
    
    def fit(self, X, y):
        """
        Fit grid search with cross-validation.
        
        Args:
            X (np.array): Feature matrix
            y (np.array): Target values
        """
        cv = CrossValidator(n_splits=self.cv)
        param_combinations = self._generate_param_combinations()
        
        results = {
            'params': [],
            'mean_test_score': [],
            'std_test_score': [],
            'rank_test_score': []
        }
        
        print(f"Fitting {len(param_combinations)} parameter combinations...")
        
        for i, params in enumerate(param_combinations):
            # Create model with current parameters
            model_instance = self.model(**params)
            
            # Perform cross-validation
            scores = cv.cross_val_score(model_instance, X, y, scoring=self.scoring)
            
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            results['params'].append(params)
            results['mean_test_score'].append(mean_score)
            results['std_test_score'].append(std_score)
            
            print(f"  {i+1:3d}/{len(param_combinations)}: {params} -> {mean_score:.6f} (±{std_score:.6f})")
        
        # Find best parameters
        if self.scoring == 'mse':
            best_idx = np.argmin(results['mean_test_score'])
        else:  # r2
            best_idx = np.argmax(results['mean_test_score'])
        
        self.best_params_ = results['params'][best_idx]
        self.best_score_ = results['mean_test_score'][best_idx]
        
        # Create ranking
        if self.scoring == 'mse':
            rank_indices = np.argsort(results['mean_test_score'])
        else:  # r2
            rank_indices = np.argsort(-np.array(results['mean_test_score']))
        
        ranks = np.empty_like(rank_indices)
        ranks[rank_indices] = np.arange(1, len(rank_indices) + 1)
        results['rank_test_score'] = ranks.tolist()
        
        # Fit best estimator
        self.best_estimator_ = self.model(**self.best_params_)
        self.best_estimator_.fit(X, y)
        
        self.cv_results_ = results
        
        return self
    
    def predict(self, X):
        """Make predictions using best estimator."""
        if self.best_estimator_ is None:
            raise ValueError("GridSearchCV must be fitted before making predictions")
        return self.best_estimator_.predict(X)


def bias_variance_decomposition(X, y, model_class, model_params, n_bootstrap=100, test_size=0.3, random_state=42):
    """
    Perform bias-variance decomposition analysis.
    
    Args:
        X (np.array): Feature matrix
        y (np.array): Target values
        model_class: Model class to analyze
        model_params (dict): Model parameters
        n_bootstrap (int): Number of bootstrap samples
        test_size (float): Proportion of test data
        random_state (int): Random seed
    
    Returns:
        dict: Dictionary containing bias, variance, and noise estimates
    """
    np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test
    
    # Split data
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    # Store predictions from all bootstrap samples
    predictions = np.zeros((n_bootstrap, len(X_test)))
    
    for i in range(n_bootstrap):
        # Bootstrap sample from training data
        bootstrap_indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_bootstrap = X_train[bootstrap_indices]
        y_bootstrap = y_train[bootstrap_indices]
        
        # Train model on bootstrap sample
        model = model_class(**model_params)
        model.fit(X_bootstrap, y_bootstrap)
        
        # Make predictions on test set
        predictions[i] = model.predict(X_test)
    
    # Calculate bias, variance, and noise
    mean_prediction = np.mean(predictions, axis=0)
    
    # Bias² = (E[f̂(x)] - f(x))²
    bias_squared = np.mean((mean_prediction - y_test) ** 2)
    
    # Variance = E[(f̂(x) - E[f̂(x)])²]
    variance = np.mean(np.var(predictions, axis=0))
    
    # Noise = irreducible error (estimate from residuals)
    noise = np.var(y_test - mean_prediction)
    
    # Total error should approximately equal bias² + variance + noise
    total_error = bias_squared + variance + noise
    
    return {
        'bias_squared': bias_squared,
        'variance': variance,
        'noise': noise,
        'total_error': total_error,
        'predictions': predictions,
        'mean_prediction': mean_prediction
    }


def plot_cross_validation_results(X, y, alphas, model_class, degree=5):
    """
    Plot cross-validation results for different regularization strengths.
    
    Args:
        X, y: Data
        alphas: List of regularization parameters
        model_class: RidgeRegression or LassoRegression
        degree: Polynomial degree for features
    """
    # Create polynomial features
    X_poly = create_polynomial_features(X, degree)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)
    
    cv = CrossValidator(n_splits=5)
    
    mean_scores = []
    std_scores = []
    
    print(f"Running cross-validation for {model_class.__name__}...")
    
    for alpha in alphas:
        model = model_class(alpha=alpha, max_iter=2000)
        scores = cv.cross_val_score(model, X_scaled, y, scoring='mse')
        
        mean_scores.append(np.mean(scores))
        std_scores.append(np.std(scores))
        
        print(f"  α = {alpha:.4f}: MSE = {np.mean(scores):.6f} (±{np.std(scores):.6f})")
    
    mean_scores = np.array(mean_scores)
    std_scores = np.array(std_scores)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.semilogx(alphas, mean_scores, 'b-o', markersize=6, linewidth=2)
    plt.fill_between(alphas, mean_scores - std_scores, mean_scores + std_scores, alpha=0.3)
    plt.xlabel('Regularization Strength (α)')
    plt.ylabel('Cross-Validation MSE')
    plt.title(f'{model_class.__name__}: Cross-Validation Results')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.semilogx(alphas, std_scores, 'r-s', markersize=6, linewidth=2)
    plt.xlabel('Regularization Strength (α)')
    plt.ylabel('Standard Deviation of CV Scores')
    plt.title('Cross-Validation Score Stability')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'plots/{model_class.__name__.lower()}_cv_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find optimal alpha
    optimal_idx = np.argmin(mean_scores)
    optimal_alpha = alphas[optimal_idx]
    optimal_score = mean_scores[optimal_idx]
    
    print(f"\nOptimal α = {optimal_alpha:.4f} with CV MSE = {optimal_score:.6f}")
    
    return optimal_alpha, optimal_score


def plot_bias_variance_analysis(X, y, model_class, alphas):
    """
    Plot bias-variance decomposition for different regularization strengths.
    
    Args:
        X, y: Data
        model_class: Model class to analyze
        alphas: List of regularization parameters
    """
    bias_squared_list = []
    variance_list = []
    noise_list = []
    total_error_list = []
    
    print(f"Running bias-variance analysis for {model_class.__name__}...")
    
    for alpha in alphas:
        print(f"  Analyzing α = {alpha:.4f}...")
        
        # Create polynomial features
        X_poly = create_polynomial_features(X, degree=6)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_poly)
        
        # Perform bias-variance decomposition
        results = bias_variance_decomposition(
            X_scaled, y, model_class, {'alpha': alpha}, 
            n_bootstrap=50, random_state=42
        )
        
        bias_squared_list.append(results['bias_squared'])
        variance_list.append(results['variance'])
        noise_list.append(results['noise'])
        total_error_list.append(results['total_error'])
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Bias-Variance decomposition
    plt.subplot(2, 2, 1)
    plt.semilogx(alphas, bias_squared_list, 'r-o', label='Bias²', markersize=6)
    plt.semilogx(alphas, variance_list, 'b-s', label='Variance', markersize=6)
    plt.semilogx(alphas, noise_list, 'g-^', label='Noise', markersize=6)
    plt.xlabel('Regularization Strength (α)')
    plt.ylabel('Error Component')
    plt.title(f'{model_class.__name__}: Bias-Variance Decomposition')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Total error
    plt.subplot(2, 2, 2)
    plt.semilogx(alphas, total_error_list, 'k-o', markersize=6, linewidth=2)
    plt.xlabel('Regularization Strength (α)')
    plt.ylabel('Total Error')
    plt.title('Total Error vs Regularization')
    plt.grid(True, alpha=0.3)
    
    # Bias vs Variance tradeoff
    plt.subplot(2, 2, 3)
    plt.plot(bias_squared_list, variance_list, 'mo-', markersize=8, linewidth=2)
    for i, alpha in enumerate(alphas[::3]):  # Label every 3rd point to avoid clutter
        idx = i * 3
        if idx < len(alphas):
            plt.annotate(f'α={alpha:.3f}', 
                        (bias_squared_list[idx], variance_list[idx]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    plt.xlabel('Bias²')
    plt.ylabel('Variance')
    plt.title('Bias-Variance Tradeoff')
    plt.grid(True, alpha=0.3)
    
    # Stacked area plot
    plt.subplot(2, 2, 4)
    plt.fill_between(alphas, 0, bias_squared_list, alpha=0.7, label='Bias²')
    plt.fill_between(alphas, bias_squared_list, 
                    np.array(bias_squared_list) + np.array(variance_list), 
                    alpha=0.7, label='Variance')
    plt.fill_between(alphas, 
                    np.array(bias_squared_list) + np.array(variance_list),
                    np.array(total_error_list), 
                    alpha=0.7, label='Noise')
    plt.xscale('log')
    plt.xlabel('Regularization Strength (α)')
    plt.ylabel('Error Component')
    plt.title('Error Decomposition (Stacked)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'plots/{model_class.__name__.lower()}_bias_variance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find optimal alpha (minimum total error)
    optimal_idx = np.argmin(total_error_list)
    optimal_alpha = alphas[optimal_idx]
    
    print(f"\nOptimal α = {optimal_alpha:.4f}")
    print(f"  Bias² = {bias_squared_list[optimal_idx]:.6f}")
    print(f"  Variance = {variance_list[optimal_idx]:.6f}")
    print(f"  Noise = {noise_list[optimal_idx]:.6f}")
    print(f"  Total Error = {total_error_list[optimal_idx]:.6f}")
    
    return optimal_alpha


def perform_grid_search_analysis(X, y, degree=6):
    """
    Perform comprehensive grid search analysis for both Ridge and Lasso.
    
    Args:
        X, y: Data
        degree: Polynomial degree for features
    """
    print("="*60)
    print("GRID SEARCH ANALYSIS")
    print("="*60)
    
    # Create polynomial features
    X_poly = create_polynomial_features(X, degree)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)
    
    # Define parameter grid
    alphas = np.logspace(-4, 2, 25)  # From 0.0001 to 100
    
    # Ridge Grid Search
    print("\n1. RIDGE REGRESSION GRID SEARCH")
    print("-" * 40)
    
    ridge_param_grid = {'alpha': alphas}
    ridge_grid = GridSearchCV(RidgeRegression, ridge_param_grid, cv=5, scoring='mse')
    ridge_grid.fit(X_scaled, y)
    
    print(f"Best Ridge parameters: {ridge_grid.best_params_}")
    print(f"Best Ridge CV score: {ridge_grid.best_score_:.6f}")
    
    # Lasso Grid Search
    print("\n2. LASSO REGRESSION GRID SEARCH")
    print("-" * 40)
    
    lasso_param_grid = {'alpha': alphas}
    lasso_grid = GridSearchCV(LassoRegression, lasso_param_grid, cv=5, scoring='mse')
    lasso_grid.fit(X_scaled, y)
    
    print(f"Best Lasso parameters: {lasso_grid.best_params_}")
    print(f"Best Lasso CV score: {lasso_grid.best_score_:.6f}")
    
    return ridge_grid, lasso_grid


def main():
    """
    Main function to run model selection and cross-validation experiments.
    """
    print("="*60)
    print("MODEL SELECTION AND CROSS-VALIDATION FROM SCRATCH")
    print("="*60)
    
    # Generate synthetic data
    print("\n1. Generating synthetic polynomial data...")
    X, y = generate_polynomial_data(n_samples=300, noise_level=0.3, degree=3, random_state=42)
    
    print(f"Data samples: {len(X)}")
    print(f"Data range: X ∈ [{X.min():.2f}, {X.max():.2f}], y ∈ [{y.min():.2f}, {y.max():.2f}]")
    
    # Cross-validation analysis
    print("\n2. Cross-validation analysis...")
    alphas = np.logspace(-3, 1, 15)
    
    ridge_optimal_alpha, _ = plot_cross_validation_results(X, y, alphas, RidgeRegression, degree=6)
    lasso_optimal_alpha, _ = plot_cross_validation_results(X, y, alphas, LassoRegression, degree=6)
    
    # Bias-variance analysis
    print("\n3. Bias-variance decomposition analysis...")
    alphas_bv = np.logspace(-2, 1, 10)
    
    ridge_bv_optimal = plot_bias_variance_analysis(X, y, RidgeRegression, alphas_bv)
    lasso_bv_optimal = plot_bias_variance_analysis(X, y, LassoRegression, alphas_bv)
    
    # Grid search analysis
    print("\n4. Grid search with cross-validation...")
    ridge_grid, lasso_grid = perform_grid_search_analysis(X, y, degree=6)
    
    # Summary of results
    print("\n" + "="*60)
    print("SUMMARY OF OPTIMAL PARAMETERS")
    print("="*60)
    print(f"Ridge Regression:")
    print(f"  CV Analysis:     α = {ridge_optimal_alpha:.4f}")
    print(f"  Bias-Var Analysis: α = {ridge_bv_optimal:.4f}")
    print(f"  Grid Search:     α = {ridge_grid.best_params_['alpha']:.4f}")
    
    print(f"\nLasso Regression:")
    print(f"  CV Analysis:     α = {lasso_optimal_alpha:.4f}")
    print(f"  Bias-Var Analysis: α = {lasso_bv_optimal:.4f}")
    print(f"  Grid Search:     α = {lasso_grid.best_params_['alpha']:.4f}")
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Check the 'plots/' directory for visualizations:")
    print("- ridgeregression_cv_results.png")
    print("- lassoregression_cv_results.png")
    print("- ridgeregression_bias_variance.png")
    print("- lassoregression_bias_variance.png")


if __name__ == "__main__":
    main() 