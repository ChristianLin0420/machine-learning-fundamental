"""
Nested Cross-Validation Pipeline

This module implements nested cross-validation where:
- Inner loop: GridSearchCV for hyperparameter optimization
- Outer loop: Cross-validation evaluation of the selected model
- Provides unbiased estimates of model performance

Author: Machine Learning Fundamentals Course
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, StratifiedKFold
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


class NestedCVPipeline:
    """
    Nested Cross-Validation Pipeline
    
    Inner loop: GridSearchCV for hyperparameter tuning
    Outer loop: Cross-validation for unbiased performance estimation
    """
    
    def __init__(self, estimator, param_grid, inner_cv=5, outer_cv=5, 
                 scoring='accuracy', random_state=42):
        """
        Initialize nested CV pipeline
        
        Parameters:
        -----------
        estimator : sklearn estimator
            Base model to optimize
        param_grid : dict
            Parameter grid for GridSearchCV
        inner_cv : int or CV object
            Inner cross-validation strategy
        outer_cv : int or CV object
            Outer cross-validation strategy
        scoring : str
            Scoring metric
        random_state : int
            Random state for reproducibility
        """
        self.estimator = estimator
        self.param_grid = param_grid
        self.inner_cv = inner_cv
        self.outer_cv = outer_cv
        self.scoring = scoring
        self.random_state = random_state
        
        # Results storage
        self.outer_scores_ = []
        self.best_params_per_fold_ = []
        self.inner_scores_per_fold_ = []
        
    def fit(self, X, y):
        """
        Perform nested cross-validation
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        
        Returns:
        --------
        self : NestedCVPipeline
            Fitted pipeline
        """
        # Setup CV strategies
        if isinstance(self.outer_cv, int):
            if len(np.unique(y)) > 1:  # Classification
                outer_cv = StratifiedKFold(n_splits=self.outer_cv, 
                                         shuffle=True, 
                                         random_state=self.random_state)
            else:
                outer_cv = KFold(n_splits=self.outer_cv, 
                               shuffle=True, 
                               random_state=self.random_state)
        else:
            outer_cv = self.outer_cv
            
        if isinstance(self.inner_cv, int):
            if len(np.unique(y)) > 1:  # Classification
                inner_cv = StratifiedKFold(n_splits=self.inner_cv, 
                                         shuffle=True, 
                                         random_state=self.random_state)
            else:
                inner_cv = KFold(n_splits=self.inner_cv, 
                               shuffle=True, 
                               random_state=self.random_state)
        else:
            inner_cv = self.inner_cv
        
        # Perform nested CV
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
            print(f"Processing outer fold {fold_idx + 1}/{outer_cv.get_n_splits(X, y)}")
            
            # Split data for current fold
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]
            
            # Inner loop: GridSearchCV for hyperparameter optimization
            grid_search = GridSearchCV(
                estimator=self.estimator,
                param_grid=self.param_grid,
                cv=inner_cv,
                scoring=self.scoring,
                n_jobs=-1
            )
            
            # Fit grid search on training data
            grid_search.fit(X_train_fold, y_train_fold)
            
            # Store best parameters and inner CV score
            self.best_params_per_fold_.append(grid_search.best_params_)
            self.inner_scores_per_fold_.append(grid_search.best_score_)
            
            # Evaluate best model on test fold (outer loop)
            best_model = grid_search.best_estimator_
            y_pred = best_model.predict(X_test_fold)
            
            # Calculate outer score
            if self.scoring == 'accuracy':
                outer_score = accuracy_score(y_test_fold, y_pred)
            elif self.scoring == 'f1':
                outer_score = f1_score(y_test_fold, y_pred, average='weighted')
            elif self.scoring == 'roc_auc':
                if hasattr(best_model, 'predict_proba'):
                    y_prob = best_model.predict_proba(X_test_fold)
                    if y_prob.shape[1] == 2:  # Binary classification
                        outer_score = roc_auc_score(y_test_fold, y_prob[:, 1])
                    else:  # Multi-class
                        outer_score = roc_auc_score(y_test_fold, y_prob, 
                                                  multi_class='ovr', average='weighted')
                else:
                    outer_score = accuracy_score(y_test_fold, y_pred)
            else:
                outer_score = accuracy_score(y_test_fold, y_pred)
            
            self.outer_scores_.append(outer_score)
            
        return self
    
    def get_performance_summary(self):
        """
        Get summary of nested CV performance
        
        Returns:
        --------
        dict : Performance summary
        """
        outer_scores = np.array(self.outer_scores_)
        inner_scores = np.array(self.inner_scores_per_fold_)
        
        summary = {
            'outer_cv_mean': np.mean(outer_scores),
            'outer_cv_std': np.std(outer_scores),
            'outer_cv_scores': outer_scores,
            'inner_cv_mean': np.mean(inner_scores),
            'inner_cv_std': np.std(inner_scores),
            'inner_cv_scores': inner_scores,
            'best_params_per_fold': self.best_params_per_fold_,
            'confidence_interval_95': (
                np.mean(outer_scores) - 1.96 * np.std(outer_scores) / np.sqrt(len(outer_scores)),
                np.mean(outer_scores) + 1.96 * np.std(outer_scores) / np.sqrt(len(outer_scores))
            )
        }
        
        return summary
    
    def plot_results(self, title="Nested Cross-Validation Results", save_path=None):
        """
        Plot nested CV results showing variance of outer scores
        
        Parameters:
        -----------
        title : str
            Plot title
        save_path : str, optional
            Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        outer_scores = np.array(self.outer_scores_)
        inner_scores = np.array(self.inner_scores_per_fold_)
        
        # 1. Outer CV scores variance
        axes[0, 0].boxplot([outer_scores], labels=['Outer CV'])
        axes[0, 0].scatter([1] * len(outer_scores), outer_scores, 
                          alpha=0.7, color='red', s=50)
        axes[0, 0].set_ylabel(f'{self.scoring.capitalize()} Score')
        axes[0, 0].set_title('Outer CV Score Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add mean and std annotations
        mean_score = np.mean(outer_scores)
        std_score = np.std(outer_scores)
        axes[0, 0].axhline(y=mean_score, color='green', linestyle='--', 
                          label=f'Mean: {mean_score:.3f}')
        axes[0, 0].axhline(y=mean_score + std_score, color='orange', 
                          linestyle=':', alpha=0.7, label=f'±1σ: {std_score:.3f}')
        axes[0, 0].axhline(y=mean_score - std_score, color='orange', 
                          linestyle=':', alpha=0.7)
        axes[0, 0].legend()
        
        # 2. Fold-by-fold comparison
        folds = range(1, len(outer_scores) + 1)
        axes[0, 1].plot(folds, outer_scores, 'o-', label='Outer CV', 
                       color='blue', linewidth=2, markersize=8)
        axes[0, 1].plot(folds, inner_scores, 's-', label='Inner CV (Best)', 
                       color='red', linewidth=2, markersize=6)
        axes[0, 1].set_xlabel('Fold')
        axes[0, 1].set_ylabel(f'{self.scoring.capitalize()} Score')
        axes[0, 1].set_title('Score per Fold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Inner vs Outer score comparison
        axes[1, 0].scatter(inner_scores, outer_scores, alpha=0.7, s=100, color='purple')
        axes[1, 0].plot([min(inner_scores.min(), outer_scores.min()), 
                        max(inner_scores.max(), outer_scores.max())],
                       [min(inner_scores.min(), outer_scores.min()), 
                        max(inner_scores.max(), outer_scores.max())],
                       'k--', alpha=0.5, label='Perfect correlation')
        axes[1, 0].set_xlabel('Inner CV Score (Best)')
        axes[1, 0].set_ylabel('Outer CV Score')
        axes[1, 0].set_title('Inner vs Outer CV Scores')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Calculate correlation
        correlation = np.corrcoef(inner_scores, outer_scores)[0, 1]
        axes[1, 0].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                       transform=axes[1, 0].transAxes, fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 4. Performance statistics
        axes[1, 1].axis('off')
        summary = self.get_performance_summary()
        
        stats_text = f"""
        Nested Cross-Validation Summary
        ================================
        
        Outer CV Performance:
        • Mean ± Std: {summary['outer_cv_mean']:.4f} ± {summary['outer_cv_std']:.4f}
        • 95% CI: [{summary['confidence_interval_95'][0]:.4f}, {summary['confidence_interval_95'][1]:.4f}]
        • Min: {np.min(outer_scores):.4f}
        • Max: {np.max(outer_scores):.4f}
        
        Inner CV Performance:
        • Mean ± Std: {summary['inner_cv_mean']:.4f} ± {summary['inner_cv_std']:.4f}
        
        Bias Estimation:
        • Optimism: {summary['inner_cv_mean'] - summary['outer_cv_mean']:.4f}
        • Relative Bias: {((summary['inner_cv_mean'] - summary['outer_cv_mean']) / summary['outer_cv_mean'] * 100):.2f}%
        """
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()


def compare_nested_vs_simple_cv():
    """
    Compare nested CV with simple CV to demonstrate bias
    """
    print("Comparing Nested CV vs Simple CV")
    print("=" * 50)
    
    # Load dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define model and parameter grid
    model = SVC(random_state=42)
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'kernel': ['rbf', 'linear']
    }
    
    # Simple CV (biased)
    print("\n1. Simple Cross-Validation (Biased):")
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    simple_scores = cross_val_score(grid_search, X_scaled, y, cv=5, scoring='accuracy')
    print(f"   Mean accuracy: {np.mean(simple_scores):.4f} ± {np.std(simple_scores):.4f}")
    
    # Nested CV (unbiased)
    print("\n2. Nested Cross-Validation (Unbiased):")
    nested_cv = NestedCVPipeline(model, param_grid, inner_cv=5, outer_cv=5, scoring='accuracy')
    nested_cv.fit(X_scaled, y)
    
    summary = nested_cv.get_performance_summary()
    print(f"   Mean accuracy: {summary['outer_cv_mean']:.4f} ± {summary['outer_cv_std']:.4f}")
    
    # Show bias
    bias = np.mean(simple_scores) - summary['outer_cv_mean']
    print(f"\n3. Optimistic Bias: {bias:.4f}")
    print(f"   Relative bias: {(bias / summary['outer_cv_mean'] * 100):.2f}%")
    
    # Plot results
    nested_cv.plot_results("Nested CV: SVM on Breast Cancer Dataset")
    
    return nested_cv, simple_scores


def demonstrate_multiple_models():
    """
    Demonstrate nested CV with multiple models
    """
    print("\nDemonstrating Nested CV with Multiple Models")
    print("=" * 60)
    
    # Load dataset
    data = load_wine()
    X, y = data.data, data.target
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define models and parameter grids
    models = {
        'Random Forest': {
            'estimator': RandomForestClassifier(random_state=42),
            'param_grid': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, None],
                'min_samples_split': [2, 5, 10]
            }
        },
        'SVM': {
            'estimator': SVC(random_state=42),
            'param_grid': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 0.01, 0.1],
                'kernel': ['rbf', 'linear']
            }
        },
        'Logistic Regression': {
            'estimator': LogisticRegression(random_state=42, max_iter=1000),
            'param_grid': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        }
    }
    
    results = {}
    
    # Perform nested CV for each model
    for model_name, model_config in models.items():
        print(f"\nProcessing {model_name}...")
        
        nested_cv = NestedCVPipeline(
            estimator=model_config['estimator'],
            param_grid=model_config['param_grid'],
            inner_cv=3,  # Smaller for faster execution
            outer_cv=5,
            scoring='accuracy'
        )
        
        nested_cv.fit(X_scaled, y)
        results[model_name] = nested_cv.get_performance_summary()
        
        print(f"   Accuracy: {results[model_name]['outer_cv_mean']:.4f} ± "
              f"{results[model_name]['outer_cv_std']:.4f}")
    
    # Compare models
    print("\n" + "=" * 60)
    print("Model Comparison Summary:")
    print("=" * 60)
    
    for model_name, summary in results.items():
        ci_low, ci_high = summary['confidence_interval_95']
        print(f"{model_name:20}: {summary['outer_cv_mean']:.4f} ± "
              f"{summary['outer_cv_std']:.4f} "
              f"(95% CI: [{ci_low:.4f}, {ci_high:.4f}])")
    
    # Find best model
    best_model = max(results.keys(), key=lambda k: results[k]['outer_cv_mean'])
    print(f"\nBest Model: {best_model}")
    print(f"Best Score: {results[best_model]['outer_cv_mean']:.4f} ± "
          f"{results[best_model]['outer_cv_std']:.4f}")
    
    return results


if __name__ == "__main__":
    # Example 1: Compare nested vs simple CV
    nested_cv, simple_scores = compare_nested_vs_simple_cv()
    
    # Example 2: Multiple models comparison
    model_results = demonstrate_multiple_models()
    
    print("\n" + "=" * 80)
    print("Nested Cross-Validation Demonstration Complete!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. Nested CV provides unbiased performance estimates")
    print("2. Simple CV often shows optimistic bias due to data leakage")
    print("3. Outer CV variance indicates model stability")
    print("4. Use nested CV for final model selection and reporting") 