"""
Model Comparison Module

This module compares multiple machine learning models on the same dataset
using consistent cross-validation schemes and provides comprehensive
performance metrics and statistical analysis.

Author: Machine Learning Fundamentals Course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_score
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class ModelComparator:
    """
    Comprehensive model comparison tool
    
    Compares multiple models using consistent cross-validation
    and provides statistical analysis of performance differences
    """
    
    def __init__(self, models=None, cv=5, scoring=['accuracy', 'f1_weighted', 'roc_auc_ovr_weighted'], 
                 random_state=42, n_jobs=-1):
        """
        Initialize model comparator
        
        Parameters:
        -----------
        models : dict, optional
            Dictionary of model name -> sklearn estimator
        cv : int or CV object
            Cross-validation strategy
        scoring : list
            List of scoring metrics
        random_state : int
            Random state for reproducibility
        n_jobs : int
            Number of parallel jobs
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models = models or self._get_default_models()
        self.cv = cv
        self.scoring = scoring
        
        # Results storage
        self.results_ = {}
        self.X_ = None
        self.y_ = None
        self.best_model_ = None
        self.best_score_ = None
        
    def _get_default_models(self):
        """Get default set of models for comparison"""
        return {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=self.random_state,
                n_jobs=self.n_jobs
            ),
            'SVM': SVC(
                random_state=self.random_state,
                probability=True  # For ROC AUC
            ),
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                n_jobs=self.n_jobs
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=self.random_state
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_jobs=self.n_jobs
            ),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(
                random_state=self.random_state
            )
        }
    
    def compare(self, X, y, scale_features=True):
        """
        Compare models on given dataset
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix
        y : array-like of shape (n_samples,)
            Target vector
        scale_features : bool
            Whether to scale features
            
        Returns:
        --------
        self : ModelComparator
            Fitted comparator
        """
        self.X_ = X
        self.y_ = y
        
        # Create pipelines with optional scaling
        pipelines = {}
        for name, model in self.models.items():
            if scale_features:
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', model)
                ])
            else:
                pipeline = Pipeline([
                    ('classifier', model)
                ])
            pipelines[name] = pipeline
        
        # Setup cross-validation
        if isinstance(self.cv, int):
            cv_strategy = StratifiedKFold(
                n_splits=self.cv, 
                shuffle=True, 
                random_state=self.random_state
            )
        else:
            cv_strategy = self.cv
        
        # Perform cross-validation for each model
        print("Comparing models...")
        print("=" * 60)
        
        for name, pipeline in pipelines.items():
            print(f"Evaluating {name}...")
            
            # Cross-validate with multiple metrics
            cv_results = cross_validate(
                pipeline, X, y,
                cv=cv_strategy,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
                return_train_score=True
            )
            
            self.results_[name] = cv_results
            
            # Print summary
            for metric in self.scoring:
                test_scores = cv_results[f'test_{metric}']
                mean_score = np.mean(test_scores)
                std_score = np.std(test_scores)
                print(f"  {metric}: {mean_score:.4f} ± {std_score:.4f}")
            print()
        
        # Find best model
        primary_metric = self.scoring[0]  # Use first metric as primary
        best_name = max(self.results_.keys(), 
                       key=lambda k: np.mean(self.results_[k][f'test_{primary_metric}']))
        
        self.best_model_ = best_name
        self.best_score_ = np.mean(self.results_[best_name][f'test_{primary_metric}'])
        
        print(f"Best Model: {self.best_model_}")
        print(f"Best {primary_metric}: {self.best_score_:.4f}")
        
        return self
    
    def get_summary_table(self):
        """
        Get summary table of all results
        
        Returns:
        --------
        pd.DataFrame : Summary table
        """
        summary_data = []
        
        for model_name, results in self.results_.items():
            row = {'Model': model_name}
            
            for metric in self.scoring:
                test_scores = results[f'test_{metric}']
                train_scores = results[f'train_{metric}']
                
                row[f'{metric}_mean'] = np.mean(test_scores)
                row[f'{metric}_std'] = np.std(test_scores)
                row[f'{metric}_train_mean'] = np.mean(train_scores)
                row[f'overfitting_{metric}'] = np.mean(train_scores) - np.mean(test_scores)
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def plot_comparison(self, figsize=(15, 10), save_path=None):
        """
        Plot comprehensive model comparison
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the plot
        """
        n_metrics = len(self.scoring)
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Model Comparison Analysis', fontsize=16, fontweight='bold')
        
        # 1. Performance comparison (boxplot)
        ax1 = axes[0, 0]
        primary_metric = self.scoring[0]
        
        data_for_box = []
        labels_for_box = []
        for name, results in self.results_.items():
            data_for_box.append(results[f'test_{primary_metric}'])
            labels_for_box.append(name)
        
        box_plot = ax1.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
        ax1.set_title(f'{primary_metric.capitalize()} Distribution')
        ax1.set_ylabel(f'{primary_metric.capitalize()} Score')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(box_plot['boxes'])))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        # 2. Mean performance with error bars
        ax2 = axes[0, 1]
        model_names = list(self.results_.keys())
        means = [np.mean(self.results_[name][f'test_{primary_metric}']) 
                for name in model_names]
        stds = [np.std(self.results_[name][f'test_{primary_metric}']) 
               for name in model_names]
        
        bars = ax2.bar(range(len(model_names)), means, yerr=stds, 
                      capsize=5, alpha=0.7, color=colors)
        ax2.set_title(f'Mean {primary_metric.capitalize()} with Std Dev')
        ax2.set_ylabel(f'{primary_metric.capitalize()} Score')
        ax2.set_xticks(range(len(model_names)))
        ax2.set_xticklabels(model_names, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + std/2,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 3. Multi-metric comparison
        ax3 = axes[1, 0]
        
        # Create heatmap data
        heatmap_data = []
        metric_labels = []
        
        for metric in self.scoring:
            metric_means = [np.mean(self.results_[name][f'test_{metric}']) 
                           for name in model_names]
            heatmap_data.append(metric_means)
            metric_labels.append(metric.replace('_', ' ').title())
        
        heatmap_data = np.array(heatmap_data)
        
        im = ax3.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        ax3.set_xticks(range(len(model_names)))
        ax3.set_xticklabels(model_names, rotation=45)
        ax3.set_yticks(range(len(metric_labels)))
        ax3.set_yticklabels(metric_labels)
        ax3.set_title('Multi-Metric Performance Heatmap')
        
        # Add text annotations
        for i in range(len(metric_labels)):
            for j in range(len(model_names)):
                text = ax3.text(j, i, f'{heatmap_data[i, j]:.3f}',
                               ha="center", va="center", color="black", fontsize=8)
        
        # Add colorbar
        plt.colorbar(im, ax=ax3, shrink=0.8)
        
        # 4. Train vs Test (overfitting analysis)
        ax4 = axes[1, 1]
        
        train_means = [np.mean(self.results_[name][f'train_{primary_metric}']) 
                      for name in model_names]
        test_means = [np.mean(self.results_[name][f'test_{primary_metric}']) 
                     for name in model_names]
        
        # Scatter plot
        scatter = ax4.scatter(train_means, test_means, 
                             c=range(len(model_names)), 
                             cmap='tab10', s=100, alpha=0.7)
        
        # Add diagonal line (perfect generalization)
        min_score = min(min(train_means), min(test_means))
        max_score = max(max(train_means), max(test_means))
        ax4.plot([min_score, max_score], [min_score, max_score], 
                'k--', alpha=0.5, label='Perfect Generalization')
        
        # Add labels for each point
        for i, name in enumerate(model_names):
            ax4.annotate(name, (train_means[i], test_means[i]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.8)
        
        ax4.set_xlabel(f'Train {primary_metric.capitalize()}')
        ax4.set_ylabel(f'Test {primary_metric.capitalize()}')
        ax4.set_title('Overfitting Analysis')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def statistical_significance_test(self, alpha=0.05):
        """
        Perform statistical significance tests between models
        
        Parameters:
        -----------
        alpha : float
            Significance level
            
        Returns:
        --------
        pd.DataFrame : Pairwise comparison results
        """
        primary_metric = self.scoring[0]
        model_names = list(self.results_.keys())
        n_models = len(model_names)
        
        # Create pairwise comparison matrix
        p_values = np.ones((n_models, n_models))
        effect_sizes = np.zeros((n_models, n_models))
        
        for i, model1 in enumerate(model_names):
            scores1 = self.results_[model1][f'test_{primary_metric}']
            
            for j, model2 in enumerate(model_names):
                if i != j:
                    scores2 = self.results_[model2][f'test_{primary_metric}']
                    
                    # Paired t-test
                    t_stat, p_val = stats.ttest_rel(scores1, scores2)
                    p_values[i, j] = p_val
                    
                    # Effect size (Cohen's d)
                    diff = np.mean(scores1) - np.mean(scores2)
                    pooled_std = np.sqrt((np.var(scores1) + np.var(scores2)) / 2)
                    effect_sizes[i, j] = diff / pooled_std if pooled_std > 0 else 0
        
        # Create summary DataFrame
        results = []
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names):
                if i < j:  # Only upper triangle
                    results.append({
                        'Model 1': model1,
                        'Model 2': model2,
                        'Mean Diff': np.mean(self.results_[model1][f'test_{primary_metric}']) - 
                                   np.mean(self.results_[model2][f'test_{primary_metric}']),
                        'P-value': p_values[i, j],
                        'Significant': p_values[i, j] < alpha,
                        'Effect Size': effect_sizes[i, j],
                        'Effect Magnitude': self._interpret_effect_size(abs(effect_sizes[i, j]))
                    })
        
        return pd.DataFrame(results).sort_values('P-value')
    
    def _interpret_effect_size(self, effect_size):
        """Interpret Cohen's d effect size"""
        if effect_size < 0.2:
            return 'Negligible'
        elif effect_size < 0.5:
            return 'Small'
        elif effect_size < 0.8:
            return 'Medium'
        else:
            return 'Large'
    
    def get_best_model_pipeline(self, scale_features=True):
        """
        Get the best model as a fitted pipeline
        
        Parameters:
        -----------
        scale_features : bool
            Whether to include scaling
            
        Returns:
        --------
        sklearn.pipeline.Pipeline : Best model pipeline
        """
        if self.best_model_ is None:
            raise ValueError("Must run compare() first")
        
        best_estimator = self.models[self.best_model_]
        
        if scale_features:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', best_estimator)
            ])
        else:
            pipeline = Pipeline([
                ('classifier', best_estimator)
            ])
        
        # Fit on full dataset
        pipeline.fit(self.X_, self.y_)
        
        return pipeline


def demonstrate_model_comparison():
    """
    Demonstrate model comparison on multiple datasets
    """
    datasets = {
        'Breast Cancer': load_breast_cancer(),
        'Wine': load_wine(),
        'Iris': load_iris()
    }
    
    for dataset_name, dataset in datasets.items():
        print(f"\n{'='*80}")
        print(f"MODEL COMPARISON: {dataset_name.upper()} DATASET")
        print(f"{'='*80}")
        
        X, y = dataset.data, dataset.target
        print(f"Dataset shape: {X.shape}")
        print(f"Number of classes: {len(np.unique(y))}")
        
        # Initialize comparator
        comparator = ModelComparator(
            cv=5,
            scoring=['accuracy', 'f1_weighted', 'precision_weighted'],
            random_state=42
        )
        
        # Compare models
        comparator.compare(X, y, scale_features=True)
        
        # Get summary table
        print(f"\n{'-'*60}")
        print("SUMMARY TABLE:")
        print(f"{'-'*60}")
        summary_df = comparator.get_summary_table()
        print(summary_df.round(4).to_string(index=False))
        
        # Statistical significance testing
        print(f"\n{'-'*60}")
        print("STATISTICAL SIGNIFICANCE TESTS:")
        print(f"{'-'*60}")
        sig_results = comparator.statistical_significance_test()
        print(sig_results.round(4).to_string(index=False))
        
        # Plot comparison
        comparator.plot_comparison(figsize=(15, 10))
        
        # Get best model
        best_pipeline = comparator.get_best_model_pipeline()
        print(f"\nBest model pipeline:")
        print(best_pipeline)


def quick_model_comparison(X, y, dataset_name="Dataset"):
    """
    Quick model comparison function
    
    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Target
    dataset_name : str
        Name of the dataset for display
        
    Returns:
    --------
    dict : Comparison results
    """
    print(f"Quick Model Comparison: {dataset_name}")
    print("=" * 50)
    
    # Initialize with subset of models for speed
    quick_models = {
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
        'SVM': SVC(random_state=42, probability=True),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42)
    }
    
    comparator = ModelComparator(
        models=quick_models,
        cv=3,  # Faster
        scoring=['accuracy', 'f1_weighted'],
        random_state=42
    )
    
    # Compare models
    comparator.compare(X, y)
    
    # Return summary
    summary = {
        'best_model': comparator.best_model_,
        'best_score': comparator.best_score_,
        'results': comparator.results_,
        'summary_table': comparator.get_summary_table()
    }
    
    return summary


if __name__ == "__main__":
    # Full demonstration
    demonstrate_model_comparison()
    
    print(f"\n{'='*80}")
    print("MODEL COMPARISON DEMONSTRATION COMPLETE!")
    print(f"{'='*80}")
    print("\nKey Features:")
    print("1. Consistent cross-validation across all models")
    print("2. Multiple evaluation metrics")
    print("3. Statistical significance testing")
    print("4. Overfitting analysis")
    print("5. Comprehensive visualizations")
    print("6. Best model pipeline extraction") 