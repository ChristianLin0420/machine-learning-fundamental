"""
Scikit-learn Cross-Validation Examples
=====================================

This module demonstrates comprehensive usage of scikit-learn's cross-validation
tools, comparing different CV strategies and their impact on model evaluation:

Cross-Validation Strategies:
- KFold: Standard k-fold cross-validation
- StratifiedKFold: Maintains class distribution
- LeaveOneOut: Exhaustive CV with single test sample
- ShuffleSplit: Random train/test splits
- GroupKFold: For grouped data

Model Comparison:
- Logistic Regression: Linear classification baseline
- SVM: Non-linear kernel-based classifier
- Random Forest: Ensemble tree-based method
- Gradient Boosting: Advanced ensemble method

Advanced Features:
- Cross-validation curves
- Learning curves analysis
- Validation curves for hyperparameters
- Statistical significance testing
- Performance visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import make_classification, load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import (
    KFold, StratifiedKFold, LeaveOneOut, ShuffleSplit, GroupKFold,
    cross_val_score, cross_validate, validation_curve, learning_curve,
    StratifiedShuffleSplit
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, make_scorer
)
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SklearnCVAnalyzer:
    """
    Comprehensive analyzer for scikit-learn cross-validation methods.
    
    This class provides tools for comparing different CV strategies,
    analyzing model performance, and visualizing results.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the analyzer.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.results = {}
    
    def compare_cv_strategies(self, models, X, y, cv_methods=None, scoring='accuracy'):
        """
        Compare different cross-validation strategies across multiple models.
        
        Parameters:
        -----------
        models : dict
            Dictionary of model names and instances
        X : array-like
            Input features
        y : array-like
            Target labels
        cv_methods : dict, optional
            Dictionary of CV method names and instances
        scoring : str or dict
            Scoring metric(s) to use
            
        Returns:
        --------
        dict : Comparison results
        """
        if cv_methods is None:
            cv_methods = self._get_default_cv_methods(X, y)
        
        results = {}
        
        print("Comparing CV strategies across models...")
        
        for model_name, model in models.items():
            print(f"\n🔍 Testing {model_name}")
            model_results = {}
            
            for cv_name, cv_method in cv_methods.items():
                print(f"  Running {cv_name}...")
                
                try:
                    # Perform cross-validation
                    if isinstance(scoring, str):
                        scores = cross_val_score(model, X, y, cv=cv_method, scoring=scoring)
                        cv_results = {scoring: scores}
                    else:
                        cv_results = cross_validate(model, X, y, cv=cv_method, scoring=scoring)
                    
                    # Calculate statistics
                    stats_results = {}
                    for metric_name, metric_scores in cv_results.items():
                        if metric_name.startswith('test_'):
                            clean_name = metric_name.replace('test_', '')
                            stats_results[clean_name] = {
                                'scores': metric_scores,
                                'mean': np.mean(metric_scores),
                                'std': np.std(metric_scores),
                                'min': np.min(metric_scores),
                                'max': np.max(metric_scores),
                                'cv_method': cv_name,
                                'n_splits': len(metric_scores)
                            }
                    
                    model_results[cv_name] = stats_results
                    
                except Exception as e:
                    print(f"    Error: {e}")
                    model_results[cv_name] = None
            
            results[model_name] = model_results
        
        self.results = results
        return results
    
    def _get_default_cv_methods(self, X, y):
        """Get default cross-validation methods."""
        n_samples = len(X)
        
        cv_methods = {
            'KFold': KFold(n_splits=5, shuffle=True, random_state=self.random_state),
            'StratifiedKFold': StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        }
        
        # Add LOOCV only for smaller datasets
        if n_samples <= 200:
            cv_methods['LeaveOneOut'] = LeaveOneOut()
        
        # Add ShuffleSplit
        cv_methods['ShuffleSplit'] = ShuffleSplit(
            n_splits=10, test_size=0.2, random_state=self.random_state
        )
        
        return cv_methods
    
    def plot_cv_comparison(self, results=None, metric='accuracy', save_path=None, figsize=(16, 12)):
        """
        Visualize cross-validation strategy comparison.
        
        Parameters:
        -----------
        results : dict, optional
            Results from compare_cv_strategies
        metric : str
            Metric to visualize
        save_path : str, optional
            Path to save the plot
        figsize : tuple
            Figure size
        """
        if results is None:
            results = self.results
        
        if not results:
            print("No results to plot. Run compare_cv_strategies first.")
            return
        
        # Prepare data for plotting
        plot_data = []
        for model_name, model_results in results.items():
            for cv_name, cv_results in model_results.items():
                if cv_results is not None and metric in cv_results:
                    scores = cv_results[metric]['scores']
                    for score in scores:
                        plot_data.append({
                            'Model': model_name,
                            'CV_Method': cv_name,
                            'Score': score,
                            'Mean_Score': cv_results[metric]['mean'],
                            'Std_Score': cv_results[metric]['std']
                        })
        
        df = pd.DataFrame(plot_data)
        
        if df.empty:
            print(f"No data available for metric: {metric}")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Box plot comparison
        ax1 = axes[0, 0]
        sns.boxplot(data=df, x='CV_Method', y='Score', hue='Model', ax=ax1)
        ax1.set_title(f'{metric.capitalize()} Scores by CV Method')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Mean scores with error bars
        ax2 = axes[0, 1]
        summary_df = df.groupby(['Model', 'CV_Method']).agg({
            'Score': ['mean', 'std']
        }).reset_index()
        summary_df.columns = ['Model', 'CV_Method', 'Mean', 'Std']
        
        for i, model in enumerate(summary_df['Model'].unique()):
            model_data = summary_df[summary_df['Model'] == model]
            x_pos = np.arange(len(model_data)) + i * 0.2
            ax2.errorbar(x_pos, model_data['Mean'], yerr=model_data['Std'],
                        fmt='o-', label=model, capsize=5, capthick=2)
        
        ax2.set_xlabel('CV Method')
        ax2.set_ylabel(f'Mean {metric.capitalize()} Score')
        ax2.set_title('Mean Scores with Standard Deviation')
        ax2.set_xticks(np.arange(len(summary_df['CV_Method'].unique())) + 0.3)
        ax2.set_xticklabels(summary_df['CV_Method'].unique())
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Score distributions
        ax3 = axes[1, 0]
        for i, model in enumerate(df['Model'].unique()):
            model_scores = df[df['Model'] == model]['Score']
            ax3.hist(model_scores, alpha=0.7, label=model, bins=15)
        
        ax3.set_xlabel(f'{metric.capitalize()} Score')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Score Distributions by Model')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Statistical summary table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create summary statistics
        summary_stats = []
        for model_name, model_results in results.items():
            for cv_name, cv_results in model_results.items():
                if cv_results is not None and metric in cv_results:
                    stats = cv_results[metric]
                    summary_stats.append([
                        model_name[:12],  # Truncate long names
                        cv_name[:12],
                        f"{stats['mean']:.3f}",
                        f"{stats['std']:.3f}",
                        f"{stats['n_splits']}"
                    ])
        
        if summary_stats:
            table = ax4.table(
                cellText=summary_stats,
                colLabels=['Model', 'CV Method', 'Mean', 'Std', 'Splits'],
                cellLoc='center',
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
        
        ax4.set_title('Performance Summary')
        
        plt.suptitle(f'Cross-Validation Strategy Comparison - {metric.capitalize()}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"CV comparison plot saved to {save_path}")
        
        plt.show()
    
    def analyze_learning_curves(self, models, X, y, cv=None, save_path=None):
        """
        Analyze learning curves for different models.
        
        Parameters:
        -----------
        models : dict
            Dictionary of model names and instances
        X : array-like
            Input features
        y : array-like
            Target labels
        cv : cross-validation generator, optional
            Cross-validation strategy
        save_path : str, optional
            Path to save the plot
        """
        if cv is None:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        n_models = len(models)
        fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(15, 10))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (model_name, model) in enumerate(models.items()):
            ax = axes[i]
            
            print(f"Analyzing learning curve for {model_name}...")
            
            # Calculate learning curve
            train_sizes, train_scores, val_scores = learning_curve(
                model, X, y, cv=cv, n_jobs=-1, random_state=self.random_state,
                train_sizes=np.linspace(0.1, 1.0, 10)
            )
            
            # Calculate mean and std
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            # Plot learning curves
            ax.plot(train_sizes, train_mean, 'o-', label='Training Score', linewidth=2)
            ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
            
            ax.plot(train_sizes, val_mean, 'o-', label='Validation Score', linewidth=2)
            ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
            
            ax.set_xlabel('Training Set Size')
            ax.set_ylabel('Accuracy Score')
            ax.set_title(f'Learning Curve - {model_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)
        
        # Hide unused subplots
        for i in range(len(models), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Learning Curves Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Learning curves plot saved to {save_path}")
        
        plt.show()
    
    def analyze_validation_curves(self, model, param_name, param_range, X, y, cv=None, save_path=None):
        """
        Analyze validation curves for hyperparameter tuning.
        
        Parameters:
        -----------
        model : estimator
            Machine learning model
        param_name : str
            Parameter name to vary
        param_range : array-like
            Range of parameter values to test
        X : array-like
            Input features
        y : array-like
            Target labels
        cv : cross-validation generator, optional
            Cross-validation strategy
        save_path : str, optional
            Path to save the plot
        """
        if cv is None:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        print(f"Analyzing validation curve for {param_name}...")
        
        # Calculate validation curve
        train_scores, val_scores = validation_curve(
            model, X, y, param_name=param_name, param_range=param_range,
            cv=cv, n_jobs=-1, scoring='accuracy'
        )
        
        # Calculate statistics
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot validation curve
        plt.figure(figsize=(10, 6))
        plt.plot(param_range, train_mean, 'o-', label='Training Score', linewidth=2)
        plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2)
        
        plt.plot(param_range, val_mean, 'o-', label='Validation Score', linewidth=2)
        plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.2)
        
        plt.xlabel(f'{param_name}')
        plt.ylabel('Accuracy Score')
        plt.title(f'Validation Curve - {param_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)
        
        # Add optimal parameter annotation
        best_idx = np.argmax(val_mean)
        best_param = param_range[best_idx]
        best_score = val_mean[best_idx]
        plt.annotate(f'Best: {param_name}={best_param}\nScore: {best_score:.3f}',
                    xy=(best_param, best_score), xytext=(best_param, best_score + 0.1),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Validation curve plot saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
        
        return best_param, best_score
    
    def statistical_significance_test(self, results, model1, model2, cv_method, metric='accuracy'):
        """
        Perform statistical significance test between two models.
        
        Parameters:
        -----------
        results : dict
            Results from compare_cv_strategies
        model1, model2 : str
            Model names to compare
        cv_method : str
            Cross-validation method used
        metric : str
            Metric to compare
            
        Returns:
        --------
        dict : Statistical test results
        """
        try:
            scores1 = results[model1][cv_method][metric]['scores']
            scores2 = results[model2][cv_method][metric]['scores']
            
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(scores1, scores2)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(scores1, ddof=1) + np.var(scores2, ddof=1)) / 2)
            cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std
            
            result = {
                'model1': model1,
                'model2': model2,
                'metric': metric,
                'cv_method': cv_method,
                'mean_diff': np.mean(scores1) - np.mean(scores2),
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant': p_value < 0.05,
                'interpretation': self._interpret_effect_size(cohens_d)
            }
            
            return result
            
        except KeyError as e:
            print(f"Error: {e}. Check model names and CV method.")
            return None
    
    def _interpret_effect_size(self, cohens_d):
        """Interpret Cohen's d effect size."""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def comprehensive_model_analysis(self, X, y, save_dir='plots'):
        """
        Perform comprehensive model analysis with multiple CV strategies.
        
        Parameters:
        -----------
        X : array-like
            Input features
        y : array-like
            Target labels
        save_dir : str
            Directory to save plots
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("Starting comprehensive model analysis...")
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'SVM': SVC(random_state=self.random_state, probability=True),
            'Random Forest': RandomForestClassifier(random_state=self.random_state, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=self.random_state),
            'Naive Bayes': GaussianNB(),
            'K-NN': KNeighborsClassifier()
        }
        
        # Define scoring metrics
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision_weighted',
            'recall': 'recall_weighted',
            'f1': 'f1_weighted',
            'roc_auc': 'roc_auc_ovr_weighted'
        }
        
        # Compare CV strategies
        results = self.compare_cv_strategies(models, X, y, scoring=scoring)
        
        # Plot comparisons for each metric
        for metric in scoring.keys():
            self.plot_cv_comparison(
                results, metric=metric,
                save_path=f'{save_dir}/cv_comparison_{metric}.png'
            )
        
        # Learning curves analysis
        self.analyze_learning_curves(
            models, X, y,
            save_path=f'{save_dir}/learning_curves.png'
        )
        
        # Validation curves for key models
        print("\nAnalyzing validation curves...")
        
        # SVM C parameter
        svm_model = SVC(random_state=self.random_state)
        c_range = np.logspace(-3, 2, 10)
        self.analyze_validation_curves(
            svm_model, 'C', c_range, X, y,
            save_path=f'{save_dir}/validation_curve_svm_C.png'
        )
        
        # Random Forest n_estimators
        rf_model = RandomForestClassifier(random_state=self.random_state)
        n_est_range = np.arange(10, 201, 20)
        self.analyze_validation_curves(
            rf_model, 'n_estimators', n_est_range, X, y,
            save_path=f'{save_dir}/validation_curve_rf_estimators.png'
        )
        
        # Statistical significance tests
        print("\nPerforming statistical significance tests...")
        cv_method = 'StratifiedKFold'
        model_pairs = [
            ('Random Forest', 'SVM'),
            ('Gradient Boosting', 'Random Forest'),
            ('Logistic Regression', 'SVM')
        ]
        
        for model1, model2 in model_pairs:
            sig_test = self.statistical_significance_test(
                results, model1, model2, cv_method, 'accuracy'
            )
            if sig_test:
                print(f"\n{model1} vs {model2}:")
                print(f"  Mean difference: {sig_test['mean_diff']:.4f}")
                print(f"  P-value: {sig_test['p_value']:.4f}")
                print(f"  Significant: {sig_test['significant']}")
                print(f"  Effect size: {sig_test['interpretation']}")
        
        return results


def load_datasets():
    """Load various datasets for analysis."""
    datasets = {}
    
    # Breast cancer dataset
    cancer = load_breast_cancer()
    datasets['breast_cancer'] = {
        'X': cancer.data,
        'y': cancer.target,
        'name': 'Breast Cancer'
    }
    
    # Wine dataset
    wine = load_wine()
    datasets['wine'] = {
        'X': wine.data,
        'y': wine.target,
        'name': 'Wine Classification'
    }
    
    # Digits dataset (subset for faster computation)
    digits = load_digits()
    datasets['digits'] = {
        'X': digits.data[:500],  # Subset for faster processing
        'y': digits.target[:500],
        'name': 'Digits Recognition'
    }
    
    # Synthetic imbalanced dataset
    X_synth, y_synth = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        n_redundant=5, n_clusters_per_class=1, weights=[0.7, 0.3],
        random_state=42
    )
    datasets['synthetic'] = {
        'X': X_synth,
        'y': y_synth,
        'name': 'Synthetic Imbalanced'
    }
    
    return datasets


def main():
    """
    Main function to demonstrate sklearn cross-validation examples.
    """
    print("🎯 SCIKIT-LEARN CROSS-VALIDATION EXAMPLES")
    print("=" * 50)
    
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Load datasets
    datasets = load_datasets()
    
    # Initialize analyzer
    analyzer = SklearnCVAnalyzer(random_state=42)
    
    # Test with breast cancer dataset
    dataset = datasets['breast_cancer']
    X, y = dataset['X'], dataset['y']
    
    print(f"\n📊 Analyzing {dataset['name']} Dataset")
    print(f"Shape: {X.shape}, Classes: {len(np.unique(y))}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Quick comparison of basic models
    print("\n🔍 Quick Model Comparison")
    print("-" * 30)
    
    basic_models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    # Basic CV comparison
    basic_results = analyzer.compare_cv_strategies(
        basic_models, X_scaled, y,
        scoring={'accuracy': 'accuracy', 'f1': 'f1_weighted'}
    )
    
    # Plot basic comparison
    analyzer.plot_cv_comparison(
        basic_results, metric='accuracy',
        save_path='plots/basic_cv_comparison.png'
    )
    
    # Comprehensive analysis
    print("\n🔬 Comprehensive Analysis")
    print("-" * 25)
    
    comprehensive_results = analyzer.comprehensive_model_analysis(
        X_scaled, y, save_dir='plots/comprehensive'
    )
    
    # Test with different datasets
    print("\n📈 Testing Multiple Datasets")
    print("-" * 30)
    
    for dataset_name, dataset_info in datasets.items():
        if dataset_name == 'breast_cancer':
            continue  # Already tested
        
        print(f"\nTesting {dataset_info['name']}...")
        X_test = dataset_info['X']
        y_test = dataset_info['y']
        
        # Standardize if needed
        if X_test.dtype != 'float64' or np.max(X_test) > 10:
            X_test = StandardScaler().fit_transform(X_test)
        
        # Quick model comparison
        quick_models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=50)
        }
        
        dataset_results = analyzer.compare_cv_strategies(
            quick_models, X_test, y_test, scoring='accuracy'
        )
        
        # Print results
        for model_name, model_results in dataset_results.items():
            for cv_name, cv_results in model_results.items():
                if cv_results is not None and 'accuracy' in cv_results:
                    acc_stats = cv_results['accuracy']
                    print(f"  {model_name} ({cv_name}): {acc_stats['mean']:.3f} ± {acc_stats['std']:.3f}")
    
    # Demonstrate specific CV methods
    print("\n🔧 Detailed CV Method Demonstration")
    print("-" * 35)
    
    # Use wine dataset for detailed demo
    wine_data = datasets['wine']
    X_wine, y_wine = wine_data['X'], wine_data['y']
    X_wine_scaled = StandardScaler().fit_transform(X_wine)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    
    # Different CV strategies
    cv_strategies = {
        'KFold (5)': KFold(n_splits=5, shuffle=True, random_state=42),
        'StratifiedKFold (5)': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        'KFold (10)': KFold(n_splits=10, shuffle=True, random_state=42),
        'LeaveOneOut': LeaveOneOut(),
        'ShuffleSplit': ShuffleSplit(n_splits=20, test_size=0.2, random_state=42)
    }
    
    print("Logistic Regression performance with different CV strategies:")
    for cv_name, cv_strategy in cv_strategies.items():
        scores = cross_val_score(model, X_wine_scaled, y_wine, cv=cv_strategy, scoring='accuracy')
        print(f"  {cv_name:>18}: {np.mean(scores):.3f} ± {np.std(scores):.3f} (n_splits={len(scores)})")
    
    # Demonstrate grouped CV
    print("\n🎲 Grouped Cross-Validation Demo")
    print("-" * 30)
    
    # Create synthetic grouped data
    np.random.seed(42)
    n_groups = 10
    samples_per_group = 20
    groups = np.repeat(range(n_groups), samples_per_group)
    
    X_grouped = np.random.randn(n_groups * samples_per_group, 5)
    # Add group-specific bias
    for i in range(n_groups):
        group_mask = groups == i
        X_grouped[group_mask] += np.random.randn(5) * 0.5
    
    y_grouped = (np.sum(X_grouped, axis=1) > 0).astype(int)
    
    # Compare regular CV vs GroupKFold
    regular_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    group_cv = GroupKFold(n_splits=5)
    
    regular_scores = cross_val_score(model, X_grouped, y_grouped, cv=regular_cv)
    group_scores = cross_val_score(model, X_grouped, y_grouped, cv=group_cv, groups=groups)
    
    print(f"Regular KFold:     {np.mean(regular_scores):.3f} ± {np.std(regular_scores):.3f}")
    print(f"GroupKFold:        {np.mean(group_scores):.3f} ± {np.std(group_scores):.3f}")
    print(f"Difference impact: {abs(np.mean(regular_scores) - np.mean(group_scores)):.3f}")
    
    print("\n✅ SKLEARN CROSS-VALIDATION EXAMPLES COMPLETE!")
    print("📁 Check the 'plots' folder for detailed visualizations.")
    print("🔧 Key demonstrations include:")
    print("   - KFold vs StratifiedKFold comparison")
    print("   - LeaveOneOut for small datasets")
    print("   - Learning curves analysis")
    print("   - Validation curves for hyperparameters")
    print("   - Statistical significance testing")
    print("   - Grouped cross-validation")
    
    return analyzer, datasets

if __name__ == "__main__":
    main() 