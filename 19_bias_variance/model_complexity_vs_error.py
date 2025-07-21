"""
Model Complexity vs Error Analysis

This module demonstrates how model complexity affects training and validation error
across different machine learning algorithms, showcasing the classic U-shaped error curve.

Algorithms analyzed:
- Decision Trees (max_depth parameter)
- Support Vector Machines (C parameter)
- k-Nearest Neighbors (k parameter)
- Random Forest (n_estimators parameter)

Author: ML Learning Series
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer, load_wine, make_classification
from sklearn.model_selection import train_test_split, validation_curve, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class ModelComplexityAnalyzer:
    """
    Comprehensive analysis of model complexity effects on bias-variance tradeoff.
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
        np.random.seed(random_state)
        
        # Store results
        self.results = {}
        self.dataset_info = {}
        
    def load_dataset(self, dataset_name='breast_cancer'):
        """
        Load and prepare dataset for analysis.
        
        Parameters:
        -----------
        dataset_name : str
            Name of dataset to load ('breast_cancer', 'wine', 'synthetic')
            
        Returns:
        --------
        X, y : arrays
            Features and target values
        """
        if dataset_name == 'breast_cancer':
            data = load_breast_cancer()
            X, y = data.data, data.target
            self.dataset_info = {
                'name': 'Breast Cancer Wisconsin',
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'n_classes': len(np.unique(y)),
                'description': 'Binary classification of breast cancer diagnosis'
            }
            
        elif dataset_name == 'wine':
            data = load_wine()
            X, y = data.data, data.target
            self.dataset_info = {
                'name': 'Wine Recognition',
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'n_classes': len(np.unique(y)),
                'description': 'Multi-class wine classification'
            }
            
        elif dataset_name == 'synthetic':
            X, y = make_classification(
                n_samples=1000, n_features=20, n_informative=10,
                n_redundant=5, n_clusters_per_class=1,
                random_state=self.random_state
            )
            self.dataset_info = {
                'name': 'Synthetic Classification',
                'n_samples': X.shape[0],
                'n_features': X.shape[1],
                'n_classes': len(np.unique(y)),
                'description': 'Synthetic binary classification dataset'
            }
            
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        print(f"Loaded {self.dataset_info['name']} dataset:")
        print(f"  Samples: {self.dataset_info['n_samples']}")
        print(f"  Features: {self.dataset_info['n_features']}")
        print(f"  Classes: {self.dataset_info['n_classes']}")
        print(f"  Description: {self.dataset_info['description']}")
        
        return X, y
    
    def analyze_decision_tree_complexity(self, X, y, max_depths=None):
        """
        Analyze how decision tree depth affects bias-variance tradeoff.
        
        Parameters:
        -----------
        X, y : arrays
            Training data
        max_depths : list, optional
            List of max_depth values to test
            
        Returns:
        --------
        results : dict
            Training and validation scores
        """
        if max_depths is None:
            max_depths = range(1, 21)
        
        print("\nAnalyzing Decision Tree complexity (max_depth)...")
        
        # Use validation curve to get training and validation scores
        train_scores, val_scores = validation_curve(
            DecisionTreeClassifier(random_state=self.random_state),
            X, y, param_name='max_depth', param_range=max_depths,
            cv=5, scoring='accuracy', n_jobs=-1
        )
        
        results = {
            'parameter': 'max_depth',
            'param_range': list(max_depths),
            'train_scores_mean': np.mean(train_scores, axis=1),
            'train_scores_std': np.std(train_scores, axis=1),
            'val_scores_mean': np.mean(val_scores, axis=1),
            'val_scores_std': np.std(val_scores, axis=1),
            'algorithm': 'Decision Tree'
        }
        
        # Find optimal depth
        optimal_idx = np.argmax(results['val_scores_mean'])
        optimal_depth = max_depths[optimal_idx]
        
        print(f"  Optimal max_depth: {optimal_depth}")
        print(f"  Best validation accuracy: {results['val_scores_mean'][optimal_idx]:.4f}")
        
        self.results['decision_tree'] = results
        return results
    
    def analyze_svm_complexity(self, X, y, C_values=None):
        """
        Analyze how SVM regularization parameter C affects bias-variance tradeoff.
        
        Parameters:
        -----------
        X, y : arrays
            Training data
        C_values : list, optional
            List of C values to test
            
        Returns:
        --------
        results : dict
            Training and validation scores
        """
        if C_values is None:
            C_values = np.logspace(-3, 2, 15)
        
        print("\nAnalyzing SVM complexity (C parameter)...")
        
        # Use validation curve
        train_scores, val_scores = validation_curve(
            SVC(kernel='rbf', random_state=self.random_state),
            X, y, param_name='C', param_range=C_values,
            cv=5, scoring='accuracy', n_jobs=-1
        )
        
        results = {
            'parameter': 'C',
            'param_range': list(C_values),
            'train_scores_mean': np.mean(train_scores, axis=1),
            'train_scores_std': np.std(train_scores, axis=1),
            'val_scores_mean': np.mean(val_scores, axis=1),
            'val_scores_std': np.std(val_scores, axis=1),
            'algorithm': 'Support Vector Machine'
        }
        
        # Find optimal C
        optimal_idx = np.argmax(results['val_scores_mean'])
        optimal_C = C_values[optimal_idx]
        
        print(f"  Optimal C: {optimal_C:.4f}")
        print(f"  Best validation accuracy: {results['val_scores_mean'][optimal_idx]:.4f}")
        
        self.results['svm'] = results
        return results
    
    def analyze_knn_complexity(self, X, y, k_values=None):
        """
        Analyze how k-NN parameter k affects bias-variance tradeoff.
        
        Parameters:
        -----------
        X, y : arrays
            Training data
        k_values : list, optional
            List of k values to test
            
        Returns:
        --------
        results : dict
            Training and validation scores
        """
        if k_values is None:
            # Use odd numbers to avoid ties
            k_values = list(range(1, min(51, len(X)//5), 2))
        
        print("\nAnalyzing k-NN complexity (k parameter)...")
        
        # Use validation curve
        train_scores, val_scores = validation_curve(
            KNeighborsClassifier(),
            X, y, param_name='n_neighbors', param_range=k_values,
            cv=5, scoring='accuracy', n_jobs=-1
        )
        
        results = {
            'parameter': 'k (n_neighbors)',
            'param_range': list(k_values),
            'train_scores_mean': np.mean(train_scores, axis=1),
            'train_scores_std': np.std(train_scores, axis=1),
            'val_scores_mean': np.mean(val_scores, axis=1),
            'val_scores_std': np.std(val_scores, axis=1),
            'algorithm': 'k-Nearest Neighbors'
        }
        
        # Find optimal k
        optimal_idx = np.argmax(results['val_scores_mean'])
        optimal_k = k_values[optimal_idx]
        
        print(f"  Optimal k: {optimal_k}")
        print(f"  Best validation accuracy: {results['val_scores_mean'][optimal_idx]:.4f}")
        
        self.results['knn'] = results
        return results
    
    def analyze_random_forest_complexity(self, X, y, n_estimators_values=None):
        """
        Analyze how Random Forest n_estimators affects bias-variance tradeoff.
        
        Parameters:
        -----------
        X, y : arrays
            Training data
        n_estimators_values : list, optional
            List of n_estimators values to test
            
        Returns:
        --------
        results : dict
            Training and validation scores
        """
        if n_estimators_values is None:
            n_estimators_values = [1, 2, 5, 10, 20, 50, 100, 200, 300, 500]
        
        print("\nAnalyzing Random Forest complexity (n_estimators)...")
        
        # Use validation curve
        train_scores, val_scores = validation_curve(
            RandomForestClassifier(random_state=self.random_state, max_depth=10),
            X, y, param_name='n_estimators', param_range=n_estimators_values,
            cv=5, scoring='accuracy', n_jobs=-1
        )
        
        results = {
            'parameter': 'n_estimators',
            'param_range': list(n_estimators_values),
            'train_scores_mean': np.mean(train_scores, axis=1),
            'train_scores_std': np.std(train_scores, axis=1),
            'val_scores_mean': np.mean(val_scores, axis=1),
            'val_scores_std': np.std(val_scores, axis=1),
            'algorithm': 'Random Forest'
        }
        
        # Find optimal n_estimators
        optimal_idx = np.argmax(results['val_scores_mean'])
        optimal_n_est = n_estimators_values[optimal_idx]
        
        print(f"  Optimal n_estimators: {optimal_n_est}")
        print(f"  Best validation accuracy: {results['val_scores_mean'][optimal_idx]:.4f}")
        
        self.results['random_forest'] = results
        return results
    
    def plot_complexity_analysis(self, save_path=None):
        """
        Create comprehensive visualization of model complexity analysis.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        """
        if not self.results:
            raise ValueError("No results available. Run analysis methods first.")
        
        n_models = len(self.results)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        fig.suptitle(f'Model Complexity Analysis: {self.dataset_info["name"]} Dataset', 
                     fontsize=16, fontweight='bold')
        
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, (model_name, results) in enumerate(self.results.items()):
            if i >= 4:  # Maximum 4 subplots
                break
                
            ax = axes[i]
            param_range = results['param_range']
            train_mean = results['train_scores_mean']
            train_std = results['train_scores_std']
            val_mean = results['val_scores_mean']
            val_std = results['val_scores_std']
            
            # Plot training and validation curves
            ax.plot(param_range, train_mean, 'o-', color=colors[i], 
                   label=f'Training Score', linewidth=2, markersize=4)
            ax.fill_between(param_range, train_mean - train_std, train_mean + train_std,
                           alpha=0.2, color=colors[i])
            
            ax.plot(param_range, val_mean, 's-', color=colors[i], alpha=0.7,
                   label=f'Validation Score', linewidth=2, markersize=4)
            ax.fill_between(param_range, val_mean - val_std, val_mean + val_std,
                           alpha=0.2, color=colors[i])
            
            # Mark optimal point
            optimal_idx = np.argmax(val_mean)
            ax.axvline(param_range[optimal_idx], color=colors[i], 
                      linestyle='--', alpha=0.7, 
                      label=f'Optimal {results["parameter"]}')
            
            # Set scale for specific parameters
            if 'C' in results['parameter'] or model_name == 'svm':
                ax.set_xscale('log')
            
            ax.set_xlabel(f'{results["parameter"]}')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'{results["algorithm"]}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add bias-variance interpretation
            underfitting_region = param_range[:optimal_idx]
            overfitting_region = param_range[optimal_idx+1:]
            
            if underfitting_region:
                ax.axvspan(min(param_range), param_range[optimal_idx], 
                          alpha=0.1, color='blue', label='High Bias')
            if overfitting_region:
                ax.axvspan(param_range[optimal_idx], max(param_range), 
                          alpha=0.1, color='red', label='High Variance')
        
        # Hide unused subplots
        for j in range(i+1, 4):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to: {save_path}")
        
        plt.show()
        return fig
    
    def plot_unified_comparison(self, save_path=None):
        """
        Create unified comparison plot showing all algorithms on same scale.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        """
        if not self.results:
            raise ValueError("No results available. Run analysis methods first.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Unified Model Complexity Comparison: {self.dataset_info["name"]}', 
                     fontsize=14, fontweight='bold')
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        # Plot 1: Validation scores (normalized parameter range)
        for i, (model_name, results) in enumerate(self.results.items()):
            param_range = results['param_range']
            val_mean = results['val_scores_mean']
            
            # Normalize parameter range to 0-1
            param_norm = np.linspace(0, 1, len(param_range))
            
            ax1.plot(param_norm, val_mean, 'o-', color=colors[i], 
                    label=f'{results["algorithm"]}', linewidth=2, markersize=4)
        
        ax1.set_xlabel('Normalized Model Complexity')
        ax1.set_ylabel('Validation Accuracy')
        ax1.set_title('Validation Performance vs Complexity')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Overfitting gap (training - validation)
        for i, (model_name, results) in enumerate(self.results.items()):
            param_range = results['param_range']
            train_mean = results['train_scores_mean']
            val_mean = results['val_scores_mean']
            overfitting_gap = train_mean - val_mean
            
            # Normalize parameter range to 0-1
            param_norm = np.linspace(0, 1, len(param_range))
            
            ax2.plot(param_norm, overfitting_gap, 's-', color=colors[i], 
                    label=f'{results["algorithm"]}', linewidth=2, markersize=4)
        
        ax2.set_xlabel('Normalized Model Complexity')
        ax2.set_ylabel('Overfitting Gap (Train - Val)')
        ax2.set_title('Overfitting vs Complexity')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nUnified comparison plot saved to: {save_path}")
        
        plt.show()
        return fig
    
    def analyze_learning_curves(self, X, y, save_path=None):
        """
        Analyze learning curves to show how training set size affects performance.
        
        Parameters:
        -----------
        X, y : arrays
            Training data
        save_path : str, optional
            Path to save the plot
        """
        print("\nAnalyzing learning curves...")
        
        algorithms = {
            'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=self.random_state),
            'SVM': SVC(C=1.0, random_state=self.random_state),
            'k-NN': KNeighborsClassifier(n_neighbors=5),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        fig.suptitle(f'Learning Curves: {self.dataset_info["name"]} Dataset', 
                     fontsize=16, fontweight='bold')
        
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, (name, model) in enumerate(algorithms.items()):
            train_sizes, train_scores, val_scores = learning_curve(
                model, X, y, cv=5, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring='accuracy'
            )
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            ax = axes[i]
            
            ax.plot(train_sizes, train_mean, 'o-', color=colors[i], 
                   label='Training Score', linewidth=2)
            ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                           alpha=0.2, color=colors[i])
            
            ax.plot(train_sizes, val_mean, 's-', color=colors[i], alpha=0.7,
                   label='Validation Score', linewidth=2)
            ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                           alpha=0.2, color=colors[i])
            
            ax.set_xlabel('Training Set Size')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'{name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Learning curves plot saved to: {save_path}")
        
        plt.show()
        return fig
    
    def print_summary(self):
        """Print comprehensive summary of the analysis."""
        if not self.results:
            print("No results available. Run analysis methods first.")
            return
        
        print("\n" + "="*70)
        print("MODEL COMPLEXITY ANALYSIS SUMMARY")
        print("="*70)
        
        print(f"Dataset: {self.dataset_info['name']}")
        print(f"  Samples: {self.dataset_info['n_samples']}")
        print(f"  Features: {self.dataset_info['n_features']}")
        print(f"  Classes: {self.dataset_info['n_classes']}")
        
        print(f"\nOptimal Parameters and Performance:")
        
        for model_name, results in self.results.items():
            optimal_idx = np.argmax(results['val_scores_mean'])
            optimal_param = results['param_range'][optimal_idx]
            best_val_score = results['val_scores_mean'][optimal_idx]
            best_train_score = results['train_scores_mean'][optimal_idx]
            overfitting_gap = best_train_score - best_val_score
            
            print(f"\n{results['algorithm']}:")
            print(f"  Optimal {results['parameter']}: {optimal_param}")
            print(f"  Best validation accuracy: {best_val_score:.4f}")
            print(f"  Training accuracy: {best_train_score:.4f}")
            print(f"  Overfitting gap: {overfitting_gap:.4f}")
            
            # Classify bias-variance characteristics
            if overfitting_gap < 0.02:
                bias_variance = "Well-balanced"
            elif overfitting_gap > 0.1:
                bias_variance = "High variance (overfitting tendency)"
            else:
                bias_variance = "Moderate variance"
            
            print(f"  Bias-variance profile: {bias_variance}")
        
        print(f"\nKey Insights:")
        print(f"  • Model complexity significantly affects bias-variance tradeoff")
        print(f"  • Training error generally decreases with complexity")
        print(f"  • Validation error typically shows U-shaped curve")
        print(f"  • Optimal complexity balances underfitting and overfitting")
        print(f"  • Different algorithms have different complexity-performance relationships")
        
        print("="*70)


def main():
    """
    Main execution function demonstrating model complexity analysis.
    """
    print("Model Complexity vs Error Analysis")
    print("="*40)
    
    # Initialize analyzer
    analyzer = ModelComplexityAnalyzer(random_state=42)
    
    # Load dataset
    print("\n1. Loading dataset...")
    X, y = analyzer.load_dataset('breast_cancer')
    
    # Analyze different algorithms
    print("\n2. Analyzing model complexity effects...")
    
    # Decision Tree analysis
    analyzer.analyze_decision_tree_complexity(X, y)
    
    # SVM analysis
    analyzer.analyze_svm_complexity(X, y)
    
    # k-NN analysis
    analyzer.analyze_knn_complexity(X, y)
    
    # Random Forest analysis
    analyzer.analyze_random_forest_complexity(X, y)
    
    # Create visualizations
    print("\n3. Creating visualizations...")
    analyzer.plot_complexity_analysis(save_path='plots/model_complexity_analysis.png')
    analyzer.plot_unified_comparison(save_path='plots/unified_complexity_comparison.png')
    analyzer.analyze_learning_curves(X, y, save_path='plots/learning_curves.png')
    
    # Print summary
    analyzer.print_summary()
    
    # Additional analysis with different dataset
    print("\n4. Additional analysis with synthetic dataset...")
    X_syn, y_syn = analyzer.load_dataset('synthetic')
    
    # Quick analysis on synthetic data
    analyzer.analyze_decision_tree_complexity(X_syn, y_syn)
    analyzer.plot_complexity_analysis(save_path='plots/synthetic_complexity_analysis.png')


if __name__ == "__main__":
    main() 