"""
Regularized Logistic Regression
===============================

This module demonstrates L1 and L2 regularized logistic regression on classification
datasets, showing how regularization affects model complexity, feature selection,
and decision boundaries.

Key Concepts:
- L1 regularization in logistic regression (Lasso-like feature selection)
- L2 regularization in logistic regression (Ridge-like coefficient shrinkage)
- Decision boundary visualization
- Regularization parameter C (inverse of alpha)
- Feature importance and sparsity effects

Author: Machine Learning Fundamentals Course
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_breast_cancer, make_classification, load_wine
from sklearn.model_selection import train_test_split, validation_curve, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RegularizedLogisticAnalyzer:
    """
    Comprehensive analysis of regularized logistic regression.
    
    This class demonstrates how L1 and L2 regularization affect logistic regression
    models, including feature selection, coefficient shrinkage, and decision boundaries.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the regularized logistic regression analyzer.
        
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
        
    def load_data(self, dataset='breast_cancer'):
        """
        Load and prepare classification dataset.
        
        Parameters:
        -----------
        dataset : str
            Dataset to use ('breast_cancer', 'wine', or 'synthetic')
            
        Returns:
        --------
        dict : Dataset information
        """
        if dataset == 'breast_cancer':
            data = load_breast_cancer()
            X, y = data.data, data.target
            feature_names = data.feature_names
            class_names = data.target_names
            print("🏥 Loading Breast Cancer Dataset")
            print("=" * 40)
            print("Target: Malignant (0) vs Benign (1)")
            
        elif dataset == 'wine':
            data = load_wine()
            X, y = data.data, data.target
            feature_names = data.feature_names
            class_names = data.target_names
            # Convert to binary classification (class 0 vs others)
            y = (y == 0).astype(int)
            print("🍷 Loading Wine Dataset (Binary: Class 0 vs Others)")
            print("=" * 55)
            
        else:  # synthetic
            X, y = make_classification(
                n_samples=1000, n_features=20, n_informative=10,
                n_redundant=10, n_clusters_per_class=1, 
                weights=[0.6, 0.4], random_state=self.random_state
            )
            feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]
            class_names = ['Class_0', 'Class_1']
            print("🔬 Creating Synthetic Binary Classification Dataset")
            print("=" * 55)
        
        print(f"Dataset shape: {X.shape}")
        print(f"Class distribution: {np.bincount(y)}")
        print(f"Features: {len(feature_names)} total")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
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
            'name': dataset,
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names,
            'class_names': class_names,
            'scaler': scaler
        }
    
    def compare_regularization_methods(self, C_range=None):
        """
        Compare L1 and L2 regularized logistic regression across different C values.
        
        Parameters:
        -----------
        C_range : array-like, optional
            Range of C values to test (C = 1/alpha)
            
        Returns:
        --------
        dict : Comparison results
        """
        if C_range is None:
            C_range = np.logspace(-3, 3, 20)  # 0.001 to 1000
        
        print("\n🔍 Comparing L1 vs L2 Regularized Logistic Regression")
        print("=" * 60)
        
        # Models to compare
        regularization_types = ['l1', 'l2', 'none']
        solvers = {'l1': 'liblinear', 'l2': 'lbfgs', 'none': 'lbfgs'}
        
        results = {}
        
        for reg_type in regularization_types:
            print(f"\nTesting {reg_type.upper()} regularization...")
            
            if reg_type == 'none':
                # No regularization (very high C)
                model = LogisticRegression(
                    penalty='l2', C=1e6, solver=solvers[reg_type], 
                    random_state=self.random_state, max_iter=2000
                )
                model.fit(self.X_train, self.y_train)
                
                y_pred_train = model.predict(self.X_train)
                y_pred_test = model.predict(self.X_test)
                y_proba_test = model.predict_proba(self.X_test)[:, 1]
                
                # Calculate metrics
                train_acc = accuracy_score(self.y_train, y_pred_train)
                test_acc = accuracy_score(self.y_test, y_pred_test)
                test_precision = precision_score(self.y_test, y_pred_test)
                test_recall = recall_score(self.y_test, y_pred_test)
                test_f1 = f1_score(self.y_test, y_pred_test)
                test_auc = roc_auc_score(self.y_test, y_proba_test)
                non_zero_coef = np.sum(np.abs(model.coef_[0]) > 1e-5)
                
                results[reg_type] = {
                    'C': [1e6],
                    'train_acc': [train_acc],
                    'test_acc': [test_acc],
                    'test_precision': [test_precision],
                    'test_recall': [test_recall],
                    'test_f1': [test_f1],
                    'test_auc': [test_auc],
                    'non_zero_coef': [non_zero_coef],
                    'coefficients': [model.coef_[0].copy()]
                }
                
                print(f"  Accuracy: {test_acc:.3f}, AUC: {test_auc:.3f}, Non-zero coef: {non_zero_coef}")
                
            else:
                # Test different C values
                C_results = {
                    'C': [],
                    'train_acc': [],
                    'test_acc': [],
                    'test_precision': [],
                    'test_recall': [],
                    'test_f1': [],
                    'test_auc': [],
                    'non_zero_coef': [],
                    'coefficients': []
                }
                
                for C in C_range:
                    model = LogisticRegression(
                        penalty=reg_type, C=C, solver=solvers[reg_type],
                        random_state=self.random_state, max_iter=2000
                    )
                    model.fit(self.X_train, self.y_train)
                    
                    y_pred_train = model.predict(self.X_train)
                    y_pred_test = model.predict(self.X_test)
                    y_proba_test = model.predict_proba(self.X_test)[:, 1]
                    
                    # Calculate metrics
                    train_acc = accuracy_score(self.y_train, y_pred_train)
                    test_acc = accuracy_score(self.y_test, y_pred_test)
                    test_precision = precision_score(self.y_test, y_pred_test, zero_division=0)
                    test_recall = recall_score(self.y_test, y_pred_test, zero_division=0)
                    test_f1 = f1_score(self.y_test, y_pred_test, zero_division=0)
                    test_auc = roc_auc_score(self.y_test, y_proba_test)
                    non_zero_coef = np.sum(np.abs(model.coef_[0]) > 1e-5)
                    
                    C_results['C'].append(C)
                    C_results['train_acc'].append(train_acc)
                    C_results['test_acc'].append(test_acc)
                    C_results['test_precision'].append(test_precision)
                    C_results['test_recall'].append(test_recall)
                    C_results['test_f1'].append(test_f1)
                    C_results['test_auc'].append(test_auc)
                    C_results['non_zero_coef'].append(non_zero_coef)
                    C_results['coefficients'].append(model.coef_[0].copy())
                
                results[reg_type] = C_results
                
                # Find best C based on test AUC
                best_idx = np.argmax(C_results['test_auc'])
                best_C = C_results['C'][best_idx]
                best_acc = C_results['test_acc'][best_idx]
                best_auc = C_results['test_auc'][best_idx]
                best_non_zero = C_results['non_zero_coef'][best_idx]
                
                print(f"  Best C: {best_C:.3f}, Accuracy: {best_acc:.3f}, AUC: {best_auc:.3f}, Non-zero coef: {best_non_zero}")
        
        self.results = results
        return results
    
    def plot_regularization_comparison(self, save_path=None, figsize=(16, 12)):
        """
        Create comprehensive plots comparing L1 and L2 regularization.
        
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
        fig.suptitle('Regularized Logistic Regression Comparison', fontsize=16, fontweight='bold')
        
        colors = {'l1': 'red', 'l2': 'blue', 'none': 'green'}
        labels = {'l1': 'L1 (Lasso)', 'l2': 'L2 (Ridge)', 'none': 'No Regularization'}
        
        # Plot 1: Accuracy vs C
        ax1 = axes[0, 0]
        for reg_type, results in self.results.items():
            if reg_type == 'none':
                ax1.axhline(y=results['test_acc'][0], color=colors[reg_type], 
                           linestyle='--', label=labels[reg_type], alpha=0.7)
            else:
                ax1.semilogx(results['C'], results['test_acc'], 
                           'o-', color=colors[reg_type], label=labels[reg_type])
        
        ax1.set_xlabel('C (Inverse Regularization Strength)')
        ax1.set_ylabel('Test Accuracy')
        ax1.set_title('Test Accuracy vs C')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: AUC vs C
        ax2 = axes[0, 1]
        for reg_type, results in self.results.items():
            if reg_type == 'none':
                ax2.axhline(y=results['test_auc'][0], color=colors[reg_type], 
                           linestyle='--', label=labels[reg_type], alpha=0.7)
            else:
                ax2.semilogx(results['C'], results['test_auc'], 
                           'o-', color=colors[reg_type], label=labels[reg_type])
        
        ax2.set_xlabel('C (Inverse Regularization Strength)')
        ax2.set_ylabel('Test AUC')
        ax2.set_title('Test AUC vs C')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Number of Non-zero Coefficients
        ax3 = axes[0, 2]
        for reg_type, results in self.results.items():
            if reg_type == 'none':
                ax3.axhline(y=results['non_zero_coef'][0], color=colors[reg_type], 
                           linestyle='--', label=labels[reg_type], alpha=0.7)
            else:
                ax3.semilogx(results['C'], results['non_zero_coef'], 
                           'o-', color=colors[reg_type], label=labels[reg_type])
        
        ax3.set_xlabel('C (Inverse Regularization Strength)')
        ax3.set_ylabel('Number of Non-zero Coefficients')
        ax3.set_title('Feature Selection Effect')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Training vs Test Accuracy (Overfitting)
        ax4 = axes[1, 0]
        for reg_type, results in self.results.items():
            if reg_type != 'none':
                ax4.semilogx(results['C'], results['train_acc'], 
                           '--', color=colors[reg_type], alpha=0.7, label=f'{labels[reg_type]} (Train)')
                ax4.semilogx(results['C'], results['test_acc'], 
                           '-', color=colors[reg_type], label=f'{labels[reg_type]} (Test)')
        
        ax4.set_xlabel('C (Inverse Regularization Strength)')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Overfitting Analysis')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Best Model Performance Comparison
        ax5 = axes[1, 1]
        best_results = {}
        metrics = ['test_acc', 'test_precision', 'test_recall', 'test_f1', 'test_auc']
        
        for reg_type, results in self.results.items():
            if reg_type == 'none':
                best_results[reg_type] = {metric: results[metric][0] for metric in metrics}
            else:
                best_idx = np.argmax(results['test_auc'])
                best_results[reg_type] = {metric: results[metric][best_idx] for metric in metrics}
        
        # Create performance heatmap
        performance_matrix = []
        reg_types = list(best_results.keys())
        
        for metric in metrics:
            row = [best_results[reg_type][metric] for reg_type in reg_types]
            performance_matrix.append(row)
        
        performance_matrix = np.array(performance_matrix)
        
        im = ax5.imshow(performance_matrix, cmap='YlOrRd', aspect='auto')
        ax5.set_xticks(range(len(reg_types)))
        ax5.set_xticklabels([labels[rt] for rt in reg_types])
        ax5.set_yticks(range(len(metrics)))
        ax5.set_yticklabels([m.replace('test_', '').upper() for m in metrics])
        ax5.set_title('Performance Heatmap (Best Models)')
        
        # Add text annotations
        for i in range(len(metrics)):
            for j in range(len(reg_types)):
                text = ax5.text(j, i, f'{performance_matrix[i, j]:.3f}',
                               ha="center", va="center", color="black", fontsize=9)
        
        plt.colorbar(im, ax=ax5, shrink=0.8)
        
        # Plot 6: Coefficient Comparison (Best models)
        ax6 = axes[1, 2]
        
        # Get best coefficients for each method
        best_coefs = {}
        for reg_type, results in self.results.items():
            if reg_type == 'none':
                best_coefs[reg_type] = results['coefficients'][0]
            else:
                best_idx = np.argmax(results['test_auc'])
                best_coefs[reg_type] = results['coefficients'][best_idx]
        
        # Show top features only
        n_top_features = min(10, len(self.feature_names))
        
        # Find most important features based on L2 regularization
        l2_coefs = np.abs(best_coefs['l2'])
        top_features_idx = np.argsort(l2_coefs)[-n_top_features:][::-1]
        top_feature_names = [self.feature_names[i] for i in top_features_idx]
        
        x_pos = np.arange(n_top_features)
        width = 0.25
        
        for i, (reg_type, coefs) in enumerate(best_coefs.items()):
            if reg_type != 'none':  # Skip none for clarity
                feature_coefs = coefs[top_features_idx]
                ax6.bar(x_pos + i * width, feature_coefs, width, 
                       label=labels[reg_type], color=colors[reg_type], alpha=0.7)
        
        ax6.set_xlabel('Top Features')
        ax6.set_ylabel('Coefficient Value')
        ax6.set_title(f'Top {n_top_features} Feature Coefficients')
        ax6.set_xticks(x_pos + width/2)
        ax6.set_xticklabels([f'F{i+1}' for i in range(n_top_features)], rotation=45)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Regularized logistic regression plot saved to {save_path}")
        
        plt.show()
    
    def plot_decision_boundaries_2d(self, save_path=None, figsize=(15, 5)):
        """
        Plot decision boundaries for 2D projections of the data.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        figsize : tuple
            Figure size
        """
        print("\n🎨 Creating Decision Boundary Visualizations")
        print("=" * 45)
        
        # Reduce dimensionality to 2D for visualization
        pca = PCA(n_components=2, random_state=self.random_state)
        X_train_2d = pca.fit_transform(self.X_train)
        X_test_2d = pca.transform(self.X_test)
        
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total variance explained: {np.sum(pca.explained_variance_ratio_):.3f}")
        
        # Train models with different regularization
        C_values = [0.1, 1.0, 10.0]
        regularizations = ['l1', 'l2']
        
        fig, axes = plt.subplots(len(regularizations), len(C_values), figsize=figsize)
        fig.suptitle('Decision Boundaries: L1 vs L2 Regularization (2D PCA)', fontsize=14, fontweight='bold')
        
        # Create a mesh for plotting decision boundaries
        h = 0.02  # step size in the mesh
        x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
        y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        for reg_idx, reg_type in enumerate(regularizations):
            for c_idx, C in enumerate(C_values):
                ax = axes[reg_idx, c_idx]
                
                # Train model
                solver = 'liblinear' if reg_type == 'l1' else 'lbfgs'
                model = LogisticRegression(
                    penalty=reg_type, C=C, solver=solver,
                    random_state=self.random_state, max_iter=2000
                )
                model.fit(X_train_2d, self.y_train)
                
                # Predict on mesh
                mesh_points = np.c_[xx.ravel(), yy.ravel()]
                Z = model.predict_proba(mesh_points)[:, 1]
                Z = Z.reshape(xx.shape)
                
                # Plot decision boundary
                contour = ax.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdYlBu')
                ax.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2)
                
                # Plot data points
                scatter = ax.scatter(X_train_2d[:, 0], X_train_2d[:, 1], 
                                   c=self.y_train, cmap='RdYlBu', edgecolors='black')
                
                # Calculate accuracy
                test_accuracy = model.score(X_test_2d, self.y_test)
                
                ax.set_title(f'{reg_type.upper()}: C={C}, Acc={test_accuracy:.3f}')
                ax.set_xlabel('First Principal Component')
                if c_idx == 0:
                    ax.set_ylabel('Second Principal Component')
                
                # Add colorbar to rightmost plots
                if c_idx == len(C_values) - 1:
                    plt.colorbar(contour, ax=ax, shrink=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Decision boundaries plot saved to {save_path}")
        
        plt.show()
    
    def feature_importance_analysis(self):
        """
        Analyze feature importance and selection patterns.
        
        Returns:
        --------
        dict : Feature importance analysis results
        """
        print("\n📊 FEATURE IMPORTANCE ANALYSIS")
        print("=" * 40)
        
        if not self.results:
            print("No results available. Run comparison first.")
            return {}
        
        # Get best models for each regularization type
        best_models = {}
        for reg_type, results in self.results.items():
            if reg_type == 'none':
                best_models[reg_type] = {
                    'coefficients': results['coefficients'][0],
                    'C': results['C'][0],
                    'accuracy': results['test_acc'][0]
                }
            else:
                best_idx = np.argmax(results['test_auc'])
                best_models[reg_type] = {
                    'coefficients': results['coefficients'][best_idx],
                    'C': results['C'][best_idx],
                    'accuracy': results['test_acc'][best_idx]
                }
        
        # Analysis results
        analysis = {}
        
        for reg_type, model_info in best_models.items():
            coefs = model_info['coefficients']
            
            # Feature importance (absolute coefficients)
            feature_importance = np.abs(coefs)
            
            # Selected features (non-zero coefficients)
            selected_features = np.where(np.abs(coefs) > 1e-5)[0]
            
            # Top features
            top_features_idx = np.argsort(feature_importance)[-10:][::-1]
            top_features = [(self.feature_names[i], feature_importance[i]) for i in top_features_idx]
            
            analysis[reg_type] = {
                'coefficients': coefs,
                'feature_importance': feature_importance,
                'selected_features': selected_features,
                'n_selected': len(selected_features),
                'top_features': top_features,
                'C': model_info['C'],
                'accuracy': model_info['accuracy']
            }
            
            print(f"\n{reg_type.upper()} Regularization:")
            print(f"  Best C: {model_info['C']:.3f}")
            print(f"  Test Accuracy: {model_info['accuracy']:.3f}")
            print(f"  Selected Features: {len(selected_features)}/{len(self.feature_names)}")
            print(f"  Top 5 Features:")
            for i, (feature, importance) in enumerate(top_features[:5]):
                print(f"    {i+1}. {feature}: {importance:.4f}")
        
        # Compare feature selection between L1 and L2
        if 'l1' in analysis and 'l2' in analysis:
            l1_selected = set(analysis['l1']['selected_features'])
            l2_selected = set(analysis['l2']['selected_features'])
            
            common_features = l1_selected.intersection(l2_selected)
            l1_only = l1_selected - l2_selected
            l2_only = l2_selected - l1_selected
            
            print(f"\nFeature Selection Comparison:")
            print(f"  L1 selected: {len(l1_selected)} features")
            print(f"  L2 selected: {len(l2_selected)} features")
            print(f"  Common: {len(common_features)} features")
            print(f"  L1 only: {len(l1_only)} features")
            print(f"  L2 only: {len(l2_only)} features")
        
        return analysis
    
    def print_classification_report(self):
        """Print detailed classification report for best models."""
        if not self.results:
            print("No results available. Run comparison first.")
            return
        
        print("\n📋 CLASSIFICATION REPORTS (Best Models)")
        print("=" * 50)
        
        for reg_type, results in self.results.items():
            if reg_type == 'none':
                continue
                
            # Get best model
            best_idx = np.argmax(results['test_auc'])
            best_C = results['C'][best_idx]
            
            # Retrain best model for detailed evaluation
            solver = 'liblinear' if reg_type == 'l1' else 'lbfgs'
            model = LogisticRegression(
                penalty=reg_type, C=best_C, solver=solver,
                random_state=self.random_state, max_iter=2000
            )
            model.fit(self.X_train, self.y_train)
            
            y_pred = model.predict(self.X_test)
            
            print(f"\n{reg_type.upper()} Regularization (C={best_C:.3f}):")
            print("-" * 30)
            print(classification_report(self.y_test, y_pred))


def main():
    """
    Main function to demonstrate regularized logistic regression.
    """
    print("🎯 REGULARIZED LOGISTIC REGRESSION ANALYSIS")
    print("=" * 55)
    
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Initialize analyzer
    analyzer = RegularizedLogisticAnalyzer(random_state=42)
    
    # Test with breast cancer dataset
    print("\n🏥 PHASE 1: Breast Cancer Dataset Analysis")
    print("=" * 45)
    
    dataset_info = analyzer.load_data('breast_cancer')
    
    # Compare regularization methods
    C_range = np.logspace(-3, 3, 25)
    results = analyzer.compare_regularization_methods(C_range)
    
    # Plot comparison
    analyzer.plot_regularization_comparison(save_path='plots/logistic_regression_comparison.png')
    
    # Plot decision boundaries
    analyzer.plot_decision_boundaries_2d(save_path='plots/decision_boundaries_2d.png')
    
    # Feature importance analysis
    feature_analysis = analyzer.feature_importance_analysis()
    
    # Print classification reports
    analyzer.print_classification_report()
    
    # Test with synthetic dataset
    print("\n\n🔬 PHASE 2: Synthetic Dataset Analysis")
    print("=" * 40)
    
    synthetic_info = analyzer.load_data('synthetic')
    
    # Quick comparison on synthetic data
    analyzer.compare_regularization_methods(C_range)
    analyzer.plot_regularization_comparison(save_path='plots/logistic_regression_synthetic.png')
    analyzer.plot_decision_boundaries_2d(save_path='plots/decision_boundaries_synthetic.png')
    
    print("\n✅ REGULARIZED LOGISTIC REGRESSION ANALYSIS COMPLETE!")
    print("📁 Check the 'plots' folder for visualizations.")
    print("🔧 Key demonstrations include:")
    print("   - L1 regularization: automatic feature selection")
    print("   - L2 regularization: coefficient shrinkage")
    print("   - Decision boundary changes with regularization strength")
    print("   - Feature importance and selection patterns")
    print("   - Performance comparison across multiple metrics")

if __name__ == "__main__":
    main() 