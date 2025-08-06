#!/usr/bin/env python3
"""
Advanced Confusion Matrix Visualization for Multiclass Classification
====================================================================

This module provides comprehensive confusion matrix visualization tools
including normalization, error highlighting, and statistical analysis.

Key Features:
- Multiple normalization strategies (true, pred, all)
- Error pattern identification and highlighting
- Statistical significance testing
- Interactive confusion matrix analysis
- Per-class performance breakdown
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_digits, load_wine, make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from scipy.stats import chi2_contingency
import itertools
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("viridis")

class ConfusionMatrixAnalyzer:
    """
    Advanced confusion matrix analyzer with multiple visualization options
    """
    
    def __init__(self, figsize=(12, 10)):
        self.figsize = figsize
        self.results = {}
        
    def plot_confusion_matrix(self, cm, class_names, title="Confusion Matrix", 
                            normalize=None, cmap='Blues', figsize=None, 
                            show_values=True, value_format='.2f'):
        """
        Plot confusion matrix with various normalization options
        
        Parameters:
        -----------
        cm : array-like
            Confusion matrix
        class_names : list
            Class labels
        normalize : str or None
            'true', 'pred', 'all', or None
        """
        if figsize is None:
            figsize = self.figsize
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Normalize confusion matrix
        if normalize == 'true':
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title += ' (Normalized by True Class)'
        elif normalize == 'pred':
            cm_norm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
            title += ' (Normalized by Predicted Class)'
        elif normalize == 'all':
            cm_norm = cm.astype('float') / cm.sum()
            title += ' (Normalized by Total)'
        else:
            cm_norm = cm
            title += ' (Absolute Counts)'
        
        # Create heatmap
        if show_values:
            annot = True
            fmt = value_format if normalize else 'd'
        else:
            annot = False
            fmt = ''
        
        sns.heatmap(cm_norm, annot=annot, fmt=fmt, cmap=cmap,
                   xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        # Rotate labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_confusion_matrix_comparison(self, confusion_matrices, class_names, 
                                       model_names, normalize='true'):
        """
        Plot multiple confusion matrices for comparison
        """
        n_models = len(confusion_matrices)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if n_models > 1 else [axes]
        else:
            axes = axes.ravel()
        
        for idx, (cm, model_name) in enumerate(zip(confusion_matrices, model_names)):
            ax = axes[idx] if n_models > 1 else axes[0]
            
            # Normalize confusion matrix
            if normalize == 'true':
                cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                title = f'{model_name}\\n(Normalized by True Class)'
            elif normalize == 'pred':
                cm_norm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
                title = f'{model_name}\\n(Normalized by Predicted Class)'
            elif normalize == 'all':
                cm_norm = cm.astype('float') / cm.sum()
                title = f'{model_name}\\n(Normalized by Total)'
            else:
                cm_norm = cm
                title = f'{model_name}\\n(Absolute Counts)'
            
            # Create heatmap
            sns.heatmap(cm_norm, annot=True, fmt='.3f' if normalize else 'd',
                       cmap='Blues', xticklabels=class_names, yticklabels=class_names,
                       ax=ax, cbar=True)
            
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            
            # Rotate labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            plt.setp(ax.get_yticklabels(), rotation=0)
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        return fig, axes
    
    def highlight_error_patterns(self, cm, class_names, threshold=0.05):
        """
        Identify and highlight common error patterns in confusion matrix
        """
        # Normalize by true class to get error rates
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Find off-diagonal elements above threshold
        error_patterns = []
        n_classes = len(class_names)
        
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j and cm_norm[i, j] > threshold:
                    error_patterns.append({
                        'true_class': class_names[i],
                        'predicted_class': class_names[j],
                        'error_rate': cm_norm[i, j],
                        'count': cm[i, j],
                        'position': (i, j)
                    })
        
        # Sort by error rate
        error_patterns.sort(key=lambda x: x['error_rate'], reverse=True)
        
        # Create visualization with highlighted errors
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Original confusion matrix
        sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax1)
        ax1.set_title('Confusion Matrix (Normalized by True Class)')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        
        # Highlight error patterns
        for pattern in error_patterns:
            i, j = pattern['position']
            # Add red border to highlight errors
            rect = plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=3)
            ax1.add_patch(rect)
        
        # Error patterns bar chart
        if error_patterns:
            error_labels = [f"{p['true_class']} → {p['predicted_class']}" for p in error_patterns[:10]]
            error_rates = [p['error_rate'] for p in error_patterns[:10]]
            
            bars = ax2.barh(range(len(error_rates)), error_rates, color='lightcoral', alpha=0.7)
            ax2.set_yticks(range(len(error_rates)))
            ax2.set_yticklabels(error_labels)
            ax2.set_xlabel('Error Rate')
            ax2.set_title('Top Error Patterns (True → Predicted)')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, rate) in enumerate(zip(bars, error_rates)):
                width = bar.get_width()
                ax2.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{rate:.3f}', ha='left', va='center')
        else:
            ax2.text(0.5, 0.5, 'No significant error patterns found',
                    ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        return fig, error_patterns
    
    def analyze_confusion_matrix_statistics(self, cm, class_names):
        """
        Perform statistical analysis of confusion matrix
        """
        print("Confusion Matrix Statistical Analysis")
        print("=" * 40)
        
        # Basic statistics
        total_samples = np.sum(cm)
        correct_predictions = np.trace(cm)
        accuracy = correct_predictions / total_samples
        
        print(f"Total Samples: {total_samples}")
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Overall Accuracy: {accuracy:.4f}")
        
        # Per-class statistics
        print("\\nPer-Class Statistics:")
        print("-" * 30)
        
        for i, class_name in enumerate(class_names):
            # True positives, false positives, false negatives
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            tn = total_samples - tp - fp - fn
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            print(f"\\n{class_name}:")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  Specificity: {specificity:.4f}")
        
        # Chi-square test for independence
        try:
            chi2, p_value, dof, expected = chi2_contingency(cm)
            print(f"\\nChi-square Test for Independence:")
            print(f"  Chi-square statistic: {chi2:.4f}")
            print(f"  p-value: {p_value:.4e}")
            print(f"  Degrees of freedom: {dof}")
            
            if p_value < 0.05:
                print("  Result: Predictions are significantly different from random (p < 0.05)")
            else:
                print("  Result: Predictions are not significantly different from random (p >= 0.05)")
        except Exception as e:
            print(f"\\nChi-square test failed: {e}")
        
        return {
            'accuracy': accuracy,
            'total_samples': total_samples,
            'correct_predictions': correct_predictions,
            'chi2_statistic': chi2 if 'chi2' in locals() else None,
            'chi2_pvalue': p_value if 'p_value' in locals() else None
        }
    
    def create_detailed_error_analysis(self, cm, class_names, model_name="Model"):
        """
        Create detailed error analysis visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Original confusion matrix
        ax1 = axes[0, 0]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax1)
        ax1.set_title(f'{model_name} - Absolute Counts')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        
        # 2. Normalized by true class (recall)
        ax2 = axes[0, 1]
        cm_recall = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_recall, annot=True, fmt='.3f', cmap='Reds',
                   xticklabels=class_names, yticklabels=class_names, ax=ax2)
        ax2.set_title(f'{model_name} - Recall (True Class Norm.)')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('True')
        
        # 3. Normalized by predicted class (precision)
        ax3 = axes[1, 0]
        cm_precision = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        sns.heatmap(cm_precision, annot=True, fmt='.3f', cmap='Greens',
                   xticklabels=class_names, yticklabels=class_names, ax=ax3)
        ax3.set_title(f'{model_name} - Precision (Pred. Class Norm.)')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('True')
        
        # 4. Error magnitude heatmap
        ax4 = axes[1, 1]
        # Calculate error magnitude (off-diagonal elements)
        error_matrix = cm.copy()
        np.fill_diagonal(error_matrix, 0)  # Remove correct predictions
        
        sns.heatmap(error_matrix, annot=True, fmt='d', cmap='OrRd',
                   xticklabels=class_names, yticklabels=class_names, ax=ax4)
        ax4.set_title(f'{model_name} - Errors Only (Absolute)')
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('True')
        
        plt.tight_layout()
        return fig
    
    def compare_models_confusion_matrices(self, models_results, class_names):
        """
        Compare confusion matrices from multiple models
        """
        model_names = list(models_results.keys())
        confusion_matrices = [results['confusion_matrix'] for results in models_results.values()]
        
        # 1. Side-by-side comparison
        fig1 = self.plot_confusion_matrix_comparison(
            confusion_matrices, class_names, model_names, normalize='true'
        )[0]
        fig1.suptitle('Model Comparison - Confusion Matrices (Normalized by True Class)', 
                     fontsize=16, fontweight='bold')
        
        # 2. Error pattern analysis for each model
        fig2, axes = plt.subplots(len(model_names), 2, figsize=(12, 4*len(model_names)))
        if len(model_names) == 1:
            axes = axes.reshape(1, -1)
        
        all_error_patterns = {}
        
        for idx, (model_name, results) in enumerate(models_results.items()):
            cm = results['confusion_matrix']
            
            # Highlight error patterns
            _, error_patterns = self.highlight_error_patterns(cm, class_names)
            all_error_patterns[model_name] = error_patterns
            
            # Plot confusion matrix with highlights
            ax1 = axes[idx, 0] if len(model_names) > 1 else axes[0]
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names, ax=ax1)
            ax1.set_title(f'{model_name} - Error Patterns Highlighted')
            ax1.set_xlabel('Predicted')
            ax1.set_ylabel('True')
            
            # Highlight significant errors
            for pattern in error_patterns[:5]:  # Top 5 errors
                i, j = pattern['position']
                rect = plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=2)
                ax1.add_patch(rect)
            
            # Plot error patterns
            ax2 = axes[idx, 1] if len(model_names) > 1 else axes[1]
            if error_patterns:
                error_labels = [f"{p['true_class']} → {p['predicted_class']}" 
                              for p in error_patterns[:8]]
                error_rates = [p['error_rate'] for p in error_patterns[:8]]
                
                bars = ax2.barh(range(len(error_rates)), error_rates, 
                               color='lightcoral', alpha=0.7)
                ax2.set_yticks(range(len(error_rates)))
                ax2.set_yticklabels(error_labels, fontsize=8)
                ax2.set_xlabel('Error Rate')
                ax2.set_title(f'{model_name} - Top Error Patterns')
                ax2.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, rate in zip(bars, error_rates):
                    width = bar.get_width()
                    ax2.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                            f'{rate:.3f}', ha='left', va='center', fontsize=8)
            else:
                ax2.text(0.5, 0.5, 'No significant error patterns',
                        ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        
        return fig1, fig2, all_error_patterns

def demonstrate_confusion_matrix_analysis():
    """
    Demonstrate comprehensive confusion matrix analysis
    """
    print("Comprehensive Confusion Matrix Analysis Demo")
    print("=" * 50)
    
    analyzer = ConfusionMatrixAnalyzer()
    
    # Load dataset (using Iris for simplicity)
    iris = load_iris()
    X, y = iris.data, iris.target
    class_names = iris.target_names
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train multiple models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42)
    }
    
    models_results = {}
    
    print("\nTraining models and generating confusion matrices...")
    
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        models_results[model_name] = {
            'model': model,
            'y_pred': y_pred,
            'confusion_matrix': cm
        }
        
        # Statistical analysis
        print(f"\nStatistical Analysis for {model_name}:")
        stats = analyzer.analyze_confusion_matrix_statistics(cm, class_names)
    
    # 1. Individual detailed analysis for best model
    best_model = max(models_results.keys(), 
                    key=lambda x: np.trace(models_results[x]['confusion_matrix']))
    
    print(f"\nCreating detailed analysis for best model: {best_model}")
    
    best_cm = models_results[best_model]['confusion_matrix']
    
    # Detailed error analysis
    fig1 = analyzer.create_detailed_error_analysis(best_cm, class_names, best_model)
    fig1.savefig('plots/detailed_confusion_analysis.png',
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Error pattern highlighting
    fig2, error_patterns = analyzer.highlight_error_patterns(best_cm, class_names)
    fig2.savefig('plots/error_patterns_highlighted.png',
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Model comparison
    print("\nComparing confusion matrices across models...")
    
    fig3, fig4, all_error_patterns = analyzer.compare_models_confusion_matrices(
        models_results, class_names
    )
    
    fig3.savefig('plots/confusion_matrices_comparison.png',
                dpi=300, bbox_inches='tight')
    fig4.savefig('plots/error_patterns_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Summary of error patterns across models
    print("\nError Pattern Summary Across Models:")
    print("=" * 40)
    
    for model_name, patterns in all_error_patterns.items():
        print(f"\n{model_name}:")
        if patterns:
            for i, pattern in enumerate(patterns[:3], 1):
                print(f"  {i}. {pattern['true_class']} → {pattern['predicted_class']}: "
                      f"{pattern['error_rate']:.3f} ({pattern['count']} cases)")
        else:
            print("  No significant error patterns found")
    
    return models_results, all_error_patterns

if __name__ == "__main__":
    results, error_patterns = demonstrate_confusion_matrix_analysis()
    print("\nConfusion matrix analysis complete! Check the plots folder for visualizations.")