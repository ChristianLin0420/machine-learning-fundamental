#!/usr/bin/env python3
"""
Multiclass Logistic Regression with Scikit-Learn
================================================

This module demonstrates multiclass classification using logistic regression
with different strategies (One-vs-Rest, multinomial) on various datasets.

Key Concepts:
- One-vs-Rest (OvR) vs Multinomial logistic regression
- Handling different dataset types and preprocessing
- Comprehensive evaluation with multiple metrics
- Class imbalance detection and handling
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_digits, load_wine, make_classification
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve, auc
)
from sklearn.multiclass import OneVsRestClassifier
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MulticlassLogisticAnalyzer:
    """
    Comprehensive analyzer for multiclass logistic regression
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results = {}
        
    def load_datasets(self):
        """Load and prepare multiple datasets for multiclass classification"""
        datasets = {}
        
        # 1. Iris dataset (3 classes, 4 features)
        iris = load_iris()
        datasets['iris'] = {
            'X': iris.data,
            'y': iris.target,
            'target_names': iris.target_names,
            'feature_names': iris.feature_names,
            'n_classes': 3,
            'description': 'Iris flowers: 3 species, 4 features (petal/sepal measurements)'
        }
        
        # 2. Wine dataset (3 classes, 13 features)
        wine = load_wine()
        datasets['wine'] = {
            'X': wine.data,
            'y': wine.target,
            'target_names': wine.target_names,
            'feature_names': wine.feature_names,
            'n_classes': 3,
            'description': 'Wine types: 3 classes, 13 chemical features'
        }
        
        # 3. Digits dataset (10 classes, 64 features)
        digits = load_digits()
        datasets['digits'] = {
            'X': digits.data,
            'y': digits.target,
            'target_names': [str(i) for i in range(10)],
            'feature_names': [f'pixel_{i}' for i in range(64)],
            'n_classes': 10,
            'description': 'Handwritten digits: 10 classes, 8x8 pixel images'
        }
        
        # 4. Synthetic imbalanced dataset
        X_synth, y_synth = make_classification(
            n_samples=2000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=5,
            n_clusters_per_class=1,
            weights=[0.5, 0.25, 0.15, 0.07, 0.03],  # Imbalanced
            random_state=self.random_state
        )
        datasets['synthetic_imbalanced'] = {
            'X': X_synth,
            'y': y_synth,
            'target_names': [f'Class_{i}' for i in range(5)],
            'feature_names': [f'feature_{i}' for i in range(20)],
            'n_classes': 5,
            'description': 'Synthetic imbalanced: 5 classes with varying sample sizes'
        }
        
        return datasets
    
    def analyze_dataset_properties(self, datasets):
        """Analyze and visualize dataset properties"""
        print("Dataset Properties Analysis")
        print("=" * 60)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for idx, (name, data) in enumerate(datasets.items()):
            if idx >= 4:  # Only plot first 4 datasets
                break
                
            X, y = data['X'], data['y']
            
            # Basic statistics
            print(f"\n{name.upper()} Dataset:")
            print(f"  Shape: {X.shape}")
            print(f"  Classes: {data['n_classes']}")
            print(f"  Features: {len(data['feature_names'])}")
            print(f"  Description: {data['description']}")
            
            # Class distribution
            unique, counts = np.unique(y, return_counts=True)
            class_dist = dict(zip(unique, counts))
            print(f"  Class distribution: {class_dist}")
            
            # Check for class imbalance
            imbalance_ratio = max(counts) / min(counts)
            print(f"  Imbalance ratio: {imbalance_ratio:.2f}")
            
            # Plot class distribution
            ax = axes[idx]
            bars = ax.bar(range(len(unique)), counts, alpha=0.7)
            ax.set_title(f'{name.title()} Class Distribution')
            ax.set_xlabel('Class')
            ax.set_ylabel('Sample Count')
            ax.set_xticks(range(len(unique)))
            ax.set_xticklabels([data['target_names'][i] for i in unique], rotation=45)
            
            # Color bars by imbalance
            colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            # Add count labels on bars
            for i, count in enumerate(counts):
                ax.text(i, count + max(counts)*0.01, str(count), 
                       ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('plots/dataset_properties.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return datasets
    
    def compare_multiclass_strategies(self, X, y, dataset_name):
        """Compare One-vs-Rest vs Multinomial logistic regression"""
        print(f"\nComparing Multiclass Strategies on {dataset_name}")
        print("-" * 50)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        strategies = {
            'One-vs-Rest': LogisticRegression(
                multi_class='ovr', solver='liblinear', 
                random_state=self.random_state, max_iter=1000
            ),
            'Multinomial': LogisticRegression(
                multi_class='multinomial', solver='lbfgs',
                random_state=self.random_state, max_iter=1000
            )
        }
        
        results = {}
        
        for strategy_name, model in strategies.items():
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            macro_f1 = f1_score(y_test, y_pred, average='macro')
            micro_f1 = f1_score(y_test, y_pred, average='micro')
            weighted_f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                      cv=StratifiedKFold(n_splits=5, shuffle=True, 
                                                       random_state=self.random_state),
                                      scoring='f1_macro')
            
            results[strategy_name] = {
                'model': model,
                'accuracy': accuracy,
                'macro_f1': macro_f1,
                'micro_f1': micro_f1,
                'weighted_f1': weighted_f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'y_test': y_test
            }
            
            print(f"{strategy_name}:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Macro F1: {macro_f1:.4f}")
            print(f"  Micro F1: {micro_f1:.4f}")
            print(f"  Weighted F1: {weighted_f1:.4f}")
            print(f"  CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return results, (X_test_scaled, y_test)
    
    def evaluate_comprehensive_metrics(self, results, dataset_name, class_names):
        """Comprehensive evaluation with multiple metrics"""
        print(f"\nComprehensive Metrics Analysis for {dataset_name}")
        print("=" * 60)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Performance comparison
        ax1 = axes[0, 0]
        metrics = ['accuracy', 'macro_f1', 'micro_f1', 'weighted_f1']
        strategies = list(results.keys())
        
        x = np.arange(len(metrics))
        width = 0.35
        
        for i, strategy in enumerate(strategies):
            values = [results[strategy][metric] for metric in metrics]
            ax1.bar(x + i*width, values, width, label=strategy, alpha=0.8)
        
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Score')
        ax1.set_title(f'Performance Comparison - {dataset_name}')
        ax1.set_xticks(x + width/2)
        ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Cross-validation comparison
        ax2 = axes[0, 1]
        cv_means = [results[s]['cv_mean'] for s in strategies]
        cv_stds = [results[s]['cv_std'] for s in strategies]
        
        bars = ax2.bar(strategies, cv_means, yerr=cv_stds, capsize=5, alpha=0.8)
        ax2.set_ylabel('Cross-Validation F1-Macro Score')
        ax2.set_title('Cross-Validation Performance')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars, cv_means, cv_stds):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.001,
                    f'{mean:.3f}±{std:.3f}', ha='center', va='bottom')
        
        # 3. Confusion matrices
        for i, (strategy, result) in enumerate(results.items()):
            ax = axes[1, i]
            cm = confusion_matrix(result['y_test'], result['y_pred'])
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names, ax=ax)
            ax.set_title(f'Confusion Matrix - {strategy}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(f'plots/{dataset_name}_comprehensive_analysis.png',
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Detailed classification reports
        for strategy, result in results.items():
            print(f"\nDetailed Classification Report - {strategy}:")
            print(classification_report(result['y_test'], result['y_pred'],
                                      target_names=class_names))
    
    def analyze_feature_importance(self, model, feature_names, dataset_name, strategy_name):
        """Analyze and visualize feature importance for logistic regression"""
        if hasattr(model, 'coef_'):
            n_classes, n_features = model.coef_.shape
            
            if n_classes == 1:  # Binary case (though we're doing multiclass)
                return
            
            fig, axes = plt.subplots(1, min(n_classes, 3), figsize=(15, 5))
            if n_classes == 1:
                axes = [axes]
            elif n_classes == 2:
                axes = axes if isinstance(axes, list) else [axes]
            
            for class_idx in range(min(n_classes, 3)):
                ax = axes[class_idx] if n_classes > 1 else axes[0]
                
                # Get coefficients for this class
                coef = model.coef_[class_idx]
                
                # Sort by absolute value
                sorted_idx = np.argsort(np.abs(coef))[::-1]
                top_features = min(10, len(coef))  # Top 10 features
                
                # Plot top features
                top_coef = coef[sorted_idx[:top_features]]
                top_names = [feature_names[i] for i in sorted_idx[:top_features]]
                
                colors = ['red' if c < 0 else 'blue' for c in top_coef]
                bars = ax.barh(range(top_features), top_coef, color=colors, alpha=0.7)
                
                ax.set_yticks(range(top_features))
                ax.set_yticklabels(top_names)
                ax.set_xlabel('Coefficient Value')
                ax.set_title(f'Class {class_idx} - Top Features\n{strategy_name}')
                ax.grid(True, alpha=0.3)
                
                # Add coefficient values
                for i, (bar, coef_val) in enumerate(zip(bars, top_coef)):
                    width = bar.get_width()
                    ax.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                           f'{coef_val:.3f}', ha='left' if width >= 0 else 'right', va='center')
            
            plt.tight_layout()
            plt.savefig(f'plots/{dataset_name}_{strategy_name.lower().replace("-", "_")}_feature_importance.png',
                       dpi=300, bbox_inches='tight')
            plt.show()

def run_comprehensive_analysis():
    """Run comprehensive multiclass logistic regression analysis"""
    print("Multiclass Logistic Regression Comprehensive Analysis")
    print("=" * 60)
    
    analyzer = MulticlassLogisticAnalyzer()
    
    # Load datasets
    datasets = analyzer.load_datasets()
    datasets = analyzer.analyze_dataset_properties(datasets)
    
    # Analyze each dataset
    all_results = {}
    
    for dataset_name, data in datasets.items():
        print(f"\n{'='*60}")
        print(f"ANALYZING DATASET: {dataset_name.upper()}")
        print(f"{'='*60}")
        
        X, y = data['X'], data['y']
        class_names = data['target_names']
        feature_names = data['feature_names']
        
        # Compare strategies
        results, test_data = analyzer.compare_multiclass_strategies(X, y, dataset_name)
        
        # Comprehensive evaluation
        analyzer.evaluate_comprehensive_metrics(results, dataset_name, class_names)
        
        # Feature importance analysis
        for strategy_name, result in results.items():
            analyzer.analyze_feature_importance(
                result['model'], feature_names, dataset_name, strategy_name
            )
        
        all_results[dataset_name] = results
    
    # Summary comparison across datasets
    print("\n" + "="*60)
    print("SUMMARY COMPARISON ACROSS ALL DATASETS")
    print("="*60)
    
    summary_data = []
    for dataset_name, results in all_results.items():
        for strategy_name, result in results.items():
            summary_data.append({
                'Dataset': dataset_name,
                'Strategy': strategy_name,
                'Accuracy': result['accuracy'],
                'Macro F1': result['macro_f1'],
                'Micro F1': result['micro_f1'],
                'CV Mean': result['cv_mean'],
                'CV Std': result['cv_std']
            })
    
    summary_df = pd.DataFrame(summary_data)
    print("\nSummary Results:")
    print(summary_df.round(4))
    
    # Create summary visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Accuracy comparison
    ax1 = axes[0, 0]
    summary_pivot = summary_df.pivot(index='Dataset', columns='Strategy', values='Accuracy')
    summary_pivot.plot(kind='bar', ax=ax1, alpha=0.8)
    ax1.set_title('Accuracy Comparison Across Datasets')
    ax1.set_ylabel('Accuracy')
    ax1.legend(title='Strategy')
    ax1.grid(True, alpha=0.3)
    
    # Macro F1 comparison
    ax2 = axes[0, 1]
    summary_pivot_f1 = summary_df.pivot(index='Dataset', columns='Strategy', values='Macro F1')
    summary_pivot_f1.plot(kind='bar', ax=ax2, alpha=0.8)
    ax2.set_title('Macro F1 Comparison Across Datasets')
    ax2.set_ylabel('Macro F1 Score')
    ax2.legend(title='Strategy')
    ax2.grid(True, alpha=0.3)
    
    # CV performance with error bars
    ax3 = axes[1, 0]
    for strategy in ['One-vs-Rest', 'Multinomial']:
        strategy_data = summary_df[summary_df['Strategy'] == strategy]
        x_pos = np.arange(len(strategy_data))
        ax3.errorbar(x_pos, strategy_data['CV Mean'], yerr=strategy_data['CV Std'],
                    label=strategy, marker='o', capsize=5)
    
    ax3.set_xlabel('Dataset')
    ax3.set_ylabel('Cross-Validation F1-Macro')
    ax3.set_title('Cross-Validation Performance Comparison')
    ax3.set_xticks(range(len(summary_df['Dataset'].unique())))
    ax3.set_xticklabels(summary_df['Dataset'].unique(), rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Performance difference
    ax4 = axes[1, 1]
    ovr_scores = summary_df[summary_df['Strategy'] == 'One-vs-Rest']['Macro F1'].values
    mult_scores = summary_df[summary_df['Strategy'] == 'Multinomial']['Macro F1'].values
    differences = mult_scores - ovr_scores
    datasets_list = summary_df['Dataset'].unique()
    
    colors = ['green' if d > 0 else 'red' for d in differences]
    bars = ax4.bar(range(len(differences)), differences, color=colors, alpha=0.7)
    ax4.set_xlabel('Dataset')
    ax4.set_ylabel('Multinomial - One-vs-Rest (F1)')
    ax4.set_title('Strategy Performance Difference')
    ax4.set_xticks(range(len(datasets_list)))
    ax4.set_xticklabels(datasets_list, rotation=45)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/summary_comparison.png',
               dpi=300, bbox_inches='tight')
    plt.show()
    
    return all_results, summary_df

if __name__ == "__main__":
    results, summary = run_comprehensive_analysis()
    print("\nAnalysis complete! Check the plots folder for visualizations.")