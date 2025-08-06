#!/usr/bin/env python3
"""
Comprehensive Multiclass Metrics Analysis
=========================================

This module provides detailed analysis of multiclass classification metrics
including macro/micro F1, per-class precision/recall, ROC-AUC, and calibration.

Key Concepts:
- Macro vs Micro vs Weighted F1 scores
- Per-class precision, recall, and F1-score analysis
- One-vs-Rest ROC-AUC for multiclass problems
- Model calibration and reliability diagrams
- Class imbalance impact on metrics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_digits, load_wine, make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    log_loss, brier_score_loss
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from itertools import cycle
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")

class MulticlassMetricsAnalyzer:
    """
    Comprehensive analyzer for multiclass classification metrics
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results = {}
        
    def load_datasets_with_imbalance(self):
        """Load datasets with different levels of class imbalance"""
        datasets = {}
        
        # 1. Balanced dataset (Iris)
        iris = load_iris()
        datasets['iris_balanced'] = {
            'X': iris.data,
            'y': iris.target,
            'target_names': iris.target_names,
            'imbalance_level': 'Balanced',
            'n_classes': 3
        }
        
        # 2. Slightly imbalanced (Wine)
        wine = load_wine()
        datasets['wine_slight'] = {
            'X': wine.data,
            'y': wine.target,
            'target_names': wine.target_names,
            'imbalance_level': 'Slight',
            'n_classes': 3
        }
        
        # 3. Moderately imbalanced synthetic
        X_mod, y_mod = make_classification(
            n_samples=2000,
            n_features=20,
            n_informative=15,
            n_classes=4,
            weights=[0.4, 0.3, 0.2, 0.1],  # Moderate imbalance
            random_state=self.random_state
        )
        datasets['synthetic_moderate'] = {
            'X': X_mod,
            'y': y_mod,
            'target_names': [f'Class_{i}' for i in range(4)],
            'imbalance_level': 'Moderate',
            'n_classes': 4
        }
        
        # 4. Highly imbalanced synthetic
        X_high, y_high = make_classification(
            n_samples=3000,
            n_features=20,
            n_informative=15,
            n_classes=5,
            weights=[0.5, 0.25, 0.15, 0.07, 0.03],  # High imbalance
            random_state=self.random_state
        )
        datasets['synthetic_high'] = {
            'X': X_high,
            'y': y_high,
            'target_names': [f'Class_{i}' for i in range(5)],
            'imbalance_level': 'High',
            'n_classes': 5
        }
        
        return datasets
    
    def analyze_class_distributions(self, datasets):
        """Analyze and visualize class distributions"""
        print("Class Distribution Analysis")
        print("=" * 40)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        imbalance_metrics = []
        
        for idx, (name, data) in enumerate(datasets.items()):
            y = data['y']
            target_names = data['target_names']
            
            # Calculate class distribution
            unique, counts = np.unique(y, return_counts=True)
            class_dist = dict(zip(unique, counts))
            
            # Calculate imbalance metrics
            max_count = max(counts)
            min_count = min(counts)
            imbalance_ratio = max_count / min_count
            
            # Gini coefficient for imbalance
            sorted_counts = np.sort(counts)
            n = len(counts)
            cumsum = np.cumsum(sorted_counts)
            gini = (2 * np.sum((np.arange(n) + 1) * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n
            
            imbalance_metrics.append({
                'Dataset': name,
                'Imbalance_Level': data['imbalance_level'],
                'Imbalance_Ratio': imbalance_ratio,
                'Gini_Coefficient': gini,
                'Num_Classes': len(unique),
                'Total_Samples': len(y)
            })
            
            print(f"\n{name}:")
            print(f"  Classes: {len(unique)}")
            print(f"  Distribution: {class_dist}")
            print(f"  Imbalance Ratio: {imbalance_ratio:.2f}")
            print(f"  Gini Coefficient: {gini:.3f}")
            
            # Plot distribution
            ax = axes[idx]
            bars = ax.bar(range(len(unique)), counts, alpha=0.7)
            ax.set_title(f'{name.replace("_", " ").title()}\\n{data["imbalance_level"]} Imbalance')
            ax.set_xlabel('Class')
            ax.set_ylabel('Sample Count')
            ax.set_xticks(range(len(unique)))
            ax.set_xticklabels([target_names[i] for i in unique], rotation=45)
            
            # Color bars by frequency
            colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            # Add count labels
            for i, count in enumerate(counts):
                ax.text(i, count + max(counts)*0.02, str(count), 
                       ha='center', va='bottom', fontweight='bold')
            
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/class_distributions.png',
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return pd.DataFrame(imbalance_metrics)
    
    def calculate_comprehensive_metrics(self, y_true, y_pred, y_pred_proba, class_names):
        """Calculate comprehensive multiclass metrics"""
        metrics = {}
        
        # Basic accuracy
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # F1 scores (different averaging methods)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro')
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
        
        # Precision and Recall (different averaging methods)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
        metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro')
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted')
        
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
        metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro')
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted')
        
        # Per-class metrics
        f1_per_class = f1_score(y_true, y_pred, average=None)
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        
        metrics['f1_per_class'] = dict(zip(class_names, f1_per_class))
        metrics['precision_per_class'] = dict(zip(class_names, precision_per_class))
        metrics['recall_per_class'] = dict(zip(class_names, recall_per_class))
        
        # ROC-AUC (One-vs-Rest)
        if y_pred_proba is not None:
            try:
                # Multi-class AUC using One-vs-Rest
                metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_pred_proba, 
                                                      multi_class='ovr', average='macro')
                metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_pred_proba, 
                                                      multi_class='ovo', average='macro')
                
                # Per-class AUC
                lb = LabelBinarizer()
                y_true_binary = lb.fit_transform(y_true)
                if y_true_binary.shape[1] == 1:  # Binary case
                    y_true_binary = np.hstack([1-y_true_binary, y_true_binary])
                
                auc_per_class = []
                for i in range(len(class_names)):
                    if i < y_pred_proba.shape[1]:
                        auc_score = roc_auc_score(y_true_binary[:, i], y_pred_proba[:, i])
                        auc_per_class.append(auc_score)
                    else:
                        auc_per_class.append(np.nan)
                
                metrics['auc_per_class'] = dict(zip(class_names, auc_per_class))
                
                # Log loss
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
                
            except Exception as e:
                print(f"Warning: Could not calculate AUC metrics: {e}")
                metrics['roc_auc_ovr'] = np.nan
                metrics['roc_auc_ovo'] = np.nan
                metrics['auc_per_class'] = {name: np.nan for name in class_names}
                metrics['log_loss'] = np.nan
        
        return metrics
    
    def compare_models_across_imbalance_levels(self, datasets):
        """Compare different models across datasets with varying imbalance"""
        print("\nComparing Models Across Imbalance Levels")
        print("=" * 50)
        
        # Define models to compare
        models = {
            'Logistic Regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            'SVM (RBF)': SVC(probability=True, random_state=self.random_state)
        }
        
        results_list = []
        detailed_results = {}
        
        for dataset_name, data in datasets.items():
            print(f"\nAnalyzing {dataset_name}...")
            X, y = data['X'], data['y']
            class_names = data['target_names']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=self.random_state, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            dataset_results = {}
            
            for model_name, model in models.items():
                print(f"  Training {model_name}...")
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Predictions
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                metrics = self.calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba, class_names)
                
                # Store results
                result_row = {
                    'Dataset': dataset_name,
                    'Imbalance_Level': data['imbalance_level'],
                    'Model': model_name,
                    **{k: v for k, v in metrics.items() if not isinstance(v, dict)}
                }
                results_list.append(result_row)
                
                # Store detailed results for visualization
                dataset_results[model_name] = {
                    'metrics': metrics,
                    'y_test': y_test,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba,
                    'model': model
                }
            
            detailed_results[dataset_name] = {
                'data': data,
                'results': dataset_results
            }
        
        results_df = pd.DataFrame(results_list)
        return results_df, detailed_results
    
    def visualize_metrics_comparison(self, results_df):
        """Visualize comprehensive metrics comparison"""
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # 1. F1 Score Comparison (Macro, Micro, Weighted)
        ax1 = axes[0, 0]
        f1_metrics = ['f1_macro', 'f1_micro', 'f1_weighted']
        f1_data = results_df.melt(
            id_vars=['Dataset', 'Model', 'Imbalance_Level'], 
            value_vars=f1_metrics,
            var_name='F1_Type', value_name='F1_Score'
        )
        
        sns.barplot(data=f1_data, x='Imbalance_Level', y='F1_Score', 
                   hue='F1_Type', ax=ax1, alpha=0.8)
        ax1.set_title('F1 Score Comparison Across Imbalance Levels')
        ax1.set_ylabel('F1 Score')
        ax1.legend(title='F1 Type')
        ax1.grid(True, alpha=0.3)
        
        # 2. Model Performance by Imbalance Level
        ax2 = axes[0, 1]
        sns.boxplot(data=results_df, x='Imbalance_Level', y='f1_macro', 
                   hue='Model', ax=ax2)
        ax2.set_title('Model Performance vs Imbalance Level')
        ax2.set_ylabel('Macro F1 Score')
        ax2.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 3. Accuracy vs F1 Macro
        ax3 = axes[1, 0]
        for model in results_df['Model'].unique():
            model_data = results_df[results_df['Model'] == model]
            ax3.scatter(model_data['accuracy'], model_data['f1_macro'], 
                       label=model, alpha=0.7, s=80)
        
        ax3.set_xlabel('Accuracy')
        ax3.set_ylabel('Macro F1')
        ax3.set_title('Accuracy vs Macro F1 Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add diagonal line for reference
        ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Agreement')
        
        # 4. ROC-AUC Comparison
        ax4 = axes[1, 1]
        auc_data = results_df.dropna(subset=['roc_auc_ovr'])
        if not auc_data.empty:
            sns.barplot(data=auc_data, x='Imbalance_Level', y='roc_auc_ovr', 
                       hue='Model', ax=ax4, alpha=0.8)
            ax4.set_title('ROC-AUC (One-vs-Rest) Comparison')
            ax4.set_ylabel('ROC-AUC Score')
            ax4.legend(title='Model')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'ROC-AUC data not available', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        # 5. Precision vs Recall Trade-off
        ax5 = axes[2, 0]
        precision_data = results_df.melt(
            id_vars=['Dataset', 'Model', 'Imbalance_Level'], 
            value_vars=['precision_macro', 'recall_macro'],
            var_name='Metric', value_name='Score'
        )
        
        sns.scatterplot(data=precision_data[precision_data['Metric'] == 'precision_macro'], 
                       x='Score', y=precision_data[precision_data['Metric'] == 'recall_macro']['Score'].values,
                       hue=precision_data[precision_data['Metric'] == 'precision_macro']['Model'].values,
                       style=precision_data[precision_data['Metric'] == 'precision_macro']['Imbalance_Level'].values,
                       ax=ax5, s=80, alpha=0.7)
        
        ax5.set_xlabel('Precision (Macro)')
        ax5.set_ylabel('Recall (Macro)')
        ax5.set_title('Precision vs Recall Trade-off')
        ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax5.grid(True, alpha=0.3)
        
        # 6. Performance Degradation with Imbalance
        ax6 = axes[2, 1]
        
        # Calculate performance degradation
        balanced_baseline = results_df[results_df['Imbalance_Level'] == 'Balanced']['f1_macro'].mean()
        degradation_data = []
        
        for level in ['Slight', 'Moderate', 'High']:
            level_data = results_df[results_df['Imbalance_Level'] == level]
            if not level_data.empty:
                for model in level_data['Model'].unique():
                    model_score = level_data[level_data['Model'] == model]['f1_macro'].mean()
                    degradation = (balanced_baseline - model_score) / balanced_baseline * 100
                    degradation_data.append({
                        'Imbalance_Level': level,
                        'Model': model,
                        'Performance_Degradation': degradation
                    })
        
        if degradation_data:
            degradation_df = pd.DataFrame(degradation_data)
            sns.barplot(data=degradation_df, x='Imbalance_Level', y='Performance_Degradation', 
                       hue='Model', ax=ax6, alpha=0.8)
            ax6.set_title('Performance Degradation with Imbalance')
            ax6.set_ylabel('Performance Degradation (%)')
            ax6.legend(title='Model')
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/comprehensive_metrics_comparison.png',
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves_multiclass(self, detailed_results):
        """Plot ROC curves for multiclass problems using One-vs-Rest"""
        n_datasets = len(detailed_results)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown'])
        
        for idx, (dataset_name, dataset_info) in enumerate(detailed_results.items()):
            if idx >= 4:  # Only plot first 4 datasets
                break
                
            ax = axes[idx]
            data = dataset_info['data']
            results = dataset_info['results']
            class_names = data['target_names']
            
            # Plot ROC curves for the best performing model
            best_model_name = max(results.keys(), 
                                key=lambda x: results[x]['metrics']['f1_macro'])
            best_result = results[best_model_name]
            
            y_test = best_result['y_test']
            y_pred_proba = best_result['y_pred_proba']
            
            if y_pred_proba is not None:
                # Binarize the output
                lb = LabelBinarizer()
                y_test_binary = lb.fit_transform(y_test)
                if y_test_binary.shape[1] == 1:  # Binary case
                    y_test_binary = np.hstack([1-y_test_binary, y_test_binary])
                
                n_classes = len(class_names)
                
                for i, color in zip(range(n_classes), colors):
                    if i < y_pred_proba.shape[1]:
                        fpr, tpr, _ = roc_curve(y_test_binary[:, i], y_pred_proba[:, i])
                        roc_auc = auc(fpr, tpr)
                        
                        ax.plot(fpr, tpr, color=color, alpha=0.8,
                               label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
                
                ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f'ROC Curves - {dataset_name}\\n{best_model_name}')
                ax.legend(loc="lower right", fontsize=8)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Probability predictions not available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{dataset_name} - No Probabilities')
        
        plt.tight_layout()
        plt.savefig('plots/roc_curves_multiclass.png',
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_metrics_summary_table(self, results_df):
        """Create a comprehensive summary table of metrics"""
        print("\nComprehensive Metrics Summary")
        print("=" * 60)
        
        # Calculate summary statistics
        summary_stats = results_df.groupby(['Imbalance_Level', 'Model']).agg({
            'accuracy': ['mean', 'std'],
            'f1_macro': ['mean', 'std'],
            'f1_micro': ['mean', 'std'],
            'f1_weighted': ['mean', 'std'],
            'precision_macro': ['mean', 'std'],
            'recall_macro': ['mean', 'std'],
            'roc_auc_ovr': ['mean', 'std']
        }).round(3)
        
        print("\nSummary Statistics (Mean ± Std):")
        print(summary_stats)
        
        # Best performing models per imbalance level
        print("\nBest Performing Models per Imbalance Level:")
        print("-" * 50)
        
        for level in results_df['Imbalance_Level'].unique():
            level_data = results_df[results_df['Imbalance_Level'] == level]
            best_model = level_data.loc[level_data['f1_macro'].idxmax()]
            
            print(f"\n{level} Imbalance:")
            print(f"  Best Model: {best_model['Model']}")
            print(f"  Dataset: {best_model['Dataset']}")
            print(f"  Accuracy: {best_model['accuracy']:.3f}")
            print(f"  Macro F1: {best_model['f1_macro']:.3f}")
            print(f"  Micro F1: {best_model['f1_micro']:.3f}")
            if not pd.isna(best_model['roc_auc_ovr']):
                print(f"  ROC-AUC: {best_model['roc_auc_ovr']:.3f}")

def run_comprehensive_metrics_analysis():
    """Run comprehensive multiclass metrics analysis"""
    print("Comprehensive Multiclass Metrics Analysis")
    print("=" * 50)
    
    analyzer = MulticlassMetricsAnalyzer()
    
    # Load datasets with different imbalance levels
    datasets = analyzer.load_datasets_with_imbalance()
    
    # Analyze class distributions
    imbalance_df = analyzer.analyze_class_distributions(datasets)
    print("\nImbalance Analysis:")
    print(imbalance_df)
    
    # Compare models across imbalance levels
    results_df, detailed_results = analyzer.compare_models_across_imbalance_levels(datasets)
    
    # Visualize metrics comparison
    analyzer.visualize_metrics_comparison(results_df)
    
    # Plot ROC curves
    analyzer.plot_roc_curves_multiclass(detailed_results)
    
    # Create summary table
    analyzer.create_metrics_summary_table(results_df)
    
    return results_df, detailed_results, imbalance_df

if __name__ == "__main__":
    results, detailed, imbalance = run_comprehensive_metrics_analysis()
    print("\\nMetrics analysis complete! Check the plots folder for visualizations.")