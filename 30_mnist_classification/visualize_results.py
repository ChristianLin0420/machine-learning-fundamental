"""
MNIST CNN Results Visualization
===============================

This module provides comprehensive visualization tools for MNIST CNN results,
including confusion matrices, misclassified examples, and prediction analysis.

Key Features:
- Confusion matrix visualization
- Misclassified examples display
- Prediction confidence analysis
- Sample predictions with uncertainty
- Model comparison visualizations

Author: Assistant
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch
import tensorflow as tf
from typing import Dict, List, Tuple, Any, Optional
import os

class MNISTVisualizer:
    """
    Comprehensive visualization tools for MNIST CNN results.
    """
    
    def __init__(self):
        """Initialize the visualizer."""
        self.class_names = [str(i) for i in range(10)]
        
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            title: str = "Confusion Matrix", 
                            figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot confusion matrix with detailed annotations.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title
            figsize: Figure size
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=self.class_names, yticklabels=self.class_names)
        ax1.set_title(f'{title} - Raw Counts')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Label')
        
        # Normalized percentages
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=ax2,
                   xticklabels=self.class_names, yticklabels=self.class_names)
        ax2.set_title(f'{title} - Normalized')
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Label')
        
        plt.tight_layout()
        plt.savefig(f'plots/{title.lower().replace(" ", "_")}_confusion_matrix.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print classification insights
        self._print_confusion_insights(cm, cm_normalized)
    
    def _print_confusion_insights(self, cm: np.ndarray, cm_normalized: np.ndarray) -> None:
        """Print insights from confusion matrix analysis."""
        print("\nConfusion Matrix Insights:")
        print("=" * 40)
        
        # Overall accuracy
        accuracy = np.trace(cm) / np.sum(cm)
        print(f"Overall Accuracy: {accuracy:.4f}")
        
        # Per-class accuracy
        per_class_acc = np.diag(cm) / np.sum(cm, axis=1)
        print("\nPer-class Accuracy:")
        for i, acc in enumerate(per_class_acc):
            print(f"  Digit {i}: {acc:.4f}")
        
        # Most confused pairs
        np.fill_diagonal(cm_normalized, 0)  # Remove diagonal for confusion analysis
        most_confused = np.unravel_index(np.argmax(cm_normalized), cm_normalized.shape)
        print(f"\nMost confused pair: {most_confused[0]} → {most_confused[1]} "
              f"({cm_normalized[most_confused]:.2%})")
        
        # Find top 3 confusion pairs
        flat_indices = np.argsort(cm_normalized.ravel())[-3:]
        top_confusions = [np.unravel_index(idx, cm_normalized.shape) for idx in flat_indices]
        
        print("\nTop 3 confusion pairs:")
        for i, (true_class, pred_class) in enumerate(reversed(top_confusions)):
            conf_rate = cm_normalized[true_class, pred_class]
            print(f"  {i+1}. Digit {true_class} → {pred_class}: {conf_rate:.2%}")
    
    def visualize_misclassified_examples(self, X_test: np.ndarray, y_true: np.ndarray, 
                                       y_pred: np.ndarray, y_proba: np.ndarray,
                                       num_examples: int = 20, figsize: Tuple[int, int] = (20, 12)) -> None:
        """
        Visualize misclassified examples with prediction probabilities.
        
        Args:
            X_test: Test images
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            num_examples: Number of examples to show
            figsize: Figure size
        """
        # Find misclassified examples
        misclassified_mask = y_pred != y_true
        misclassified_indices = np.where(misclassified_mask)[0]
        
        if len(misclassified_indices) == 0:
            print("No misclassified examples found!")
            return
        
        # Sort by confidence (most confident wrong predictions first)
        confidences = np.max(y_proba[misclassified_indices], axis=1)
        sorted_indices = misclassified_indices[np.argsort(confidences)[::-1]]
        
        # Select examples to display
        num_examples = min(num_examples, len(sorted_indices))
        selected_indices = sorted_indices[:num_examples]
        
        # Create subplot grid
        cols = 5
        rows = (num_examples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx in range(num_examples):
            row, col = idx // cols, idx % cols
            img_idx = selected_indices[idx]
            
            # Get image (handle different input formats)
            if X_test.ndim == 4:
                if X_test.shape[-1] == 1:  # Keras format (H, W, 1)
                    img = X_test[img_idx].squeeze()
                else:  # PyTorch format (1, H, W)
                    img = X_test[img_idx].squeeze()
            else:
                img = X_test[img_idx]
            
            axes[row, col].imshow(img, cmap='gray')
            
            true_label = y_true[img_idx]
            pred_label = y_pred[img_idx]
            confidence = np.max(y_proba[img_idx])
            
            # Color based on confidence (red for high confidence errors)
            color = 'red' if confidence > 0.8 else 'orange'
            
            axes[row, col].set_title(
                f'True: {true_label}, Pred: {pred_label}\nConf: {confidence:.3f}',
                color=color, fontsize=10
            )
            axes[row, col].axis('off')
        
        # Hide empty subplots
        for idx in range(num_examples, rows * cols):
            row, col = idx // cols, idx % cols
            axes[row, col].axis('off')
        
        plt.suptitle(f'Top {num_examples} Misclassified Examples (Most Confident Errors)', 
                    fontsize=16)
        plt.tight_layout()
        plt.savefig('plots/misclassified_examples.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nShowing {num_examples} misclassified examples out of {len(misclassified_indices)} total errors")
    
    def visualize_prediction_confidence(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      y_proba: np.ndarray, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Analyze and visualize prediction confidence patterns.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            figsize: Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Confidence distribution for correct vs incorrect predictions
        correct_mask = y_pred == y_true
        correct_confidences = np.max(y_proba[correct_mask], axis=1)
        incorrect_confidences = np.max(y_proba[~correct_mask], axis=1)
        
        axes[0, 0].hist(correct_confidences, bins=50, alpha=0.7, label='Correct', color='green')
        axes[0, 0].hist(incorrect_confidences, bins=50, alpha=0.7, label='Incorrect', color='red')
        axes[0, 0].set_xlabel('Prediction Confidence')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Confidence Distribution: Correct vs Incorrect')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Per-class confidence analysis
        class_confidences = []
        class_accuracies = []
        
        for class_idx in range(10):
            class_mask = y_true == class_idx
            if np.sum(class_mask) > 0:
                class_proba = y_proba[class_mask]
                class_pred = y_pred[class_mask]
                
                # Average confidence for this class
                avg_confidence = np.mean(np.max(class_proba, axis=1))
                class_confidences.append(avg_confidence)
                
                # Accuracy for this class
                class_accuracy = np.mean(class_pred == class_idx)
                class_accuracies.append(class_accuracy)
            else:
                class_confidences.append(0)
                class_accuracies.append(0)
        
        x_pos = np.arange(10)
        axes[0, 1].bar(x_pos, class_confidences, alpha=0.7, color='skyblue')
        axes[0, 1].set_xlabel('Digit Class')
        axes[0, 1].set_ylabel('Average Confidence')
        axes[0, 1].set_title('Average Prediction Confidence by Class')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(self.class_names)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Confidence vs Accuracy scatter plot
        confidence_bins = np.linspace(0, 1, 11)
        bin_accuracies = []
        bin_centers = []
        
        for i in range(len(confidence_bins) - 1):
            bin_mask = (np.max(y_proba, axis=1) >= confidence_bins[i]) & \
                      (np.max(y_proba, axis=1) < confidence_bins[i + 1])
            if np.sum(bin_mask) > 0:
                bin_accuracy = np.mean(y_pred[bin_mask] == y_true[bin_mask])
                bin_accuracies.append(bin_accuracy)
                bin_centers.append((confidence_bins[i] + confidence_bins[i + 1]) / 2)
        
        axes[1, 0].plot(bin_centers, bin_accuracies, 'bo-', linewidth=2, markersize=8)
        axes[1, 0].plot([0, 1], [0, 1], 'r--', alpha=0.8, label='Perfect Calibration')
        axes[1, 0].set_xlabel('Prediction Confidence')
        axes[1, 0].set_ylabel('Actual Accuracy')
        axes[1, 0].set_title('Calibration Plot: Confidence vs Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_ylim(0, 1)
        
        # 4. Uncertainty analysis (entropy of predictions)
        entropies = -np.sum(y_proba * np.log(y_proba + 1e-8), axis=1)
        
        axes[1, 1].scatter(entropies[correct_mask], correct_confidences, 
                          alpha=0.6, s=10, label='Correct', color='green')
        axes[1, 1].scatter(entropies[~correct_mask], incorrect_confidences, 
                          alpha=0.6, s=10, label='Incorrect', color='red')
        axes[1, 1].set_xlabel('Prediction Entropy (Uncertainty)')
        axes[1, 1].set_ylabel('Max Prediction Probability')
        axes[1, 1].set_title('Confidence vs Uncertainty')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/prediction_confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print confidence statistics
        print("\nPrediction Confidence Analysis:")
        print("=" * 40)
        print(f"Average confidence (correct): {np.mean(correct_confidences):.4f}")
        print(f"Average confidence (incorrect): {np.mean(incorrect_confidences):.4f}")
        print(f"High confidence errors (>0.9): {np.sum(incorrect_confidences > 0.9)}")
        print(f"Low confidence correct (>0.5): {np.sum(correct_confidences < 0.5)}")
    
    def visualize_sample_predictions(self, X_test: np.ndarray, y_true: np.ndarray, 
                                   y_pred: np.ndarray, y_proba: np.ndarray,
                                   num_samples: int = 16, figsize: Tuple[int, int] = (16, 12)) -> None:
        """
        Visualize random sample predictions with probability distributions.
        
        Args:
            X_test: Test images
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            num_samples: Number of samples to show
            figsize: Figure size
        """
        # Select random samples
        indices = np.random.choice(len(X_test), num_samples, replace=False)
        
        # Create subplot grid
        cols = 4
        rows = num_samples // cols
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        for i, idx in enumerate(indices):
            row, col = i // cols, i % cols
            
            # Get image (handle different input formats)
            if X_test.ndim == 4:
                if X_test.shape[-1] == 1:  # Keras format (H, W, 1)
                    img = X_test[idx].squeeze()
                else:  # PyTorch format (1, H, W)
                    img = X_test[idx].squeeze()
            else:
                img = X_test[idx]
            
            # Create subplot with image and probability bar
            ax_img = plt.subplot2grid((rows, cols), (row, col))
            
            ax_img.imshow(img, cmap='gray')
            
            true_label = y_true[idx]
            pred_label = y_pred[idx]
            probabilities = y_proba[idx]
            
            # Color based on correctness
            color = 'green' if true_label == pred_label else 'red'
            
            ax_img.set_title(f'True: {true_label}, Pred: {pred_label}', 
                           color=color, fontsize=10)
            ax_img.axis('off')
            
            # Add small probability distribution
            prob_text = f"P({pred_label}): {probabilities[pred_label]:.3f}"
            if true_label != pred_label:
                prob_text += f"\nP({true_label}): {probabilities[true_label]:.3f}"
            
            ax_img.text(0.02, 0.98, prob_text, transform=ax_img.transAxes, 
                       fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle(f'Sample Predictions with Probabilities', fontsize=16)
        plt.tight_layout()
        plt.savefig('plots/sample_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_model_predictions(self, keras_results: Dict[str, Any], 
                                pytorch_results: Dict[str, Any],
                                X_test: np.ndarray, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Compare predictions between Keras and PyTorch models.
        
        Args:
            keras_results: Results from Keras model evaluation
            pytorch_results: Results from PyTorch model evaluation
            X_test: Test images
            figsize: Figure size
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        keras_pred = keras_results['predictions']
        pytorch_pred = pytorch_results['predictions']
        keras_acc = keras_results['test_accuracy']
        pytorch_acc = pytorch_results['test_accuracy']
        
        # 1. Accuracy comparison
        models = ['Keras CNN', 'PyTorch CNN']
        accuracies = [keras_acc, pytorch_acc]
        
        bars = axes[0, 0].bar(models, accuracies, color=['blue', 'orange'], alpha=0.7)
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Test Accuracy')
        axes[0, 0].set_ylim(0.95, 1.0)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                          f'{acc:.4f}', ha='center', va='bottom')
        
        # 2. Agreement analysis
        agreement = keras_pred == pytorch_pred
        agreement_rate = np.mean(agreement)
        
        labels = ['Agree', 'Disagree']
        sizes = [agreement_rate, 1 - agreement_rate]
        colors = ['lightgreen', 'lightcoral']
        
        axes[0, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.2f%%')
        axes[0, 1].set_title(f'Model Agreement\n({agreement_rate:.2%})')
        
        # 3. Disagreement analysis
        if np.sum(~agreement) > 0:
            disagreement_indices = np.where(~agreement)[0]
            
            # Show some disagreement examples
            num_examples = min(8, len(disagreement_indices))
            selected_indices = disagreement_indices[:num_examples]
            
            for i, idx in enumerate(selected_indices):
                if i >= 8:  # Limit to 8 examples
                    break
                    
                row, col = (i // 4) + 1, i % 4
                if row >= 2:  # Only use bottom row
                    continue
                    
                # Get image
                if X_test.ndim == 4:
                    if X_test.shape[-1] == 1:
                        img = X_test[idx].squeeze()
                    else:
                        img = X_test[idx].squeeze()
                else:
                    img = X_test[idx]
                
                ax_pos = (1, col) if i < 4 else (1, col)
                if col < 2:  # Only show first 2 in bottom row for space
                    axes[ax_pos].imshow(img, cmap='gray')
                    axes[ax_pos].set_title(f'K: {keras_pred[idx]}, P: {pytorch_pred[idx]}', 
                                         fontsize=10)
                    axes[ax_pos].axis('off')
        
        # 4. Performance metrics comparison
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        # Calculate metrics from classification reports
        keras_metrics = [
            keras_results['classification_report']['accuracy'],
            keras_results['classification_report']['macro avg']['precision'],
            keras_results['classification_report']['macro avg']['recall'],
            keras_results['classification_report']['macro avg']['f1-score']
        ]
        
        pytorch_metrics = [
            pytorch_results['classification_report']['accuracy'],
            pytorch_results['classification_report']['macro avg']['precision'],
            pytorch_results['classification_report']['macro avg']['recall'],
            pytorch_results['classification_report']['macro avg']['f1-score']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        if len(axes[1]) > 3:  # Only if we have enough subplots
            axes[1, 3].bar(x - width/2, keras_metrics, width, label='Keras', 
                          color='blue', alpha=0.7)
            axes[1, 3].bar(x + width/2, pytorch_metrics, width, label='PyTorch', 
                          color='orange', alpha=0.7)
            
            axes[1, 3].set_xlabel('Metrics')
            axes[1, 3].set_ylabel('Score')
            axes[1, 3].set_title('Detailed Metrics Comparison')
            axes[1, 3].set_xticks(x)
            axes[1, 3].set_xticklabels(metrics, rotation=45)
            axes[1, 3].legend()
            axes[1, 3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nModel Comparison Results:")
        print(f"Keras CNN Accuracy: {keras_acc:.4f}")
        print(f"PyTorch CNN Accuracy: {pytorch_acc:.4f}")
        print(f"Model Agreement: {agreement_rate:.2%}")
        print(f"Disagreements: {np.sum(~agreement)} out of {len(agreement)} predictions")


def demonstrate_visualization_capabilities():
    """
    Demonstrate all visualization capabilities with sample data.
    """
    print("MNIST CNN Visualization Demonstration")
    print("=" * 50)
    
    # Create output directory
    os.makedirs('plots', exist_ok=True)
    
    # Initialize visualizer
    visualizer = MNISTVisualizer()
    
    # Generate sample data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate MNIST-like data
    X_sample = np.random.rand(n_samples, 28, 28)
    y_true = np.random.randint(0, 10, n_samples)
    
    # Simulate predictions (with some errors)
    y_pred = y_true.copy()
    error_indices = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    y_pred[error_indices] = np.random.randint(0, 10, len(error_indices))
    
    # Simulate prediction probabilities
    y_proba = np.random.rand(n_samples, 10)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
    
    # Make predictions more confident for correct predictions
    for i in range(n_samples):
        if y_pred[i] == y_true[i]:
            y_proba[i, y_true[i]] = np.random.uniform(0.8, 0.99)
            remaining_prob = 1 - y_proba[i, y_true[i]]
            other_indices = [j for j in range(10) if j != y_true[i]]
            y_proba[i, other_indices] = np.random.dirichlet(np.ones(9)) * remaining_prob
    
    print("Demonstrating visualization capabilities...")
    
    # 1. Confusion Matrix
    print("\n1. Plotting confusion matrix...")
    visualizer.plot_confusion_matrix(y_true, y_pred, "Sample MNIST Results")
    
    # 2. Misclassified Examples
    print("\n2. Visualizing misclassified examples...")
    visualizer.visualize_misclassified_examples(X_sample, y_true, y_pred, y_proba, num_examples=12)
    
    # 3. Prediction Confidence Analysis
    print("\n3. Analyzing prediction confidence...")
    visualizer.visualize_prediction_confidence(y_true, y_pred, y_proba)
    
    # 4. Sample Predictions
    print("\n4. Showing sample predictions...")
    visualizer.visualize_sample_predictions(X_sample, y_true, y_pred, y_proba, num_samples=8)
    
    print("\n" + "=" * 50)
    print("Visualization demonstration complete!")
    print("Check the 'plots/' directory for generated visualizations.")


if __name__ == "__main__":
    # Run the demonstration
    demonstrate_visualization_capabilities()