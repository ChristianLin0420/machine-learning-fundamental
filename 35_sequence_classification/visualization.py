#!/usr/bin/env python3
"""
Visualization utilities for sequence classification analysis.

This module provides tools for visualizing:
- Attention weights and heatmaps
- Training curves and model comparison
- Sequence length analysis
- Embedding visualizations
- Model performance metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd
from collections import defaultdict
import os

# Set style
plt.style.use('default')
sns.set_palette("husl")

class SequenceClassificationVisualizer:
    """Comprehensive visualization tools for sequence classification."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), save_dir: str = "plots"):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size
            save_dir: Directory to save plots
        """
        self.figsize = figsize
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Color palette
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
    
    def plot_attention_weights(self, attention_weights: np.ndarray, 
                             tokens: List[str], predicted_class: str,
                             save_path: Optional[str] = None,
                             title: str = "Attention Weights Visualization"):
        """
        Visualize attention weights as a heatmap.
        
        Args:
            attention_weights: (seq_length,) attention weights
            tokens: List of tokens in the sequence
            predicted_class: Predicted class label
            save_path: Path to save the plot
            title: Plot title
        """
        plt.figure(figsize=(max(len(tokens) * 0.5, 8), 4))
        
        # Create heatmap
        weights_2d = attention_weights.reshape(1, -1)
        
        # Create heatmap
        sns.heatmap(
            weights_2d,
            xticklabels=tokens,
            yticklabels=[predicted_class],
            cmap='Reds',
            cbar_kws={'label': 'Attention Weight'},
            annot=True,
            fmt='.3f',
            square=False
        )
        
        plt.title(f"{title}\nPredicted Class: {predicted_class}")
        plt.xlabel("Tokens")
        plt.ylabel("Prediction")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.save_dir, save_path), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_attention_heatmap_multiple(self, attention_data: List[Dict],
                                      save_path: Optional[str] = None,
                                      title: str = "Attention Patterns Across Samples"):
        """
        Plot multiple attention patterns in a grid.
        
        Args:
            attention_data: List of dicts with 'weights', 'tokens', 'prediction', 'true_label'
            save_path: Path to save the plot
            title: Plot title
        """
        n_samples = len(attention_data)
        cols = min(3, n_samples)
        rows = (n_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows))
        if n_samples == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, data in enumerate(attention_data):
            ax = axes[i] if i < len(axes) else None
            if ax is None:
                break
            
            weights = data['weights'].reshape(1, -1)
            tokens = data['tokens']
            pred = data['prediction']
            true_label = data.get('true_label', 'Unknown')
            
            # Truncate long sequences for visualization
            if len(tokens) > 20:
                mid = len(tokens) // 2
                tokens = tokens[:10] + ['...'] + tokens[-10:]
                weights = np.concatenate([
                    weights[:, :10], 
                    [[weights.mean()]], 
                    weights[:, -10:]
                ], axis=1)
            
            im = ax.imshow(weights, cmap='Reds', aspect='auto')
            
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
            ax.set_yticks([0])
            ax.set_yticklabels([f"Pred: {pred}\nTrue: {true_label}"])
            ax.set_title(f"Sample {i+1}", fontsize=10)
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide empty subplots
        for i in range(n_samples, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.save_dir, save_path), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_curves(self, history: Dict, save_path: Optional[str] = None,
                           title: str = "Training Curves"):
        """
        Plot training and validation curves.
        
        Args:
            history: Dictionary with 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'
            save_path: Path to save the plot
            title: Plot title
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        if 'val_loss' in history and history['val_loss']:
            ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(epochs, history['train_accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        if 'val_accuracy' in history and history['val_accuracy']:
            ax2.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.save_dir, save_path), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, results: Dict[str, Dict],
                            save_path: Optional[str] = None,
                            title: str = "Model Performance Comparison"):
        """
        Compare multiple models' performance.
        
        Args:
            results: Dict of {model_name: {'accuracy': float, 'loss': float, 'f1': float}}
            save_path: Path to save the plot
            title: Plot title
        """
        models = list(results.keys())
        metrics = ['accuracy', 'loss', 'f1']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, max(5, 0.7 * len(models))))
        
        for i, metric in enumerate(metrics):
            values = [results[model].get(metric, 0) for model in models]
            
            bars = axes[i].bar(models, values, color=self.colors[:len(models)])
            axes[i].set_title(f'{metric.capitalize()}')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
            
            axes[i].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.save_dir, save_path), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            class_names: List[str], normalize: bool = True,
                            save_path: Optional[str] = None,
                            title: str = "Confusion Matrix"):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            normalize: Whether to normalize the matrix
            save_path: Path to save the plot
            title: Plot title
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.save_dir, save_path), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_sequence_length_analysis(self, length_performance: Dict[int, Dict],
                                    save_path: Optional[str] = None,
                                    title: str = "Performance vs Sequence Length"):
        """
        Analyze model performance across different sequence lengths.
        
        Args:
            length_performance: Dict of {length: {'accuracy': float, 'loss': float, 'count': int}}
            save_path: Path to save the plot
            title: Plot title
        """
        lengths = sorted(length_performance.keys())
        accuracies = [length_performance[l]['accuracy'] for l in lengths]
        losses = [length_performance[l]['loss'] for l in lengths]
        counts = [length_performance[l]['count'] for l in lengths]
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Accuracy vs length
        ax1.plot(lengths, accuracies, 'bo-', linewidth=2, markersize=6)
        ax1.set_title('Accuracy vs Sequence Length')
        ax1.set_ylabel('Accuracy')
        ax1.grid(True, alpha=0.3)
        
        # Loss vs length
        ax2.plot(lengths, losses, 'ro-', linewidth=2, markersize=6)
        ax2.set_title('Loss vs Sequence Length')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)
        
        # Sample count distribution
        ax3.bar(lengths, counts, alpha=0.7, color='green')
        ax3.set_title('Number of Samples per Length')
        ax3.set_xlabel('Sequence Length')
        ax3.set_ylabel('Count')
        ax3.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.save_dir, save_path), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_embedding_analysis(self, embeddings: np.ndarray, labels: np.ndarray,
                              class_names: List[str], method: str = 'tsne',
                              save_path: Optional[str] = None,
                              title: str = "Embedding Visualization"):
        """
        Visualize learned embeddings using dimensionality reduction.
        
        Args:
            embeddings: (n_samples, embed_dim) embedding vectors
            labels: (n_samples,) class labels
            class_names: List of class names
            method: Dimensionality reduction method ('tsne' or 'pca')
            save_path: Path to save the plot
            title: Plot title
        """
        # Apply dimensionality reduction
        if method.lower() == 'tsne':
            # Adjust perplexity for small datasets
            perplexity = min(30, len(embeddings) - 1, 10)
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        else:  # PCA
            reducer = PCA(n_components=2)
        
        embeddings_2d = reducer.fit_transform(embeddings)
        
        plt.figure(figsize=self.figsize)
        
        # Plot each class
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                       c=[self.colors[i]], label=class_names[label],
                       alpha=0.7, s=60)
        
        plt.title(f"{title} ({method.upper()})")
        plt.xlabel(f'{method.upper()} 1')
        plt.ylabel(f'{method.upper()} 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.save_dir, save_path), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_learning_curves_comparison(self, histories: Dict[str, Dict],
                                      save_path: Optional[str] = None,
                                      title: str = "Learning Curves Comparison"):
        """
        Compare learning curves of multiple models.
        
        Args:
            histories: Dict of {model_name: history_dict}
            save_path: Path to save the plot
            title: Plot title
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        for i, (model_name, history) in enumerate(histories.items()):
            color = self.colors[i % len(self.colors)]
            epochs = range(1, len(history['train_loss']) + 1)
            
            # Training loss
            ax1.plot(epochs, history['train_loss'], color=color, 
                    label=f'{model_name}', linewidth=2)
            
            # Validation loss
            if 'val_loss' in history and history['val_loss']:
                ax2.plot(epochs, history['val_loss'], color=color,
                        label=f'{model_name}', linewidth=2)
            
            # Training accuracy
            ax3.plot(epochs, history['train_accuracy'], color=color,
                    label=f'{model_name}', linewidth=2)
            
            # Validation accuracy
            if 'val_accuracy' in history and history['val_accuracy']:
                ax4.plot(epochs, history['val_accuracy'], color=color,
                        label=f'{model_name}', linewidth=2)
        
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('Validation Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3.set_title('Training Accuracy')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Accuracy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4.set_title('Validation Accuracy')
        ax4.set_xlabel('Epochs')
        ax4.set_ylabel('Accuracy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.save_dir, save_path), dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_attention_patterns(self, attention_weights_list: List[np.ndarray],
                                 tokens_list: List[List[str]],
                                 predictions_list: List[str],
                                 save_path: Optional[str] = None,
                                 title: str = "Attention Pattern Analysis"):
        """
        Analyze attention patterns across multiple samples.
        
        Args:
            attention_weights_list: List of attention weight arrays
            tokens_list: List of token lists
            predictions_list: List of predictions
            save_path: Path to save the plot
            title: Plot title
        """
        # Find most attended words across all samples
        word_attention = defaultdict(list)
        
        for weights, tokens in zip(attention_weights_list, tokens_list):
            for token, weight in zip(tokens, weights):
                word_attention[token].append(weight)
        
        # Calculate average attention for each word
        avg_attention = {word: np.mean(weights) 
                        for word, weights in word_attention.items() 
                        if len(weights) >= 2}  # Only words appearing multiple times
        
        # Sort by attention score
        sorted_words = sorted(avg_attention.items(), key=lambda x: x[1], reverse=True)
        
        # Plot top attended words
        plt.figure(figsize=(12, 8))
        
        top_words = sorted_words[:20]  # Top 20 words
        words, scores = zip(*top_words) if top_words else ([], [])
        
        plt.barh(range(len(words)), scores, color=self.colors[0])
        plt.yticks(range(len(words)), words)
        plt.xlabel('Average Attention Weight')
        plt.title(f'{title}\nTop Attended Words Across All Samples')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.save_dir, save_path), dpi=300, bbox_inches='tight')
        plt.show()
        
        return avg_attention
    
    def create_classification_report_plot(self, y_true: np.ndarray, y_pred: np.ndarray,
                                        class_names: List[str],
                                        save_path: Optional[str] = None,
                                        title: str = "Classification Report"):
        """
        Create a visual classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            save_path: Path to save the plot
            title: Plot title
        """
        from sklearn.metrics import precision_recall_fscore_support
        
        # Calculate metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(len(class_names))
        )
        
        # Create DataFrame for visualization
        metrics_data = {
            'Class': class_names,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        }
        df = pd.DataFrame(metrics_data)
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        
        # Precision
        axes[0].bar(df['Class'], df['Precision'], color=self.colors[0], alpha=0.7)
        axes[0].set_title('Precision by Class')
        axes[0].set_ylabel('Precision')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Recall
        axes[1].bar(df['Class'], df['Recall'], color=self.colors[1], alpha=0.7)
        axes[1].set_title('Recall by Class')
        axes[1].set_ylabel('Recall')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # F1-Score
        axes[2].bar(df['Class'], df['F1-Score'], color=self.colors[2], alpha=0.7)
        axes[2].set_title('F1-Score by Class')
        axes[2].set_ylabel('F1-Score')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(self.save_dir, save_path), dpi=300, bbox_inches='tight')
        plt.show()
        
        return df


def extract_attention_weights(model, sequences: torch.Tensor, vocabulary,
                            class_names: List[str]) -> List[Dict]:
    """
    Extract attention weights from attention-based models.
    
    Args:
        model: Attention-based PyTorch model
        sequences: Input sequences
        vocabulary: Vocabulary object
        class_names: List of class names
        
    Returns:
        attention_data: List of attention visualization data
    """
    model.eval()
    attention_data = []
    
    with torch.no_grad():
        outputs = model(sequences)
        predictions = torch.argmax(outputs, dim=1)
        
        # Check if model has attention weights
        if hasattr(model, 'last_attention_weights'):
            attention_weights = model.last_attention_weights.cpu().numpy()
            
            for i in range(len(sequences)):
                # Get tokens (excluding padding)
                seq = sequences[i].cpu().numpy()
                mask = seq != 0  # Assuming 0 is padding
                tokens = [vocabulary.idx_to_token.get(idx, '<UNK>') 
                         for idx in seq[mask]]
                
                # Get corresponding attention weights
                weights = attention_weights[i][mask]
                
                # Get prediction
                pred_idx = predictions[i].item()
                predicted_class = class_names[pred_idx]
                
                attention_data.append({
                    'tokens': tokens,
                    'weights': weights,
                    'prediction': predicted_class
                })
    
    return attention_data


if __name__ == "__main__":
    # Test visualization utilities
    print("Testing Sequence Classification Visualizations")
    print("=" * 50)
    
    # Create test data
    np.random.seed(42)
    
    # Create visualizer
    visualizer = SequenceClassificationVisualizer(save_dir="test_plots")
    
    # Test attention visualization
    print("Testing attention visualization...")
    tokens = ['this', 'movie', 'is', 'really', 'great', 'and', 'amazing']
    attention_weights = np.array([0.1, 0.15, 0.05, 0.3, 0.25, 0.1, 0.05])
    
    visualizer.plot_attention_weights(
        attention_weights, tokens, "positive",
        save_path="test_attention.png"
    )
    
    # Test training curves
    print("Testing training curves...")
    history = {
        'train_loss': [1.2, 0.8, 0.6, 0.4, 0.3],
        'val_loss': [1.1, 0.9, 0.7, 0.5, 0.4],
        'train_accuracy': [0.4, 0.6, 0.7, 0.8, 0.85],
        'val_accuracy': [0.45, 0.62, 0.68, 0.75, 0.78]
    }
    
    visualizer.plot_training_curves(
        history, save_path="test_training_curves.png"
    )
    
    # Test model comparison
    print("Testing model comparison...")
    results = {
        'RNN': {'accuracy': 0.75, 'loss': 0.6, 'f1': 0.74},
        'LSTM': {'accuracy': 0.82, 'loss': 0.45, 'f1': 0.81},
        'Attention': {'accuracy': 0.87, 'loss': 0.35, 'f1': 0.86}
    }
    
    visualizer.plot_model_comparison(
        results, save_path="test_model_comparison.png"
    )
    
    # Test confusion matrix
    print("Testing confusion matrix...")
    y_true = np.random.randint(0, 3, 100)
    y_pred = np.random.randint(0, 3, 100)
    class_names = ['Negative', 'Neutral', 'Positive']
    
    visualizer.plot_confusion_matrix(
        y_true, y_pred, class_names,
        save_path="test_confusion_matrix.png"
    )
    
    # Test embedding visualization
    print("Testing embedding visualization...")
    embeddings = np.random.randn(100, 50)
    labels = np.random.randint(0, 3, 100)
    
    visualizer.plot_embedding_analysis(
        embeddings, labels, class_names, method='pca',
        save_path="test_embeddings.png"
    )
    
    print("Visualization tests completed!")
    print("Check the 'test_plots' directory for generated visualizations.")