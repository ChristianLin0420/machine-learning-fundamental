"""
Word Embedding Visualization Tools
==================================

Comprehensive visualization suite for word embeddings including:
- t-SNE and PCA dimensionality reduction
- Semantic cluster visualization
- Word analogy visualization
- Embedding space exploration
- Comparative visualizations between models
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib.patches import FancyBboxPatch
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
import os

class EmbeddingVisualizer:
    """Comprehensive visualization toolkit for word embeddings"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_tsne_embeddings(self, model, words: List[str] = None, 
                           categories: Dict[str, List[str]] = None,
                           perplexity: int = 30, random_state: int = 42,
                           save_path: str = None, title: str = "t-SNE Word Embeddings"):
        """
        Create t-SNE visualization of word embeddings
        
        Args:
            model: Model with get_word_vector() method
            words: Specific words to visualize (if None, uses sample from vocab)
            categories: Dictionary of {category: [words]} for color coding
            perplexity: t-SNE perplexity parameter
            random_state: Random seed for reproducibility
            save_path: Path to save the plot
            title: Plot title
        """
        print(f"Creating t-SNE visualization...")
        
        # Get words to visualize
        if words is None:
            if hasattr(model, 'vocab'):
                # Sample words from vocabulary
                vocab_words = list(model.vocab.keys())
                words = vocab_words[:min(100, len(vocab_words))]  # Limit for visualization
            else:
                raise ValueError("Must provide words list or model must have vocab attribute")
        
        # Get word vectors
        vectors = []
        valid_words = []
        
        for word in words:
            vec = model.get_word_vector(word)
            if vec is not None:
                vectors.append(vec)
                valid_words.append(word)
        
        if len(vectors) == 0:
            print("No valid word vectors found!")
            return
        
        vectors = np.array(vectors)
        
        # Apply t-SNE
        print(f"Applying t-SNE to {len(vectors)} word vectors...")
        # Adjust perplexity to be less than number of samples
        adjusted_perplexity = min(perplexity, len(vectors) - 1, 30)
        print(f"Using perplexity: {adjusted_perplexity}")
        tsne = TSNE(n_components=2, perplexity=adjusted_perplexity, random_state=random_state, 
                   n_iter=1000, learning_rate=200.0)
        embeddings_2d = tsne.fit_transform(vectors)
        
        # Create plot
        plt.figure(figsize=self.figsize)
        
        # Color coding based on categories
        if categories:
            colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
            word_to_category = {}
            category_colors = {}
            
            for i, (category, cat_words) in enumerate(categories.items()):
                category_colors[category] = colors[i]
                for word in cat_words:
                    if word in valid_words:
                        word_to_category[word] = category
            
            # Plot points by category
            for category, color in category_colors.items():
                cat_indices = [i for i, word in enumerate(valid_words) 
                              if word_to_category.get(word) == category]
                if cat_indices:
                    cat_x = embeddings_2d[cat_indices, 0]
                    cat_y = embeddings_2d[cat_indices, 1]
                    plt.scatter(cat_x, cat_y, c=[color], label=category, alpha=0.7, s=60)
            
            # Plot uncategorized words
            uncat_indices = [i for i, word in enumerate(valid_words) 
                           if word not in word_to_category]
            if uncat_indices:
                uncat_x = embeddings_2d[uncat_indices, 0]
                uncat_y = embeddings_2d[uncat_indices, 1]
                plt.scatter(uncat_x, uncat_y, c='gray', label='Other', alpha=0.5, s=40)
            
            plt.legend()
        else:
            # Single color for all points
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=60)
        
        # Add word labels
        for i, word in enumerate(valid_words):
            plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
        
        plt.title(title)
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"t-SNE plot saved to {save_path}")
        
        plt.close()
    
    def plot_pca_embeddings(self, model, words: List[str] = None,
                           categories: Dict[str, List[str]] = None,
                           save_path: str = None, title: str = "PCA Word Embeddings"):
        """
        Create PCA visualization of word embeddings
        
        Args:
            model: Model with get_word_vector() method
            words: Specific words to visualize
            categories: Dictionary of {category: [words]} for color coding
            save_path: Path to save the plot
            title: Plot title
        """
        print(f"Creating PCA visualization...")
        
        # Get words to visualize
        if words is None:
            if hasattr(model, 'vocab'):
                vocab_words = list(model.vocab.keys())
                words = vocab_words[:min(100, len(vocab_words))]
            else:
                raise ValueError("Must provide words list or model must have vocab attribute")
        
        # Get word vectors
        vectors = []
        valid_words = []
        
        for word in words:
            vec = model.get_word_vector(word)
            if vec is not None:
                vectors.append(vec)
                valid_words.append(word)
        
        if len(vectors) == 0:
            print("No valid word vectors found!")
            return
        
        vectors = np.array(vectors)
        
        # Apply PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(vectors)
        
        # Create plot (similar to t-SNE plot)
        plt.figure(figsize=self.figsize)
        
        if categories:
            colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
            word_to_category = {}
            category_colors = {}
            
            for i, (category, cat_words) in enumerate(categories.items()):
                category_colors[category] = colors[i]
                for word in cat_words:
                    if word in valid_words:
                        word_to_category[word] = category
            
            for category, color in category_colors.items():
                cat_indices = [i for i, word in enumerate(valid_words) 
                              if word_to_category.get(word) == category]
                if cat_indices:
                    cat_x = embeddings_2d[cat_indices, 0]
                    cat_y = embeddings_2d[cat_indices, 1]
                    plt.scatter(cat_x, cat_y, c=[color], label=category, alpha=0.7, s=60)
            
            uncat_indices = [i for i, word in enumerate(valid_words) 
                           if word not in word_to_category]
            if uncat_indices:
                uncat_x = embeddings_2d[uncat_indices, 0]
                uncat_y = embeddings_2d[uncat_indices, 1]
                plt.scatter(uncat_x, uncat_y, c='gray', label='Other', alpha=0.5, s=40)
            
            plt.legend()
        else:
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=60)
        
        # Add word labels
        for i, word in enumerate(valid_words):
            plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
        
        plt.title(title)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PCA plot saved to {save_path}")
        
        plt.close()
    
    def plot_analogy_visualization(self, model, word_a: str, word_b: str, word_c: str,
                                  candidates: List[str] = None, save_path: str = None):
        """
        Visualize word analogy in 2D space using PCA
        
        Args:
            model: Model with get_word_vector() and analogy() methods
            word_a, word_b, word_c: Analogy words (A is to B as C is to ?)
            candidates: Candidate words to show
            save_path: Path to save the plot
        """
        print(f"Visualizing analogy: {word_a} - {word_b} + {word_c} = ?")
        
        # Get analogy prediction
        analogy_results = model.analogy(word_a, word_b, word_c, top_k=10)
        
        # Collect all words to visualize
        words_to_plot = [word_a, word_b, word_c]
        if analogy_results:
            predicted_words = [pred[0] for pred in analogy_results[:5]]  # Top 5 predictions
            words_to_plot.extend(predicted_words)
        
        if candidates:
            words_to_plot.extend(candidates)
        
        # Remove duplicates
        words_to_plot = list(set(words_to_plot))
        
        # Get vectors
        vectors = []
        valid_words = []
        
        for word in words_to_plot:
            vec = model.get_word_vector(word)
            if vec is not None:
                vectors.append(vec)
                valid_words.append(word)
        
        if len(vectors) < 4:
            print("Not enough valid vectors for visualization!")
            return
        
        # Calculate analogy vector
        vec_a = model.get_word_vector(word_a)
        vec_b = model.get_word_vector(word_b)
        vec_c = model.get_word_vector(word_c)
        analogy_vec = vec_b - vec_a + vec_c
        
        # Add analogy vector to visualization
        vectors.append(analogy_vec)
        valid_words.append(f"{word_b}-{word_a}+{word_c}")
        
        vectors = np.array(vectors)
        
        # Apply PCA for 2D visualization
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(vectors)
        
        # Create plot
        plt.figure(figsize=self.figsize)
        
        # Color code different types of words
        colors = {'input': 'red', 'predicted': 'blue', 'analogy_vector': 'green', 'other': 'gray'}
        
        for i, word in enumerate(valid_words):
            if word in [word_a, word_b, word_c]:
                color = colors['input']
                marker = 'o'
                size = 100
            elif word == f"{word_b}-{word_a}+{word_c}":
                color = colors['analogy_vector']
                marker = '*'
                size = 200
            elif analogy_results and word in [pred[0] for pred in analogy_results]:
                color = colors['predicted']
                marker = 's'
                size = 80
            else:
                color = colors['other']
                marker = 'o'
                size = 60
            
            plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], 
                       c=color, marker=marker, s=size, alpha=0.7)
            plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # Draw analogy vector
        if word_a in valid_words and word_b in valid_words and word_c in valid_words:
            idx_a = valid_words.index(word_a)
            idx_b = valid_words.index(word_b)
            idx_c = valid_words.index(word_c)
            idx_analogy = valid_words.index(f"{word_b}-{word_a}+{word_c}")
            
            # Draw vector from A to B
            plt.arrow(embeddings_2d[idx_a, 0], embeddings_2d[idx_a, 1],
                     embeddings_2d[idx_b, 0] - embeddings_2d[idx_a, 0],
                     embeddings_2d[idx_b, 1] - embeddings_2d[idx_a, 1],
                     head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.6)
            
            # Draw vector from C to analogy point
            plt.arrow(embeddings_2d[idx_c, 0], embeddings_2d[idx_c, 1],
                     embeddings_2d[idx_analogy, 0] - embeddings_2d[idx_c, 0],
                     embeddings_2d[idx_analogy, 1] - embeddings_2d[idx_c, 1],
                     head_width=0.1, head_length=0.1, fc='green', ec='green', alpha=0.6)
        
        # Create legend
        legend_elements = [
            plt.scatter([], [], c=colors['input'], marker='o', s=100, label='Input words'),
            plt.scatter([], [], c=colors['predicted'], marker='s', s=80, label='Predictions'),
            plt.scatter([], [], c=colors['analogy_vector'], marker='*', s=200, label='Analogy target'),
            plt.scatter([], [], c=colors['other'], marker='o', s=60, label='Other words')
        ]
        plt.legend(handles=legend_elements)
        
        plt.title(f'Analogy Visualization: {word_a} - {word_b} + {word_c} = ?')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Analogy visualization saved to {save_path}")
        
        plt.close()
    
    def plot_similarity_heatmap(self, model, words: List[str], save_path: str = None,
                               title: str = "Word Similarity Heatmap"):
        """
        Create similarity heatmap for a set of words
        
        Args:
            model: Model with get_word_vector() method
            words: List of words to compare
            save_path: Path to save the plot
            title: Plot title
        """
        print(f"Creating similarity heatmap for {len(words)} words...")
        
        # Get word vectors
        vectors = []
        valid_words = []
        
        for word in words:
            vec = model.get_word_vector(word)
            if vec is not None:
                vectors.append(vec)
                valid_words.append(word)
        
        if len(vectors) == 0:
            print("No valid word vectors found!")
            return
        
        vectors = np.array(vectors)
        
        # Calculate similarity matrix
        n = len(vectors)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    vec1, vec2 = vectors[i], vectors[j]
                    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    similarity_matrix[i, j] = similarity
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(similarity_matrix, 
                   xticklabels=valid_words, 
                   yticklabels=valid_words,
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True,
                   fmt='.2f')
        
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Similarity heatmap saved to {save_path}")
        
        plt.close()
    
    def compare_embeddings_2d(self, models: Dict[str, object], words: List[str],
                            method: str = 'tsne', save_path: str = None):
        """
        Compare multiple embedding models in 2D space
        
        Args:
            models: Dictionary of {name: model}
            words: Words to visualize
            method: 'tsne' or 'pca'
            save_path: Path to save the plot
        """
        print(f"Comparing {len(models)} models using {method.upper()}...")
        
        n_models = len(models)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, model) in enumerate(models.items()):
            # Get word vectors
            vectors = []
            valid_words = []
            
            for word in words:
                vec = model.get_word_vector(word)
                if vec is not None:
                    vectors.append(vec)
                    valid_words.append(word)
            
            if len(vectors) == 0:
                axes[i].text(0.5, 0.5, f'No valid vectors\nfor {model_name}', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(model_name)
                continue
            
            vectors = np.array(vectors)
            
            # Apply dimensionality reduction
            if method.lower() == 'tsne':
                # Adjust perplexity for t-SNE
                adjusted_perplexity = min(30, len(vectors) - 1, 15)
                reducer = TSNE(n_components=2, perplexity=adjusted_perplexity, random_state=42)
            else:  # PCA
                reducer = PCA(n_components=2)
            
            embeddings_2d = reducer.fit_transform(vectors)
            
            # Plot
            axes[i].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=60)
            
            # Add word labels
            for j, word in enumerate(valid_words):
                axes[i].annotate(word, (embeddings_2d[j, 0], embeddings_2d[j, 1]), 
                               xytext=(3, 3), textcoords='offset points', fontsize=8)
            
            axes[i].set_title(f'{model_name} ({method.upper()})')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved to {save_path}")
        
        plt.close()


def create_visualization_categories():
    """Create word categories for visualization"""
    categories = {
        'Royalty': ['king', 'queen', 'prince', 'princess'],
        'Gender': ['man', 'woman'],
        'Animals': ['dog', 'cat', 'bird', 'fish', 'lion', 'tiger'],
        'Sizes': ['big', 'large', 'huge', 'small', 'tiny'],
        'Colors': ['red', 'blue', 'green', 'yellow', 'black', 'white'],
        'Nature': ['tree', 'flower', 'forest', 'garden', 'mountain', 'ocean'],
        'Young': ['puppy', 'kitten', 'child', 'baby']
    }
    return categories


def demonstrate_visualization():
    """Demonstrate visualization functionality"""
    print("Word Embedding Visualization Demonstration")
    print("=" * 50)
    
    # Create visualization categories
    categories = create_visualization_categories()
    
    print(f"Created visualization categories:")
    for category, words in categories.items():
        print(f"  {category}: {words}")
    
    # Note: This would typically be used with trained models
    print("\nVisualization framework ready!")
    print("To use with trained models:")
    print("1. Load your trained Word2Vec/GloVe models")
    print("2. Use EmbeddingVisualizer.plot_tsne_embeddings()")
    print("3. Create analogy and similarity visualizations")
    
    return categories


if __name__ == "__main__":
    demonstrate_visualization()