"""
GloVe (Global Vectors) Implementation from Scratch
==================================================

Implements GloVe embeddings using matrix factorization of word co-occurrence statistics.

Key Components:
- Co-occurrence matrix construction
- Weighting function for rare/frequent word pairs
- Bilinear objective function optimization
- AdaGrad optimizer for convergence

GloVe combines the advantages of:
- Global matrix factorization methods (like LSA)
- Local context window methods (like Word2Vec)
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import re
from typing import List, Tuple, Dict, Optional
import pickle
import os
from tqdm import tqdm
from scipy.sparse import coo_matrix, csr_matrix

class GloVe:
    """GloVe implementation using weighted least squares on co-occurrence statistics"""
    
    def __init__(self,
                 embedding_dim: int = 100,
                 window_size: int = 5,
                 min_count: int = 5,
                 learning_rate: float = 0.05,
                 max_iter: int = 100,
                 x_max: float = 100.0,
                 alpha: float = 0.75,
                 random_state: int = 42):
        """
        Initialize GloVe model
        
        Args:
            embedding_dim: Dimensionality of word vectors
            window_size: Context window size for co-occurrence
            min_count: Minimum word frequency to include in vocabulary
            learning_rate: Learning rate for optimization
            max_iter: Maximum number of training iterations
            x_max: Cutoff for weighting function (Xmax)
            alpha: Exponent for weighting function
            random_state: Random seed for reproducibility
        """
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.min_count = min_count
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.x_max = x_max
        self.alpha = alpha
        self.random_state = random_state
        
        # Model parameters
        self.vocab = {}  # word -> index
        self.index_to_word = {}  # index -> word
        self.word_counts = {}
        self.vocab_size = 0
        
        # Weight matrices and biases
        self.W = None      # Main word vectors (vocab_size x embedding_dim)
        self.W_tilde = None  # Context word vectors (vocab_size x embedding_dim)
        self.b = None      # Main word biases (vocab_size,)
        self.b_tilde = None  # Context word biases (vocab_size,)
        
        # Co-occurrence matrix
        self.cooccur_matrix = None
        
        # Training statistics
        self.loss_history = []
        
        np.random.seed(random_state)
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text: lowercase, remove punctuation, split into words"""
        # Convert to lowercase and remove punctuation
        text = re.sub(r'[^\w\s]', '', text.lower())
        # Split into words
        words = text.split()
        return words
    
    def _build_vocabulary(self, corpus: List[str]):
        """Build vocabulary from corpus"""
        print("Building vocabulary...")
        
        # Count word frequencies
        all_words = []
        for text in corpus:
            words = self._preprocess_text(text)
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        
        # Filter words by minimum count
        filtered_words = {word: count for word, count in word_counts.items() 
                         if count >= self.min_count}
        
        # Create vocabulary mappings
        self.vocab = {word: idx for idx, word in enumerate(filtered_words.keys())}
        self.index_to_word = {idx: word for word, idx in self.vocab.items()}
        self.word_counts = filtered_words
        self.vocab_size = len(self.vocab)
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Total words processed: {len(all_words)}")
    
    def _build_cooccurrence_matrix(self, corpus: List[str]):
        """Build word co-occurrence matrix"""
        print("Building co-occurrence matrix...")
        
        # Initialize co-occurrence dictionary
        cooccur_dict = defaultdict(float)
        
        for text in tqdm(corpus, desc="Processing corpus"):
            words = self._preprocess_text(text)
            # Convert to indices, filtering out unknown words
            word_indices = [self.vocab[word] for word in words if word in self.vocab]
            
            # Build co-occurrence within window
            for i, center_idx in enumerate(word_indices):
                # Define context window
                start = max(0, i - self.window_size)
                end = min(len(word_indices), i + self.window_size + 1)
                
                for j in range(start, end):
                    if i != j:  # Skip center word
                        context_idx = word_indices[j]
                        # Weight by distance from center word
                        distance = abs(i - j)
                        weight = 1.0 / distance
                        
                        # Add to co-occurrence (symmetric)
                        cooccur_dict[(center_idx, context_idx)] += weight
        
        print(f"Co-occurrence entries: {len(cooccur_dict)}")
        
        # Convert to sparse matrix format
        rows, cols, data = [], [], []
        for (i, j), count in cooccur_dict.items():
            rows.append(i)
            cols.append(j)
            data.append(count)
        
        self.cooccur_matrix = coo_matrix(
            (data, (rows, cols)), 
            shape=(self.vocab_size, self.vocab_size)
        ).tocsr()
        
        print(f"Co-occurrence matrix shape: {self.cooccur_matrix.shape}")
        print(f"Non-zero entries: {self.cooccur_matrix.nnz}")
    
    def _weighting_function(self, x):
        """
        Weighting function f(X_ij) for co-occurrence counts
        
        f(x) = (x/x_max)^alpha if x < x_max else 1
        """
        weights = np.ones_like(x, dtype=np.float32)
        mask = x < self.x_max
        weights[mask] = (x[mask] / self.x_max) ** self.alpha
        return weights
    
    def _initialize_parameters(self):
        """Initialize word vectors and biases randomly"""
        print("Initializing parameters...")
        
        # Initialize word vectors with small random values
        scale = 1.0 / np.sqrt(self.embedding_dim)
        self.W = np.random.uniform(-scale, scale, (self.vocab_size, self.embedding_dim))
        self.W_tilde = np.random.uniform(-scale, scale, (self.vocab_size, self.embedding_dim))
        
        # Initialize biases to zero
        self.b = np.zeros(self.vocab_size)
        self.b_tilde = np.zeros(self.vocab_size)
        
        print(f"Initialized embeddings: {self.W.shape}")
    
    def _compute_loss_and_gradients(self):
        """Compute loss and gradients for all co-occurrence pairs"""
        # Get non-zero entries from co-occurrence matrix
        coo = self.cooccur_matrix.tocoo()
        i_indices = coo.row
        j_indices = coo.col
        x_ij = coo.data
        
        # Compute predictions: w_i^T * w_j + b_i + b_j
        w_i = self.W[i_indices]  # (nnz, embedding_dim)
        w_j = self.W_tilde[j_indices]  # (nnz, embedding_dim)
        b_i = self.b[i_indices]  # (nnz,)
        b_j = self.b_tilde[j_indices]  # (nnz,)
        
        # Dot products
        predictions = np.sum(w_i * w_j, axis=1) + b_i + b_j  # (nnz,)
        
        # Target: log(X_ij)
        targets = np.log(x_ij)
        
        # Weights
        weights = self._weighting_function(x_ij)
        
        # Weighted squared error
        errors = predictions - targets  # (nnz,)
        weighted_errors = weights * errors  # (nnz,)
        
        # Total loss
        loss = 0.5 * np.sum(weighted_errors * errors)
        
        # Gradients
        # For main word vectors W
        grad_W = np.zeros_like(self.W)
        np.add.at(grad_W, i_indices, (weighted_errors[:, None] * w_j))
        
        # For context word vectors W_tilde
        grad_W_tilde = np.zeros_like(self.W_tilde)
        np.add.at(grad_W_tilde, j_indices, (weighted_errors[:, None] * w_i))
        
        # For main word biases
        grad_b = np.zeros_like(self.b)
        np.add.at(grad_b, i_indices, weighted_errors)
        
        # For context word biases
        grad_b_tilde = np.zeros_like(self.b_tilde)
        np.add.at(grad_b_tilde, j_indices, weighted_errors)
        
        return loss, grad_W, grad_W_tilde, grad_b, grad_b_tilde
    
    def fit(self, corpus: List[str]):
        """Train GloVe model on corpus"""
        print(f"Training GloVe model...")
        
        # Build vocabulary and co-occurrence matrix
        self._build_vocabulary(corpus)
        self._build_cooccurrence_matrix(corpus)
        
        # Initialize parameters
        self._initialize_parameters()
        
        # AdaGrad accumulators
        grad_sq_W = np.ones_like(self.W)
        grad_sq_W_tilde = np.ones_like(self.W_tilde)
        grad_sq_b = np.ones_like(self.b)
        grad_sq_b_tilde = np.ones_like(self.b_tilde)
        
        print(f"\nStarting training for {self.max_iter} iterations...")
        
        for iteration in range(self.max_iter):
            # Compute loss and gradients
            loss, grad_W, grad_W_tilde, grad_b, grad_b_tilde = self._compute_loss_and_gradients()
            
            # AdaGrad updates
            grad_sq_W += grad_W ** 2
            grad_sq_W_tilde += grad_W_tilde ** 2
            grad_sq_b += grad_b ** 2
            grad_sq_b_tilde += grad_b_tilde ** 2
            
            # Update parameters
            self.W -= self.learning_rate * grad_W / np.sqrt(grad_sq_W)
            self.W_tilde -= self.learning_rate * grad_W_tilde / np.sqrt(grad_sq_W_tilde)
            self.b -= self.learning_rate * grad_b / np.sqrt(grad_sq_b)
            self.b_tilde -= self.learning_rate * grad_b_tilde / np.sqrt(grad_sq_b_tilde)
            
            self.loss_history.append(loss)
            
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{self.max_iter}, Loss: {loss:.6f}")
        
        print("Training completed!")
    
    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """Get word vector for a given word (average of main and context vectors)"""
        if word in self.vocab:
            idx = self.vocab[word]
            # GloVe uses the sum of main and context vectors
            return self.W[idx] + self.W_tilde[idx]
        return None
    
    def most_similar(self, word: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find most similar words to a given word"""
        if word not in self.vocab:
            return []
        
        word_vector = self.get_word_vector(word)
        
        # Calculate cosine similarities
        similarities = []
        for other_word, idx in self.vocab.items():
            if other_word != word:
                other_vector = self.get_word_vector(other_word)
                
                # Cosine similarity
                dot_product = np.dot(word_vector, other_vector)
                norm_product = np.linalg.norm(word_vector) * np.linalg.norm(other_vector)
                
                if norm_product > 0:
                    similarity = dot_product / norm_product
                    similarities.append((other_word, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def analogy(self, word_a: str, word_b: str, word_c: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Solve analogy: word_a is to word_b as word_c is to ?
        Example: king - man + woman = queen
        """
        if not all(word in self.vocab for word in [word_a, word_b, word_c]):
            return []
        
        # Get vectors
        vec_a = self.get_word_vector(word_a)
        vec_b = self.get_word_vector(word_b)
        vec_c = self.get_word_vector(word_c)
        
        # Calculate analogy vector: vec_b - vec_a + vec_c
        analogy_vector = vec_b - vec_a + vec_c
        
        # Find most similar words to analogy vector
        similarities = []
        for word, idx in self.vocab.items():
            if word not in [word_a, word_b, word_c]:  # Exclude input words
                word_vector = self.get_word_vector(word)
                
                # Cosine similarity
                dot_product = np.dot(analogy_vector, word_vector)
                norm_product = np.linalg.norm(analogy_vector) * np.linalg.norm(word_vector)
                
                if norm_product > 0:
                    similarity = dot_product / norm_product
                    similarities.append((word, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_cooccurrence_stats(self):
        """Get statistics about the co-occurrence matrix"""
        coo = self.cooccur_matrix.tocoo()
        counts = coo.data
        
        stats = {
            'total_pairs': len(counts),
            'total_count': np.sum(counts),
            'mean_count': np.mean(counts),
            'median_count': np.median(counts),
            'max_count': np.max(counts),
            'min_count': np.min(counts),
            'sparsity': 1.0 - (len(counts) / (self.vocab_size ** 2))
        }
        
        return stats
    
    def save_model(self, filepath: str):
        """Save trained model"""
        model_data = {
            'embedding_dim': self.embedding_dim,
            'window_size': self.window_size,
            'vocab': self.vocab,
            'index_to_word': self.index_to_word,
            'word_counts': self.word_counts,
            'vocab_size': self.vocab_size,
            'W': self.W,
            'W_tilde': self.W_tilde,
            'b': self.b,
            'b_tilde': self.b_tilde,
            'loss_history': self.loss_history,
            'x_max': self.x_max,
            'alpha': self.alpha
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.embedding_dim = model_data['embedding_dim']
        self.window_size = model_data['window_size']
        self.vocab = model_data['vocab']
        self.index_to_word = model_data['index_to_word']
        self.word_counts = model_data['word_counts']
        self.vocab_size = model_data['vocab_size']
        self.W = model_data['W']
        self.W_tilde = model_data['W_tilde']
        self.b = model_data['b']
        self.b_tilde = model_data['b_tilde']
        self.loss_history = model_data['loss_history']
        self.x_max = model_data['x_max']
        self.alpha = model_data['alpha']
        print(f"Model loaded from {filepath}")


def create_extended_corpus():
    """Create an extended corpus for GloVe demonstration"""
    # Base corpus with semantic relationships
    base_corpus = [
        "the king is strong and powerful ruler",
        "the queen is beautiful and elegant ruler", 
        "man and woman are different genders",
        "prince and princess are royal children",
        "dog and cat are common pets",
        "puppy is a young small dog",
        "kitten is a young small cat",
        "bird flies high in the blue sky",
        "fish swims deep in clear water",
        "tree grows tall in green forest",
        "flower blooms bright in beautiful garden",
        "sun shines warm during bright day",
        "moon glows soft at quiet night",
        "rain falls heavy from dark clouds",
        "wind blows strong through tall trees",
        "fire burns hot and bright",
        "water flows cold in mountain river",
        "mountain stands high and tall",
        "valley lies low and green",
        "ocean is vast deep and blue",
        "king rules his large kingdom wisely",
        "queen helps the kind king rule",
        "prince will become future king",
        "princess will become future queen",
        "man works very hard daily",
        "woman is very intelligent and wise",
        "child plays happily in garden",
        "baby sleeps peacefully in crib",
        "dog barks loudly at strangers",
        "cat meows softly for food",
        "bird sings beautifully in morning",
        "fish swims gracefully in pond"
    ]
    
    # Additional semantic clusters
    animals = [
        "lion is king of jungle",
        "tiger is powerful striped cat",
        "elephant is large gray animal",
        "mouse is small quick animal",
        "horse runs fast in field",
        "cow gives fresh milk",
        "sheep provides warm wool",
        "pig is pink farm animal"
    ]
    
    colors = [
        "red is color of fire",
        "blue is color of sky",
        "green is color of grass",
        "yellow is color of sun",
        "black is dark color",
        "white is light color",
        "purple is royal color",
        "orange is fruit color"
    ]
    
    sizes = [
        "big elephant is very large",
        "small mouse is very tiny",
        "huge mountain is extremely large",
        "tiny ant is extremely small",
        "giant tree is very big",
        "mini car is very small"
    ]
    
    # Combine all and repeat for more training data
    corpus = (base_corpus + animals + colors + sizes) * 25
    
    return corpus


def demonstrate_glove():
    """Demonstrate GloVe functionality"""
    print("GloVe Demonstration")
    print("=" * 50)
    
    # Create extended corpus
    corpus = create_extended_corpus()
    print(f"Corpus size: {len(corpus)} sentences")
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Train GloVe model
    print("\n" + "=" * 50)
    print("Training GloVe Model")
    print("=" * 50)
    
    glove_model = GloVe(
        embedding_dim=50,
        window_size=5,
        min_count=3,
        learning_rate=0.05,
        max_iter=100,
        x_max=100.0,
        alpha=0.75,
        random_state=42
    )
    
    glove_model.fit(corpus)
    glove_model.save_model('plots/glove_model.pkl')
    
    # Display co-occurrence statistics
    print("\n" + "=" * 50)
    print("Co-occurrence Matrix Statistics")
    print("=" * 50)
    
    cooccur_stats = glove_model.get_cooccurrence_stats()
    for key, value in cooccur_stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Test word similarities
    print("\n" + "=" * 50)
    print("Word Similarities")
    print("=" * 50)
    
    test_words = ['king', 'queen', 'man', 'woman', 'dog', 'cat', 'big', 'small']
    
    for word in test_words:
        if word in glove_model.vocab:
            print(f"\nWords similar to '{word}':")
            similar_words = glove_model.most_similar(word, top_k=5)
            for similar_word, score in similar_words:
                print(f"  {similar_word}: {score:.3f}")
    
    # Test analogies
    print("\n" + "=" * 50)
    print("Analogy Tests")
    print("=" * 50)
    
    analogy_tests = [
        ('king', 'man', 'queen'),    # king - man + queen = ?
        ('big', 'small', 'large'),   # big - small + large = ?
        ('dog', 'puppy', 'cat'),     # dog - puppy + cat = ?
        ('prince', 'king', 'princess'), # prince - king + princess = ?
    ]
    
    for word_a, word_b, word_c in analogy_tests:
        print(f"\n{word_a} - {word_b} + {word_c} = ?")
        analogy_results = glove_model.analogy(word_a, word_b, word_c, top_k=3)
        for word, score in analogy_results:
            print(f"  {word}: {score:.3f}")
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(glove_model.loss_history, color='green', linewidth=2)
    plt.title('GloVe Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale for better visualization
    plt.savefig('plots/glove_training_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot co-occurrence distribution
    coo = glove_model.cooccur_matrix.tocoo()
    counts = coo.data
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(counts, bins=50, alpha=0.7, color='green', edgecolor='black')
    plt.title('Co-occurrence Count Distribution')
    plt.xlabel('Co-occurrence Count')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(np.log(counts + 1), bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Log Co-occurrence Count Distribution')
    plt.xlabel('Log(Co-occurrence Count + 1)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/glove_cooccurrence_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to 'plots/' directory")
    print("GloVe demonstration completed!")
    
    return glove_model


if __name__ == "__main__":
    demonstrate_glove()