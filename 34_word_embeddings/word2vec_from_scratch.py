"""
Word2Vec Implementation from Scratch
===================================

Implements both Skip-gram and CBOW (Continuous Bag of Words) models
with negative sampling for efficient training.

Key Components:
- Vocabulary building with subsampling
- Skip-gram: predict context words from target word
- CBOW: predict target word from context words
- Negative sampling for computational efficiency
- Hierarchical softmax alternative
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import re
from typing import List, Tuple, Dict, Optional
import pickle
import os
from tqdm import tqdm

class Word2Vec:
    """Word2Vec implementation with Skip-gram and CBOW models"""
    
    def __init__(self, 
                 embedding_dim: int = 100,
                 window_size: int = 5,
                 min_count: int = 5,
                 negative_samples: int = 5,
                 learning_rate: float = 0.025,
                 epochs: int = 5,
                 model_type: str = 'skipgram',
                 subsample_threshold: float = 1e-3):
        """
        Initialize Word2Vec model
        
        Args:
            embedding_dim: Dimensionality of word vectors
            window_size: Context window size
            min_count: Minimum word frequency to include in vocabulary
            negative_samples: Number of negative samples per positive sample
            learning_rate: Initial learning rate
            epochs: Number of training epochs
            model_type: 'skipgram' or 'cbow'
            subsample_threshold: Threshold for subsampling frequent words
        """
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.min_count = min_count
        self.negative_samples = negative_samples
        self.learning_rate = learning_rate
        self.initial_lr = learning_rate
        self.epochs = epochs
        self.model_type = model_type.lower()
        self.subsample_threshold = subsample_threshold
        
        # Model parameters
        self.vocab = {}  # word -> index
        self.index_to_word = {}  # index -> word
        self.word_counts = {}
        self.vocab_size = 0
        self.total_words = 0
        
        # Weight matrices
        self.W_in = None   # Input embeddings (vocab_size x embedding_dim)
        self.W_out = None  # Output embeddings (vocab_size x embedding_dim)
        
        # For negative sampling
        self.unigram_table = None
        self.unigram_table_size = int(1e8)
        
        # Training statistics
        self.loss_history = []
        
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
        self.total_words = len(all_words)
        
        # Filter words by minimum count
        filtered_words = {word: count for word, count in word_counts.items() 
                         if count >= self.min_count}
        
        # Create vocabulary mappings
        self.vocab = {word: idx for idx, word in enumerate(filtered_words.keys())}
        self.index_to_word = {idx: word for word, idx in self.vocab.items()}
        self.word_counts = filtered_words
        self.vocab_size = len(self.vocab)
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Total words: {self.total_words}")
        
    def _create_unigram_table(self):
        """Create unigram table for negative sampling"""
        print("Creating unigram table for negative sampling...")
        
        # Calculate word probabilities (with 3/4 power)
        word_probs = []
        for word in self.vocab.keys():
            count = self.word_counts[word]
            prob = (count / self.total_words) ** 0.75
            word_probs.append(prob)
        
        # Normalize probabilities
        word_probs = np.array(word_probs)
        word_probs = word_probs / np.sum(word_probs)
        
        # Create lookup table
        self.unigram_table = np.zeros(self.unigram_table_size, dtype=np.int32)
        
        word_idx = 0
        cumulative_prob = word_probs[word_idx]
        
        for i in range(self.unigram_table_size):
            self.unigram_table[i] = word_idx
            
            if i / self.unigram_table_size > cumulative_prob:
                word_idx += 1
                if word_idx < len(word_probs):
                    cumulative_prob += word_probs[word_idx]
    
    def _subsample_words(self, words: List[str]) -> List[str]:
        """Apply subsampling to frequent words"""
        if self.subsample_threshold <= 0:
            return words
        
        subsampled = []
        for word in words:
            if word in self.vocab:
                freq = self.word_counts[word] / self.total_words
                prob = (np.sqrt(freq / self.subsample_threshold) + 1) * \
                       (self.subsample_threshold / freq)
                
                if np.random.rand() < prob:
                    subsampled.append(word)
            else:
                # Keep unknown words (though they won't be in training)
                subsampled.append(word)
        
        return subsampled
    
    def _get_negative_samples(self, positive_word_idx: int) -> List[int]:
        """Get negative samples for a positive word"""
        negative_samples = []
        
        while len(negative_samples) < self.negative_samples:
            # Sample from unigram table
            idx = np.random.randint(0, self.unigram_table_size)
            candidate = self.unigram_table[idx]
            
            # Avoid sampling the positive word
            if candidate != positive_word_idx:
                negative_samples.append(candidate)
        
        return negative_samples
    
    def _sigmoid(self, x):
        """Stable sigmoid function"""
        x = np.clip(x, -250, 250)  # Prevent overflow
        return 1 / (1 + np.exp(-x))
    
    def _train_skipgram(self, center_word_idx: int, context_word_idx: int):
        """Train skip-gram model on a word pair"""
        # Get embeddings
        center_embed = self.W_in[center_word_idx]  # (embedding_dim,)
        context_embed = self.W_out[context_word_idx]  # (embedding_dim,)
        
        # Positive sample
        score = np.dot(center_embed, context_embed)
        pred = self._sigmoid(score)
        
        # Gradient for positive sample
        grad_center = (1 - pred) * context_embed
        grad_context = (1 - pred) * center_embed
        
        # Update for positive sample
        self.W_in[center_word_idx] += self.learning_rate * grad_center
        self.W_out[context_word_idx] += self.learning_rate * grad_context
        
        # Negative samples
        negative_samples = self._get_negative_samples(context_word_idx)
        
        for neg_word_idx in negative_samples:
            neg_embed = self.W_out[neg_word_idx]
            
            # Negative sample
            neg_score = np.dot(center_embed, neg_embed)
            neg_pred = self._sigmoid(neg_score)
            
            # Gradient for negative sample
            neg_grad_center = -neg_pred * neg_embed
            neg_grad_neg = -neg_pred * center_embed
            
            # Update for negative sample
            self.W_in[center_word_idx] += self.learning_rate * neg_grad_center
            self.W_out[neg_word_idx] += self.learning_rate * neg_grad_neg
        
        # Calculate loss (for monitoring)
        pos_loss = -np.log(pred + 1e-10)
        neg_loss = -np.sum([np.log(1 - self._sigmoid(np.dot(center_embed, self.W_out[neg])) + 1e-10) 
                           for neg in negative_samples])
        
        return pos_loss + neg_loss
    
    def _train_cbow(self, context_word_indices: List[int], target_word_idx: int):
        """Train CBOW model on context-target pair"""
        # Average context embeddings
        context_embeds = self.W_in[context_word_indices]  # (context_size, embedding_dim)
        avg_context = np.mean(context_embeds, axis=0)  # (embedding_dim,)
        
        target_embed = self.W_out[target_word_idx]  # (embedding_dim,)
        
        # Positive sample
        score = np.dot(avg_context, target_embed)
        pred = self._sigmoid(score)
        
        # Gradient for positive sample
        grad_context = (1 - pred) * target_embed / len(context_word_indices)
        grad_target = (1 - pred) * avg_context
        
        # Update for positive sample
        for ctx_idx in context_word_indices:
            self.W_in[ctx_idx] += self.learning_rate * grad_context
        self.W_out[target_word_idx] += self.learning_rate * grad_target
        
        # Negative samples
        negative_samples = self._get_negative_samples(target_word_idx)
        
        for neg_word_idx in negative_samples:
            neg_embed = self.W_out[neg_word_idx]
            
            # Negative sample
            neg_score = np.dot(avg_context, neg_embed)
            neg_pred = self._sigmoid(neg_score)
            
            # Gradient for negative sample
            neg_grad_context = -neg_pred * neg_embed / len(context_word_indices)
            neg_grad_neg = -neg_pred * avg_context
            
            # Update for negative sample
            for ctx_idx in context_word_indices:
                self.W_in[ctx_idx] += self.learning_rate * neg_grad_context
            self.W_out[neg_word_idx] += self.learning_rate * neg_grad_neg
        
        # Calculate loss
        pos_loss = -np.log(pred + 1e-10)
        neg_loss = -np.sum([np.log(1 - self._sigmoid(np.dot(avg_context, self.W_out[neg])) + 1e-10) 
                           for neg in negative_samples])
        
        return pos_loss + neg_loss
    
    def _generate_training_pairs(self, words: List[str]):
        """Generate training pairs from a sentence"""
        word_indices = [self.vocab[word] for word in words if word in self.vocab]
        
        pairs = []
        for i, center_word_idx in enumerate(word_indices):
            # Define context window
            start = max(0, i - self.window_size)
            end = min(len(word_indices), i + self.window_size + 1)
            
            context_indices = []
            for j in range(start, end):
                if j != i:  # Skip center word
                    context_indices.append(word_indices[j])
            
            if self.model_type == 'skipgram':
                # Skip-gram: predict each context word from center word
                for context_idx in context_indices:
                    pairs.append((center_word_idx, context_idx))
            else:  # CBOW
                # CBOW: predict center word from context
                if context_indices:  # Make sure we have context
                    pairs.append((context_indices, center_word_idx))
        
        return pairs
    
    def fit(self, corpus: List[str]):
        """Train Word2Vec model on corpus"""
        print(f"Training {self.model_type.upper()} model...")
        
        # Build vocabulary
        self._build_vocabulary(corpus)
        
        # Initialize weight matrices
        self.W_in = np.random.uniform(-0.5, 0.5, (self.vocab_size, self.embedding_dim)) / self.embedding_dim
        self.W_out = np.random.uniform(-0.5, 0.5, (self.vocab_size, self.embedding_dim)) / self.embedding_dim
        
        # Create unigram table for negative sampling
        self._create_unigram_table()
        
        # Training loop
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            
            epoch_loss = 0
            num_pairs = 0
            
            # Shuffle corpus
            shuffled_corpus = corpus.copy()
            np.random.shuffle(shuffled_corpus)
            
            for text in tqdm(shuffled_corpus, desc="Training"):
                words = self._preprocess_text(text)
                words = self._subsample_words(words)  # Apply subsampling
                
                # Generate training pairs
                pairs = self._generate_training_pairs(words)
                
                # Train on each pair
                for pair in pairs:
                    if self.model_type == 'skipgram':
                        center_idx, context_idx = pair
                        loss = self._train_skipgram(center_idx, context_idx)
                    else:  # CBOW
                        context_indices, target_idx = pair
                        loss = self._train_cbow(context_indices, target_idx)
                    
                    epoch_loss += loss
                    num_pairs += 1
            
            # Update learning rate (linear decay)
            progress = (epoch + 1) / self.epochs
            self.learning_rate = self.initial_lr * (1 - progress)
            
            avg_loss = epoch_loss / max(num_pairs, 1)
            self.loss_history.append(avg_loss)
            print(f"Average loss: {avg_loss:.4f}, Learning rate: {self.learning_rate:.6f}")
    
    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        """Get word vector for a given word"""
        if word in self.vocab:
            return self.W_in[self.vocab[word]]
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
                other_vector = self.W_in[idx]
                
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
                word_vector = self.W_in[idx]
                
                # Cosine similarity
                dot_product = np.dot(analogy_vector, word_vector)
                norm_product = np.linalg.norm(analogy_vector) * np.linalg.norm(word_vector)
                
                if norm_product > 0:
                    similarity = dot_product / norm_product
                    similarities.append((word, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def save_model(self, filepath: str):
        """Save trained model"""
        model_data = {
            'embedding_dim': self.embedding_dim,
            'window_size': self.window_size,
            'model_type': self.model_type,
            'vocab': self.vocab,
            'index_to_word': self.index_to_word,
            'word_counts': self.word_counts,
            'vocab_size': self.vocab_size,
            'W_in': self.W_in,
            'W_out': self.W_out,
            'loss_history': self.loss_history
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
        self.model_type = model_data['model_type']
        self.vocab = model_data['vocab']
        self.index_to_word = model_data['index_to_word']
        self.word_counts = model_data['word_counts']
        self.vocab_size = model_data['vocab_size']
        self.W_in = model_data['W_in']
        self.W_out = model_data['W_out']
        self.loss_history = model_data['loss_history']
        print(f"Model loaded from {filepath}")


def create_sample_corpus():
    """Create a sample corpus for demonstration"""
    corpus = [
        "the king is strong and powerful",
        "the queen is beautiful and elegant", 
        "man and woman are different",
        "prince and princess live in castle",
        "dog and cat are pets",
        "puppy is a young dog",
        "kitten is a young cat",
        "bird flies in the sky",
        "fish swims in water",
        "tree grows in forest",
        "flower blooms in garden",
        "sun shines during day",
        "moon glows at night",
        "rain falls from clouds",
        "wind blows through trees",
        "fire burns bright",
        "water flows in river",
        "mountain stands tall",
        "valley lies low",
        "ocean is vast and deep",
        "king rules the kingdom",
        "queen helps the king",
        "prince will become king",
        "princess will become queen",
        "man works hard",
        "woman is intelligent",
        "child plays happily",
        "baby sleeps peacefully",
        "dog barks loudly",
        "cat meows softly",
        "bird sings beautifully",
        "fish swims gracefully"
    ] * 20  # Repeat for more training data
    
    return corpus


def demonstrate_word2vec():
    """Demonstrate Word2Vec functionality"""
    print("Word2Vec Demonstration")
    print("=" * 50)
    
    # Create sample corpus
    corpus = create_sample_corpus()
    print(f"Corpus size: {len(corpus)} sentences")
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Train Skip-gram model
    print("\n" + "=" * 50)
    print("Training Skip-gram Model")
    print("=" * 50)
    
    skipgram_model = Word2Vec(
        embedding_dim=50,
        window_size=3,
        min_count=2,
        negative_samples=5,
        learning_rate=0.025,
        epochs=10,
        model_type='skipgram'
    )
    
    skipgram_model.fit(corpus)
    skipgram_model.save_model('plots/skipgram_model.pkl')
    
    # Train CBOW model
    print("\n" + "=" * 50)
    print("Training CBOW Model")
    print("=" * 50)
    
    cbow_model = Word2Vec(
        embedding_dim=50,
        window_size=3,
        min_count=2,
        negative_samples=5,
        learning_rate=0.025,
        epochs=10,
        model_type='cbow'
    )
    
    cbow_model.fit(corpus)
    cbow_model.save_model('plots/cbow_model.pkl')
    
    # Compare models
    print("\n" + "=" * 50)
    print("Model Comparison")
    print("=" * 50)
    
    test_words = ['king', 'queen', 'man', 'woman', 'dog', 'cat']
    
    for word in test_words:
        if word in skipgram_model.vocab:
            print(f"\nSimilar words to '{word}':")
            
            print("Skip-gram:")
            sg_similar = skipgram_model.most_similar(word, top_k=5)
            for similar_word, score in sg_similar:
                print(f"  {similar_word}: {score:.3f}")
            
            print("CBOW:")
            cbow_similar = cbow_model.most_similar(word, top_k=5)
            for similar_word, score in cbow_similar:
                print(f"  {similar_word}: {score:.3f}")
    
    # Test analogies
    print("\n" + "=" * 50)
    print("Analogy Tests")
    print("=" * 50)
    
    analogy_tests = [
        ('king', 'man', 'queen'),  # king - man + queen = ?
        ('prince', 'king', 'princess'),  # prince - king + princess = ?
        ('dog', 'puppy', 'cat'),  # dog - puppy + cat = ?
    ]
    
    for word_a, word_b, word_c in analogy_tests:
        print(f"\n{word_a} - {word_b} + {word_c} = ?")
        
        print("Skip-gram:")
        sg_analogy = skipgram_model.analogy(word_a, word_b, word_c, top_k=3)
        for word, score in sg_analogy:
            print(f"  {word}: {score:.3f}")
        
        print("CBOW:")
        cbow_analogy = cbow_model.analogy(word_a, word_b, word_c, top_k=3)
        for word, score in cbow_analogy:
            print(f"  {word}: {score:.3f}")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(skipgram_model.loss_history, label='Skip-gram', color='blue')
    plt.title('Skip-gram Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(cbow_model.loss_history, label='CBOW', color='red')
    plt.title('CBOW Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('plots/word2vec_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Compare training curves
    plt.figure(figsize=(10, 6))
    plt.plot(skipgram_model.loss_history, label='Skip-gram', color='blue', linewidth=2)
    plt.plot(cbow_model.loss_history, label='CBOW', color='red', linewidth=2)
    plt.title('Word2Vec Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('plots/word2vec_loss_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to 'plots/' directory")
    print("Word2Vec demonstration completed!")
    
    return skipgram_model, cbow_model


if __name__ == "__main__":
    demonstrate_word2vec()