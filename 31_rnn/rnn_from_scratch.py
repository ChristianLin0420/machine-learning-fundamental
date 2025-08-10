"""
RNN from Scratch Implementation
===============================

This module implements a Recurrent Neural Network (RNN) from scratch using only NumPy.
It demonstrates the core concepts of sequence processing, hidden states, and 
Backpropagation Through Time (BPTT).

Key Features:
- Character-level language modeling
- Forward pass with hidden state propagation
- Backpropagation Through Time (BPTT)
- Text generation and sampling
- Gradient clipping for stability
- Comprehensive analysis and visualization

Author: Assistant
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import os
import time
import pickle

# Set random seed for reproducibility
np.random.seed(42)

class VanillaRNN:
    """
    Vanilla RNN implementation from scratch for character-level language modeling.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 learning_rate: float = 0.1, seq_length: int = 25):
        """
        Initialize the RNN with random weights.
        
        Args:
            input_size: Size of input vocabulary
            hidden_size: Number of hidden units
            output_size: Size of output vocabulary (usually same as input_size)
            learning_rate: Learning rate for gradient descent
            seq_length: Length of sequences for truncated BPTT
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.seq_length = seq_length
        
        # Initialize weights with more conservative Xavier initialization
        self.W_xh = np.random.randn(hidden_size, input_size) * np.sqrt(1.0 / input_size)
        self.W_hh = np.random.randn(hidden_size, hidden_size) * np.sqrt(1.0 / hidden_size)
        self.W_hy = np.random.randn(output_size, hidden_size) * np.sqrt(1.0 / hidden_size)
        
        # Initialize biases
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))
        
        # Initialize hidden state
        self.h_prev = np.zeros((hidden_size, 1))
        
        # For gradient clipping and monitoring
        self.gradient_norms = []
        self.losses = []
        self.training_history = {
            'loss': [],
            'perplexity': [],
            'gradient_norm': [],
            'iteration': []
        }
    
    def _stable_softmax(self, x):
        """
        Numerically stable softmax implementation.
        
        Args:
            x: Input logits
            
        Returns:
            Softmax probabilities
        """
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
        
    def forward(self, inputs: List[int], h_prev: np.ndarray) -> Tuple[Dict, Dict, Dict, Dict, float]:
        """
        Forward pass through the RNN.
        
        Args:
            inputs: List of input character indices
            h_prev: Previous hidden state
            
        Returns:
            Tuple of (xs, hs, ys, ps, loss) where:
                xs: input vectors (one-hot)
                hs: hidden states
                ys: output logits
                ps: output probabilities
                loss: cross-entropy loss
        """
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(h_prev)
        loss = 0
        
        # Forward pass through sequence
        for t in range(len(inputs)):
            # Convert input to one-hot vector
            xs[t] = np.zeros((self.input_size, 1))
            xs[t][inputs[t]] = 1
            
            # Hidden state computation
            hs[t] = np.tanh(np.dot(self.W_xh, xs[t]) + 
                           np.dot(self.W_hh, hs[t-1]) + self.b_h)
            
            # Output computation
            ys[t] = np.dot(self.W_hy, hs[t]) + self.b_y
            
            # Softmax for probabilities (numerically stable)
            ps[t] = self._stable_softmax(ys[t])
            
            # Cross-entropy loss (if we have targets)
            if t < len(inputs) - 1:  # We predict next character
                target = inputs[t + 1]
                # Clip probability to avoid log(0)
                prob = np.clip(ps[t][target, 0], 1e-15, 1.0)
                loss += -np.log(prob)
        
        return xs, hs, ys, ps, loss
    
    def backward(self, inputs: List[int], xs: Dict, hs: Dict, 
                ys: Dict, ps: Dict) -> Tuple[Dict, float]:
        """
        Backward pass (Backpropagation Through Time).
        
        Args:
            inputs: List of input character indices
            xs: Input vectors from forward pass
            hs: Hidden states from forward pass
            ys: Output logits from forward pass
            ps: Output probabilities from forward pass
            
        Returns:
            Tuple of (gradients, gradient_norm)
        """
        # Initialize gradients
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        dh_next = np.zeros_like(hs[0])
        
        # Backward pass through sequence
        for t in reversed(range(len(inputs) - 1)):
            # Output layer gradients
            dy = np.copy(ps[t])
            dy[inputs[t + 1]] -= 1  # Cross-entropy derivative
            
            dW_hy += np.dot(dy, hs[t].T)
            db_y += dy
            
            # Hidden layer gradients
            dh = np.dot(self.W_hy.T, dy) + dh_next
            
            # Through tanh nonlinearity
            dh_raw = (1 - hs[t] * hs[t]) * dh
            
            # Input-to-hidden gradients
            dW_xh += np.dot(dh_raw, xs[t].T)
            
            # Hidden-to-hidden gradients
            dW_hh += np.dot(dh_raw, hs[t-1].T)
            
            # Bias gradients
            db_h += dh_raw
            
            # Gradient for next timestep
            dh_next = np.dot(self.W_hh.T, dh_raw)
        
        # Clip gradients to prevent exploding gradients
        gradients = {'dW_xh': dW_xh, 'dW_hh': dW_hh, 'dW_hy': dW_hy,
                    'db_h': db_h, 'db_y': db_y}
        
        # Calculate gradient norm for monitoring
        gradient_norm = 0
        for grad in gradients.values():
            gradient_norm += np.sum(grad * grad)
        gradient_norm = np.sqrt(gradient_norm)
        
        # Clip gradients more conservatively
        for key in gradients:
            np.clip(gradients[key], -1, 1, out=gradients[key])
        
        return gradients, gradient_norm
    
    def update_weights(self, gradients: Dict) -> None:
        """
        Update weights using gradient descent.
        
        Args:
            gradients: Dictionary of gradients for each parameter
        """
        self.W_xh -= self.learning_rate * gradients['dW_xh']
        self.W_hh -= self.learning_rate * gradients['dW_hh']
        self.W_hy -= self.learning_rate * gradients['dW_hy']
        self.b_h -= self.learning_rate * gradients['db_h']
        self.b_y -= self.learning_rate * gradients['db_y']
    
    def sample(self, seed_ix: int, n: int, h_prev: Optional[np.ndarray] = None) -> List[int]:
        """
        Sample a sequence of characters from the model.
        
        Args:
            seed_ix: Starting character index
            n: Number of characters to generate
            h_prev: Initial hidden state (optional)
            
        Returns:
            List of generated character indices
        """
        if h_prev is None:
            h = np.zeros((self.hidden_size, 1))
        else:
            h = np.copy(h_prev)
        
        x = np.zeros((self.input_size, 1))
        x[seed_ix] = 1
        
        generated_indices = []
        
        for t in range(n):
            # Forward pass for single timestep
            h = np.tanh(np.dot(self.W_xh, x) + np.dot(self.W_hh, h) + self.b_h)
            y = np.dot(self.W_hy, h) + self.b_y
            p = self._stable_softmax(y)
            
            # Sample next character (handle potential NaN/inf values)
            p_flat = p.ravel()
            if np.any(np.isnan(p_flat)) or np.any(np.isinf(p_flat)):
                # If probabilities are invalid, use uniform distribution
                p_flat = np.ones(self.input_size) / self.input_size
            else:
                # Normalize to ensure probabilities sum to 1
                p_flat = p_flat / np.sum(p_flat)
            
            ix = np.random.choice(range(self.input_size), p=p_flat)
            
            # Prepare input for next timestep
            x = np.zeros((self.input_size, 1))
            x[ix] = 1
            
            generated_indices.append(ix)
        
        return generated_indices
    
    def train_step(self, inputs: List[int]) -> Tuple[float, float]:
        """
        Perform one training step.
        
        Args:
            inputs: List of input character indices
            
        Returns:
            Tuple of (loss, gradient_norm)
        """
        # Forward pass
        xs, hs, ys, ps, loss = self.forward(inputs, self.h_prev)
        
        # Backward pass
        gradients, gradient_norm = self.backward(inputs, xs, hs, ys, ps)
        
        # Update weights
        self.update_weights(gradients)
        
        # Update hidden state for next iteration
        self.h_prev = hs[len(inputs) - 2]  # Last hidden state
        
        return loss, gradient_norm
    
    def train(self, data: str, epochs: int, verbose: bool = True) -> Dict[str, List]:
        """
        Train the RNN on text data.
        
        Args:
            data: Training text
            epochs: Number of training epochs
            verbose: Whether to print progress
            
        Returns:
            Training history dictionary
        """
        if verbose:
            print(f"Training RNN for {epochs} epochs...")
            print(f"Data length: {len(data)} characters")
            print(f"Vocabulary size: {self.input_size}")
            print(f"Hidden size: {self.hidden_size}")
            print(f"Sequence length: {self.seq_length}")
        
        # Character to index mapping
        chars = list(set(data))
        char_to_ix = {ch: i for i, ch in enumerate(chars)}
        ix_to_char = {i: ch for i, ch in enumerate(chars)}
        
        # Store mappings
        self.char_to_ix = char_to_ix
        self.ix_to_char = ix_to_char
        
        data_size = len(data)
        iter_num = 0
        smooth_loss = -np.log(1.0 / self.input_size) * self.seq_length
        
        for epoch in range(epochs):
            # Reset hidden state for new epoch
            self.h_prev = np.zeros((self.hidden_size, 1))
            p = 0  # Data pointer
            
            while p < data_size - self.seq_length - 1:
                # Prepare inputs and targets
                inputs = [char_to_ix[ch] for ch in data[p:p + self.seq_length]]
                
                # Training step
                loss, gradient_norm = self.train_step(inputs)
                
                # Smooth loss for monitoring
                smooth_loss = smooth_loss * 0.999 + loss * 0.001
                
                # Store training history
                if iter_num % 100 == 0:
                    perplexity = np.exp(smooth_loss / self.seq_length)
                    self.training_history['loss'].append(smooth_loss)
                    self.training_history['perplexity'].append(perplexity)
                    self.training_history['gradient_norm'].append(gradient_norm)
                    self.training_history['iteration'].append(iter_num)
                    
                    if verbose and iter_num % 1000 == 0:
                        print(f'Epoch {epoch+1}, Iter {iter_num}, Loss: {smooth_loss:.4f}, '
                              f'Perplexity: {perplexity:.4f}, Grad Norm: {gradient_norm:.4f}')
                        
                        # Generate sample text
                        sample_ix = self.sample(char_to_ix[data[p]], 100, self.h_prev)
                        sample_text = ''.join([ix_to_char[ix] for ix in sample_ix])
                        print(f'Sample: "{sample_text[:50]}..."')
                        print('-' * 80)
                
                p += self.seq_length
                iter_num += 1
        
        if verbose:
            print("Training completed!")
        
        return self.training_history


class TextDataLoader:
    """
    Text data loader for character-level language modeling.
    """
    
    def __init__(self):
        """Initialize the data loader."""
        self.sample_texts = {
            'simple': "hello world! this is a simple test for our rnn implementation. " * 50,
            'shakespeare': self._get_shakespeare_sample(),
            'code': self._get_code_sample(),
            'numbers': ''.join([str(i % 10) for i in range(1000)])
        }
    
    def _get_shakespeare_sample(self) -> str:
        """Get a sample of Shakespeare-like text."""
        return """
        To be or not to be, that is the question:
        Whether 'tis nobler in the mind to suffer
        The slings and arrows of outrageous fortune,
        Or to take arms against a sea of troubles
        And by opposing end them. To die—to sleep,
        No more; and by a sleep to say we end
        The heart-ache and the thousand natural shocks
        That flesh is heir to: 'tis a consummation
        Devoutly to be wish'd. To die, to sleep;
        To sleep, perchance to dream—ay, there's the rub:
        For in that sleep of death what dreams may come,
        When we have shuffled off this mortal coil,
        Must give us pause—there's the respect
        That makes calamity of so long life.
        """ * 10
    
    def _get_code_sample(self) -> str:
        """Get a sample of code-like text."""
        return """
        def factorial(n):
            if n <= 1:
                return 1
            else:
                return n * factorial(n - 1)
        
        def fibonacci(n):
            if n <= 1:
                return n
            else:
                return fibonacci(n - 1) + fibonacci(n - 2)
        
        for i in range(10):
            print(f"factorial({i}) = {factorial(i)}")
            print(f"fibonacci({i}) = {fibonacci(i)}")
        """ * 5
    
    def get_data(self, dataset_name: str = 'simple') -> str:
        """
        Get training data.
        
        Args:
            dataset_name: Name of dataset to load
            
        Returns:
            Training text
        """
        if dataset_name not in self.sample_texts:
            raise ValueError(f"Dataset {dataset_name} not available. "
                           f"Choose from: {list(self.sample_texts.keys())}")
        
        return self.sample_texts[dataset_name]
    
    def analyze_data(self, text: str) -> Dict[str, Any]:
        """
        Analyze text data properties.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with analysis results
        """
        chars = list(set(text))
        char_counts = {ch: text.count(ch) for ch in chars}
        
        analysis = {
            'length': len(text),
            'vocab_size': len(chars),
            'chars': sorted(chars),
            'char_counts': char_counts,
            'most_common': sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'entropy': self._calculate_entropy(char_counts, len(text))
        }
        
        return analysis
    
    def _calculate_entropy(self, char_counts: Dict[str, int], total_chars: int) -> float:
        """Calculate entropy of text."""
        entropy = 0
        for count in char_counts.values():
            p = count / total_chars
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy


def run_rnn_experiments():
    """
    Run comprehensive RNN experiments.
    
    Returns:
        Dictionary with experiment results
    """
    print("RNN from Scratch - Comprehensive Experiments")
    print("=" * 60)
    
    # Create output directory
    os.makedirs('plots', exist_ok=True)
    
    # Initialize data loader
    data_loader = TextDataLoader()
    
    results = {}
    
    # Experiment 1: Simple text
    print("\n" + "="*50)
    print("EXPERIMENT 1: Simple Text")
    print("="*50)
    
    simple_text = data_loader.get_data('simple')
    simple_analysis = data_loader.analyze_data(simple_text)
    
    print(f"Text length: {simple_analysis['length']}")
    print(f"Vocabulary size: {simple_analysis['vocab_size']}")
    print(f"Entropy: {simple_analysis['entropy']:.2f} bits")
    print(f"Most common chars: {simple_analysis['most_common'][:5]}")
    
    # Train RNN on simple text
    rnn_simple = VanillaRNN(
        input_size=simple_analysis['vocab_size'],
        hidden_size=50,
        output_size=simple_analysis['vocab_size'],
        learning_rate=0.1,
        seq_length=25
    )
    
    history_simple = rnn_simple.train(simple_text, epochs=10, verbose=True)
    
    # Generate sample text
    print("\nGenerating sample text...")
    seed_char = simple_text[0]
    seed_ix = rnn_simple.char_to_ix[seed_char]
    sample_indices = rnn_simple.sample(seed_ix, 200)
    sample_text = ''.join([rnn_simple.ix_to_char[ix] for ix in sample_indices])
    print(f"Generated text: {sample_text}")
    
    results['simple'] = {
        'rnn': rnn_simple,
        'history': history_simple,
        'analysis': simple_analysis,
        'sample_text': sample_text
    }
    
    # Experiment 2: Different architectures
    print("\n" + "="*50)
    print("EXPERIMENT 2: Architecture Comparison")
    print("="*50)
    
    architectures = {
        'Small': {'hidden_size': 25, 'seq_length': 15},
        'Medium': {'hidden_size': 50, 'seq_length': 25},
        'Large': {'hidden_size': 100, 'seq_length': 35}
    }
    
    arch_results = {}
    
    for name, config in architectures.items():
        print(f"\nTraining {name} RNN...")
        rnn = VanillaRNN(
            input_size=simple_analysis['vocab_size'],
            hidden_size=config['hidden_size'],
            output_size=simple_analysis['vocab_size'],
            learning_rate=0.1,
            seq_length=config['seq_length']
        )
        
        # Train for fewer epochs for comparison
        history = rnn.train(simple_text, epochs=5, verbose=False)
        
        # Generate sample
        seed_ix = rnn.char_to_ix[simple_text[0]]
        sample_indices = rnn.sample(seed_ix, 100)
        sample_text = ''.join([rnn.ix_to_char[ix] for ix in sample_indices])
        
        arch_results[name] = {
            'rnn': rnn,
            'history': history,
            'config': config,
            'sample_text': sample_text,
            'final_loss': history['loss'][-1] if history['loss'] else float('inf')
        }
        
        print(f"Final loss: {arch_results[name]['final_loss']:.4f}")
        print(f"Sample: {sample_text[:50]}...")
    
    results['architectures'] = arch_results
    
    # Experiment 3: Different datasets
    print("\n" + "="*50)
    print("EXPERIMENT 3: Dataset Comparison")
    print("="*50)
    
    datasets = ['simple', 'shakespeare', 'code', 'numbers']
    dataset_results = {}
    
    for dataset_name in datasets:
        print(f"\nTraining on {dataset_name} dataset...")
        
        text = data_loader.get_data(dataset_name)
        analysis = data_loader.analyze_data(text)
        
        print(f"Vocabulary size: {analysis['vocab_size']}")
        print(f"Text entropy: {analysis['entropy']:.2f} bits")
        
        rnn = VanillaRNN(
            input_size=analysis['vocab_size'],
            hidden_size=50,
            output_size=analysis['vocab_size'],
            learning_rate=0.1,
            seq_length=25
        )
        
        # Train for fewer epochs for comparison
        history = rnn.train(text, epochs=3, verbose=False)
        
        # Generate sample
        if text:
            seed_ix = rnn.char_to_ix[text[0]]
            sample_indices = rnn.sample(seed_ix, 150)
            sample_text = ''.join([rnn.ix_to_char[ix] for ix in sample_indices])
        else:
            sample_text = ""
        
        dataset_results[dataset_name] = {
            'rnn': rnn,
            'history': history,
            'analysis': analysis,
            'sample_text': sample_text,
            'final_loss': history['loss'][-1] if history['loss'] else float('inf')
        }
        
        print(f"Final loss: {dataset_results[dataset_name]['final_loss']:.4f}")
        print(f"Sample: {sample_text[:50]}...")
    
    results['datasets'] = dataset_results
    
    return results


def visualize_rnn_results(results: Dict[str, Any]) -> None:
    """
    Create comprehensive visualizations of RNN results.
    
    Args:
        results: Results from run_rnn_experiments()
    """
    print("\nCreating visualizations...")
    
    # 1. Training curves for simple RNN
    if 'simple' in results:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        history = results['simple']['history']
        
        # Loss curve
        axes[0, 0].plot(history['iteration'], history['loss'], 'b-', linewidth=2)
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Perplexity curve
        axes[0, 1].plot(history['iteration'], history['perplexity'], 'r-', linewidth=2)
        axes[0, 1].set_title('Perplexity')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Perplexity')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Gradient norm
        axes[1, 0].plot(history['iteration'], history['gradient_norm'], 'g-', linewidth=2)
        axes[1, 0].set_title('Gradient Norm')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Gradient Norm')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Character frequency in original vs generated text
        original_text = results['simple']['analysis']['char_counts']
        sample_text = results['simple']['sample_text']
        sample_counts = {ch: sample_text.count(ch) for ch in set(sample_text)}
        
        chars = sorted(set(list(original_text.keys()) + list(sample_counts.keys())))
        orig_freqs = [original_text.get(ch, 0) for ch in chars]
        sample_freqs = [sample_counts.get(ch, 0) for ch in chars]
        
        x = np.arange(len(chars))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, orig_freqs, width, label='Original', alpha=0.8)
        axes[1, 1].bar(x + width/2, sample_freqs, width, label='Generated', alpha=0.8)
        axes[1, 1].set_title('Character Frequency Comparison')
        axes[1, 1].set_xlabel('Characters')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(chars, rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/rnn_training_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 2. Architecture comparison
    if 'architectures' in results:
        arch_results = results['architectures']
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Final loss comparison
        names = list(arch_results.keys())
        losses = [arch_results[name]['final_loss'] for name in names]
        hidden_sizes = [arch_results[name]['config']['hidden_size'] for name in names]
        
        bars = axes[0].bar(names, losses, color=['skyblue', 'lightgreen', 'salmon'])
        axes[0].set_title('Final Loss by Architecture')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, loss in zip(bars, losses):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{loss:.3f}', ha='center', va='bottom')
        
        # Training curves comparison
        for name in names:
            history = arch_results[name]['history']
            if history['iteration']:
                axes[1].plot(history['iteration'], history['loss'], 
                           label=f'{name} (h={arch_results[name]["config"]["hidden_size"]})',
                           linewidth=2)
        
        axes[1].set_title('Training Loss Comparison')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/rnn_architecture_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 3. Dataset comparison
    if 'datasets' in results:
        dataset_results = results['datasets']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Dataset properties
        dataset_names = list(dataset_results.keys())
        vocab_sizes = [dataset_results[name]['analysis']['vocab_size'] for name in dataset_names]
        entropies = [dataset_results[name]['analysis']['entropy'] for name in dataset_names]
        final_losses = [dataset_results[name]['final_loss'] for name in dataset_names]
        
        # Vocabulary size
        bars1 = axes[0, 0].bar(dataset_names, vocab_sizes, color='lightblue')
        axes[0, 0].set_title('Vocabulary Size by Dataset')
        axes[0, 0].set_ylabel('Vocabulary Size')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, size in zip(bars1, vocab_sizes):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                          f'{size}', ha='center', va='bottom')
        
        # Entropy
        bars2 = axes[0, 1].bar(dataset_names, entropies, color='lightcoral')
        axes[0, 1].set_title('Text Entropy by Dataset')
        axes[0, 1].set_ylabel('Entropy (bits)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Final loss
        bars3 = axes[1, 0].bar(dataset_names, final_losses, color='lightgreen')
        axes[1, 0].set_title('Final Training Loss by Dataset')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Training curves for all datasets
        for name in dataset_names:
            history = dataset_results[name]['history']
            if history['iteration']:
                axes[1, 1].plot(history['iteration'], history['loss'], 
                               label=name, linewidth=2)
        
        axes[1, 1].set_title('Training Loss by Dataset')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/rnn_dataset_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    # Run comprehensive RNN experiments
    print("Starting RNN from Scratch Implementation...")
    
    results = run_rnn_experiments()
    
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    visualize_rnn_results(results)
    
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    # Print summary of results
    if 'simple' in results:
        final_loss = results['simple']['history']['loss'][-1]
        print(f"Simple RNN final loss: {final_loss:.4f}")
        print(f"Generated sample: {results['simple']['sample_text'][:100]}...")
    
    if 'architectures' in results:
        print("\nArchitecture Comparison:")
        for name, result in results['architectures'].items():
            print(f"  {name}: Loss = {result['final_loss']:.4f}")
    
    if 'datasets' in results:
        print("\nDataset Comparison:")
        for name, result in results['datasets'].items():
            print(f"  {name}: Loss = {result['final_loss']:.4f}, "
                  f"Vocab = {result['analysis']['vocab_size']}")
    
    print("\n" + "="*60)
    print("RNN IMPLEMENTATION COMPLETE!")
    print("="*60)
    print("Check the 'plots/' directory for visualizations.")
    
    # Save results for later analysis
    with open('plots/rnn_results.pkl', 'wb') as f:
        # Save only serializable parts
        save_results = {
            'simple_history': results.get('simple', {}).get('history', {}),
            'architecture_comparison': {
                name: {'history': result['history'], 'config': result['config'], 'final_loss': result['final_loss']}
                for name, result in results.get('architectures', {}).items()
            },
            'dataset_comparison': {
                name: {'history': result['history'], 'analysis': result['analysis'], 'final_loss': result['final_loss']}
                for name, result in results.get('datasets', {}).items()
            }
        }
        pickle.dump(save_results, f)
    
    print("Results saved to 'plots/rnn_results.pkl'")