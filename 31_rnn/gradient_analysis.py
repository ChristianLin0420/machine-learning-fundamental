"""
RNN Gradient Analysis
====================

This module analyzes the vanishing gradient problem in RNNs and demonstrates
various techniques to understand and mitigate gradient-related issues.

Key Features:
- Gradient flow analysis through time
- Vanishing/exploding gradient detection
- Sequence length impact on gradients
- Weight initialization effects
- Gradient clipping analysis

Author: Assistant
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import os
from rnn_from_scratch import VanillaRNN, TextDataLoader

class GradientAnalyzer:
    """
    Analyze gradient behavior in RNNs.
    """
    
    def __init__(self):
        """Initialize the gradient analyzer."""
        self.analysis_results = {}
    
    def analyze_gradient_flow(self, rnn: VanillaRNN, text: str, 
                            sequence_lengths: List[int] = [5, 10, 25, 50]) -> Dict[str, Any]:
        """
        Analyze gradient flow for different sequence lengths.
        
        Args:
            rnn: RNN model
            text: Training text
            sequence_lengths: List of sequence lengths to test
            
        Returns:
            Dictionary with gradient analysis results
        """
        print("Analyzing gradient flow...")
        
        # Character to index mapping
        chars = list(set(text))
        char_to_ix = {ch: i for i, ch in enumerate(chars)}
        
        results = {}
        
        for seq_len in sequence_lengths:
            print(f"Analyzing sequence length: {seq_len}")
            
            # Create a copy of RNN with specific sequence length
            test_rnn = VanillaRNN(
                input_size=len(chars),
                hidden_size=rnn.hidden_size,
                output_size=len(chars),
                learning_rate=rnn.learning_rate,
                seq_length=seq_len
            )
            
            # Copy weights from original RNN
            test_rnn.W_xh = rnn.W_xh.copy()
            test_rnn.W_hh = rnn.W_hh.copy()
            test_rnn.W_hy = rnn.W_hy.copy()
            test_rnn.b_h = rnn.b_h.copy()
            test_rnn.b_y = rnn.b_y.copy()
            
            # Prepare input sequence
            inputs = [char_to_ix[ch] for ch in text[:seq_len]]
            
            # Forward pass
            xs, hs, ys, ps, loss = test_rnn.forward(inputs, test_rnn.h_prev)
            
            # Modified backward pass to track gradient magnitudes through time
            gradient_magnitudes = self._analyze_bptt_gradients(test_rnn, inputs, xs, hs, ys, ps)
            
            results[seq_len] = {
                'gradient_magnitudes': gradient_magnitudes,
                'loss': loss,
                'final_gradient_norm': np.mean([np.linalg.norm(grad) for grad in gradient_magnitudes])
            }
        
        return results
    
    def _analyze_bptt_gradients(self, rnn: VanillaRNN, inputs: List[int], 
                               xs: Dict, hs: Dict, ys: Dict, ps: Dict) -> List[np.ndarray]:
        """
        Analyze gradients during BPTT with detailed tracking.
        
        Args:
            rnn: RNN model
            inputs: Input sequence
            xs, hs, ys, ps: Forward pass results
            
        Returns:
            List of gradient magnitudes at each timestep
        """
        gradient_magnitudes = []
        dh_next = np.zeros_like(hs[0])
        
        for t in reversed(range(len(inputs) - 1)):
            # Output layer gradients
            dy = np.copy(ps[t])
            dy[inputs[t + 1]] -= 1
            
            # Hidden layer gradients
            dh = np.dot(rnn.W_hy.T, dy) + dh_next
            
            # Store gradient magnitude
            gradient_magnitudes.append(dh.copy())
            
            # Through tanh nonlinearity
            dh_raw = (1 - hs[t] * hs[t]) * dh
            
            # Gradient for next timestep
            dh_next = np.dot(rnn.W_hh.T, dh_raw)
        
        # Reverse to get chronological order
        return list(reversed(gradient_magnitudes))
    
    def analyze_weight_initialization_effects(self, vocab_size: int, hidden_size: int,
                                            text: str) -> Dict[str, Any]:
        """
        Analyze effect of different weight initializations on gradient flow.
        
        Args:
            vocab_size: Size of vocabulary
            hidden_size: Hidden layer size
            text: Training text
            
        Returns:
            Dictionary with initialization analysis results
        """
        print("Analyzing weight initialization effects...")
        
        initialization_methods = {
            'Xavier': lambda fan_in, fan_out: np.random.randn(fan_out, fan_in) * np.sqrt(2.0 / fan_in),
            'Small Random': lambda fan_in, fan_out: np.random.randn(fan_out, fan_in) * 0.01,
            'Large Random': lambda fan_in, fan_out: np.random.randn(fan_out, fan_in) * 0.5,
            'Identity': lambda fan_in, fan_out: np.eye(fan_out, fan_in) if fan_in == fan_out else np.random.randn(fan_out, fan_in) * 0.01
        }
        
        results = {}
        chars = list(set(text))
        char_to_ix = {ch: i for i, ch in enumerate(chars)}
        
        for method_name, init_func in initialization_methods.items():
            print(f"Testing {method_name} initialization...")
            
            # Create RNN with specific initialization
            rnn = VanillaRNN(vocab_size, hidden_size, vocab_size)
            
            # Apply initialization
            rnn.W_xh = init_func(vocab_size, hidden_size)
            if method_name == 'Identity' and vocab_size == hidden_size:
                rnn.W_hh = np.eye(hidden_size) * 0.5  # Scaled identity
            else:
                rnn.W_hh = init_func(hidden_size, hidden_size)
            rnn.W_hy = init_func(hidden_size, vocab_size)
            
            # Test gradient flow
            seq_len = 25
            inputs = [char_to_ix[ch] for ch in text[:seq_len]]
            
            xs, hs, ys, ps, loss = rnn.forward(inputs, rnn.h_prev)
            gradient_magnitudes = self._analyze_bptt_gradients(rnn, inputs, xs, hs, ys, ps)
            
            # Calculate statistics
            grad_norms = [np.linalg.norm(grad) for grad in gradient_magnitudes]
            
            results[method_name] = {
                'gradient_norms': grad_norms,
                'mean_gradient_norm': np.mean(grad_norms),
                'gradient_variance': np.var(grad_norms),
                'gradient_decay_rate': self._calculate_decay_rate(grad_norms),
                'loss': loss
            }
        
        return results
    
    def _calculate_decay_rate(self, gradient_norms: List[float]) -> float:
        """Calculate the rate of gradient decay through time."""
        if len(gradient_norms) < 2:
            return 0.0
        
        # Fit exponential decay: grad(t) = grad(0) * exp(-decay_rate * t)
        log_grads = np.log(np.maximum(gradient_norms, 1e-10))
        timesteps = np.arange(len(gradient_norms))
        
        # Linear regression on log scale
        A = np.vstack([timesteps, np.ones(len(timesteps))]).T
        decay_rate, _ = np.linalg.lstsq(A, log_grads, rcond=None)[0]
        
        return -decay_rate  # Negative because we want decay rate
    
    def analyze_sequence_length_impact(self, vocab_size: int, hidden_size: int,
                                     text: str, max_seq_length: int = 100) -> Dict[str, Any]:
        """
        Analyze impact of sequence length on training dynamics.
        
        Args:
            vocab_size: Size of vocabulary
            hidden_size: Hidden layer size
            text: Training text
            max_seq_length: Maximum sequence length to test
            
        Returns:
            Dictionary with sequence length analysis results
        """
        print("Analyzing sequence length impact...")
        
        sequence_lengths = list(range(5, max_seq_length + 1, 5))
        chars = list(set(text))
        char_to_ix = {ch: i for i, ch in enumerate(chars)}
        
        results = {
            'sequence_lengths': sequence_lengths,
            'losses': [],
            'gradient_norms': [],
            'training_times': [],
            'perplexities': []
        }
        
        for seq_len in sequence_lengths:
            print(f"Testing sequence length: {seq_len}")
            
            # Create RNN
            rnn = VanillaRNN(vocab_size, hidden_size, vocab_size, seq_length=seq_len)
            
            # Train for a few steps
            inputs = [char_to_ix[ch] for ch in text[:seq_len]]
            
            import time
            start_time = time.time()
            
            # Multiple training steps
            total_loss = 0
            total_grad_norm = 0
            num_steps = 10
            
            for step in range(num_steps):
                loss, grad_norm = rnn.train_step(inputs)
                total_loss += loss
                total_grad_norm += grad_norm
            
            training_time = time.time() - start_time
            
            avg_loss = total_loss / num_steps
            avg_grad_norm = total_grad_norm / num_steps
            perplexity = np.exp(avg_loss / seq_len)
            
            results['losses'].append(avg_loss)
            results['gradient_norms'].append(avg_grad_norm)
            results['training_times'].append(training_time)
            results['perplexities'].append(perplexity)
        
        return results
    
    def demonstrate_gradient_clipping(self, vocab_size: int, hidden_size: int,
                                    text: str) -> Dict[str, Any]:
        """
        Demonstrate the effect of gradient clipping.
        
        Args:
            vocab_size: Size of vocabulary
            hidden_size: Hidden layer size
            text: Training text
            
        Returns:
            Dictionary with gradient clipping analysis results
        """
        print("Analyzing gradient clipping effects...")
        
        chars = list(set(text))
        char_to_ix = {ch: i for i, ch in enumerate(chars)}
        
        # Test different clipping strategies
        clipping_methods = {
            'No Clipping': None,
            'Clip at 5': 5.0,
            'Clip at 1': 1.0,
            'Clip at 0.5': 0.5
        }
        
        results = {}
        
        for method_name, clip_value in clipping_methods.items():
            print(f"Testing {method_name}...")
            
            # Create RNN
            rnn = VanillaRNN(vocab_size, hidden_size, vocab_size)
            
            # Override gradient clipping if specified
            if clip_value is not None:
                original_backward = rnn.backward
                
                def clipped_backward(inputs, xs, hs, ys, ps):
                    gradients, grad_norm = original_backward(inputs, xs, hs, ys, ps)
                    
                    # Apply custom clipping
                    for key in gradients:
                        np.clip(gradients[key], -clip_value, clip_value, out=gradients[key])
                    
                    # Recalculate gradient norm after clipping
                    clipped_grad_norm = 0
                    for grad in gradients.values():
                        clipped_grad_norm += np.sum(grad * grad)
                    clipped_grad_norm = np.sqrt(clipped_grad_norm)
                    
                    return gradients, clipped_grad_norm
                
                rnn.backward = clipped_backward
            
            # Train for several steps and track progress
            seq_len = 25
            inputs = [char_to_ix[ch] for ch in text[:seq_len]]
            
            losses = []
            grad_norms = []
            
            for step in range(50):
                loss, grad_norm = rnn.train_step(inputs)
                losses.append(loss)
                grad_norms.append(grad_norm)
            
            results[method_name] = {
                'losses': losses,
                'gradient_norms': grad_norms,
                'final_loss': losses[-1],
                'mean_gradient_norm': np.mean(grad_norms),
                'gradient_stability': np.std(grad_norms)
            }
        
        return results


def visualize_gradient_analysis(analysis_results: Dict[str, Any]) -> None:
    """
    Create comprehensive visualizations of gradient analysis results.
    
    Args:
        analysis_results: Results from gradient analysis
    """
    print("Creating gradient analysis visualizations...")
    
    # Create output directory
    os.makedirs('plots', exist_ok=True)
    
    # 1. Gradient flow through time
    if 'gradient_flow' in analysis_results:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        gradient_results = analysis_results['gradient_flow']
        
        # Gradient magnitude vs timestep for different sequence lengths
        for seq_len, results in gradient_results.items():
            grad_norms = [np.linalg.norm(grad) for grad in results['gradient_magnitudes']]
            timesteps = range(len(grad_norms))
            axes[0, 0].plot(timesteps, grad_norms, 'o-', label=f'Seq Len {seq_len}', linewidth=2)
        
        axes[0, 0].set_title('Gradient Magnitude Through Time')
        axes[0, 0].set_xlabel('Timestep (backwards)')
        axes[0, 0].set_ylabel('Gradient Magnitude')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_yscale('log')
        
        # Final gradient norm vs sequence length
        seq_lens = list(gradient_results.keys())
        final_norms = [gradient_results[seq_len]['final_gradient_norm'] for seq_len in seq_lens]
        
        axes[0, 1].plot(seq_lens, final_norms, 'ro-', linewidth=2, markersize=8)
        axes[0, 1].set_title('Final Gradient Norm vs Sequence Length')
        axes[0, 1].set_xlabel('Sequence Length')
        axes[0, 1].set_ylabel('Final Gradient Norm')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('plots/gradient_flow_analysis.png', dpi=300, bbox_inches='tight')
    
    # 2. Weight initialization effects
    if 'weight_initialization' in analysis_results:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        init_results = analysis_results['weight_initialization']
        
        # Mean gradient norms
        methods = list(init_results.keys())
        mean_norms = [init_results[method]['mean_gradient_norm'] for method in methods]
        
        bars1 = axes[0, 0].bar(methods, mean_norms, color='skyblue', alpha=0.8)
        axes[0, 0].set_title('Mean Gradient Norm by Initialization')
        axes[0, 0].set_ylabel('Mean Gradient Norm')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Gradient variance
        variances = [init_results[method]['gradient_variance'] for method in methods]
        
        bars2 = axes[0, 1].bar(methods, variances, color='lightcoral', alpha=0.8)
        axes[0, 1].set_title('Gradient Variance by Initialization')
        axes[0, 1].set_ylabel('Gradient Variance')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Gradient decay rates
        decay_rates = [init_results[method]['gradient_decay_rate'] for method in methods]
        
        bars3 = axes[1, 0].bar(methods, decay_rates, color='lightgreen', alpha=0.8)
        axes[1, 0].set_title('Gradient Decay Rate by Initialization')
        axes[1, 0].set_ylabel('Decay Rate')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Gradient norms through time for each method
        for method in methods:
            grad_norms = init_results[method]['gradient_norms']
            timesteps = range(len(grad_norms))
            axes[1, 1].plot(timesteps, grad_norms, 'o-', label=method, linewidth=2)
        
        axes[1, 1].set_title('Gradient Norms Through Time')
        axes[1, 1].set_xlabel('Timestep (backwards)')
        axes[1, 1].set_ylabel('Gradient Norm')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('plots/weight_initialization_analysis.png', dpi=300, bbox_inches='tight')
    
    # 3. Sequence length impact
    if 'sequence_length' in analysis_results:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        seq_results = analysis_results['sequence_length']
        
        # Loss vs sequence length
        axes[0, 0].plot(seq_results['sequence_lengths'], seq_results['losses'], 
                       'bo-', linewidth=2, markersize=6)
        axes[0, 0].set_title('Loss vs Sequence Length')
        axes[0, 0].set_xlabel('Sequence Length')
        axes[0, 0].set_ylabel('Average Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Gradient norm vs sequence length
        axes[0, 1].plot(seq_results['sequence_lengths'], seq_results['gradient_norms'], 
                       'ro-', linewidth=2, markersize=6)
        axes[0, 1].set_title('Gradient Norm vs Sequence Length')
        axes[0, 1].set_xlabel('Sequence Length')
        axes[0, 1].set_ylabel('Average Gradient Norm')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
        
        # Training time vs sequence length
        axes[1, 0].plot(seq_results['sequence_lengths'], seq_results['training_times'], 
                       'go-', linewidth=2, markersize=6)
        axes[1, 0].set_title('Training Time vs Sequence Length')
        axes[1, 0].set_xlabel('Sequence Length')
        axes[1, 0].set_ylabel('Training Time (seconds)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Perplexity vs sequence length
        axes[1, 1].plot(seq_results['sequence_lengths'], seq_results['perplexities'], 
                       'mo-', linewidth=2, markersize=6)
        axes[1, 1].set_title('Perplexity vs Sequence Length')
        axes[1, 1].set_xlabel('Sequence Length')
        axes[1, 1].set_ylabel('Perplexity')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/sequence_length_analysis.png', dpi=300, bbox_inches='tight')
    
    # 4. Gradient clipping effects
    if 'gradient_clipping' in analysis_results:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        clip_results = analysis_results['gradient_clipping']
        
        # Training curves with different clipping
        for method, results in clip_results.items():
            steps = range(len(results['losses']))
            axes[0, 0].plot(steps, results['losses'], label=method, linewidth=2)
        
        axes[0, 0].set_title('Training Loss with Different Clipping')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Gradient norm curves
        for method, results in clip_results.items():
            steps = range(len(results['gradient_norms']))
            axes[0, 1].plot(steps, results['gradient_norms'], label=method, linewidth=2)
        
        axes[0, 1].set_title('Gradient Norms with Different Clipping')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Gradient Norm')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')
        
        # Final performance comparison
        methods = list(clip_results.keys())
        final_losses = [clip_results[method]['final_loss'] for method in methods]
        
        bars1 = axes[1, 0].bar(methods, final_losses, color='lightblue', alpha=0.8)
        axes[1, 0].set_title('Final Loss by Clipping Method')
        axes[1, 0].set_ylabel('Final Loss')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Gradient stability
        stabilities = [clip_results[method]['gradient_stability'] for method in methods]
        
        bars2 = axes[1, 1].bar(methods, stabilities, color='lightgreen', alpha=0.8)
        axes[1, 1].set_title('Gradient Stability by Clipping Method')
        axes[1, 1].set_ylabel('Gradient Standard Deviation')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/gradient_clipping_analysis.png', dpi=300, bbox_inches='tight')
    
    plt.show()


def run_comprehensive_gradient_analysis():
    """
    Run comprehensive gradient analysis experiments.
    
    Returns:
        Dictionary with all analysis results
    """
    print("RNN Gradient Analysis - Comprehensive Study")
    print("=" * 60)
    
    # Initialize data loader and get sample text
    data_loader = TextDataLoader()
    text = data_loader.get_data('simple')
    chars = list(set(text))
    vocab_size = len(chars)
    hidden_size = 50
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Hidden size: {hidden_size}")
    print(f"Text length: {len(text)}")
    
    # Initialize analyzer
    analyzer = GradientAnalyzer()
    
    # Create a baseline RNN
    baseline_rnn = VanillaRNN(vocab_size, hidden_size, vocab_size)
    
    analysis_results = {}
    
    # 1. Gradient flow analysis
    print("\n1. Analyzing gradient flow through time...")
    gradient_flow_results = analyzer.analyze_gradient_flow(baseline_rnn, text)
    analysis_results['gradient_flow'] = gradient_flow_results
    
    # 2. Weight initialization effects
    print("\n2. Analyzing weight initialization effects...")
    init_results = analyzer.analyze_weight_initialization_effects(vocab_size, hidden_size, text)
    analysis_results['weight_initialization'] = init_results
    
    # 3. Sequence length impact
    print("\n3. Analyzing sequence length impact...")
    seq_len_results = analyzer.analyze_sequence_length_impact(vocab_size, hidden_size, text, max_seq_length=50)
    analysis_results['sequence_length'] = seq_len_results
    
    # 4. Gradient clipping demonstration
    print("\n4. Analyzing gradient clipping effects...")
    clipping_results = analyzer.demonstrate_gradient_clipping(vocab_size, hidden_size, text)
    analysis_results['gradient_clipping'] = clipping_results
    
    return analysis_results


if __name__ == "__main__":
    # Run comprehensive gradient analysis
    analysis_results = run_comprehensive_gradient_analysis()
    
    print("\n" + "="*60)
    print("CREATING GRADIENT ANALYSIS VISUALIZATIONS")
    print("="*60)
    
    # Create visualizations
    visualize_gradient_analysis(analysis_results)
    
    print("\n" + "="*60)
    print("GRADIENT ANALYSIS SUMMARY")
    print("="*60)
    
    # Print key findings
    if 'gradient_flow' in analysis_results:
        print("\nGradient Flow Analysis:")
        for seq_len, results in analysis_results['gradient_flow'].items():
            print(f"  Sequence length {seq_len}: Final gradient norm = {results['final_gradient_norm']:.6f}")
    
    if 'weight_initialization' in analysis_results:
        print("\nWeight Initialization Analysis:")
        for method, results in analysis_results['weight_initialization'].items():
            print(f"  {method}: Mean gradient norm = {results['mean_gradient_norm']:.6f}, "
                  f"Decay rate = {results['gradient_decay_rate']:.6f}")
    
    if 'sequence_length' in analysis_results:
        seq_results = analysis_results['sequence_length']
        min_loss_idx = np.argmin(seq_results['losses'])
        optimal_seq_len = seq_results['sequence_lengths'][min_loss_idx]
        print(f"\nSequence Length Analysis:")
        print(f"  Optimal sequence length: {optimal_seq_len}")
        print(f"  Minimum loss: {seq_results['losses'][min_loss_idx]:.6f}")
    
    if 'gradient_clipping' in analysis_results:
        print("\nGradient Clipping Analysis:")
        for method, results in analysis_results['gradient_clipping'].items():
            print(f"  {method}: Final loss = {results['final_loss']:.6f}, "
                  f"Gradient stability = {results['gradient_stability']:.6f}")
    
    print("\n" + "="*60)
    print("GRADIENT ANALYSIS COMPLETE!")
    print("="*60)
    print("Check the 'plots/' directory for detailed visualizations.")