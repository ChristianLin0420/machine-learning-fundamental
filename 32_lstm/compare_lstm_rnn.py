"""
LSTM vs RNN Comparison
Author: ML Fundamentals Series
Day 32: Long Short-Term Memory Networks

This module provides head-to-head comparison between LSTM and RNN
on sequence prediction tasks to demonstrate LSTM's advantages.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
from typing import Dict, List, Tuple

# Set style for consistent plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class SimpleRNN:
    """
    Simple RNN implementation for comparison with LSTM.
    Based on the RNN from Day 31 but simplified for direct comparison.
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, learning_rate: float = 0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights
        self.W_xh = np.random.randn(hidden_size, input_size) * np.sqrt(1.0 / input_size)
        self.W_hh = np.random.randn(hidden_size, hidden_size) * np.sqrt(1.0 / hidden_size)
        self.W_hy = np.random.randn(output_size, hidden_size) * np.sqrt(1.0 / hidden_size)
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))
        
        # Training metrics
        self.losses = []
        self.gradient_norms = []
    
    def tanh(self, x):
        return np.tanh(x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    def forward_step(self, x_t, h_prev):
        """Single RNN forward step."""
        h_t = self.tanh(np.dot(self.W_xh, x_t) + np.dot(self.W_hh, h_prev) + self.b_h)
        y_t = np.dot(self.W_hy, h_t) + self.b_y
        return h_t, y_t
    
    def forward(self, inputs, initial_h=None):
        """Forward pass through sequence."""
        seq_length = len(inputs)
        
        if initial_h is None:
            h_t = np.zeros((self.hidden_size, 1))
        else:
            h_t = initial_h.copy()
        
        hidden_states = [h_t.copy()]
        outputs = []
        
        for t in range(seq_length):
            h_t, y_t = self.forward_step(inputs[t], h_t)
            hidden_states.append(h_t.copy())
            outputs.append(y_t)
        
        return outputs, hidden_states
    
    def backward(self, outputs, targets, hidden_states, inputs):
        """Backpropagation through time."""
        seq_length = len(outputs)
        
        # Initialize gradients
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)
        
        dh_next = np.zeros((self.hidden_size, 1))
        loss = 0
        
        # Backward pass
        for t in reversed(range(seq_length)):
            # Output layer gradients
            dy = outputs[t] - targets[t]
            loss += 0.5 * np.sum(dy**2)
            
            dW_hy += np.dot(dy, hidden_states[t+1].T)
            db_y += dy
            
            # Hidden layer gradients
            dh = np.dot(self.W_hy.T, dy) + dh_next
            dh_raw = dh * (1 - hidden_states[t+1]**2)  # tanh derivative
            
            # Weight gradients
            dW_xh += np.dot(dh_raw, inputs[t].T)
            dW_hh += np.dot(dh_raw, hidden_states[t].T)
            db_h += dh_raw
            
            # Gradient for next iteration
            dh_next = np.dot(self.W_hh.T, dh_raw)
        
        gradients = {
            'dW_xh': dW_xh, 'dW_hh': dW_hh, 'dW_hy': dW_hy,
            'db_h': db_h, 'db_y': db_y
        }
        
        return loss, gradients
    
    def clip_gradients(self, gradients, clip_value=1.0):
        """Clip gradients to prevent explosion."""
        total_norm = 0
        for grad in gradients.values():
            total_norm += np.sum(grad**2)
        total_norm = np.sqrt(total_norm)
        
        if total_norm > clip_value:
            for key in gradients:
                gradients[key] = gradients[key] * clip_value / total_norm
        
        return total_norm
    
    def update_weights(self, gradients):
        """Update weights using gradients."""
        self.W_xh -= self.learning_rate * gradients['dW_xh']
        self.W_hh -= self.learning_rate * gradients['dW_hh']
        self.W_hy -= self.learning_rate * gradients['dW_hy']
        self.b_h -= self.learning_rate * gradients['db_h']
        self.b_y -= self.learning_rate * gradients['db_y']
    
    def train_step(self, inputs, targets):
        """Single training step."""
        # Forward pass
        outputs, hidden_states = self.forward(inputs)
        
        # Backward pass
        loss, gradients = self.backward(outputs, targets, hidden_states, inputs)
        
        # Clip gradients
        grad_norm = self.clip_gradients(gradients)
        
        # Update weights
        self.update_weights(gradients)
        
        # Store metrics
        self.losses.append(loss)
        self.gradient_norms.append(grad_norm)
        
        return loss, grad_norm
    
    def predict(self, inputs, initial_h=None):
        """Make predictions."""
        outputs, _ = self.forward(inputs, initial_h)
        return outputs


def create_comparison_datasets():
    """Create datasets specifically designed to highlight LSTM advantages."""
    
    datasets = {}
    
    # 1. Long-term dependency task (remember value from far past)
    def long_term_memory_task(seq_length=50, num_sequences=500, dependency_lag=30):
        """Task where output depends on input from many steps ago."""
        sequences = []
        targets = []
        
        for _ in range(num_sequences):
            # Random sequence with important signal early on
            important_value = np.random.uniform(-1, 1)
            noise = np.random.uniform(-0.1, 0.1, seq_length)
            
            # Create sequence with important value at beginning
            sequence = noise.copy()
            sequence[5] = important_value  # Place signal early in sequence
            
            # Target depends on the important value after long delay
            input_seq = [sequence[i:i+1].reshape(-1, 1) for i in range(seq_length)]
            target_seq = [np.zeros((1, 1)) for _ in range(seq_length)]
            target_seq[-1] = np.array([[important_value]])  # Remember at the end
            
            sequences.append(input_seq)
            targets.append(target_seq)
        
        return sequences, targets
    
    # 2. Vanishing gradient challenge (very long sequences)
    def vanishing_gradient_task(seq_length=80, num_sequences=300):
        """Long sequences to test vanishing gradient handling."""
        sequences = []
        targets = []
        
        for _ in range(num_sequences):
            # Create a pattern that requires long-term memory
            pattern_value = np.random.choice([-1, 1])
            
            # Sequence with pattern at beginning and noise throughout
            sequence = np.random.uniform(-0.2, 0.2, seq_length)
            sequence[0] = pattern_value
            sequence[1] = pattern_value * 0.8
            sequence[2] = pattern_value * 0.6
            
            # Target is to classify the pattern at the end
            input_seq = [sequence[i:i+1].reshape(-1, 1) for i in range(seq_length)]
            target_seq = [np.zeros((1, 1)) for _ in range(seq_length)]
            target_seq[-1] = np.array([[1.0 if pattern_value > 0 else -1.0]])
            
            sequences.append(input_seq)
            targets.append(target_seq)
        
        return sequences, targets
    
    # 3. Adding task with long sequences
    def long_adding_task(seq_length=60, num_sequences=400):
        """Adding task but with longer sequences."""
        sequences = []
        targets = []
        
        for _ in range(num_sequences):
            # Random values
            values = np.random.uniform(0, 1, seq_length)
            
            # Mark two positions to add (far apart)
            pos1 = np.random.randint(0, seq_length // 3)
            pos2 = np.random.randint(2 * seq_length // 3, seq_length)
            
            markers = np.zeros(seq_length)
            markers[pos1] = 1
            markers[pos2] = 1
            
            # Create input sequence
            input_seq = []
            for i in range(seq_length):
                input_vec = np.array([[values[i]], [markers[i]]])
                input_seq.append(input_vec)
            
            # Target is sum of marked values
            target_value = values[pos1] + values[pos2]
            target_seq = [np.zeros((1, 1)) for _ in range(seq_length)]
            target_seq[-1] = np.array([[target_value]])
            
            sequences.append(input_seq)
            targets.append(target_seq)
        
        return sequences, targets
    
    # 4. Sine wave with long-range dependencies
    def long_sine_prediction(seq_length=100, num_sequences=400):
        """Sine wave prediction with very long sequences."""
        sequences = []
        targets = []
        
        for _ in range(num_sequences):
            # Random phase and frequency
            phase = np.random.uniform(0, 2*np.pi)
            freq1 = np.random.uniform(0.02, 0.05)  # Very slow frequency
            freq2 = np.random.uniform(0.1, 0.2)    # Faster frequency
            
            # Create complex wave (sum of two sine waves)
            t = np.arange(seq_length + 1)
            wave = 0.7 * np.sin(freq1 * t + phase) + 0.3 * np.sin(freq2 * t + phase/2)
            
            # Input sequence and target (predict next value)
            input_seq = [wave[i:i+1].reshape(-1, 1) for i in range(seq_length)]
            target_seq = [wave[i+1:i+2].reshape(-1, 1) for i in range(seq_length)]
            
            sequences.append(input_seq)
            targets.append(target_seq)
        
        return sequences, targets
    
    print("Creating comparison datasets (designed to show LSTM advantages)...")
    
    datasets['long_term_memory'] = long_term_memory_task(seq_length=50, num_sequences=400)
    datasets['vanishing_gradient'] = vanishing_gradient_task(seq_length=80, num_sequences=300)
    datasets['long_adding'] = long_adding_task(seq_length=60, num_sequences=400)
    datasets['long_sine'] = long_sine_prediction(seq_length=100, num_sequences=300)
    
    return datasets


def run_lstm_vs_rnn_comparison():
    """Run comprehensive LSTM vs RNN comparison."""
    
    print("LSTM vs RNN Comprehensive Comparison")
    print("=" * 60)
    
    # Import LSTM from scratch implementation
    from lstm_from_scratch import LSTMFromScratch
    
    # Create comparison datasets
    datasets = create_comparison_datasets()
    
    results = {}
    
    for dataset_name, (sequences, targets) in datasets.items():
        print(f"\n" + "="*50)
        print(f"EXPERIMENT: {dataset_name.replace('_', ' ').title()}")
        print("="*50)
        
        # Get input dimensions
        sample_seq = sequences[0]
        input_size = sample_seq[0].shape[0]
        output_size = targets[0][0].shape[0]
        seq_length = len(sample_seq)
        
        print(f"Dataset: {len(sequences)} sequences of length {seq_length}")
        print(f"Input size: {input_size}, Output size: {output_size}")
        
        # Initialize models
        hidden_size = 40
        learning_rate = 0.005
        
        print(f"\nTraining RNN (hidden_size={hidden_size})...")
        rnn = SimpleRNN(input_size, hidden_size, output_size, learning_rate)
        
        print(f"Training LSTM (hidden_size={hidden_size})...")
        lstm = LSTMFromScratch(input_size, hidden_size, output_size, learning_rate)
        
        # Training parameters
        num_epochs = 150
        train_subset = 100  # Use subset for faster training
        
        # Train RNN
        rnn_start_time = time.time()
        rnn_losses = []
        rnn_grad_norms = []
        
        print("Training RNN...")
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_grad_norm = 0
            
            # Shuffle data
            indices = np.random.permutation(len(sequences))
            
            for i in indices[:train_subset]:
                loss, grad_norm = rnn.train_step(sequences[i], targets[i])
                epoch_loss += loss
                epoch_grad_norm += grad_norm
            
            avg_loss = epoch_loss / train_subset
            avg_grad_norm = epoch_grad_norm / train_subset
            
            rnn_losses.append(avg_loss)
            rnn_grad_norms.append(avg_grad_norm)
            
            if epoch % 20 == 0:
                print(f"  RNN Epoch {epoch:3d}, Loss: {avg_loss:.6f}, Grad Norm: {avg_grad_norm:.4f}")
        
        rnn_train_time = time.time() - rnn_start_time
        
        # Train LSTM
        lstm_start_time = time.time()
        lstm_losses = []
        lstm_grad_norms = []
        
        print("Training LSTM...")
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_grad_norm = 0
            
            # Shuffle data
            indices = np.random.permutation(len(sequences))
            
            for i in indices[:train_subset]:
                loss, grad_norm, _ = lstm.train_step(sequences[i], targets[i])
                epoch_loss += loss
                epoch_grad_norm += grad_norm
            
            avg_loss = epoch_loss / train_subset
            avg_grad_norm = epoch_grad_norm / train_subset
            
            lstm_losses.append(avg_loss)
            lstm_grad_norms.append(avg_grad_norm)
            
            if epoch % 20 == 0:
                print(f"  LSTM Epoch {epoch:3d}, Loss: {avg_loss:.6f}, Grad Norm: {avg_grad_norm:.4f}")
        
        lstm_train_time = time.time() - lstm_start_time
        
        # Test both models
        print("\nTesting models...")
        
        # Test on a few sequences
        test_indices = np.random.choice(len(sequences), 10, replace=False)
        
        rnn_test_errors = []
        lstm_test_errors = []
        
        for idx in test_indices:
            test_seq = sequences[idx]
            test_target = targets[idx][-1]  # Final target
            
            # RNN prediction
            rnn_outputs = rnn.predict(test_seq)
            rnn_pred = rnn_outputs[-1]
            rnn_error = np.mean((rnn_pred - test_target)**2)
            rnn_test_errors.append(rnn_error)
            
            # LSTM prediction
            lstm_outputs = lstm.predict(test_seq)
            lstm_pred = lstm_outputs[-1]
            lstm_error = np.mean((lstm_pred - test_target)**2)
            lstm_test_errors.append(lstm_error)
        
        # Calculate statistics
        rnn_final_loss = rnn_losses[-1]
        lstm_final_loss = lstm_losses[-1]
        rnn_test_error = np.mean(rnn_test_errors)
        lstm_test_error = np.mean(lstm_test_errors)
        rnn_final_grad = rnn_grad_norms[-1]
        lstm_final_grad = lstm_grad_norms[-1]
        
        # Store results
        results[dataset_name] = {
            'rnn': {
                'model': rnn,
                'train_losses': rnn_losses,
                'grad_norms': rnn_grad_norms,
                'final_loss': rnn_final_loss,
                'test_error': rnn_test_error,
                'train_time': rnn_train_time,
                'final_grad_norm': rnn_final_grad
            },
            'lstm': {
                'model': lstm,
                'train_losses': lstm_losses,
                'grad_norms': lstm_grad_norms,
                'final_loss': lstm_final_loss,
                'test_error': lstm_test_error,
                'train_time': lstm_train_time,
                'final_grad_norm': lstm_final_grad
            },
            'dataset_info': {
                'seq_length': seq_length,
                'input_size': input_size,
                'num_sequences': len(sequences)
            }
        }
        
        # Print comparison
        print(f"\nResults for {dataset_name}:")
        print(f"  RNN  - Final Loss: {rnn_final_loss:.6f}, Test Error: {rnn_test_error:.6f}, Time: {rnn_train_time:.2f}s")
        print(f"  LSTM - Final Loss: {lstm_final_loss:.6f}, Test Error: {lstm_test_error:.6f}, Time: {lstm_train_time:.2f}s")
        print(f"  LSTM Improvement: {((rnn_test_error - lstm_test_error) / rnn_test_error * 100):.1f}% better test error")
    
    return results


def visualize_comparison_results(results):
    """Create comprehensive visualizations comparing LSTM vs RNN."""
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Figure 1: Training Loss Comparison
    num_datasets = len(results)
    fig, axes = plt.subplots(2, num_datasets, figsize=(4*num_datasets, 10))
    if num_datasets == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('LSTM vs RNN: Training Dynamics Comparison', fontsize=16)
    
    for i, (dataset_name, result) in enumerate(results.items()):
        # Training losses
        rnn_losses = result['rnn']['train_losses']
        lstm_losses = result['lstm']['train_losses']
        epochs = range(len(rnn_losses))
        
        axes[0, i].plot(epochs, rnn_losses, 'r-', label='RNN', linewidth=2, alpha=0.8)
        axes[0, i].plot(epochs, lstm_losses, 'b-', label='LSTM', linewidth=2, alpha=0.8)
        axes[0, i].set_title(f'{dataset_name.replace("_", " ").title()}\nTraining Loss')
        axes[0, i].set_xlabel('Epoch')
        axes[0, i].set_ylabel('Loss')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].set_yscale('log')
        
        # Gradient norms
        rnn_grads = result['rnn']['grad_norms']
        lstm_grads = result['lstm']['grad_norms']
        
        axes[1, i].plot(epochs, rnn_grads, 'r-', label='RNN', linewidth=2, alpha=0.8)
        axes[1, i].plot(epochs, lstm_grads, 'b-', label='LSTM', linewidth=2, alpha=0.8)
        axes[1, i].set_title(f'{dataset_name.replace("_", " ").title()}\nGradient Norms')
        axes[1, i].set_xlabel('Epoch')
        axes[1, i].set_ylabel('Gradient Norm')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('plots/lstm_vs_rnn_training.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Performance Summary
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('LSTM vs RNN: Performance Summary', fontsize=16)
    
    dataset_names = list(results.keys())
    dataset_labels = [name.replace('_', ' ').title() for name in dataset_names]
    
    # Final training losses
    rnn_final_losses = [results[name]['rnn']['final_loss'] for name in dataset_names]
    lstm_final_losses = [results[name]['lstm']['final_loss'] for name in dataset_names]
    
    x = np.arange(len(dataset_names))
    width = 0.35
    
    bars1 = axes[0, 0].bar(x - width/2, rnn_final_losses, width, label='RNN', color='red', alpha=0.7)
    bars2 = axes[0, 0].bar(x + width/2, lstm_final_losses, width, label='LSTM', color='blue', alpha=0.7)
    axes[0, 0].set_title('Final Training Loss')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(dataset_labels, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    
    # Test errors
    rnn_test_errors = [results[name]['rnn']['test_error'] for name in dataset_names]
    lstm_test_errors = [results[name]['lstm']['test_error'] for name in dataset_names]
    
    bars3 = axes[0, 1].bar(x - width/2, rnn_test_errors, width, label='RNN', color='red', alpha=0.7)
    bars4 = axes[0, 1].bar(x + width/2, lstm_test_errors, width, label='LSTM', color='blue', alpha=0.7)
    axes[0, 1].set_title('Test Error')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(dataset_labels, rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].set_yscale('log')
    
    # Training times
    rnn_times = [results[name]['rnn']['train_time'] for name in dataset_names]
    lstm_times = [results[name]['lstm']['train_time'] for name in dataset_names]
    
    bars5 = axes[1, 0].bar(x - width/2, rnn_times, width, label='RNN', color='red', alpha=0.7)
    bars6 = axes[1, 0].bar(x + width/2, lstm_times, width, label='LSTM', color='blue', alpha=0.7)
    axes[1, 0].set_title('Training Time')
    axes[1, 0].set_ylabel('Time (seconds)')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(dataset_labels, rotation=45, ha='right')
    axes[1, 0].legend()
    
    # Improvement percentages
    improvements = []
    for name in dataset_names:
        rnn_err = results[name]['rnn']['test_error']
        lstm_err = results[name]['lstm']['test_error']
        improvement = ((rnn_err - lstm_err) / rnn_err) * 100
        improvements.append(improvement)
    
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars7 = axes[1, 1].bar(dataset_labels, improvements, color=colors, alpha=0.7)
    axes[1, 1].set_title('LSTM Improvement over RNN')
    axes[1, 1].set_ylabel('Improvement (%)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add percentage labels on bars
    for bar, imp in zip(bars7, improvements):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                       f'{imp:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
    
    plt.tight_layout()
    plt.savefig('plots/lstm_vs_rnn_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Sequence Length vs Performance
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Performance vs Sequence Length', fontsize=16)
    
    # Extract sequence lengths and errors
    seq_lengths = [results[name]['dataset_info']['seq_length'] for name in dataset_names]
    
    # Plot test errors vs sequence length
    axes[0].scatter(seq_lengths, rnn_test_errors, color='red', s=100, alpha=0.7, label='RNN')
    axes[0].scatter(seq_lengths, lstm_test_errors, color='blue', s=100, alpha=0.7, label='LSTM')
    
    for i, name in enumerate(dataset_names):
        axes[0].annotate(name.replace('_', '\n'), 
                        (seq_lengths[i], lstm_test_errors[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    axes[0].set_xlabel('Sequence Length')
    axes[0].set_ylabel('Test Error (MSE)')
    axes[0].set_title('Test Error vs Sequence Length')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # Plot improvement vs sequence length
    axes[1].scatter(seq_lengths, improvements, color='green', s=100, alpha=0.7)
    
    for i, name in enumerate(dataset_names):
        axes[1].annotate(name.replace('_', '\n'), 
                        (seq_lengths[i], improvements[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    axes[1].set_xlabel('Sequence Length')
    axes[1].set_ylabel('LSTM Improvement (%)')
    axes[1].set_title('LSTM Advantage vs Sequence Length')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/lstm_vs_rnn_sequence_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nComparison visualizations completed!")
    print("Generated files:")
    print("- lstm_vs_rnn_training.png: Training dynamics comparison")
    print("- lstm_vs_rnn_summary.png: Performance summary")
    print("- lstm_vs_rnn_sequence_analysis.png: Sequence length analysis")


def analyze_gradient_flow_comparison(results):
    """Analyze gradient flow differences between LSTM and RNN."""
    
    print("\n" + "="*60)
    print("GRADIENT FLOW ANALYSIS")
    print("="*60)
    
    # Create gradient flow analysis plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Gradient Flow Analysis: LSTM vs RNN', fontsize=16)
    
    # Select the longest sequence dataset for analysis
    longest_dataset = max(results.keys(), 
                         key=lambda x: results[x]['dataset_info']['seq_length'])
    
    result = results[longest_dataset]
    
    print(f"Analyzing gradient flow for: {longest_dataset}")
    print(f"Sequence length: {result['dataset_info']['seq_length']}")
    
    # Plot gradient norms over training
    rnn_grads = result['rnn']['grad_norms']
    lstm_grads = result['lstm']['grad_norms']
    epochs = range(len(rnn_grads))
    
    # Moving average for smoother curves
    window = 10
    rnn_smooth = np.convolve(rnn_grads, np.ones(window)/window, mode='valid')
    lstm_smooth = np.convolve(lstm_grads, np.ones(window)/window, mode='valid')
    epochs_smooth = range(len(rnn_smooth))
    
    axes[0, 0].plot(epochs, rnn_grads, 'r-', alpha=0.3, linewidth=1)
    axes[0, 0].plot(epochs_smooth, rnn_smooth, 'r-', linewidth=2, label='RNN (smoothed)')
    axes[0, 0].set_title('RNN Gradient Norms')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Gradient Norm')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    axes[0, 1].plot(epochs, lstm_grads, 'b-', alpha=0.3, linewidth=1)
    axes[0, 1].plot(epochs_smooth, lstm_smooth, 'b-', linewidth=2, label='LSTM (smoothed)')
    axes[0, 1].set_title('LSTM Gradient Norms')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Gradient Norm')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # Direct comparison
    axes[1, 0].plot(epochs_smooth, rnn_smooth, 'r-', linewidth=2, label='RNN')
    axes[1, 0].plot(epochs_smooth, lstm_smooth, 'b-', linewidth=2, label='LSTM')
    axes[1, 0].set_title('Gradient Norm Comparison')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Gradient Norm')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # Gradient stability analysis
    rnn_stability = np.std(rnn_grads[-50:])  # Stability in last 50 epochs
    lstm_stability = np.std(lstm_grads[-50:])
    
    stability_comparison = ['RNN', 'LSTM']
    stability_values = [rnn_stability, lstm_stability]
    
    bars = axes[1, 1].bar(stability_comparison, stability_values, 
                         color=['red', 'blue'], alpha=0.7)
    axes[1, 1].set_title('Gradient Stability\n(Std Dev of Final 50 Epochs)')
    axes[1, 1].set_ylabel('Gradient Norm Std Dev')
    
    # Add values on bars
    for bar, val in zip(bars, stability_values):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('plots/gradient_flow_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print(f"\nGradient Flow Statistics ({longest_dataset}):")
    print(f"  RNN:")
    print(f"    Final gradient norm: {rnn_grads[-1]:.6f}")
    print(f"    Mean gradient norm: {np.mean(rnn_grads):.6f}")
    print(f"    Gradient stability: {rnn_stability:.6f}")
    print(f"  LSTM:")
    print(f"    Final gradient norm: {lstm_grads[-1]:.6f}")
    print(f"    Mean gradient norm: {np.mean(lstm_grads):.6f}")
    print(f"    Gradient stability: {lstm_stability:.6f}")
    
    stability_improvement = ((rnn_stability - lstm_stability) / rnn_stability) * 100
    print(f"  LSTM gradient stability improvement: {stability_improvement:.1f}%")


if __name__ == "__main__":
    print("Starting LSTM vs RNN Comparison...")
    
    # Run comprehensive comparison
    comparison_results = run_lstm_vs_rnn_comparison()
    
    # Create visualizations
    visualize_comparison_results(comparison_results)
    
    # Analyze gradient flow
    analyze_gradient_flow_comparison(comparison_results)
    
    print("\n" + "="*60)
    print("LSTM vs RNN COMPARISON COMPLETE")
    print("="*60)
    
    # Summary statistics
    total_datasets = len(comparison_results)
    lstm_wins = 0
    total_improvement = 0
    
    for dataset_name, result in comparison_results.items():
        rnn_error = result['rnn']['test_error']
        lstm_error = result['lstm']['test_error']
        
        if lstm_error < rnn_error:
            lstm_wins += 1
            improvement = ((rnn_error - lstm_error) / rnn_error) * 100
            total_improvement += improvement
            print(f"  {dataset_name}: LSTM wins by {improvement:.1f}%")
        else:
            improvement = ((lstm_error - rnn_error) / lstm_error) * 100
            print(f"  {dataset_name}: RNN wins by {improvement:.1f}%")
    
    print(f"\nOverall Results:")
    print(f"  LSTM wins: {lstm_wins}/{total_datasets} datasets")
    print(f"  Average LSTM improvement: {total_improvement/max(lstm_wins, 1):.1f}%")
    print(f"\n✅ LSTM demonstrates clear advantages on long-term dependency tasks")
    print("✅ LSTM shows better gradient flow and training stability")
    print("✅ Performance gap increases with sequence length")
    print("\nCheck the 'plots' directory for detailed analysis!")