"""
LSTM from Scratch Implementation
Author: ML Fundamentals Series
Day 32: Long Short-Term Memory Networks

This module implements LSTM from scratch using only NumPy to provide
deep understanding of the LSTM architecture and gate mechanisms.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import time
import os

# Set style for consistent plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class LSTMFromScratch:
    """
    Complete LSTM implementation from scratch using NumPy.
    
    Features:
    - All three gates (forget, input, output) 
    - Cell state and hidden state management
    - Forward pass with proper gate computations
    - Backpropagation through time (BPTT)
    - Gradient clipping for stability
    - Multiple sequence prediction tasks
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 learning_rate: float = 0.001):
        """
        Initialize LSTM with Xavier/Glorot initialization.
        
        Args:
            input_size: Dimension of input vectors
            hidden_size: Dimension of hidden/cell states
            output_size: Dimension of output vectors
            learning_rate: Learning rate for optimization
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Initialize weights using Xavier initialization
        self._initialize_weights()
        
        # Training metrics
        self.losses = []
        self.perplexities = []
        self.gradient_norms = []
        
    def _initialize_weights(self):
        """Initialize all LSTM weights and biases using Xavier initialization."""
        
        # Input to hidden weights for all gates + candidate
        # Each gate needs weights from input and previous hidden state
        input_hidden_size = self.input_size + self.hidden_size
        
        # Forget gate weights
        self.W_f = np.random.randn(self.hidden_size, input_hidden_size) * np.sqrt(2.0 / input_hidden_size)
        self.b_f = np.ones((self.hidden_size, 1))  # Initialize forget bias to 1 (remember by default)
        
        # Input gate weights
        self.W_i = np.random.randn(self.hidden_size, input_hidden_size) * np.sqrt(2.0 / input_hidden_size)
        self.b_i = np.zeros((self.hidden_size, 1))
        
        # Candidate values weights  
        self.W_C = np.random.randn(self.hidden_size, input_hidden_size) * np.sqrt(2.0 / input_hidden_size)
        self.b_C = np.zeros((self.hidden_size, 1))
        
        # Output gate weights
        self.W_o = np.random.randn(self.hidden_size, input_hidden_size) * np.sqrt(2.0 / input_hidden_size)
        self.b_o = np.zeros((self.hidden_size, 1))
        
        # Hidden to output weights
        self.W_hy = np.random.randn(self.output_size, self.hidden_size) * np.sqrt(2.0 / self.hidden_size)
        self.b_y = np.zeros((self.output_size, 1))
        
    def sigmoid(self, x):
        """Numerically stable sigmoid function."""
        return np.where(x >= 0, 
                       1 / (1 + np.exp(-x)), 
                       np.exp(x) / (1 + np.exp(x)))
    
    def tanh(self, x):
        """Tanh activation function."""
        return np.tanh(x)
    
    def softmax(self, x):
        """Numerically stable softmax function."""
        exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    def forward_step(self, x_t, h_prev, C_prev):
        """
        Single LSTM forward step.
        
        Args:
            x_t: Input at time t (input_size, 1)
            h_prev: Previous hidden state (hidden_size, 1)
            C_prev: Previous cell state (hidden_size, 1)
            
        Returns:
            h_t: Current hidden state
            C_t: Current cell state
            gate_values: Dictionary with all gate activations for analysis
        """
        
        # Concatenate input and previous hidden state
        concat_input = np.vstack([h_prev, x_t])  # (input_size + hidden_size, 1)
        
        # Compute gates
        f_t = self.sigmoid(np.dot(self.W_f, concat_input) + self.b_f)  # Forget gate
        i_t = self.sigmoid(np.dot(self.W_i, concat_input) + self.b_i)  # Input gate
        C_tilde = self.tanh(np.dot(self.W_C, concat_input) + self.b_C)  # Candidate values
        o_t = self.sigmoid(np.dot(self.W_o, concat_input) + self.b_o)  # Output gate
        
        # Update cell state: C_t = f_t * C_prev + i_t * C_tilde
        C_t = f_t * C_prev + i_t * C_tilde
        
        # Update hidden state: h_t = o_t * tanh(C_t)
        h_t = o_t * self.tanh(C_t)
        
        # Store gate values for analysis
        gate_values = {
            'forget': f_t,
            'input': i_t, 
            'candidate': C_tilde,
            'output': o_t,
            'cell_state': C_t,
            'concat_input': concat_input
        }
        
        return h_t, C_t, gate_values
    
    def forward(self, inputs, initial_h=None, initial_C=None):
        """
        Forward pass through sequence.
        
        Args:
            inputs: List of input vectors [(input_size, 1), ...]
            initial_h: Initial hidden state
            initial_C: Initial cell state
            
        Returns:
            outputs: List of output vectors
            hidden_states: List of hidden states  
            cell_states: List of cell states
            all_gates: List of gate values for each timestep
        """
        seq_length = len(inputs)
        
        # Initialize states
        if initial_h is None:
            h_t = np.zeros((self.hidden_size, 1))
        else:
            h_t = initial_h.copy()
            
        if initial_C is None:
            C_t = np.zeros((self.hidden_size, 1))
        else:
            C_t = initial_C.copy()
        
        # Store all states and outputs
        hidden_states = [h_t.copy()]
        cell_states = [C_t.copy()]
        all_gates = []
        outputs = []
        
        # Forward pass through sequence
        for t in range(seq_length):
            h_t, C_t, gate_values = self.forward_step(inputs[t], h_t, C_t)
            
            # Compute output
            y_t = np.dot(self.W_hy, h_t) + self.b_y
            
            # Store values
            hidden_states.append(h_t.copy())
            cell_states.append(C_t.copy())
            all_gates.append(gate_values)
            outputs.append(y_t)
            
        return outputs, hidden_states, cell_states, all_gates
    
    def backward_step(self, dh_next, dC_next, gate_values, h_prev, C_prev):
        """
        Single LSTM backward step.
        
        Args:
            dh_next: Gradient w.r.t. next hidden state
            dC_next: Gradient w.r.t. next cell state
            gate_values: Gate values from forward pass
            h_prev: Previous hidden state
            C_prev: Previous cell state
            
        Returns:
            Gradients for all parameters and previous states
        """
        
        # Extract gate values
        f_t = gate_values['forget']
        i_t = gate_values['input']
        C_tilde = gate_values['candidate']
        o_t = gate_values['output']
        C_t = gate_values['cell_state']
        concat_input = gate_values['concat_input']
        
        # Gradient w.r.t. output gate
        do_t = dh_next * self.tanh(C_t)
        do_t_input = do_t * o_t * (1 - o_t)  # Sigmoid derivative
        
        # Gradient w.r.t. cell state
        dC_t = dC_next + dh_next * o_t * (1 - self.tanh(C_t)**2)  # tanh derivative
        
        # Gradient w.r.t. forget gate
        df_t = dC_t * C_prev
        df_t_input = df_t * f_t * (1 - f_t)  # Sigmoid derivative
        
        # Gradient w.r.t. input gate
        di_t = dC_t * C_tilde
        di_t_input = di_t * i_t * (1 - i_t)  # Sigmoid derivative
        
        # Gradient w.r.t. candidate values
        dC_tilde = dC_t * i_t
        dC_tilde_input = dC_tilde * (1 - C_tilde**2)  # tanh derivative
        
        # Gradients w.r.t. weights and biases
        dW_f = np.dot(df_t_input, concat_input.T)
        db_f = df_t_input
        
        dW_i = np.dot(di_t_input, concat_input.T)
        db_i = di_t_input
        
        dW_C = np.dot(dC_tilde_input, concat_input.T)
        db_C = dC_tilde_input
        
        dW_o = np.dot(do_t_input, concat_input.T)
        db_o = do_t_input
        
        # Gradient w.r.t. concat input
        dconcat = (np.dot(self.W_f.T, df_t_input) + 
                  np.dot(self.W_i.T, di_t_input) +
                  np.dot(self.W_C.T, dC_tilde_input) + 
                  np.dot(self.W_o.T, do_t_input))
        
        # Split gradients for previous hidden state and input
        dh_prev = dconcat[:self.hidden_size]
        dx_t = dconcat[self.hidden_size:]
        
        # Gradient w.r.t. previous cell state
        dC_prev = dC_t * f_t
        
        gradients = {
            'dW_f': dW_f, 'db_f': db_f,
            'dW_i': dW_i, 'db_i': db_i, 
            'dW_C': dW_C, 'db_C': db_C,
            'dW_o': dW_o, 'db_o': db_o,
            'dh_prev': dh_prev, 'dC_prev': dC_prev, 'dx_t': dx_t
        }
        
        return gradients
    
    def backward(self, outputs, targets, hidden_states, cell_states, all_gates, inputs):
        """
        Backpropagation through time for LSTM.
        
        Args:
            outputs: Forward pass outputs
            targets: Target values
            hidden_states: All hidden states from forward pass
            cell_states: All cell states from forward pass
            all_gates: All gate values from forward pass
            inputs: Input sequence
            
        Returns:
            loss: Total loss for sequence
            gradients: All parameter gradients
        """
        seq_length = len(outputs)
        
        # Initialize gradients
        dW_f = np.zeros_like(self.W_f)
        db_f = np.zeros_like(self.b_f)
        dW_i = np.zeros_like(self.W_i)
        db_i = np.zeros_like(self.b_i)
        dW_C = np.zeros_like(self.W_C)
        db_C = np.zeros_like(self.b_C)
        dW_o = np.zeros_like(self.W_o)
        db_o = np.zeros_like(self.b_o)
        dW_hy = np.zeros_like(self.W_hy)
        db_y = np.zeros_like(self.b_y)
        
        # Initialize gradients for states
        dh_next = np.zeros((self.hidden_size, 1))
        dC_next = np.zeros((self.hidden_size, 1))
        
        loss = 0
        
        # Backward pass through sequence
        for t in reversed(range(seq_length)):
            # Output layer gradients
            dy = outputs[t] - targets[t]
            loss += 0.5 * np.sum(dy**2)
            
            dW_hy += np.dot(dy, hidden_states[t+1].T)
            db_y += dy
            
            # Gradient w.r.t. hidden state
            dh = np.dot(self.W_hy.T, dy) + dh_next
            
            # LSTM backward step
            if t > 0:
                gradients = self.backward_step(dh, dC_next, all_gates[t], 
                                             hidden_states[t], cell_states[t])
            else:
                # For t=0, use initial states (zeros)
                h_prev = np.zeros((self.hidden_size, 1))
                C_prev = np.zeros((self.hidden_size, 1))
                gradients = self.backward_step(dh, dC_next, all_gates[t], h_prev, C_prev)
            
            # Accumulate gradients
            dW_f += gradients['dW_f']
            db_f += gradients['db_f']
            dW_i += gradients['dW_i']
            db_i += gradients['db_i']
            dW_C += gradients['dW_C']
            db_C += gradients['db_C']
            dW_o += gradients['dW_o']
            db_o += gradients['db_o']
            
            # Update gradients for next iteration
            dh_next = gradients['dh_prev']
            dC_next = gradients['dC_prev']
        
        # Package all gradients
        all_gradients = {
            'dW_f': dW_f, 'db_f': db_f,
            'dW_i': dW_i, 'db_i': db_i,
            'dW_C': dW_C, 'db_C': db_C,
            'dW_o': dW_o, 'db_o': db_o,
            'dW_hy': dW_hy, 'db_y': db_y
        }
        
        return loss, all_gradients
    
    def clip_gradients(self, gradients, clip_value=1.0):
        """Clip gradients to prevent exploding gradients."""
        
        # Calculate total gradient norm
        total_norm = 0
        for grad in gradients.values():
            total_norm += np.sum(grad**2)
        total_norm = np.sqrt(total_norm)
        
        # Clip if necessary
        if total_norm > clip_value:
            for key in gradients:
                gradients[key] = gradients[key] * clip_value / total_norm
        
        return total_norm
    
    def update_weights(self, gradients):
        """Update all LSTM weights using gradients."""
        
        self.W_f -= self.learning_rate * gradients['dW_f']
        self.b_f -= self.learning_rate * gradients['db_f']
        self.W_i -= self.learning_rate * gradients['dW_i']
        self.b_i -= self.learning_rate * gradients['db_i']
        self.W_C -= self.learning_rate * gradients['dW_C']
        self.b_C -= self.learning_rate * gradients['db_C']
        self.W_o -= self.learning_rate * gradients['dW_o']
        self.b_o -= self.learning_rate * gradients['db_o']
        self.W_hy -= self.learning_rate * gradients['dW_hy']
        self.b_y -= self.learning_rate * gradients['db_y']
    
    def train_step(self, inputs, targets):
        """Single training step."""
        
        # Forward pass
        outputs, hidden_states, cell_states, all_gates = self.forward(inputs)
        
        # Backward pass
        loss, gradients = self.backward(outputs, targets, hidden_states, 
                                      cell_states, all_gates, inputs)
        
        # Clip gradients
        grad_norm = self.clip_gradients(gradients)
        
        # Update weights
        self.update_weights(gradients)
        
        # Store metrics
        self.losses.append(loss)
        self.gradient_norms.append(grad_norm)
        
        return loss, grad_norm, all_gates
    
    def predict(self, inputs, initial_h=None, initial_C=None):
        """Make predictions without training."""
        outputs, _, _, _ = self.forward(inputs, initial_h, initial_C)
        return outputs
    
    def generate_sequence(self, seed_input, length, temperature=1.0):
        """
        Generate a sequence starting from seed input.
        
        Args:
            seed_input: Initial input to start generation
            length: Length of sequence to generate
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            generated_sequence: List of generated outputs
        """
        generated = []
        current_input = seed_input.copy()
        h_t = np.zeros((self.hidden_size, 1))
        C_t = np.zeros((self.hidden_size, 1))
        
        for _ in range(length):
            # Forward step
            h_t, C_t, _ = self.forward_step(current_input, h_t, C_t)
            
            # Generate output
            y_t = np.dot(self.W_hy, h_t) + self.b_y
            
            # Apply temperature and sample
            if temperature != 1.0:
                y_t = y_t / temperature
            
            # For regression tasks, just use the output
            generated.append(y_t.copy())
            
            # Use output as next input (for sequence generation)
            current_input = y_t
        
        return generated


def create_sequence_datasets():
    """Create various sequence prediction datasets for LSTM experiments."""
    
    datasets = {}
    
    # 1. Sine wave prediction
    def sine_wave_data(seq_length=50, num_sequences=1000):
        sequences = []
        targets = []
        
        for _ in range(num_sequences):
            # Random phase and frequency
            phase = np.random.uniform(0, 2*np.pi)
            freq = np.random.uniform(0.1, 0.5)
            
            # Generate sequence
            t = np.arange(seq_length + 1)
            wave = np.sin(freq * t + phase)
            
            # Input sequence and target (next value)
            input_seq = [wave[i:i+1].reshape(-1, 1) for i in range(seq_length)]
            target_seq = [wave[i+1:i+2].reshape(-1, 1) for i in range(seq_length)]
            
            sequences.append(input_seq)
            targets.append(target_seq)
        
        return sequences, targets
    
    # 2. Memory task (remember value from beginning)
    def memory_task_data(seq_length=20, num_sequences=1000):
        sequences = []
        targets = []
        
        for _ in range(num_sequences):
            # Random sequence with important value at beginning
            important_value = np.random.uniform(-1, 1)
            noise_sequence = np.random.uniform(-0.1, 0.1, seq_length-1)
            
            # Create input sequence
            full_sequence = np.concatenate([[important_value], noise_sequence])
            input_seq = [full_sequence[i:i+1].reshape(-1, 1) for i in range(seq_length)]
            
            # Target is to output the important value at the end
            target_seq = [np.zeros((1, 1)) for _ in range(seq_length-1)]
            target_seq.append(np.array([[important_value]]))
            
            sequences.append(input_seq)
            targets.append(target_seq)
        
        return sequences, targets
    
    # 3. Adding problem (sum two marked numbers)
    def adding_problem_data(seq_length=20, num_sequences=1000):
        sequences = []
        targets = []
        
        for _ in range(num_sequences):
            # Random sequence
            values = np.random.uniform(0, 1, seq_length)
            
            # Mark two positions to add
            markers = np.zeros(seq_length)
            positions = np.random.choice(seq_length, 2, replace=False)
            markers[positions] = 1
            
            # Create input (value and marker channels)
            input_seq = []
            for i in range(seq_length):
                input_vec = np.array([[values[i]], [markers[i]]])
                input_seq.append(input_vec)
            
            # Target is sum of marked values
            target_value = values[positions[0]] + values[positions[1]]
            target_seq = [np.zeros((1, 1)) for _ in range(seq_length-1)]
            target_seq.append(np.array([[target_value]]))
            
            sequences.append(input_seq)
            targets.append(target_seq)
        
        return sequences, targets
    
    print("Creating LSTM sequence datasets...")
    
    datasets['sine_wave'] = sine_wave_data(seq_length=30, num_sequences=500)
    datasets['memory_task'] = memory_task_data(seq_length=25, num_sequences=500)
    datasets['adding_problem'] = adding_problem_data(seq_length=20, num_sequences=500)
    
    return datasets


def run_lstm_experiments():
    """Run comprehensive LSTM experiments and comparisons."""
    
    print("LSTM from Scratch - Comprehensive Experiments")
    print("=" * 60)
    
    # Create datasets
    datasets = create_sequence_datasets()
    
    results = {}
    
    # Experiment 1: Sine Wave Prediction
    print("\n" + "="*50)
    print("EXPERIMENT 1: Sine Wave Prediction")
    print("="*50)
    
    sequences, targets = datasets['sine_wave']
    
    # Initialize LSTM
    lstm = LSTMFromScratch(input_size=1, hidden_size=50, output_size=1, learning_rate=0.01)
    
    # Training
    num_epochs = 100
    train_losses = []
    
    print(f"Training LSTM for {num_epochs} epochs...")
    print(f"Dataset: {len(sequences)} sequences of length {len(sequences[0])}")
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        
        # Shuffle data
        indices = np.random.permutation(len(sequences))
        
        for i in indices[:100]:  # Use subset for faster training
            loss, grad_norm, gates = lstm.train_step(sequences[i], targets[i])
            epoch_loss += loss
        
        avg_loss = epoch_loss / 100
        train_losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}, Loss: {avg_loss:.6f}, Grad Norm: {grad_norm:.4f}")
    
    training_time = time.time() - start_time
    
    # Test prediction
    test_seq = sequences[0]
    predictions = lstm.predict(test_seq)
    actual = targets[0]
    
    # Calculate test error
    test_error = np.mean([(pred - target)**2 for pred, target in zip(predictions, actual)])
    
    results['sine_wave'] = {
        'train_losses': train_losses,
        'test_error': test_error,
        'training_time': training_time,
        'predictions': predictions[:10],  # Store first 10 predictions
        'actual': actual[:10],
        'model': lstm
    }
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Final train loss: {train_losses[-1]:.6f}")
    print(f"Test MSE: {test_error:.6f}")
    
    # Experiment 2: Memory Task
    print("\n" + "="*50)
    print("EXPERIMENT 2: Memory Task")
    print("="*50)
    
    sequences, targets = datasets['memory_task']
    
    # Initialize LSTM for memory task
    lstm_memory = LSTMFromScratch(input_size=1, hidden_size=30, output_size=1, learning_rate=0.005)
    
    # Training
    memory_losses = []
    
    print(f"Training LSTM for memory task...")
    
    for epoch in range(150):
        epoch_loss = 0
        
        for i in range(50):  # Use subset
            loss, grad_norm, gates = lstm_memory.train_step(sequences[i], targets[i])
            epoch_loss += loss
        
        avg_loss = epoch_loss / 50
        memory_losses.append(avg_loss)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}, Loss: {avg_loss:.6f}")
    
    # Test memory capability
    test_seq = sequences[0]
    memory_pred = lstm_memory.predict(test_seq)
    memory_target = targets[0][-1]  # Last target (the important value)
    memory_error = abs(memory_pred[-1][0, 0] - memory_target[0, 0])
    
    results['memory_task'] = {
        'train_losses': memory_losses,
        'memory_error': memory_error,
        'predicted_value': memory_pred[-1][0, 0],
        'target_value': memory_target[0, 0],
        'model': lstm_memory
    }
    
    print(f"Memory test - Predicted: {memory_pred[-1][0, 0]:.4f}, Target: {memory_target[0, 0]:.4f}")
    print(f"Memory error: {memory_error:.6f}")
    
    # Experiment 3: Adding Problem
    print("\n" + "="*50)
    print("EXPERIMENT 3: Adding Problem")
    print("="*50)
    
    sequences, targets = datasets['adding_problem']
    
    # Initialize LSTM for adding problem
    lstm_adding = LSTMFromScratch(input_size=2, hidden_size=40, output_size=1, learning_rate=0.003)
    
    # Training
    adding_losses = []
    
    for epoch in range(200):
        epoch_loss = 0
        
        for i in range(50):
            loss, grad_norm, gates = lstm_adding.train_step(sequences[i], targets[i])
            epoch_loss += loss
        
        avg_loss = epoch_loss / 50
        adding_losses.append(avg_loss)
        
        if epoch % 30 == 0:
            print(f"Epoch {epoch:3d}, Loss: {avg_loss:.6f}")
    
    # Test adding capability
    test_seq = sequences[0]
    adding_pred = lstm_adding.predict(test_seq)
    adding_target = targets[0][-1]
    adding_error = abs(adding_pred[-1][0, 0] - adding_target[0, 0])
    
    results['adding_problem'] = {
        'train_losses': adding_losses,
        'adding_error': adding_error,
        'predicted_sum': adding_pred[-1][0, 0],
        'target_sum': adding_target[0, 0],
        'model': lstm_adding
    }
    
    print(f"Adding test - Predicted: {adding_pred[-1][0, 0]:.4f}, Target: {adding_target[0, 0]:.4f}")
    print(f"Adding error: {adding_error:.6f}")
    
    return results


def visualize_lstm_results(results):
    """Create comprehensive visualizations of LSTM experiments."""
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Figure 1: Training Dynamics Comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('LSTM Training Dynamics Across Different Tasks', fontsize=16)
    
    # Loss curves
    tasks = ['sine_wave', 'memory_task', 'adding_problem']
    task_names = ['Sine Wave Prediction', 'Memory Task', 'Adding Problem']
    
    for i, (task, name) in enumerate(zip(tasks, task_names)):
        # Training loss
        axes[0, i].plot(results[task]['train_losses'], 'b-', alpha=0.8)
        axes[0, i].set_title(f'{name}\nTraining Loss')
        axes[0, i].set_xlabel('Epoch')
        axes[0, i].set_ylabel('Loss')
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].set_yscale('log')
        
        # Gradient norms (if available)
        model = results[task]['model']
        if hasattr(model, 'gradient_norms') and model.gradient_norms:
            axes[1, i].plot(model.gradient_norms[-100:], 'r-', alpha=0.8)
            axes[1, i].set_title(f'{name}\nGradient Norms (Last 100 Steps)')
            axes[1, i].set_xlabel('Training Step')
            axes[1, i].set_ylabel('Gradient Norm')
            axes[1, i].grid(True, alpha=0.3)
        else:
            axes[1, i].text(0.5, 0.5, 'Gradient norms\nnot available', 
                           ha='center', va='center', transform=axes[1, i].transAxes)
            axes[1, i].set_title(f'{name}\nGradient Analysis')
    
    plt.tight_layout()
    plt.savefig('plots/lstm_training_dynamics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Performance Summary
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('LSTM Performance Summary', fontsize=16)
    
    # Performance metrics
    task_errors = [
        results['sine_wave']['test_error'],
        results['memory_task']['memory_error'], 
        results['adding_problem']['adding_error']
    ]
    
    colors = ['steelblue', 'darkgreen', 'darkorange']
    
    bars = axes[0].bar(task_names, task_errors, color=colors, alpha=0.7)
    axes[0].set_title('Final Test Errors')
    axes[0].set_ylabel('Error')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add error values on bars
    for bar, error in zip(bars, task_errors):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{error:.4f}', ha='center', va='bottom')
    
    # Training times (if available)
    training_times = []
    for task in tasks:
        if 'training_time' in results[task]:
            training_times.append(results[task]['training_time'])
        else:
            training_times.append(0)
    
    if any(training_times):
        bars2 = axes[1].bar(task_names, training_times, color=colors, alpha=0.7)
        axes[1].set_title('Training Time (seconds)')
        axes[1].set_ylabel('Time (s)')
        axes[1].tick_params(axis='x', rotation=45)
        
        for bar, time_val in zip(bars2, training_times):
            if time_val > 0:
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height,
                            f'{time_val:.1f}s', ha='center', va='bottom')
    
    # Final loss comparison
    final_losses = [results[task]['train_losses'][-1] for task in tasks]
    bars3 = axes[2].bar(task_names, final_losses, color=colors, alpha=0.7)
    axes[2].set_title('Final Training Loss')
    axes[2].set_ylabel('Loss')
    axes[2].set_yscale('log')
    axes[2].tick_params(axis='x', rotation=45)
    
    for bar, loss in zip(bars3, final_losses):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{loss:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('plots/lstm_performance_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Prediction Examples
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('LSTM Prediction Examples', fontsize=16)
    
    # Sine wave predictions
    if 'predictions' in results['sine_wave']:
        predictions = results['sine_wave']['predictions']
        actual = results['sine_wave']['actual']
        
        pred_values = [p[0, 0] for p in predictions]
        actual_values = [a[0, 0] for a in actual]
        
        time_steps = range(len(pred_values))
        axes[0].plot(time_steps, actual_values, 'b-', label='Actual', linewidth=2)
        axes[0].plot(time_steps, pred_values, 'r--', label='Predicted', linewidth=2)
        axes[0].set_title('Sine Wave Prediction')
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Value')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Memory task visualization
    memory_pred = results['memory_task']['predicted_value']
    memory_target = results['memory_task']['target_value']
    
    axes[1].bar(['Predicted', 'Target'], [memory_pred, memory_target], 
               color=['red', 'blue'], alpha=0.7)
    axes[1].set_title('Memory Task: Remember First Value')
    axes[1].set_ylabel('Value')
    
    # Add values on bars
    axes[1].text(0, memory_pred, f'{memory_pred:.3f}', ha='center', va='bottom')
    axes[1].text(1, memory_target, f'{memory_target:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('plots/lstm_prediction_examples.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nVisualization completed! Check the 'plots' directory for:")
    print("- lstm_training_dynamics.png: Training loss and gradient analysis")
    print("- lstm_performance_summary.png: Performance metrics comparison")
    print("- lstm_prediction_examples.png: Example predictions")


def analyze_gate_activations(model, test_sequence):
    """Analyze and visualize LSTM gate activations."""
    
    print("\nAnalyzing LSTM gate activations...")
    
    # Forward pass to get gate values
    outputs, hidden_states, cell_states, all_gates = model.forward(test_sequence)
    
    # Extract gate values over time
    seq_length = len(all_gates)
    hidden_size = model.hidden_size
    
    forget_gates = np.array([gates['forget'].flatten() for gates in all_gates])
    input_gates = np.array([gates['input'].flatten() for gates in all_gates])
    output_gates = np.array([gates['output'].flatten() for gates in all_gates])
    cell_states_vals = np.array([gates['cell_state'].flatten() for gates in all_gates])
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('LSTM Gate Activations Over Time', fontsize=16)
    
    # Time steps
    time_steps = range(seq_length)
    
    # Plot average gate activations
    axes[0, 0].plot(time_steps, np.mean(forget_gates, axis=1), 'b-', label='Forget Gate', linewidth=2)
    axes[0, 0].fill_between(time_steps, 
                           np.mean(forget_gates, axis=1) - np.std(forget_gates, axis=1),
                           np.mean(forget_gates, axis=1) + np.std(forget_gates, axis=1),
                           alpha=0.3)
    axes[0, 0].set_title('Forget Gate Activation')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Activation')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    
    axes[0, 1].plot(time_steps, np.mean(input_gates, axis=1), 'g-', label='Input Gate', linewidth=2)
    axes[0, 1].fill_between(time_steps,
                           np.mean(input_gates, axis=1) - np.std(input_gates, axis=1),
                           np.mean(input_gates, axis=1) + np.std(input_gates, axis=1),
                           alpha=0.3)
    axes[0, 1].set_title('Input Gate Activation')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Activation')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 1)
    
    axes[1, 0].plot(time_steps, np.mean(output_gates, axis=1), 'r-', label='Output Gate', linewidth=2)
    axes[1, 0].fill_between(time_steps,
                           np.mean(output_gates, axis=1) - np.std(output_gates, axis=1),
                           np.mean(output_gates, axis=1) + np.std(output_gates, axis=1),
                           alpha=0.3)
    axes[1, 0].set_title('Output Gate Activation')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Activation')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)
    
    # Cell state evolution
    axes[1, 1].plot(time_steps, np.mean(cell_states_vals, axis=1), 'purple', linewidth=2)
    axes[1, 1].fill_between(time_steps,
                           np.mean(cell_states_vals, axis=1) - np.std(cell_states_vals, axis=1),
                           np.mean(cell_states_vals, axis=1) + np.std(cell_states_vals, axis=1),
                           alpha=0.3)
    axes[1, 1].set_title('Cell State Evolution')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Cell State Value')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/lstm_gate_activations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Gate statistics
    print(f"Gate Activation Statistics:")
    print(f"  Forget Gate - Mean: {np.mean(forget_gates):.3f}, Std: {np.std(forget_gates):.3f}")
    print(f"  Input Gate  - Mean: {np.mean(input_gates):.3f}, Std: {np.std(input_gates):.3f}")
    print(f"  Output Gate - Mean: {np.mean(output_gates):.3f}, Std: {np.std(output_gates):.3f}")
    print(f"  Cell State  - Mean: {np.mean(cell_states_vals):.3f}, Std: {np.std(cell_states_vals):.3f}")
    
    return {
        'forget_gates': forget_gates,
        'input_gates': input_gates,
        'output_gates': output_gates,
        'cell_states': cell_states_vals
    }


if __name__ == "__main__":
    print("Starting LSTM from Scratch Implementation...")
    
    # Run main experiments
    results = run_lstm_experiments()
    
    # Create visualizations
    visualize_lstm_results(results)
    
    # Analyze gate activations for sine wave model
    print("\n" + "="*60)
    print("GATE ACTIVATION ANALYSIS")
    print("="*60)
    
    # Use a test sequence from sine wave dataset
    datasets = create_sequence_datasets()
    test_seq = datasets['sine_wave'][0][0]  # First sequence
    gate_analysis = analyze_gate_activations(results['sine_wave']['model'], test_seq)
    
    print("\n" + "="*60)
    print("LSTM IMPLEMENTATION COMPLETE")
    print("="*60)
    print("✅ Successfully implemented LSTM from scratch")
    print("✅ Trained on multiple sequence prediction tasks")
    print("✅ Analyzed gate mechanisms and cell state evolution")
    print("✅ Generated comprehensive visualizations")
    print("\nCheck the 'plots' directory for detailed analysis!")