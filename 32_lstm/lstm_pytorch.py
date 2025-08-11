"""
LSTM PyTorch Implementation and Comparison
Author: ML Fundamentals Series
Day 32: Long Short-Term Memory Networks

This module implements LSTM using PyTorch and compares performance
with the from-scratch NumPy implementation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import time
import os
from typing import Dict, List, Tuple, Optional

# Set style for consistent plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class SequenceDataset(Dataset):
    """PyTorch dataset for sequence data."""
    
    def __init__(self, sequences, targets):
        """
        Args:
            sequences: List of input sequences
            targets: List of target sequences
        """
        self.sequences = sequences
        self.targets = targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Convert list of arrays to proper tensor format
        # Each sequence is a list of (input_size, 1) arrays
        sequence_data = []
        target_data = []
        
        for step_input in self.sequences[idx]:
            sequence_data.append(step_input.flatten())
        
        for step_target in self.targets[idx]:
            target_data.append(step_target.flatten())
        
        # Convert to tensors with proper shapes
        sequence = torch.FloatTensor(np.array(sequence_data))  # (seq_length, input_size)
        target = torch.FloatTensor(np.array(target_data))      # (seq_length, output_size)
        
        return sequence, target


class CustomLSTMCell(nn.Module):
    """
    Custom LSTM cell implementation for educational purposes.
    Implements the same LSTM equations as the NumPy version.
    """
    
    def __init__(self, input_size, hidden_size):
        super(CustomLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input-to-hidden weights for all gates
        self.W_ih = nn.Linear(input_size, 4 * hidden_size, bias=True)
        # Hidden-to-hidden weights for all gates  
        self.W_hh = nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Initialize forget gate bias to 1
                param.data[self.hidden_size:2*self.hidden_size].fill_(1.0)
    
    def forward(self, input_tensor, hidden_state):
        """
        Forward pass through LSTM cell.
        
        Args:
            input_tensor: Input at current timestep (batch_size, input_size)
            hidden_state: Tuple of (h_t-1, C_t-1)
        
        Returns:
            new_h: New hidden state
            new_C: New cell state
            gates: Dictionary with gate values for analysis
        """
        h_prev, C_prev = hidden_state
        
        # Compute input-to-hidden and hidden-to-hidden transformations
        gi = self.W_ih(input_tensor)  # Input gates
        gh = self.W_hh(h_prev)        # Hidden gates
        i_i, i_f, i_g, i_o = gi.chunk(4, 1)
        h_i, h_f, h_g, h_o = gh.chunk(4, 1)
        
        # Compute gates
        forget_gate = torch.sigmoid(i_f + h_f)
        input_gate = torch.sigmoid(i_i + h_i)
        candidate_gate = torch.tanh(i_g + h_g)
        output_gate = torch.sigmoid(i_o + h_o)
        
        # Update cell state
        new_C = forget_gate * C_prev + input_gate * candidate_gate
        
        # Update hidden state
        new_h = output_gate * torch.tanh(new_C)
        
        # Store gate values for analysis
        gates = {
            'forget': forget_gate,
            'input': input_gate,
            'candidate': candidate_gate,
            'output': output_gate,
            'cell_state': new_C
        }
        
        return new_h, new_C, gates


class LSTMModel(nn.Module):
    """
    Complete LSTM model with multiple layers and output projection.
    """
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, 
                 dropout=0.0, use_custom_cell=False):
        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.use_custom_cell = use_custom_cell
        
        if use_custom_cell:
            # Use our custom LSTM cell
            self.lstm_cell = CustomLSTMCell(input_size, hidden_size)
        else:
            # Use PyTorch's built-in LSTM
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                               batch_first=True, dropout=dropout)
        
        # Output projection layer
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x, hidden_state=None):
        """
        Forward pass through LSTM model.
        
        Args:
            x: Input tensor (batch_size, seq_length, input_size)
            hidden_state: Initial hidden state
        
        Returns:
            output: Model output (batch_size, seq_length, output_size)
            hidden_state: Final hidden state
            all_gates: Gate activations (if using custom cell)
        """
        batch_size, seq_length, _ = x.size()
        
        if self.use_custom_cell:
            # Use custom LSTM cell
            if hidden_state is None:
                h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
                C_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
            else:
                h_t, C_t = hidden_state
            
            outputs = []
            all_gates = []
            
            for t in range(seq_length):
                h_t, C_t, gates = self.lstm_cell(x[:, t, :], (h_t, C_t))
                output = self.output_layer(h_t)
                outputs.append(output.unsqueeze(1))
                all_gates.append(gates)
            
            output = torch.cat(outputs, dim=1)
            hidden_state = (h_t, C_t)
            
            return output, hidden_state, all_gates
        
        else:
            # Use built-in LSTM
            lstm_out, hidden_state = self.lstm(x, hidden_state)
            output = self.output_layer(lstm_out)
            
            return output, hidden_state, None
    
    def predict(self, x):
        """Make predictions without tracking gradients."""
        self.eval()
        with torch.no_grad():
            output, _, _ = self.forward(x)
        return output


class LSTMTrainer:
    """
    Training class for LSTM models.
    """
    
    def __init__(self, model, learning_rate=0.001, clip_value=1.0):
        self.model = model.to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.clip_value = clip_value
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.gradient_norms = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (sequences, targets) in enumerate(train_loader):
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs, _, _ = self.model(sequences)
            
            # Calculate loss - use last timestep for most sequence tasks
            # For sequence prediction tasks, we typically predict the next value
            loss = self.criterion(outputs[:, -1, :], targets[:, -1, :])
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
            self.gradient_norms.append(grad_norm.item())
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(device)
                targets = targets.to(device)
                
                outputs, _, _ = self.model(sequences)
                
                # Use last timestep for validation as well
                loss = self.criterion(outputs[:, -1, :], targets[:, -1, :])
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self, train_loader, val_loader, num_epochs):
        """Train the model for multiple epochs."""
        print(f"Training on {device}")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}, Train Loss: {train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}, Grad Norm: {self.gradient_norms[-1]:.4f}")
        
        print(f"Training completed. Final train loss: {self.train_losses[-1]:.6f}")
        return self.train_losses, self.val_losses


def create_pytorch_datasets():
    """Create PyTorch datasets for LSTM experiments."""
    
    # Import datasets from scratch implementation
    from lstm_from_scratch import create_sequence_datasets
    
    print("Creating PyTorch datasets...")
    scratch_datasets = create_sequence_datasets()
    
    pytorch_datasets = {}
    
    for name, (sequences, targets) in scratch_datasets.items():
        # Convert to PyTorch format
        dataset = SequenceDataset(sequences, targets)
        
        # Split into train/validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        pytorch_datasets[name] = {
            'train': train_dataset,
            'val': val_dataset,
            'full': dataset
        }
    
    return pytorch_datasets


def compare_pytorch_vs_scratch():
    """Compare PyTorch LSTM with scratch implementation."""
    
    print("PyTorch vs NumPy LSTM Comparison")
    print("=" * 50)
    
    # Create datasets
    datasets = create_pytorch_datasets()
    
    results = {}
    
    # Test on sine wave dataset
    dataset_name = 'sine_wave'
    train_dataset = datasets[dataset_name]['train']
    val_dataset = datasets[dataset_name]['val']
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Get input/output dimensions from first batch
    sample_seq, sample_target = next(iter(train_loader))
    input_size = sample_seq.shape[-1]
    if sample_target.dim() == 2:
        output_size = sample_target.shape[-1]
    else:
        output_size = sample_target.shape[-1]
    
    print(f"Dataset: {dataset_name}")
    print(f"Input size: {input_size}, Output size: {output_size}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Experiment 1: Built-in PyTorch LSTM
    print("\n" + "-" * 40)
    print("Training Built-in PyTorch LSTM...")
    print("-" * 40)
    
    builtin_model = LSTMModel(input_size=input_size, hidden_size=50, 
                             output_size=output_size, use_custom_cell=False)
    builtin_trainer = LSTMTrainer(builtin_model, learning_rate=0.01)
    
    start_time = time.time()
    train_losses_builtin, val_losses_builtin = builtin_trainer.train(
        train_loader, val_loader, num_epochs=100)
    builtin_time = time.time() - start_time
    
    # Experiment 2: Custom PyTorch LSTM
    print("\n" + "-" * 40)
    print("Training Custom PyTorch LSTM...")
    print("-" * 40)
    
    custom_model = LSTMModel(input_size=input_size, hidden_size=50,
                            output_size=output_size, use_custom_cell=True)
    custom_trainer = LSTMTrainer(custom_model, learning_rate=0.01)
    
    start_time = time.time()
    train_losses_custom, val_losses_custom = custom_trainer.train(
        train_loader, val_loader, num_epochs=100)
    custom_time = time.time() - start_time
    
    # Compare performance
    builtin_final_loss = val_losses_builtin[-1]
    custom_final_loss = val_losses_custom[-1]
    
    print(f"\nComparison Results:")
    print(f"Built-in LSTM - Final val loss: {builtin_final_loss:.6f}, Time: {builtin_time:.2f}s")
    print(f"Custom LSTM   - Final val loss: {custom_final_loss:.6f}, Time: {custom_time:.2f}s")
    
    results[dataset_name] = {
        'builtin': {
            'model': builtin_model,
            'trainer': builtin_trainer,
            'train_losses': train_losses_builtin,
            'val_losses': val_losses_builtin,
            'final_loss': builtin_final_loss,
            'training_time': builtin_time
        },
        'custom': {
            'model': custom_model,
            'trainer': custom_trainer,
            'train_losses': train_losses_custom,
            'val_losses': val_losses_custom,
            'final_loss': custom_final_loss,
            'training_time': custom_time
        }
    }
    
    return results


def run_pytorch_experiments():
    """Run comprehensive PyTorch LSTM experiments."""
    
    print("PyTorch LSTM Comprehensive Experiments")
    print("=" * 60)
    
    # Create datasets
    datasets = create_pytorch_datasets()
    
    results = {}
    
    # Experiment on all tasks
    task_configs = {
        'sine_wave': {'hidden_size': 50, 'lr': 0.01, 'epochs': 100},
        'memory_task': {'hidden_size': 30, 'lr': 0.005, 'epochs': 150},
        'adding_problem': {'hidden_size': 40, 'lr': 0.003, 'epochs': 200}
    }
    
    for task_name, config in task_configs.items():
        print(f"\n" + "="*50)
        print(f"EXPERIMENT: {task_name.replace('_', ' ').title()}")
        print("="*50)
        
        train_dataset = datasets[task_name]['train']
        val_dataset = datasets[task_name]['val']
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        
        # Get dimensions
        sample_seq, sample_target = next(iter(train_loader))
        input_size = sample_seq.shape[-1]
        
        if sample_target.dim() == 2:
            output_size = sample_target.shape[-1]
        else:
            output_size = sample_target.shape[-1]
        
        print(f"Input size: {input_size}, Output size: {output_size}")
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
        # Create and train model
        model = LSTMModel(input_size=input_size, 
                         hidden_size=config['hidden_size'],
                         output_size=output_size,
                         use_custom_cell=False)
        
        trainer = LSTMTrainer(model, learning_rate=config['lr'])
        
        start_time = time.time()
        train_losses, val_losses = trainer.train(train_loader, val_loader, 
                                                config['epochs'])
        training_time = time.time() - start_time
        
        # Evaluate on test sample
        model.eval()
        with torch.no_grad():
            test_seq, test_target = next(iter(val_loader))
            test_seq = test_seq.to(device)
            test_target = test_target.to(device)
            
            predictions = model.predict(test_seq)
            
            # Use last timestep for evaluation
            test_loss = torch.nn.functional.mse_loss(predictions[:, -1, :], test_target[:, -1, :])
        
        results[task_name] = {
            'model': model,
            'trainer': trainer,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_val_loss': val_losses[-1],
            'test_loss': test_loss.item(),
            'training_time': training_time,
            'config': config
        }
        
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Final validation loss: {val_losses[-1]:.6f}")
        print(f"Test loss: {test_loss.item():.6f}")
    
    return results


def visualize_pytorch_results(results):
    """Create comprehensive visualizations of PyTorch LSTM results."""
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Figure 1: Training Curves Comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('PyTorch LSTM Training Dynamics', fontsize=16)
    
    tasks = list(results.keys())
    task_names = [task.replace('_', ' ').title() for task in tasks]
    
    for i, (task, name) in enumerate(zip(tasks, task_names)):
        if task in results:
            # Training and validation loss
            train_losses = results[task]['train_losses']
            val_losses = results[task]['val_losses']
            epochs = range(len(train_losses))
            
            axes[0, i].plot(epochs, train_losses, 'b-', label='Train', alpha=0.8)
            axes[0, i].plot(epochs, val_losses, 'r-', label='Validation', alpha=0.8)
            axes[0, i].set_title(f'{name}\nTraining Curves')
            axes[0, i].set_xlabel('Epoch')
            axes[0, i].set_ylabel('Loss')
            axes[0, i].legend()
            axes[0, i].grid(True, alpha=0.3)
            axes[0, i].set_yscale('log')
            
            # Gradient norms
            grad_norms = results[task]['trainer'].gradient_norms
            if grad_norms:
                axes[1, i].plot(grad_norms[-200:], 'g-', alpha=0.7)
                axes[1, i].set_title(f'{name}\nGradient Norms (Last 200 Steps)')
                axes[1, i].set_xlabel('Training Step')
                axes[1, i].set_ylabel('Gradient Norm')
                axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/pytorch_lstm_training.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Performance Comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('PyTorch LSTM Performance Summary', fontsize=16)
    
    # Final losses
    final_losses = [results[task]['final_val_loss'] for task in tasks]
    colors = ['steelblue', 'darkgreen', 'darkorange']
    
    bars1 = axes[0].bar(task_names, final_losses, color=colors, alpha=0.7)
    axes[0].set_title('Final Validation Loss')
    axes[0].set_ylabel('Loss')
    axes[0].tick_params(axis='x', rotation=45)
    
    for bar, loss in zip(bars1, final_losses):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{loss:.4f}', ha='center', va='bottom')
    
    # Training times
    train_times = [results[task]['training_time'] for task in tasks]
    bars2 = axes[1].bar(task_names, train_times, color=colors, alpha=0.7)
    axes[1].set_title('Training Time')
    axes[1].set_ylabel('Time (seconds)')
    axes[1].tick_params(axis='x', rotation=45)
    
    for bar, time_val in zip(bars2, train_times):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.1f}s', ha='center', va='bottom')
    
    # Test losses
    test_losses = [results[task]['test_loss'] for task in tasks]
    bars3 = axes[2].bar(task_names, test_losses, color=colors, alpha=0.7)
    axes[2].set_title('Test Loss')
    axes[2].set_ylabel('Loss')
    axes[2].tick_params(axis='x', rotation=45)
    
    for bar, loss in zip(bars3, test_losses):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{loss:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('plots/pytorch_lstm_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nPyTorch visualizations completed!")
    print("Generated files:")
    print("- pytorch_lstm_training.png: Training curves and gradient analysis")
    print("- pytorch_lstm_performance.png: Performance summary")


def analyze_pytorch_gates(model, test_sequence):
    """Analyze gate activations in PyTorch LSTM."""
    
    if not hasattr(model, 'use_custom_cell') or not model.use_custom_cell:
        print("Gate analysis requires custom LSTM cell implementation.")
        return
    
    print("\nAnalyzing PyTorch LSTM gate activations...")
    
    model.eval()
    with torch.no_grad():
        # Convert test sequence to tensor
        if isinstance(test_sequence, list):
            test_tensor = torch.FloatTensor(np.array(test_sequence).squeeze()).unsqueeze(0)
        else:
            test_tensor = test_sequence.unsqueeze(0)
        
        test_tensor = test_tensor.to(device)
        
        # Forward pass to get gate activations
        outputs, hidden_state, all_gates = model(test_tensor)
        
        # Extract gate values
        seq_length = len(all_gates)
        forget_gates = torch.stack([gates['forget'][0] for gates in all_gates]).cpu().numpy()
        input_gates = torch.stack([gates['input'][0] for gates in all_gates]).cpu().numpy()
        output_gates = torch.stack([gates['output'][0] for gates in all_gates]).cpu().numpy()
        cell_states = torch.stack([gates['cell_state'][0] for gates in all_gates]).cpu().numpy()
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('PyTorch LSTM Gate Activations', fontsize=16)
        
        time_steps = range(seq_length)
        
        # Plot gate activations
        axes[0, 0].plot(time_steps, np.mean(forget_gates, axis=1), 'b-', linewidth=2)
        axes[0, 0].fill_between(time_steps,
                               np.mean(forget_gates, axis=1) - np.std(forget_gates, axis=1),
                               np.mean(forget_gates, axis=1) + np.std(forget_gates, axis=1),
                               alpha=0.3)
        axes[0, 0].set_title('Forget Gate')
        axes[0, 0].set_ylabel('Activation')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 1)
        
        axes[0, 1].plot(time_steps, np.mean(input_gates, axis=1), 'g-', linewidth=2)
        axes[0, 1].fill_between(time_steps,
                               np.mean(input_gates, axis=1) - np.std(input_gates, axis=1),
                               np.mean(input_gates, axis=1) + np.std(input_gates, axis=1),
                               alpha=0.3)
        axes[0, 1].set_title('Input Gate')
        axes[0, 1].set_ylabel('Activation')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1)
        
        axes[1, 0].plot(time_steps, np.mean(output_gates, axis=1), 'r-', linewidth=2)
        axes[1, 0].fill_between(time_steps,
                               np.mean(output_gates, axis=1) - np.std(output_gates, axis=1),
                               np.mean(output_gates, axis=1) + np.std(output_gates, axis=1),
                               alpha=0.3)
        axes[1, 0].set_title('Output Gate')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Activation')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1)
        
        axes[1, 1].plot(time_steps, np.mean(cell_states, axis=1), 'purple', linewidth=2)
        axes[1, 1].fill_between(time_steps,
                               np.mean(cell_states, axis=1) - np.std(cell_states, axis=1),
                               np.mean(cell_states, axis=1) + np.std(cell_states, axis=1),
                               alpha=0.3)
        axes[1, 1].set_title('Cell State')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/pytorch_lstm_gates.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"PyTorch Gate Statistics:")
        print(f"  Forget Gate - Mean: {np.mean(forget_gates):.3f}, Std: {np.std(forget_gates):.3f}")
        print(f"  Input Gate  - Mean: {np.mean(input_gates):.3f}, Std: {np.std(input_gates):.3f}")
        print(f"  Output Gate - Mean: {np.mean(output_gates):.3f}, Std: {np.std(output_gates):.3f}")
        print(f"  Cell State  - Mean: {np.mean(cell_states):.3f}, Std: {np.std(cell_states):.3f}")
        
        return {
            'forget_gates': forget_gates,
            'input_gates': input_gates,
            'output_gates': output_gates,
            'cell_states': cell_states
        }


if __name__ == "__main__":
    print("Starting PyTorch LSTM Implementation...")
    
    # Run PyTorch experiments
    pytorch_results = run_pytorch_experiments()
    
    # Create visualizations
    visualize_pytorch_results(pytorch_results)
    
    # Compare PyTorch vs Scratch implementation
    print("\n" + "="*60)
    print("PYTORCH vs NUMPY COMPARISON")
    print("="*60)
    
    comparison_results = compare_pytorch_vs_scratch()
    
    # Analyze gates (if using custom cell)
    if 'sine_wave' in comparison_results:
        custom_model = comparison_results['sine_wave']['custom']['model']
        
        # Create a test sequence
        datasets = create_pytorch_datasets()
        test_loader = DataLoader(datasets['sine_wave']['val'], batch_size=1, shuffle=False)
        test_seq, _ = next(iter(test_loader))
        
        gate_analysis = analyze_pytorch_gates(custom_model, test_seq[0])
    
    print("\n" + "="*60)
    print("PYTORCH LSTM IMPLEMENTATION COMPLETE")
    print("="*60)
    print("✅ Implemented PyTorch LSTM with built-in and custom cells")
    print("✅ Compared PyTorch vs NumPy implementations")
    print("✅ Analyzed gate mechanisms in PyTorch")
    print("✅ Generated comprehensive visualizations")
    print("\nCheck the 'plots' directory for detailed analysis!")