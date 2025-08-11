"""
LSTM Gate Visualization and Analysis
Author: ML Fundamentals Series
Day 32: Long Short-Term Memory Networks

This module provides specialized tools for visualizing and analyzing
LSTM gate activations, cell states, and internal dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import os
from typing import Dict, List, Tuple

# Set style for consistent plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class LSTMGateAnalyzer:
    """
    Comprehensive analyzer for LSTM gate behaviors and cell state dynamics.
    """
    
    def __init__(self):
        self.gate_data = {}
        self.sequence_data = {}
        
    def collect_gate_data(self, lstm_model, test_sequences, sequence_names=None):
        """
        Collect gate activation data from LSTM model on test sequences.
        
        Args:
            lstm_model: Trained LSTM model
            test_sequences: List of test sequences
            sequence_names: Optional names for sequences
        """
        if sequence_names is None:
            sequence_names = [f"Sequence_{i}" for i in range(len(test_sequences))]
        
        print("Collecting gate activation data...")
        
        for seq_idx, (sequence, name) in enumerate(zip(test_sequences, sequence_names)):
            # Forward pass to get gate values
            outputs, hidden_states, cell_states, all_gates = lstm_model.forward(sequence)
            
            # Extract gate activations
            seq_length = len(all_gates)
            hidden_size = lstm_model.hidden_size
            
            # Organize gate data
            gate_data = {
                'forget_gates': np.array([gates['forget'].flatten() for gates in all_gates]),
                'input_gates': np.array([gates['input'].flatten() for gates in all_gates]),
                'candidate_values': np.array([gates['candidate'].flatten() for gates in all_gates]),
                'output_gates': np.array([gates['output'].flatten() for gates in all_gates]),
                'cell_states': np.array([gates['cell_state'].flatten() for gates in all_gates]),
                'hidden_states': np.array([h.flatten() for h in hidden_states[1:]]),  # Skip initial state
                'outputs': np.array([o.flatten() for o in outputs]),
                'sequence_length': seq_length,
                'hidden_size': hidden_size
            }
            
            self.gate_data[name] = gate_data
            self.sequence_data[name] = sequence
        
        print(f"Collected data for {len(test_sequences)} sequences")
    
    def plot_gate_activations_over_time(self, sequence_name, save_path=None):
        """Plot gate activations over time for a specific sequence."""
        
        if sequence_name not in self.gate_data:
            raise ValueError(f"Sequence '{sequence_name}' not found in collected data")
        
        data = self.gate_data[sequence_name]
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(f'LSTM Gate Activations Over Time: {sequence_name}', fontsize=16)
        
        time_steps = range(data['sequence_length'])
        
        # Forget gate
        axes[0, 0].plot(time_steps, np.mean(data['forget_gates'], axis=1), 'b-', linewidth=2)
        axes[0, 0].fill_between(time_steps,
                               np.mean(data['forget_gates'], axis=1) - np.std(data['forget_gates'], axis=1),
                               np.mean(data['forget_gates'], axis=1) + np.std(data['forget_gates'], axis=1),
                               alpha=0.3)
        axes[0, 0].set_title('Forget Gate Activation')
        axes[0, 0].set_ylabel('Activation')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 1)
        
        # Input gate
        axes[0, 1].plot(time_steps, np.mean(data['input_gates'], axis=1), 'g-', linewidth=2)
        axes[0, 1].fill_between(time_steps,
                               np.mean(data['input_gates'], axis=1) - np.std(data['input_gates'], axis=1),
                               np.mean(data['input_gates'], axis=1) + np.std(data['input_gates'], axis=1),
                               alpha=0.3)
        axes[0, 1].set_title('Input Gate Activation')
        axes[0, 1].set_ylabel('Activation')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1)
        
        # Output gate
        axes[1, 0].plot(time_steps, np.mean(data['output_gates'], axis=1), 'r-', linewidth=2)
        axes[1, 0].fill_between(time_steps,
                               np.mean(data['output_gates'], axis=1) - np.std(data['output_gates'], axis=1),
                               np.mean(data['output_gates'], axis=1) + np.std(data['output_gates'], axis=1),
                               alpha=0.3)
        axes[1, 0].set_title('Output Gate Activation')
        axes[1, 0].set_ylabel('Activation')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1)
        
        # Candidate values
        axes[1, 1].plot(time_steps, np.mean(data['candidate_values'], axis=1), 'orange', linewidth=2)
        axes[1, 1].fill_between(time_steps,
                               np.mean(data['candidate_values'], axis=1) - np.std(data['candidate_values'], axis=1),
                               np.mean(data['candidate_values'], axis=1) + np.std(data['candidate_values'], axis=1),
                               alpha=0.3)
        axes[1, 1].set_title('Candidate Values')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(-1, 1)
        
        # Cell state
        axes[2, 0].plot(time_steps, np.mean(data['cell_states'], axis=1), 'purple', linewidth=2)
        axes[2, 0].fill_between(time_steps,
                               np.mean(data['cell_states'], axis=1) - np.std(data['cell_states'], axis=1),
                               np.mean(data['cell_states'], axis=1) + np.std(data['cell_states'], axis=1),
                               alpha=0.3)
        axes[2, 0].set_title('Cell State Evolution')
        axes[2, 0].set_xlabel('Time Step')
        axes[2, 0].set_ylabel('Cell State')
        axes[2, 0].grid(True, alpha=0.3)
        
        # Hidden state
        axes[2, 1].plot(time_steps, np.mean(data['hidden_states'], axis=1), 'brown', linewidth=2)
        axes[2, 1].fill_between(time_steps,
                               np.mean(data['hidden_states'], axis=1) - np.std(data['hidden_states'], axis=1),
                               np.mean(data['hidden_states'], axis=1) + np.std(data['hidden_states'], axis=1),
                               alpha=0.3)
        axes[2, 1].set_title('Hidden State Evolution')
        axes[2, 1].set_xlabel('Time Step')
        axes[2, 1].set_ylabel('Hidden State')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_gate_heatmaps(self, sequence_name, save_path=None):
        """Create heatmaps showing gate activations across hidden units and time."""
        
        if sequence_name not in self.gate_data:
            raise ValueError(f"Sequence '{sequence_name}' not found in collected data")
        
        data = self.gate_data[sequence_name]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'LSTM Gate Activation Heatmaps: {sequence_name}', fontsize=16)
        
        # Transpose data for heatmap (hidden_units x time_steps)
        forget_heatmap = data['forget_gates'].T
        input_heatmap = data['input_gates'].T
        output_heatmap = data['output_gates'].T
        candidate_heatmap = data['candidate_values'].T
        cell_heatmap = data['cell_states'].T
        hidden_heatmap = data['hidden_states'].T
        
        # Plot heatmaps
        im1 = axes[0, 0].imshow(forget_heatmap, aspect='auto', cmap='Blues', vmin=0, vmax=1)
        axes[0, 0].set_title('Forget Gate')
        axes[0, 0].set_ylabel('Hidden Unit')
        plt.colorbar(im1, ax=axes[0, 0])
        
        im2 = axes[0, 1].imshow(input_heatmap, aspect='auto', cmap='Greens', vmin=0, vmax=1)
        axes[0, 1].set_title('Input Gate')
        axes[0, 1].set_ylabel('Hidden Unit')
        plt.colorbar(im2, ax=axes[0, 1])
        
        im3 = axes[0, 2].imshow(output_heatmap, aspect='auto', cmap='Reds', vmin=0, vmax=1)
        axes[0, 2].set_title('Output Gate')
        axes[0, 2].set_ylabel('Hidden Unit')
        plt.colorbar(im3, ax=axes[0, 2])
        
        im4 = axes[1, 0].imshow(candidate_heatmap, aspect='auto', cmap='Oranges', vmin=-1, vmax=1)
        axes[1, 0].set_title('Candidate Values')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Hidden Unit')
        plt.colorbar(im4, ax=axes[1, 0])
        
        im5 = axes[1, 1].imshow(cell_heatmap, aspect='auto', cmap='Purples')
        axes[1, 1].set_title('Cell State')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Hidden Unit')
        plt.colorbar(im5, ax=axes[1, 1])
        
        im6 = axes[1, 2].imshow(hidden_heatmap, aspect='auto', cmap='viridis')
        axes[1, 2].set_title('Hidden State')
        axes[1, 2].set_xlabel('Time Step')
        axes[1, 2].set_ylabel('Hidden Unit')
        plt.colorbar(im6, ax=axes[1, 2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def analyze_gate_correlations(self, sequence_name, save_path=None):
        """Analyze correlations between different gates."""
        
        if sequence_name not in self.gate_data:
            raise ValueError(f"Sequence '{sequence_name}' not found in collected data")
        
        data = self.gate_data[sequence_name]
        
        # Calculate average gate activations over time
        forget_avg = np.mean(data['forget_gates'], axis=1)
        input_avg = np.mean(data['input_gates'], axis=1)
        output_avg = np.mean(data['output_gates'], axis=1)
        candidate_avg = np.mean(data['candidate_values'], axis=1)
        
        # Create correlation matrix
        gate_matrix = np.column_stack([forget_avg, input_avg, output_avg, candidate_avg])
        correlation_matrix = np.corrcoef(gate_matrix.T)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'LSTM Gate Analysis: {sequence_name}', fontsize=16)
        
        # Correlation heatmap
        gate_names = ['Forget', 'Input', 'Output', 'Candidate']
        im = axes[0].imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0].set_xticks(range(len(gate_names)))
        axes[0].set_yticks(range(len(gate_names)))
        axes[0].set_xticklabels(gate_names)
        axes[0].set_yticklabels(gate_names)
        axes[0].set_title('Gate Correlation Matrix')
        
        # Add correlation values to heatmap
        for i in range(len(gate_names)):
            for j in range(len(gate_names)):
                text = axes[0].text(j, i, f'{correlation_matrix[i, j]:.2f}',
                                   ha="center", va="center", color="black" if abs(correlation_matrix[i, j]) < 0.5 else "white")
        
        plt.colorbar(im, ax=axes[0])
        
        # Gate activation distributions
        axes[1].hist(forget_avg, alpha=0.7, label='Forget', bins=20, density=True)
        axes[1].hist(input_avg, alpha=0.7, label='Input', bins=20, density=True)
        axes[1].hist(output_avg, alpha=0.7, label='Output', bins=20, density=True)
        axes[1].set_title('Gate Activation Distributions')
        axes[1].set_xlabel('Activation Value')
        axes[1].set_ylabel('Density')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Forget vs Input gate relationship
        axes[2].scatter(forget_avg, input_avg, alpha=0.6, c=range(len(forget_avg)), cmap='viridis')
        axes[2].set_xlabel('Forget Gate')
        axes[2].set_ylabel('Input Gate')
        axes[2].set_title('Forget vs Input Gate\n(Color = Time)')
        axes[2].grid(True, alpha=0.3)
        
        # Add colorbar for time
        cbar = plt.colorbar(axes[2].collections[0], ax=axes[2])
        cbar.set_label('Time Step')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        # Print correlation statistics
        print(f"\nGate Correlation Analysis for {sequence_name}:")
        print(f"  Forget-Input correlation: {correlation_matrix[0, 1]:.3f}")
        print(f"  Forget-Output correlation: {correlation_matrix[0, 2]:.3f}")
        print(f"  Input-Output correlation: {correlation_matrix[1, 2]:.3f}")
        
        return correlation_matrix
    
    def compare_sequences(self, save_path=None):
        """Compare gate behaviors across different sequences."""
        
        if len(self.gate_data) < 2:
            print("Need at least 2 sequences for comparison")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('LSTM Gate Behavior Comparison Across Sequences', fontsize=16)
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        # Compare average gate activations
        for i, (seq_name, data) in enumerate(self.gate_data.items()):
            color = colors[i % len(colors)]
            
            # Average activations over hidden units
            forget_avg = np.mean(data['forget_gates'], axis=1)
            input_avg = np.mean(data['input_gates'], axis=1)
            output_avg = np.mean(data['output_gates'], axis=1)
            
            time_steps = range(len(forget_avg))
            
            axes[0, 0].plot(time_steps, forget_avg, color=color, label=seq_name, linewidth=2)
            axes[0, 1].plot(time_steps, input_avg, color=color, label=seq_name, linewidth=2)
            axes[1, 0].plot(time_steps, output_avg, color=color, label=seq_name, linewidth=2)
            
            # Cell state magnitude
            cell_mag = np.mean(np.abs(data['cell_states']), axis=1)
            axes[1, 1].plot(time_steps, cell_mag, color=color, label=seq_name, linewidth=2)
        
        axes[0, 0].set_title('Forget Gate Comparison')
        axes[0, 0].set_ylabel('Average Activation')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 1)
        
        axes[0, 1].set_title('Input Gate Comparison')
        axes[0, 1].set_ylabel('Average Activation')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1)
        
        axes[1, 0].set_title('Output Gate Comparison')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Average Activation')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1)
        
        axes[1, 1].set_title('Cell State Magnitude Comparison')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Average |Cell State|')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def create_comprehensive_report(self, output_dir='plots'):
        """Create a comprehensive visualization report."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("Creating comprehensive LSTM gate analysis report...")
        
        # 1. Individual sequence analysis
        for seq_name in self.gate_data.keys():
            # Gate activations over time
            self.plot_gate_activations_over_time(
                seq_name, 
                save_path=os.path.join(output_dir, f'gate_activations_{seq_name}.png')
            )
            
            # Gate heatmaps
            self.plot_gate_heatmaps(
                seq_name,
                save_path=os.path.join(output_dir, f'gate_heatmaps_{seq_name}.png')
            )
            
            # Gate correlations
            self.analyze_gate_correlations(
                seq_name,
                save_path=os.path.join(output_dir, f'gate_correlations_{seq_name}.png')
            )
        
        # 2. Cross-sequence comparison
        if len(self.gate_data) > 1:
            self.compare_sequences(
                save_path=os.path.join(output_dir, 'gate_sequence_comparison.png')
            )
        
        # 3. Summary statistics
        self.print_summary_statistics()
        
        print(f"\nComprehensive report created in '{output_dir}' directory!")
    
    def print_summary_statistics(self):
        """Print summary statistics for all sequences."""
        
        print("\n" + "="*60)
        print("LSTM GATE ACTIVATION SUMMARY STATISTICS")
        print("="*60)
        
        for seq_name, data in self.gate_data.items():
            print(f"\nSequence: {seq_name}")
            print(f"  Length: {data['sequence_length']} steps")
            print(f"  Hidden Size: {data['hidden_size']} units")
            
            # Calculate statistics
            forget_stats = {
                'mean': np.mean(data['forget_gates']),
                'std': np.std(data['forget_gates']),
                'min': np.min(data['forget_gates']),
                'max': np.max(data['forget_gates'])
            }
            
            input_stats = {
                'mean': np.mean(data['input_gates']),
                'std': np.std(data['input_gates']),
                'min': np.min(data['input_gates']),
                'max': np.max(data['input_gates'])
            }
            
            output_stats = {
                'mean': np.mean(data['output_gates']),
                'std': np.std(data['output_gates']),
                'min': np.min(data['output_gates']),
                'max': np.max(data['output_gates'])
            }
            
            cell_stats = {
                'mean': np.mean(data['cell_states']),
                'std': np.std(data['cell_states']),
                'min': np.min(data['cell_states']),
                'max': np.max(data['cell_states'])
            }
            
            print(f"  Forget Gate - Mean: {forget_stats['mean']:.3f}, Std: {forget_stats['std']:.3f}")
            print(f"  Input Gate  - Mean: {input_stats['mean']:.3f}, Std: {input_stats['std']:.3f}")
            print(f"  Output Gate - Mean: {output_stats['mean']:.3f}, Std: {output_stats['std']:.3f}")
            print(f"  Cell State  - Mean: {cell_stats['mean']:.3f}, Std: {cell_stats['std']:.3f}")


def run_gate_visualization_demo():
    """Run demonstration of LSTM gate visualization capabilities."""
    
    print("LSTM Gate Visualization Demo")
    print("=" * 50)
    
    # Import LSTM model
    from lstm_from_scratch import LSTMFromScratch, create_sequence_datasets
    
    # Create test data
    datasets = create_sequence_datasets()
    
    # Train a small LSTM for demonstration
    print("Training demonstration LSTM...")
    sequences, targets = datasets['sine_wave']
    
    lstm_model = LSTMFromScratch(input_size=1, hidden_size=20, output_size=1, learning_rate=0.01)
    
    # Quick training (just a few epochs for demo)
    for epoch in range(30):
        for i in range(min(20, len(sequences))):
            lstm_model.train_step(sequences[i], targets[i])
        
        if epoch % 10 == 0:
            print(f"  Demo training epoch {epoch}")
    
    # Select test sequences
    test_sequences = sequences[:3]
    sequence_names = ['Sine_Wave_1', 'Sine_Wave_2', 'Sine_Wave_3']
    
    # Create analyzer and collect data
    analyzer = LSTMGateAnalyzer()
    analyzer.collect_gate_data(lstm_model, test_sequences, sequence_names)
    
    # Create comprehensive report
    analyzer.create_comprehensive_report()
    
    print("\n" + "="*50)
    print("GATE VISUALIZATION DEMO COMPLETE")
    print("="*50)
    print("✅ Analyzed gate activations for multiple sequences")
    print("✅ Created heatmaps showing gate patterns")
    print("✅ Analyzed gate correlations and relationships")
    print("✅ Generated comprehensive visualization report")
    print("\nCheck the 'plots' directory for all visualizations!")


if __name__ == "__main__":
    print("Starting LSTM Gate Visualization...")
    
    # Run demonstration
    run_gate_visualization_demo()
    
    print("\n" + "="*60)
    print("LSTM GATE VISUALIZATION COMPLETE")
    print("="*60)
    print("✅ Comprehensive gate analysis tools implemented")
    print("✅ Multiple visualization types available")
    print("✅ Statistical analysis included")
    print("✅ Ready for detailed LSTM interpretation")