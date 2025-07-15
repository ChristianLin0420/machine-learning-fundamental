"""
HMM Analysis and Visualization Module
====================================

This module provides comprehensive analysis and visualization tools for
Hidden Markov Models, including:

- Training convergence analysis
- State sequence visualization
- Parameter comparison and validation
- Performance metrics and evaluation
- Algorithm comparison (Forward vs Viterbi)
- Real-world application examples

Features:
- Interactive visualizations
- Statistical analysis of results
- Comparative studies across different HMM configurations
- Detailed performance metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from hmm_from_scratch import HiddenMarkovModelScratch, HMMDataGenerator
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class HMMAnalyzer:
    """
    Comprehensive analyzer for Hidden Markov Models.
    
    This class provides tools for analyzing HMM performance,
    visualizing results, and comparing different configurations.
    """
    
    def __init__(self):
        """Initialize the HMM analyzer."""
        self.results = {}
        
    def analyze_training_convergence(self, hmm_models, data_info, save_path=None):
        """
        Analyze and visualize training convergence for multiple HMM models.
        
        Parameters:
        -----------
        hmm_models : dict
            Dictionary of trained HMM models
        data_info : dict
            Information about the datasets used
        save_path : str, optional
            Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Log-likelihood convergence
        ax1 = axes[0, 0]
        for name, hmm in hmm_models.items():
            if hasattr(hmm, 'log_likelihood_history') and hmm.log_likelihood_history:
                ax1.plot(hmm.log_likelihood_history, linewidth=2, marker='o', 
                        markersize=4, label=f'{name} (iter: {hmm.n_iter})')
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Log-Likelihood')
        ax1.set_title('Training Convergence: Log-Likelihood')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Convergence speed comparison
        ax2 = axes[0, 1]
        model_names = []
        iterations = []
        converged_status = []
        
        for name, hmm in hmm_models.items():
            if hasattr(hmm, 'n_iter'):
                model_names.append(name)
                iterations.append(hmm.n_iter)
                converged_status.append(hmm.converged if hasattr(hmm, 'converged') else True)
        
        colors = ['green' if conv else 'red' for conv in converged_status]
        bars = ax2.bar(model_names, iterations, color=colors, alpha=0.7)
        
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Iterations to Convergence')
        ax2.set_title('Convergence Speed Comparison')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add convergence status annotations
        for bar, conv in zip(bars, converged_status):
            height = bar.get_height()
            status = 'Converged' if conv else 'Max iter'
            ax2.annotate(status, xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
        
        # Plot 3: Final log-likelihood comparison
        ax3 = axes[1, 0]
        final_ll = []
        for name in model_names:
            hmm = hmm_models[name]
            if hasattr(hmm, 'log_likelihood_history') and hmm.log_likelihood_history:
                final_ll.append(hmm.log_likelihood_history[-1])
            else:
                final_ll.append(0)
        
        ax3.bar(model_names, final_ll, alpha=0.7)
        ax3.set_xlabel('Model')
        ax3.set_ylabel('Final Log-Likelihood')
        ax3.set_title('Final Model Performance')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Model complexity comparison
        ax4 = axes[1, 1]
        n_states = []
        n_params = []
        
        for name, hmm in hmm_models.items():
            n_states.append(hmm.n_states)
            # Estimate number of parameters
            if hmm.observation_type == 'discrete':
                n_param = hmm.n_states**2 + hmm.n_states * hmm.n_observations + hmm.n_states
            else:
                n_param = hmm.n_states**2 + hmm.n_states * 2 + hmm.n_states  # means + variances
            n_params.append(n_param)
        
        scatter = ax4.scatter(n_states, n_params, s=100, alpha=0.7)
        
        for i, name in enumerate(model_names):
            ax4.annotate(name, (n_states[i], n_params[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax4.set_xlabel('Number of States')
        ax4.set_ylabel('Number of Parameters')
        ax4.set_title('Model Complexity')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training convergence analysis saved to {save_path}")
        
        plt.show()
    
    def visualize_state_sequences(self, hmm, data, sequence_idx=0, save_path=None):
        """
        Visualize predicted vs true state sequences.
        
        Parameters:
        -----------
        hmm : HiddenMarkovModelScratch
            Trained HMM model
        data : dict
            Data dictionary with observations and true states
        sequence_idx : int
            Index of sequence to visualize
        save_path : str, optional
            Path to save the plot
        """
        observations = data['observations'][sequence_idx]
        true_states = data['states'][sequence_idx]
        
        # Get predictions
        predicted_states, log_prob = hmm.predict_states(observations)
        
        # Get state probabilities using forward-backward
        log_prob_seq, log_alpha = hmm.forward(observations, log_space=True)
        log_beta = hmm.backward(observations, log_space=True)
        gamma = hmm._compute_gamma(log_alpha, log_beta)
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        # Plot 1: Observations
        ax1 = axes[0]
        ax1.plot(observations, 'bo-', linewidth=2, markersize=6)
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Observation')
        ax1.set_title(f'Observation Sequence (Log-prob: {log_prob:.2f})')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: True vs Predicted States
        ax2 = axes[1]
        time_steps = range(len(observations))
        
        ax2.plot(time_steps, true_states, 'g-', linewidth=3, label='True States', alpha=0.7)
        ax2.plot(time_steps, predicted_states, 'r--', linewidth=2, label='Predicted States', marker='o', markersize=4)
        
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('State')
        ax2.set_title('State Sequence Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: State Probabilities (Heatmap)
        ax3 = axes[2]
        im = ax3.imshow(gamma.T, cmap='Blues', aspect='auto', interpolation='nearest')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('State')
        ax3.set_title('State Probability Distribution')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Probability')
        
        # Add true state overlay
        ax3.plot(time_steps, true_states, 'g-', linewidth=2, label='True States', alpha=0.8)
        ax3.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"State sequence visualization saved to {save_path}")
        
        plt.show()
        
        # Calculate and print accuracy
        accuracy = accuracy_score(true_states, predicted_states)
        print(f"State prediction accuracy: {accuracy:.4f}")
        
        return {
            'accuracy': accuracy,
            'log_probability': log_prob,
            'predicted_states': predicted_states,
            'state_probabilities': gamma
        }
    
    def compare_algorithms(self, hmm, data, save_path=None):
        """
        Compare Forward algorithm vs Viterbi algorithm performance.
        
        Parameters:
        -----------
        hmm : HiddenMarkovModelScratch
            Trained HMM model
        data : dict
            Data dictionary with observations and true states
        save_path : str, optional
            Path to save the plot
        """
        n_sequences = min(10, len(data['observations']))
        
        forward_probs = []
        viterbi_probs = []
        forward_times = []
        viterbi_times = []
        
        for i in range(n_sequences):
            observations = data['observations'][i]
            
            # Time forward algorithm
            import time
            start_time = time.time()
            forward_prob, _ = hmm.forward(observations, log_space=True)
            forward_time = time.time() - start_time
            
            # Time Viterbi algorithm
            start_time = time.time()
            _, viterbi_prob = hmm.viterbi(observations)
            viterbi_time = time.time() - start_time
            
            forward_probs.append(forward_prob)
            viterbi_probs.append(viterbi_prob)
            forward_times.append(forward_time)
            viterbi_times.append(viterbi_time)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Probability comparison
        ax1 = axes[0, 0]
        ax1.scatter(forward_probs, viterbi_probs, alpha=0.7, s=60)
        
        # Add diagonal line
        min_prob = min(min(forward_probs), min(viterbi_probs))
        max_prob = max(max(forward_probs), max(viterbi_probs))
        ax1.plot([min_prob, max_prob], [min_prob, max_prob], 'r--', alpha=0.7, label='y=x')
        
        ax1.set_xlabel('Forward Log-Probability')
        ax1.set_ylabel('Viterbi Log-Probability')
        ax1.set_title('Forward vs Viterbi Probabilities')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Timing comparison
        ax2 = axes[0, 1]
        x = np.arange(n_sequences)
        width = 0.35
        
        ax2.bar(x - width/2, forward_times, width, label='Forward', alpha=0.7)
        ax2.bar(x + width/2, viterbi_times, width, label='Viterbi', alpha=0.7)
        
        ax2.set_xlabel('Sequence Index')
        ax2.set_ylabel('Computation Time (seconds)')
        ax2.set_title('Algorithm Timing Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Probability difference distribution
        ax3 = axes[1, 0]
        prob_diff = np.array(forward_probs) - np.array(viterbi_probs)
        ax3.hist(prob_diff, bins=15, alpha=0.7, edgecolor='black')
        ax3.axvline(np.mean(prob_diff), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(prob_diff):.3f}')
        ax3.set_xlabel('Forward - Viterbi (Log-Probability)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Probability Difference Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Average timing
        ax4 = axes[1, 1]
        avg_times = [np.mean(forward_times), np.mean(viterbi_times)]
        std_times = [np.std(forward_times), np.std(viterbi_times)]
        
        bars = ax4.bar(['Forward', 'Viterbi'], avg_times, yerr=std_times, 
                      capsize=5, alpha=0.7)
        ax4.set_ylabel('Average Time (seconds)')
        ax4.set_title('Average Algorithm Performance')
        ax4.grid(True, alpha=0.3)
        
        # Add value annotations
        for bar, avg_time in zip(bars, avg_times):
            height = bar.get_height()
            ax4.annotate(f'{avg_time:.4f}s', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Algorithm comparison saved to {save_path}")
        
        plt.show()
        
        return {
            'forward_probs': forward_probs,
            'viterbi_probs': viterbi_probs,
            'forward_times': forward_times,
            'viterbi_times': viterbi_times,
            'prob_difference': prob_diff
        }
    
    def analyze_parameter_learning(self, hmm, true_params, save_path=None):
        """
        Analyze how well the model learned the true parameters.
        
        Parameters:
        -----------
        hmm : HiddenMarkovModelScratch
            Trained HMM model
        true_params : dict
            Dictionary with true parameters
        save_path : str, optional
            Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Transition matrix comparison
        ax1 = axes[0, 0]
        if 'true_A' in true_params:
            true_A = true_params['true_A']
            learned_A = hmm.A
            
            # Flatten matrices for comparison
            true_flat = true_A.flatten()
            learned_flat = learned_A.flatten()
            
            ax1.scatter(true_flat, learned_flat, alpha=0.7, s=60)
            ax1.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Perfect match')
            ax1.set_xlabel('True Transition Probabilities')
            ax1.set_ylabel('Learned Transition Probabilities')
            ax1.set_title('Transition Matrix Learning')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Initial state distribution comparison
        ax2 = axes[0, 1]
        if 'true_pi' in true_params:
            true_pi = true_params['true_pi']
            learned_pi = hmm.pi
            
            x = np.arange(len(true_pi))
            width = 0.35
            
            ax2.bar(x - width/2, true_pi, width, label='True', alpha=0.7)
            ax2.bar(x + width/2, learned_pi, width, label='Learned', alpha=0.7)
            
            ax2.set_xlabel('State')
            ax2.set_ylabel('Initial Probability')
            ax2.set_title('Initial State Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Emission parameters (for discrete HMM)
        ax3 = axes[1, 0]
        if 'true_B' in true_params and hmm.observation_type == 'discrete':
            true_B = true_params['true_B']
            learned_B = hmm.B
            
            # Show as heatmap comparison
            vmin = min(true_B.min(), learned_B.min())
            vmax = max(true_B.max(), learned_B.max())
            
            # Create subplot within subplot for side-by-side heatmaps
            ax3.set_title('Emission Matrix Comparison')
            ax3.axis('off')
            
            # True emission matrix
            ax3_left = fig.add_subplot(2, 4, 7)
            im1 = ax3_left.imshow(true_B, cmap='Blues', vmin=vmin, vmax=vmax)
            ax3_left.set_title('True B')
            ax3_left.set_xlabel('Observation')
            ax3_left.set_ylabel('State')
            
            # Learned emission matrix
            ax3_right = fig.add_subplot(2, 4, 8)
            im2 = ax3_right.imshow(learned_B, cmap='Blues', vmin=vmin, vmax=vmax)
            ax3_right.set_title('Learned B')
            ax3_right.set_xlabel('Observation')
            ax3_right.set_ylabel('State')
            
            # Add colorbar
            plt.colorbar(im2, ax=[ax3_left, ax3_right], shrink=0.6)
        
        # Plot 4: Parameter error analysis
        ax4 = axes[1, 1]
        errors = []
        param_names = []
        
        if 'true_A' in true_params:
            error_A = np.mean(np.abs(true_params['true_A'] - hmm.A))
            errors.append(error_A)
            param_names.append('Transition Matrix')
        
        if 'true_pi' in true_params:
            error_pi = np.mean(np.abs(true_params['true_pi'] - hmm.pi))
            errors.append(error_pi)
            param_names.append('Initial Distribution')
        
        if 'true_B' in true_params and hmm.observation_type == 'discrete':
            error_B = np.mean(np.abs(true_params['true_B'] - hmm.B))
            errors.append(error_B)
            param_names.append('Emission Matrix')
        
        if errors:
            bars = ax4.bar(param_names, errors, alpha=0.7)
            ax4.set_ylabel('Mean Absolute Error')
            ax4.set_title('Parameter Learning Accuracy')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
            
            # Add value annotations
            for bar, error in zip(bars, errors):
                height = bar.get_height()
                ax4.annotate(f'{error:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Parameter learning analysis saved to {save_path}")
        
        plt.show()
        
        return {
            'parameter_errors': dict(zip(param_names, errors)) if errors else {},
            'learned_parameters': {
                'A': hmm.A,
                'pi': hmm.pi,
                'B': hmm.B if hmm.observation_type == 'discrete' else None
            }
        }
    
    def evaluate_model_performance(self, hmm, test_data, save_path=None):
        """
        Comprehensive evaluation of model performance.
        
        Parameters:
        -----------
        hmm : HiddenMarkovModelScratch
            Trained HMM model
        test_data : dict
            Test data with observations and true states
        save_path : str, optional
            Path to save the plot
        """
        all_true_states = []
        all_predicted_states = []
        sequence_accuracies = []
        sequence_log_probs = []
        
        for i in range(len(test_data['observations'])):
            observations = test_data['observations'][i]
            true_states = test_data['states'][i]
            
            # Get predictions
            predicted_states, log_prob = hmm.predict_states(observations)
            
            # Calculate sequence-level metrics
            accuracy = accuracy_score(true_states, predicted_states)
            sequence_accuracies.append(accuracy)
            sequence_log_probs.append(log_prob)
            
            all_true_states.extend(true_states)
            all_predicted_states.extend(predicted_states)
        
        # Overall accuracy
        overall_accuracy = accuracy_score(all_true_states, all_predicted_states)
        
        # Confusion matrix
        cm = confusion_matrix(all_true_states, all_predicted_states)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Sequence accuracy distribution
        ax1 = axes[0, 0]
        ax1.hist(sequence_accuracies, bins=15, alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(sequence_accuracies), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(sequence_accuracies):.3f}')
        ax1.set_xlabel('Sequence Accuracy')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Sequence-Level Accuracy Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Log-probability distribution
        ax2 = axes[0, 1]
        ax2.hist(sequence_log_probs, bins=15, alpha=0.7, edgecolor='black')
        ax2.axvline(np.mean(sequence_log_probs), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(sequence_log_probs):.2f}')
        ax2.set_xlabel('Sequence Log-Probability')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Sequence Log-Probability Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Confusion matrix
        ax3 = axes[1, 0]
        im = ax3.imshow(cm, cmap='Blues', interpolation='nearest')
        ax3.set_xlabel('Predicted State')
        ax3.set_ylabel('True State')
        ax3.set_title(f'Confusion Matrix (Accuracy: {overall_accuracy:.3f})')
        
        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax3.text(j, i, cm[i, j], ha='center', va='center', 
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        
        plt.colorbar(im, ax=ax3)
        
        # Plot 4: Performance metrics summary
        ax4 = axes[1, 1]
        
        # Calculate per-class metrics
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_true_states, all_predicted_states, average=None, zero_division=0
        )
        
        x = np.arange(len(precision))
        width = 0.25
        
        ax4.bar(x - width, precision, width, label='Precision', alpha=0.7)
        ax4.bar(x, recall, width, label='Recall', alpha=0.7)
        ax4.bar(x + width, f1, width, label='F1-Score', alpha=0.7)
        
        ax4.set_xlabel('State')
        ax4.set_ylabel('Score')
        ax4.set_title('Per-State Performance Metrics')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model performance evaluation saved to {save_path}")
        
        plt.show()
        
        # Print detailed classification report
        print("\nDetailed Classification Report:")
        print(classification_report(all_true_states, all_predicted_states))
        
        return {
            'overall_accuracy': overall_accuracy,
            'sequence_accuracies': sequence_accuracies,
            'sequence_log_probs': sequence_log_probs,
            'confusion_matrix': cm,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

def comprehensive_hmm_analysis():
    """
    Run comprehensive HMM analysis with multiple datasets and visualizations.
    """
    print("🔍 COMPREHENSIVE HMM ANALYSIS")
    print("=" * 50)
    
    # Create analyzer
    analyzer = HMMAnalyzer()
    
    # Generate different types of data
    print("\n📊 Generating test datasets...")
    pos_data = HMMDataGenerator.generate_pos_tagging_data(n_sequences=30, sequence_length=20)
    dna_data = HMMDataGenerator.generate_dna_sequence_data(n_sequences=25, sequence_length=50)
    gaussian_data = HMMDataGenerator.generate_gaussian_hmm_data(n_sequences=20, sequence_length=30)
    
    # Train different HMM models
    print("\n🔧 Training HMM models...")
    
    # POS Tagging HMM
    hmm_pos = HiddenMarkovModelScratch(
        n_states=pos_data['n_states'],
        n_observations=pos_data['n_observations'],
        observation_type='discrete'
    )
    hmm_pos.initialize_parameters('random')
    hmm_pos.baum_welch(pos_data['observations'][:20], max_iter=30)
    
    # DNA Sequence HMM
    hmm_dna = HiddenMarkovModelScratch(
        n_states=dna_data['n_states'],
        n_observations=dna_data['n_observations'],
        observation_type='discrete'
    )
    hmm_dna.initialize_parameters('random')
    hmm_dna.baum_welch(dna_data['observations'][:15], max_iter=30)
    
    # Gaussian HMM
    hmm_gaussian = HiddenMarkovModelScratch(
        n_states=gaussian_data['n_states'],
        observation_type='gaussian'
    )
    hmm_gaussian.initialize_parameters('random')
    hmm_gaussian.baum_welch(gaussian_data['observations'][:15], max_iter=30)
    
    # Collect models
    hmm_models = {
        'POS Tagging': hmm_pos,
        'DNA Sequence': hmm_dna,
        'Gaussian': hmm_gaussian
    }
    
    data_info = {
        'POS Tagging': pos_data,
        'DNA Sequence': dna_data,
        'Gaussian': gaussian_data
    }
    
    # Analysis 1: Training convergence
    print("\n📈 Analyzing training convergence...")
    analyzer.analyze_training_convergence(hmm_models, data_info, 'plots/hmm_training_convergence.png')
    
    # Analysis 2: State sequence visualization
    print("\n🔗 Visualizing state sequences...")
    
    # POS tagging visualization
    pos_results = analyzer.visualize_state_sequences(
        hmm_pos, pos_data, sequence_idx=0, 
        save_path='plots/hmm_pos_states.png'
    )
    
    # DNA sequence visualization
    dna_results = analyzer.visualize_state_sequences(
        hmm_dna, dna_data, sequence_idx=0,
        save_path='plots/hmm_dna_states.png'
    )
    
    # Analysis 3: Algorithm comparison
    print("\n⚡ Comparing algorithms...")
    algorithm_results = analyzer.compare_algorithms(
        hmm_pos, pos_data,
        save_path='plots/hmm_algorithm_comparison.png'
    )
    
    # Analysis 4: Parameter learning analysis
    print("\n🎯 Analyzing parameter learning...")
    
    # For POS tagging (has true parameters)
    pos_param_results = analyzer.analyze_parameter_learning(
        hmm_pos, pos_data,
        save_path='plots/hmm_parameter_learning.png'
    )
    
    # Analysis 5: Model performance evaluation
    print("\n📊 Evaluating model performance...")
    
    # Use remaining data for testing
    test_pos_data = {
        'observations': pos_data['observations'][20:],
        'states': pos_data['states'][20:]
    }
    
    performance_results = analyzer.evaluate_model_performance(
        hmm_pos, test_pos_data,
        save_path='plots/hmm_performance_evaluation.png'
    )
    
    print("\n✅ COMPREHENSIVE HMM ANALYSIS COMPLETE!")
    print("📁 Check the 'plots' folder for generated visualizations.")
    print("\n📋 SUMMARY RESULTS:")
    print(f"• POS Tagging Accuracy: {pos_results['accuracy']:.4f}")
    print(f"• DNA Sequence Accuracy: {dna_results['accuracy']:.4f}")
    print(f"• Overall Test Accuracy: {performance_results['overall_accuracy']:.4f}")
    print(f"• Average Forward Time: {np.mean(algorithm_results['forward_times']):.4f}s")
    print(f"• Average Viterbi Time: {np.mean(algorithm_results['viterbi_times']):.4f}s")
    
    return {
        'models': hmm_models,
        'data': data_info,
        'results': {
            'pos_states': pos_results,
            'dna_states': dna_results,
            'algorithms': algorithm_results,
            'parameters': pos_param_results,
            'performance': performance_results
        }
    }

def main():
    """
    Main function to run HMM analysis.
    """
    print("🔍 HMM ANALYSIS MODULE")
    print("=" * 30)
    
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Run comprehensive analysis
    results = comprehensive_hmm_analysis()
    
    return results

if __name__ == "__main__":
    main() 