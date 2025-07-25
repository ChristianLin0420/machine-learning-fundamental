"""
Manual Backpropagation with Detailed Gradient Logging

This module extends the MLP implementation with comprehensive gradient logging,
analysis, and visualization capabilities. Provides detailed insights into
gradient flow, vanishing/exploding gradient detection, and gradient clipping.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings

# Import from previous implementations
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '22_ffnn'))

from mlp_numpy import FeedforwardNeuralNet
from activation_functions import get_activation_function, get_activation_derivative


class MLPWithGradientLogging(FeedforwardNeuralNet):
    """
    Enhanced MLP with comprehensive gradient logging and analysis.
    
    Tracks all gradients during training to analyze gradient flow,
    detect vanishing/exploding gradients, and provide detailed insights
    into training dynamics.
    """
    
    def __init__(self, layers, activations=None, output_activation='sigmoid', 
                 weight_init='xavier', random_state=42, gradient_clip_value=None):
        """
        Initialize MLP with gradient logging capabilities.
        
        Args:
            layers: Network architecture
            activations: Hidden layer activations
            output_activation: Output layer activation
            weight_init: Weight initialization method
            random_state: Random seed
            gradient_clip_value: Value for gradient clipping (None = no clipping)
        """
        super().__init__(layers, activations, output_activation, weight_init, random_state)
        
        self.gradient_clip_value = gradient_clip_value
        
        # Gradient logging
        self.gradient_logs = {
            'weight_gradients': [],      # Weight gradients per epoch
            'bias_gradients': [],        # Bias gradients per epoch
            'weight_norms': [],          # Weight gradient norms per layer
            'bias_norms': [],            # Bias gradient norms per layer
            'activation_gradients': [],   # Gradients w.r.t activations
            'pre_activation_gradients': [], # Gradients w.r.t pre-activations
            'gradient_ratios': [],       # Ratio of gradients between layers
            'weight_updates': [],        # Actual weight updates
            'bias_updates': []           # Actual bias updates
        }
        
        # Gradient statistics
        self.gradient_stats = {
            'max_gradient': [],
            'min_gradient': [],
            'mean_gradient': [],
            'std_gradient': [],
            'vanishing_layers': [],      # Layers with vanishing gradients
            'exploding_layers': []       # Layers with exploding gradients
        }
    
    def _log_gradients(self, weight_gradients: List[np.ndarray], 
                      bias_gradients: List[np.ndarray],
                      activations: List[np.ndarray],
                      pre_activations: List[np.ndarray]) -> None:
        """
        Log detailed gradient information.
        
        Args:
            weight_gradients: Weight gradients for each layer
            bias_gradients: Bias gradients for each layer
            activations: Activations for each layer
            pre_activations: Pre-activations for each layer
        """
        # Store raw gradients
        self.gradient_logs['weight_gradients'].append([g.copy() for g in weight_gradients])
        self.gradient_logs['bias_gradients'].append([g.copy() for g in bias_gradients])
        
        # Compute gradient norms
        weight_norms = [np.linalg.norm(g) for g in weight_gradients]
        bias_norms = [np.linalg.norm(g) for g in bias_gradients]
        
        self.gradient_logs['weight_norms'].append(weight_norms)
        self.gradient_logs['bias_norms'].append(bias_norms)
        
        # Compute activation gradients (simplified approximation)
        activation_grads = []
        for i in range(len(activations) - 1):  # Exclude input
            # Approximate gradient magnitude
            if i < len(weight_gradients):
                # Use weight gradients as proxy for activation gradients
                activation_grad_norm = np.mean(np.abs(weight_gradients[i]))
                activation_grads.append(activation_grad_norm)
        
        self.gradient_logs['activation_gradients'].append(activation_grads)
        
        # Compute gradient ratios between consecutive layers
        gradient_ratios = []
        for i in range(len(weight_norms) - 1):
            if weight_norms[i+1] > 1e-10:  # Avoid division by zero
                ratio = weight_norms[i] / weight_norms[i+1]
                gradient_ratios.append(ratio)
        
        self.gradient_logs['gradient_ratios'].append(gradient_ratios)
        
        # Overall gradient statistics
        all_gradients = np.concatenate([g.flatten() for g in weight_gradients])
        
        self.gradient_stats['max_gradient'].append(np.max(np.abs(all_gradients)))
        self.gradient_stats['min_gradient'].append(np.min(np.abs(all_gradients)))
        self.gradient_stats['mean_gradient'].append(np.mean(np.abs(all_gradients)))
        self.gradient_stats['std_gradient'].append(np.std(all_gradients))
        
        # Detect vanishing/exploding gradients
        vanishing_threshold = 1e-6
        exploding_threshold = 10.0
        
        vanishing_layers = []
        exploding_layers = []
        
        for i, norm in enumerate(weight_norms):
            if norm < vanishing_threshold:
                vanishing_layers.append(i)
            elif norm > exploding_threshold:
                exploding_layers.append(i)
        
        self.gradient_stats['vanishing_layers'].append(vanishing_layers)
        self.gradient_stats['exploding_layers'].append(exploding_layers)
    
    def _clip_gradients(self, weight_gradients: List[np.ndarray], 
                       bias_gradients: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Apply gradient clipping if specified.
        
        Args:
            weight_gradients: Original weight gradients
            bias_gradients: Original bias gradients
            
        Returns:
            tuple: (clipped_weight_gradients, clipped_bias_gradients)
        """
        if self.gradient_clip_value is None:
            return weight_gradients, bias_gradients
        
        clipped_weight_grads = []
        clipped_bias_grads = []
        
        for w_grad in weight_gradients:
            norm = np.linalg.norm(w_grad)
            if norm > self.gradient_clip_value:
                clipped_grad = w_grad * (self.gradient_clip_value / norm)
                clipped_weight_grads.append(clipped_grad)
            else:
                clipped_weight_grads.append(w_grad)
        
        for b_grad in bias_gradients:
            norm = np.linalg.norm(b_grad)
            if norm > self.gradient_clip_value:
                clipped_grad = b_grad * (self.gradient_clip_value / norm)
                clipped_bias_grads.append(clipped_grad)
            else:
                clipped_bias_grads.append(b_grad)
        
        return clipped_weight_grads, clipped_bias_grads
    
    def backward_with_logging(self, X: np.ndarray, y: np.ndarray, 
                             activations: List[np.ndarray], 
                             pre_activations: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Enhanced backward pass with detailed logging.
        
        Args:
            X: Input data
            y: Target labels
            activations: Forward pass activations
            pre_activations: Forward pass pre-activations
            
        Returns:
            tuple: (weight_gradients, bias_gradients)
        """
        # Perform standard backpropagation
        weight_gradients, bias_gradients = self.backward(X, y, activations, pre_activations)
        
        # Log gradients before clipping
        self._log_gradients(weight_gradients, bias_gradients, activations, pre_activations)
        
        # Apply gradient clipping
        clipped_weight_grads, clipped_bias_grads = self._clip_gradients(weight_gradients, bias_gradients)
        
        # Log weight updates
        weight_updates = [lr * g for g, lr in zip(clipped_weight_grads, [0.01] * len(clipped_weight_grads))]
        bias_updates = [lr * g for g, lr in zip(clipped_bias_grads, [0.01] * len(clipped_bias_grads))]
        
        self.gradient_logs['weight_updates'].append(weight_updates)
        self.gradient_logs['bias_updates'].append(bias_updates)
        
        return clipped_weight_grads, clipped_bias_grads
    
    def fit_with_logging(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, 
                        learning_rate: float = 0.01, batch_size: Optional[int] = None,
                        validation_data: Optional[Tuple] = None, verbose: bool = True) -> 'MLPWithGradientLogging':
        """
        Training with comprehensive gradient logging.
        
        Args:
            X: Training features
            y: Training labels
            epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size (None = full batch)
            validation_data: Validation data tuple
            verbose: Print progress
            
        Returns:
            self: Trained model with gradient logs
        """
        # Clear previous logs
        self.gradient_logs = {key: [] for key in self.gradient_logs.keys()}
        self.gradient_stats = {key: [] for key in self.gradient_stats.keys()}
        
        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # Set batch size
        if batch_size is None:
            batch_size = X.shape[0]
        
        # Reset history
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        if verbose:
            print(f"Training MLP with gradient logging:")
            print(f"  Architecture: {self.layers}")
            print(f"  Gradient clipping: {self.gradient_clip_value}")
            print(f"  Epochs: {epochs}, Learning rate: {learning_rate}")
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_accuracy = 0
            n_batches = 0
            
            # Mini-batch training
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                
                # Forward pass
                activations, pre_activations = self.forward(X_batch)
                y_pred = activations[-1]
                
                # Compute loss and accuracy
                batch_loss = self._compute_loss(y_batch, y_pred)
                batch_accuracy = self._compute_accuracy(y_batch, y_pred)
                
                # Enhanced backward pass with logging
                weight_gradients, bias_gradients = self.backward_with_logging(
                    X_batch, y_batch, activations, pre_activations)
                
                # Update weights
                self.update_weights(weight_gradients, bias_gradients, learning_rate)
                
                epoch_loss += batch_loss
                epoch_accuracy += batch_accuracy
                n_batches += 1
            
            # Average metrics
            avg_loss = epoch_loss / n_batches
            avg_accuracy = epoch_accuracy / n_batches
            
            self.history['loss'].append(avg_loss)
            self.history['accuracy'].append(avg_accuracy)
            
            # Validation metrics
            if validation_data is not None:
                X_val, y_val = validation_data
                if y_val.ndim == 1:
                    y_val = y_val.reshape(-1, 1)
                
                val_pred = self.predict_proba(X_val)
                val_loss = self._compute_loss(y_val, val_pred)
                val_accuracy = self._compute_accuracy(y_val, val_pred)
                
                self.history['val_loss'].append(val_loss)
                self.history['val_accuracy'].append(val_accuracy)
            
            # Print progress
            if verbose and (epoch + 1) % (epochs // 10 if epochs >= 10 else 1) == 0:
                msg = f"Epoch {epoch + 1}/{epochs} - "
                msg += f"loss: {avg_loss:.4f} - accuracy: {avg_accuracy:.4f}"
                
                if validation_data is not None:
                    msg += f" - val_loss: {val_loss:.4f} - val_accuracy: {val_accuracy:.4f}"
                
                # Add gradient info
                current_max_grad = self.gradient_stats['max_gradient'][-1] if self.gradient_stats['max_gradient'] else 0
                msg += f" - max_grad: {current_max_grad:.2e}"
                
                print(msg)
        
        return self
    
    def analyze_gradient_flow(self) -> Dict:
        """
        Analyze gradient flow throughout training.
        
        Returns:
            dict: Comprehensive gradient analysis
        """
        if not self.gradient_logs['weight_norms']:
            raise ValueError("No gradient logs available. Train the model first.")
        
        analysis = {
            'gradient_flow_summary': {},
            'vanishing_gradient_analysis': {},
            'exploding_gradient_analysis': {},
            'layer_wise_statistics': {},
            'training_stability': {}
        }
        
        # Convert logs to numpy arrays for analysis
        weight_norms = np.array(self.gradient_logs['weight_norms'])  # (epochs, layers)
        bias_norms = np.array(self.gradient_logs['bias_norms'])
        
        # Gradient flow summary
        analysis['gradient_flow_summary'] = {
            'mean_gradient_norm_per_layer': np.mean(weight_norms, axis=0),
            'std_gradient_norm_per_layer': np.std(weight_norms, axis=0),
            'final_gradient_norms': weight_norms[-1] if len(weight_norms) > 0 else [],
            'gradient_norm_evolution': weight_norms
        }
        
        # Vanishing gradient analysis
        vanishing_epochs = []
        for epoch_vanishing in self.gradient_stats['vanishing_layers']:
            vanishing_epochs.extend(epoch_vanishing)
        
        analysis['vanishing_gradient_analysis'] = {
            'vanishing_layer_frequency': np.bincount(vanishing_epochs) if vanishing_epochs else [],
            'epochs_with_vanishing': sum(1 for v in self.gradient_stats['vanishing_layers'] if v),
            'total_epochs': len(self.gradient_stats['vanishing_layers'])
        }
        
        # Exploding gradient analysis
        exploding_epochs = []
        for epoch_exploding in self.gradient_stats['exploding_layers']:
            exploding_epochs.extend(epoch_exploding)
        
        analysis['exploding_gradient_analysis'] = {
            'exploding_layer_frequency': np.bincount(exploding_epochs) if exploding_epochs else [],
            'epochs_with_exploding': sum(1 for e in self.gradient_stats['exploding_layers'] if e),
            'max_gradient_ever': max(self.gradient_stats['max_gradient']) if self.gradient_stats['max_gradient'] else 0
        }
        
        # Layer-wise statistics
        layer_stats = {}
        for layer_idx in range(len(self.layers) - 1):
            layer_grads = weight_norms[:, layer_idx] if layer_idx < weight_norms.shape[1] else []
            layer_stats[f'layer_{layer_idx}'] = {
                'mean_gradient_norm': np.mean(layer_grads) if len(layer_grads) > 0 else 0,
                'gradient_norm_trend': np.polyfit(range(len(layer_grads)), layer_grads, 1)[0] if len(layer_grads) > 1 else 0,
                'gradient_stability': np.std(layer_grads) / (np.mean(layer_grads) + 1e-8) if len(layer_grads) > 0 else 0
            }
        
        analysis['layer_wise_statistics'] = layer_stats
        
        # Training stability
        gradient_ratios = np.array(self.gradient_logs['gradient_ratios'])
        analysis['training_stability'] = {
            'gradient_ratio_stability': np.std(gradient_ratios.flatten()) if gradient_ratios.size > 0 else 0,
            'mean_gradient_ratio': np.mean(gradient_ratios.flatten()) if gradient_ratios.size > 0 else 0,
            'gradient_flow_balance': np.mean(np.abs(np.log(gradient_ratios.flatten() + 1e-8))) if gradient_ratios.size > 0 else 0
        }
        
        return analysis
    
    def visualize_gradient_flow(self, save_path: str = None, show_plots: bool = True):
        """
        Create comprehensive visualizations of gradient flow.
        
        Args:
            save_path: Base path for saving plots
            show_plots: Whether to display plots
        """
        if not self.gradient_logs['weight_norms']:
            raise ValueError("No gradient logs available. Train the model first.")
        
        # Convert to numpy arrays
        weight_norms = np.array(self.gradient_logs['weight_norms'])
        max_gradients = np.array(self.gradient_stats['max_gradient'])
        mean_gradients = np.array(self.gradient_stats['mean_gradient'])
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Gradient norms across layers and epochs
        ax1 = plt.subplot(3, 3, 1)
        if weight_norms.size > 0:
            im = ax1.imshow(weight_norms.T, aspect='auto', cmap='viridis', origin='lower')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Layer')
            ax1.set_title('Gradient Norms Heatmap')
            plt.colorbar(im, ax=ax1)
        
        # 2. Gradient norms per layer over time
        ax2 = plt.subplot(3, 3, 2)
        if weight_norms.size > 0:
            for layer_idx in range(weight_norms.shape[1]):
                ax2.plot(weight_norms[:, layer_idx], label=f'Layer {layer_idx}', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Gradient Norm')
            ax2.set_title('Gradient Norms Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
        
        # 3. Maximum gradient evolution
        ax3 = plt.subplot(3, 3, 3)
        if len(max_gradients) > 0:
            ax3.plot(max_gradients, 'r-', linewidth=2, label='Max Gradient')
            ax3.plot(mean_gradients, 'b-', linewidth=2, label='Mean Gradient')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Gradient Magnitude')
            ax3.set_title('Gradient Magnitude Evolution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_yscale('log')
        
        # 4. Gradient ratios between layers
        ax4 = plt.subplot(3, 3, 4)
        gradient_ratios = self.gradient_logs['gradient_ratios']
        if gradient_ratios:
            ratios_array = np.array(gradient_ratios)
            if ratios_array.size > 0:
                for i in range(ratios_array.shape[1]):
                    ax4.plot(ratios_array[:, i], label=f'Layers {i}-{i+1}', linewidth=2)
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel('Gradient Ratio')
                ax4.set_title('Gradient Ratios Between Layers')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                ax4.set_yscale('log')
        
        # 5. Vanishing/Exploding gradient detection
        ax5 = plt.subplot(3, 3, 5)
        vanishing_counts = [len(v) for v in self.gradient_stats['vanishing_layers']]
        exploding_counts = [len(e) for e in self.gradient_stats['exploding_layers']]
        
        epochs = range(len(vanishing_counts))
        ax5.plot(epochs, vanishing_counts, 'b-', label='Vanishing Layers', linewidth=2)
        ax5.plot(epochs, exploding_counts, 'r-', label='Exploding Layers', linewidth=2)
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Number of Problematic Layers')
        ax5.set_title('Vanishing/Exploding Gradient Detection')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Weight update magnitudes
        ax6 = plt.subplot(3, 3, 6)
        if self.gradient_logs['weight_updates']:
            update_norms = []
            for epoch_updates in self.gradient_logs['weight_updates']:
                epoch_norm = np.mean([np.linalg.norm(u) for u in epoch_updates])
                update_norms.append(epoch_norm)
            
            ax6.plot(update_norms, 'g-', linewidth=2)
            ax6.set_xlabel('Epoch')
            ax6.set_ylabel('Mean Update Magnitude')
            ax6.set_title('Weight Update Magnitudes')
            ax6.grid(True, alpha=0.3)
            ax6.set_yscale('log')
        
        # 7. Gradient distribution (final epoch)
        ax7 = plt.subplot(3, 3, 7)
        if self.gradient_logs['weight_gradients']:
            final_gradients = self.gradient_logs['weight_gradients'][-1]
            all_final_grads = np.concatenate([g.flatten() for g in final_gradients])
            ax7.hist(all_final_grads, bins=50, alpha=0.7, edgecolor='black')
            ax7.set_xlabel('Gradient Value')
            ax7.set_ylabel('Frequency')
            ax7.set_title('Final Epoch Gradient Distribution')
            ax7.grid(True, alpha=0.3)
        
        # 8. Training loss with gradient info
        ax8 = plt.subplot(3, 3, 8)
        if self.history['loss']:
            ax8_twin = ax8.twinx()
            
            # Plot loss
            line1 = ax8.plot(self.history['loss'], 'b-', linewidth=2, label='Training Loss')
            ax8.set_xlabel('Epoch')
            ax8.set_ylabel('Loss', color='b')
            ax8.tick_params(axis='y', labelcolor='b')
            
            # Plot max gradient
            if len(max_gradients) > 0:
                line2 = ax8_twin.plot(max_gradients, 'r--', linewidth=2, label='Max Gradient')
                ax8_twin.set_ylabel('Max Gradient', color='r')
                ax8_twin.tick_params(axis='y', labelcolor='r')
                ax8_twin.set_yscale('log')
            
            ax8.set_title('Loss vs Gradient Magnitude')
            ax8.grid(True, alpha=0.3)
        
        # 9. Layer gradient balance
        ax9 = plt.subplot(3, 3, 9)
        if weight_norms.size > 0:
            # Show final gradient norms per layer
            final_norms = weight_norms[-1]
            layers = range(len(final_norms))
            bars = ax9.bar(layers, final_norms, alpha=0.7)
            ax9.set_xlabel('Layer Index')
            ax9.set_ylabel('Final Gradient Norm')
            ax9.set_title('Final Gradient Balance Across Layers')
            ax9.grid(True, alpha=0.3)
            
            # Color bars based on magnitude
            max_norm = max(final_norms) if final_norms.size > 0 else 1
            for bar, norm in zip(bars, final_norms):
                if norm < 1e-6:
                    bar.set_color('red')  # Vanishing
                elif norm > max_norm * 0.1:
                    bar.set_color('green')  # Healthy
                else:
                    bar.set_color('orange')  # Weak
        
        plt.suptitle('Comprehensive Gradient Flow Analysis', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_gradient_flow.png", dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close()


def demonstrate_gradient_logging():
    """
    Demonstrate gradient logging capabilities on different problems.
    """
    # Create visualizations directory
    import os
    os.makedirs("visualizations", exist_ok=True)
    
    print("Gradient Logging Demonstration")
    print("=" * 50)
    
    # Import dataset creation
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '22_ffnn'))
    from datasets import make_xor_dataset, make_spiral_dataset, create_train_test_split, normalize_features
    
    # Test on different problems
    problems = {
        "XOR": make_xor_dataset(n_samples=200, noise=0.05, random_state=42),
        "Spiral": make_spiral_dataset(n_samples=200, noise=0.1, random_state=42)
    }
    
    for problem_name, (X, y) in problems.items():
        print(f"\n{problem_name} Problem Analysis")
        print("-" * 30)
        
        # Split data
        X_train, X_test, y_train, y_test = create_train_test_split(X, y, test_size=0.2)
        X_train_norm, X_test_norm, _, _ = normalize_features(X_train, X_test)
        
        # Test different gradient clipping values
        clip_values = [None, 1.0, 0.1]
        
        for clip_value in clip_values:
            clip_str = f"clip_{clip_value}" if clip_value else "no_clip"
            print(f"\nTraining with gradient clipping: {clip_value}")
            
            # Create model with gradient logging
            mlp = MLPWithGradientLogging(
                layers=[2, 10, 10, 1],
                activations=['relu', 'relu'],
                output_activation='sigmoid',
                gradient_clip_value=clip_value,
                random_state=42
            )
            
            # Train with logging
            mlp.fit_with_logging(
                X_train_norm, y_train,
                epochs=100,
                learning_rate=0.01,
                validation_data=(X_test_norm, y_test),
                verbose=False
            )
            
            # Analyze gradients
            analysis = mlp.analyze_gradient_flow()
            
            print(f"  Final accuracy: {mlp.evaluate(X_test_norm, y_test)['accuracy']:.4f}")
            print(f"  Max gradient: {analysis['exploding_gradient_analysis']['max_gradient_ever']:.2e}")
            print(f"  Epochs with vanishing gradients: {analysis['vanishing_gradient_analysis']['epochs_with_vanishing']}")
            print(f"  Epochs with exploding gradients: {analysis['exploding_gradient_analysis']['epochs_with_exploding']}")
            
            # Create visualizations
            mlp.visualize_gradient_flow(
                save_path=f"visualizations/{problem_name.lower()}_{clip_str}",
                show_plots=False
            )
    
    print("\nGradient logging demonstration complete!")
    print("Check the visualizations/ directory for detailed plots.")


if __name__ == "__main__":
    demonstrate_gradient_logging() 