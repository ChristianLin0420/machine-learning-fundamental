"""
Decision Boundary Visualization for Perceptron

This module provides comprehensive visualization tools for perceptron decision boundaries,
training dynamics, and classification regions.
"""

import numpy as np
import matplotlib.pyplot as plt
try:
    import matplotlib.animation as animation
    ANIMATION_AVAILABLE = True
except ImportError:
    ANIMATION_AVAILABLE = False
    print("Animation not available. Some features will be disabled.")
from matplotlib.colors import ListedColormap
from perceptron import Perceptron
from synthetic_data import (make_linearly_separable_data, make_noisy_linear_data, 
                           make_xor_data, make_concentric_circles_data)


def plot_decision_boundary_2d(classifier, X, y, title="Decision Boundary", 
                             save_path=None, h=0.01, show_support_vectors=False):
    """
    Plot 2D decision boundary for a trained classifier.
    
    Args:
        classifier: Trained classifier with predict method
        X (np.ndarray): 2D training data
        y (np.ndarray): Training labels
        title (str): Plot title
        save_path (str): Path to save plot
        h (float): Step size for meshgrid
        show_support_vectors (bool): Whether to highlight support vectors
    """
    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    
    # Create a mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Make predictions on the mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = classifier.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=cmap_light)
    
    # Plot the training points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, 
                         edgecolors='black', s=60)
    
    # Plot decision boundary line for perceptron
    if hasattr(classifier, 'get_decision_boundary_params'):
        try:
            slope, intercept = classifier.get_decision_boundary_params()
            x_range = np.linspace(x_min, x_max, 100)
            
            if slope is not None:
                y_boundary = slope * x_range + intercept
                # Only plot within the y range
                mask = (y_boundary >= y_min) & (y_boundary <= y_max)
                plt.plot(x_range[mask], y_boundary[mask], 'k-', linewidth=3, 
                        label='Decision Boundary')
            else:
                plt.axvline(x=intercept, color='k', linewidth=3, 
                           label='Decision Boundary')
        except:
            pass
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_animation(X, y, learning_rate=1.0, max_epochs=100, 
                          save_path=None, interval=200):
    """
    Create an animation showing perceptron training process.
    
    Args:
        X (np.ndarray): Training data
        y (np.ndarray): Training labels
        learning_rate (float): Learning rate
        max_epochs (int): Maximum epochs
        save_path (str): Path to save animation
        interval (int): Animation interval in milliseconds
    """
    if not ANIMATION_AVAILABLE:
        print("Animation not available. Creating static plots instead.")
        # Create a static visualization instead
        perceptron = Perceptron(learning_rate=learning_rate, max_epochs=max_epochs)
        perceptron.fit(X, y, verbose=False)
        plot_decision_boundary_2d(perceptron, X, y, 
                                 title="Perceptron Final Result",
                                 save_path=save_path.replace('.gif', '_static.png') if save_path else None)
        return None
    # Custom perceptron that stores all weight updates
    class AnimatedPerceptron(Perceptron):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.weight_updates = []
            self.misclassified_points = []
        
        def fit(self, X, y, verbose=False):
            X_with_bias = self._add_bias_term(X)
            n_samples, n_features = X_with_bias.shape
            self._initialize_weights(n_features)
            
            self.weight_updates = [self.weights.copy()]
            self.misclassified_points = []
            
            for epoch in range(self.max_epochs):
                epoch_misclassified = []
                errors = 0
                
                for i in range(n_samples):
                    x_i = X_with_bias[i]
                    y_i = y[i]
                    prediction = self._predict_sample(x_i)
                    
                    if prediction != y_i:
                        epoch_misclassified.append(i)
                        self.weights += self.learning_rate * y_i * x_i
                        self.weight_updates.append(self.weights.copy())
                        errors += 1
                
                self.misclassified_points.append(epoch_misclassified)
                
                if errors == 0:
                    break
            
            return self
    
    # Train the animated perceptron
    perceptron = AnimatedPerceptron(learning_rate=learning_rate, max_epochs=max_epochs)
    perceptron.fit(X, y)
    
    # Set up the figure and axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot setup
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    def animate(frame):
        ax1.clear()
        ax2.clear()
        
        if frame < len(perceptron.weight_updates):
            # Get current weights
            current_weights = perceptron.weight_updates[frame]
            
            # Plot data points
            mask_pos = y == 1
            mask_neg = y == -1
            
            ax1.scatter(X[mask_pos, 0], X[mask_pos, 1], c='red', marker='o', 
                       label='Class +1', s=60, alpha=0.7)
            ax1.scatter(X[mask_neg, 0], X[mask_neg, 1], c='blue', marker='s', 
                       label='Class -1', s=60, alpha=0.7)
            
            # Highlight misclassified points
            if frame < len(perceptron.misclassified_points):
                misclassified = perceptron.misclassified_points[frame]
                if misclassified:
                    ax1.scatter(X[misclassified, 0], X[misclassified, 1], 
                               c='yellow', marker='x', s=100, linewidth=3,
                               label='Misclassified')
            
            # Plot decision boundary
            if len(current_weights) == 3 and abs(current_weights[2]) > 1e-10:
                w0, w1, w2 = current_weights
                slope = -w1 / w2
                intercept = -w0 / w2
                
                x_range = np.linspace(x_min, x_max, 100)
                y_boundary = slope * x_range + intercept
                
                # Only plot within bounds
                mask = (y_boundary >= y_min) & (y_boundary <= y_max)
                ax1.plot(x_range[mask], y_boundary[mask], 'g-', linewidth=2, 
                        label='Decision Boundary')
            
            ax1.set_xlim(x_min, x_max)
            ax1.set_ylim(y_min, y_max)
            ax1.set_xlabel('Feature 1')
            ax1.set_ylabel('Feature 2')
            ax1.set_title(f'Perceptron Training - Update {frame}')
            # Only add legend if there are labeled elements
            handles, labels = ax1.get_legend_handles_labels()
            if handles:
                ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot weight evolution
            updates_so_far = perceptron.weight_updates[:frame+1]
            if len(updates_so_far) > 1:
                updates_array = np.array(updates_so_far)
                ax2.plot(updates_array[:, 0], label='Bias (w0)', linewidth=2)
                ax2.plot(updates_array[:, 1], label='Weight 1 (w1)', linewidth=2)
                ax2.plot(updates_array[:, 2], label='Weight 2 (w2)', linewidth=2)
            
            ax2.set_xlabel('Update Step')
            ax2.set_ylabel('Weight Value')
            ax2.set_title('Weight Evolution')
            # Only add legend if there are labeled elements
            handles, labels = ax2.get_legend_handles_labels()
            if handles:
                ax2.legend()
            ax2.grid(True, alpha=0.3)
    
    # Create animation
    try:
        anim = animation.FuncAnimation(fig, animate, frames=len(perceptron.weight_updates),
                                      interval=interval, repeat=True, blit=False)
        
        plt.tight_layout()
        
        if save_path:
            # Save as GIF
            try:
                writer = animation.PillowWriter(fps=5)
                anim.save(save_path, writer=writer)
                print(f"Animation saved to {save_path}")
            except Exception as e:
                print(f"Could not save animation: {e}")
                # Save final frame instead
                plt.savefig(save_path.replace('.gif', '_final_frame.png'), dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return anim
    except Exception as e:
        print(f"Animation creation failed: {e}")
        plt.close()
        return None


def plot_convergence_analysis(datasets_info, save_path=None):
    """
    Analyze convergence behavior across different datasets.
    
    Args:
        datasets_info (dict): Dictionary with dataset names and generation functions
        save_path (str): Path to save plot
    """
    fig, axes = plt.subplots(2, len(datasets_info), figsize=(5*len(datasets_info), 10))
    
    if len(datasets_info) == 1:
        axes = axes.reshape(-1, 1)
    
    results = {}
    
    for col, (name, data_func) in enumerate(datasets_info.items()):
        # Generate data
        X, y = data_func()
        
        # Train perceptron
        perceptron = Perceptron(learning_rate=1.0, max_epochs=1000, random_state=42)
        perceptron.fit(X, y, verbose=False)
        
        # Store results
        results[name] = {
            'converged': perceptron.history['converged_epoch'] is not None,
            'accuracy': perceptron.score(X, y),
            'epochs': perceptron.history['converged_epoch'] or 1000,
            'final_errors': perceptron.history['errors'][-1]
        }
        
        # Plot decision boundary
        plot_decision_boundary_2d(perceptron, X, y, title=f'{name}\nDecision Boundary')
        
        # Save the current plot
        plt.savefig(f"plots/temp_{name.lower().replace(' ', '_')}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot error evolution
        axes[0, col].plot(perceptron.history['errors'], 'b-', linewidth=2)
        axes[0, col].set_title(f'{name}\nError Evolution')
        axes[0, col].set_xlabel('Epoch')
        axes[0, col].set_ylabel('Number of Errors')
        axes[0, col].grid(True, alpha=0.3)
        
        if perceptron.history['converged_epoch'] is not None:
            axes[0, col].axvline(x=perceptron.history['converged_epoch'], 
                                color='r', linestyle='--', alpha=0.7,
                                label=f'Converged at {perceptron.history["converged_epoch"]}')
            axes[0, col].legend()
        
        # Plot weight evolution (for 2D case)
        if len(perceptron.weights) == 3:
            weights_history = np.array(perceptron.history['weights'])
            axes[1, col].plot(weights_history[:, 0], label='Bias (w0)', linewidth=2)
            axes[1, col].plot(weights_history[:, 1], label='Weight 1 (w1)', linewidth=2)
            axes[1, col].plot(weights_history[:, 2], label='Weight 2 (w2)', linewidth=2)
            axes[1, col].set_title(f'{name}\nWeight Evolution')
            axes[1, col].set_xlabel('Epoch')
            axes[1, col].set_ylabel('Weight Value')
            axes[1, col].legend()
            axes[1, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return results


def plot_perceptron_limitations():
    """
    Demonstrate perceptron limitations on non-linearly separable data.
    """
    # Test on XOR and concentric circles
    datasets = {
        "XOR Problem": make_xor_data(n_samples=200, random_state=42),
        "Concentric Circles": make_concentric_circles_data(n_samples=200, random_state=42)
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for col, (name, (X, y)) in enumerate(datasets.items()):
        # Train perceptron
        perceptron = Perceptron(learning_rate=1.0, max_epochs=1000, random_state=42)
        perceptron.fit(X, y, verbose=False)
        
        accuracy = perceptron.score(X, y)
        
        # Plot original data
        mask_pos = y == 1
        mask_neg = y == -1
        
        axes[0, col].scatter(X[mask_pos, 0], X[mask_pos, 1], c='red', marker='o', 
                           label='Class +1', alpha=0.7, s=50)
        axes[0, col].scatter(X[mask_neg, 0], X[mask_neg, 1], c='blue', marker='s', 
                           label='Class -1', alpha=0.7, s=50)
        axes[0, col].set_title(f'{name}\nOriginal Data')
        axes[0, col].legend()
        axes[0, col].grid(True, alpha=0.3)
        
        # Plot decision boundary
        plot_decision_boundary_2d(perceptron, X, y, 
                                 title=f'{name}\nPerceptron Attempt (Acc: {accuracy:.3f})')
        plt.savefig(f"plots/temp_limitation_{col}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot training errors
        axes[1, col].plot(perceptron.history['errors'], 'r-', linewidth=2)
        axes[1, col].set_title(f'{name}\nTraining Errors')
        axes[1, col].set_xlabel('Epoch')
        axes[1, col].set_ylabel('Number of Errors')
        axes[1, col].grid(True, alpha=0.3)
        
        # Add text with final results
        final_errors = perceptron.history['errors'][-1]
        converged = perceptron.history['converged_epoch'] is not None
        
        text = f"Final Errors: {final_errors}\n"
        text += f"Accuracy: {accuracy:.3f}\n"
        text += f"Converged: {converged}"
        
        axes[1, col].text(0.05, 0.95, text, transform=axes[1, col].transAxes,
                         verticalalignment='top', bbox=dict(boxstyle='round', 
                         facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig("plots/perceptron_limitations.png", dpi=300, bbox_inches='tight')
    plt.show()


def create_comprehensive_visualization():
    """
    Create a comprehensive visualization of perceptron behavior.
    """
    print("Creating comprehensive perceptron visualizations...")
    
    # Create plots directory
    import os
    os.makedirs("plots", exist_ok=True)
    
    # 1. Convergence analysis on different datasets
    print("1. Analyzing convergence on different datasets...")
    
    datasets_info = {
        "Linearly Separable": lambda: make_linearly_separable_data(n_samples=100, random_state=42),
        "Noisy Linear": lambda: make_noisy_linear_data(n_samples=100, noise_level=0.2, random_state=42),
        "Very Noisy": lambda: make_noisy_linear_data(n_samples=100, noise_level=0.4, random_state=42)
    }
    
    results = plot_convergence_analysis(datasets_info, "plots/perceptron_convergence_analysis.png")
    
    # 2. Demonstrate limitations
    print("2. Demonstrating perceptron limitations...")
    plot_perceptron_limitations()
    
    # 3. Create training animation for linearly separable data
    print("3. Creating training animation...")
    X, y = make_linearly_separable_data(n_samples=20, random_state=42)  # Small dataset for clear animation
    
    anim = plot_training_animation(X, y, learning_rate=1.0, max_epochs=50,
                                  save_path="plots/perceptron_training_animation.gif",
                                  interval=500)
    
    # 4. Learning rate comparison visualization
    print("4. Analyzing different learning rates...")
    
    X, y = make_linearly_separable_data(n_samples=100, random_state=42)
    learning_rates = [0.1, 0.5, 1.0, 2.0]
    
    fig, axes = plt.subplots(2, len(learning_rates), figsize=(16, 8))
    
    for i, lr in enumerate(learning_rates):
        perceptron = Perceptron(learning_rate=lr, max_epochs=1000, random_state=42)
        perceptron.fit(X, y, verbose=False)
        
        # Plot error evolution
        axes[0, i].plot(perceptron.history['errors'], 'b-', linewidth=2)
        axes[0, i].set_title(f'Learning Rate = {lr}')
        axes[0, i].set_xlabel('Epoch')
        axes[0, i].set_ylabel('Errors')
        axes[0, i].grid(True, alpha=0.3)
        
        if perceptron.history['converged_epoch'] is not None:
            axes[0, i].axvline(x=perceptron.history['converged_epoch'], 
                              color='r', linestyle='--', alpha=0.7,
                              label=f'Converged at {perceptron.history["converged_epoch"]}')
            axes[0, i].legend()
        
        # Plot decision boundary
        plot_decision_boundary_2d(perceptron, X, y, 
                                 title=f'LR = {lr}, Acc = {perceptron.score(X, y):.3f}')
        plt.savefig(f"plots/temp_lr_{i}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot weight evolution
        if len(perceptron.weights) == 3:
            weights_history = np.array(perceptron.history['weights'])
            axes[1, i].plot(weights_history[:, 0], label='Bias', linewidth=2)
            axes[1, i].plot(weights_history[:, 1], label='w1', linewidth=2)
            axes[1, i].plot(weights_history[:, 2], label='w2', linewidth=2)
            axes[1, i].set_title(f'Weight Evolution (LR = {lr})')
            axes[1, i].set_xlabel('Epoch')
            axes[1, i].set_ylabel('Weight Value')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("plots/learning_rate_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Comprehensive visualization complete!")
    return results


if __name__ == "__main__":
    create_comprehensive_visualization() 