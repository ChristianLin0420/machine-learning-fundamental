"""
Kernel Perceptron Implementation

This module implements the kernel perceptron algorithm using the dual formulation,
which allows for non-linear classification through kernel functions.

The dual perceptron maintains alpha coefficients for each training example
and makes predictions using: f(x) = sign(Σ α_i * y_i * K(x_i, x))
"""

import numpy as np
import matplotlib.pyplot as plt
from synthetic_data import (make_xor_data, make_concentric_circles_data, 
                           make_moons_data, make_linearly_separable_data)
from plot_decision_boundary import plot_decision_boundary_2d


class KernelPerceptron:
    """
    Kernel Perceptron Classifier
    
    Uses the dual formulation of the perceptron algorithm with kernel functions
    to handle non-linearly separable data.
    """
    
    def __init__(self, kernel='rbf', kernel_params=None, max_epochs=1000, random_state=42):
        """
        Initialize the kernel perceptron.
        
        Args:
            kernel (str): Kernel type ('linear', 'polynomial', 'rbf')
            kernel_params (dict): Kernel-specific parameters
            max_epochs (int): Maximum number of training epochs
            random_state (int): Random seed
        """
        self.kernel = kernel
        self.kernel_params = kernel_params or {}
        self.max_epochs = max_epochs
        self.random_state = random_state
        
        # Training data and coefficients
        self.X_train = None
        self.y_train = None
        self.alphas = None
        
        # Training history
        self.history = {
            'errors': [],
            'converged_epoch': None,
            'alpha_updates': []
        }
    
    def _kernel_function(self, x1, x2):
        """
        Compute kernel function between two vectors.
        
        Args:
            x1, x2 (np.ndarray): Input vectors
            
        Returns:
            float: Kernel value
        """
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        
        elif self.kernel == 'polynomial':
            degree = self.kernel_params.get('degree', 3)
            coef0 = self.kernel_params.get('coef0', 1)
            return (np.dot(x1, x2) + coef0) ** degree
        
        elif self.kernel == 'rbf':
            gamma = self.kernel_params.get('gamma', 1.0)
            return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)
        
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def _compute_kernel_matrix(self, X1, X2=None):
        """
        Compute kernel matrix between two sets of points.
        
        Args:
            X1 (np.ndarray): First set of points
            X2 (np.ndarray): Second set of points (default: X1)
            
        Returns:
            np.ndarray: Kernel matrix
        """
        if X2 is None:
            X2 = X1
        
        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                K[i, j] = self._kernel_function(X1[i], X2[j])
        
        return K
    
    def _predict_sample(self, x):
        """
        Predict class for a single sample using dual formulation.
        
        Args:
            x (np.ndarray): Single sample
            
        Returns:
            int: Predicted class {-1, 1}
        """
        if self.X_train is None:
            raise ValueError("Model must be trained first")
        
        # Compute kernel values with all training samples
        kernel_values = np.array([self._kernel_function(x_train, x) 
                                 for x_train in self.X_train])
        
        # Dual prediction: f(x) = Σ α_i * y_i * K(x_i, x)
        decision_value = np.sum(self.alphas * self.y_train * kernel_values)
        
        return 1 if decision_value > 0 else -1
    
    def predict(self, X):
        """
        Predict classes for multiple samples.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predicted classes
        """
        predictions = []
        for x in X:
            predictions.append(self._predict_sample(x))
        return np.array(predictions)
    
    def decision_function(self, X):
        """
        Compute decision function values (before sign).
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Decision function values
        """
        decision_values = []
        for x in X:
            kernel_values = np.array([self._kernel_function(x_train, x) 
                                     for x_train in self.X_train])
            decision_value = np.sum(self.alphas * self.y_train * kernel_values)
            decision_values.append(decision_value)
        return np.array(decision_values)
    
    def fit(self, X, y, verbose=True):
        """
        Train the kernel perceptron using the dual algorithm.
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training labels {-1, 1}
            verbose (bool): Whether to print training progress
            
        Returns:
            self: Returns self for method chaining
        """
        self.X_train = X.copy()
        self.y_train = y.copy()
        n_samples = len(X)
        
        # Initialize alpha coefficients
        self.alphas = np.zeros(n_samples)
        
        # Reset history
        self.history = {
            'errors': [],
            'converged_epoch': None,
            'alpha_updates': []
        }
        
        if verbose:
            print(f"Training kernel perceptron with {n_samples} samples")
            print(f"Kernel: {self.kernel}, Parameters: {self.kernel_params}")
            print(f"Max epochs: {self.max_epochs}")
        
        # Precompute kernel matrix for efficiency (optional optimization)
        # K = self._compute_kernel_matrix(X)
        
        # Training loop
        for epoch in range(self.max_epochs):
            errors = 0
            
            # Store current alpha state
            self.history['alpha_updates'].append(self.alphas.copy())
            
            # Go through each training sample
            for i in range(n_samples):
                # Make prediction for current sample
                prediction = self._predict_sample(X[i])
                
                # Check if misclassified
                if prediction != y[i]:
                    # Update alpha coefficient (dual perceptron update)
                    # In dual form: α_i ← α_i + 1 when sample i is misclassified
                    self.alphas[i] += 1
                    errors += 1
            
            self.history['errors'].append(errors)
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: {errors} errors")
            
            # Check for convergence
            if errors == 0:
                self.history['converged_epoch'] = epoch
                if verbose:
                    print(f"Converged after {epoch + 1} epochs!")
                break
        
        if self.history['converged_epoch'] is None and verbose:
            print(f"Did not converge after {self.max_epochs} epochs")
        
        # Store final state
        self.history['alpha_updates'].append(self.alphas.copy())
        
        return self
    
    def score(self, X, y):
        """
        Calculate accuracy on given data.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): True labels
            
        Returns:
            float: Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def get_support_vectors(self):
        """
        Get indices of support vectors (training samples with α > 0).
        
        Returns:
            np.ndarray: Indices of support vectors
        """
        return np.where(self.alphas > 0)[0]
    
    def get_alpha_summary(self):
        """
        Get summary statistics about alpha coefficients.
        
        Returns:
            dict: Summary statistics
        """
        support_vectors = self.get_support_vectors()
        
        return {
            'n_support_vectors': len(support_vectors),
            'support_vector_ratio': len(support_vectors) / len(self.alphas),
            'max_alpha': np.max(self.alphas),
            'mean_alpha': np.mean(self.alphas[support_vectors]) if len(support_vectors) > 0 else 0,
            'alpha_distribution': np.histogram(self.alphas[self.alphas > 0], bins=10)[0] if len(support_vectors) > 0 else []
        }


def compare_kernels_on_dataset(X, y, dataset_name, kernels_to_test=None):
    """
    Compare different kernel functions on a given dataset.
    
    Args:
        X (np.ndarray): Training data
        y (np.ndarray): Training labels
        dataset_name (str): Name of the dataset
        kernels_to_test (list): List of kernel configurations to test
        
    Returns:
        dict: Results for each kernel
    """
    if kernels_to_test is None:
        kernels_to_test = [
            {'kernel': 'linear', 'params': {}},
            {'kernel': 'polynomial', 'params': {'degree': 2, 'coef0': 1}},
            {'kernel': 'polynomial', 'params': {'degree': 3, 'coef0': 1}},
            {'kernel': 'rbf', 'params': {'gamma': 0.5}},
            {'kernel': 'rbf', 'params': {'gamma': 1.0}},
            {'kernel': 'rbf', 'params': {'gamma': 2.0}}
        ]
    
    results = {}
    
    print(f"\n{'='*60}")
    print(f"Testing Kernels on {dataset_name} Dataset")
    print(f"{'='*60}")
    
    for kernel_config in kernels_to_test:
        kernel_name = kernel_config['kernel']
        params = kernel_config['params']
        
        # Create descriptive name
        if kernel_name == 'linear':
            full_name = 'Linear'
        elif kernel_name == 'polynomial':
            degree = params.get('degree', 3)
            full_name = f'Polynomial (degree={degree})'
        elif kernel_name == 'rbf':
            gamma = params.get('gamma', 1.0)
            full_name = f'RBF (γ={gamma})'
        else:
            full_name = kernel_name
        
        print(f"\nTesting {full_name} kernel...")
        
        # Train kernel perceptron
        kp = KernelPerceptron(kernel=kernel_name, kernel_params=params, 
                             max_epochs=1000, random_state=42)
        kp.fit(X, y, verbose=False)
        
        # Evaluate
        accuracy = kp.score(X, y)
        alpha_summary = kp.get_alpha_summary()
        converged = kp.history['converged_epoch'] is not None
        
        results[full_name] = {
            'kernel': kernel_name,
            'params': params,
            'accuracy': accuracy,
            'converged': converged,
            'epochs': kp.history['converged_epoch'] if converged else 1000,
            'final_errors': kp.history['errors'][-1],
            'n_support_vectors': alpha_summary['n_support_vectors'],
            'support_vector_ratio': alpha_summary['support_vector_ratio'],
            'classifier': kp
        }
        
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Converged: {converged}")
        print(f"  Support vectors: {alpha_summary['n_support_vectors']}/{len(X)} "
              f"({alpha_summary['support_vector_ratio']:.1%})")
    
    return results


def visualize_kernel_comparison(X, y, results, dataset_name, save_path=None):
    """
    Visualize the comparison of different kernels.
    
    Args:
        X (np.ndarray): Training data
        y (np.ndarray): Training labels
        results (dict): Results from compare_kernels_on_dataset
        dataset_name (str): Name of the dataset
        save_path (str): Path to save plot
    """
    n_kernels = len(results)
    fig, axes = plt.subplots(2, n_kernels, figsize=(4*n_kernels, 8))
    
    if n_kernels == 1:
        axes = axes.reshape(-1, 1)
    
    for col, (kernel_name, result) in enumerate(results.items()):
        classifier = result['classifier']
        accuracy = result['accuracy']
        
        # Plot decision boundary
        ax1 = axes[0, col]
        
        # Create mesh for decision boundary
        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = classifier.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot decision regions
        ax1.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
        
        # Plot training points
        mask_pos = y == 1
        mask_neg = y == -1
        
        ax1.scatter(X[mask_pos, 0], X[mask_pos, 1], c='red', marker='o', 
                   label='Class +1', s=50, alpha=0.8, edgecolors='black')
        ax1.scatter(X[mask_neg, 0], X[mask_neg, 1], c='blue', marker='s', 
                   label='Class -1', s=50, alpha=0.8, edgecolors='black')
        
        # Highlight support vectors
        support_vectors = classifier.get_support_vectors()
        if len(support_vectors) > 0:
            ax1.scatter(X[support_vectors, 0], X[support_vectors, 1], 
                       s=100, facecolors='none', edgecolors='yellow', 
                       linewidth=2, label='Support Vectors')
        
        ax1.set_xlim(xx.min(), xx.max())
        ax1.set_ylim(yy.min(), yy.max())
        ax1.set_title(f'{kernel_name}\nAccuracy: {accuracy:.3f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot training convergence
        ax2 = axes[1, col]
        ax2.plot(classifier.history['errors'], 'b-', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Number of Errors')
        ax2.set_title(f'Training Convergence')
        ax2.grid(True, alpha=0.3)
        
        if classifier.history['converged_epoch'] is not None:
            ax2.axvline(x=classifier.history['converged_epoch'], 
                       color='r', linestyle='--', alpha=0.7,
                       label=f'Converged at {classifier.history["converged_epoch"]}')
            ax2.legend()
    
    plt.suptitle(f'Kernel Comparison on {dataset_name} Dataset', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def demonstrate_kernel_perceptron():
    """
    Comprehensive demonstration of kernel perceptron on various datasets.
    """
    # Create plots directory
    import os
    os.makedirs("plots", exist_ok=True)
    
    print("Kernel Perceptron Demonstration")
    print("="*50)
    
    # Test datasets
    datasets = {
        "XOR": make_xor_data(n_samples=200, noise=0.1, random_state=42),
        "Concentric Circles": make_concentric_circles_data(n_samples=200, noise=0.1, random_state=42),
        "Moons": make_moons_data(n_samples=200, noise=0.1, random_state=42),
        "Linearly Separable": make_linearly_separable_data(n_samples=100, random_state=42)
    }
    
    all_results = {}
    
    for dataset_name, (X, y) in datasets.items():
        # Compare kernels on this dataset
        results = compare_kernels_on_dataset(X, y, dataset_name)
        all_results[dataset_name] = results
        
        # Visualize comparison
        visualize_kernel_comparison(X, y, results, dataset_name, 
                                   f"plots/kernel_comparison_{dataset_name.lower().replace(' ', '_')}.png")
    
    # Create summary comparison
    create_kernel_summary(all_results)
    
    return all_results


def create_kernel_summary(all_results):
    """
    Create a summary comparison of all kernel results.
    
    Args:
        all_results (dict): Results from all datasets and kernels
    """
    # Collect data for summary
    summary_data = []
    
    for dataset_name, dataset_results in all_results.items():
        for kernel_name, result in dataset_results.items():
            summary_data.append({
                'Dataset': dataset_name,
                'Kernel': kernel_name,
                'Accuracy': result['accuracy'],
                'Converged': result['converged'],
                'Support Vector Ratio': result['support_vector_ratio'],
                'Final Errors': result['final_errors']
            })
    
    # Create summary plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy heatmap
    datasets = list(all_results.keys())
    kernels = list(next(iter(all_results.values())).keys())
    
    accuracy_matrix = np.zeros((len(datasets), len(kernels)))
    
    for i, dataset in enumerate(datasets):
        for j, kernel in enumerate(kernels):
            accuracy_matrix[i, j] = all_results[dataset][kernel]['accuracy']
    
    im1 = axes[0, 0].imshow(accuracy_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    axes[0, 0].set_xticks(range(len(kernels)))
    axes[0, 0].set_xticklabels(kernels, rotation=45, ha='right')
    axes[0, 0].set_yticks(range(len(datasets)))
    axes[0, 0].set_yticklabels(datasets)
    axes[0, 0].set_title('Accuracy Heatmap')
    
    # Add text annotations
    for i in range(len(datasets)):
        for j in range(len(kernels)):
            axes[0, 0].text(j, i, f'{accuracy_matrix[i, j]:.2f}', 
                           ha='center', va='center', color='black')
    
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Support vector ratio comparison
    axes[0, 1].set_title('Support Vector Ratios')
    for dataset in datasets:
        sv_ratios = [all_results[dataset][kernel]['support_vector_ratio'] 
                    for kernel in kernels]
        axes[0, 1].plot(kernels, sv_ratios, marker='o', label=dataset, linewidth=2)
    
    axes[0, 1].set_ylabel('Support Vector Ratio')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Convergence analysis
    convergence_data = {}
    for dataset in datasets:
        convergence_data[dataset] = sum(1 for kernel in kernels 
                                       if all_results[dataset][kernel]['converged'])
    
    axes[1, 0].bar(convergence_data.keys(), convergence_data.values(), alpha=0.7)
    axes[1, 0].set_ylabel('Number of Converged Kernels')
    axes[1, 0].set_title('Convergence by Dataset')
    axes[1, 0].grid(True, alpha=0.3)
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Best kernel for each dataset
    best_kernels = {}
    for dataset in datasets:
        best_kernel = max(all_results[dataset].keys(), 
                         key=lambda k: all_results[dataset][k]['accuracy'])
        best_kernels[dataset] = best_kernel
    
    # Count kernel preferences
    kernel_counts = {}
    for kernel in kernels:
        kernel_counts[kernel] = sum(1 for best in best_kernels.values() if best == kernel)
    
    axes[1, 1].pie(kernel_counts.values(), labels=kernel_counts.keys(), autopct='%1.0f%%')
    axes[1, 1].set_title('Best Performing Kernel Distribution')
    
    plt.tight_layout()
    plt.savefig("plots/kernel_summary_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary table
    print(f"\n{'='*80}")
    print("KERNEL PERCEPTRON SUMMARY")
    print(f"{'='*80}")
    
    print(f"{'Dataset':<20} {'Best Kernel':<25} {'Accuracy':<10} {'SV Ratio':<10}")
    print("-" * 80)
    
    for dataset in datasets:
        best_kernel = max(all_results[dataset].keys(), 
                         key=lambda k: all_results[dataset][k]['accuracy'])
        best_result = all_results[dataset][best_kernel]
        
        print(f"{dataset:<20} {best_kernel:<25} {best_result['accuracy']:<10.3f} "
              f"{best_result['support_vector_ratio']:<10.1%}")


if __name__ == "__main__":
    # Run comprehensive demonstration
    results = demonstrate_kernel_perceptron()
    
    print("\nKernel perceptron implementation complete!") 