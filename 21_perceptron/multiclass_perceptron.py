"""
Multi-class Perceptron Implementation

This module implements multi-class perceptron algorithms using different strategies:
1. One-vs-Rest (OvR): Train one binary classifier per class
2. One-vs-One (OvO): Train one binary classifier per pair of classes
3. Multi-class Perceptron: Direct extension with multiple weight vectors

Each approach has different characteristics and trade-offs.
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from perceptron import Perceptron
from synthetic_data import make_multiclass_data
from sklearn.datasets import make_blobs


class OneVsRestPerceptron:
    """
    One-vs-Rest Multi-class Perceptron
    
    Trains one binary perceptron for each class vs all other classes.
    Prediction is made by choosing the class with highest confidence.
    """
    
    def __init__(self, learning_rate=1.0, max_epochs=1000, random_state=42):
        """
        Initialize One-vs-Rest perceptron.
        
        Args:
            learning_rate (float): Learning rate for binary perceptrons
            max_epochs (int): Maximum epochs for each binary classifier
            random_state (int): Random seed
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.random_state = random_state
        
        # Will be set during training
        self.classes_ = None
        self.classifiers = {}
        self.n_classes = 0
    
    def fit(self, X, y, verbose=True):
        """
        Train one binary perceptron for each class.
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training labels
            verbose (bool): Print training progress
            
        Returns:
            self: Returns self for method chaining
        """
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        
        if verbose:
            print(f"Training One-vs-Rest perceptron with {self.n_classes} classes")
            print(f"Classes: {self.classes_}")
        
        # Train one binary classifier for each class
        for class_label in self.classes_:
            if verbose:
                print(f"\nTraining classifier for class {class_label} vs rest...")
            
            # Create binary labels: +1 for current class, -1 for all others
            y_binary = np.where(y == class_label, 1, -1)
            
            # Train binary perceptron
            classifier = Perceptron(
                learning_rate=self.learning_rate,
                max_epochs=self.max_epochs,
                random_state=self.random_state
            )
            classifier.fit(X, y_binary, verbose=verbose)
            
            self.classifiers[class_label] = classifier
            
            if verbose:
                accuracy = classifier.score(X, y_binary)
                print(f"Binary accuracy for class {class_label}: {accuracy:.3f}")
        
        return self
    
    def decision_function(self, X):
        """
        Compute decision function values for all classes.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Decision values (n_samples, n_classes)
        """
        if self.classes_ is None:
            raise ValueError("Model must be trained first")
        
        n_samples = X.shape[0]
        decision_values = np.zeros((n_samples, self.n_classes))
        
        for i, class_label in enumerate(self.classes_):
            classifier = self.classifiers[class_label]
            
            # Get decision function values (w·x + b)
            X_with_bias = np.c_[np.ones(X.shape[0]), X]
            decision_values[:, i] = np.dot(X_with_bias, classifier.weights)
        
        return decision_values
    
    def predict(self, X):
        """
        Predict class labels for samples.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predicted class labels
        """
        decision_values = self.decision_function(X)
        
        # Choose class with highest decision value
        class_indices = np.argmax(decision_values, axis=1)
        
        return self.classes_[class_indices]
    
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


class OneVsOnePerceptron:
    """
    One-vs-One Multi-class Perceptron
    
    Trains one binary perceptron for each pair of classes.
    Prediction is made by majority voting among all binary classifiers.
    """
    
    def __init__(self, learning_rate=1.0, max_epochs=1000, random_state=42):
        """
        Initialize One-vs-One perceptron.
        
        Args:
            learning_rate (float): Learning rate for binary perceptrons
            max_epochs (int): Maximum epochs for each binary classifier
            random_state (int): Random seed
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.random_state = random_state
        
        # Will be set during training
        self.classes_ = None
        self.classifiers = {}
        self.n_classes = 0
        self.class_pairs = []
    
    def fit(self, X, y, verbose=True):
        """
        Train one binary perceptron for each pair of classes.
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training labels
            verbose (bool): Print training progress
            
        Returns:
            self: Returns self for method chaining
        """
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        self.class_pairs = list(combinations(self.classes_, 2))
        
        if verbose:
            print(f"Training One-vs-One perceptron with {self.n_classes} classes")
            print(f"Classes: {self.classes_}")
            print(f"Number of binary classifiers: {len(self.class_pairs)}")
        
        # Train one binary classifier for each pair of classes
        for class_pair in self.class_pairs:
            class_a, class_b = class_pair
            
            if verbose:
                print(f"\nTraining classifier for {class_a} vs {class_b}...")
            
            # Filter data to only include samples from these two classes
            mask = (y == class_a) | (y == class_b)
            X_pair = X[mask]
            y_pair = y[mask]
            
            # Create binary labels: +1 for class_a, -1 for class_b
            y_binary = np.where(y_pair == class_a, 1, -1)
            
            # Train binary perceptron
            classifier = Perceptron(
                learning_rate=self.learning_rate,
                max_epochs=self.max_epochs,
                random_state=self.random_state
            )
            classifier.fit(X_pair, y_binary, verbose=False)
            
            self.classifiers[class_pair] = classifier
            
            if verbose:
                accuracy = classifier.score(X_pair, y_binary)
                print(f"Binary accuracy for {class_a} vs {class_b}: {accuracy:.3f}")
        
        return self
    
    def predict(self, X):
        """
        Predict class labels using majority voting.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predicted class labels
        """
        if self.classes_ is None:
            raise ValueError("Model must be trained first")
        
        n_samples = X.shape[0]
        votes = np.zeros((n_samples, self.n_classes))
        
        # Get predictions from each binary classifier
        for class_pair in self.class_pairs:
            class_a, class_b = class_pair
            classifier = self.classifiers[class_pair]
            
            # Get binary predictions
            binary_predictions = classifier.predict(X)
            
            # Convert to votes
            class_a_idx = np.where(self.classes_ == class_a)[0][0]
            class_b_idx = np.where(self.classes_ == class_b)[0][0]
            
            # Add votes based on binary predictions
            votes[binary_predictions == 1, class_a_idx] += 1
            votes[binary_predictions == -1, class_b_idx] += 1
        
        # Choose class with most votes
        class_indices = np.argmax(votes, axis=1)
        
        return self.classes_[class_indices]
    
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


class MultiClassPerceptron:
    """
    Direct Multi-class Perceptron
    
    Maintains separate weight vectors for each class and updates
    them using the multi-class perceptron learning rule.
    """
    
    def __init__(self, learning_rate=1.0, max_epochs=1000, random_state=42):
        """
        Initialize multi-class perceptron.
        
        Args:
            learning_rate (float): Learning rate
            max_epochs (int): Maximum epochs
            random_state (int): Random seed
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.random_state = random_state
        
        # Will be set during training
        self.classes_ = None
        self.weights = None  # Will be (n_classes, n_features + 1) for bias
        self.n_classes = 0
        self.history = {
            'errors': [],
            'converged_epoch': None
        }
    
    def _add_bias_term(self, X):
        """Add bias term to input matrix."""
        return np.c_[np.ones(X.shape[0]), X]
    
    def _initialize_weights(self, n_features):
        """Initialize weight matrix."""
        np.random.seed(self.random_state)
        self.weights = np.random.normal(0, 0.01, (self.n_classes, n_features))
    
    def _predict_sample(self, x):
        """
        Predict class for a single sample.
        
        Args:
            x (np.ndarray): Single sample with bias term
            
        Returns:
            int: Predicted class index
        """
        # Compute activations for all classes
        activations = np.dot(self.weights, x)
        
        # Return class with highest activation
        return np.argmax(activations)
    
    def predict(self, X):
        """
        Predict classes for multiple samples.
        
        Args:
            X (np.ndarray): Input features
            
        Returns:
            np.ndarray: Predicted class labels
        """
        if self.weights is None:
            raise ValueError("Model must be trained first")
        
        X_with_bias = self._add_bias_term(X)
        predictions = []
        
        for x in X_with_bias:
            class_idx = self._predict_sample(x)
            predictions.append(self.classes_[class_idx])
        
        return np.array(predictions)
    
    def fit(self, X, y, verbose=True):
        """
        Train the multi-class perceptron.
        
        Args:
            X (np.ndarray): Training features
            y (np.ndarray): Training labels
            verbose (bool): Print training progress
            
        Returns:
            self: Returns self for method chaining
        """
        self.classes_ = np.unique(y)
        self.n_classes = len(self.classes_)
        
        # Create class index mapping
        class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}
        y_indexed = np.array([class_to_idx[cls] for cls in y])
        
        # Add bias term
        X_with_bias = self._add_bias_term(X)
        n_samples, n_features = X_with_bias.shape
        
        # Initialize weights
        self._initialize_weights(n_features)
        
        # Reset history
        self.history = {
            'errors': [],
            'converged_epoch': None
        }
        
        if verbose:
            print(f"Training multi-class perceptron with {n_samples} samples, "
                  f"{n_features-1} features, {self.n_classes} classes")
            print(f"Classes: {self.classes_}")
        
        # Training loop
        for epoch in range(self.max_epochs):
            errors = 0
            
            # Go through each training sample
            for i in range(n_samples):
                x_i = X_with_bias[i]
                true_class_idx = y_indexed[i]
                
                # Make prediction
                predicted_class_idx = self._predict_sample(x_i)
                
                # Check if misclassified
                if predicted_class_idx != true_class_idx:
                    # Multi-class perceptron update rule:
                    # w_true ← w_true + η * x_i
                    # w_predicted ← w_predicted - η * x_i
                    self.weights[true_class_idx] += self.learning_rate * x_i
                    self.weights[predicted_class_idx] -= self.learning_rate * x_i
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


def compare_multiclass_strategies(X, y, strategies=None):
    """
    Compare different multi-class strategies on a dataset.
    
    Args:
        X (np.ndarray): Training data
        y (np.ndarray): Training labels
        strategies (list): List of strategy names to test
        
    Returns:
        dict: Results for each strategy
    """
    if strategies is None:
        strategies = ['OneVsRest', 'OneVsOne', 'MultiClass']
    
    results = {}
    
    print(f"Comparing multi-class strategies on dataset with {len(np.unique(y))} classes")
    print(f"Dataset shape: {X.shape}")
    
    for strategy in strategies:
        print(f"\n{'='*50}")
        print(f"Testing {strategy} Strategy")
        print(f"{'='*50}")
        
        # Initialize classifier based on strategy
        if strategy == 'OneVsRest':
            classifier = OneVsRestPerceptron(learning_rate=1.0, max_epochs=1000, random_state=42)
        elif strategy == 'OneVsOne':
            classifier = OneVsOnePerceptron(learning_rate=1.0, max_epochs=1000, random_state=42)
        elif strategy == 'MultiClass':
            classifier = MultiClassPerceptron(learning_rate=1.0, max_epochs=1000, random_state=42)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Train classifier
        classifier.fit(X, y, verbose=True)
        
        # Evaluate
        accuracy = classifier.score(X, y)
        
        # Get additional metrics
        if hasattr(classifier, 'history'):
            converged = classifier.history['converged_epoch'] is not None
            epochs = classifier.history['converged_epoch'] if converged else 1000
            final_errors = classifier.history['errors'][-1] if classifier.history['errors'] else 0
        else:
            converged = None
            epochs = None
            final_errors = None
        
        results[strategy] = {
            'classifier': classifier,
            'accuracy': accuracy,
            'converged': converged,
            'epochs': epochs,
            'final_errors': final_errors
        }
        
        print(f"Final accuracy: {accuracy:.3f}")
        if converged is not None:
            print(f"Converged: {converged}")
            if converged:
                print(f"Epochs to convergence: {epochs}")
    
    return results


def plot_multiclass_comparison(X, y, results, dataset_name="Multi-class Dataset", save_path=None):
    """
    Visualize comparison of multi-class strategies.
    
    Args:
        X (np.ndarray): Training data (2D only)
        y (np.ndarray): Training labels
        results (dict): Results from compare_multiclass_strategies
        dataset_name (str): Name of dataset
        save_path (str): Path to save plot
    """
    if X.shape[1] != 2:
        print("Visualization only available for 2D data")
        return
    
    n_strategies = len(results)
    fig, axes = plt.subplots(2, n_strategies, figsize=(5*n_strategies, 10))
    
    if n_strategies == 1:
        axes = axes.reshape(-1, 1)
    
    classes = np.unique(y)
    colors = plt.cm.Set1(np.linspace(0, 1, len(classes)))
    
    for col, (strategy, result) in enumerate(results.items()):
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
        
        # Convert class labels to numerical for plotting
        class_to_num = {cls: i for i, cls in enumerate(classes)}
        Z_num = np.array([class_to_num[cls] for cls in Z])
        Z_num = Z_num.reshape(xx.shape)
        
        # Plot decision regions
        ax1.contourf(xx, yy, Z_num, alpha=0.4, cmap=plt.cm.Set1, levels=len(classes)-1)
        
        # Plot training points
        for i, cls in enumerate(classes):
            mask = y == cls
            ax1.scatter(X[mask, 0], X[mask, 1], c=[colors[i]], 
                       label=f'Class {cls}', s=50, alpha=0.8, edgecolors='black')
        
        ax1.set_xlim(xx.min(), xx.max())
        ax1.set_ylim(yy.min(), yy.max())
        ax1.set_title(f'{strategy}\nAccuracy: {accuracy:.3f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot convergence (if available)
        ax2 = axes[1, col]
        
        if hasattr(classifier, 'history') and classifier.history['errors']:
            ax2.plot(classifier.history['errors'], 'b-', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Number of Errors')
            ax2.set_title(f'{strategy} Convergence')
            ax2.grid(True, alpha=0.3)
            
            if classifier.history['converged_epoch'] is not None:
                ax2.axvline(x=classifier.history['converged_epoch'], 
                           color='r', linestyle='--', alpha=0.7,
                           label=f'Converged at {classifier.history["converged_epoch"]}')
                ax2.legend()
        else:
            # For strategies without convergence history, show accuracy bar
            ax2.bar([strategy], [accuracy], color='skyblue', alpha=0.7)
            ax2.set_ylabel('Accuracy')
            ax2.set_title(f'{strategy} Performance')
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Multi-class Strategy Comparison: {dataset_name}', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def demonstrate_multiclass_perceptron():
    """
    Comprehensive demonstration of multi-class perceptron strategies.
    """
    # Create plots directory
    import os
    os.makedirs("plots", exist_ok=True)
    
    print("Multi-class Perceptron Demonstration")
    print("="*50)
    
    # Test datasets
    datasets = {
        "3-Class Blobs": make_multiclass_data(n_samples=300, n_classes=3, random_state=42),
        "4-Class Blobs": make_multiclass_data(n_samples=400, n_classes=4, random_state=42),
        "Overlapping Classes": make_blobs(n_samples=300, centers=3, cluster_std=2.0, 
                                        center_box=(-5, 5), random_state=42)
    }
    
    all_results = {}
    
    for dataset_name, (X, y) in datasets.items():
        print(f"\n{'='*60}")
        print(f"Testing on {dataset_name}")
        print(f"{'='*60}")
        
        # Compare strategies
        results = compare_multiclass_strategies(X, y)
        all_results[dataset_name] = results
        
        # Visualize comparison
        plot_multiclass_comparison(X, y, results, dataset_name, 
                                  f"plots/multiclass_comparison_{dataset_name.lower().replace(' ', '_').replace('-', '_')}.png")
    
    # Create summary analysis
    create_multiclass_summary(all_results)
    
    return all_results


def create_multiclass_summary(all_results):
    """
    Create summary analysis of multi-class strategy performance.
    
    Args:
        all_results (dict): Results from all datasets and strategies
    """
    # Collect summary data
    strategies = list(next(iter(all_results.values())).keys())
    datasets = list(all_results.keys())
    
    # Create summary plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison
    accuracy_data = {}
    for strategy in strategies:
        accuracy_data[strategy] = [all_results[dataset][strategy]['accuracy'] 
                                  for dataset in datasets]
    
    x_pos = np.arange(len(datasets))
    width = 0.25
    
    for i, (strategy, accuracies) in enumerate(accuracy_data.items()):
        axes[0, 0].bar(x_pos + i*width, accuracies, width, label=strategy, alpha=0.8)
    
    axes[0, 0].set_xlabel('Dataset')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy Comparison Across Datasets')
    axes[0, 0].set_xticks(x_pos + width)
    axes[0, 0].set_xticklabels(datasets, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Convergence analysis (only for strategies with history)
    convergence_strategies = []
    convergence_data = {}
    
    for strategy in strategies:
        # Check if any result has convergence info
        has_convergence = any(all_results[dataset][strategy]['converged'] is not None 
                             for dataset in datasets)
        if has_convergence:
            convergence_strategies.append(strategy)
            convergence_data[strategy] = []
            
            for dataset in datasets:
                result = all_results[dataset][strategy]
                if result['converged']:
                    convergence_data[strategy].append(result['epochs'])
                else:
                    convergence_data[strategy].append(1000)  # Max epochs
    
    if convergence_data:
        x_pos = np.arange(len(datasets))
        width = 0.35
        
        for i, (strategy, epochs) in enumerate(convergence_data.items()):
            axes[0, 1].bar(x_pos + i*width, epochs, width, label=strategy, alpha=0.8)
        
        axes[0, 1].set_xlabel('Dataset')
        axes[0, 1].set_ylabel('Epochs to Convergence')
        axes[0, 1].set_title('Convergence Speed Comparison')
        axes[0, 1].set_xticks(x_pos + width/2)
        axes[0, 1].set_xticklabels(datasets, rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Strategy preference across datasets
    best_strategies = {}
    for dataset in datasets:
        best_strategy = max(strategies, key=lambda s: all_results[dataset][s]['accuracy'])
        best_strategies[dataset] = best_strategy
    
    strategy_counts = {strategy: 0 for strategy in strategies}
    for best in best_strategies.values():
        strategy_counts[best] += 1
    
    axes[1, 0].pie(strategy_counts.values(), labels=strategy_counts.keys(), 
                   autopct='%1.0f%%', startangle=90)
    axes[1, 0].set_title('Best Strategy Distribution')
    
    # Average accuracy by strategy
    avg_accuracies = {}
    for strategy in strategies:
        accuracies = [all_results[dataset][strategy]['accuracy'] for dataset in datasets]
        avg_accuracies[strategy] = np.mean(accuracies)
    
    axes[1, 1].bar(avg_accuracies.keys(), avg_accuracies.values(), alpha=0.8, color='lightcoral')
    axes[1, 1].set_ylabel('Average Accuracy')
    axes[1, 1].set_title('Average Accuracy Across All Datasets')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add values on bars
    for strategy, avg_acc in avg_accuracies.items():
        axes[1, 1].text(strategy, avg_acc + 0.01, f'{avg_acc:.3f}', 
                       ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("plots/multiclass_summary_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary table
    print(f"\n{'='*80}")
    print("MULTI-CLASS STRATEGY SUMMARY")
    print(f"{'='*80}")
    
    print(f"{'Dataset':<20} {'Best Strategy':<15} {'Accuracy':<10} {'OvR Acc':<10} {'OvO Acc':<10} {'MC Acc':<10}")
    print("-" * 85)
    
    for dataset in datasets:
        best_strategy = max(strategies, key=lambda s: all_results[dataset][s]['accuracy'])
        best_acc = all_results[dataset][best_strategy]['accuracy']
        
        ovr_acc = all_results[dataset]['OneVsRest']['accuracy']
        ovo_acc = all_results[dataset]['OneVsOne']['accuracy']
        mc_acc = all_results[dataset]['MultiClass']['accuracy']
        
        print(f"{dataset:<20} {best_strategy:<15} {best_acc:<10.3f} "
              f"{ovr_acc:<10.3f} {ovo_acc:<10.3f} {mc_acc:<10.3f}")


if __name__ == "__main__":
    # Run comprehensive demonstration
    results = demonstrate_multiclass_perceptron()
    
    print("\nMulti-class perceptron implementation complete!") 