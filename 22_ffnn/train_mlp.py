"""
Training Experiments for Multi-Layer Perceptron

This module provides comprehensive experiments to analyze MLP performance
on various datasets, including decision boundary visualization and
comparison of different activation functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time

from mlp_numpy import FeedforwardNeuralNet
from datasets import (make_xor_dataset, make_spiral_dataset, make_moons_dataset, 
                     make_circles_dataset, make_blobs_dataset, create_train_test_split,
                     normalize_features)


def plot_decision_boundary(model, X, y, title="Decision Boundary", save_path=None, h=0.01):
    """
    Plot 2D decision boundary for a trained model.
    
    Args:
        model: Trained model with predict_proba method
        X (np.ndarray): 2D training data
        y (np.ndarray): Training labels
        title (str): Plot title
        save_path (str): Path to save plot
        h (float): Step size for meshgrid
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
    Z = model.predict_proba(mesh_points)
    
    # Handle different output formats
    if Z.shape[1] == 1:
        # Binary classification
        Z = Z.ravel()
        Z = (Z > 0.5).astype(int)
    else:
        # Multi-class classification
        Z = np.argmax(Z, axis=1)
    
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=cmap_light)
    
    # Plot the training points
    if y.min() == -1:
        # Convert {-1, 1} to {0, 1} for plotting
        y_plot = (y + 1) // 2
    else:
        y_plot = y
    
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y_plot.ravel(), cmap=cmap_bold, 
                         edgecolors='black', s=60)
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.colorbar(scatter)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def compare_activation_functions(X, y, dataset_name="Dataset"):
    """
    Compare different activation functions on a dataset.
    
    Args:
        X (np.ndarray): Features
        y (np.ndarray): Labels
        dataset_name (str): Name of the dataset
        
    Returns:
        dict: Results for each activation function
    """
    print(f"\nComparing Activation Functions on {dataset_name}")
    print("-" * 60)
    
    # Split data
    X_train, X_test, y_train, y_test = create_train_test_split(X, y, test_size=0.2)
    
    # Normalize features
    X_train_norm, X_test_norm, _, _ = normalize_features(X_train, X_test)
    
    # Activation functions to test
    activations_to_test = [
        ('ReLU', ['relu', 'relu']),
        ('Sigmoid', ['sigmoid', 'sigmoid']),
        ('Tanh', ['tanh', 'tanh']),
        ('Mixed', ['relu', 'tanh'])
    ]
    
    results = {}
    
    for name, activations in activations_to_test:
        print(f"\nTesting {name} activation...")
        
        # Create model
        mlp = FeedforwardNeuralNet(
            layers=[2, 10, 10, 1],
            activations=activations,
            output_activation='sigmoid',
            weight_init='xavier',
            random_state=42
        )
        
        # Train model
        start_time = time.time()
        mlp.fit(X_train_norm, y_train, epochs=300, learning_rate=0.01, 
                validation_data=(X_test_norm, y_test), verbose=False)
        training_time = time.time() - start_time
        
        # Evaluate
        train_results = mlp.evaluate(X_train_norm, y_train)
        test_results = mlp.evaluate(X_test_norm, y_test)
        
        results[name] = {
            'model': mlp,
            'train_accuracy': train_results['accuracy'],
            'test_accuracy': test_results['accuracy'],
            'train_loss': train_results['loss'],
            'test_loss': test_results['loss'],
            'training_time': training_time,
            'final_train_loss': mlp.history['loss'][-1],
            'final_val_loss': mlp.history['val_loss'][-1] if mlp.history['val_loss'] else None
        }
        
        print(f"  Train accuracy: {train_results['accuracy']:.4f}")
        print(f"  Test accuracy: {test_results['accuracy']:.4f}")
        print(f"  Training time: {training_time:.2f}s")
    
    return results, X_train_norm, X_test_norm, y_train, y_test


def plot_activation_comparison(results, X_train, y_train, dataset_name):
    """
    Plot comparison of different activation functions.
    
    Args:
        results (dict): Results from compare_activation_functions
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        dataset_name (str): Name of the dataset
    """
    n_activations = len(results)
    fig, axes = plt.subplots(2, n_activations, figsize=(5*n_activations, 10))
    
    if n_activations == 1:
        axes = axes.reshape(-1, 1)
    
    for col, (activation_name, result) in enumerate(results.items()):
        model = result['model']
        
        # Plot decision boundary
        ax1 = axes[0, col]
        
        # Create mesh for decision boundary
        h = 0.02
        x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
        y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = model.predict_proba(mesh_points)
        Z = Z.ravel()
        Z = Z.reshape(xx.shape)
        
        # Plot decision regions
        ax1.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap=plt.cm.RdYlBu)
        
        # Plot training points
        y_plot = (y_train + 1) // 2 if y_train.min() == -1 else y_train
        colors = ['red', 'blue']
        for i, color in enumerate(colors):
            mask = y_plot.ravel() == i
            label = f'Class {-1 if i == 0 and y_train.min() == -1 else i}'
            ax1.scatter(X_train[mask, 0], X_train[mask, 1], c=color, 
                       label=label, alpha=0.7, s=30, edgecolors='black')
        
        ax1.set_xlim(xx.min(), xx.max())
        ax1.set_ylim(yy.min(), yy.max())
        ax1.set_title(f'{activation_name}\nAcc: {result["test_accuracy"]:.3f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot training history
        ax2 = axes[1, col]
        ax2.plot(result['model'].history['loss'], 'b-', label='Training Loss', linewidth=2)
        if result['model'].history['val_loss']:
            ax2.plot(result['model'].history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title(f'{activation_name} Training')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
    
    plt.suptitle(f'Activation Function Comparison: {dataset_name}', fontsize=16)
    plt.tight_layout()
    
    save_path = f"plots/activation_comparison_{dataset_name.lower().replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def analyze_network_depth(X, y, dataset_name="Dataset"):
    """
    Analyze the effect of network depth on performance.
    
    Args:
        X (np.ndarray): Features
        y (np.ndarray): Labels
        dataset_name (str): Name of the dataset
        
    Returns:
        dict: Results for different network depths
    """
    print(f"\nAnalyzing Network Depth on {dataset_name}")
    print("-" * 50)
    
    # Split and normalize data
    X_train, X_test, y_train, y_test = create_train_test_split(X, y, test_size=0.2)
    X_train_norm, X_test_norm, _, _ = normalize_features(X_train, X_test)
    
    # Different network architectures
    architectures = [
        ("Shallow (1 hidden)", [2, 20, 1]),
        ("Medium (2 hidden)", [2, 15, 15, 1]),
        ("Deep (3 hidden)", [2, 10, 10, 10, 1]),
        ("Very Deep (4 hidden)", [2, 8, 8, 8, 8, 1])
    ]
    
    results = {}
    
    for name, layers in architectures:
        print(f"\nTesting {name}: {layers}")
        
        # Create activations list
        activations = ['relu'] * (len(layers) - 2)
        
        # Create model
        mlp = FeedforwardNeuralNet(
            layers=layers,
            activations=activations,
            output_activation='sigmoid',
            weight_init='xavier',
            random_state=42
        )
        
        # Train model
        start_time = time.time()
        mlp.fit(X_train_norm, y_train, epochs=400, learning_rate=0.01, 
                validation_data=(X_test_norm, y_test), verbose=False)
        training_time = time.time() - start_time
        
        # Evaluate
        train_results = mlp.evaluate(X_train_norm, y_train)
        test_results = mlp.evaluate(X_test_norm, y_test)
        
        # Get model summary
        summary = mlp.get_model_summary()
        
        results[name] = {
            'model': mlp,
            'layers': layers,
            'parameters': summary['total_parameters'],
            'train_accuracy': train_results['accuracy'],
            'test_accuracy': test_results['accuracy'],
            'training_time': training_time,
            'convergence_epoch': len(mlp.history['loss'])
        }
        
        print(f"  Parameters: {summary['total_parameters']}")
        print(f"  Train accuracy: {train_results['accuracy']:.4f}")
        print(f"  Test accuracy: {test_results['accuracy']:.4f}")
        print(f"  Training time: {training_time:.2f}s")
    
    return results


def plot_depth_analysis(results, dataset_name):
    """
    Plot analysis of network depth effects.
    
    Args:
        results (dict): Results from analyze_network_depth
        dataset_name (str): Name of the dataset
    """
    names = list(results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy vs Depth
    train_accs = [results[name]['train_accuracy'] for name in names]
    test_accs = [results[name]['test_accuracy'] for name in names]
    
    x_pos = np.arange(len(names))
    width = 0.35
    
    axes[0, 0].bar(x_pos - width/2, train_accs, width, label='Training', alpha=0.8)
    axes[0, 0].bar(x_pos + width/2, test_accs, width, label='Testing', alpha=0.8)
    axes[0, 0].set_xlabel('Network Architecture')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy vs Network Depth')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Parameters vs Performance
    params = [results[name]['parameters'] for name in names]
    axes[0, 1].scatter(params, test_accs, s=100, alpha=0.7)
    for i, name in enumerate(names):
        axes[0, 1].annotate(name.split(' ')[0], (params[i], test_accs[i]), 
                           xytext=(5, 5), textcoords='offset points')
    axes[0, 1].set_xlabel('Number of Parameters')
    axes[0, 1].set_ylabel('Test Accuracy')
    axes[0, 1].set_title('Test Accuracy vs Model Complexity')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Training Time vs Depth
    training_times = [results[name]['training_time'] for name in names]
    axes[1, 0].bar(names, training_times, alpha=0.8, color='orange')
    axes[1, 0].set_xlabel('Network Architecture')
    axes[1, 0].set_ylabel('Training Time (seconds)')
    axes[1, 0].set_title('Training Time vs Network Depth')
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss curves for all models
    for name in names:
        model = results[name]['model']
        axes[1, 1].plot(model.history['loss'], label=f'{name} (Train)', linewidth=2)
        if model.history['val_loss']:
            axes[1, 1].plot(model.history['val_loss'], '--', 
                           label=f'{name} (Val)', linewidth=2)
    
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Training Curves Comparison')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.suptitle(f'Network Depth Analysis: {dataset_name}', fontsize=16)
    plt.tight_layout()
    
    save_path = f"plots/depth_analysis_{dataset_name.lower().replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def comprehensive_mlp_experiments():
    """
    Run comprehensive experiments on multiple datasets.
    """
    # Create plots directory
    import os
    os.makedirs("plots", exist_ok=True)
    
    print("Comprehensive MLP Experiments")
    print("=" * 60)
    
    # Test datasets
    datasets = {
        "XOR": make_xor_dataset(n_samples=800, noise=0.05, random_state=42),
        "Spiral": make_spiral_dataset(n_samples=800, noise=0.1, random_state=42),
        "Moons": make_moons_dataset(n_samples=800, noise=0.1, random_state=42),
        "Circles": make_circles_dataset(n_samples=800, noise=0.1, random_state=42)
    }
    
    all_results = {}
    
    # Test each dataset
    for dataset_name, (X, y) in datasets.items():
        print(f"\n{'='*60}")
        print(f"Testing on {dataset_name} Dataset")
        print(f"{'='*60}")
        
        # 1. Compare activation functions
        activation_results, X_train, X_test, y_train, y_test = compare_activation_functions(X, y, dataset_name)
        plot_activation_comparison(activation_results, X_train, y_train, dataset_name)
        
        # 2. Analyze network depth
        depth_results = analyze_network_depth(X, y, dataset_name)
        plot_depth_analysis(depth_results, dataset_name)
        
        all_results[dataset_name] = {
            'activation_results': activation_results,
            'depth_results': depth_results
        }
    
    # Create summary analysis
    create_summary_analysis(all_results)
    
    return all_results


def create_summary_analysis(all_results):
    """
    Create a summary analysis across all datasets and experiments.
    
    Args:
        all_results (dict): Results from all experiments
    """
    print(f"\n{'='*60}")
    print("SUMMARY ANALYSIS")
    print(f"{'='*60}")
    
    # Best activation function for each dataset
    print("\nBest Activation Function by Dataset:")
    print("-" * 50)
    
    activation_winners = {}
    for dataset_name, results in all_results.items():
        activation_results = results['activation_results']
        best_activation = max(activation_results.keys(), 
                            key=lambda k: activation_results[k]['test_accuracy'])
        best_accuracy = activation_results[best_activation]['test_accuracy']
        
        activation_winners[dataset_name] = best_activation
        print(f"{dataset_name:<12}: {best_activation:<10} (Acc: {best_accuracy:.3f})")
    
    # Best depth for each dataset
    print("\nBest Network Depth by Dataset:")
    print("-" * 50)
    
    depth_winners = {}
    for dataset_name, results in all_results.items():
        depth_results = results['depth_results']
        best_depth = max(depth_results.keys(), 
                        key=lambda k: depth_results[k]['test_accuracy'])
        best_accuracy = depth_results[best_depth]['test_accuracy']
        
        depth_winners[dataset_name] = best_depth
        print(f"{dataset_name:<12}: {best_depth:<20} (Acc: {best_accuracy:.3f})")
    
    # Create summary plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Activation function preference
    activation_counts = {}
    for winner in activation_winners.values():
        activation_counts[winner] = activation_counts.get(winner, 0) + 1
    
    axes[0, 0].pie(activation_counts.values(), labels=activation_counts.keys(), 
                   autopct='%1.0f%%', startangle=90)
    axes[0, 0].set_title('Best Activation Function Distribution')
    
    # Depth preference
    depth_counts = {}
    for winner in depth_winners.values():
        depth_key = winner.split(' ')[0]  # Extract depth level
        depth_counts[depth_key] = depth_counts.get(depth_key, 0) + 1
    
    axes[0, 1].pie(depth_counts.values(), labels=depth_counts.keys(), 
                   autopct='%1.0f%%', startangle=90)
    axes[0, 1].set_title('Best Network Depth Distribution')
    
    # Performance comparison across datasets
    datasets = list(all_results.keys())
    
    # Get best accuracies for each dataset
    best_accuracies = []
    for dataset_name in datasets:
        activation_results = all_results[dataset_name]['activation_results']
        best_acc = max(result['test_accuracy'] for result in activation_results.values())
        best_accuracies.append(best_acc)
    
    bars = axes[1, 0].bar(datasets, best_accuracies, alpha=0.8, color='skyblue')
    axes[1, 0].set_ylabel('Best Test Accuracy')
    axes[1, 0].set_title('Best Performance by Dataset')
    axes[1, 0].set_ylim(0, 1.1)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add values on bars
    for bar, acc in zip(bars, best_accuracies):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                       f'{acc:.3f}', ha='center', va='bottom')
    
    # Training time comparison
    avg_times = []
    for dataset_name in datasets:
        activation_results = all_results[dataset_name]['activation_results']
        avg_time = np.mean([result['training_time'] for result in activation_results.values()])
        avg_times.append(avg_time)
    
    axes[1, 1].bar(datasets, avg_times, alpha=0.8, color='lightcoral')
    axes[1, 1].set_ylabel('Average Training Time (s)')
    axes[1, 1].set_title('Training Time by Dataset')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("plots/mlp_summary_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Performance table
    print("\nPerformance Summary Table:")
    print("-" * 80)
    print(f"{'Dataset':<12} {'Best Activation':<15} {'Best Depth':<20} {'Accuracy':<10} {'Complexity':<15}")
    print("-" * 80)
    
    for dataset_name in datasets:
        best_activation = activation_winners[dataset_name]
        best_depth = depth_winners[dataset_name]
        best_acc = max(all_results[dataset_name]['activation_results'][act]['test_accuracy'] 
                      for act in all_results[dataset_name]['activation_results'])
        
        # Get complexity from best depth model
        depth_results = all_results[dataset_name]['depth_results']
        complexity = depth_results[best_depth]['parameters']
        
        print(f"{dataset_name:<12} {best_activation:<15} {best_depth:<20} "
              f"{best_acc:<10.3f} {complexity:<15}")


if __name__ == "__main__":
    # Run comprehensive experiments
    results = comprehensive_mlp_experiments()
    
    print("\nMLP training experiments complete!") 