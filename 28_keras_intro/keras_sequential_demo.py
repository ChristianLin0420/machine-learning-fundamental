"""
Day 28 - Keras Sequential API Demo
=================================

This module demonstrates the Keras Sequential API for building neural networks.
We'll create simple feedforward networks and train them on various datasets.

Author: ML Fundamentals Course
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import make_moons, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_plots_dir():
    """Create plots directory if it doesn't exist."""
    if not os.path.exists('plots'):
        os.makedirs('plots')

def sequential_binary_classification():
    """
    Demonstrate Sequential API for binary classification using make_moons dataset.
    """
    print("="*60)
    print("SEQUENTIAL API - BINARY CLASSIFICATION")
    print("="*60)
    
    # Generate dataset
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Build Sequential model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(2,), name='hidden_1'),
        tf.keras.layers.Dense(8, activation='relu', name='hidden_2'),
        tf.keras.layers.Dense(1, activation='sigmoid', name='output')
    ], name='Sequential_Binary_Model')
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Display model architecture
    print(f"\nModel Summary:")
    model.summary()
    
    # Train model
    print(f"\nTraining model...")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test_scaled, y_test),
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"\nTest Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_accuracy:.4f}")
    
    return model, history, X_train_scaled, X_test_scaled, y_train, y_test, scaler

def sequential_multiclass_classification():
    """
    Demonstrate Sequential API for multiclass classification using Iris dataset.
    """
    print("\n" + "="*60)
    print("SEQUENTIAL API - MULTICLASS CLASSIFICATION")
    print("="*60)
    
    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    print(f"Class names: {iris.target_names}")
    print(f"Feature names: {iris.feature_names}")
    
    # Build Sequential model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(4,), name='hidden_1'),
        tf.keras.layers.Dropout(0.2, name='dropout_1'),
        tf.keras.layers.Dense(32, activation='relu', name='hidden_2'),
        tf.keras.layers.Dropout(0.1, name='dropout_2'),
        tf.keras.layers.Dense(3, activation='softmax', name='output')
    ], name='Sequential_Multiclass_Model')
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Display model architecture
    print(f"\nModel Summary:")
    model.summary()
    
    # Train model
    print(f"\nTraining model...")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=16,
        validation_data=(X_test_scaled, y_test),
        verbose=1
    )
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"\nTest Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_accuracy:.4f}")
    
    # Make predictions
    predictions = model.predict(X_test_scaled, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    
    print(f"\nSample Predictions:")
    for i in range(5):
        print(f"Sample {i+1}: True={iris.target_names[y_test[i]]}, "
              f"Pred={iris.target_names[predicted_classes[i]]}, "
              f"Confidence={predictions[i].max():.3f}")
    
    return model, history, X_test_scaled, y_test, predictions

def plot_training_history(history, title="Training History"):
    """
    Plot training and validation loss and accuracy.
    """
    create_plots_dir()
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'plots/sequential_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_decision_boundary_binary(model, X, y, scaler, title="Decision Boundary"):
    """
    Plot decision boundary for binary classification.
    """
    create_plots_dir()
    
    # Create a mesh
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Scale the mesh points
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_points_scaled = scaler.transform(mesh_points)
    
    # Make predictions
    Z = model.predict(mesh_points_scaled, verbose=0)
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
    plt.colorbar(label='Prediction Probability')
    
    # Plot data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/sequential_decision_boundary.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_optimizers():
    """
    Compare different optimizers using Sequential API.
    """
    print("\n" + "="*60)
    print("OPTIMIZER COMPARISON")
    print("="*60)
    
    # Generate dataset
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    optimizers = {
        'SGD': tf.keras.optimizers.SGD(learning_rate=0.01),
        'Adam': tf.keras.optimizers.Adam(learning_rate=0.001),
        'RMSprop': tf.keras.optimizers.RMSprop(learning_rate=0.001),
        'Adagrad': tf.keras.optimizers.Adagrad(learning_rate=0.01)
    }
    
    results = {}
    
    for opt_name, optimizer in optimizers.items():
        print(f"\nTraining with {opt_name}...")
        
        # Create model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test_scaled, y_test),
            verbose=0
        )
        
        # Evaluate
        test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
        
        results[opt_name] = {
            'history': history,
            'test_loss': test_loss,
            'test_accuracy': test_accuracy
        }
        
        print(f"{opt_name} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    # Plot comparison
    plot_optimizer_comparison(results)
    
    return results

def plot_optimizer_comparison(results):
    """
    Plot comparison of different optimizers.
    """
    create_plots_dir()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Training loss
    for opt_name, result in results.items():
        axes[0, 0].plot(result['history'].history['loss'], 
                       label=f'{opt_name}', linewidth=2)
    axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation loss
    for opt_name, result in results.items():
        axes[0, 1].plot(result['history'].history['val_loss'], 
                       label=f'{opt_name}', linewidth=2)
    axes[0, 1].set_title('Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Training accuracy
    for opt_name, result in results.items():
        axes[1, 0].plot(result['history'].history['accuracy'], 
                       label=f'{opt_name}', linewidth=2)
    axes[1, 0].set_title('Training Accuracy', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Validation accuracy
    for opt_name, result in results.items():
        axes[1, 1].plot(result['history'].history['val_accuracy'], 
                       label=f'{opt_name}', linewidth=2)
    axes[1, 1].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Optimizer Comparison - Sequential API', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/sequential_optimizer_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Bar plot of final test accuracies
    plt.figure(figsize=(10, 6))
    opt_names = list(results.keys())
    test_accuracies = [results[name]['test_accuracy'] for name in opt_names]
    
    bars = plt.bar(opt_names, test_accuracies, color=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    plt.title('Final Test Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Test Accuracy')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, test_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('plots/sequential_test_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def demonstrate_sequential_api():
    """
    Main function to demonstrate Sequential API capabilities.
    """
    print("Keras Sequential API Comprehensive Demo")
    print("="*60)
    print("TensorFlow version:", tf.__version__)
    print("Keras version: Built-in with TensorFlow", tf.__version__)
    print()
    
    # Binary classification demo
    binary_model, binary_history, X_train, X_test, y_train, y_test, scaler = sequential_binary_classification()
    
    # Plot training history
    plot_training_history(binary_history, "Binary Classification Training")
    
    # Plot decision boundary
    X_combined = np.vstack([X_train, X_test])
    y_combined = np.hstack([y_train, y_test])
    plot_decision_boundary_binary(binary_model, X_combined, y_combined, scaler, 
                                 "Sequential API - Binary Classification Decision Boundary")
    
    # Multiclass classification demo
    multiclass_model, multiclass_history, X_test_iris, y_test_iris, predictions = sequential_multiclass_classification()
    
    # Optimizer comparison
    optimizer_results = compare_optimizers()
    
    print("\n" + "="*60)
    print("SEQUENTIAL API DEMONSTRATION COMPLETE")
    print("="*60)
    print("✅ Binary classification with Sequential API")
    print("✅ Multiclass classification with Sequential API")
    print("✅ Optimizer comparison")
    print("✅ Training visualization")
    print("✅ Decision boundary visualization")
    print("\nAll plots saved to 'plots/' directory")

if __name__ == "__main__":
    demonstrate_sequential_api()