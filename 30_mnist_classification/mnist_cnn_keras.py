"""
MNIST CNN Implementation using Keras/TensorFlow
==================================================

This module implements a Convolutional Neural Network for MNIST digit classification
using Keras/TensorFlow. It includes comprehensive training, evaluation, and analysis.

Key Features:
- CNN architecture with multiple layers
- Data preprocessing and augmentation
- Training with callbacks
- Comprehensive evaluation metrics
- Model comparison and analysis

Author: Assistant
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import os
import time
from typing import Dict, List, Tuple, Any

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class MNISTCNNKeras:
    """
    Keras/TensorFlow CNN implementation for MNIST digit classification.
    """
    
    def __init__(self):
        """Initialize the MNIST CNN class."""
        self.model = None
        self.history = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.class_names = [str(i) for i in range(10)]
        
    def load_and_preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess MNIST dataset.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print("Loading MNIST dataset...")
        
        # Load MNIST data
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
        
        # Normalize pixel values to [0, 1]
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        # Reshape for CNN (add channel dimension)
        X_train = X_train.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)
        
        # Convert labels to categorical (one-hot encoding)
        y_train_cat = keras.utils.to_categorical(y_train, 10)
        y_test_cat = keras.utils.to_categorical(y_test, 10)
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Training labels shape: {y_train_cat.shape}")
        print(f"Test labels shape: {y_test_cat.shape}")
        
        # Store for later use
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        
        return X_train, X_test, y_train_cat, y_test_cat
    
    def create_simple_cnn(self) -> keras.Model:
        """
        Create a simple CNN architecture.
        
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_deeper_cnn(self) -> keras.Model:
        """
        Create a deeper CNN architecture with batch normalization.
        
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_regularized_cnn(self) -> keras.Model:
        """
        Create a CNN with L2 regularization and different activation.
        
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1),
                         kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu',
                         kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu',
                         kernel_regularizer=keras.regularizers.l2(0.001)),
            
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_model(self, model: keras.Model, X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray, epochs: int = 10,
                   batch_size: int = 128) -> keras.callbacks.History:
        """
        Train the CNN model with callbacks.
        
        Args:
            model: Keras model to train
            X_train: Training images
            y_train: Training labels (one-hot encoded)
            X_test: Test images
            y_test: Test labels (one-hot encoded)
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        print(f"\nTraining model for {epochs} epochs...")
        
        # Create callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=3, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7
            )
        ]
        
        # Train the model
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        return history
    
    def evaluate_model(self, model: keras.Model, X_test: np.ndarray, 
                      y_test_cat: np.ndarray, y_test_orig: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the trained model comprehensively.
        
        Args:
            model: Trained Keras model
            X_test: Test images
            y_test_cat: Test labels (one-hot encoded)
            y_test_orig: Test labels (original integers)
            
        Returns:
            Dictionary with evaluation results
        """
        print("\nEvaluating model...")
        
        # Get predictions
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
        
        # Classification report
        class_report = classification_report(
            y_test_orig, y_pred, target_names=self.class_names, output_dict=True
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test_orig, y_pred)
        
        # Find misclassified examples
        misclassified_indices = np.where(y_pred != y_test_orig)[0]
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'misclassified_indices': misclassified_indices
        }
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Number of misclassified samples: {len(misclassified_indices)}")
        
        return results
    
    def compare_architectures(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Compare different CNN architectures.
        
        Args:
            X_train: Training images
            y_train: Training labels (one-hot encoded)
            X_test: Test images
            y_test: Test labels (one-hot encoded)
            
        Returns:
            Dictionary with comparison results
        """
        print("\n" + "="*60)
        print("COMPARING CNN ARCHITECTURES")
        print("="*60)
        
        architectures = {
            'Simple CNN': self.create_simple_cnn,
            'Deeper CNN': self.create_deeper_cnn,
            'Regularized CNN': self.create_regularized_cnn
        }
        
        results = {}
        
        for name, create_model_func in architectures.items():
            print(f"\nTraining {name}...")
            
            # Create and train model
            model = create_model_func()
            print(f"Model parameters: {model.count_params():,}")
            
            # Train with fewer epochs for comparison
            history = self.train_model(model, X_train, y_train, X_test, y_test, epochs=5)
            
            # Evaluate
            evaluation = self.evaluate_model(model, X_test, y_test, self.y_test)
            
            results[name] = {
                'model': model,
                'history': history,
                'evaluation': evaluation,
                'params': model.count_params()
            }
        
        return results
    
    def visualize_training_history(self, history: keras.callbacks.History, 
                                 model_name: str = "CNN") -> None:
        """
        Visualize training history (loss and accuracy curves).
        
        Args:
            history: Training history from model.fit()
            model_name: Name of the model for plot title
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(history.history['loss']) + 1)
        
        # Loss curves
        ax1.plot(epochs, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title(f'{model_name} - Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(epochs, history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title(f'{model_name} - Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'plots/{model_name.lower().replace(" ", "_")}_training_history.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_architecture_comparison(self, comparison_results: Dict[str, Any]) -> None:
        """
        Visualize comparison between different architectures.
        
        Args:
            comparison_results: Results from compare_architectures()
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data for comparison
        model_names = list(comparison_results.keys())
        accuracies = [comparison_results[name]['evaluation']['test_accuracy'] 
                     for name in model_names]
        losses = [comparison_results[name]['evaluation']['test_loss'] 
                 for name in model_names]
        params = [comparison_results[name]['params'] for name in model_names]
        
        # 1. Test Accuracy Comparison
        bars1 = axes[0, 0].bar(model_names, accuracies, color=['skyblue', 'lightgreen', 'salmon'])
        axes[0, 0].set_title('Test Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0.95, 1.0)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                          f'{acc:.4f}', ha='center', va='bottom')
        
        # 2. Model Parameters Comparison
        bars2 = axes[0, 1].bar(model_names, params, color=['skyblue', 'lightgreen', 'salmon'])
        axes[0, 1].set_title('Model Parameters Comparison')
        axes[0, 1].set_ylabel('Number of Parameters')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, param in zip(bars2, params):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(params)*0.01,
                          f'{param:,}', ha='center', va='bottom', rotation=45)
        
        # 3. Training Loss Curves
        for name in model_names:
            history = comparison_results[name]['history']
            epochs = range(1, len(history.history['loss']) + 1)
            axes[1, 0].plot(epochs, history.history['loss'], label=f'{name} (Train)', linewidth=2)
            axes[1, 0].plot(epochs, history.history['val_loss'], '--', label=f'{name} (Val)', linewidth=2)
        
        axes[1, 0].set_title('Training Loss Comparison')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Training Accuracy Curves
        for name in model_names:
            history = comparison_results[name]['history']
            epochs = range(1, len(history.history['accuracy']) + 1)
            axes[1, 1].plot(epochs, history.history['accuracy'], label=f'{name} (Train)', linewidth=2)
            axes[1, 1].plot(epochs, history.history['val_accuracy'], '--', label=f'{name} (Val)', linewidth=2)
        
        axes[1, 1].set_title('Training Accuracy Comparison')
        axes[1, 1].set_xlabel('Epochs')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/architecture_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()


def run_comprehensive_mnist_analysis():
    """
    Run comprehensive MNIST CNN analysis with Keras.
    
    Returns:
        Dictionary with all results
    """
    print("MNIST CNN with Keras - Comprehensive Analysis")
    print("=" * 60)
    
    # Check TensorFlow and GPU availability
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    # Create output directory
    os.makedirs('plots', exist_ok=True)
    
    # Initialize classifier
    mnist_cnn = MNISTCNNKeras()
    
    # Load and preprocess data
    X_train, X_test, y_train_cat, y_test_cat = mnist_cnn.load_and_preprocess_data()
    
    print("\n" + "="*60)
    print("TRAINING BEST MODEL")
    print("="*60)
    
    # Create and train the best model (deeper CNN)
    best_model = mnist_cnn.create_deeper_cnn()
    print(f"Model parameters: {best_model.count_params():,}")
    print("\nModel architecture:")
    best_model.summary()
    
    # Train the model
    history = mnist_cnn.train_model(
        best_model, X_train, y_train_cat, X_test, y_test_cat, epochs=15
    )
    
    # Evaluate the model
    evaluation = mnist_cnn.evaluate_model(best_model, X_test, y_test_cat, mnist_cnn.y_test)
    
    # Store results
    mnist_cnn.model = best_model
    mnist_cnn.history = history
    
    # Visualize training history
    mnist_cnn.visualize_training_history(history, "Best CNN")
    
    print("\n" + "="*60)
    print("ARCHITECTURE COMPARISON")
    print("="*60)
    
    # Compare different architectures
    comparison_results = mnist_cnn.compare_architectures(X_train, y_train_cat, X_test, y_test_cat)
    
    # Visualize comparison
    mnist_cnn.visualize_architecture_comparison(comparison_results)
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    
    print(f"Best Model Test Accuracy: {evaluation['test_accuracy']:.4f}")
    print(f"Best Model Test Loss: {evaluation['test_loss']:.4f}")
    print(f"Misclassified samples: {len(evaluation['misclassified_indices'])}/10000")
    
    print("\nArchitecture Comparison:")
    for name, results in comparison_results.items():
        acc = results['evaluation']['test_accuracy']
        params = results['params']
        print(f"  {name}: {acc:.4f} accuracy, {params:,} parameters")
    
    # Save the best model
    best_model.save('plots/best_mnist_cnn_model.h5')
    print(f"\nBest model saved to 'plots/best_mnist_cnn_model.h5'")
    
    return {
        'mnist_cnn': mnist_cnn,
        'best_model': best_model,
        'history': history,
        'evaluation': evaluation,
        'comparison_results': comparison_results
    }


if __name__ == "__main__":
    # Run the comprehensive analysis
    results = run_comprehensive_mnist_analysis()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("Check the 'plots/' directory for visualizations.")
    print("The trained model has been saved for further use.")