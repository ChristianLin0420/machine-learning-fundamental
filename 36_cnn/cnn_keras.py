#!/usr/bin/env python3
"""
CNN Keras Implementation
========================

Modern CNN implementation using Keras/TensorFlow with various architectures
and comprehensive experiments for image classification.

Author: ML Fundamentals Course
Day: 36 - CNNs
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Any
import time
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Check TensorFlow version and GPU
print(f"TensorFlow version: {tf.__version__}")
if tf.config.list_physical_devices('GPU'):
    print("GPU is available")
    # Enable memory growth to avoid allocating all GPU memory
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)
else:
    print("GPU not available, using CPU")


def create_simple_cnn(input_shape: Tuple[int, int, int], num_classes: int) -> keras.Model:
    """
    Create a simple CNN using Keras Sequential API.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # First conv block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Second conv block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third conv block
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_deep_cnn(input_shape: Tuple[int, int, int], num_classes: int) -> keras.Model:
    """
    Create a deeper CNN with batch normalization.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # First conv block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second conv block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third conv block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_functional_cnn(input_shape: Tuple[int, int, int], num_classes: int) -> keras.Model:
    """
    Create a CNN using Keras Functional API with skip connections.
    
    Args:
        input_shape: Input image shape (height, width, channels)
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape)
    
    # First conv block
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    
    # Residual-like block
    residual = x
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])  # Skip connection
    
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Second conv block
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Another residual-like block
    residual = x
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])
    
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Third conv block
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Global average pooling instead of flatten
    x = layers.GlobalAveragePooling2D()(x)
    
    # Classifier
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def load_keras_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess MNIST for Keras.
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("Loading MNIST dataset for Keras...")
    
    # Load MNIST
    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize pixel values
    X_train_full = X_train_full.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Reshape for CNN (add channel dimension)
    X_train_full = X_train_full.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    
    # Create train/validation split
    validation_split = 0.1
    val_size = int(len(X_train_full) * validation_split)
    
    X_train = X_train_full[val_size:]
    X_val = X_train_full[:val_size]
    y_train = y_train_full[val_size:]
    y_val = y_train_full[:val_size]
    
    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_keras_cifar10() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess CIFAR-10 for Keras.
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("Loading CIFAR-10 dataset for Keras...")
    
    # Load CIFAR-10
    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()
    
    # Normalize pixel values
    X_train_full = X_train_full.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Flatten labels
    y_train_full = y_train_full.flatten()
    y_test = y_test.flatten()
    
    # Create train/validation split
    validation_split = 0.1
    val_size = int(len(X_train_full) * validation_split)
    
    X_train = X_train_full[val_size:]
    X_val = X_train_full[:val_size]
    y_train = y_train_full[val_size:]
    y_val = y_train_full[:val_size]
    
    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def visualize_keras_training(history: keras.callbacks.History, title: str = "Training History",
                           save_path: str = None) -> None:
    """Visualize Keras training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(history.history['loss'], label='Training Loss', marker='o')
    if 'val_loss' in history.history:
        axes[0].plot(history.history['val_loss'], label='Validation Loss', marker='s')
    axes[0].set_title(f'{title} - Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    if 'val_accuracy' in history.history:
        axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
    axes[1].set_title(f'{title} - Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history saved to {save_path}")
    
    plt.show()


def visualize_keras_filters(model: keras.Model, layer_name: str = None, 
                          save_path: str = None) -> None:
    """Visualize learned filters from Keras model."""
    # Find first conv layer
    conv_layer = None
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D):
            conv_layer = layer
            break
    
    if conv_layer is None:
        print("No convolution layer found!")
        return
    
    # Get filter weights
    filters = conv_layer.get_weights()[0]  # Shape: (h, w, in_channels, out_channels)
    
    # Transpose to match PyTorch format: (out_channels, in_channels, h, w)
    filters = np.transpose(filters, (3, 2, 0, 1))
    
    num_filters = min(16, filters.shape[0])
    cols = 4
    rows = (num_filters + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_filters):
        row = i // cols
        col = i % cols
        
        # Get first input channel
        filter_img = filters[i, 0]  # Shape: (h, w)
        
        # Normalize for visualization
        filter_img = (filter_img - filter_img.min()) / (filter_img.max() - filter_img.min() + 1e-8)
        
        axes[row, col].imshow(filter_img, cmap='RdBu')
        axes[row, col].set_title(f'Filter {i+1}')
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(num_filters, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.suptitle('Learned Convolution Filters (Keras)', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Filter visualization saved to {save_path}")
    
    plt.show()


def get_keras_feature_maps(model: keras.Model, input_sample: np.ndarray, 
                          layer_index: int = 0) -> np.ndarray:
    """Extract feature maps from a specific layer in Keras model."""
    # Find conv layers
    conv_layers = [layer for layer in model.layers if isinstance(layer, layers.Conv2D)]
    
    if not conv_layers or layer_index >= len(conv_layers):
        print("No convolution layers found or invalid layer index!")
        return None
    
    # Create intermediate model
    target_layer = conv_layers[layer_index]
    intermediate_model = keras.Model(inputs=model.input, outputs=target_layer.output)
    
    # Get feature maps
    feature_maps = intermediate_model.predict(input_sample.reshape(1, *input_sample.shape))
    
    return feature_maps


def visualize_keras_feature_maps(model: keras.Model, input_sample: np.ndarray,
                               save_path: str = None) -> None:
    """Visualize feature maps from Keras model."""
    # Get feature maps from first conv layer
    feature_maps = get_keras_feature_maps(model, input_sample, layer_index=0)
    
    if feature_maps is None:
        return
    
    feature_maps = feature_maps[0]  # Remove batch dimension
    
    # Show subset of feature maps
    num_maps = min(16, feature_maps.shape[-1])
    cols = 4
    rows = (num_maps + cols - 1) // cols + 1  # +1 for original image
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 2))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # Original image
    if input_sample.shape[-1] == 1:  # Grayscale
        axes[0, 0].imshow(input_sample[:, :, 0], cmap='gray')
    else:  # RGB
        axes[0, 0].imshow(input_sample)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Hide other subplots in first row
    for i in range(1, cols):
        axes[0, i].axis('off')
    
    # Feature maps
    for i in range(num_maps):
        row = (i // cols) + 1
        col = i % cols
        
        feature_map = feature_maps[:, :, i]
        im = axes[row, col].imshow(feature_map, cmap='viridis')
        axes[row, col].set_title(f'Feature {i+1}')
        axes[row, col].axis('off')
        plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
    
    plt.suptitle('Feature Maps from First Conv Layer (Keras)', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature map visualization saved to {save_path}")
    
    plt.show()


def create_keras_confusion_matrix(model: keras.Model, X_test: np.ndarray, y_test: np.ndarray,
                                class_names: List[str] = None, save_path: str = None) -> None:
    """Create confusion matrix for Keras model."""
    # Make predictions
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names if class_names else range(len(cm)),
                yticklabels=class_names if class_names else range(len(cm)))
    plt.title('Confusion Matrix (Keras CNN)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def analyze_keras_misclassifications(model: keras.Model, X_test: np.ndarray, y_test: np.ndarray,
                                   class_names: List[str] = None, save_path: str = None) -> None:
    """Analyze misclassified examples from Keras model."""
    # Make predictions
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)
    
    # Find misclassified examples
    wrong_indices = np.where(y_pred != y_test)[0]
    
    if len(wrong_indices) == 0:
        print("No misclassifications found!")
        return
    
    # Select random misclassified examples
    num_examples = min(20, len(wrong_indices))
    selected_indices = np.random.choice(wrong_indices, num_examples, replace=False)
    
    # Visualize
    cols = 5
    rows = (num_examples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(selected_indices):
        row = i // cols
        col = i % cols
        
        image = X_test[idx]
        
        if image.shape[-1] == 1:  # Grayscale
            axes[row, col].imshow(image[:, :, 0], cmap='gray')
        else:  # RGB
            axes[row, col].imshow(image)
        
        pred_label = class_names[y_pred[idx]] if class_names else y_pred[idx]
        true_label = class_names[y_test[idx]] if class_names else y_test[idx]
        
        axes[row, col].set_title(
            f"Pred: {pred_label}, True: {true_label}\nConf: {confidences[idx]:.2f}"
        )
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(num_examples, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.suptitle('Misclassified Examples (Keras CNN)', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Misclassification analysis saved to {save_path}")
    
    plt.show()


def compare_keras_architectures() -> Dict[str, Any]:
    """Compare different CNN architectures using Keras."""
    print("\nComparing Keras CNN Architectures on MNIST...")
    print("=" * 50)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_keras_mnist()
    input_shape = X_train.shape[1:]  # (height, width, channels)
    num_classes = 10
    
    # Define architectures
    architectures = {
        'Simple CNN': create_simple_cnn(input_shape, num_classes),
        'Deep CNN': create_deep_cnn(input_shape, num_classes),
        'Functional CNN': create_functional_cnn(input_shape, num_classes)
    }
    
    results = {}
    
    for arch_name, model in architectures.items():
        print(f"\nTraining {arch_name}...")
        
        # Model summary
        total_params = model.count_params()
        print(f"  Total parameters: {total_params:,}")
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(factor=0.2, patience=3)
        ]
        
        # Train model
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=128,
            callbacks=callbacks_list,
            verbose=0
        )
        
        training_time = time.time() - start_time
        
        # Evaluate on test set
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
        print(f"  Training time: {training_time:.1f}s")
        print(f"  Test accuracy: {test_acc:.4f}")
        
        results[arch_name] = {
            'model': model,
            'history': history,
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'total_params': total_params,
            'training_time': training_time
        }
    
    return results


def data_augmentation_comparison() -> Dict[str, Any]:
    """Compare models with and without data augmentation."""
    print("\nComparing Data Augmentation Effects...")
    print("=" * 50)
    
    # Load CIFAR-10 (more challenging dataset)
    X_train, X_val, X_test, y_train, y_val, y_test = load_keras_cifar10()
    input_shape = X_train.shape[1:]
    num_classes = 10
    
    results = {}
    
    # Without augmentation
    print("\nTraining without data augmentation...")
    model_no_aug = create_deep_cnn(input_shape, num_classes)
    
    history_no_aug = model_no_aug.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=128,
        verbose=0
    )
    
    test_acc_no_aug = model_no_aug.evaluate(X_test, y_test, verbose=0)[1]
    
    # With augmentation
    print("Training with data augmentation...")
    model_aug = create_deep_cnn(input_shape, num_classes)
    
    # Data augmentation
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1
    )
    datagen.fit(X_train)
    
    history_aug = model_aug.fit(
        datagen.flow(X_train, y_train, batch_size=128),
        steps_per_epoch=len(X_train) // 128,
        validation_data=(X_val, y_val),
        epochs=10,
        verbose=0
    )
    
    test_acc_aug = model_aug.evaluate(X_test, y_test, verbose=0)[1]
    
    print(f"Test accuracy without augmentation: {test_acc_no_aug:.4f}")
    print(f"Test accuracy with augmentation: {test_acc_aug:.4f}")
    print(f"Improvement: {test_acc_aug - test_acc_no_aug:.4f}")
    
    results = {
        'no_augmentation': {
            'model': model_no_aug,
            'history': history_no_aug,
            'test_accuracy': test_acc_no_aug
        },
        'with_augmentation': {
            'model': model_aug,
            'history': history_aug,
            'test_accuracy': test_acc_aug
        }
    }
    
    return results


def activation_function_comparison() -> Dict[str, Any]:
    """Compare different activation functions in CNNs."""
    print("\nComparing Activation Functions...")
    print("=" * 50)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_keras_mnist()
    input_shape = X_train.shape[1:]
    num_classes = 10
    
    # Define activations to test
    activations = ['relu', 'leaky_relu', 'elu', 'swish']
    results = {}
    
    for activation in activations:
        print(f"\nTraining with {activation} activation...")
        
        # Create model with specific activation
        if activation == 'leaky_relu':
            model = models.Sequential([
                layers.Conv2D(32, (3, 3), input_shape=input_shape),
                layers.LeakyReLU(alpha=0.01),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3)),
                layers.LeakyReLU(alpha=0.01),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3)),
                layers.LeakyReLU(alpha=0.01),
                layers.Flatten(),
                layers.Dense(64),
                layers.LeakyReLU(alpha=0.01),
                layers.Dense(num_classes, activation='softmax')
            ])
        elif activation == 'swish':
            model = models.Sequential([
                layers.Conv2D(32, (3, 3), activation='swish', input_shape=input_shape),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='swish'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='swish'),
                layers.Flatten(),
                layers.Dense(64, activation='swish'),
                layers.Dense(num_classes, activation='softmax')
            ])
        else:
            model = models.Sequential([
                layers.Conv2D(32, (3, 3), activation=activation, input_shape=input_shape),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation=activation),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation=activation),
                layers.Flatten(),
                layers.Dense(64, activation=activation),
                layers.Dense(num_classes, activation='softmax')
            ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=128,
            verbose=0
        )
        
        # Evaluate
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
        print(f"  Test accuracy: {test_acc:.4f}")
        
        results[activation] = {
            'model': model,
            'history': history,
            'test_accuracy': test_acc,
            'test_loss': test_loss
        }
    
    return results


def run_comprehensive_keras_experiments() -> Dict[str, Any]:
    """Run comprehensive Keras CNN experiments."""
    print("Starting Comprehensive Keras CNN Experiments")
    print("=" * 60)
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    all_results = {}
    
    # Experiment 1: Architecture comparison
    print("\n" + "=" * 50)
    print("EXPERIMENT 1: Architecture Comparison")
    print("=" * 50)
    
    arch_results = compare_keras_architectures()
    all_results['architectures'] = arch_results
    
    # Find best model for visualization
    best_arch = max(arch_results.keys(), 
                   key=lambda k: arch_results[k]['test_accuracy'])
    best_model = arch_results[best_arch]['model']
    best_history = arch_results[best_arch]['history']
    
    print(f"\nBest architecture: {best_arch}")
    print(f"Test accuracy: {arch_results[best_arch]['test_accuracy']:.4f}")
    
    # Visualizations
    print("\nCreating visualizations...")
    
    # Training history
    visualize_keras_training(best_history, f"Best Model ({best_arch})",
                           'plots/keras_best_training.png')
    
    # Filters
    visualize_keras_filters(best_model, save_path='plots/keras_learned_filters.png')
    
    # Feature maps
    _, _, X_test, _, _, y_test = load_keras_mnist()
    sample_input = X_test[0]
    visualize_keras_feature_maps(best_model, sample_input, 
                               'plots/keras_feature_maps.png')
    
    # Confusion matrix
    create_keras_confusion_matrix(best_model, X_test, y_test,
                                class_names=[str(i) for i in range(10)],
                                save_path='plots/keras_confusion_matrix.png')
    
    # Misclassifications
    analyze_keras_misclassifications(best_model, X_test, y_test,
                                   class_names=[str(i) for i in range(10)],
                                   save_path='plots/keras_misclassifications.png')
    
    # Experiment 2: Activation function comparison
    print("\n" + "=" * 50)
    print("EXPERIMENT 2: Activation Function Comparison")
    print("=" * 50)
    
    activation_results = activation_function_comparison()
    all_results['activations'] = activation_results
    
    # Experiment 3: Data augmentation comparison (on CIFAR-10)
    print("\n" + "=" * 50)
    print("EXPERIMENT 3: Data Augmentation Comparison")
    print("=" * 50)
    
    # Note: This might take longer, so we'll use a smaller subset or fewer epochs
    # aug_results = data_augmentation_comparison()
    # all_results['augmentation'] = aug_results
    
    # Create comprehensive comparison plots
    create_keras_comparison_plots(all_results)
    
    print("\nKeras CNN experiments completed!")
    print("Check the plots/ directory for all visualizations.")
    
    return all_results


def create_keras_comparison_plots(results: Dict[str, Any]) -> None:
    """Create comprehensive comparison plots."""
    
    # Architecture comparison
    if 'architectures' in results:
        arch_results = results['architectures']
        
        plt.figure(figsize=(15, 10))
        
        # Test accuracies
        plt.subplot(2, 3, 1)
        arch_names = list(arch_results.keys())
        test_accs = [arch_results[name]['test_accuracy'] for name in arch_names]
        param_counts = [arch_results[name]['total_params'] for name in arch_names]
        
        bars = plt.bar(arch_names, test_accs, color=['skyblue', 'lightcoral', 'lightgreen'])
        plt.title('Test Accuracy by Architecture')
        plt.ylabel('Test Accuracy')
        plt.xticks(rotation=45)
        
        # Add parameter count labels
        for bar, param_count in zip(bars, param_counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{param_count:,}', ha='center', va='bottom', fontsize=8)
        
        # Training time comparison
        plt.subplot(2, 3, 2)
        training_times = [arch_results[name]['training_time'] for name in arch_names]
        plt.bar(arch_names, training_times, color=['skyblue', 'lightcoral', 'lightgreen'])
        plt.title('Training Time by Architecture')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=45)
        
        # Parameter efficiency
        plt.subplot(2, 3, 3)
        plt.scatter(param_counts, test_accs, s=100, alpha=0.7)
        for i, name in enumerate(arch_names):
            plt.annotate(name, (param_counts[i], test_accs[i]), 
                        xytext=(5, 5), textcoords='offset points')
        plt.title('Parameter Efficiency')
        plt.xlabel('Number of Parameters')
        plt.ylabel('Test Accuracy')
        plt.grid(True)
        
        # Validation accuracy curves
        plt.subplot(2, 3, 4)
        for arch_name in arch_names:
            history = arch_results[arch_name]['history']
            plt.plot(history.history['val_accuracy'], label=arch_name, marker='o')
        plt.title('Validation Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Loss curves
        plt.subplot(2, 3, 5)
        for arch_name in arch_names:
            history = arch_results[arch_name]['history']
            plt.plot(history.history['val_loss'], label=arch_name, marker='s')
        plt.title('Validation Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss')
        plt.legend()
        plt.grid(True)
    
    # Activation function comparison
    if 'activations' in results:
        activation_results = results['activations']
        
        plt.subplot(2, 3, 6)
        activation_names = list(activation_results.keys())
        activation_accs = [activation_results[name]['test_accuracy'] for name in activation_names]
        
        plt.bar(activation_names, activation_accs, color='lightsteelblue')
        plt.title('Activation Function Comparison')
        plt.ylabel('Test Accuracy')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('plots/keras_comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    results = run_comprehensive_keras_experiments()