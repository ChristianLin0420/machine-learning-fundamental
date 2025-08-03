"""
Day 28 - Keras CNN on MNIST Demo
===============================

This module demonstrates building and training a Convolutional Neural Network
on the MNIST dataset using Keras. Target: >97% accuracy.

Author: ML Fundamentals Course
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import time

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_plots_dir():
    """Create plots directory if it doesn't exist."""
    if not os.path.exists('plots'):
        os.makedirs('plots')

def load_and_preprocess_mnist():
    """
    Load and preprocess the MNIST dataset.
    """
    print("="*60)
    print("LOADING AND PREPROCESSING MNIST DATASET")
    print("="*60)
    
    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    print(f"Original shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")
    
    # Normalize pixel values to [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Reshape to add channel dimension (for CNN)
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    
    print(f"\nAfter preprocessing:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"Pixel value range: [{X_train.min():.1f}, {X_train.max():.1f}]")
    print(f"Number of classes: {len(np.unique(y_train))}")
    print(f"Class distribution: {np.bincount(y_train)}")
    
    return (X_train, y_train), (X_test, y_test)

def visualize_mnist_samples(X_train, y_train, num_samples=15):
    """
    Visualize sample images from MNIST dataset.
    """
    create_plots_dir()
    
    fig, axes = plt.subplots(3, 5, figsize=(12, 8))
    axes = axes.ravel()
    
    for i in range(num_samples):
        img = X_train[i].reshape(28, 28)
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Label: {y_train[i]}', fontsize=12)
        axes[i].axis('off')
    
    plt.suptitle('MNIST Dataset Samples', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/mnist_samples.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_simple_cnn():
    """
    Create a simple CNN architecture for MNIST.
    """
    print("\n" + "="*60)
    print("CREATING SIMPLE CNN ARCHITECTURE")
    print("="*60)
    
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name='conv1'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        
        # Third Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', name='conv3'),
        
        # Classifier
        layers.Flatten(name='flatten'),
        layers.Dense(64, activation='relu', name='dense1'),
        layers.Dropout(0.5, name='dropout'),
        layers.Dense(10, activation='softmax', name='output')
    ], name='Simple_CNN')
    
    print("Simple CNN Architecture:")
    model.summary()
    
    return model

def create_improved_cnn():
    """
    Create an improved CNN architecture with better performance.
    """
    print("\n" + "="*60)
    print("CREATING IMPROVED CNN ARCHITECTURE")
    print("="*60)
    
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name='conv1'),
        layers.BatchNormalization(name='bn1'),
        layers.Conv2D(32, (3, 3), activation='relu', name='conv2'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        layers.Dropout(0.25, name='dropout1'),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', name='conv3'),
        layers.BatchNormalization(name='bn2'),
        layers.Conv2D(64, (3, 3), activation='relu', name='conv4'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        layers.Dropout(0.25, name='dropout2'),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', name='conv5'),
        layers.BatchNormalization(name='bn3'),
        layers.Dropout(0.25, name='dropout3'),
        
        # Classifier
        layers.Flatten(name='flatten'),
        layers.Dense(512, activation='relu', name='dense1'),
        layers.BatchNormalization(name='bn4'),
        layers.Dropout(0.5, name='dropout4'),
        layers.Dense(10, activation='softmax', name='output')
    ], name='Improved_CNN')
    
    print("Improved CNN Architecture:")
    model.summary()
    
    return model

def create_functional_cnn():
    """
    Create a CNN using Functional API to demonstrate flexibility.
    """
    print("\n" + "="*60)
    print("CREATING FUNCTIONAL API CNN")
    print("="*60)
    
    # Input layer
    inputs = layers.Input(shape=(28, 28, 1), name='input')
    
    # Feature extraction branch
    x = layers.Conv2D(32, (3, 3), activation='relu', name='conv1')(inputs)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', name='conv2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', name='conv3')(x)
    
    # Auxiliary branch for regularization
    aux_branch = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    aux_output = layers.Dense(10, activation='softmax', name='aux_output')(aux_branch)
    
    # Main branch
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(64, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.5, name='dropout')(x)
    main_output = layers.Dense(10, activation='softmax', name='main_output')(x)
    
    # Create model with multiple outputs
    model = models.Model(
        inputs=inputs, 
        outputs=[main_output, aux_output], 
        name='Functional_CNN'
    )
    
    print("Functional API CNN Architecture:")
    model.summary()
    
    return model

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, 
                           model_name="CNN", epochs=10, use_callbacks=True):
    """
    Train and evaluate a CNN model.
    """
    print(f"\n" + "="*60)
    print(f"TRAINING {model_name}")
    print("="*60)
    
    # Compile model
    if isinstance(model.output, list):  # Multi-output model
        model.compile(
            optimizer='adam',
            loss={'main_output': 'sparse_categorical_crossentropy', 
                  'aux_output': 'sparse_categorical_crossentropy'},
            loss_weights={'main_output': 1.0, 'aux_output': 0.3},
            metrics=['accuracy']
        )
        train_targets = {'main_output': y_train, 'aux_output': y_train}
        val_targets = {'main_output': y_test, 'aux_output': y_test}
    else:  # Single output model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        train_targets = y_train
        val_targets = y_test
    
    # Setup callbacks
    callbacks_list = []
    if use_callbacks:
        callbacks_list = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=0.0001,
                verbose=1
            )
        ]
    
    # Train model
    print(f"Starting training for {epochs} epochs...")
    start_time = time.time()
    
    history = model.fit(
        X_train, train_targets,
        batch_size=128,
        epochs=epochs,
        validation_data=(X_test, val_targets),
        callbacks=callbacks_list,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Evaluate model
    test_results = model.evaluate(X_test, val_targets, verbose=0)
    
    if isinstance(model.output, list):
        test_loss = test_results[0]
        test_accuracy = test_results[3]  # main_output_accuracy
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
    else:
        test_loss, test_accuracy = test_results
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Check if target accuracy achieved
    if test_accuracy > 0.97:
        print(f"🎉 TARGET ACHIEVED! Accuracy > 97%")
    else:
        print(f"Target not reached. Current: {test_accuracy:.1%}, Target: 97%")
    
    return history, test_accuracy, training_time

def plot_training_history(histories, model_names):
    """
    Plot training histories for multiple models.
    """
    create_plots_dir()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Training loss
    for history, name in zip(histories, model_names):
        if 'loss' in history.history:
            axes[0, 0].plot(history.history['loss'], label=f'{name}', linewidth=2)
    axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation loss
    for history, name in zip(histories, model_names):
        if 'val_loss' in history.history:
            axes[0, 1].plot(history.history['val_loss'], label=f'{name}', linewidth=2)
        elif 'val_main_output_loss' in history.history:
            axes[0, 1].plot(history.history['val_main_output_loss'], label=f'{name}', linewidth=2)
    axes[0, 1].set_title('Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Training accuracy
    for history, name in zip(histories, model_names):
        if 'accuracy' in history.history:
            axes[1, 0].plot(history.history['accuracy'], label=f'{name}', linewidth=2)
        elif 'main_output_accuracy' in history.history:
            axes[1, 0].plot(history.history['main_output_accuracy'], label=f'{name}', linewidth=2)
    axes[1, 0].set_title('Training Accuracy', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Validation accuracy
    for history, name in zip(histories, model_names):
        if 'val_accuracy' in history.history:
            axes[1, 1].plot(history.history['val_accuracy'], label=f'{name}', linewidth=2)
        elif 'val_main_output_accuracy' in history.history:
            axes[1, 1].plot(history.history['val_main_output_accuracy'], label=f'{name}', linewidth=2)
    axes[1, 1].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('CNN Models Training Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/cnn_training_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_predictions(model, X_test, y_test, num_samples=15):
    """
    Visualize model predictions on test samples.
    """
    create_plots_dir()
    
    # Make predictions
    if isinstance(model.output, list):
        predictions = model.predict(X_test[:num_samples], verbose=0)[0]  # Main output
    else:
        predictions = model.predict(X_test[:num_samples], verbose=0)
    
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = y_test[:num_samples]
    
    # Plot samples with predictions
    fig, axes = plt.subplots(3, 5, figsize=(15, 10))
    axes = axes.ravel()
    
    for i in range(num_samples):
        img = X_test[i].reshape(28, 28)
        axes[i].imshow(img, cmap='gray')
        
        # Color code: green for correct, red for incorrect
        color = 'green' if predicted_classes[i] == true_classes[i] else 'red'
        confidence = predictions[i][predicted_classes[i]]
        
        title = f'True: {true_classes[i]}, Pred: {predicted_classes[i]}\nConf: {confidence:.3f}'
        axes[i].set_title(title, fontsize=10, color=color, fontweight='bold')
        axes[i].axis('off')
    
    plt.suptitle('Model Predictions on Test Samples\n(Green: Correct, Red: Incorrect)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/cnn_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_model_performance(model, X_test, y_test):
    """
    Analyze model performance with confusion matrix and classification report.
    """
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    
    create_plots_dir()
    
    # Make predictions
    if isinstance(model.output, list):
        predictions = model.predict(X_test, verbose=0)[0]  # Main output
    else:
        predictions = model.predict(X_test, verbose=0)
    
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, predicted_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, predicted_classes))
    
    # Per-class accuracy
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(10), per_class_accuracy, color='skyblue', edgecolor='navy')
    plt.title('Per-Class Accuracy', fontsize=16, fontweight='bold')
    plt.xlabel('Digit Class')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, per_class_accuracy):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('plots/per_class_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_model_architectures():
    """
    Compare different CNN architectures.
    """
    print("\n" + "="*80)
    print("COMPARING CNN ARCHITECTURES")
    print("="*80)
    
    # Load data
    (X_train, y_train), (X_test, y_test) = load_and_preprocess_mnist()
    
    # Visualize samples
    visualize_mnist_samples(X_train, y_train)
    
    # Create models
    simple_cnn = create_simple_cnn()
    improved_cnn = create_improved_cnn()
    functional_cnn = create_functional_cnn()
    
    models = [simple_cnn, improved_cnn, functional_cnn]
    model_names = ['Simple CNN', 'Improved CNN', 'Functional CNN']
    
    # Train and evaluate all models
    histories = []
    accuracies = []
    training_times = []
    
    for model, name in zip(models, model_names):
        history, accuracy, train_time = train_and_evaluate_model(
            model, X_train, y_train, X_test, y_test, 
            model_name=name, epochs=15
        )
        histories.append(history)
        accuracies.append(accuracy)
        training_times.append(train_time)
    
    # Plot training comparison
    plot_training_history(histories, model_names)
    
    # Performance summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    results_df = []
    for name, accuracy, train_time in zip(model_names, accuracies, training_times):
        results_df.append({
            'Model': name,
            'Test Accuracy': f"{accuracy:.4f}",
            'Training Time': f"{train_time:.1f}s",
            'Target Achieved': "✅" if accuracy > 0.97 else "❌"
        })
        print(f"{name:15} | Accuracy: {accuracy:.4f} | Time: {train_time:.1f}s | Target: {'✅' if accuracy > 0.97 else '❌'}")
    
    # Bar chart comparison
    create_plots_dir()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    bars1 = axes[0].bar(model_names, accuracies, color=['#ff7f0e', '#2ca02c', '#d62728'])
    axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Test Accuracy')
    axes[0].set_ylim(0.95, 1.0)
    axes[0].axhline(y=0.97, color='red', linestyle='--', label='Target (97%)')
    axes[0].legend()
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Training time comparison
    bars2 = axes[1].bar(model_names, training_times, color=['#ff7f0e', '#2ca02c', '#d62728'])
    axes[1].set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Training Time (seconds)')
    
    # Add value labels on bars
    for bar, time in zip(bars2, training_times):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{time:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/model_comparison_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Select best model for detailed analysis
    best_idx = np.argmax(accuracies)
    best_model = models[best_idx]
    best_name = model_names[best_idx]
    
    print(f"\nDetailed analysis of best model: {best_name}")
    visualize_predictions(best_model, X_test, y_test)
    analyze_model_performance(best_model, X_test, y_test)
    
    return models, histories, accuracies, training_times

def demonstrate_cnn_mnist():
    """
    Main function to demonstrate CNN on MNIST.
    """
    print("Keras CNN on MNIST Comprehensive Demo")
    print("="*80)
    print("TensorFlow version:", tf.__version__)
    print("Keras version: Built-in with TensorFlow", tf.__version__)
    print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    print()
    
    # Compare different architectures
    models, histories, accuracies, training_times = compare_model_architectures()
    
    print("\n" + "="*80)
    print("CNN MNIST DEMONSTRATION COMPLETE")
    print("="*80)
    print("✅ Simple CNN implementation")
    print("✅ Improved CNN with BatchNorm and Dropout")
    print("✅ Functional API CNN with auxiliary output")
    print("✅ Model training with callbacks")
    print("✅ Performance comparison and analysis")
    print("✅ Prediction visualization")
    print("✅ Detailed performance metrics")
    print(f"\nBest accuracy achieved: {max(accuracies):.4f}")
    if max(accuracies) > 0.97:
        print("🎉 Target accuracy (>97%) achieved!")
    print("\nAll plots and analysis saved to 'plots/' directory")

if __name__ == "__main__":
    demonstrate_cnn_mnist()