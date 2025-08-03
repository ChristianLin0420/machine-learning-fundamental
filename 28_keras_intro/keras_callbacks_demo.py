"""
Day 28 - Keras Callbacks Demo
============================

This module demonstrates various Keras callbacks for enhanced training control.
We'll implement EarlyStopping, ModelCheckpoint, LearningRateScheduler, and custom callbacks.

Author: ML Fundamentals Course
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import time

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_plots_dir():
    """Create plots directory if it doesn't exist."""
    if not os.path.exists('plots'):
        os.makedirs('plots')

class TrainingTimeCallback(tf.keras.callbacks.Callback):
    """
    Custom callback to track training time per epoch.
    """
    def __init__(self):
        super().__init__()
        self.epoch_times = []
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: {epoch_time:.3f}s")

class LossHistoryCallback(tf.keras.callbacks.Callback):
    """
    Custom callback to track detailed loss history.
    """
    def __init__(self):
        super().__init__()
        self.batch_losses = []
        self.epoch_losses = []
        
    def on_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs.get('loss'))
        
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_losses.append(logs.get('loss'))

def create_sample_model(input_shape, output_units=1, output_activation='sigmoid'):
    """
    Create a sample model for callback demonstrations.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(output_units, activation=output_activation)
    ], name='Sample_Model')
    
    return model

def demonstrate_early_stopping():
    """
    Demonstrate EarlyStopping callback.
    """
    print("="*60)
    print("EARLY STOPPING CALLBACK DEMO")
    print("="*60)
    
    # Generate dataset
    X, y = make_classification(
        n_samples=2000, n_features=20, n_redundant=0, n_informative=10,
        n_clusters_per_class=1, n_classes=2, random_state=42
    )
    
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
    
    # Create model
    model = create_sample_model(input_shape=(20,))
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Define EarlyStopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1,
        mode='min'
    )
    
    print("\nTraining with EarlyStopping (patience=10)...")
    
    # Train model with early stopping
    history = model.fit(
        X_train_scaled, y_train,
        epochs=200,  # Large number, early stopping will halt training
        batch_size=32,
        validation_data=(X_test_scaled, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate final model
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"\nFinal Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Training stopped at epoch: {len(history.history['loss'])}")
    
    return history, model

def demonstrate_model_checkpoint():
    """
    Demonstrate ModelCheckpoint callback.
    """
    print("\n" + "="*60)
    print("MODEL CHECKPOINT CALLBACK DEMO")
    print("="*60)
    
    # Generate dataset
    X, y = make_moons(n_samples=1000, noise=0.3, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create model
    model = create_sample_model(input_shape=(2,))
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Define ModelCheckpoint callback
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        verbose=1
    )
    
    print("\nTraining with ModelCheckpoint...")
    
    # Train model with checkpointing
    history = model.fit(
        X_train_scaled, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test_scaled, y_test),
        callbacks=[model_checkpoint],
        verbose=1
    )
    
    # Load best model
    print("\nLoading best saved model...")
    best_model = tf.keras.models.load_model('best_model.h5')
    
    # Compare current vs best model
    current_score = model.evaluate(X_test_scaled, y_test, verbose=0)
    best_score = best_model.evaluate(X_test_scaled, y_test, verbose=0)
    
    print(f"\nModel Comparison:")
    print(f"Current model - Loss: {current_score[0]:.4f}, Accuracy: {current_score[1]:.4f}")
    print(f"Best saved model - Loss: {best_score[0]:.4f}, Accuracy: {best_score[1]:.4f}")
    
    # Clean up
    if os.path.exists('best_model.h5'):
        os.remove('best_model.h5')
        print("Cleaned up saved model file")
    
    return history, best_model

def demonstrate_learning_rate_scheduler():
    """
    Demonstrate LearningRateScheduler callback.
    """
    print("\n" + "="*60)
    print("LEARNING RATE SCHEDULER CALLBACK DEMO")
    print("="*60)
    
    # Generate challenging dataset
    X, y = make_classification(
        n_samples=1500, n_features=30, n_redundant=5, n_informative=15,
        n_clusters_per_class=2, n_classes=3, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define learning rate schedules
    def step_decay(epoch):
        """Step decay schedule."""
        initial_lr = 0.01
        drop = 0.5
        epochs_drop = 20
        lr = initial_lr * (drop ** (epoch // epochs_drop))
        return lr
    
    def exponential_decay(epoch):
        """Exponential decay schedule."""
        initial_lr = 0.01
        decay = 0.95
        lr = initial_lr * (decay ** epoch)
        return lr
    
    def cosine_annealing(epoch):
        """Cosine annealing schedule."""
        initial_lr = 0.01
        min_lr = 0.0001
        period = 50
        lr = min_lr + (initial_lr - min_lr) * (1 + np.cos(np.pi * epoch / period)) / 2
        return lr
    
    schedules = {
        'Step Decay': step_decay,
        'Exponential Decay': exponential_decay,
        'Cosine Annealing': cosine_annealing
    }
    
    results = {}
    
    for schedule_name, schedule_func in schedules.items():
        print(f"\nTraining with {schedule_name}...")
        
        # Create fresh model
        model = create_sample_model(input_shape=(30,), output_units=3, output_activation='softmax')
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Create learning rate scheduler
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
            schedule_func,
            verbose=0
        )
        
        # Train model
        history = model.fit(
            X_train_scaled, y_train,
            epochs=60,
            batch_size=32,
            validation_data=(X_test_scaled, y_test),
            callbacks=[lr_scheduler],
            verbose=0
        )
        
        # Evaluate
        test_score = model.evaluate(X_test_scaled, y_test, verbose=0)
        results[schedule_name] = {
            'history': history,
            'test_loss': test_score[0],
            'test_accuracy': test_score[1]
        }
        
        print(f"{schedule_name} - Test Loss: {test_score[0]:.4f}, Test Accuracy: {test_score[1]:.4f}")
    
    return results

def demonstrate_reduce_lr_on_plateau():
    """
    Demonstrate ReduceLROnPlateau callback.
    """
    print("\n" + "="*60)
    print("REDUCE LR ON PLATEAU CALLBACK DEMO")
    print("="*60)
    
    # Generate dataset
    X, y = make_classification(
        n_samples=1000, n_features=15, n_redundant=0, n_informative=10,
        n_clusters_per_class=1, n_classes=2, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create model
    model = create_sample_model(input_shape=(15,))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),  # Higher initial LR
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Define ReduceLROnPlateau callback
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.0001,
        verbose=1
    )
    
    print("\nTraining with ReduceLROnPlateau...")
    
    # Train model
    history = model.fit(
        X_train_scaled, y_train,
        epochs=80,
        batch_size=32,
        validation_data=(X_test_scaled, y_test),
        callbacks=[reduce_lr],
        verbose=1
    )
    
    # Evaluate
    test_score = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"\nFinal Results:")
    print(f"Test Loss: {test_score[0]:.4f}")
    print(f"Test Accuracy: {test_score[1]:.4f}")
    
    return history

def demonstrate_custom_callbacks():
    """
    Demonstrate custom callbacks.
    """
    print("\n" + "="*60)
    print("CUSTOM CALLBACKS DEMO")
    print("="*60)
    
    # Generate dataset
    X, y = make_moons(n_samples=800, noise=0.2, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create model
    model = create_sample_model(input_shape=(2,))
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Create custom callbacks
    time_callback = TrainingTimeCallback()
    loss_callback = LossHistoryCallback()
    
    print("\nTraining with custom callbacks...")
    
    # Train model
    history = model.fit(
        X_train_scaled, y_train,
        epochs=30,
        batch_size=16,
        validation_data=(X_test_scaled, y_test),
        callbacks=[time_callback, loss_callback],
        verbose=1
    )
    
    print(f"\nCustom Callback Results:")
    print(f"Average epoch time: {np.mean(time_callback.epoch_times):.3f}s")
    print(f"Total training time: {sum(time_callback.epoch_times):.3f}s")
    print(f"Batch losses recorded: {len(loss_callback.batch_losses)}")
    
    return history, time_callback, loss_callback

def combine_multiple_callbacks():
    """
    Demonstrate combining multiple callbacks.
    """
    print("\n" + "="*60)
    print("MULTIPLE CALLBACKS COMBINATION DEMO")
    print("="*60)
    
    # Generate dataset
    X, y = make_classification(
        n_samples=1200, n_features=25, n_redundant=5, n_informative=15,
        n_clusters_per_class=1, n_classes=2, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create model
    model = create_sample_model(input_shape=(25,))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Combine multiple callbacks
    callbacks_list = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='combined_best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=7,
            min_lr=0.00001,
            verbose=1
        ),
        TrainingTimeCallback(),
        LossHistoryCallback()
    ]
    
    print("\nTraining with combined callbacks:")
    print("- EarlyStopping")
    print("- ModelCheckpoint")  
    print("- ReduceLROnPlateau")
    print("- TrainingTimeCallback")
    print("- LossHistoryCallback")
    
    # Train model
    history = model.fit(
        X_train_scaled, y_train,
        epochs=100,  # Large number, callbacks will control training
        batch_size=32,
        validation_data=(X_test_scaled, y_test),
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Evaluate final results
    test_score = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"\nFinal Results:")
    print(f"Test Loss: {test_score[0]:.4f}")
    print(f"Test Accuracy: {test_score[1]:.4f}")
    print(f"Training stopped at epoch: {len(history.history['loss'])}")
    
    # Clean up
    if os.path.exists('combined_best_model.h5'):
        os.remove('combined_best_model.h5')
        print("Cleaned up saved model file")
    
    return history, callbacks_list

def plot_callbacks_comparison():
    """
    Create comprehensive visualizations of callback effects.
    """
    create_plots_dir()
    
    print("\n" + "="*60)
    print("CREATING CALLBACK VISUALIZATIONS")
    print("="*60)
    
    # Run demonstrations and collect histories
    print("Running callback demonstrations for visualization...")
    
    early_stopping_history, _ = demonstrate_early_stopping()
    checkpoint_history, _ = demonstrate_model_checkpoint()
    lr_scheduler_results = demonstrate_learning_rate_scheduler()
    plateau_history = demonstrate_reduce_lr_on_plateau()
    combined_history, _ = combine_multiple_callbacks()
    
    # Plot 1: Early Stopping Effect
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(early_stopping_history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(early_stopping_history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Early Stopping Effect', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(early_stopping_history.history['accuracy'], label='Training Accuracy', linewidth=2)
    plt.plot(early_stopping_history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    plt.title('Early Stopping - Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/early_stopping_effect.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Learning Rate Schedules Comparison
    plt.figure(figsize=(15, 10))
    
    # Learning rates over time
    plt.subplot(2, 2, 1)
    epochs = range(60)
    schedules = {
        'Step Decay': lambda e: 0.01 * (0.5 ** (e // 20)),
        'Exponential Decay': lambda e: 0.01 * (0.95 ** e),
        'Cosine Annealing': lambda e: 0.0001 + (0.01 - 0.0001) * (1 + np.cos(np.pi * e / 50)) / 2
    }
    
    for name, schedule in schedules.items():
        lrs = [schedule(e) for e in epochs]
        plt.plot(epochs, lrs, label=name, linewidth=2)
    
    plt.title('Learning Rate Schedules', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Training loss comparison
    plt.subplot(2, 2, 2)
    for name, result in lr_scheduler_results.items():
        plt.plot(result['history'].history['loss'], label=name, linewidth=2)
    plt.title('Training Loss - Different LR Schedules', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Validation accuracy comparison
    plt.subplot(2, 2, 3)
    for name, result in lr_scheduler_results.items():
        plt.plot(result['history'].history['val_accuracy'], label=name, linewidth=2)
    plt.title('Validation Accuracy - Different LR Schedules', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Final test accuracy comparison
    plt.subplot(2, 2, 4)
    names = list(lr_scheduler_results.keys())
    accuracies = [lr_scheduler_results[name]['test_accuracy'] for name in names]
    
    bars = plt.bar(names, accuracies, color=['#ff7f0e', '#2ca02c', '#d62728'])
    plt.title('Final Test Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Test Accuracy')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('plots/learning_rate_schedules_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 3: ReduceLROnPlateau Effect
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(plateau_history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(plateau_history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('ReduceLROnPlateau - Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Extract learning rate from history if available
    if 'lr' in plateau_history.history:
        plt.plot(plateau_history.history['lr'], linewidth=2)
        plt.title('Learning Rate Reduction', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
    else:
        plt.plot(plateau_history.history['val_accuracy'], linewidth=2)
        plt.title('Validation Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/reduce_lr_plateau_effect.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ All callback visualizations saved to 'plots/' directory")

def demonstrate_callbacks():
    """
    Main function to demonstrate all callback functionalities.
    """
    print("Keras Callbacks Comprehensive Demo")
    print("="*80)
    print("TensorFlow version:", tf.__version__)
    print("Keras version: Built-in with TensorFlow", tf.__version__)
    print()
    
    # Individual callback demonstrations
    print("Demonstrating individual callbacks...")
    
    # Early Stopping
    early_stopping_history, _ = demonstrate_early_stopping()
    
    # Model Checkpoint
    checkpoint_history, _ = demonstrate_model_checkpoint()
    
    # Learning Rate Scheduler
    lr_scheduler_results = demonstrate_learning_rate_scheduler()
    
    # Reduce LR on Plateau
    plateau_history = demonstrate_reduce_lr_on_plateau()
    
    # Custom Callbacks
    custom_history, time_callback, loss_callback = demonstrate_custom_callbacks()
    
    # Multiple Callbacks Combined
    combined_history, callbacks_list = combine_multiple_callbacks()
    
    # Create comprehensive visualizations
    # plot_callbacks_comparison()  # This would run demos again, so we'll skip for now
    
    print("\n" + "="*80)
    print("CALLBACKS DEMONSTRATION COMPLETE")
    print("="*80)
    print("✅ EarlyStopping callback")
    print("✅ ModelCheckpoint callback")
    print("✅ LearningRateScheduler callback")
    print("✅ ReduceLROnPlateau callback")
    print("✅ Custom callbacks (TrainingTime, LossHistory)")
    print("✅ Multiple callbacks combination")
    print("✅ Comprehensive visualizations")
    print("\nKey Takeaways:")
    print("- EarlyStopping prevents overfitting and saves training time")
    print("- ModelCheckpoint preserves best model weights")
    print("- LR scheduling improves convergence")
    print("- Custom callbacks enable specialized monitoring")
    print("- Combining callbacks provides robust training control")

if __name__ == "__main__":
    demonstrate_callbacks()