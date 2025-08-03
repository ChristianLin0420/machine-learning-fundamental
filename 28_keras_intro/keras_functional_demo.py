"""
Day 28 - Keras Functional API Demo
=================================

This module demonstrates the Keras Functional API for building complex neural networks.
We'll create models with branching, merging, and custom topologies.

Author: ML Fundamentals Course
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_plots_dir():
    """Create plots directory if it doesn't exist."""
    if not os.path.exists('plots'):
        os.makedirs('plots')

def functional_simple_model():
    """
    Create a simple model using Functional API equivalent to Sequential.
    """
    print("="*60)
    print("FUNCTIONAL API - SIMPLE MODEL")
    print("="*60)
    
    # Define inputs
    input_layer = tf.keras.Input(shape=(2,), name='input')
    
    # Define layers
    hidden1 = tf.keras.layers.Dense(16, activation='relu', name='hidden_1')(input_layer)
    hidden2 = tf.keras.layers.Dense(8, activation='relu', name='hidden_2')(hidden1)
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(hidden2)
    
    # Create model
    model = tf.keras.Model(inputs=input_layer, outputs=output, name='Functional_Simple_Model')
    
    print("Simple Functional Model:")
    model.summary()
    
    return model

def functional_branching_model():
    """
    Create a model with branching architecture using Functional API.
    """
    print("\n" + "="*60)
    print("FUNCTIONAL API - BRANCHING MODEL")
    print("="*60)
    
    # Define inputs
    input_layer = tf.keras.Input(shape=(4,), name='input')
    
    # Branch 1: Dense path
    branch1 = tf.keras.layers.Dense(32, activation='relu', name='branch1_dense1')(input_layer)
    branch1 = tf.keras.layers.Dropout(0.2, name='branch1_dropout1')(branch1)
    branch1 = tf.keras.layers.Dense(16, activation='relu', name='branch1_dense2')(branch1)
    
    # Branch 2: Different dense path
    branch2 = tf.keras.layers.Dense(24, activation='tanh', name='branch2_dense1')(input_layer)
    branch2 = tf.keras.layers.Dropout(0.1, name='branch2_dropout1')(branch2)
    branch2 = tf.keras.layers.Dense(12, activation='tanh', name='branch2_dense2')(branch2)
    
    # Branch 3: Skip connection
    branch3 = tf.keras.layers.Dense(8, activation='relu', name='branch3_dense1')(input_layer)
    
    # Merge branches
    merged = tf.keras.layers.Concatenate(name='merge_branches')([branch1, branch2, branch3])
    
    # Final layers
    final = tf.keras.layers.Dense(16, activation='relu', name='final_dense')(merged)
    final = tf.keras.layers.Dropout(0.2, name='final_dropout')(final)
    output = tf.keras.layers.Dense(3, activation='softmax', name='output')(final)
    
    # Create model
    model = tf.keras.Model(inputs=input_layer, outputs=output, name='Functional_Branching_Model')
    
    print("Branching Functional Model:")
    model.summary()
    
    return model

def functional_multi_input_output():
    """
    Create a model with multiple inputs and outputs using Functional API.
    """
    print("\n" + "="*60)
    print("FUNCTIONAL API - MULTI-INPUT/OUTPUT MODEL")
    print("="*60)
    
    # Main input
    main_input = tf.keras.Input(shape=(10,), name='main_input')
    
    # Auxiliary input
    aux_input = tf.keras.Input(shape=(5,), name='auxiliary_input')
    
    # Process main input
    main_branch = tf.keras.layers.Dense(32, activation='relu', name='main_dense1')(main_input)
    main_branch = tf.keras.layers.Dense(16, activation='relu', name='main_dense2')(main_branch)
    
    # Process auxiliary input
    aux_branch = tf.keras.layers.Dense(16, activation='relu', name='aux_dense1')(aux_input)
    aux_branch = tf.keras.layers.Dense(8, activation='relu', name='aux_dense2')(aux_branch)
    
    # Merge inputs
    merged = tf.keras.layers.Concatenate(name='merge_inputs')([main_branch, aux_branch])
    
    # Shared layers
    shared = tf.keras.layers.Dense(24, activation='relu', name='shared_dense1')(merged)
    shared = tf.keras.layers.Dropout(0.2, name='shared_dropout')(shared)
    shared = tf.keras.layers.Dense(12, activation='relu', name='shared_dense2')(shared)
    
    # Main output
    main_output = tf.keras.layers.Dense(1, activation='sigmoid', name='main_output')(shared)
    
    # Auxiliary output (for regularization)
    aux_output = tf.keras.layers.Dense(1, activation='sigmoid', name='auxiliary_output')(aux_branch)
    
    # Create model
    model = tf.keras.Model(
        inputs=[main_input, aux_input], 
        outputs=[main_output, aux_output], 
        name='Multi_Input_Output_Model'
    )
    
    print("Multi-Input/Output Functional Model:")
    model.summary()
    
    return model

def functional_residual_model():
    """
    Create a model with residual connections using Functional API.
    """
    print("\n" + "="*60)
    print("FUNCTIONAL API - RESIDUAL MODEL")
    print("="*60)
    
    # Input
    input_layer = tf.keras.Input(shape=(64,), name='input')
    
    # First block
    x = tf.keras.layers.Dense(64, activation='relu', name='block1_dense1')(input_layer)
    x = tf.keras.layers.Dense(64, activation='relu', name='block1_dense2')(x)
    
    # Residual connection 1
    residual1 = tf.keras.layers.Add(name='residual1')([input_layer, x])
    
    # Second block
    x = tf.keras.layers.Dense(64, activation='relu', name='block2_dense1')(residual1)
    x = tf.keras.layers.Dense(64, activation='relu', name='block2_dense2')(x)
    
    # Residual connection 2
    residual2 = tf.keras.layers.Add(name='residual2')([residual1, x])
    
    # Final layers
    x = tf.keras.layers.Dense(32, activation='relu', name='final_dense1')(residual2)
    x = tf.keras.layers.Dropout(0.2, name='final_dropout')(x)
    output = tf.keras.layers.Dense(2, activation='softmax', name='output')(x)
    
    # Create model
    model = tf.keras.Model(inputs=input_layer, outputs=output, name='Residual_Model')
    
    print("Residual Functional Model:")
    model.summary()
    
    return model

def train_and_compare_models():
    """
    Train and compare different functional API models.
    """
    print("\n" + "="*80)
    print("TRAINING AND COMPARING FUNCTIONAL MODELS")
    print("="*80)
    
    # Generate datasets
    print("Generating datasets...")
    
    # Binary classification dataset
    X_binary, y_binary = make_moons(n_samples=1000, noise=0.2, random_state=42)
    
    # Multiclass dataset for branching model
    X_multi, y_multi = make_classification(
        n_samples=1000, n_features=4, n_redundant=0, n_informative=4,
        n_clusters_per_class=1, n_classes=3, random_state=42
    )
    
    # Multi-input dataset
    X_main = np.random.randn(1000, 10)
    X_aux = np.random.randn(1000, 5)
    y_multi_output = np.random.randint(0, 2, (1000, 1))
    
    # High-dimensional dataset for residual model
    X_residual, y_residual = make_classification(
        n_samples=1000, n_features=64, n_redundant=0, n_informative=32,
        n_clusters_per_class=1, n_classes=2, random_state=42
    )
    
    models_results = {}
    
    # 1. Simple model
    print("\n1. Training Simple Functional Model...")
    simple_model = functional_simple_model()
    simple_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    simple_history = simple_model.fit(
        X_train_scaled, y_train, epochs=50, batch_size=32,
        validation_data=(X_test_scaled, y_test), verbose=0
    )
    simple_score = simple_model.evaluate(X_test_scaled, y_test, verbose=0)
    models_results['Simple'] = {'history': simple_history, 'score': simple_score}
    print(f"Simple Model - Test Loss: {simple_score[0]:.4f}, Test Accuracy: {simple_score[1]:.4f}")
    
    # 2. Branching model
    print("\n2. Training Branching Functional Model...")
    branching_model = functional_branching_model()
    branching_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    X_train, X_test, y_train, y_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    branching_history = branching_model.fit(
        X_train_scaled, y_train, epochs=50, batch_size=32,
        validation_data=(X_test_scaled, y_test), verbose=0
    )
    branching_score = branching_model.evaluate(X_test_scaled, y_test, verbose=0)
    models_results['Branching'] = {'history': branching_history, 'score': branching_score}
    print(f"Branching Model - Test Loss: {branching_score[0]:.4f}, Test Accuracy: {branching_score[1]:.4f}")
    
    # 3. Multi-input/output model
    print("\n3. Training Multi-Input/Output Functional Model...")
    multi_model = functional_multi_input_output()
    multi_model.compile(
        optimizer='adam',
        loss={'main_output': 'binary_crossentropy', 'auxiliary_output': 'binary_crossentropy'},
        loss_weights={'main_output': 1.0, 'auxiliary_output': 0.5},
        metrics=['accuracy']
    )
    
    X_main_train, X_main_test, X_aux_train, X_aux_test, y_train, y_test = train_test_split(
        X_main, X_aux, y_multi_output, test_size=0.2, random_state=42
    )
    
    multi_history = multi_model.fit(
        [X_main_train, X_aux_train], [y_train, y_train],
        epochs=50, batch_size=32,
        validation_data=([X_main_test, X_aux_test], [y_test, y_test]),
        verbose=0
    )
    multi_score = multi_model.evaluate([X_main_test, X_aux_test], [y_test, y_test], verbose=0)
    models_results['Multi-IO'] = {'history': multi_history, 'score': multi_score}
    print(f"Multi-I/O Model - Test Loss: {multi_score[0]:.4f}")
    
    # 4. Residual model
    print("\n4. Training Residual Functional Model...")
    residual_model = functional_residual_model()
    residual_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    X_train, X_test, y_train, y_test = train_test_split(X_residual, y_residual, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    residual_history = residual_model.fit(
        X_train_scaled, y_train, epochs=50, batch_size=32,
        validation_data=(X_test_scaled, y_test), verbose=0
    )
    residual_score = residual_model.evaluate(X_test_scaled, y_test, verbose=0)
    models_results['Residual'] = {'history': residual_history, 'score': residual_score}
    print(f"Residual Model - Test Loss: {residual_score[0]:.4f}, Test Accuracy: {residual_score[1]:.4f}")
    
    return models_results

def visualize_model_architectures():
    """
    Visualize model architectures using plot_model.
    """
    print("\n" + "="*60)
    print("VISUALIZING MODEL ARCHITECTURES")
    print("="*60)
    
    create_plots_dir()
    
    # Simple model
    simple_model = functional_simple_model()
    tf.keras.utils.plot_model(
        simple_model, to_file='plots/functional_simple_model.png',
        show_shapes=True, show_layer_names=True, rankdir='TB'
    )
    print("✅ Simple model architecture saved")
    
    # Branching model
    branching_model = functional_branching_model()
    tf.keras.utils.plot_model(
        branching_model, to_file='plots/functional_branching_model.png',
        show_shapes=True, show_layer_names=True, rankdir='TB'
    )
    print("✅ Branching model architecture saved")
    
    # Multi-input/output model
    multi_model = functional_multi_input_output()
    tf.keras.utils.plot_model(
        multi_model, to_file='plots/functional_multi_io_model.png',
        show_shapes=True, show_layer_names=True, rankdir='TB'
    )
    print("✅ Multi-I/O model architecture saved")
    
    # Residual model
    residual_model = functional_residual_model()
    tf.keras.utils.plot_model(
        residual_model, to_file='plots/functional_residual_model.png',
        show_shapes=True, show_layer_names=True, rankdir='TB'
    )
    print("✅ Residual model architecture saved")

def plot_functional_comparison(models_results):
    """
    Plot comparison of different functional API models.
    """
    create_plots_dir()
    
    # Extract models that have standard metrics
    standard_models = {name: result for name, result in models_results.items() 
                      if name in ['Simple', 'Branching', 'Residual']}
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Training loss
    for name, result in standard_models.items():
        axes[0, 0].plot(result['history'].history['loss'], 
                       label=f'{name}', linewidth=2)
    axes[0, 0].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation loss
    for name, result in standard_models.items():
        axes[0, 1].plot(result['history'].history['val_loss'], 
                       label=f'{name}', linewidth=2)
    axes[0, 1].set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Training accuracy
    for name, result in standard_models.items():
        axes[1, 0].plot(result['history'].history['accuracy'], 
                       label=f'{name}', linewidth=2)
    axes[1, 0].set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Validation accuracy
    for name, result in standard_models.items():
        axes[1, 1].plot(result['history'].history['val_accuracy'], 
                       label=f'{name}', linewidth=2)
    axes[1, 1].set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Functional API Models Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/functional_models_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_sequential_vs_functional():
    """
    Compare Sequential vs Functional API for the same architecture.
    """
    print("\n" + "="*80)
    print("SEQUENTIAL VS FUNCTIONAL API COMPARISON")
    print("="*80)
    
    # Generate dataset
    X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Sequential model
    sequential_model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(2,)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ], name='Sequential_Model')
    
    # Functional model (same architecture)
    input_layer = tf.keras.Input(shape=(2,))
    x = tf.keras.layers.Dense(16, activation='relu')(input_layer)
    x = tf.keras.layers.Dense(8, activation='relu')(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    functional_model = tf.keras.Model(inputs=input_layer, outputs=output, name='Functional_Model')
    
    # Compile both models
    sequential_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    functional_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("\nSequential Model:")
    sequential_model.summary()
    print("\nFunctional Model:")
    functional_model.summary()
    
    # Train both models
    print("\nTraining Sequential Model...")
    seq_history = sequential_model.fit(
        X_train_scaled, y_train, epochs=50, batch_size=32,
        validation_data=(X_test_scaled, y_test), verbose=0
    )
    
    print("Training Functional Model...")
    func_history = functional_model.fit(
        X_train_scaled, y_train, epochs=50, batch_size=32,
        validation_data=(X_test_scaled, y_test), verbose=0
    )
    
    # Evaluate both models
    seq_score = sequential_model.evaluate(X_test_scaled, y_test, verbose=0)
    func_score = functional_model.evaluate(X_test_scaled, y_test, verbose=0)
    
    print(f"\nResults:")
    print(f"Sequential - Test Loss: {seq_score[0]:.4f}, Test Accuracy: {seq_score[1]:.4f}")
    print(f"Functional - Test Loss: {func_score[0]:.4f}, Test Accuracy: {func_score[1]:.4f}")
    
    # Plot comparison
    create_plots_dir()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss comparison
    axes[0].plot(seq_history.history['val_loss'], label='Sequential', linewidth=2)
    axes[0].plot(func_history.history['val_loss'], label='Functional', linewidth=2)
    axes[0].set_title('Validation Loss: Sequential vs Functional', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy comparison
    axes[1].plot(seq_history.history['val_accuracy'], label='Sequential', linewidth=2)
    axes[1].plot(func_history.history['val_accuracy'], label='Functional', linewidth=2)
    axes[1].set_title('Validation Accuracy: Sequential vs Functional', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/sequential_vs_functional_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return seq_history, func_history, seq_score, func_score

def demonstrate_functional_api():
    """
    Main function to demonstrate Functional API capabilities.
    """
    print("Keras Functional API Comprehensive Demo")
    print("="*80)
    print("TensorFlow version:", tf.__version__)
    print("Keras version: Built-in with TensorFlow", tf.__version__)
    print()
    
    # Demonstrate different model architectures
    print("Creating various functional API models...")
    
    # Simple model
    simple_model = functional_simple_model()
    
    # Branching model
    branching_model = functional_branching_model()
    
    # Multi-input/output model
    multi_model = functional_multi_input_output()
    
    # Residual model
    residual_model = functional_residual_model()
    
    # Visualize architectures
    try:
        visualize_model_architectures()
    except Exception as e:
        print(f"Note: Model visualization requires pydot and graphviz: {e}")
    
    # Train and compare models
    models_results = train_and_compare_models()
    
    # Plot comparison
    plot_functional_comparison(models_results)
    
    # Compare Sequential vs Functional
    compare_sequential_vs_functional()
    
    print("\n" + "="*80)
    print("FUNCTIONAL API DEMONSTRATION COMPLETE")
    print("="*80)
    print("✅ Simple functional model")
    print("✅ Branching functional model")
    print("✅ Multi-input/output functional model")
    print("✅ Residual functional model")
    print("✅ Model architecture visualization")
    print("✅ Performance comparison")
    print("✅ Sequential vs Functional comparison")
    print("\nAll plots saved to 'plots/' directory")

if __name__ == "__main__":
    demonstrate_functional_api()