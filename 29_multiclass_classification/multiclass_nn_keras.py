#!/usr/bin/env python3
"""
Multiclass Neural Networks with Keras/TensorFlow
================================================

This module demonstrates multiclass classification using neural networks
with proper softmax activation and categorical crossentropy loss.

Key Concepts:
- Softmax activation for multiclass output
- Categorical vs Sparse Categorical Crossentropy
- One-hot vs label encoding comparison
- Network architecture impact on multiclass performance
- Class probability interpretation and calibration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import to_categorical, plot_model
from sklearn.datasets import load_iris, load_digits, load_wine, make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, log_loss
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MulticlassNeuralNetAnalyzer:
    """
    Comprehensive analyzer for multiclass neural networks using Keras
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
    def load_and_prepare_datasets(self):
        """Load and prepare datasets for neural network training"""
        datasets = {}
        
        # 1. Iris dataset (3 classes)
        iris = load_iris()
        datasets['iris'] = {
            'X': iris.data,
            'y': iris.target,
            'target_names': iris.target_names,
            'feature_names': iris.feature_names,
            'n_classes': 3,
            'input_dim': iris.data.shape[1]
        }
        
        # 2. Wine dataset (3 classes)
        wine = load_wine()
        datasets['wine'] = {
            'X': wine.data,
            'y': wine.target,
            'target_names': wine.target_names,
            'feature_names': wine.feature_names,
            'n_classes': 3,
            'input_dim': wine.data.shape[1]
        }
        
        # 3. Digits dataset (10 classes) - subset for faster training
        digits = load_digits()
        # Take subset for faster training
        subset_size = 1000
        indices = np.random.choice(len(digits.data), subset_size, replace=False)
        datasets['digits'] = {
            'X': digits.data[indices],
            'y': digits.target[indices],
            'target_names': [str(i) for i in range(10)],
            'feature_names': [f'pixel_{i}' for i in range(64)],
            'n_classes': 10,
            'input_dim': digits.data.shape[1]
        }
        
        # 4. Synthetic multiclass dataset
        X_synth, y_synth = make_classification(
            n_samples=3000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_classes=6,
            n_clusters_per_class=1,
            weights=[0.3, 0.2, 0.2, 0.15, 0.1, 0.05],  # Somewhat imbalanced
            random_state=self.random_state
        )
        datasets['synthetic'] = {
            'X': X_synth,
            'y': y_synth,
            'target_names': [f'Class_{i}' for i in range(6)],
            'feature_names': [f'feature_{i}' for i in range(20)],
            'n_classes': 6,
            'input_dim': 20
        }
        
        return datasets
    
    def create_neural_networks(self, input_dim, n_classes):
        """Create different neural network architectures for comparison"""
        network_models = {}
        
        # 1. Simple Network
        simple_model = tf.keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(n_classes, activation='softmax')
        ])
        network_models['Simple'] = simple_model
        
        # 2. Deep Network
        deep_model = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(n_classes, activation='softmax')
        ])
        network_models['Deep'] = deep_model
        
        # 3. Wide Network
        wide_model = tf.keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(n_classes, activation='softmax')
        ])
        network_models['Wide'] = wide_model
        
        # 4. Functional API with skip connections (for larger datasets)
        if input_dim >= 10:
            input_layer = layers.Input(shape=(input_dim,))
            x1 = layers.Dense(64, activation='relu')(input_layer)
            x1 = layers.Dropout(0.3)(x1)
            x2 = layers.Dense(32, activation='relu')(x1)
            x2 = layers.Dropout(0.3)(x2)
            
            # Skip connection
            concat = layers.Concatenate()([x1, x2])
            x3 = layers.Dense(32, activation='relu')(concat)
            x3 = layers.Dropout(0.2)(x3)
            output = layers.Dense(n_classes, activation='softmax')(x3)
            
            functional_model = tf.keras.Model(inputs=input_layer, outputs=output)
            network_models['Functional_Skip'] = functional_model
        
        return network_models
    
    def compare_encoding_strategies(self, X, y, n_classes, dataset_name):
        """Compare one-hot vs sparse categorical encoding"""
        print(f"\nComparing Encoding Strategies on {dataset_name}")
        print("-" * 50)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Prepare labels for different encoding strategies
        y_train_onehot = to_categorical(y_train, num_classes=n_classes)
        y_test_onehot = to_categorical(y_test, num_classes=n_classes)
        
        # Create model for each encoding strategy
        input_dim = X_train_scaled.shape[1]
        
        # Model with one-hot encoding
        model_onehot = tf.keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(n_classes, activation='softmax')
        ])
        
        # Model with sparse categorical encoding
        model_sparse = tf.keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(n_classes, activation='softmax')
        ])
        
        # Compile models with different loss functions
        model_onehot.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model_sparse.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
        )
        
        # Train models
        print("Training with one-hot encoding...")
        history_onehot = model_onehot.fit(
            X_train_scaled, y_train_onehot,
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        print("Training with sparse categorical encoding...")
        history_sparse = model_sparse.fit(
            X_train_scaled, y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Evaluate models
        results = {}
        
        # One-hot encoding results
        pred_onehot = model_onehot.predict(X_test_scaled, verbose=0)
        pred_onehot_classes = np.argmax(pred_onehot, axis=1)
        
        results['One-Hot Encoding'] = {
            'model': model_onehot,
            'history': history_onehot,
            'predictions': pred_onehot,
            'predicted_classes': pred_onehot_classes,
            'accuracy': accuracy_score(y_test, pred_onehot_classes),
            'f1_macro': f1_score(y_test, pred_onehot_classes, average='macro'),
            'f1_micro': f1_score(y_test, pred_onehot_classes, average='micro'),
            'log_loss': log_loss(y_test_onehot, pred_onehot)
        }
        
        # Sparse categorical encoding results
        pred_sparse = model_sparse.predict(X_test_scaled, verbose=0)
        pred_sparse_classes = np.argmax(pred_sparse, axis=1)
        
        results['Sparse Categorical'] = {
            'model': model_sparse,
            'history': history_sparse,
            'predictions': pred_sparse,
            'predicted_classes': pred_sparse_classes,
            'accuracy': accuracy_score(y_test, pred_sparse_classes),
            'f1_macro': f1_score(y_test, pred_sparse_classes, average='macro'),
            'f1_micro': f1_score(y_test, pred_sparse_classes, average='micro'),
            'log_loss': log_loss(to_categorical(y_test, n_classes), pred_sparse)
        }
        
        # Print results
        for encoding, result in results.items():
            print(f"\n{encoding}:")
            print(f"  Accuracy: {result['accuracy']:.4f}")
            print(f"  Macro F1: {result['f1_macro']:.4f}")
            print(f"  Micro F1: {result['f1_micro']:.4f}")
            print(f"  Log Loss: {result['log_loss']:.4f}")
        
        return results, (X_test_scaled, y_test)
    
    def compare_network_architectures(self, X, y, n_classes, dataset_name):
        """Compare different network architectures"""
        print(f"\nComparing Network Architectures on {dataset_name}")
        print("-" * 50)
        
        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        input_dim = X_train_scaled.shape[1]
        
        # Create models
        network_models = self.create_neural_networks(input_dim, n_classes)
        
        results = {}
        
        for model_name, model in network_models.items():
            print(f"Training {model_name} network...")
            
            # Compile model
            model.compile(
                optimizer=optimizers.Adam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Callbacks
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss', patience=15, restore_best_weights=True
            )
            
            reduce_lr = callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6
            )
            
            # Train model
            history = model.fit(
                X_train_scaled, y_train,
                validation_split=0.2,
                epochs=100,
                batch_size=32,
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # Evaluate model
            predictions = model.predict(X_test_scaled, verbose=0)
            predicted_classes = np.argmax(predictions, axis=1)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, predicted_classes)
            f1_macro = f1_score(y_test, predicted_classes, average='macro')
            f1_micro = f1_score(y_test, predicted_classes, average='micro')
            
            # Model complexity
            total_params = model.count_params()
            trainable_params = sum([np.prod(w.shape) for w in model.trainable_weights])
            
            results[model_name] = {
                'model': model,
                'history': history,
                'predictions': predictions,
                'predicted_classes': predicted_classes,
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'f1_micro': f1_micro,
                'total_params': total_params,
                'trainable_params': trainable_params,
                'epochs_trained': len(history.history['loss'])
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Macro F1: {f1_macro:.4f}")
            print(f"  Parameters: {total_params:,}")
            print(f"  Epochs trained: {len(history.history['loss'])}")
        
        return results, (X_test_scaled, y_test)
    
    def analyze_class_probabilities(self, models_results, y_test, class_names, dataset_name):
        """Analyze class probability distributions and calibration"""
        print(f"\nAnalyzing Class Probabilities for {dataset_name}")
        print("-" * 50)
        
        n_models = len(models_results)
        fig, axes = plt.subplots(2, n_models, figsize=(5*n_models, 10))
        if n_models == 1:
            axes = axes.reshape(-1, 1)
        
        for col, (model_name, result) in enumerate(models_results.items()):
            predictions = result['predictions']
            predicted_classes = result['predicted_classes']
            
            # 1. Probability distribution by class
            ax1 = axes[0, col]
            max_probs = np.max(predictions, axis=1)
            correct_mask = predicted_classes == y_test
            
            # Plot histograms for correct and incorrect predictions
            ax1.hist(max_probs[correct_mask], bins=20, alpha=0.7, 
                    label='Correct', color='green', density=True)
            ax1.hist(max_probs[~correct_mask], bins=20, alpha=0.7, 
                    label='Incorrect', color='red', density=True)
            ax1.set_xlabel('Maximum Probability')
            ax1.set_ylabel('Density')
            ax1.set_title(f'{model_name}\\nProbability Distribution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Calibration plot
            ax2 = axes[1, col]
            n_bins = 10
            
            # Calculate calibration for each class and average
            calibration_scores = []
            for class_idx in range(len(class_names)):
                if class_idx in y_test:  # Only if class exists in test set
                    y_binary = (y_test == class_idx).astype(int)
                    prob_class = predictions[:, class_idx]
                    
                    fraction_of_positives, mean_predicted_value = calibration_curve(
                        y_binary, prob_class, n_bins=n_bins
                    )
                    
                    ax2.plot(mean_predicted_value, fraction_of_positives, 
                            marker='o', alpha=0.7, label=f'Class {class_names[class_idx]}'[:10])
            
            # Perfect calibration line
            ax2.plot([0, 1], [0, 1], 'k--', alpha=0.8, label='Perfect Calibration')
            ax2.set_xlabel('Mean Predicted Probability')
            ax2.set_ylabel('Fraction of Positives')
            ax2.set_title(f'{model_name}\\nCalibration Plot')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'plots/{dataset_name}_probability_analysis.png',
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_training_dynamics(self, models_results, dataset_name):
        """Visualize training dynamics for different models"""
        n_models = len(models_results)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Training/Validation Loss
        ax1 = axes[0, 0]
        for model_name, result in models_results.items():
            history = result['history']
            epochs = range(1, len(history.history['loss']) + 1)
            ax1.plot(epochs, history.history['loss'], label=f'{model_name} Train', alpha=0.8)
            ax1.plot(epochs, history.history['val_loss'], '--', label=f'{model_name} Val', alpha=0.8)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training/Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Training/Validation Accuracy
        ax2 = axes[0, 1]
        for model_name, result in models_results.items():
            history = result['history']
            epochs = range(1, len(history.history['accuracy']) + 1)
            ax2.plot(epochs, history.history['accuracy'], label=f'{model_name} Train', alpha=0.8)
            ax2.plot(epochs, history.history['val_accuracy'], '--', label=f'{model_name} Val', alpha=0.8)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training/Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Final Performance Comparison
        ax3 = axes[1, 0]
        model_names = list(models_results.keys())
        accuracies = [models_results[name]['accuracy'] for name in model_names]
        f1_scores = [models_results[name]['f1_macro'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
        bars2 = ax3.bar(x + width/2, f1_scores, width, label='Macro F1', alpha=0.8)
        
        ax3.set_xlabel('Model')
        ax3.set_ylabel('Score')
        ax3.set_title('Final Performance Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(model_names, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
        
        # 4. Model Complexity vs Performance
        ax4 = axes[1, 1]
        params = [models_results[name]['total_params'] for name in model_names]
        
        scatter = ax4.scatter(params, accuracies, c=f1_scores, s=100, alpha=0.7, cmap='viridis')
        
        for i, name in enumerate(model_names):
            ax4.annotate(name, (params[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        ax4.set_xlabel('Number of Parameters')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Model Complexity vs Performance')
        ax4.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Macro F1 Score')
        
        plt.tight_layout()
        plt.savefig(f'plots/{dataset_name}_training_dynamics.png',
                   dpi=300, bbox_inches='tight')
        plt.show()

def run_comprehensive_neural_network_analysis():
    """Run comprehensive multiclass neural network analysis"""
    print("Multiclass Neural Networks Comprehensive Analysis")
    print("=" * 60)
    
    analyzer = MulticlassNeuralNetAnalyzer()
    
    # Load datasets
    datasets = analyzer.load_and_prepare_datasets()
    
    # Analysis results storage
    all_results = {}
    
    for dataset_name, data in datasets.items():
        print(f"\n{'='*60}")
        print(f"ANALYZING DATASET: {dataset_name.upper()}")
        print(f"{'='*60}")
        
        X, y = data['X'], data['y']
        n_classes = data['n_classes']
        class_names = data['target_names']
        
        # 1. Compare encoding strategies
        encoding_results, test_data = analyzer.compare_encoding_strategies(
            X, y, n_classes, dataset_name
        )
        
        # 2. Compare network architectures
        architecture_results, test_data = analyzer.compare_network_architectures(
            X, y, n_classes, dataset_name
        )
        
        # 3. Analyze class probabilities
        analyzer.analyze_class_probabilities(
            architecture_results, test_data[1], class_names, dataset_name
        )
        
        # 4. Visualize training dynamics
        analyzer.visualize_training_dynamics(architecture_results, dataset_name)
        
        all_results[dataset_name] = {
            'encoding': encoding_results,
            'architecture': architecture_results,
            'test_data': test_data
        }
    
    # Create summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON ACROSS ALL DATASETS")
    print(f"{'='*60}")
    
    # Collect summary data
    summary_data = []
    for dataset_name, results in all_results.items():
        # Architecture results
        for model_name, result in results['architecture'].items():
            summary_data.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Accuracy': result['accuracy'],
                'Macro F1': result['f1_macro'],
                'Micro F1': result['f1_micro'],
                'Parameters': result['total_params'],
                'Epochs': result['epochs_trained']
            })
    
    summary_df = pd.DataFrame(summary_data)
    print("\nArchitecture Summary Results:")
    print(summary_df.round(4))
    
    # Create overall summary visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Performance by dataset
    ax1 = axes[0, 0]
    for dataset in summary_df['Dataset'].unique():
        dataset_data = summary_df[summary_df['Dataset'] == dataset]
        ax1.plot(dataset_data['Model'], dataset_data['Accuracy'], 
                marker='o', label=dataset, alpha=0.8)
    
    ax1.set_xlabel('Model Architecture')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy by Model Architecture')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.get_xticklabels(), rotation=45)
    
    # Efficiency analysis
    ax2 = axes[0, 1]
    scatter = ax2.scatter(summary_df['Parameters'], summary_df['Accuracy'], 
                         c=summary_df['Epochs'], s=60, alpha=0.7, cmap='plasma')
    ax2.set_xlabel('Number of Parameters')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Model Efficiency Analysis')
    ax2.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Training Epochs')
    
    # Best model per dataset
    ax3 = axes[1, 0]
    best_models = summary_df.loc[summary_df.groupby('Dataset')['Accuracy'].idxmax()]
    bars = ax3.bar(best_models['Dataset'], best_models['Accuracy'], alpha=0.8)
    ax3.set_xlabel('Dataset')
    ax3.set_ylabel('Best Accuracy')
    ax3.set_title('Best Model Performance per Dataset')
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.get_xticklabels(), rotation=45)
    
    # Add model names on bars
    for bar, model in zip(bars, best_models['Model']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                model, ha='center', va='bottom', rotation=45, fontsize=8)
    
    # F1 Score comparison
    ax4 = axes[1, 1]
    summary_pivot = summary_df.pivot_table(
        values='Macro F1', index='Dataset', columns='Model', aggfunc='mean'
    )
    sns.heatmap(summary_pivot, annot=True, fmt='.3f', cmap='viridis', ax=ax4)
    ax4.set_title('Macro F1 Score Heatmap')
    
    plt.tight_layout()
    plt.savefig('plots/neural_network_summary.png',
               dpi=300, bbox_inches='tight')
    plt.show()
    
    return all_results, summary_df

if __name__ == "__main__":
    results, summary = run_comprehensive_neural_network_analysis()
    print("\nNeural network analysis complete! Check the plots folder for visualizations.")