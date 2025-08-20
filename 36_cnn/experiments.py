#!/usr/bin/env python3
"""
CNN Comprehensive Experiments
=============================

Comprehensive experiments comparing CNN implementations:
- NumPy vs PyTorch vs Keras
- Different architectures
- Various datasets
- Performance analysis and visualization

Author: ML Fundamentals Course  
Day: 36 - CNNs
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import pickle
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Import our implementations
from cnn_from_scratch import CNN as NumpyCNN, load_mnist_data, create_simple_dataset
from visualize_features import (
    FilterVisualizer, FeatureMapVisualizer, LayerActivationVisualizer,
    ConvolutionVisualizer, create_synthetic_images
)

# Framework imports
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from cnn_pytorch import SimpleCNN, DeepCNN, ResNetCNN, CNNTrainer, load_mnist_pytorch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    from cnn_keras import (
        create_simple_cnn, create_deep_cnn, create_functional_cnn,
        load_keras_mnist, visualize_keras_training
    )
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False


class CNNExperiments:
    """Comprehensive CNN experiments and analysis."""
    
    def __init__(self):
        self.results = defaultdict(dict)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if PYTORCH_AVAILABLE else 'cpu'
        
        # Create plots directory
        os.makedirs('plots', exist_ok=True)
        
        print(f"CNN Experiments initialized")
        if PYTORCH_AVAILABLE:
            print(f"PyTorch available - Device: {self.device}")
        if KERAS_AVAILABLE:
            print(f"Keras/TensorFlow available")
    
    def compare_frameworks_mnist(self) -> Dict[str, Any]:
        """Compare NumPy, PyTorch, and Keras on MNIST."""
        print("\nComparing Frameworks on MNIST...")
        print("=" * 40)
        
        results = {}
        
        # NumPy implementation
        print("\n1. Testing NumPy CNN...")
        try:
            X_train, X_test, y_train, y_test = create_simple_dataset()
            
            input_shape = X_train.shape[1:]  # (channels, height, width)
            num_classes = y_train.shape[1]
            
            numpy_cnn = NumpyCNN(input_shape, num_classes)
            
            start_time = time.time()
            numpy_history = numpy_cnn.fit(X_train[:400], y_train[:400], 
                                        X_test[:100], y_test[:100],
                                        epochs=5, learning_rate=0.01, batch_size=16)
            numpy_time = time.time() - start_time
            
            test_loss, test_acc = numpy_cnn.evaluate(X_test[:100], y_test[:100])
            
            results['NumPy'] = {
                'test_accuracy': test_acc,
                'test_loss': test_loss,
                'training_time': numpy_time,
                'history': numpy_history,
                'model': numpy_cnn
            }
            
            print(f"   Test Accuracy: {test_acc:.4f}")
            print(f"   Training Time: {numpy_time:.1f}s")
            
        except Exception as e:
            print(f"   NumPy CNN failed: {e}")
            results['NumPy'] = {'error': str(e)}
        
        # PyTorch implementation
        if PYTORCH_AVAILABLE:
            print("\n2. Testing PyTorch CNN...")
            try:
                # Use subset of MNIST for fair comparison
                train_loader, val_loader, test_loader = load_mnist_pytorch()
                
                # Create simple model
                pytorch_model = SimpleCNN(input_channels=1, num_classes=10)
                trainer = CNNTrainer(pytorch_model, self.device)
                
                start_time = time.time()
                pytorch_history = trainer.fit(train_loader, val_loader, epochs=5, verbose=False)
                pytorch_time = time.time() - start_time
                
                # Evaluate
                test_loss, test_acc = trainer.validate(test_loader, nn.CrossEntropyLoss())
                
                results['PyTorch'] = {
                    'test_accuracy': test_acc,
                    'test_loss': test_loss,
                    'training_time': pytorch_time,
                    'history': pytorch_history,
                    'model': pytorch_model
                }
                
                print(f"   Test Accuracy: {test_acc:.4f}")
                print(f"   Training Time: {pytorch_time:.1f}s")
                
            except Exception as e:
                print(f"   PyTorch CNN failed: {e}")
                results['PyTorch'] = {'error': str(e)}
        
        # Keras implementation
        if KERAS_AVAILABLE:
            print("\n3. Testing Keras CNN...")
            try:
                X_train, X_val, X_test, y_train, y_val, y_test = load_keras_mnist()
                
                # Use subset for fair comparison
                X_train_sub = X_train[:5000]
                y_train_sub = y_train[:5000]
                X_val_sub = X_val[:1000]
                y_val_sub = y_val[:1000]
                X_test_sub = X_test[:1000]
                y_test_sub = y_test[:1000]
                
                keras_model = create_simple_cnn((28, 28, 1), 10)
                
                start_time = time.time()
                keras_history = keras_model.fit(
                    X_train_sub, y_train_sub,
                    validation_data=(X_val_sub, y_val_sub),
                    epochs=5, batch_size=32, verbose=0
                )
                keras_time = time.time() - start_time
                
                # Evaluate
                test_loss, test_acc = keras_model.evaluate(X_test_sub, y_test_sub, verbose=0)
                
                results['Keras'] = {
                    'test_accuracy': test_acc,
                    'test_loss': test_loss,
                    'training_time': keras_time,
                    'history': keras_history,
                    'model': keras_model
                }
                
                print(f"   Test Accuracy: {test_acc:.4f}")
                print(f"   Training Time: {keras_time:.1f}s")
                
            except Exception as e:
                print(f"   Keras CNN failed: {e}")
                results['Keras'] = {'error': str(e)}
        
        return results
    
    def analyze_cnn_components(self) -> Dict[str, Any]:
        """Analyze individual CNN components."""
        print("\nAnalyzing CNN Components...")
        print("=" * 40)
        
        results = {}
        
        # Create test data
        synthetic_images = create_synthetic_images()
        
        # Test different filter types
        print("1. Analyzing filter responses...")
        
        filters = {
            'Edge Detector': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
            'Blur Filter': np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
            'Sharpening': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
            'Ridge Detector': np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]])
        }
        
        filter_responses = {}
        
        for filter_name, filter_kernel in filters.items():
            responses = {}
            for img_name, img in synthetic_images.items():
                # Apply convolution (simplified)
                if img.shape[0] >= filter_kernel.shape[0] and img.shape[1] >= filter_kernel.shape[1]:
                    conv_result = self._apply_convolution_2d(img, filter_kernel)
                    responses[img_name] = {
                        'output': conv_result,
                        'max_response': np.max(conv_result),
                        'mean_response': np.mean(conv_result),
                        'response_variance': np.var(conv_result)
                    }
            
            filter_responses[filter_name] = responses
        
        results['filter_responses'] = filter_responses
        
        # Visualize filter responses
        self._visualize_filter_responses(synthetic_images, filters, filter_responses)
        
        # Test pooling effects
        print("2. Analyzing pooling effects...")
        pooling_analysis = self._analyze_pooling_effects(synthetic_images)
        results['pooling_analysis'] = pooling_analysis
        
        return results
    
    def _apply_convolution_2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Apply 2D convolution (simplified implementation)."""
        h_out = image.shape[0] - kernel.shape[0] + 1
        w_out = image.shape[1] - kernel.shape[1] + 1
        
        output = np.zeros((h_out, w_out))
        
        for h in range(h_out):
            for w in range(w_out):
                region = image[h:h+kernel.shape[0], w:w+kernel.shape[1]]
                output[h, w] = np.sum(region * kernel)
        
        return output
    
    def _visualize_filter_responses(self, images: Dict[str, np.ndarray], 
                                  filters: Dict[str, np.ndarray],
                                  responses: Dict[str, Dict[str, Any]]) -> None:
        """Visualize how different filters respond to different images."""
        fig, axes = plt.subplots(len(filters), len(images) + 1, 
                                figsize=(len(images) * 3 + 3, len(filters) * 2.5))
        
        if len(filters) == 1:
            axes = axes.reshape(1, -1)
        
        for i, (filter_name, filter_kernel) in enumerate(filters.items()):
            # Show filter
            im = axes[i, 0].imshow(filter_kernel, cmap='RdBu')
            axes[i, 0].set_title(f'{filter_name}\nFilter')
            axes[i, 0].axis('off')
            plt.colorbar(im, ax=axes[i, 0], fraction=0.046, pad=0.04)
            
            # Show responses to each image
            for j, (img_name, img) in enumerate(images.items()):
                if filter_name in responses and img_name in responses[filter_name]:
                    response = responses[filter_name][img_name]['output']
                    max_resp = responses[filter_name][img_name]['max_response']
                    
                    im = axes[i, j+1].imshow(response, cmap='viridis')
                    axes[i, j+1].set_title(f'{img_name}\nMax: {max_resp:.2f}')
                    axes[i, j+1].axis('off')
                    plt.colorbar(im, ax=axes[i, j+1], fraction=0.046, pad=0.04)
                else:
                    axes[i, j+1].axis('off')
        
        plt.suptitle('Filter Response Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig('plots/filter_response_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _analyze_pooling_effects(self, images: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Analyze effects of different pooling operations."""
        pooling_results = {}
        
        for img_name, img in images.items():
            results = {}
            
            # Apply different pooling operations
            for pool_size in [2, 3, 4]:
                h_out = img.shape[0] // pool_size
                w_out = img.shape[1] // pool_size
                
                max_pooled = np.zeros((h_out, w_out))
                avg_pooled = np.zeros((h_out, w_out))
                
                for i in range(h_out):
                    for j in range(w_out):
                        region = img[i*pool_size:(i+1)*pool_size, 
                                   j*pool_size:(j+1)*pool_size]
                        max_pooled[i, j] = np.max(region)
                        avg_pooled[i, j] = np.mean(region)
                
                results[f'max_pool_{pool_size}'] = max_pooled
                results[f'avg_pool_{pool_size}'] = avg_pooled
                
                # Calculate information retention
                original_info = np.var(img)
                max_info = np.var(max_pooled)
                avg_info = np.var(avg_pooled)
                
                results[f'max_info_retention_{pool_size}'] = max_info / original_info
                results[f'avg_info_retention_{pool_size}'] = avg_info / original_info
            
            pooling_results[img_name] = results
        
        return pooling_results
    
    def performance_comparison(self) -> Dict[str, Any]:
        """Compare performance across all implementations."""
        print("\nPerformance Comparison Across Frameworks...")
        print("=" * 50)
        
        # Get framework comparison results
        framework_results = self.compare_frameworks_mnist()
        
        # Performance metrics to track
        metrics = ['test_accuracy', 'training_time']
        
        # Create comparison plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Test accuracy comparison
        frameworks = [name for name in framework_results.keys() if 'error' not in framework_results[name]]
        accuracies = [framework_results[name]['test_accuracy'] for name in frameworks]
        
        bars1 = axes[0].bar(frameworks, accuracies, color=['lightblue', 'lightcoral', 'lightgreen'])
        axes[0].set_title('Test Accuracy Comparison')
        axes[0].set_ylabel('Test Accuracy')
        axes[0].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom')
        
        # Training time comparison
        times = [framework_results[name]['training_time'] for name in frameworks]
        bars2 = axes[1].bar(frameworks, times, color=['lightblue', 'lightcoral', 'lightgreen'])
        axes[1].set_title('Training Time Comparison')
        axes[1].set_ylabel('Time (seconds)')
        
        # Add value labels
        for bar, time_val in zip(bars2, times):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{time_val:.1f}s', ha='center', va='bottom')
        
        # Efficiency plot (accuracy vs time)
        axes[2].scatter(times, accuracies, s=100, alpha=0.7)
        for i, framework in enumerate(frameworks):
            axes[2].annotate(framework, (times[i], accuracies[i]), 
                           xytext=(5, 5), textcoords='offset points')
        axes[2].set_title('Efficiency: Accuracy vs Training Time')
        axes[2].set_xlabel('Training Time (seconds)')
        axes[2].set_ylabel('Test Accuracy')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/framework_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return framework_results
    
    def architecture_depth_analysis(self) -> Dict[str, Any]:
        """Analyze effect of network depth on performance."""
        print("\nAnalyzing Network Depth Effects...")
        print("=" * 40)
        
        if not KERAS_AVAILABLE:
            print("Keras not available for depth analysis")
            return {}
        
        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test = load_keras_mnist()
        
        # Use subset for faster experiments
        X_train_sub = X_train[:3000]
        y_train_sub = y_train[:3000]
        X_val_sub = X_val[:500]
        y_val_sub = y_val[:500]
        X_test_sub = X_test[:500]
        y_test_sub = y_test[:500]
        
        # Define different depths
        depth_configs = {
            'Shallow (1 Conv)': self._create_depth_model(1, (28, 28, 1), 10),
            'Medium (2 Conv)': self._create_depth_model(2, (28, 28, 1), 10),
            'Deep (3 Conv)': self._create_depth_model(3, (28, 28, 1), 10),
            'Very Deep (4 Conv)': self._create_depth_model(4, (28, 28, 1), 10)
        }
        
        depth_results = {}
        
        for depth_name, model in depth_configs.items():
            print(f"Training {depth_name}...")
            
            # Count parameters
            total_params = model.count_params()
            
            start_time = time.time()
            history = model.fit(
                X_train_sub, y_train_sub,
                validation_data=(X_val_sub, y_val_sub),
                epochs=8, batch_size=64, verbose=0
            )
            training_time = time.time() - start_time
            
            # Evaluate
            test_loss, test_acc = model.evaluate(X_test_sub, y_test_sub, verbose=0)
            
            depth_results[depth_name] = {
                'test_accuracy': test_acc,
                'test_loss': test_loss,
                'training_time': training_time,
                'total_params': total_params,
                'history': history
            }
            
            print(f"  Params: {total_params:,}, Accuracy: {test_acc:.3f}, Time: {training_time:.1f}s")
        
        # Visualize depth analysis
        self._visualize_depth_analysis(depth_results)
        
        return depth_results
    
    def _create_depth_model(self, num_conv_layers: int, input_shape: Tuple[int, int, int], 
                           num_classes: int) -> keras.Model:
        """Create model with specified number of conv layers."""
        model = keras.Sequential()
        
        # First layer
        model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        
        # Additional conv layers
        filters = 32
        for i in range(num_conv_layers - 1):
            if i % 2 == 1:  # Increase filters every 2 layers
                filters = min(filters * 2, 128)
            
            model.add(keras.layers.Conv2D(filters, (3, 3), activation='relu'))
            
            if i % 2 == 1:  # Add pooling every 2 layers
                model.add(keras.layers.MaxPooling2D((2, 2)))
        
        # Final pooling if not added
        if num_conv_layers % 2 == 1:
            model.add(keras.layers.MaxPooling2D((2, 2)))
        
        # Classifier
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(num_classes, activation='softmax'))
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _visualize_depth_analysis(self, results: Dict[str, Any]) -> None:
        """Visualize depth analysis results."""
        depth_names = list(results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy vs depth
        accuracies = [results[name]['test_accuracy'] for name in depth_names]
        axes[0, 0].plot(range(1, len(depth_names) + 1), accuracies, 'o-', linewidth=2, markersize=8)
        axes[0, 0].set_title('Test Accuracy vs Network Depth')
        axes[0, 0].set_xlabel('Number of Conv Layers')
        axes[0, 0].set_ylabel('Test Accuracy')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xticks(range(1, len(depth_names) + 1))
        axes[0, 0].set_xticklabels([name.split('(')[0].strip() for name in depth_names], rotation=45)
        
        # Parameters vs accuracy
        param_counts = [results[name]['total_params'] for name in depth_names]
        axes[0, 1].scatter(param_counts, accuracies, s=100, alpha=0.7)
        for i, name in enumerate(depth_names):
            axes[0, 1].annotate(name.split('(')[0].strip(), 
                              (param_counts[i], accuracies[i]), 
                              xytext=(5, 5), textcoords='offset points')
        axes[0, 1].set_title('Parameter Efficiency')
        axes[0, 1].set_xlabel('Number of Parameters')
        axes[0, 1].set_ylabel('Test Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Training time vs depth
        training_times = [results[name]['training_time'] for name in depth_names]
        axes[1, 0].plot(range(1, len(depth_names) + 1), training_times, 's-', 
                       linewidth=2, markersize=8, color='orange')
        axes[1, 0].set_title('Training Time vs Network Depth')
        axes[1, 0].set_xlabel('Number of Conv Layers')
        axes[1, 0].set_ylabel('Training Time (seconds)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xticks(range(1, len(depth_names) + 1))
        axes[1, 0].set_xticklabels([name.split('(')[0].strip() for name in depth_names], rotation=45)
        
        # Validation curves
        for name in depth_names:
            history = results[name]['history']
            axes[1, 1].plot(history.history['val_accuracy'], label=name.split('(')[0].strip())
        
        axes[1, 1].set_title('Validation Accuracy Curves')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Validation Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/depth_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def activation_function_analysis(self) -> Dict[str, Any]:
        """Compare different activation functions in CNNs."""
        print("\nAnalyzing Activation Functions...")
        print("=" * 40)
        
        if not KERAS_AVAILABLE:
            print("Keras not available for activation analysis")
            return {}
        
        # Load data
        X_train, X_val, X_test, y_train, y_val, y_test = load_keras_mnist()
        
        # Use subset
        X_train_sub = X_train[:2000]
        y_train_sub = y_train[:2000]
        X_val_sub = X_val[:400]
        y_val_sub = y_val[:400]
        X_test_sub = X_test[:400]
        y_test_sub = y_test[:400]
        
        # Test different activations
        activations = ['relu', 'leaky_relu', 'elu', 'tanh']
        activation_results = {}
        
        for activation in activations:
            print(f"Testing {activation} activation...")
            
            # Create model with specific activation
            model = self._create_activation_model(activation, (28, 28, 1), 10)
            
            start_time = time.time()
            history = model.fit(
                X_train_sub, y_train_sub,
                validation_data=(X_val_sub, y_val_sub),
                epochs=6, batch_size=64, verbose=0
            )
            training_time = time.time() - start_time
            
            # Evaluate
            test_loss, test_acc = model.evaluate(X_test_sub, y_test_sub, verbose=0)
            
            activation_results[activation] = {
                'test_accuracy': test_acc,
                'test_loss': test_loss,
                'training_time': training_time,
                'history': history,
                'model': model
            }
            
            print(f"  Accuracy: {test_acc:.3f}, Time: {training_time:.1f}s")
        
        # Visualize activation comparison
        self._visualize_activation_analysis(activation_results)
        
        return activation_results
    
    def _create_activation_model(self, activation: str, input_shape: Tuple[int, int, int], 
                               num_classes: int) -> keras.Model:
        """Create model with specific activation function."""
        model = keras.Sequential()
        
        if activation == 'leaky_relu':
            model.add(keras.layers.Conv2D(32, (3, 3), input_shape=input_shape))
            model.add(keras.layers.LeakyReLU(alpha=0.01))
            model.add(keras.layers.MaxPooling2D((2, 2)))
            
            model.add(keras.layers.Conv2D(64, (3, 3)))
            model.add(keras.layers.LeakyReLU(alpha=0.01))
            model.add(keras.layers.MaxPooling2D((2, 2)))
            
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(64))
            model.add(keras.layers.LeakyReLU(alpha=0.01))
            model.add(keras.layers.Dense(num_classes, activation='softmax'))
        else:
            model.add(keras.layers.Conv2D(32, (3, 3), activation=activation, input_shape=input_shape))
            model.add(keras.layers.MaxPooling2D((2, 2)))
            model.add(keras.layers.Conv2D(64, (3, 3), activation=activation))
            model.add(keras.layers.MaxPooling2D((2, 2)))
            model.add(keras.layers.Flatten())
            model.add(keras.layers.Dense(64, activation=activation))
            model.add(keras.layers.Dense(num_classes, activation='softmax'))
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _visualize_activation_analysis(self, results: Dict[str, Any]) -> None:
        """Visualize activation function analysis."""
        activation_names = list(results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Test accuracies
        accuracies = [results[name]['test_accuracy'] for name in activation_names]
        bars = axes[0, 0].bar(activation_names, accuracies, color='skyblue')
        axes[0, 0].set_title('Test Accuracy by Activation Function')
        axes[0, 0].set_ylabel('Test Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                           f'{acc:.3f}', ha='center', va='bottom')
        
        # Training times
        training_times = [results[name]['training_time'] for name in activation_names]
        axes[0, 1].bar(activation_names, training_times, color='lightcoral')
        axes[0, 1].set_title('Training Time by Activation Function')
        axes[0, 1].set_ylabel('Training Time (seconds)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Convergence curves
        for name in activation_names:
            history = results[name]['history']
            axes[1, 0].plot(history.history['val_accuracy'], label=name, marker='o')
        
        axes[1, 0].set_title('Validation Accuracy Convergence')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Validation Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Loss curves
        for name in activation_names:
            history = results[name]['history']
            axes[1, 1].plot(history.history['val_loss'], label=name, marker='s')
        
        axes[1, 1].set_title('Validation Loss Curves')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Validation Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/activation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_summary(self, all_results: Dict[str, Any]) -> None:
        """Create comprehensive summary of all experiments."""
        print("\nCreating Comprehensive Summary...")
        
        # Create mega-plot with all key results
        fig = plt.figure(figsize=(20, 15))
        
        # Framework comparison (if available)
        if 'framework_comparison' in all_results:
            framework_results = all_results['framework_comparison']
            frameworks = [name for name in framework_results.keys() 
                         if 'error' not in framework_results[name]]
            
            if frameworks:
                plt.subplot(3, 3, 1)
                accuracies = [framework_results[name]['test_accuracy'] for name in frameworks]
                plt.bar(frameworks, accuracies, color=['lightblue', 'lightcoral', 'lightgreen'])
                plt.title('Framework Accuracy Comparison')
                plt.ylabel('Test Accuracy')
                plt.xticks(rotation=45)
        
        # Depth analysis (if available)
        if 'depth_analysis' in all_results:
            depth_results = all_results['depth_analysis']
            depth_names = list(depth_results.keys())
            
            plt.subplot(3, 3, 2)
            depth_accuracies = [depth_results[name]['test_accuracy'] for name in depth_names]
            plt.plot(range(1, len(depth_names) + 1), depth_accuracies, 'o-', linewidth=2)
            plt.title('Accuracy vs Network Depth')
            plt.xlabel('Number of Conv Layers')
            plt.ylabel('Test Accuracy')
            plt.grid(True, alpha=0.3)
        
        # Activation function analysis (if available)
        if 'activation_analysis' in all_results:
            activation_results = all_results['activation_analysis']
            activation_names = list(activation_results.keys())
            
            plt.subplot(3, 3, 3)
            activation_accuracies = [activation_results[name]['test_accuracy'] for name in activation_names]
            plt.bar(activation_names, activation_accuracies, color='lightsteelblue')
            plt.title('Activation Function Comparison')
            plt.ylabel('Test Accuracy')
            plt.xticks(rotation=45)
        
        # Component analysis visualization
        plt.subplot(3, 3, 4)
        # Create a conceptual diagram
        concepts = ['Convolution', 'Pooling', 'ReLU', 'Fully Connected']
        importance = [0.9, 0.7, 0.8, 0.6]  # Conceptual importance
        colors = ['red', 'blue', 'green', 'orange']
        
        bars = plt.bar(concepts, importance, color=colors, alpha=0.7)
        plt.title('CNN Component Importance')
        plt.ylabel('Relative Importance')
        plt.xticks(rotation=45)
        
        # CNN evolution diagram
        plt.subplot(3, 3, 5)
        milestones = ['LeNet\n(1998)', 'AlexNet\n(2012)', 'VGG\n(2014)', 'ResNet\n(2015)']
        accuracies_historic = [0.99, 0.8, 0.92, 0.96]  # Approximate historical accuracies
        
        plt.plot(milestones, accuracies_historic, 'o-', linewidth=2, markersize=8, color='purple')
        plt.title('CNN Architecture Evolution')
        plt.ylabel('Approximate Accuracy')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Key insights text
        insights_text = """
        Key CNN Insights:
        
        ✓ Local Connectivity
          Filters detect local patterns
        
        ✓ Parameter Sharing
          Same filter across image
        
        ✓ Translation Invariance
          Features detected anywhere
        
        ✓ Hierarchical Learning
          Simple → Complex features
        
        ✓ Spatial Structure
          Preserves image topology
        """
        
        plt.subplot(3, 3, 6)
        plt.text(0.05, 0.95, insights_text, transform=plt.gca().transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        plt.title('CNN Key Insights')
        plt.axis('off')
        
        # Performance summary table
        plt.subplot(3, 3, 7)
        if 'framework_comparison' in all_results:
            framework_results = all_results['framework_comparison']
            frameworks = [name for name in framework_results.keys() 
                         if 'error' not in framework_results[name]]
            
            if frameworks:
                table_data = []
                for framework in frameworks:
                    result = framework_results[framework]
                    table_data.append([
                        framework,
                        f"{result['test_accuracy']:.3f}",
                        f"{result['training_time']:.1f}s"
                    ])
                
                table = plt.table(cellText=table_data,
                                colLabels=['Framework', 'Accuracy', 'Time'],
                                cellLoc='center',
                                loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1.2, 1.5)
                plt.title('Performance Summary')
                plt.axis('off')
        
        # Future directions
        future_text = """
        Advanced CNN Topics:
        
        • ResNet & Skip Connections
        • DenseNet & Dense Blocks  
        • Attention Mechanisms
        • Depthwise Separable Conv
        • Neural Architecture Search
        • EfficientNet & Scaling
        • Vision Transformers
        """
        
        plt.subplot(3, 3, 8)
        plt.text(0.05, 0.95, future_text, transform=plt.gca().transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))
        plt.title('Advanced CNN Topics')
        plt.axis('off')
        
        # Best practices
        practices_text = """
        CNN Best Practices:
        
        ✓ Data Augmentation
        ✓ Batch Normalization
        ✓ Dropout Regularization
        ✓ Learning Rate Scheduling
        ✓ Proper Weight Init
        ✓ Gradient Clipping
        ✓ Transfer Learning
        """
        
        plt.subplot(3, 3, 9)
        plt.text(0.05, 0.95, practices_text, transform=plt.gca().transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        plt.title('Best Practices')
        plt.axis('off')
        
        plt.suptitle('CNN Comprehensive Summary', fontsize=18)
        plt.tight_layout()
        plt.savefig('plots/cnn_comprehensive_summary.png', dpi=300, bbox_inches='tight')
        plt.show()


def set_all_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    if PYTORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    if KERAS_AVAILABLE:
        tf.random.set_seed(seed)


def run_comprehensive_cnn_experiments() -> Dict[str, Any]:
    """Run all CNN experiments."""
    print("Starting Comprehensive CNN Experiments")
    print("=" * 60)
    
    # Set seeds for reproducibility
    set_all_seeds(42)
    
    # Initialize experiments
    experiments = CNNExperiments()
    
    all_results = {}
    
    # Experiment 1: Framework comparison
    print("\n" + "=" * 50)
    print("EXPERIMENT 1: Framework Comparison")
    print("=" * 50)
    
    framework_results = experiments.performance_comparison()
    all_results['framework_comparison'] = framework_results
    
    # Experiment 2: CNN component analysis
    print("\n" + "=" * 50)
    print("EXPERIMENT 2: CNN Component Analysis")
    print("=" * 50)
    
    component_results = experiments.analyze_cnn_components()
    all_results['component_analysis'] = component_results
    
    # Experiment 3: Architecture depth analysis
    print("\n" + "=" * 50)
    print("EXPERIMENT 3: Network Depth Analysis")
    print("=" * 50)
    
    depth_results = experiments.architecture_depth_analysis()
    all_results['depth_analysis'] = depth_results
    
    # Experiment 4: Activation function analysis
    print("\n" + "=" * 50)
    print("EXPERIMENT 4: Activation Function Analysis")
    print("=" * 50)
    
    activation_results = experiments.activation_function_analysis()
    all_results['activation_analysis'] = activation_results
    
    # Create comprehensive summary
    print("\n" + "=" * 50)
    print("CREATING COMPREHENSIVE SUMMARY")
    print("=" * 50)
    
    experiments.create_comprehensive_summary(all_results)
    
    # Save results
    print("\nSaving experiment results...")
    
    # Prepare results for saving (remove large objects)
    results_to_save = {}
    for exp_name, exp_results in all_results.items():
        if exp_name == 'framework_comparison':
            results_to_save[exp_name] = {}
            for framework, result in exp_results.items():
                if 'error' not in result:
                    results_to_save[exp_name][framework] = {
                        'test_accuracy': result['test_accuracy'],
                        'test_loss': result['test_loss'],
                        'training_time': result['training_time']
                    }
                else:
                    results_to_save[exp_name][framework] = result
        else:
            # For other experiments, save metrics only
            if isinstance(exp_results, dict):
                results_to_save[exp_name] = {}
                for key, value in exp_results.items():
                    if isinstance(value, dict) and 'model' in value:
                        # Remove model, keep metrics
                        results_to_save[exp_name][key] = {
                            k: v for k, v in value.items() if k != 'model'
                        }
                    else:
                        results_to_save[exp_name][key] = value
            else:
                results_to_save[exp_name] = exp_results
    
    with open('plots/cnn_experiment_results.pkl', 'wb') as f:
        pickle.dump(results_to_save, f)
    
    print("CNN experiments completed!")
    print("Results saved to plots/cnn_experiment_results.pkl")
    print("Check the plots/ directory for all visualizations.")
    
    return all_results


if __name__ == "__main__":
    results = run_comprehensive_cnn_experiments()