#!/usr/bin/env python3
"""
Comprehensive demonstration of sequence classification implementations.

This script demonstrates:
1. Data preprocessing and dataset creation
2. Training different model architectures
3. Model comparison and evaluation
4. Attention visualization
5. Performance analysis
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import time
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader

# Import our modules
from data_preprocessing import create_sequence_classification_data
from sequence_models import LSTMClassifier, AttentionLSTMClassifier, SequenceClassifierTrainer
from pytorch_models import (
    create_pytorch_model, SequenceClassifierTrainerPyTorch, SequenceDatasetPyTorch
)
from visualization import SequenceClassificationVisualizer, extract_attention_weights

def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def demonstrate_data_preprocessing():
    """Demonstrate data preprocessing capabilities."""
    print("=" * 60)
    print("DEMONSTRATING DATA PREPROCESSING")
    print("=" * 60)
    
    # Create different types of datasets
    datasets_info = []
    
    # IMDB sentiment analysis
    print("\n1. Creating IMDB-like sentiment dataset...")
    imdb_data = create_sequence_classification_data(
        dataset_name="imdb",
        n_samples=500,
        max_vocab_size=2000,
        max_length=50
    )
    datasets_info.append(("IMDB Sentiment", imdb_data))
    
    # 20 Newsgroups text classification
    print("\n2. Creating 20 Newsgroups-like dataset...")
    news_data = create_sequence_classification_data(
        dataset_name="20newsgroups",
        n_samples=400,
        max_vocab_size=1500,
        max_length=60
    )
    datasets_info.append(("20 Newsgroups", news_data))
    
    # Synthetic sequences
    print("\n3. Creating synthetic sequence dataset...")
    synthetic_data = create_sequence_classification_data(
        dataset_name="synthetic",
        n_samples=600,
        max_vocab_size=1000,
        max_length=40
    )
    datasets_info.append(("Synthetic", synthetic_data))
    
    # Display dataset information
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    
    for name, data in datasets_info:
        print(f"\n{name}:")
        print(f"  Classes: {data['class_names']}")
        print(f"  Vocabulary size: {data['vocab_size']}")
        print(f"  Max sequence length: {data['max_length']}")
        print(f"  Train samples: {data['dataset_info']['train_size']}")
        print(f"  Test samples: {data['dataset_info']['test_size']}")
        
        # Show sample
        sample_seq, sample_label = data['train_dataset'][0]
        sample_tokens = data['vocabulary'].decode(sample_seq.tolist())
        print(f"  Sample: {' '.join(sample_tokens[:8])}...")
        print(f"  Label: {data['class_names'][sample_label]}")
    
    return datasets_info

def demonstrate_numpy_models(dataset_info: Tuple[str, Dict]):
    """Demonstrate NumPy model implementations."""
    name, data = dataset_info
    print(f"\n{'='*60}")
    print(f"DEMONSTRATING NUMPY MODELS - {name.upper()}")
    print(f"{'='*60}")
    
    results = {}
    
    # Model configurations
    model_configs = [
        {
            'name': 'LSTM',
            'class': LSTMClassifier,
            'params': {
                'vocab_size': data['vocab_size'],
                'embed_dim': 32,
                'hidden_dim': 64,
                'n_classes': data['n_classes'],
                'max_length': data['max_length']
            }
        },
        {
            'name': 'AttentionLSTM',
            'class': AttentionLSTMClassifier,
            'params': {
                'vocab_size': data['vocab_size'],
                'embed_dim': 32,
                'hidden_dim': 64,
                'n_classes': data['n_classes'],
                'max_length': data['max_length'],
                'attention_dim': 32
            }
        }
    ]
    
    for config in model_configs:
        print(f"\nTraining {config['name']}...")
        
        # Create model
        model = config['class'](**config['params'])
        
        # Create trainer
        trainer = SequenceClassifierTrainer(
            model=model,
            learning_rate=0.001,
            clip_grad=5.0
        )
        
        # Train model
        start_time = time.time()
        history = trainer.train(
            train_data=data['train_dataset'],
            test_data=data['test_dataset'],
            epochs=20,
            batch_size=16,
            verbose=True
        )
        training_time = time.time() - start_time
        
        # Evaluate
        test_loss, test_accuracy = trainer.evaluate(data['test_dataset'])
        
        # Store results
        results[config['name']] = {
            'model': model,
            'history': history,
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'training_time': training_time
        }
        
        print(f"Final Test Accuracy: {test_accuracy:.4f}")
        print(f"Final Test Loss: {test_loss:.4f}")
        print(f"Training Time: {training_time:.2f}s")
    
    return results

def demonstrate_pytorch_models(dataset_info: Tuple[str, Dict]):
    """Demonstrate PyTorch model implementations."""
    name, data = dataset_info
    print(f"\n{'='*60}")
    print(f"DEMONSTRATING PYTORCH MODELS - {name.upper()}")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    results = {}
    
    # Create PyTorch datasets
    train_pytorch = SequenceDatasetPyTorch(
        data['train_dataset'].sequences,
        data['train_dataset'].labels
    )
    test_pytorch = SequenceDatasetPyTorch(
        data['test_dataset'].sequences,
        data['test_dataset'].labels
    )
    
    # Create data loaders
    train_loader = DataLoader(train_pytorch, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_pytorch, batch_size=16, shuffle=False)
    
    # Model configurations
    model_configs = [
        {
            'name': 'LSTM',
            'type': 'lstm',
            'kwargs': {'bidirectional': False}
        },
        {
            'name': 'BiLSTM',
            'type': 'lstm',
            'kwargs': {'bidirectional': True}
        },
        {
            'name': 'GRU',
            'type': 'gru',
            'kwargs': {'bidirectional': False}
        },
        {
            'name': 'Attention',
            'type': 'attention',
            'kwargs': {'bidirectional': True}
        },
        {
            'name': 'CNN',
            'type': 'cnn',
            'kwargs': {'filter_sizes': [3, 4, 5], 'num_filters': 64}
        }
    ]
    
    for config in model_configs:
        print(f"\nTraining {config['name']}...")
        
        try:
            # Create model
            model = create_pytorch_model(
                model_type=config['type'],
                vocab_size=data['vocab_size'],
                embed_dim=64,
                hidden_dim=128,
                n_classes=data['n_classes'],
                **config['kwargs']
            )
            
            # Create trainer
            trainer = SequenceClassifierTrainerPyTorch(
                model=model,
                device=device,
                learning_rate=0.001
            )
            
            # Train model
            start_time = time.time()
            history = trainer.train(
                train_loader=train_loader,
                val_loader=test_loader,
                epochs=20,
                verbose=True
            )
            training_time = time.time() - start_time
            
            # Evaluate
            test_loss, test_accuracy = trainer.evaluate(test_loader)
            
            # Store results
            results[config['name']] = {
                'model': model,
                'trainer': trainer,
                'history': history,
                'test_accuracy': test_accuracy,
                'test_loss': test_loss,
                'training_time': training_time
            }
            
            print(f"Final Test Accuracy: {test_accuracy:.4f}")
            print(f"Final Test Loss: {test_loss:.4f}")
            print(f"Training Time: {training_time:.2f}s")
            
        except Exception as e:
            print(f"Error training {config['name']}: {e}")
            continue
    
    return results, test_loader

def demonstrate_attention_visualization(pytorch_results: Dict, data: Dict, test_loader: DataLoader):
    """Demonstrate attention weight visualization."""
    print(f"\n{'='*60}")
    print("DEMONSTRATING ATTENTION VISUALIZATION")
    print(f"{'='*60}")
    
    if 'Attention' not in pytorch_results:
        print("No attention model available for visualization.")
        return
    
    # Get attention model
    attention_model = pytorch_results['Attention']['model']
    
    # Get a few test samples
    test_sequences = []
    test_labels = []
    for batch_sequences, batch_labels in test_loader:
        test_sequences.append(batch_sequences)
        test_labels.append(batch_labels)
        if len(test_sequences) >= 3:  # Get 3 batches
            break
    
    if not test_sequences:
        print("No test data available.")
        return
    
    # Use first batch
    sample_sequences = test_sequences[0][:5]  # First 5 samples
    sample_labels = test_labels[0][:5]
    
    # Extract attention weights
    attention_data = extract_attention_weights(
        attention_model, sample_sequences, data['vocabulary'], data['class_names']
    )
    
    # Add true labels
    for i, att_data in enumerate(attention_data):
        true_label_idx = sample_labels[i].item()
        att_data['true_label'] = data['class_names'][true_label_idx]
    
    # Create visualizer
    visualizer = SequenceClassificationVisualizer(save_dir="demo_plots")
    
    # Visualize individual attention patterns
    for i, att_data in enumerate(attention_data[:3]):  # Show first 3
        visualizer.plot_attention_weights(
            att_data['weights'],
            att_data['tokens'],
            att_data['prediction'],
            save_path=f"attention_sample_{i+1}.png",
            title=f"Attention Weights - Sample {i+1}"
        )
    
    # Visualize multiple patterns
    visualizer.plot_attention_heatmap_multiple(
        attention_data,
        save_path="attention_heatmap_multiple.png",
        title="Attention Patterns Across Multiple Samples"
    )
    
    # Analyze attention patterns
    attention_weights_list = [data['weights'] for data in attention_data]
    tokens_list = [data['tokens'] for data in attention_data]
    predictions_list = [data['prediction'] for data in attention_data]
    
    avg_attention = visualizer.analyze_attention_patterns(
        attention_weights_list, tokens_list, predictions_list,
        save_path="attention_pattern_analysis.png",
        title="Attention Pattern Analysis"
    )
    
    print("\nTop 10 most attended words:")
    sorted_attention = sorted(avg_attention.items(), key=lambda x: x[1], reverse=True)
    for word, weight in sorted_attention[:10]:
        print(f"  {word}: {weight:.4f}")
    
    return attention_data

def demonstrate_model_comparison(numpy_results: Dict, pytorch_results: Dict, dataset_name: str):
    """Demonstrate model comparison and visualization."""
    print(f"\n{'='*60}")
    print("DEMONSTRATING MODEL COMPARISON")
    print(f"{'='*60}")
    
    # Create visualizer
    visualizer = SequenceClassificationVisualizer(save_dir="demo_plots")
    
    # Prepare comparison data
    comparison_data = {}
    
    # Add NumPy results
    for model_name, result in numpy_results.items():
        comparison_data[f"{model_name}_NumPy"] = {
            'accuracy': result['test_accuracy'],
            'loss': result['test_loss'],
            'f1': result['test_accuracy'],  # Approximation
            'time': result['training_time']
        }
    
    # Add PyTorch results
    for model_name, result in pytorch_results.items():
        comparison_data[f"{model_name}_PyTorch"] = {
            'accuracy': result['test_accuracy'],
            'loss': result['test_loss'],
            'f1': result['test_accuracy'],  # Approximation
            'time': result['training_time']
        }
    
    # Create comparison plot
    visualizer.plot_model_comparison(
        comparison_data,
        save_path=f"model_comparison_{dataset_name.lower().replace(' ', '_')}.png",
        title=f"Model Performance Comparison - {dataset_name}"
    )
    
    # Plot learning curves comparison
    pytorch_histories = {name: result['history'] for name, result in pytorch_results.items()}
    visualizer.plot_learning_curves_comparison(
        pytorch_histories,
        save_path=f"learning_curves_{dataset_name.lower().replace(' ', '_')}.png",
        title=f"Learning Curves Comparison - {dataset_name}"
    )
    
    # Print comparison table
    print(f"\nPerformance Comparison - {dataset_name}:")
    print("-" * 60)
    print(f"{'Model':<20} {'Accuracy':<10} {'Loss':<10} {'Time (s)':<10}")
    print("-" * 60)
    
    for model_name, metrics in comparison_data.items():
        print(f"{model_name:<20} {metrics['accuracy']:<10.4f} "
              f"{metrics['loss']:<10.4f} {metrics['time']:<10.2f}")
    
    return comparison_data

def demonstrate_error_analysis(pytorch_results: Dict, data: Dict, test_loader: DataLoader):
    """Demonstrate error analysis and confusion matrix."""
    print(f"\n{'='*60}")
    print("DEMONSTRATING ERROR ANALYSIS")
    print(f"{'='*60}")
    
    if not pytorch_results:
        print("No PyTorch results available for error analysis.")
        return
    
    # Use the best performing model
    best_model_name = max(pytorch_results.keys(), 
                         key=lambda k: pytorch_results[k]['test_accuracy'])
    best_result = pytorch_results[best_model_name]
    
    print(f"Analyzing errors for best model: {best_model_name}")
    
    # Get predictions
    trainer = best_result['trainer']
    predictions, probabilities = trainer.predict(test_loader)
    
    # Get true labels
    true_labels = []
    for _, labels in test_loader:
        true_labels.extend(labels.numpy())
    true_labels = np.array(true_labels)
    
    # Create visualizer
    visualizer = SequenceClassificationVisualizer(save_dir="demo_plots")
    
    # Plot confusion matrix
    visualizer.plot_confusion_matrix(
        true_labels, predictions, data['class_names'],
        save_path=f"confusion_matrix_{best_model_name.lower()}.png",
        title=f"Confusion Matrix - {best_model_name}"
    )
    
    # Create classification report visualization
    visualizer.create_classification_report_plot(
        true_labels, predictions, data['class_names'],
        save_path=f"classification_report_{best_model_name.lower()}.png",
        title=f"Classification Report - {best_model_name}"
    )
    
    # Analyze misclassified samples
    misclassified_indices = np.where(predictions != true_labels)[0]
    print(f"\nMisclassification Analysis:")
    print(f"Total samples: {len(true_labels)}")
    print(f"Misclassified: {len(misclassified_indices)}")
    print(f"Error rate: {len(misclassified_indices) / len(true_labels):.4f}")
    
    # Show some misclassified examples
    if len(misclassified_indices) > 0:
        print(f"\nSample misclassifications:")
        for i in misclassified_indices[:5]:  # Show first 5
            true_class = data['class_names'][true_labels[i]]
            pred_class = data['class_names'][predictions[i]]
            confidence = probabilities[i][predictions[i]]
            print(f"  True: {true_class}, Predicted: {pred_class}, "
                  f"Confidence: {confidence:.4f}")
    
    return {
        'true_labels': true_labels,
        'predictions': predictions,
        'probabilities': probabilities,
        'misclassified_indices': misclassified_indices
    }

def run_comprehensive_demo():
    """Run the complete comprehensive demonstration."""
    print("COMPREHENSIVE SEQUENCE CLASSIFICATION DEMONSTRATION")
    print("=" * 70)
    print("This demo showcases the complete sequence classification pipeline:")
    print("1. Data preprocessing and dataset creation")
    print("2. NumPy model implementations")
    print("3. PyTorch model implementations")
    print("4. Attention visualization")
    print("5. Model comparison and evaluation")
    print("6. Error analysis")
    print("=" * 70)
    
    # Set seeds for reproducibility
    set_seeds(42)
    
    # Create plots directory
    os.makedirs("demo_plots", exist_ok=True)
    
    # Step 1: Demonstrate data preprocessing
    datasets_info = demonstrate_data_preprocessing()
    
    # Use the first dataset for detailed demonstration
    demo_dataset = datasets_info[0]  # IMDB dataset
    
    # Step 2: Demonstrate NumPy models
    numpy_results = demonstrate_numpy_models(demo_dataset)
    
    # Step 3: Demonstrate PyTorch models
    pytorch_results, test_loader = demonstrate_pytorch_models(demo_dataset)
    
    # Step 4: Demonstrate attention visualization
    if pytorch_results:
        attention_data = demonstrate_attention_visualization(
            pytorch_results, demo_dataset[1], test_loader
        )
    
    # Step 5: Demonstrate model comparison
    if numpy_results and pytorch_results:
        comparison_data = demonstrate_model_comparison(
            numpy_results, pytorch_results, demo_dataset[0]
        )
    
    # Step 6: Demonstrate error analysis
    if pytorch_results:
        error_analysis = demonstrate_error_analysis(
            pytorch_results, demo_dataset[1], test_loader
        )
    
    # Summary
    print(f"\n{'='*70}")
    print("DEMONSTRATION SUMMARY")
    print(f"{'='*70}")
    print(f"✓ Created and preprocessed {len(datasets_info)} different datasets")
    print(f"✓ Trained {len(numpy_results)} NumPy models")
    print(f"✓ Trained {len(pytorch_results)} PyTorch models")
    print(f"✓ Generated attention visualizations")
    print(f"✓ Created model comparison plots")
    print(f"✓ Performed error analysis")
    print(f"\nAll plots and results saved to: ./demo_plots/")
    print(f"{'='*70}")
    
    return {
        'datasets': datasets_info,
        'numpy_results': numpy_results,
        'pytorch_results': pytorch_results,
        'demo_dataset': demo_dataset
    }

if __name__ == "__main__":
    demo_results = run_comprehensive_demo()