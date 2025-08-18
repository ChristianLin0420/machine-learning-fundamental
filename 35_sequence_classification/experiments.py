#!/usr/bin/env python3
"""
Comprehensive experiments for sequence classification models.

This module provides experiments comparing:
- Different model architectures (RNN, LSTM, GRU, Attention, Transformer, CNN)
- NumPy vs PyTorch implementations
- Different datasets (IMDB, 20 Newsgroups, Synthetic)
- Hyperparameter effects
- Sequence length analysis
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import pickle
import os
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import pandas as pd

# Import our modules
from data_preprocessing import create_sequence_classification_data
from sequence_models import (
    SimpleRNNClassifier, LSTMClassifier, AttentionLSTMClassifier,
    SequenceClassifierTrainer
)
from pytorch_models import (
    create_pytorch_model, SequenceClassifierTrainerPyTorch,
    SequenceDatasetPyTorch
)
from visualization import SequenceClassificationVisualizer, extract_attention_weights

# Set random seeds for reproducibility
def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

class SequenceClassificationExperiments:
    """Comprehensive sequence classification experiments."""
    
    def __init__(self, save_dir: str = "experiments", device: str = 'auto'):
        """
        Initialize experiments.
        
        Args:
            save_dir: Directory to save results
            device: Device to use ('auto', 'cpu', 'cuda')
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize visualizer
        self.visualizer = SequenceClassificationVisualizer(
            save_dir=os.path.join(save_dir, "plots")
        )
        
        # Results storage
        self.results = defaultdict(dict)
    
    def create_datasets(self, dataset_configs: List[Dict]) -> Dict[str, Dict]:
        """
        Create datasets for experiments.
        
        Args:
            dataset_configs: List of dataset configuration dictionaries
            
        Returns:
            datasets: Dictionary of dataset name to data dictionary
        """
        datasets = {}
        
        for config in dataset_configs:
            dataset_name = config['dataset_name']
            print(f"Creating {dataset_name} dataset...")
            
            data = create_sequence_classification_data(**config)
            datasets[dataset_name] = data
            
            print(f"  Classes: {data['class_names']}")
            print(f"  Vocab size: {data['vocab_size']}")
            print(f"  Train samples: {data['dataset_info']['train_size']}")
            print(f"  Test samples: {data['dataset_info']['test_size']}")
        
        return datasets
    
    def run_numpy_experiments(self, datasets: Dict[str, Dict],
                             model_configs: List[Dict],
                             training_config: Dict) -> Dict[str, Dict]:
        """
        Run experiments with NumPy implementations.
        
        Args:
            datasets: Dictionary of datasets
            model_configs: List of model configurations
            training_config: Training configuration
            
        Returns:
            results: Experiment results
        """
        numpy_results = {}
        
        for dataset_name, data in datasets.items():
            print(f"\n{'='*60}")
            print(f"NUMPY EXPERIMENTS - {dataset_name.upper()}")
            print(f"{'='*60}")
            
            dataset_results = {}
            
            for model_config in model_configs:
                model_name = model_config['name']
                print(f"\nTraining {model_name} on {dataset_name}...")
                
                # Create model
                if model_name == 'SimpleRNN':
                    model = SimpleRNNClassifier(
                        vocab_size=data['vocab_size'],
                        embed_dim=model_config['embed_dim'],
                        hidden_dim=model_config['hidden_dim'],
                        n_classes=data['n_classes'],
                        max_length=data['max_length']
                    )
                elif model_name == 'LSTM':
                    model = LSTMClassifier(
                        vocab_size=data['vocab_size'],
                        embed_dim=model_config['embed_dim'],
                        hidden_dim=model_config['hidden_dim'],
                        n_classes=data['n_classes'],
                        max_length=data['max_length']
                    )
                elif model_name == 'AttentionLSTM':
                    model = AttentionLSTMClassifier(
                        vocab_size=data['vocab_size'],
                        embed_dim=model_config['embed_dim'],
                        hidden_dim=model_config['hidden_dim'],
                        n_classes=data['n_classes'],
                        max_length=data['max_length'],
                        attention_dim=model_config.get('attention_dim', 64)
                    )
                else:
                    print(f"Unknown model: {model_name}")
                    continue
                
                # Create trainer
                trainer = SequenceClassifierTrainer(
                    model=model,
                    learning_rate=training_config['learning_rate'],
                    clip_grad=training_config['clip_grad']
                )
                
                # Train model
                start_time = time.time()
                history = trainer.train(
                    train_data=data['train_dataset'],
                    test_data=data['test_dataset'],
                    epochs=training_config['epochs'],
                    batch_size=training_config['batch_size'],
                    verbose=False
                )
                training_time = time.time() - start_time
                
                # Evaluate
                test_loss, test_accuracy = trainer.evaluate(
                    data['test_dataset'], 
                    batch_size=training_config['batch_size']
                )
                
                # Get predictions for additional metrics
                test_sequences = data['test_dataset'].sequences
                test_labels = data['test_dataset'].labels
                predictions = model.predict(test_sequences)
                
                # Calculate F1 score
                from sklearn.metrics import f1_score
                f1 = f1_score(test_labels, predictions, average='weighted')
                
                # Store results
                dataset_results[model_name] = {
                    'accuracy': test_accuracy,
                    'loss': test_loss,
                    'f1': f1,
                    'training_time': training_time,
                    'history': history,
                    'predictions': predictions,
                    'model': model
                }
                
                print(f"  Test Accuracy: {test_accuracy:.4f}")
                print(f"  Test Loss: {test_loss:.4f}")
                print(f"  F1 Score: {f1:.4f}")
                print(f"  Training Time: {training_time:.2f}s")
            
            numpy_results[dataset_name] = dataset_results
        
        return numpy_results
    
    def run_pytorch_experiments(self, datasets: Dict[str, Dict],
                               model_configs: List[Dict],
                               training_config: Dict) -> Dict[str, Dict]:
        """
        Run experiments with PyTorch implementations.
        
        Args:
            datasets: Dictionary of datasets
            model_configs: List of model configurations
            training_config: Training configuration
            
        Returns:
            results: Experiment results
        """
        pytorch_results = {}
        
        for dataset_name, data in datasets.items():
            print(f"\n{'='*60}")
            print(f"PYTORCH EXPERIMENTS - {dataset_name.upper()}")
            print(f"{'='*60}")
            
            dataset_results = {}
            
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
            train_loader = DataLoader(
                train_pytorch,
                batch_size=training_config['batch_size'],
                shuffle=True
            )
            test_loader = DataLoader(
                test_pytorch,
                batch_size=training_config['batch_size'],
                shuffle=False
            )
            
            for model_config in model_configs:
                model_name = model_config['name']
                print(f"\nTraining {model_name} on {dataset_name}...")
                
                try:
                    # Create model
                    model = create_pytorch_model(
                        model_type=model_config['type'],
                        vocab_size=data['vocab_size'],
                        embed_dim=model_config['embed_dim'],
                        hidden_dim=model_config['hidden_dim'],
                        n_classes=data['n_classes'],
                        **model_config.get('kwargs', {})
                    )
                    
                    # Create trainer
                    trainer = SequenceClassifierTrainerPyTorch(
                        model=model,
                        device=self.device,
                        learning_rate=training_config['learning_rate']
                    )
                    
                    # Train model
                    start_time = time.time()
                    history = trainer.train(
                        train_loader=train_loader,
                        val_loader=test_loader,
                        epochs=training_config['epochs'],
                        verbose=False
                    )
                    training_time = time.time() - start_time
                    
                    # Evaluate
                    test_loss, test_accuracy = trainer.evaluate(test_loader)
                    
                    # Get predictions
                    predictions, probabilities = trainer.predict(test_loader)
                    
                    # Calculate F1 score
                    from sklearn.metrics import f1_score
                    f1 = f1_score(data['test_dataset'].labels, predictions, average='weighted')
                    
                    # Store results
                    dataset_results[model_name] = {
                        'accuracy': test_accuracy,
                        'loss': test_loss,
                        'f1': f1,
                        'training_time': training_time,
                        'history': history,
                        'predictions': predictions,
                        'probabilities': probabilities,
                        'model': model
                    }
                    
                    print(f"  Test Accuracy: {test_accuracy:.4f}")
                    print(f"  Test Loss: {test_loss:.4f}")
                    print(f"  F1 Score: {f1:.4f}")
                    print(f"  Training Time: {training_time:.2f}s")
                    
                except Exception as e:
                    print(f"  Error training {model_name}: {e}")
                    continue
            
            pytorch_results[dataset_name] = dataset_results
        
        return pytorch_results
    
    def analyze_sequence_length_impact(self, dataset: Dict, model_configs: List[Dict],
                                     training_config: Dict) -> Dict:
        """
        Analyze the impact of sequence length on model performance.
        
        Args:
            dataset: Dataset dictionary
            model_configs: List of model configurations
            training_config: Training configuration
            
        Returns:
            length_analysis: Results of sequence length analysis
        """
        print(f"\n{'='*60}")
        print(f"SEQUENCE LENGTH ANALYSIS")
        print(f"{'='*60}")
        
        # Group test samples by sequence length
        test_sequences = dataset['test_dataset'].sequences
        test_labels = dataset['test_dataset'].labels
        
        length_groups = defaultdict(list)
        for i, seq in enumerate(test_sequences):
            # Calculate actual length (excluding padding)
            actual_length = np.sum(seq != 0)  # Assuming 0 is padding
            length_groups[actual_length].append(i)
        
        # Filter groups with at least 10 samples
        valid_lengths = {length: indices for length, indices in length_groups.items() 
                        if len(indices) >= 10}
        
        length_results = {}
        
        for model_config in model_configs[:2]:  # Test on first 2 models to save time
            model_name = model_config['name']
            print(f"\nAnalyzing {model_name}...")
            
            # Train model
            if model_name == 'LSTM':
                model = LSTMClassifier(
                    vocab_size=dataset['vocab_size'],
                    embed_dim=model_config['embed_dim'],
                    hidden_dim=model_config['hidden_dim'],
                    n_classes=dataset['n_classes'],
                    max_length=dataset['max_length']
                )
            else:
                continue
            
            trainer = SequenceClassifierTrainer(
                model=model,
                learning_rate=training_config['learning_rate']
            )
            
            # Quick training
            trainer.train(
                train_data=dataset['train_dataset'],
                epochs=min(20, training_config['epochs']),
                batch_size=training_config['batch_size'],
                verbose=False
            )
            
            # Analyze performance by length
            length_performance = {}
            for length, indices in valid_lengths.items():
                if len(indices) < 5:  # Skip small groups
                    continue
                
                # Get subset
                subset_sequences = test_sequences[indices]
                subset_labels = test_labels[indices]
                
                # Predict
                predictions = model.predict(subset_sequences)
                
                # Calculate metrics
                accuracy = np.mean(predictions == subset_labels)
                
                # Calculate loss manually
                outputs, _ = model.forward(subset_sequences)
                loss = model.compute_loss(outputs, subset_labels)
                
                length_performance[length] = {
                    'accuracy': accuracy,
                    'loss': loss,
                    'count': len(indices)
                }
            
            length_results[model_name] = length_performance
            
            # Visualize
            self.visualizer.plot_sequence_length_analysis(
                length_performance,
                save_path=f"length_analysis_{model_name.lower()}.png",
                title=f"Performance vs Sequence Length - {model_name}"
            )
        
        return length_results
    
    def compare_frameworks(self, results_numpy: Dict, results_pytorch: Dict) -> Dict:
        """
        Compare NumPy vs PyTorch implementations.
        
        Args:
            results_numpy: NumPy experiment results
            results_pytorch: PyTorch experiment results
            
        Returns:
            comparison: Framework comparison results
        """
        print(f"\n{'='*60}")
        print(f"FRAMEWORK COMPARISON")
        print(f"{'='*60}")
        
        comparison_results = {}
        
        for dataset_name in results_numpy.keys():
            if dataset_name not in results_pytorch:
                continue
            
            print(f"\nComparing on {dataset_name}:")
            print("-" * 40)
            
            dataset_comparison = {}
            
            # Compare common models
            common_models = set(results_numpy[dataset_name].keys()) & set(results_pytorch[dataset_name].keys())
            
            for model_name in common_models:
                numpy_result = results_numpy[dataset_name][model_name]
                pytorch_result = results_pytorch[dataset_name][model_name]
                
                comparison = {
                    'numpy_accuracy': numpy_result['accuracy'],
                    'pytorch_accuracy': pytorch_result['accuracy'],
                    'accuracy_diff': pytorch_result['accuracy'] - numpy_result['accuracy'],
                    'numpy_time': numpy_result['training_time'],
                    'pytorch_time': pytorch_result['training_time'],
                    'time_ratio': pytorch_result['training_time'] / numpy_result['training_time']
                }
                
                dataset_comparison[model_name] = comparison
                
                print(f"{model_name}:")
                print(f"  NumPy accuracy: {numpy_result['accuracy']:.4f}")
                print(f"  PyTorch accuracy: {pytorch_result['accuracy']:.4f}")
                print(f"  Accuracy difference: {comparison['accuracy_diff']:+.4f}")
                print(f"  NumPy time: {numpy_result['training_time']:.2f}s")
                print(f"  PyTorch time: {pytorch_result['training_time']:.2f}s")
                print(f"  Time ratio: {comparison['time_ratio']:.2f}x")
            
            comparison_results[dataset_name] = dataset_comparison
        
        return comparison_results
    
    def visualize_attention_analysis(self, pytorch_results: Dict, datasets: Dict):
        """
        Analyze and visualize attention patterns.
        
        Args:
            pytorch_results: PyTorch experiment results
            datasets: Datasets dictionary
        """
        print(f"\n{'='*60}")
        print(f"ATTENTION ANALYSIS")
        print(f"{'='*60}")
        
        for dataset_name, results in pytorch_results.items():
            if 'Attention' not in results:
                continue
            
            print(f"\nAnalyzing attention on {dataset_name}...")
            
            dataset = datasets[dataset_name]
            model = results['Attention']['model']
            
            # Get test samples for attention analysis
            test_sequences = torch.tensor(dataset['test_dataset'].sequences[:20])  # First 20 samples
            test_labels = dataset['test_dataset'].labels[:20]
            
            # Extract attention weights
            attention_data = extract_attention_weights(
                model, test_sequences, dataset['vocabulary'], dataset['class_names']
            )
            
            # Add true labels
            for i, data in enumerate(attention_data):
                data['true_label'] = dataset['class_names'][test_labels[i]]
            
            # Visualize attention patterns
            self.visualizer.plot_attention_heatmap_multiple(
                attention_data[:6],  # Show first 6 samples
                save_path=f"attention_patterns_{dataset_name}.png",
                title=f"Attention Patterns - {dataset_name.title()}"
            )
            
            # Analyze attention patterns
            attention_weights_list = [data['weights'] for data in attention_data]
            tokens_list = [data['tokens'] for data in attention_data]
            predictions_list = [data['prediction'] for data in attention_data]
            
            avg_attention = self.visualizer.analyze_attention_patterns(
                attention_weights_list, tokens_list, predictions_list,
                save_path=f"attention_analysis_{dataset_name}.png",
                title=f"Attention Pattern Analysis - {dataset_name.title()}"
            )
            
            # Save attention analysis
            with open(os.path.join(self.save_dir, f"attention_analysis_{dataset_name}.pkl"), 'wb') as f:
                pickle.dump({
                    'attention_data': attention_data,
                    'avg_attention': avg_attention
                }, f)
    
    def save_results(self, results: Dict, filename: str):
        """Save experiment results to file."""
        # Create a serializable version of results
        serializable_results = {}
        
        for dataset_name, dataset_results in results.items():
            serializable_results[dataset_name] = {}
            for model_name, model_results in dataset_results.items():
                # Remove non-serializable objects
                clean_results = {k: v for k, v in model_results.items() 
                               if k not in ['model']}
                serializable_results[dataset_name][model_name] = clean_results
        
        with open(os.path.join(self.save_dir, filename), 'wb') as f:
            pickle.dump(serializable_results, f)
    
    def create_summary_report(self, results_numpy: Dict, results_pytorch: Dict) -> pd.DataFrame:
        """
        Create a comprehensive summary report.
        
        Args:
            results_numpy: NumPy experiment results
            results_pytorch: PyTorch experiment results
            
        Returns:
            summary_df: Summary DataFrame
        """
        summary_data = []
        
        # Process NumPy results
        for dataset_name, dataset_results in results_numpy.items():
            for model_name, model_results in dataset_results.items():
                summary_data.append({
                    'Dataset': dataset_name,
                    'Model': model_name,
                    'Framework': 'NumPy',
                    'Accuracy': model_results['accuracy'],
                    'Loss': model_results['loss'],
                    'F1_Score': model_results['f1'],
                    'Training_Time': model_results['training_time']
                })
        
        # Process PyTorch results
        for dataset_name, dataset_results in results_pytorch.items():
            for model_name, model_results in dataset_results.items():
                summary_data.append({
                    'Dataset': dataset_name,
                    'Model': model_name,
                    'Framework': 'PyTorch',
                    'Accuracy': model_results['accuracy'],
                    'Loss': model_results['loss'],
                    'F1_Score': model_results['f1'],
                    'Training_Time': model_results['training_time']
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        summary_df.to_csv(os.path.join(self.save_dir, "experiment_summary.csv"), index=False)
        
        return summary_df


def run_comprehensive_experiments():
    """Run all comprehensive experiments."""
    print("Starting Comprehensive Sequence Classification Experiments")
    print("=" * 70)
    
    # Set seeds for reproducibility
    set_seeds(42)
    
    # Initialize experiments
    experiments = SequenceClassificationExperiments()
    
    # Dataset configurations
    dataset_configs = [
        {
            'dataset_name': 'imdb',
            'n_samples': 1000,
            'max_vocab_size': 5000,
            'max_length': 100
        },
        {
            'dataset_name': '20newsgroups',
            'n_samples': 800,
            'max_vocab_size': 3000,
            'max_length': 80
        }
    ]
    
    # Model configurations for NumPy experiments
    numpy_model_configs = [
        {
            'name': 'SimpleRNN',
            'embed_dim': 32,
            'hidden_dim': 64
        },
        {
            'name': 'LSTM',
            'embed_dim': 32,
            'hidden_dim': 64
        },
        {
            'name': 'AttentionLSTM',
            'embed_dim': 32,
            'hidden_dim': 64,
            'attention_dim': 32
        }
    ]
    
    # Model configurations for PyTorch experiments
    pytorch_model_configs = [
        {
            'name': 'RNN',
            'type': 'rnn',
            'embed_dim': 64,
            'hidden_dim': 128
        },
        {
            'name': 'LSTM',
            'type': 'lstm',
            'embed_dim': 64,
            'hidden_dim': 128
        },
        {
            'name': 'GRU',
            'type': 'gru',
            'embed_dim': 64,
            'hidden_dim': 128
        },
        {
            'name': 'Attention',
            'type': 'attention',
            'embed_dim': 64,
            'hidden_dim': 128,
            'kwargs': {'bidirectional': True}
        },
        {
            'name': 'CNN',
            'type': 'cnn',
            'embed_dim': 64,
            'hidden_dim': 128,  # Not used for CNN
            'kwargs': {'filter_sizes': [3, 4, 5], 'num_filters': 64}
        }
    ]
    
    # Training configuration
    training_config = {
        'epochs': 30,
        'batch_size': 32,
        'learning_rate': 0.001,
        'clip_grad': 5.0
    }
    
    # Create datasets
    datasets = experiments.create_datasets(dataset_configs)
    
    # Run NumPy experiments
    print(f"\n{'='*70}")
    print("RUNNING NUMPY EXPERIMENTS")
    print(f"{'='*70}")
    results_numpy = experiments.run_numpy_experiments(
        datasets, numpy_model_configs, training_config
    )
    
    # Run PyTorch experiments
    print(f"\n{'='*70}")
    print("RUNNING PYTORCH EXPERIMENTS")
    print(f"{'='*70}")
    results_pytorch = experiments.run_pytorch_experiments(
        datasets, pytorch_model_configs, training_config
    )
    
    # Compare frameworks
    framework_comparison = experiments.compare_frameworks(results_numpy, results_pytorch)
    
    # Sequence length analysis
    length_analysis = experiments.analyze_sequence_length_impact(
        datasets['imdb'], numpy_model_configs, training_config
    )
    
    # Attention analysis
    experiments.visualize_attention_analysis(results_pytorch, datasets)
    
    # Create visualizations
    print(f"\n{'='*70}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*70}")
    
    # Model comparison plots
    for dataset_name in datasets.keys():
        if dataset_name in results_pytorch:
            experiments.visualizer.plot_model_comparison(
                results_pytorch[dataset_name],
                save_path=f"model_comparison_{dataset_name}.png",
                title=f"Model Performance Comparison - {dataset_name.title()}"
            )
    
    # Learning curves comparison
    for dataset_name in datasets.keys():
        if dataset_name in results_pytorch:
            histories = {name: result['history'] 
                        for name, result in results_pytorch[dataset_name].items()}
            experiments.visualizer.plot_learning_curves_comparison(
                histories,
                save_path=f"learning_curves_{dataset_name}.png",
                title=f"Learning Curves Comparison - {dataset_name.title()}"
            )
    
    # Save results
    experiments.save_results(results_numpy, "numpy_results.pkl")
    experiments.save_results(results_pytorch, "pytorch_results.pkl")
    
    # Create summary report
    summary_df = experiments.create_summary_report(results_numpy, results_pytorch)
    
    print(f"\n{'='*70}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*70}")
    print(summary_df.to_string(index=False))
    
    print(f"\n{'='*70}")
    print("EXPERIMENTS COMPLETED")
    print(f"{'='*70}")
    print(f"Results saved to: {experiments.save_dir}")
    print(f"Plots saved to: {os.path.join(experiments.save_dir, 'plots')}")
    
    return {
        'numpy_results': results_numpy,
        'pytorch_results': results_pytorch,
        'framework_comparison': framework_comparison,
        'length_analysis': length_analysis,
        'summary': summary_df,
        'datasets': datasets
    }


if __name__ == "__main__":
    results = run_comprehensive_experiments()