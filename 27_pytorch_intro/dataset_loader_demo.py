"""
PyTorch Dataset and DataLoader Demo
===================================

This module demonstrates PyTorch's data loading pipeline:
- torch.utils.data.Dataset and DataLoader
- Custom datasets for different data types
- Data transformations and augmentations
- Batching, shuffling, and parallel loading
- Working with various data formats
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_classification, load_digits
from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import Tuple, List, Optional, Callable, Any
import os
import pickle
from PIL import Image

class SyntheticDataset(Dataset):
    """Custom dataset for synthetic data"""
    
    def __init__(self, 
                 n_samples: int = 1000,
                 n_features: int = 20,
                 n_classes: int = 2,
                 noise: float = 0.1,
                 random_state: int = 42,
                 transform: Optional[Callable] = None):
        self.transform = transform
        
        # Generate synthetic data
        if n_classes == 2:
            X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        else:
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_classes=n_classes,
                n_redundant=0,
                n_informative=min(n_features, n_classes),
                random_state=random_state
            )
        
        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        self.data = torch.FloatTensor(X)
        self.targets = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, target

class DigitsDataset(Dataset):
    """Custom dataset for sklearn digits (8x8 images)"""
    
    def __init__(self, transform: Optional[Callable] = None):
        self.transform = transform
        
        # Load digits dataset
        digits = load_digits()
        self.data = torch.FloatTensor(digits.data).reshape(-1, 1, 8, 8)  # Add channel dimension
        self.targets = torch.LongTensor(digits.target)
        
        # Normalize to [0, 1]
        self.data = self.data / 16.0
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        target = self.targets[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, target

class TimeSeriesDataset(Dataset):
    """Custom dataset for time series data"""
    
    def __init__(self, 
                 sequence_length: int = 50,
                 n_sequences: int = 1000,
                 n_features: int = 1,
                 pattern: str = 'sine'):
        self.sequence_length = sequence_length
        
        # Generate time series data
        sequences = []
        targets = []
        
        for i in range(n_sequences):
            t = np.linspace(0, 4*np.pi, sequence_length + 1)
            
            if pattern == 'sine':
                # Sine wave with random frequency and phase
                freq = np.random.uniform(0.5, 2.0)
                phase = np.random.uniform(0, 2*np.pi)
                series = np.sin(freq * t + phase)
            elif pattern == 'trend':
                # Linear trend with noise
                slope = np.random.uniform(-1, 1)
                noise = np.random.normal(0, 0.1, len(t))
                series = slope * t + noise
            else:
                # Random walk
                series = np.cumsum(np.random.normal(0, 0.1, len(t)))
            
            # Input: first sequence_length points, Target: next point
            sequences.append(series[:-1])
            targets.append(series[-1])
        
        self.data = torch.FloatTensor(sequences).unsqueeze(-1)  # Add feature dimension
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class TextDataset(Dataset):
    """Simple text dataset for demonstration"""
    
    def __init__(self, texts: List[str], labels: List[int], vocab_size: int = 1000):
        self.texts = texts
        self.labels = labels
        
        # Simple tokenization (character-level)
        all_chars = set(''.join(texts))
        self.char_to_idx = {char: idx for idx, char in enumerate(sorted(all_chars))}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        
        # Convert texts to indices
        self.encoded_texts = []
        max_length = max(len(text) for text in texts)
        
        for text in texts:
            indices = [self.char_to_idx[char] for char in text]
            # Pad to max length
            indices += [0] * (max_length - len(indices))
            self.encoded_texts.append(indices)
        
        self.data = torch.LongTensor(self.encoded_texts)
        self.targets = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class MemoryMappedDataset(Dataset):
    """Dataset that uses memory mapping for large datasets"""
    
    def __init__(self, data_file: str, targets_file: str):
        # For demonstration, we'll create the files if they don't exist
        if not os.path.exists(data_file):
            self._create_dummy_files(data_file, targets_file)
        
        # Memory map the data
        self.data = np.memmap(data_file, dtype=np.float32, mode='r')
        self.targets = np.memmap(targets_file, dtype=np.int64, mode='r')
        
        # Reshape data (assuming 2D)
        n_samples = len(self.targets)
        n_features = len(self.data) // n_samples
        self.data = self.data.reshape(n_samples, n_features)
    
    def _create_dummy_files(self, data_file: str, targets_file: str):
        """Create dummy memory-mapped files"""
        n_samples = 10000
        n_features = 100
        
        # Create data
        data = np.random.randn(n_samples, n_features).astype(np.float32)
        targets = np.random.randint(0, 5, n_samples).astype(np.int64)
        
        # Save as memory-mapped files
        data_mmap = np.memmap(data_file, dtype=np.float32, mode='w+', shape=(n_samples, n_features))
        targets_mmap = np.memmap(targets_file, dtype=np.int64, mode='w+', shape=(n_samples,))
        
        data_mmap[:] = data[:]
        targets_mmap[:] = targets[:]
        
        del data_mmap, targets_mmap  # Close files
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.LongTensor([self.targets[idx]])[0]

def demonstrate_basic_dataset():
    """Demonstrate basic Dataset and DataLoader usage"""
    print("=" * 60)
    print("BASIC DATASET AND DATALOADER")
    print("=" * 60)
    
    # Create synthetic dataset
    dataset = SyntheticDataset(n_samples=1000, n_classes=2)
    
    print(f"Dataset length: {len(dataset)}")
    print(f"Sample 0: {dataset[0]}")
    print(f"Data shape: {dataset[0][0].shape}")
    print(f"Target type: {type(dataset[0][1])}")
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=0  # Set to 0 for compatibility
    )
    
    print(f"\nDataLoader info:")
    print(f"Number of batches: {len(dataloader)}")
    print(f"Batch size: {dataloader.batch_size}")
    print(f"Shuffle: {dataloader.sampler is not None}")
    
    # Iterate through first few batches
    for i, (batch_data, batch_targets) in enumerate(dataloader):
        print(f"Batch {i}: data shape {batch_data.shape}, targets shape {batch_targets.shape}")
        if i >= 2:  # Show only first 3 batches
            break
    
    return dataset, dataloader

def demonstrate_data_splitting():
    """Demonstrate train/validation/test splits"""
    print("\n" + "=" * 60)
    print("DATA SPLITTING")
    print("=" * 60)
    
    # Create dataset
    dataset = SyntheticDataset(n_samples=1000, n_features=10, n_classes=3)
    
    # Method 1: random_split
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Original dataset size: {len(dataset)}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    # Method 2: Manual indices
    indices = torch.randperm(len(dataset))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)
    
    print(f"\nUsing Subset:")
    print(f"Train subset size: {len(train_subset)}")
    print(f"Validation subset size: {len(val_subset)}")
    print(f"Test subset size: {len(test_subset)}")
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    print(f"\nDataLoader batches:")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    return {
        'datasets': (train_dataset, val_dataset, test_dataset),
        'subsets': (train_subset, val_subset, test_subset),
        'loaders': (train_loader, val_loader, test_loader)
    }

def demonstrate_transforms():
    """Demonstrate data transformations"""
    print("\n" + "=" * 60)
    print("DATA TRANSFORMATIONS")
    print("=" * 60)
    
    # Define custom transforms
    class AddNoise:
        def __init__(self, noise_level=0.1):
            self.noise_level = noise_level
        
        def __call__(self, tensor):
            noise = torch.randn_like(tensor) * self.noise_level
            return tensor + noise
    
    class Normalize:
        def __init__(self, mean=0.0, std=1.0):
            self.mean = mean
            self.std = std
        
        def __call__(self, tensor):
            return (tensor - self.mean) / self.std
    
    # Create transform pipeline
    transform = transforms.Compose([
        AddNoise(noise_level=0.05),
        Normalize(mean=0.0, std=1.0)
    ])
    
    # Create datasets with and without transforms
    dataset_no_transform = SyntheticDataset(n_samples=100, transform=None)
    dataset_with_transform = SyntheticDataset(n_samples=100, transform=transform)
    
    # Compare samples
    original_sample = dataset_no_transform[0][0]
    transformed_sample = dataset_with_transform[0][0]
    
    print("Original sample statistics:")
    print(f"  Mean: {original_sample.mean().item():.4f}")
    print(f"  Std: {original_sample.std().item():.4f}")
    print(f"  Min: {original_sample.min().item():.4f}")
    print(f"  Max: {original_sample.max().item():.4f}")
    
    print("\nTransformed sample statistics:")
    print(f"  Mean: {transformed_sample.mean().item():.4f}")
    print(f"  Std: {transformed_sample.std().item():.4f}")
    print(f"  Min: {transformed_sample.min().item():.4f}")
    print(f"  Max: {transformed_sample.max().item():.4f}")
    
    return dataset_no_transform, dataset_with_transform, transform

def demonstrate_different_data_types():
    """Demonstrate datasets for different data types"""
    print("\n" + "=" * 60)
    print("DIFFERENT DATA TYPES")
    print("=" * 60)
    
    # 1. Image data (digits)
    print("1. Image Dataset (8x8 Digits):")
    digits_dataset = DigitsDataset()
    digits_loader = DataLoader(digits_dataset, batch_size=16, shuffle=True)
    
    print(f"  Dataset size: {len(digits_dataset)}")
    print(f"  Image shape: {digits_dataset[0][0].shape}")
    print(f"  Number of classes: {len(torch.unique(digits_dataset.targets))}")
    
    # 2. Time series data
    print("\n2. Time Series Dataset:")
    ts_dataset = TimeSeriesDataset(sequence_length=30, n_sequences=500, pattern='sine')
    ts_loader = DataLoader(ts_dataset, batch_size=8, shuffle=True)
    
    print(f"  Dataset size: {len(ts_dataset)}")
    print(f"  Sequence shape: {ts_dataset[0][0].shape}")
    print(f"  Target shape: {ts_dataset[0][1].shape}")
    
    # 3. Text data
    print("\n3. Text Dataset:")
    texts = ["hello world", "machine learning", "pytorch dataset", "data loader", "neural network"]
    labels = [0, 1, 1, 0, 1]  # Binary classification
    
    text_dataset = TextDataset(texts, labels)
    text_loader = DataLoader(text_dataset, batch_size=2, shuffle=True)
    
    print(f"  Dataset size: {len(text_dataset)}")
    print(f"  Vocabulary size: {text_dataset.vocab_size}")
    print(f"  Text shape: {text_dataset[0][0].shape}")
    print(f"  Sample text (encoded): {text_dataset[0][0][:10]}")
    
    # 4. Memory-mapped dataset
    print("\n4. Memory-Mapped Dataset:")
    mmap_dataset = MemoryMappedDataset('temp_data.dat', 'temp_targets.dat')
    mmap_loader = DataLoader(mmap_dataset, batch_size=32, shuffle=True)
    
    print(f"  Dataset size: {len(mmap_dataset)}")
    print(f"  Data shape: {mmap_dataset[0][0].shape}")
    print(f"  Memory-mapped: Efficient for large datasets")
    
    # Cleanup temporary files
    try:
        os.remove('temp_data.dat')
        os.remove('temp_targets.dat')
    except:
        pass
    
    return {
        'digits': (digits_dataset, digits_loader),
        'time_series': (ts_dataset, ts_loader),
        'text': (text_dataset, text_loader),
        'memory_mapped': (mmap_dataset, mmap_loader)
    }

def demonstrate_advanced_features():
    """Demonstrate advanced DataLoader features"""
    print("\n" + "=" * 60)
    print("ADVANCED DATALOADER FEATURES")
    print("=" * 60)
    
    dataset = SyntheticDataset(n_samples=1000, n_features=5)
    
    # 1. Different sampling strategies
    print("1. Sampling strategies:")
    
    # Sequential sampler
    sequential_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    print(f"  Sequential: {len(sequential_loader)} batches")
    
    # Random sampler
    random_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"  Random: {len(random_loader)} batches")
    
    # Custom weighted sampler (for imbalanced datasets)
    from torch.utils.data import WeightedRandomSampler
    
    # Simulate class imbalance
    class_counts = torch.bincount(dataset.targets)
    weights = 1.0 / class_counts.float()
    sample_weights = weights[dataset.targets]
    
    weighted_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )
    
    weighted_loader = DataLoader(dataset, batch_size=32, sampler=weighted_sampler)
    print(f"  Weighted: {len(weighted_loader)} batches")
    
    # 2. Custom collate function
    print("\n2. Custom collate function:")
    
    def custom_collate(batch):
        """Custom collate function that adds batch statistics"""
        data, targets = zip(*batch)
        
        data_tensor = torch.stack(data)
        targets_tensor = torch.stack(targets)
        
        # Add batch statistics
        batch_stats = {
            'mean': data_tensor.mean(dim=0),
            'std': data_tensor.std(dim=0),
            'size': len(batch)
        }
        
        return data_tensor, targets_tensor, batch_stats
    
    custom_loader = DataLoader(
        dataset, 
        batch_size=16, 
        shuffle=True, 
        collate_fn=custom_collate
    )
    
    sample_batch = next(iter(custom_loader))
    print(f"  Custom collate output length: {len(sample_batch)}")
    print(f"  Batch stats keys: {sample_batch[2].keys()}")
    
    # 3. Pin memory and num_workers
    print("\n3. Performance options:")
    
    # Pin memory (useful for GPU training)
    pin_memory_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=0  # Keep at 0 for demonstration
    )
    
    print(f"  Pin memory: {pin_memory_loader.pin_memory}")
    print(f"  Num workers: {pin_memory_loader.num_workers}")
    
    return {
        'samplers': (sequential_loader, random_loader, weighted_loader),
        'custom_collate': custom_loader,
        'performance': pin_memory_loader
    }

def benchmark_dataloader_performance():
    """Benchmark DataLoader performance with different settings"""
    print("\n" + "=" * 60)
    print("DATALOADER PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # Create a larger dataset for meaningful benchmarks
    large_dataset = SyntheticDataset(n_samples=10000, n_features=100)
    
    import time
    
    configs = [
        {"batch_size": 32, "num_workers": 0, "pin_memory": False},
        {"batch_size": 64, "num_workers": 0, "pin_memory": False},
        {"batch_size": 128, "num_workers": 0, "pin_memory": False},
        {"batch_size": 32, "num_workers": 0, "pin_memory": True},
    ]
    
    results = {}
    
    for config in configs:
        config_name = f"bs{config['batch_size']}_w{config['num_workers']}_pm{config['pin_memory']}"
        
        loader = DataLoader(large_dataset, **config)
        
        # Benchmark loading time
        start_time = time.time()
        total_samples = 0
        
        for batch_data, batch_targets in loader:
            total_samples += len(batch_data)
            # Simulate some computation
            _ = batch_data.mean()
        
        end_time = time.time()
        
        loading_time = end_time - start_time
        samples_per_second = total_samples / loading_time
        
        results[config_name] = {
            'time': loading_time,
            'samples_per_second': samples_per_second,
            'config': config
        }
        
        print(f"{config_name}:")
        print(f"  Time: {loading_time:.3f}s")
        print(f"  Samples/sec: {samples_per_second:.1f}")
    
    return results

def visualize_datasets_and_loaders(data_results):
    """Visualize different datasets and DataLoader behavior"""
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Synthetic 2D data
    ax1 = plt.subplot(4, 4, 1)
    dataset = SyntheticDataset(n_samples=500, n_classes=2)
    data, targets = dataset.data, dataset.targets
    
    for class_idx in torch.unique(targets):
        mask = targets == class_idx
        plt.scatter(data[mask, 0], data[mask, 1], label=f'Class {class_idx}', alpha=0.7)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Synthetic 2D Dataset')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Digits visualization
    ax2 = plt.subplot(4, 4, 2)
    digits_dataset, digits_loader = data_results['digits']
    
    # Show first few digit images
    fig_inner, axes_inner = plt.subplots(2, 5, figsize=(8, 3))
    for i in range(10):
        row, col = i // 5, i % 5
        image, label = digits_dataset[i]
        axes_inner[row, col].imshow(image.squeeze(), cmap='gray')
        axes_inner[row, col].set_title(f'Label: {label}')
        axes_inner[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('27_pytorch_intro/plots/digits_samples.png', dpi=150, bbox_inches='tight')
    plt.close(fig_inner)
    
    # For main plot, show digit class distribution
    digit_counts = torch.bincount(digits_dataset.targets)
    ax2.bar(range(len(digit_counts)), digit_counts.numpy())
    ax2.set_xlabel('Digit Class')
    ax2.set_ylabel('Count')
    ax2.set_title('Digits Dataset Class Distribution')
    ax2.grid(True, alpha=0.3)
    
    # 3. Time series visualization
    ax3 = plt.subplot(4, 4, 3)
    ts_dataset, ts_loader = data_results['time_series']
    
    # Show first few time series
    for i in range(5):
        sequence, target = ts_dataset[i]
        plt.plot(sequence.squeeze().numpy(), alpha=0.7, label=f'Series {i}')
    
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Time Series Samples')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Batch size effect on training
    ax4 = plt.subplot(4, 4, 4)
    batch_sizes = [16, 32, 64, 128, 256]
    num_batches = [len(DataLoader(dataset, batch_size=bs)) for bs in batch_sizes]
    
    ax4.plot(batch_sizes, num_batches, 'o-', linewidth=2, markersize=8)
    ax4.set_xlabel('Batch Size')
    ax4.set_ylabel('Number of Batches')
    ax4.set_title('Batch Size vs Number of Batches')
    ax4.grid(True, alpha=0.3)
    
    # 5. Data distribution before and after transforms
    ax5 = plt.subplot(4, 4, 5)
    original_data = SyntheticDataset(n_samples=1000, transform=None).data.flatten()
    
    class AddGaussianNoise:
        def __call__(self, tensor):
            return tensor + torch.randn_like(tensor) * 0.1
    
    transformed_data = SyntheticDataset(n_samples=1000, transform=AddGaussianNoise()).data.flatten()
    
    ax5.hist(original_data.numpy(), bins=50, alpha=0.7, label='Original', density=True)
    ax5.hist(transformed_data.numpy(), bins=50, alpha=0.7, label='With Noise', density=True)
    ax5.set_xlabel('Value')
    ax5.set_ylabel('Density')
    ax5.set_title('Effect of Transforms')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Memory usage comparison
    ax6 = plt.subplot(4, 4, 6)
    dataset_sizes = [100, 500, 1000, 5000, 10000]
    memory_usage = []
    
    for size in dataset_sizes:
        temp_dataset = SyntheticDataset(n_samples=size)
        # Approximate memory usage (in MB)
        mem_usage = temp_dataset.data.numel() * 4 / (1024**2)  # 4 bytes per float32
        memory_usage.append(mem_usage)
    
    ax6.plot(dataset_sizes, memory_usage, 'o-', linewidth=2, markersize=8, color='red')
    ax6.set_xlabel('Dataset Size')
    ax6.set_ylabel('Memory Usage (MB)')
    ax6.set_title('Dataset Memory Usage')
    ax6.grid(True, alpha=0.3)
    
    # 7. DataLoader iteration timing
    ax7 = plt.subplot(4, 4, 7)
    batch_sizes = [16, 32, 64, 128]
    iteration_times = []
    
    test_dataset = SyntheticDataset(n_samples=1000)
    
    for bs in batch_sizes:
        loader = DataLoader(test_dataset, batch_size=bs)
        
        import time
        start_time = time.time()
        for batch in loader:
            pass  # Just iterate
        end_time = time.time()
        
        iteration_times.append(end_time - start_time)
    
    ax7.bar(range(len(batch_sizes)), iteration_times, color='orange')
    ax7.set_xlabel('Batch Size')
    ax7.set_ylabel('Iteration Time (s)')
    ax7.set_title('DataLoader Iteration Performance')
    ax7.set_xticks(range(len(batch_sizes)))
    ax7.set_xticklabels(batch_sizes)
    ax7.grid(True, alpha=0.3)
    
    # 8. Shuffling effect visualization
    ax8 = plt.subplot(4, 4, 8)
    
    # Create a small dataset with clear order
    ordered_data = torch.arange(100).float().unsqueeze(1)
    ordered_targets = torch.zeros(100)
    ordered_dataset = torch.utils.data.TensorDataset(ordered_data, ordered_targets)
    
    # Compare shuffled vs non-shuffled
    no_shuffle_loader = DataLoader(ordered_dataset, batch_size=10, shuffle=False)
    shuffle_loader = DataLoader(ordered_dataset, batch_size=10, shuffle=True)
    
    no_shuffle_first_batch = next(iter(no_shuffle_loader))[0].flatten()
    shuffle_first_batch = next(iter(shuffle_loader))[0].flatten()
    
    ax8.plot(no_shuffle_first_batch.numpy(), 'o-', label='No Shuffle', linewidth=2)
    ax8.plot(shuffle_first_batch.numpy(), 's-', label='Shuffled', linewidth=2)
    ax8.set_xlabel('Sample Index in Batch')
    ax8.set_ylabel('Original Data Index')
    ax8.set_title('Shuffling Effect on First Batch')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Add more subplots for comprehensive analysis
    # 9. Class balance in batches
    ax9 = plt.subplot(4, 4, 9)
    imbalanced_targets = torch.cat([torch.zeros(800), torch.ones(200)])  # Imbalanced
    imbalanced_data = torch.randn(1000, 5)
    imbalanced_dataset = torch.utils.data.TensorDataset(imbalanced_data, imbalanced_targets)
    
    loader = DataLoader(imbalanced_dataset, batch_size=50, shuffle=True)
    batch_ratios = []
    
    for batch_data, batch_targets in loader:
        ratio = (batch_targets == 1).float().mean().item()
        batch_ratios.append(ratio)
    
    ax9.hist(batch_ratios, bins=15, alpha=0.7, color='purple')
    ax9.axvline(0.2, color='red', linestyle='--', label='True Ratio')
    ax9.set_xlabel('Batch Class 1 Ratio')
    ax9.set_ylabel('Frequency')
    ax9.set_title('Class Balance Across Batches')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # 10. Loading time vs dataset size
    ax10 = plt.subplot(4, 4, 10)
    sizes = [100, 500, 1000, 2000]
    loading_times = []
    
    for size in sizes:
        temp_dataset = SyntheticDataset(n_samples=size)
        loader = DataLoader(temp_dataset, batch_size=32)
        
        import time
        start_time = time.time()
        for _ in loader:
            pass
        loading_times.append(time.time() - start_time)
    
    ax10.plot(sizes, loading_times, 'o-', linewidth=2, markersize=8, color='green')
    ax10.set_xlabel('Dataset Size')
    ax10.set_ylabel('Loading Time (s)')
    ax10.set_title('Loading Time vs Dataset Size')
    ax10.grid(True, alpha=0.3)
    
    # Fill remaining subplots with useful information
    # 11. Text tokenization visualization
    ax11 = plt.subplot(4, 4, 11)
    text_dataset, text_loader = data_results['text']
    
    # Show vocabulary distribution
    vocab_usage = torch.zeros(text_dataset.vocab_size)
    for text_indices in text_dataset.encoded_texts:
        for idx in text_indices:
            if idx > 0:  # Skip padding
                vocab_usage[idx] += 1
    
    top_chars = torch.topk(vocab_usage, 10)
    char_labels = [text_dataset.idx_to_char[idx.item()] for idx in top_chars.indices]
    
    ax11.bar(range(10), top_chars.values.numpy())
    ax11.set_xlabel('Character')
    ax11.set_ylabel('Frequency')
    ax11.set_title('Top 10 Characters in Text Dataset')
    ax11.set_xticks(range(10))
    ax11.set_xticklabels(char_labels)
    ax11.grid(True, alpha=0.3)
    
    # 12. Feature correlation heatmap
    ax12 = plt.subplot(4, 4, 12)
    multi_feature_dataset = SyntheticDataset(n_samples=1000, n_features=8)
    correlation_matrix = torch.corrcoef(multi_feature_dataset.data.T)
    
    im = ax12.imshow(correlation_matrix.numpy(), cmap='coolwarm', vmin=-1, vmax=1)
    ax12.set_xlabel('Feature')
    ax12.set_ylabel('Feature')
    ax12.set_title('Feature Correlation Matrix')
    plt.colorbar(im, ax=ax12)
    
    plt.tight_layout()
    plt.savefig('27_pytorch_intro/plots/dataset_loader_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def comprehensive_dataset_demo():
    """Run comprehensive dataset and DataLoader demonstration"""
    print("PyTorch Dataset and DataLoader Comprehensive Demo")
    print("===============================================")
    
    # Run all demonstrations
    basic_results = demonstrate_basic_dataset()
    split_results = demonstrate_data_splitting()
    transform_results = demonstrate_transforms()
    data_type_results = demonstrate_different_data_types()
    advanced_results = demonstrate_advanced_features()
    benchmark_results = benchmark_dataloader_performance()
    
    # Create visualizations
    fig = visualize_datasets_and_loaders(data_type_results)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✅ Basic Dataset and DataLoader usage")
    print("✅ Train/validation/test data splitting")
    print("✅ Data transformations and augmentations")
    print("✅ Multiple data types (tabular, image, time series, text)")
    print("✅ Advanced DataLoader features (sampling, collate_fn)")
    print("✅ Performance benchmarking")
    print("✅ Memory-mapped datasets for large data")
    print("✅ Comprehensive visualizations and analysis")
    
    return {
        'basic': basic_results,
        'splitting': split_results,
        'transforms': transform_results,
        'data_types': data_type_results,
        'advanced': advanced_results,
        'benchmarks': benchmark_results,
        'visualization': fig
    }

if __name__ == "__main__":
    results = comprehensive_dataset_demo() 