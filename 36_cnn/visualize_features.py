#!/usr/bin/env python3
"""
CNN Feature Visualization Tools
===============================

Comprehensive tools for visualizing and understanding CNN internals:
- Filter visualizations
- Feature map activations
- Gradient-based feature importance
- Class activation maps
- Filter response analysis

Author: ML Fundamentals Course
Day: 36 - CNNs
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Any, Optional, Union
import cv2
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Framework imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False


class FilterVisualizer:
    """Visualize CNN filters and their responses."""
    
    @staticmethod
    def plot_filters_grid(filters: np.ndarray, title: str = "CNN Filters",
                         max_filters: int = 16, save_path: str = None) -> None:
        """
        Plot CNN filters in a grid layout.
        
        Args:
            filters: Filter weights array
            title: Plot title
            max_filters: Maximum number of filters to show
            save_path: Path to save the plot
        """
        if len(filters.shape) == 4:
            # Format: (num_filters, channels, height, width) or (height, width, in_ch, out_ch)
            if filters.shape[0] > filters.shape[-1]:  # Likely (h, w, in_ch, out_ch)
                filters = np.transpose(filters, (3, 2, 0, 1))
            
            num_filters = min(max_filters, filters.shape[0])
            filter_size = filters.shape[-1]  # Assuming square filters
            
            # Use first input channel for visualization
            filters_to_plot = filters[:num_filters, 0] if filters.shape[1] > 1 else filters[:num_filters, 0]
        else:
            print(f"Unexpected filter shape: {filters.shape}")
            return
        
        # Determine grid layout
        cols = min(8, int(np.ceil(np.sqrt(num_filters))))
        rows = (num_filters + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(num_filters):
            row = i // cols
            col = i % cols
            
            filter_img = filters_to_plot[i]
            
            # Normalize filter for better visualization
            filter_min, filter_max = filter_img.min(), filter_img.max()
            if filter_max > filter_min:
                filter_normalized = (filter_img - filter_min) / (filter_max - filter_min)
            else:
                filter_normalized = filter_img
            
            im = axes[row, col].imshow(filter_normalized, cmap='RdBu_r', interpolation='nearest')
            axes[row, col].set_title(f'F{i+1}', fontsize=10)
            axes[row, col].axis('off')
            
            # Add colorbar for first few filters
            if i < 4:
                plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for i in range(num_filters, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Filter visualization saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def analyze_filter_properties(filters: np.ndarray) -> Dict[str, Any]:
        """
        Analyze statistical properties of filters.
        
        Args:
            filters: Filter weights array
            
        Returns:
            Dictionary of filter statistics
        """
        if len(filters.shape) == 4 and filters.shape[0] > filters.shape[-1]:
            filters = np.transpose(filters, (3, 2, 0, 1))
        
        num_filters = filters.shape[0]
        
        # Calculate statistics
        stats = {
            'num_filters': num_filters,
            'filter_shape': filters.shape[1:],
            'mean_values': np.mean(filters, axis=(1, 2, 3)),
            'std_values': np.std(filters, axis=(1, 2, 3)),
            'l2_norms': np.linalg.norm(filters.reshape(num_filters, -1), axis=1),
            'sparsity': np.mean(np.abs(filters) < 1e-3, axis=(1, 2, 3))
        }
        
        return stats
    
    @staticmethod
    def plot_filter_statistics(stats: Dict[str, Any], save_path: str = None) -> None:
        """Plot filter statistics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Mean values
        axes[0, 0].hist(stats['mean_values'], bins=20, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Distribution of Filter Mean Values')
        axes[0, 0].set_xlabel('Mean Value')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Standard deviations
        axes[0, 1].hist(stats['std_values'], bins=20, alpha=0.7, color='lightcoral')
        axes[0, 1].set_title('Distribution of Filter Standard Deviations')
        axes[0, 1].set_xlabel('Standard Deviation')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].grid(True, alpha=0.3)
        
        # L2 norms
        axes[1, 0].hist(stats['l2_norms'], bins=20, alpha=0.7, color='lightgreen')
        axes[1, 0].set_title('Distribution of Filter L2 Norms')
        axes[1, 0].set_xlabel('L2 Norm')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Sparsity
        axes[1, 1].hist(stats['sparsity'], bins=20, alpha=0.7, color='plum')
        axes[1, 1].set_title('Distribution of Filter Sparsity')
        axes[1, 1].set_xlabel('Sparsity (fraction of near-zero weights)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Filter statistics saved to {save_path}")
        
        plt.show()


class FeatureMapVisualizer:
    """Visualize CNN feature maps and activations."""
    
    @staticmethod
    def plot_feature_maps_detailed(feature_maps: np.ndarray, original_image: np.ndarray,
                                  title: str = "Feature Maps", max_maps: int = 16,
                                  save_path: str = None) -> None:
        """
        Plot feature maps with original image for comparison.
        
        Args:
            feature_maps: Feature maps array (height, width, channels)
            original_image: Original input image
            title: Plot title
            max_maps: Maximum number of feature maps to show
            save_path: Path to save the plot
        """
        if len(feature_maps.shape) == 4:
            feature_maps = feature_maps[0]  # Remove batch dimension
        
        num_channels = feature_maps.shape[-1]
        num_maps = min(max_maps, num_channels)
        
        # Calculate grid layout
        cols = 4
        rows = (num_maps + cols - 1) // cols + 1  # +1 row for original and summary
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        # Original image
        if len(original_image.shape) == 3 and original_image.shape[-1] == 1:
            axes[0, 0].imshow(original_image[:, :, 0], cmap='gray')
        elif len(original_image.shape) == 3:
            axes[0, 0].imshow(original_image)
        else:
            axes[0, 0].imshow(original_image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Feature map statistics
        axes[0, 1].bar(['Min', 'Max', 'Mean', 'Std'], 
                      [feature_maps.min(), feature_maps.max(), 
                       feature_maps.mean(), feature_maps.std()])
        axes[0, 1].set_title('Feature Map Statistics')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Average activation across all channels
        avg_activation = np.mean(feature_maps, axis=-1)
        im = axes[0, 2].imshow(avg_activation, cmap='viridis', interpolation='bilinear')
        axes[0, 2].set_title('Average Activation')
        axes[0, 2].axis('off')
        plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        # Max activation across all channels
        max_activation = np.max(feature_maps, axis=-1)
        im = axes[0, 3].imshow(max_activation, cmap='hot', interpolation='bilinear')
        axes[0, 3].set_title('Max Activation')
        axes[0, 3].axis('off')
        plt.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.04)
        
        # Individual feature maps
        for i in range(num_maps):
            row = (i // cols) + 1
            col = i % cols
            
            feature_map = feature_maps[:, :, i]
            im = axes[row, col].imshow(feature_map, cmap='viridis', interpolation='bilinear')
            axes[row, col].set_title(f'Channel {i+1}')
            axes[row, col].axis('off')
            
            # Add activation statistics
            axes[row, col].text(0.02, 0.98, f'Max: {feature_map.max():.2f}', 
                              transform=axes[row, col].transAxes, 
                              verticalalignment='top', bbox=dict(boxstyle='round', 
                              facecolor='white', alpha=0.8), fontsize=8)
        
        # Hide unused subplots
        for i in range(num_maps, (rows - 1) * cols):
            row = (i // cols) + 1
            col = i % cols
            if row < rows:
                axes[row, col].axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature map visualization saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def analyze_activation_patterns(feature_maps: np.ndarray) -> Dict[str, Any]:
        """
        Analyze activation patterns in feature maps.
        
        Args:
            feature_maps: Feature maps array
            
        Returns:
            Dictionary of activation statistics
        """
        if len(feature_maps.shape) == 4:
            feature_maps = feature_maps[0]  # Remove batch dimension
        
        stats = {
            'total_activations': feature_maps.size,
            'active_neurons': np.sum(feature_maps > 0),
            'sparsity': np.mean(feature_maps == 0),
            'mean_activation': np.mean(feature_maps),
            'max_activation': np.max(feature_maps),
            'channel_means': np.mean(feature_maps, axis=(0, 1)),
            'channel_stds': np.std(feature_maps, axis=(0, 1)),
            'spatial_variance': np.var(feature_maps, axis=(0, 1))
        }
        
        # Dead filters (channels with very low activation)
        dead_threshold = 0.01
        stats['dead_filters'] = np.sum(stats['channel_means'] < dead_threshold)
        
        return stats


class LayerActivationVisualizer:
    """Visualize activations across different layers."""
    
    def __init__(self):
        self.activations = {}
    
    def extract_all_activations(self, model, input_sample: np.ndarray,
                              framework: str = 'keras') -> Dict[str, np.ndarray]:
        """
        Extract activations from all convolutional layers.
        
        Args:
            model: CNN model (Keras or PyTorch)
            input_sample: Input image sample
            framework: 'keras' or 'pytorch'
            
        Returns:
            Dictionary of layer activations
        """
        activations = {}
        
        if framework == 'keras' and KERAS_AVAILABLE:
            activations = self._extract_keras_activations(model, input_sample)
        elif framework == 'pytorch' and PYTORCH_AVAILABLE:
            activations = self._extract_pytorch_activations(model, input_sample)
        else:
            print(f"Framework {framework} not available or not supported")
        
        return activations
    
    def _extract_keras_activations(self, model: keras.Model, 
                                  input_sample: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract activations from Keras model."""
        activations = {}
        
        # Find all conv layers
        conv_layers = []
        for i, layer in enumerate(model.layers):
            if isinstance(layer, (tf.keras.layers.Conv2D)):
                conv_layers.append((i, layer))
        
        # Extract activations
        for layer_idx, layer in conv_layers:
            intermediate_model = keras.Model(inputs=model.input, outputs=layer.output)
            activation = intermediate_model.predict(input_sample.reshape(1, *input_sample.shape), 
                                                   verbose=0)
            activations[f"conv_{layer_idx}_{layer.name}"] = activation[0]
        
        return activations
    
    def _extract_pytorch_activations(self, model: nn.Module, 
                                    input_sample: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract activations from PyTorch model."""
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach().cpu().numpy()
            return hook
        
        # Register hooks for conv layers
        hooks = []
        layer_count = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                hook = module.register_forward_hook(hook_fn(f"conv_{layer_count}_{name}"))
                hooks.append(hook)
                layer_count += 1
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            if isinstance(input_sample, np.ndarray):
                input_tensor = torch.FloatTensor(input_sample).unsqueeze(0)
                if len(input_sample.shape) == 2:  # Add channel dimension
                    input_tensor = input_tensor.unsqueeze(0)
                elif len(input_sample.shape) == 3 and input_sample.shape[-1] in [1, 3]:
                    # Convert HWC to CHW
                    input_tensor = input_tensor.permute(0, 3, 1, 2)
            
            _ = model(input_tensor)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Remove batch dimension
        for key in activations:
            activations[key] = activations[key][0]
        
        return activations
    
    def visualize_layer_progression(self, activations: Dict[str, np.ndarray],
                                  original_image: np.ndarray, save_path: str = None) -> None:
        """
        Visualize how feature maps evolve through layers.
        
        Args:
            activations: Dictionary of layer activations
            original_image: Original input image
            save_path: Path to save the plot
        """
        layer_names = list(activations.keys())
        num_layers = len(layer_names)
        
        if num_layers == 0:
            print("No activations to visualize!")
            return
        
        # Create subplot grid
        fig, axes = plt.subplots(num_layers + 1, 4, figsize=(16, (num_layers + 1) * 3))
        if num_layers == 0:
            axes = axes.reshape(1, -1)
        
        # Original image
        if len(original_image.shape) == 3 and original_image.shape[-1] == 1:
            axes[0, 0].imshow(original_image[:, :, 0], cmap='gray')
        elif len(original_image.shape) == 3:
            axes[0, 0].imshow(original_image)
        else:
            axes[0, 0].imshow(original_image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Image statistics
        axes[0, 1].bar(['Min', 'Max', 'Mean', 'Std'], 
                      [original_image.min(), original_image.max(), 
                       original_image.mean(), original_image.std()])
        axes[0, 1].set_title('Image Statistics')
        
        # Hide unused subplots in first row
        axes[0, 2].axis('off')
        axes[0, 3].axis('off')
        
        # Layer activations
        for i, layer_name in enumerate(layer_names):
            activation = activations[layer_name]
            
            # Average across channels
            avg_activation = np.mean(activation, axis=-1) if len(activation.shape) == 3 else activation
            
            # Plot average activation
            im1 = axes[i+1, 0].imshow(avg_activation, cmap='viridis', interpolation='bilinear')
            axes[i+1, 0].set_title(f'{layer_name}\nAverage Activation')
            axes[i+1, 0].axis('off')
            plt.colorbar(im1, ax=axes[i+1, 0], fraction=0.046, pad=0.04)
            
            # Plot max activation
            max_activation = np.max(activation, axis=-1) if len(activation.shape) == 3 else activation
            im2 = axes[i+1, 1].imshow(max_activation, cmap='hot', interpolation='bilinear')
            axes[i+1, 1].set_title('Max Activation')
            axes[i+1, 1].axis('off')
            plt.colorbar(im2, ax=axes[i+1, 1], fraction=0.046, pad=0.04)
            
            # Activation statistics
            stats = ['Shape', 'Active%', 'Mean', 'Std']
            values = [
                f"{activation.shape}",
                f"{np.mean(activation > 0):.1%}",
                f"{activation.mean():.3f}",
                f"{activation.std():.3f}"
            ]
            
            axes[i+1, 2].barh(range(len(stats)), [1, np.mean(activation > 0), 
                                                 activation.mean(), activation.std()])
            axes[i+1, 2].set_yticks(range(len(stats)))
            axes[i+1, 2].set_yticklabels(stats)
            axes[i+1, 2].set_title('Layer Statistics')
            
            # Channel activation distribution
            if len(activation.shape) == 3:
                channel_means = np.mean(activation, axis=(0, 1))
                axes[i+1, 3].hist(channel_means, bins=min(20, len(channel_means)), 
                                alpha=0.7, color='orange')
                axes[i+1, 3].set_title('Channel Activation Distribution')
                axes[i+1, 3].set_xlabel('Mean Activation')
                axes[i+1, 3].set_ylabel('Count')
            else:
                axes[i+1, 3].axis('off')
        
        plt.suptitle('Layer Activation Progression', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Layer progression visualization saved to {save_path}")
        
        plt.show()


class ConvolutionVisualizer:
    """Visualize convolution operations step by step."""
    
    @staticmethod
    def demonstrate_convolution(image: np.ndarray, kernel: np.ndarray, 
                              stride: int = 1, padding: int = 0,
                              save_path: str = None) -> None:
        """
        Demonstrate convolution operation step by step.
        
        Args:
            image: Input image (2D)
            kernel: Convolution kernel (2D)
            stride: Convolution stride
            padding: Padding amount
            save_path: Path to save the plot
        """
        # Add padding
        if padding > 0:
            image_padded = np.pad(image, padding, mode='constant', constant_values=0)
        else:
            image_padded = image
        
        # Calculate output dimensions
        h_out = (image_padded.shape[0] - kernel.shape[0]) // stride + 1
        w_out = (image_padded.shape[1] - kernel.shape[1]) // stride + 1
        
        # Perform convolution
        output = np.zeros((h_out, w_out))
        step_visualizations = []
        
        for h in range(h_out):
            for w in range(w_out):
                h_start = h * stride
                h_end = h_start + kernel.shape[0]
                w_start = w * stride
                w_end = w_start + kernel.shape[1]
                
                region = image_padded[h_start:h_end, w_start:w_end]
                conv_result = np.sum(region * kernel)
                output[h, w] = conv_result
                
                # Store visualization data for first few steps
                if len(step_visualizations) < 6:
                    step_visualizations.append({
                        'region': region.copy(),
                        'position': (h, w),
                        'result': conv_result
                    })
        
        # Create visualization
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        
        # Original image
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Kernel
        im = axes[0, 1].imshow(kernel, cmap='RdBu')
        axes[0, 1].set_title('Convolution Kernel')
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # Padded image
        axes[0, 2].imshow(image_padded, cmap='gray')
        axes[0, 2].set_title(f'Padded Image (p={padding})')
        axes[0, 2].axis('off')
        
        # Final output
        im = axes[0, 3].imshow(output, cmap='viridis')
        axes[0, 3].set_title('Convolution Output')
        axes[0, 3].axis('off')
        plt.colorbar(im, ax=axes[0, 3], fraction=0.046, pad=0.04)
        
        # Step-by-step convolution
        for i, step_data in enumerate(step_visualizations):
            row = (i // 2) + 1
            col = (i % 2) * 2
            
            # Show region being convolved
            im1 = axes[row, col].imshow(step_data['region'], cmap='gray')
            axes[row, col].set_title(f'Step {i+1}: Region at {step_data["position"]}')
            axes[row, col].axis('off')
            
            # Show element-wise multiplication
            multiplication = step_data['region'] * kernel
            im2 = axes[row, col+1].imshow(multiplication, cmap='RdBu')
            axes[row, col+1].set_title(f'Element-wise Product\nSum = {step_data["result"]:.2f}')
            axes[row, col+1].axis('off')
            plt.colorbar(im2, ax=axes[row, col+1], fraction=0.046, pad=0.04)
        
        plt.suptitle('Convolution Operation Demonstration', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Convolution demonstration saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def compare_pooling_operations(feature_map: np.ndarray, pool_size: int = 2,
                                 save_path: str = None) -> None:
        """
        Compare max pooling vs average pooling.
        
        Args:
            feature_map: Input feature map (2D)
            pool_size: Pooling window size
            save_path: Path to save the plot
        """
        h, w = feature_map.shape
        h_out = h // pool_size
        w_out = w // pool_size
        
        # Max pooling
        max_pooled = np.zeros((h_out, w_out))
        avg_pooled = np.zeros((h_out, w_out))
        
        for i in range(h_out):
            for j in range(w_out):
                region = feature_map[i*pool_size:(i+1)*pool_size, 
                                   j*pool_size:(j+1)*pool_size]
                max_pooled[i, j] = np.max(region)
                avg_pooled[i, j] = np.mean(region)
        
        # Visualize
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original
        im1 = axes[0].imshow(feature_map, cmap='viridis')
        axes[0].set_title('Original Feature Map')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        
        # Max pooling
        im2 = axes[1].imshow(max_pooled, cmap='viridis', interpolation='nearest')
        axes[1].set_title(f'Max Pooling ({pool_size}x{pool_size})')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Average pooling
        im3 = axes[2].imshow(avg_pooled, cmap='viridis', interpolation='nearest')
        axes[2].set_title(f'Average Pooling ({pool_size}x{pool_size})')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Pooling comparison saved to {save_path}")
        
        plt.show()


def create_synthetic_images() -> Dict[str, np.ndarray]:
    """Create synthetic images to test CNN visualizations."""
    print("Creating synthetic test images...")
    
    images = {}
    
    # Edge detection test image
    edge_img = np.zeros((28, 28))
    edge_img[10:18, 5:23] = 1.0  # Rectangle
    edge_img[12:16, 7:21] = 0.0  # Inner rectangle
    images['edges'] = edge_img
    
    # Texture test image
    texture_img = np.random.rand(28, 28)
    texture_img = cv2.GaussianBlur(texture_img, (5, 5), 1.0) if 'cv2' in globals() else texture_img
    images['texture'] = texture_img
    
    # Gradient test image
    gradient_img = np.zeros((28, 28))
    for i in range(28):
        gradient_img[i, :] = i / 27.0
    images['gradient'] = gradient_img
    
    # Checkerboard pattern
    checker_img = np.zeros((28, 28))
    for i in range(0, 28, 4):
        for j in range(0, 28, 4):
            if (i // 4 + j // 4) % 2 == 0:
                checker_img[i:i+4, j:j+4] = 1.0
    images['checkerboard'] = checker_img
    
    return images


def demonstrate_convolution_basics() -> None:
    """Demonstrate basic convolution operations."""
    print("\nDemonstrating Convolution Basics...")
    print("=" * 40)
    
    # Create synthetic images
    images = create_synthetic_images()
    
    # Define common kernels
    kernels = {
        'Vertical Edge': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
        'Horizontal Edge': np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
        'Blur': np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
        'Sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    }
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # Demonstrate convolution for each kernel
    for i, (kernel_name, kernel) in enumerate(kernels.items()):
        print(f"Demonstrating {kernel_name} kernel...")
        
        # Use edge image for demonstration
        ConvolutionVisualizer.demonstrate_convolution(
            images['edges'], kernel, stride=1, padding=0,
            save_path=f'plots/convolution_demo_{kernel_name.lower().replace(" ", "_")}.png'
        )
    
    # Demonstrate pooling
    print("Demonstrating pooling operations...")
    ConvolutionVisualizer.compare_pooling_operations(
        images['texture'], pool_size=2,
        save_path='plots/pooling_comparison.png'
    )


def run_comprehensive_visualization_experiments() -> Dict[str, Any]:
    """Run comprehensive CNN visualization experiments."""
    print("Starting Comprehensive CNN Visualization Experiments")
    print("=" * 60)
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    results = {}
    
    # Basic convolution demonstrations
    print("\n" + "=" * 50)
    print("EXPERIMENT 1: Basic Convolution Operations")
    print("=" * 50)
    
    demonstrate_convolution_basics()
    
    # Test with Keras models if available
    if KERAS_AVAILABLE:
        print("\n" + "=" * 50)
        print("EXPERIMENT 2: Keras Model Visualizations")
        print("=" * 50)
        
        try:
            # Load MNIST for quick testing
            (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
            X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
            X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
            
            # Create and train a simple model
            model = create_simple_cnn((28, 28, 1), 10)
            print("Training simple CNN for visualization...")
            
            # Quick training (just 3 epochs for demo)
            history = model.fit(X_train[:5000], y_train[:5000], 
                              validation_data=(X_test[:1000], y_test[:1000]),
                              epochs=3, batch_size=128, verbose=0)
            
            # Visualizations
            print("Creating Keras visualizations...")
            
            # Filter visualization
            visualize_keras_filters(model, save_path='plots/keras_filters_detailed.png')
            
            # Feature maps
            sample_image = X_test[0]
            visualize_keras_feature_maps(model, sample_image, 
                                       save_path='plots/keras_feature_maps_detailed.png')
            
            # Layer progression
            visualizer = LayerActivationVisualizer()
            activations = visualizer.extract_all_activations(model, sample_image, 'keras')
            visualizer.visualize_layer_progression(activations, sample_image,
                                                 'plots/keras_layer_progression.png')
            
            results['keras_model'] = {
                'model': model,
                'history': history,
                'activations': activations
            }
            
        except Exception as e:
            print(f"Keras visualization failed: {e}")
    
    # Create summary visualization
    create_visualization_summary()
    
    print("\nVisualization experiments completed!")
    print("Check the plots/ directory for all visualizations.")
    
    return results


def create_visualization_summary() -> None:
    """Create a summary of all CNN concepts."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # CNN Architecture diagram (conceptual)
    axes[0, 0].text(0.5, 0.8, 'INPUT\n28×28×1', ha='center', va='center', 
                   bbox=dict(boxstyle='round', facecolor='lightblue'), fontsize=12)
    axes[0, 0].text(0.5, 0.6, '↓ CONV 3×3', ha='center', va='center', fontsize=10)
    axes[0, 0].text(0.5, 0.4, 'FEATURE MAPS\n26×26×32', ha='center', va='center', 
                   bbox=dict(boxstyle='round', facecolor='lightgreen'), fontsize=12)
    axes[0, 0].text(0.5, 0.2, '↓ POOL 2×2', ha='center', va='center', fontsize=10)
    axes[0, 0].text(0.5, 0.0, 'OUTPUT\n13×13×32', ha='center', va='center', 
                   bbox=dict(boxstyle='round', facecolor='lightcoral'), fontsize=12)
    axes[0, 0].set_title('CNN Layer Flow')
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].axis('off')
    
    # Filter examples
    edge_filter = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    im1 = axes[0, 1].imshow(edge_filter, cmap='RdBu')
    axes[0, 1].set_title('Horizontal Edge Filter')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Pooling visualization
    sample_data = np.random.rand(8, 8)
    pooled_data = sample_data[::2, ::2]  # Simple 2x2 max pooling simulation
    
    axes[0, 2].imshow(sample_data, cmap='viridis')
    axes[0, 2].set_title('Before Pooling (8×8)')
    axes[0, 2].axis('off')
    
    # CNN advantages
    advantages_text = """
    CNN Advantages:
    
    • Parameter Sharing
      Same filter across image
    
    • Translation Invariance
      Detects features anywhere
    
    • Hierarchical Learning
      Simple → Complex features
    
    • Spatial Structure
      Preserves image topology
    """
    
    axes[1, 0].text(0.05, 0.95, advantages_text, transform=axes[1, 0].transAxes,
                   verticalalignment='top', fontsize=11, 
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[1, 0].set_title('Key CNN Concepts')
    axes[1, 0].axis('off')
    
    # Pooling result
    im2 = axes[1, 1].imshow(pooled_data, cmap='viridis', interpolation='nearest')
    axes[1, 1].set_title('After Pooling (4×4)')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # Feature hierarchy
    hierarchy_text = """
    Feature Hierarchy:
    
    Layer 1: Edges, Lines
    ──────────────────
    
    Layer 2: Shapes, Textures
    ──────────────────────
    
    Layer 3: Parts, Patterns
    ─────────────────────
    
    Layer 4: Objects, Concepts
    ──────────────────────────
    """
    
    axes[1, 2].text(0.05, 0.95, hierarchy_text, transform=axes[1, 2].transAxes,
                   verticalalignment='top', fontsize=11, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    axes[1, 2].set_title('Hierarchical Feature Learning')
    axes[1, 2].axis('off')
    
    plt.suptitle('CNN Concepts and Architecture Overview', fontsize=16)
    plt.tight_layout()
    plt.savefig('plots/cnn_concepts_overview.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # Set random seeds
    np.random.seed(42)
    if KERAS_AVAILABLE:
        tf.random.set_seed(42)
    
    results = run_comprehensive_visualization_experiments()