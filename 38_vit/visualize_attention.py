"""
Attention Visualization for Vision Transformer

Tools for visualizing attention maps to understand what the model focuses on.
Includes utilities for creating attention heatmaps, attention rollout,
and multi-layer attention analysis.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from models import vit_tiny, vit_small, vit_base


class AttentionVisualizer:
    """
    Visualize attention maps from Vision Transformer.
    
    Provides various visualization methods:
    - Raw attention maps from specific heads/layers
    - Attention rollout (accumulated attention)
    - Class attention maps (CLS token attention)
    - Multi-head attention comparison
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
        # Get model parameters
        self.patch_size = model.patch_embed.patch_size
        self.num_patches = model.patch_embed.num_patches
        self.patches_per_side = int(np.sqrt(self.num_patches))
        
    def get_attention_maps(self, image, layer_idx=None):
        """
        Get attention maps from the model.
        
        Args:
            image: Input image tensor (1, 3, H, W)
            layer_idx: Specific layer index (if None, returns all layers)
            
        Returns:
            attention_maps: Attention weights
        """
        with torch.no_grad():
            image = image.to(self.device)
            _, attention_maps = self.model.forward_with_attention(image)
            
        if layer_idx is not None:
            return attention_maps[layer_idx]
        return attention_maps
    
    def visualize_attention_heads(self, image, layer_idx=-1, save_path=None):
        """
        Visualize attention from different heads in a specific layer.
        
        Args:
            image: Input image tensor (1, 3, H, W) 
            layer_idx: Layer index to visualize (-1 for last layer)
            save_path: Path to save the visualization
            
        Returns:
            Figure object
        """
        attention = self.get_attention_maps(image, layer_idx)
        
        # Handle the case where attention might be 3D or 4D
        if attention.dim() == 4:
            # (B, H, N, N) - take first batch element
            attention = attention[0]  # (H, N, N)
        elif attention.dim() == 3:
            # Already (H, N, N)
            pass
        else:
            raise ValueError(f"Unexpected attention tensor dimensions: {attention.shape}")
            
        num_heads = attention.shape[0]
        seq_len = attention.shape[1]
        
        # Focus on CLS token attention (first token)
        cls_attention = attention[:, 0, 1:].cpu().numpy()  # (num_heads, num_patches)
        
        # Reshape to spatial dimensions
        cls_attention = cls_attention.reshape(num_heads, self.patches_per_side, self.patches_per_side)
        
        # Create subplot for each head
        fig, axes = plt.subplots(1, num_heads, figsize=(4 * num_heads, 4))
        if num_heads == 1:
            axes = [axes]
            
        # Original image for reference
        original_image = self._tensor_to_image(image[0])
        
        for head_idx in range(num_heads):
            ax = axes[head_idx]
            
            # Upsample attention map to image size
            attention_map = cls_attention[head_idx]
            attention_resized = self._resize_attention_map(attention_map, original_image.shape[:2])
            
            # Show original image
            ax.imshow(original_image, alpha=0.7)
            
            # Overlay attention heatmap
            im = ax.imshow(attention_resized, alpha=0.6, cmap='jet')
            ax.set_title(f'Head {head_idx}')
            ax.axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.suptitle(f'Attention Heads - Layer {layer_idx}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention visualization saved to {save_path}")
            
        return fig
    
    def visualize_attention_rollout(self, image, save_path=None):
        """
        Visualize attention rollout across all layers.
        
        Attention rollout shows the cumulative attention from input to output.
        
        Args:
            image: Input image tensor (1, 3, H, W)
            save_path: Path to save the visualization
            
        Returns:
            Figure object
        """
        attention_maps = self.get_attention_maps(image)
        
        # Compute attention rollout
        rollout = self._compute_attention_rollout(attention_maps)
        
        # Focus on CLS token attention
        cls_rollout = rollout[0, 0, 1:].cpu().numpy()  # (num_patches,)
        cls_rollout = cls_rollout.reshape(self.patches_per_side, self.patches_per_side)
        
        # Original image
        original_image = self._tensor_to_image(image[0])
        
        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        ax1.imshow(original_image)
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # Raw attention rollout
        im2 = ax2.imshow(cls_rollout, cmap='jet')
        ax2.set_title('Attention Rollout')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        
        # Overlay on original image
        attention_resized = self._resize_attention_map(cls_rollout, original_image.shape[:2])
        ax3.imshow(original_image, alpha=0.7)
        im3 = ax3.imshow(attention_resized, alpha=0.6, cmap='jet')
        ax3.set_title('Attention Overlay')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention rollout saved to {save_path}")
            
        return fig
    
    def visualize_attention_layers(self, image, save_path=None):
        """
        Compare attention across different layers.
        
        Args:
            image: Input image tensor (1, 3, H, W)
            save_path: Path to save the visualization
            
        Returns:
            Figure object
        """
        attention_maps = self.get_attention_maps(image)
        num_layers = len(attention_maps)
        
        # Create grid of subplots
        cols = min(4, num_layers)
        rows = (num_layers + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        
        if rows == 1:
            axes = axes.reshape(1, -1) if num_layers > 1 else [[axes]]
        
        original_image = self._tensor_to_image(image[0])
        
        for layer_idx in range(num_layers):
            row = layer_idx // cols
            col = layer_idx % cols
            ax = axes[row][col]
            
            # Get attention from this layer
            attention = attention_maps[layer_idx]
            
            # Handle different tensor dimensions
            if attention.dim() == 4:
                attention = attention[0]  # Take first batch element: (H, N, N)
            
            # Average attention across heads
            avg_attention = attention.mean(dim=0)  # (seq_len, seq_len)
            
            # CLS token attention
            cls_attention = avg_attention[0, 1:].cpu().numpy()  # (num_patches,)
            cls_attention = cls_attention.reshape(self.patches_per_side, self.patches_per_side)
            
            # Resize and overlay
            attention_resized = self._resize_attention_map(cls_attention, original_image.shape[:2])
            ax.imshow(original_image, alpha=0.7)
            im = ax.imshow(attention_resized, alpha=0.6, cmap='jet')
            ax.set_title(f'Layer {layer_idx}')
            ax.axis('off')
            
        # Remove empty subplots
        for layer_idx in range(num_layers, rows * cols):
            row = layer_idx // cols
            col = layer_idx % cols
            axes[row][col].remove()
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Layer attention comparison saved to {save_path}")
            
        return fig
    
    def _compute_attention_rollout(self, attention_maps, discard_ratio=0.1):
        """
        Compute attention rollout as in the original paper.
        
        Args:
            attention_maps: List of attention tensors from all layers
            discard_ratio: Ratio of attention to discard (set to 0)
            
        Returns:
            Rolled out attention matrix
        """
        # Start with identity matrix
        result = torch.eye(attention_maps[0].size(-1), device=self.device)
        result = result.unsqueeze(0)  # Add batch dimension
        
        for attention in attention_maps:
            # Handle different tensor dimensions
            if attention.dim() == 4:
                attention_heads_fused = attention.mean(dim=1)  # (batch, seq_len, seq_len)
            elif attention.dim() == 3:
                attention_heads_fused = attention.mean(dim=0).unsqueeze(0)  # Add batch dim and average heads
            else:
                raise ValueError(f'Unexpected attention tensor dimensions: {attention.shape}')
            
            # Add residual connection (identity matrix)
            I = torch.eye(attention_heads_fused.size(-1), device=self.device).unsqueeze(0)
            a = (attention_heads_fused + I) / 2
            
            # Re-normalize
            a = a / a.sum(dim=-1, keepdim=True)
            
            # Multiply with previous result
            result = torch.matmul(a, result)
            
        return result
    
    def _tensor_to_image(self, tensor):
        """Convert normalized tensor to displayable image."""
        # Denormalize CIFAR-10 normalization
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        
        image = tensor.cpu().numpy().transpose(1, 2, 0)
        image = image * std + mean
        image = np.clip(image, 0, 1)
        
        return image
    
    def _resize_attention_map(self, attention_map, target_size):
        """Resize attention map to target image size."""
        attention_tensor = torch.from_numpy(attention_map).unsqueeze(0).unsqueeze(0).float()
        resized = F.interpolate(
            attention_tensor, size=target_size, mode='bicubic', align_corners=False
        )
        return resized.squeeze().numpy()


def create_attention_visualization_demo():
    """
    Create a comprehensive attention visualization demo.
    
    This function loads a trained model and creates various attention visualizations
    on sample CIFAR-10 images.
    """
    print("Creating attention visualization demo...")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load a sample model (you can replace with trained model)
    model = vit_tiny(num_classes=10)
    
    # Initialize with random weights for demo (replace with actual trained weights)
    print("Note: Using randomly initialized model for demo. Load trained weights for meaningful visualizations.")
    
    # Create visualizer
    visualizer = AttentionVisualizer(model, device)
    
    # Load sample CIFAR-10 images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Create some sample images (or load from CIFAR-10)
    try:
        dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        sample_images = [dataset[i][0] for i in range(3)]
        sample_labels = [dataset[i][1] for i in range(3)]
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    except:
        print("CIFAR-10 dataset not available, creating random sample images")
        sample_images = [torch.randn(3, 32, 32) for _ in range(3)]
        sample_labels = [0, 1, 2]
        class_names = ['class_0', 'class_1', 'class_2']
    
    # Create visualizations directory
    viz_dir = Path('plots')
    viz_dir.mkdir(exist_ok=True)
    
    # Generate visualizations for each sample image
    for i, (image, label) in enumerate(zip(sample_images, sample_labels)):
        image_batch = image.unsqueeze(0)  # Add batch dimension
        class_name = class_names[label]
        
        print(f"Creating visualizations for sample {i+1} (class: {class_name})...")
        
        # 1. Multi-head attention visualization
        fig1 = visualizer.visualize_attention_heads(
            image_batch, 
            layer_idx=-1,
            save_path=viz_dir / f'attention_heads_sample_{i+1}_{class_name}.png'
        )
        plt.close(fig1)
        
        # 2. Attention rollout
        fig2 = visualizer.visualize_attention_rollout(
            image_batch,
            save_path=viz_dir / f'attention_rollout_sample_{i+1}_{class_name}.png'
        )
        plt.close(fig2)
        
        # 3. Layer-wise attention comparison
        fig3 = visualizer.visualize_attention_layers(
            image_batch,
            save_path=viz_dir / f'attention_layers_sample_{i+1}_{class_name}.png'
        )
        plt.close(fig3)
    
    # Create a summary visualization combining multiple samples
    create_attention_summary(visualizer, sample_images[:3], sample_labels[:3], 
                           class_names, viz_dir / 'attention_map.png')
    
    print(f"All attention visualizations saved to {viz_dir}/")


def create_attention_summary(visualizer, images, labels, class_names, save_path):
    """
    Create a summary visualization combining multiple samples.
    
    This creates the main attention_map.png mentioned in the requirements.
    """
    num_samples = len(images)
    fig, axes = plt.subplots(2, num_samples, figsize=(6 * num_samples, 12))
    
    if num_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (image, label) in enumerate(zip(images, labels)):
        image_batch = image.unsqueeze(0)
        class_name = class_names[label]
        
        # Original image
        original_image = visualizer._tensor_to_image(image)
        axes[0, i].imshow(original_image)
        axes[0, i].set_title(f'Original: {class_name}')
        axes[0, i].axis('off')
        
        # Attention rollout
        attention_maps = visualizer.get_attention_maps(image_batch)
        rollout = visualizer._compute_attention_rollout(attention_maps)
        cls_rollout = rollout[0, 0, 1:].cpu().numpy()
        cls_rollout = cls_rollout.reshape(visualizer.patches_per_side, visualizer.patches_per_side)
        
        # Overlay attention on original image
        attention_resized = visualizer._resize_attention_map(cls_rollout, original_image.shape[:2])
        axes[1, i].imshow(original_image, alpha=0.7)
        im = axes[1, i].imshow(attention_resized, alpha=0.6, cmap='jet')
        axes[1, i].set_title(f'Attention: {class_name}')
        axes[1, i].axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
    
    plt.suptitle('Vision Transformer Attention Visualization', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Attention summary visualization saved to {save_path}")


if __name__ == "__main__":
    # Create comprehensive attention visualization demo
    create_attention_visualization_demo()
    
    print("\nAttention visualization demo completed! ✅")
    print("\nGenerated visualizations:")
    print("  - attention_heads_*.png: Multi-head attention from different heads")  
    print("  - attention_rollout_*.png: Attention rollout across all layers")
    print("  - attention_layers_*.png: Comparison across different layers")
    print("  - attention_map.png: Main summary visualization")
    print("\nTo use with trained models:")
    print("  1. Load your trained ViT model")
    print("  2. Replace the random model in create_attention_visualization_demo()")
    print("  3. Run the visualization on your test images")