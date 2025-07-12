"""
SVD Image Compression - Advanced Implementation
==============================================

This module implements SVD-based image compression including:
- Grayscale image compression using rank-k approximation
- PSNR (Peak Signal-to-Noise Ratio) analysis
- Compression ratio calculations
- Visual quality assessment
- Progressive compression demonstration

Mathematical Foundation:
- Image matrix I = UΣV^T
- Rank-k approximation: I_k = Σ(i=1 to k) σ_i * u_i * v_i^T
- PSNR = 20 * log10(MAX_I / RMSE)
- Compression ratio = original_size / compressed_size
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import requests
from io import BytesIO
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SVDImageCompressor:
    """
    SVD-based image compression with quality analysis.
    
    Features:
    - Load and process grayscale images
    - Apply SVD compression with various ranks
    - Calculate PSNR and compression ratios
    - Generate comparative visualizations
    - Progressive compression analysis
    """
    
    def __init__(self, image_path=None, image_array=None):
        """
        Initialize image compressor.
        
        Parameters:
        -----------
        image_path : str, optional
            Path to image file
        image_array : np.ndarray, optional
            Image array (grayscale)
        """
        self.original_image = None
        self.compressed_images = {}
        self.compression_stats = {}
        
        if image_path:
            self.load_image(image_path)
        elif image_array is not None:
            self.original_image = image_array.astype(np.float64)
            
    def load_image(self, image_path):
        """
        Load image from file or URL.
        
        Parameters:
        -----------
        image_path : str
            Path to image file or URL
        """
        try:
            if image_path.startswith('http'):
                # Download image from URL
                response = requests.get(image_path)
                img = Image.open(BytesIO(response.content))
            else:
                # Load local image
                img = Image.open(image_path)
                
            # Convert to grayscale and numpy array
            img_gray = img.convert('L')
            self.original_image = np.array(img_gray, dtype=np.float64)
            
            print(f"Image loaded successfully!")
            print(f"Shape: {self.original_image.shape}")
            print(f"Data type: {self.original_image.dtype}")
            print(f"Value range: [{self.original_image.min():.1f}, {self.original_image.max():.1f}]")
            
        except Exception as e:
            print(f"Error loading image: {e}")
            # Create a synthetic test image
            self.create_test_image()
            
    def create_test_image(self, size=(256, 256)):
        """
        Create a synthetic test image for demonstration.
        
        Parameters:
        -----------
        size : tuple
            Image dimensions (height, width)
        """
        print("Creating synthetic test image...")
        
        h, w = size
        x = np.linspace(-2, 2, w)
        y = np.linspace(-2, 2, h)
        X, Y = np.meshgrid(x, y)
        
        # Create interesting pattern with multiple frequency components
        image = (
            128 + 64 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y) +
            32 * np.sin(4 * np.pi * X) +
            16 * np.cos(6 * np.pi * Y) +
            8 * np.sin(8 * np.pi * (X + Y))
        )
        
        # Add some noise
        image += 5 * np.random.randn(h, w)
        
        # Clip to valid range
        self.original_image = np.clip(image, 0, 255)
        
        print(f"Synthetic image created: {self.original_image.shape}")
        
    def compress_image(self, k):
        """
        Compress image using rank-k SVD approximation.
        
        Parameters:
        -----------
        k : int
            Number of singular values to keep
            
        Returns:
        --------
        np.ndarray : Compressed image
        """
        if self.original_image is None:
            raise ValueError("No image loaded")
            
        # Perform SVD
        U, s, Vt = np.linalg.svd(self.original_image, full_matrices=False)
        
        # Keep only top-k components
        k = min(k, len(s))
        U_k = U[:, :k]
        s_k = s[:k]
        Vt_k = Vt[:k, :]
        
        # Reconstruct image
        compressed = U_k @ np.diag(s_k) @ Vt_k
        
        # Clip to valid pixel range
        compressed = np.clip(compressed, 0, 255)
        
        return compressed, U_k, s_k, Vt_k
    
    def calculate_psnr(self, original, compressed):
        """
        Calculate Peak Signal-to-Noise Ratio.
        
        Parameters:
        -----------
        original : np.ndarray
            Original image
        compressed : np.ndarray
            Compressed image
            
        Returns:
        --------
        float : PSNR in dB
        """
        mse = np.mean((original - compressed) ** 2)
        if mse == 0:
            return float('inf')
        
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr
    
    def calculate_compression_ratio(self, k):
        """
        Calculate compression ratio for rank-k approximation.
        
        Parameters:
        -----------
        k : int
            Number of singular values kept
            
        Returns:
        --------
        float : Compression ratio
        """
        h, w = self.original_image.shape
        original_size = h * w
        
        # Compressed representation: U_k + s_k + Vt_k
        compressed_size = k * (h + 1 + w)
        
        return original_size / compressed_size
    
    def calculate_storage_savings(self, k):
        """
        Calculate storage savings percentage.
        
        Parameters:
        -----------
        k : int
            Number of singular values kept
            
        Returns:
        --------
        float : Storage savings percentage
        """
        ratio = self.calculate_compression_ratio(k)
        return (1 - 1/ratio) * 100
    
    def analyze_compression_range(self, k_values):
        """
        Analyze compression for multiple k values.
        
        Parameters:
        -----------
        k_values : list
            List of k values to test
            
        Returns:
        --------
        dict : Compression analysis results
        """
        if self.original_image is None:
            raise ValueError("No image loaded")
            
        results = {
            'k_values': [],
            'psnr_values': [],
            'compression_ratios': [],
            'storage_savings': [],
            'compressed_images': []
        }
        
        print("Analyzing compression for different ranks...")
        print("Rank |   PSNR   | Comp. Ratio | Storage Savings")
        print("-" * 50)
        
        for k in k_values:
            compressed, U_k, s_k, Vt_k = self.compress_image(k)
            psnr = self.calculate_psnr(self.original_image, compressed)
            ratio = self.calculate_compression_ratio(k)
            savings = self.calculate_storage_savings(k)
            
            results['k_values'].append(k)
            results['psnr_values'].append(psnr)
            results['compression_ratios'].append(ratio)
            results['storage_savings'].append(savings)
            results['compressed_images'].append(compressed)
            
            print(f"{k:4d} | {psnr:8.2f} | {ratio:11.2f} | {savings:13.1f}%")
            
        return results
    
    def plot_compression_analysis(self, k_values, save_path=None):
        """
        Plot comprehensive compression analysis.
        
        Parameters:
        -----------
        k_values : list
            List of k values to analyze
        save_path : str, optional
            Path to save the plot
        """
        results = self.analyze_compression_range(k_values)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. PSNR vs Rank
        ax1.plot(results['k_values'], results['psnr_values'], 'bo-', linewidth=2, markersize=6)
        ax1.set_xlabel('Rank k')
        ax1.set_ylabel('PSNR (dB)')
        ax1.set_title('Image Quality vs Compression Rank')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=30, color='r', linestyle='--', alpha=0.7, label='Good Quality (30dB)')
        ax1.axhline(y=40, color='g', linestyle='--', alpha=0.7, label='Excellent Quality (40dB)')
        ax1.legend()
        
        # 2. Compression Ratio vs Rank
        ax2.plot(results['k_values'], results['compression_ratios'], 'ro-', linewidth=2, markersize=6)
        ax2.set_xlabel('Rank k')
        ax2.set_ylabel('Compression Ratio')
        ax2.set_title('Compression Ratio vs Rank')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1, color='k', linestyle='--', alpha=0.7, label='No Compression')
        ax2.legend()
        
        # 3. PSNR vs Compression Ratio (Rate-Distortion)
        ax3.plot(results['compression_ratios'], results['psnr_values'], 'go-', linewidth=2, markersize=6)
        ax3.set_xlabel('Compression Ratio')
        ax3.set_ylabel('PSNR (dB)')
        ax3.set_title('Rate-Distortion Curve')
        ax3.grid(True, alpha=0.3)
        
        # Add annotations for specific points
        for i, k in enumerate(results['k_values'][::max(1, len(results['k_values'])//5)]):
            idx = results['k_values'].index(k)
            ax3.annotate(f'k={k}', 
                        (results['compression_ratios'][idx], results['psnr_values'][idx]),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 4. Storage Savings vs Rank
        ax4.plot(results['k_values'], results['storage_savings'], 'mo-', linewidth=2, markersize=6)
        ax4.set_xlabel('Rank k')
        ax4.set_ylabel('Storage Savings (%)')
        ax4.set_title('Storage Savings vs Rank')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='50% Savings')
        ax4.axhline(y=90, color='r', linestyle='--', alpha=0.7, label='90% Savings')
        ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Compression analysis plot saved to {save_path}")
            
        plt.show()
        
        return results
    
    def plot_compressed_images(self, k_values, save_path=None):
        """
        Plot original and compressed images for comparison.
        
        Parameters:
        -----------
        k_values : list
            List of k values to display
        save_path : str, optional
            Path to save the plot
        """
        if self.original_image is None:
            raise ValueError("No image loaded")
            
        n_images = len(k_values) + 1  # +1 for original
        cols = min(4, n_images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        # Plot original image
        ax = axes[0, 0]
        ax.imshow(self.original_image, cmap='gray', vmin=0, vmax=255)
        ax.set_title('Original Image')
        ax.axis('off')
        
        # Plot compressed images
        for i, k in enumerate(k_values):
            row = (i + 1) // cols
            col = (i + 1) % cols
            
            compressed, _, _, _ = self.compress_image(k)
            psnr = self.calculate_psnr(self.original_image, compressed)
            ratio = self.calculate_compression_ratio(k)
            
            ax = axes[row, col]
            ax.imshow(compressed, cmap='gray', vmin=0, vmax=255)
            ax.set_title(f'Rank {k}\nPSNR: {psnr:.1f}dB\nRatio: {ratio:.1f}x')
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(n_images, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Compressed images plot saved to {save_path}")
            
        plt.show()
    
    def plot_singular_values(self, save_path=None):
        """
        Plot singular values of the image matrix.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        """
        if self.original_image is None:
            raise ValueError("No image loaded")
            
        # Perform SVD
        U, s, Vt = np.linalg.svd(self.original_image, full_matrices=False)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Singular values (linear scale)
        ax1.plot(range(1, len(s) + 1), s, 'b-', linewidth=2)
        ax1.set_xlabel('Index')
        ax1.set_ylabel('Singular Value')
        ax1.set_title('Singular Values (Linear Scale)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Singular values (log scale)
        ax2.semilogy(range(1, len(s) + 1), s, 'r-', linewidth=2)
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Singular Value (log scale)')
        ax2.set_title('Singular Values (Log Scale)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Cumulative energy
        cumulative_energy = np.cumsum(s ** 2) / np.sum(s ** 2)
        ax3.plot(range(1, len(s) + 1), cumulative_energy, 'g-', linewidth=2)
        ax3.axhline(y=0.9, color='r', linestyle='--', alpha=0.7, label='90% Energy')
        ax3.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='95% Energy')
        ax3.axhline(y=0.99, color='purple', linestyle='--', alpha=0.7, label='99% Energy')
        ax3.set_xlabel('Number of Components')
        ax3.set_ylabel('Cumulative Energy')
        ax3.set_title('Cumulative Energy Preservation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Energy distribution (first 50 components)
        energy_ratios = (s ** 2) / np.sum(s ** 2)
        n_show = min(50, len(s))
        ax4.bar(range(1, n_show + 1), energy_ratios[:n_show], alpha=0.7)
        ax4.set_xlabel('Component Index')
        ax4.set_ylabel('Energy Ratio')
        ax4.set_title(f'Energy Distribution (First {n_show} Components)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Singular values plot saved to {save_path}")
            
        plt.show()
        
        return s

def demonstrate_image_compression():
    """
    Demonstrate SVD image compression with comprehensive analysis.
    """
    print("=" * 70)
    print("SVD IMAGE COMPRESSION DEMONSTRATION")
    print("=" * 70)
    
    # Try to load a famous test image
    test_images = [
        "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png",
        "https://homepages.cae.wisc.edu/~ece533/images/lena.png",
        "https://www.ece.rice.edu/~wakin/images/lena512.bmp"
    ]
    
    compressor = None
    for img_url in test_images:
        try:
            print(f"Attempting to load image from: {img_url}")
            compressor = SVDImageCompressor(img_url)
            break
        except:
            continue
    
    if compressor is None or compressor.original_image is None:
        print("Could not load test image from URLs, creating synthetic image...")
        compressor = SVDImageCompressor()
        compressor.create_test_image((256, 256))
    
    return compressor

def main():
    """
    Main function to run SVD image compression analysis.
    """
    print("🖼️ SVD IMAGE COMPRESSION - ADVANCED IMPLEMENTATION")
    print("=" * 80)
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    # 1. Load and analyze image
    compressor = demonstrate_image_compression()
    
    # 2. Analyze singular values
    print("\n📊 ANALYZING IMAGE SINGULAR VALUES...")
    print("-" * 50)
    singular_values = compressor.plot_singular_values('plots/svd_image_singular_values.png')
    
    print(f"Image dimensions: {compressor.original_image.shape}")
    print(f"Number of singular values: {len(singular_values)}")
    print(f"Largest singular value: {singular_values[0]:.2f}")
    print(f"Smallest singular value: {singular_values[-1]:.2e}")
    print(f"Condition number: {singular_values[0] / singular_values[-1]:.2e}")
    
    # 3. Test different compression levels
    print("\n🗜️ TESTING COMPRESSION LEVELS...")
    print("-" * 40)
    
    # Define test ranks based on image size
    max_rank = min(compressor.original_image.shape)
    test_ranks = [1, 5, 10, 20, 50, 100, 200]
    test_ranks = [k for k in test_ranks if k <= max_rank]
    
    # Add some percentage-based ranks
    percentage_ranks = [int(0.01 * max_rank), int(0.05 * max_rank), int(0.1 * max_rank), int(0.25 * max_rank)]
    test_ranks.extend([k for k in percentage_ranks if k not in test_ranks and k <= max_rank])
    test_ranks = sorted(list(set(test_ranks)))
    
    print(f"Testing ranks: {test_ranks}")
    
    # 4. Comprehensive compression analysis
    results = compressor.plot_compression_analysis(test_ranks, 'plots/svd_image_compression_analysis.png')
    
    # 5. Visual comparison of compressed images
    display_ranks = [1, 5, 20, 50] if max_rank >= 50 else test_ranks[:4]
    compressor.plot_compressed_images(display_ranks, 'plots/svd_image_compression_comparison.png')
    
    # 6. Find optimal compression points
    print("\n🎯 OPTIMAL COMPRESSION ANALYSIS...")
    print("-" * 45)
    
    # Find rank for different quality thresholds
    quality_thresholds = [25, 30, 35, 40]  # PSNR in dB
    
    print("Quality Threshold | Optimal Rank | Compression Ratio | Storage Savings")
    print("-" * 70)
    
    for threshold in quality_thresholds:
        # Find the smallest rank that achieves the threshold
        optimal_rank = None
        for i, psnr in enumerate(results['psnr_values']):
            if psnr >= threshold:
                optimal_rank = results['k_values'][i]
                break
        
        if optimal_rank:
            idx = results['k_values'].index(optimal_rank)
            ratio = results['compression_ratios'][idx]
            savings = results['storage_savings'][idx]
            print(f"{threshold:15.0f}dB | {optimal_rank:12d} | {ratio:17.2f} | {savings:13.1f}%")
        else:
            print(f"{threshold:15.0f}dB | {'Not achievable':>12} | {'N/A':>17} | {'N/A':>13}")
    
    # 7. Storage efficiency analysis
    print("\n💾 STORAGE EFFICIENCY ANALYSIS...")
    print("-" * 42)
    
    # Calculate bits per pixel for different compressions
    h, w = compressor.original_image.shape
    original_bpp = 8  # 8 bits per pixel for grayscale
    
    print("Rank | Compression Ratio | Effective BPP | Quality (PSNR)")
    print("-" * 55)
    
    for i, k in enumerate(results['k_values'][::max(1, len(results['k_values'])//8)]):
        idx = results['k_values'].index(k)
        ratio = results['compression_ratios'][idx]
        psnr = results['psnr_values'][idx]
        effective_bpp = original_bpp / ratio
        
        print(f"{k:4d} | {ratio:17.2f} | {effective_bpp:13.3f} | {psnr:13.2f}dB")
    
    print("\n✅ SVD IMAGE COMPRESSION ANALYSIS COMPLETE!")
    print("📁 Check the 'plots' folder for generated visualizations.")
    print("\n📋 SUMMARY:")
    print(f"• Analyzed compression for {len(test_ranks)} different ranks")
    print(f"• Generated 3 comprehensive visualization files")
    print(f"• Image dimensions: {compressor.original_image.shape}")
    print(f"• Maximum achievable compression ratio: {max(results['compression_ratios']):.1f}x")
    print(f"• Best quality at 10x compression: {results['psnr_values'][min(range(len(results['compression_ratios'])), key=lambda i: abs(results['compression_ratios'][i] - 10))]:.1f}dB")

if __name__ == "__main__":
    main() 