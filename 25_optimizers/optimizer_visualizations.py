"""
Optimizer Trajectory Visualization

This module provides tools for visualizing optimizer behavior on 2D loss surfaces.
Includes various test functions and trajectory plotting capabilities.

Key Features:
- 2D loss landscape visualization (contour plots, 3D surfaces)
- Optimizer trajectory tracking and animation
- Multiple test functions (Rosenbrock, Beale, Himmelblau, etc.)
- Comparative analysis of optimizer paths
- Convergence rate visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
from typing import Callable, List, Tuple, Dict, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from optimizers import *

class TestFunctions:
    """Collection of test functions for optimizer visualization"""
    
    @staticmethod
    def rosenbrock(x: np.ndarray, a: float = 1.0, b: float = 100.0) -> float:
        """
        Rosenbrock function: f(x,y) = (a-x)² + b(y-x²)²
        
        Global minimum at (a, a²) with value 0.
        Classic optimization test function with banana-shaped valley.
        """
        return (a - x[0])**2 + b * (x[1] - x[0]**2)**2
    
    @staticmethod
    def rosenbrock_grad(x: np.ndarray, a: float = 1.0, b: float = 100.0) -> np.ndarray:
        """Gradient of Rosenbrock function"""
        dx = -2*(a - x[0]) - 4*b*x[0]*(x[1] - x[0]**2)
        dy = 2*b*(x[1] - x[0]**2)
        return np.array([dx, dy])
    
    @staticmethod
    def beale(x: np.ndarray) -> float:
        """
        Beale function: f(x,y) = (1.5-x+xy)² + (2.25-x+xy²)² + (2.625-x+xy³)²
        
        Global minimum at (3, 0.5) with value 0.
        """
        term1 = (1.5 - x[0] + x[0]*x[1])**2
        term2 = (2.25 - x[0] + x[0]*x[1]**2)**2  
        term3 = (2.625 - x[0] + x[0]*x[1]**3)**2
        return term1 + term2 + term3
    
    @staticmethod
    def beale_grad(x: np.ndarray) -> np.ndarray:
        """Gradient of Beale function"""
        # Partial derivatives computed analytically
        dx = (2*(1.5 - x[0] + x[0]*x[1])*(x[1] - 1) + 
              2*(2.25 - x[0] + x[0]*x[1]**2)*(x[1]**2 - 1) +
              2*(2.625 - x[0] + x[0]*x[1]**3)*(x[1]**3 - 1))
        
        dy = (2*(1.5 - x[0] + x[0]*x[1])*x[0] + 
              2*(2.25 - x[0] + x[0]*x[1]**2)*x[0]*2*x[1] +
              2*(2.625 - x[0] + x[0]*x[1]**3)*x[0]*3*x[1]**2)
        
        return np.array([dx, dy])
    
    @staticmethod
    def himmelblau(x: np.ndarray) -> float:
        """
        Himmelblau function: f(x,y) = (x²+y-11)² + (x+y²-7)²
        
        Four global minima at:
        (3, 2), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)
        """
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
    
    @staticmethod  
    def himmelblau_grad(x: np.ndarray) -> np.ndarray:
        """Gradient of Himmelblau function"""
        dx = 4*x[0]*(x[0]**2 + x[1] - 11) + 2*(x[0] + x[1]**2 - 7)
        dy = 2*(x[0]**2 + x[1] - 11) + 4*x[1]*(x[0] + x[1]**2 - 7)
        return np.array([dx, dy])
    
    @staticmethod
    def rastrigin(x: np.ndarray, A: float = 10.0) -> float:
        """
        Rastrigin function: f(x,y) = A*n + Σ[x_i² - A*cos(2πx_i)]
        
        Global minimum at (0, 0) with value 0.
        Highly multimodal with many local minima.
        """
        n = len(x)
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    @staticmethod
    def rastrigin_grad(x: np.ndarray, A: float = 10.0) -> np.ndarray:
        """Gradient of Rastrigin function"""
        return 2*x + 2*A*np.pi*np.sin(2*np.pi*x)
    
    @staticmethod
    def saddle_point(x: np.ndarray) -> float:
        """
        Simple saddle point: f(x,y) = x² - y²
        
        Saddle point at (0, 0). Useful for testing optimizer behavior
        near saddle points.
        """
        return x[0]**2 - x[1]**2
    
    @staticmethod
    def saddle_point_grad(x: np.ndarray) -> np.ndarray:
        """Gradient of saddle point function"""
        return np.array([2*x[0], -2*x[1]])
    
    @staticmethod
    def sphere(x: np.ndarray) -> float:
        """
        Sphere function: f(x) = Σx_i²
        
        Simple convex function for testing basic convergence.
        """
        return np.sum(x**2)
    
    @staticmethod
    def sphere_grad(x: np.ndarray) -> np.ndarray:
        """Gradient of sphere function"""
        return 2*x

class OptimizerVisualizer:
    """Main class for visualizing optimizer trajectories"""
    
    def __init__(self, func: Callable, grad_func: Callable, 
                 x_range: Tuple[float, float] = (-2, 2),
                 y_range: Tuple[float, float] = (-2, 2),
                 resolution: int = 100):
        """
        Initialize visualizer
        
        Args:
            func: Objective function f(x) -> scalar
            grad_func: Gradient function ∇f(x) -> array
            x_range: Range for x-axis
            y_range: Range for y-axis  
            resolution: Grid resolution for contour plots
        """
        self.func = func
        self.grad_func = grad_func
        self.x_range = x_range
        self.y_range = y_range
        self.resolution = resolution
        
        # Create meshgrid for visualization
        self.X, self.Y = np.meshgrid(
            np.linspace(x_range[0], x_range[1], resolution),
            np.linspace(y_range[0], y_range[1], resolution)
        )
        
        # Compute function values on grid
        self.Z = np.zeros_like(self.X)
        for i in range(resolution):
            for j in range(resolution):
                self.Z[i, j] = self.func(np.array([self.X[i, j], self.Y[i, j]]))
    
    def plot_3d_surface(self, figsize: Tuple[int, int] = (12, 8), 
                       title: str = "Loss Surface") -> plt.Figure:
        """
        Plot 3D surface of the loss function
        
        Args:
            figsize: Figure size
            title: Plot title
            
        Returns:
            Figure object
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot surface
        surf = ax.plot_surface(self.X, self.Y, self.Z, 
                              cmap='viridis', alpha=0.8,
                              linewidth=0, antialiased=True)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y') 
        ax.set_zlabel('Loss')
        ax.set_title(title)
        
        # Add colorbar
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        return fig
    
    def plot_contour(self, figsize: Tuple[int, int] = (10, 8),
                    levels: int = 20, title: str = "Loss Contours") -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot contour map of the loss function
        
        Args:
            figsize: Figure size
            levels: Number of contour levels
            title: Plot title
            
        Returns:
            Figure and axes objects
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create contour plot
        contour = ax.contour(self.X, self.Y, self.Z, levels=levels, colors='gray', alpha=0.6)
        contourf = ax.contourf(self.X, self.Y, self.Z, levels=levels, cmap='viridis', alpha=0.8)
        
        # Add colorbar
        fig.colorbar(contourf, ax=ax)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    def run_optimizer(self, optimizer: BaseOptimizer, 
                     initial_point: np.ndarray,
                     num_steps: int = 100) -> Dict[str, List]:
        """
        Run optimizer and track trajectory
        
        Args:
            optimizer: Optimizer instance
            initial_point: Starting point [x, y]
            num_steps: Number of optimization steps
            
        Returns:
            Dictionary with trajectory history
        """
        # Reset optimizer
        optimizer.reset()
        
        # Initialize parameters (ensure float dtype)
        params = {'weights': initial_point.copy().astype(np.float64)}
        
        # Track history
        history = {
            'params': [initial_point.copy()],
            'losses': [self.func(initial_point)],
            'grad_norms': []
        }
        
        for step in range(num_steps):
            # Compute gradients
            grad = self.grad_func(params['weights'])
            grad_norm = np.linalg.norm(grad)
            
            # Early stopping if gradient is very small
            if grad_norm < 1e-8:
                break
                
            # Update parameters
            grads = {'weights': grad}
            optimizer.step(params, grads)
            
            # Store history
            history['params'].append(params['weights'].copy())
            history['losses'].append(self.func(params['weights']))
            history['grad_norms'].append(grad_norm)
        
        return history
    
    def compare_optimizers(self, optimizers: List[BaseOptimizer],
                          initial_point: np.ndarray,
                          num_steps: int = 100,
                          figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Compare multiple optimizers on the same loss surface
        
        Args:
            optimizers: List of optimizer instances
            initial_point: Starting point [x, y]
            num_steps: Number of optimization steps
            figsize: Figure size
            
        Returns:
            Figure object
        """
        # Create subplots
        fig = plt.figure(figsize=figsize)
        
        # Plot 1: Contour with trajectories
        ax1 = plt.subplot(2, 2, 1)
        contour = ax1.contour(self.X, self.Y, self.Z, levels=20, colors='gray', alpha=0.6)
        contourf = ax1.contourf(self.X, self.Y, self.Z, levels=20, cmap='viridis', alpha=0.3)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(optimizers)))
        trajectories = {}
        
        for i, optimizer in enumerate(optimizers):
            history = self.run_optimizer(optimizer, initial_point, num_steps)
            trajectories[optimizer.__class__.__name__] = history
            
            # Plot trajectory
            params_array = np.array(history['params'])
            ax1.plot(params_array[:, 0], params_array[:, 1], 
                    color=colors[i], linewidth=2, alpha=0.8,
                    label=optimizer.__class__.__name__, marker='o', markersize=3)
        
        # Mark starting point
        ax1.plot(initial_point[0], initial_point[1], 'r*', markersize=15, label='Start')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('Optimizer Trajectories')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Loss curves
        ax2 = plt.subplot(2, 2, 2)
        for i, (name, history) in enumerate(trajectories.items()):
            ax2.plot(history['losses'], color=colors[i], linewidth=2, label=name)
        
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss')
        ax2.set_title('Convergence Curves')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Gradient norms
        ax3 = plt.subplot(2, 2, 3)
        for i, (name, history) in enumerate(trajectories.items()):
            if history['grad_norms']:  # Check if not empty
                ax3.plot(history['grad_norms'], color=colors[i], linewidth=2, label=name)
        
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Gradient Norm')
        ax3.set_title('Gradient Magnitudes')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Final positions
        ax4 = plt.subplot(2, 2, 4)
        final_losses = []
        names = []
        
        for i, (name, history) in enumerate(trajectories.items()):
            final_loss = history['losses'][-1]
            final_losses.append(final_loss)
            names.append(name)
            
        bars = ax4.bar(names, final_losses, color=colors[:len(names)])
        ax4.set_ylabel('Final Loss')
        ax4.set_title('Final Performance')
        ax4.set_yscale('log')
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    def create_animation(self, optimizer: BaseOptimizer,
                        initial_point: np.ndarray,
                        num_steps: int = 100,
                        interval: int = 100,
                        save_path: Optional[str] = None) -> animation.FuncAnimation:
        """
        Create animated visualization of optimizer trajectory
        
        Args:
            optimizer: Optimizer instance
            initial_point: Starting point [x, y]
            num_steps: Number of optimization steps
            interval: Animation interval in milliseconds
            save_path: Path to save animation (optional)
            
        Returns:
            Animation object
        """
        # Run optimizer to get full trajectory
        history = self.run_optimizer(optimizer, initial_point, num_steps)
        params_array = np.array(history['params'])
        
        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot contour
        contour = ax.contour(self.X, self.Y, self.Z, levels=20, colors='gray', alpha=0.6)
        contourf = ax.contourf(self.X, self.Y, self.Z, levels=20, cmap='viridis', alpha=0.3)
        
        # Initialize trajectory line and point
        line, = ax.plot([], [], 'r-', linewidth=2, alpha=0.8, label='Trajectory')
        point, = ax.plot([], [], 'ro', markersize=8, label='Current Position')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'{optimizer.__class__.__name__} Optimization')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        def animate(frame):
            # Update trajectory up to current frame
            if frame < len(params_array):
                x_data = params_array[:frame+1, 0]
                y_data = params_array[:frame+1, 1]
                line.set_data(x_data, y_data)
                point.set_data([params_array[frame, 0]], [params_array[frame, 1]])
            
            return line, point
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=len(params_array),
                                     interval=interval, blit=True, repeat=True)
        
        # Save if path provided
        if save_path:
            try:
                anim.save(save_path, writer='pillow', fps=10)
                print(f"Animation saved to {save_path}")
            except Exception as e:
                print(f"Could not save animation: {e}")
        
        return anim

def demonstrate_test_functions():
    """Demonstrate various test functions"""
    test_functions = [
        ("Rosenbrock", TestFunctions.rosenbrock, TestFunctions.rosenbrock_grad, (-2, 2), (-1, 3)),
        ("Beale", TestFunctions.beale, TestFunctions.beale_grad, (-4.5, 4.5), (-4.5, 4.5)),
        ("Himmelblau", TestFunctions.himmelblau, TestFunctions.himmelblau_grad, (-5, 5), (-5, 5)),
        ("Saddle Point", TestFunctions.saddle_point, TestFunctions.saddle_point_grad, (-2, 2), (-2, 2)),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (name, func, grad_func, x_range, y_range) in enumerate(test_functions):
        visualizer = OptimizerVisualizer(func, grad_func, x_range, y_range)
        
        # Plot contour
        contourf = axes[i].contourf(visualizer.X, visualizer.Y, visualizer.Z, 
                                   levels=20, cmap='viridis', alpha=0.8)
        contour = axes[i].contour(visualizer.X, visualizer.Y, visualizer.Z, 
                                 levels=20, colors='black', alpha=0.4)
        
        axes[i].set_title(f'{name} Function')
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')
        
        # Add colorbar
        fig.colorbar(contourf, ax=axes[i])
    
    plt.tight_layout()
    return fig

def comprehensive_optimizer_analysis():
    """Run comprehensive analysis of all optimizers on multiple test functions"""
    
    # Test functions to use
    test_cases = [
        ("Rosenbrock", TestFunctions.rosenbrock, TestFunctions.rosenbrock_grad, 
         (-2, 2), (-1, 3), np.array([-1.5, 1.5])),
        ("Beale", TestFunctions.beale, TestFunctions.beale_grad, 
         (-4.5, 4.5), (-4.5, 4.5), np.array([0, 0])),
        ("Himmelblau", TestFunctions.himmelblau, TestFunctions.himmelblau_grad, 
         (-5, 5), (-5, 5), np.array([0, 0])),
        ("Saddle Point", TestFunctions.saddle_point, TestFunctions.saddle_point_grad, 
         (-2, 2), (-2, 2), np.array([1, 1])),
    ]
    
    # Create optimizers
    optimizers = [
        SGD(learning_rate=0.01),
        Momentum(learning_rate=0.01, momentum=0.9),
        Nesterov(learning_rate=0.01, momentum=0.9),
        RMSProp(learning_rate=0.01),
        Adam(learning_rate=0.01)
    ]
    
    # Run analysis for each test function
    for name, func, grad_func, x_range, y_range, initial_point in test_cases:
        print(f"\nAnalyzing optimizers on {name} function...")
        
        visualizer = OptimizerVisualizer(func, grad_func, x_range, y_range)
        fig = visualizer.compare_optimizers(optimizers, initial_point, num_steps=200)
        
        plt.suptitle(f'Optimizer Comparison: {name} Function', fontsize=16)
        plt.show()

if __name__ == "__main__":
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    print("Optimizer Visualization Demo")
    print("=" * 50)
    
    # 1. Demonstrate test functions
    print("1. Plotting test functions...")
    fig = demonstrate_test_functions()
    plt.savefig('plots/test_functions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Quick comparison on Rosenbrock function
    print("\n2. Quick optimizer comparison on Rosenbrock function...")
    visualizer = OptimizerVisualizer(
        TestFunctions.rosenbrock, 
        TestFunctions.rosenbrock_grad,
        x_range=(-2, 2), 
        y_range=(-1, 3)
    )
    
    optimizers = [
        SGD(learning_rate=0.01),
        Momentum(learning_rate=0.01, momentum=0.9),
        Adam(learning_rate=0.01)
    ]
    
    fig = visualizer.compare_optimizers(optimizers, np.array([-1.5, 1.5]), num_steps=200)
    plt.savefig('plots/optimizer_comparison_rosenbrock.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nVisualization complete! Check the plots/ directory for saved figures.") 