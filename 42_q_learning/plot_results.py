"""
Visualization Tools for Q-Learning Results
==========================================

Comprehensive visualization tools for analyzing Q-learning performance,
policies, and value functions across different environments.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from scipy.ndimage import gaussian_filter
import os
from datetime import datetime


class QLearningVisualizer:
    """
    Comprehensive visualization toolkit for Q-learning results.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), 
                 save_dir: str = "q_learning_plots"):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size for plots
            save_dir: Directory to save plots (created if doesn't exist)
        """
        self.figsize = figsize
        self.colors = plt.cm.Set1.colors
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            try:
                plt.style.use('seaborn')
            except OSError:
                pass  # Use default matplotlib style
        
        # Custom colormap for Q-values
        self.q_cmap = LinearSegmentedColormap.from_list(
            'q_values', ['red', 'yellow', 'green'], N=256
        )
        
        # Setup save directory structure
        self.base_save_dir = save_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._create_save_directories()
    
    def _create_save_directories(self) -> None:
        """Create directory structure for saving plots."""
        directories = [
            self.base_save_dir,
            os.path.join(self.base_save_dir, "training_curves"),
            os.path.join(self.base_save_dir, "policies"),
            os.path.join(self.base_save_dir, "q_values"),
            os.path.join(self.base_save_dir, "analysis"),
            os.path.join(self.base_save_dir, "dashboards")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        print(f"📁 Plot save directories created in: {self.base_save_dir}")
    
    def _get_default_save_path(self, plot_type: str, filename: str) -> str:
        """
        Generate default save path for plots.
        
        Args:
            plot_type: Type of plot (e.g., 'training_curves', 'policies')
            filename: Base filename without extension
            
        Returns:
            Full path for saving the plot
        """
        return os.path.join(self.base_save_dir, plot_type, 
                          f"{filename}_{self.timestamp}.png")
    
    def plot_training_curves(self, results: Dict[str, Any], 
                           smooth_window: int = 100, 
                           save_path: Optional[str] = None) -> None:
        """
        Plot comprehensive training curves.
        
        Args:
            results: Dictionary of experiment results
            smooth_window: Window size for smoothing
            save_path: Path to save the plot
        """
        n_experiments = len(results)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Raw episode rewards
        ax1 = axes[0, 0]
        for i, (name, result) in enumerate(results.items()):
            rewards = result['training_stats']['episode_rewards']
            ax1.plot(rewards, alpha=0.3, color=self.colors[i % len(self.colors)])
            
            # Add smoothed curve
            if len(rewards) > smooth_window:
                smoothed = self._smooth_curve(rewards, smooth_window)
                ax1.plot(smoothed, label=name, color=self.colors[i % len(self.colors)], linewidth=2)
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Episode Reward')
        ax1.set_title('Training Curves (Raw + Smoothed)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Success rate over time
        ax2 = axes[0, 1]
        for i, (name, result) in enumerate(results.items()):
            rewards = np.array(result['training_stats']['episode_rewards'])
            success_rate = self._calculate_success_rate(rewards, window=smooth_window)
            ax2.plot(success_rate, label=name, color=self.colors[i % len(self.colors)], linewidth=2)
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Success Rate')
        ax2.set_title(f'Success Rate (Moving Average, window={smooth_window})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Plot 3: Episode length over time
        ax3 = axes[1, 0]
        for i, (name, result) in enumerate(results.items()):
            lengths = result['training_stats']['episode_lengths']
            if len(lengths) > smooth_window:
                smoothed_lengths = self._smooth_curve(lengths, smooth_window)
                ax3.plot(smoothed_lengths, label=name, color=self.colors[i % len(self.colors)], linewidth=2)
        
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Episode Length')
        ax3.set_title('Episode Length Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Final performance comparison
        ax4 = axes[1, 1]
        metrics = ['Success Rate', 'Avg Reward', 'Avg Length']
        x = np.arange(len(results))
        width = 0.25
        
        success_rates = [results[name]['evaluation_stats']['success_rate'] for name in results.keys()]
        avg_rewards = [results[name]['evaluation_stats']['avg_reward'] for name in results.keys()]
        avg_lengths = [results[name]['evaluation_stats']['avg_length'] / 100 for name in results.keys()]  # Normalized
        
        ax4.bar(x - width, success_rates, width, label='Success Rate', alpha=0.7)
        ax4.bar(x, avg_rewards, width, label='Avg Reward', alpha=0.7)
        ax4.bar(x + width, avg_lengths, width, label='Avg Length (÷100)', alpha=0.7)
        
        ax4.set_xlabel('Experiment')
        ax4.set_ylabel('Normalized Performance')
        ax4.set_title('Final Performance Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(list(results.keys()), rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Use default save path if none provided
        if save_path is None:
            save_path = self._get_default_save_path("training_curves", "training_comparison")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Training curves saved to: {save_path}")
        
        plt.show()
    
    def visualize_gridworld_policy(self, env, policy: np.ndarray, q_table: np.ndarray,
                                 title: str = "GridWorld Policy", 
                                 save_path: Optional[str] = None) -> None:
        """
        Visualize policy and values for GridWorld environment.
        
        Args:
            env: GridWorld environment
            policy: Learned policy
            q_table: Q-table
            title: Plot title
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Get state values
        state_values = np.max(q_table, axis=1)
        value_grid = state_values.reshape(env.height, env.width)
        
        # Plot 1: State values as heatmap
        ax1 = axes[0]
        im1 = ax1.imshow(value_grid, cmap='RdYlBu_r')
        
        # Add value text
        for i in range(env.height):
            for j in range(env.width):
                state = i * env.width + j
                pos = (i, j)
                
                if pos in env.obstacles:
                    ax1.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, color='black', alpha=0.8))
                    ax1.text(j, i, '#', ha='center', va='center', color='white', fontsize=16, weight='bold')
                elif pos in env.cliffs:
                    ax1.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, color='red', alpha=0.8))
                    ax1.text(j, i, 'X', ha='center', va='center', color='white', fontsize=16, weight='bold')
                elif pos == env.goal_pos:
                    ax1.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, color='green', alpha=0.8))
                    ax1.text(j, i, f'G\n{state_values[state]:.2f}', ha='center', va='center', 
                            color='white', fontsize=10, weight='bold')
                else:
                    ax1.text(j, i, f'{state_values[state]:.2f}', ha='center', va='center', fontsize=10)
        
        ax1.set_title('State Values V(s) = max_a Q(s,a)')
        plt.colorbar(im1, ax=ax1, label='State Value')
        ax1.set_xticks(range(env.width))
        ax1.set_yticks(range(env.height))
        
        # Plot 2: Policy with arrows
        ax2 = axes[1]
        im2 = ax2.imshow(value_grid, cmap='RdYlBu_r', alpha=0.3)
        
        # Action arrows
        action_arrows = {0: '↑', 1: '↓', 2: '←', 3: '→'}
        action_directions = {0: (0, 0.3), 1: (0, -0.3), 2: (-0.3, 0), 3: (0.3, 0)}
        
        for i in range(env.height):
            for j in range(env.width):
                state = i * env.width + j
                pos = (i, j)
                
                if pos in env.obstacles:
                    ax2.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, color='black', alpha=0.8))
                    ax2.text(j, i, '#', ha='center', va='center', color='white', fontsize=16, weight='bold')
                elif pos in env.cliffs:
                    ax2.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, color='red', alpha=0.8))
                    ax2.text(j, i, 'X', ha='center', va='center', color='white', fontsize=16, weight='bold')
                elif pos == env.goal_pos:
                    ax2.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, color='green', alpha=0.8))
                    ax2.text(j, i, 'G', ha='center', va='center', color='white', fontsize=16, weight='bold')
                else:
                    action = policy[state]
                    dx, dy = action_directions[action]
                    ax2.arrow(j, i, dx, dy, head_width=0.15, head_length=0.1, 
                             fc='black', ec='black', linewidth=2)
        
        ax2.set_title('Learned Policy π(s)')
        ax2.set_xticks(range(env.width))
        ax2.set_yticks(range(env.height))
        
        plt.suptitle(title, fontsize=16, y=1.02)
        plt.tight_layout()
        
        # Use default save path if none provided
        if save_path is None:
            safe_title = title.replace(" ", "_").replace("/", "_").lower()
            save_path = self._get_default_save_path("policies", f"gridworld_policy_{safe_title}")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 GridWorld policy saved to: {save_path}")
        
        plt.show()
    
    def visualize_frozenlake_policy(self, q_table: np.ndarray, map_size: int = 4,
                                  title: str = "FrozenLake Policy",
                                  save_path: Optional[str] = None) -> None:
        """
        Visualize policy for FrozenLake environment.
        
        Args:
            q_table: Q-table
            map_size: Size of the map (4 or 8)
            title: Plot title
            save_path: Path to save the plot
        """
        policy = np.argmax(q_table, axis=1)
        state_values = np.max(q_table, axis=1)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Q-values heatmap
        ax1 = axes[0]
        value_grid = state_values.reshape(map_size, map_size)
        im1 = ax1.imshow(value_grid, cmap='RdYlBu_r')
        
        # Add values as text
        for i in range(map_size):
            for j in range(map_size):
                state = i * map_size + j
                ax1.text(j, i, f'{state_values[state]:.2f}', ha='center', va='center', fontsize=8)
        
        ax1.set_title('State Values')
        plt.colorbar(im1, ax=ax1)
        
        # Plot 2: Policy arrows
        ax2 = axes[1]
        im2 = ax2.imshow(value_grid, cmap='RdYlBu_r', alpha=0.3)
        
        action_arrows = {0: '←', 1: '↓', 2: '→', 3: '↑'}
        
        for i in range(map_size):
            for j in range(map_size):
                state = i * map_size + j
                action = policy[state]
                ax2.text(j, i, action_arrows[action], ha='center', va='center', 
                        fontsize=16, fontweight='bold')
        
        ax2.set_title('Policy')
        ax2.set_xticks(range(map_size))
        ax2.set_yticks(range(map_size))
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        # Use default save path if none provided
        if save_path is None:
            safe_title = title.replace(" ", "_").lower()
            save_path = self._get_default_save_path("policies", f"frozenlake_policy_{map_size}x{map_size}_{safe_title}")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 FrozenLake policy saved to: {save_path}")
        
        plt.show()
    
    def plot_q_value_heatmaps(self, q_table: np.ndarray, map_size: int = 4,
                            title: str = "Q-Value Heatmaps",
                            save_path: Optional[str] = None) -> None:
        """
        Plot Q-values for each action as separate heatmaps.
        
        Args:
            q_table: Q-table
            map_size: Size of the environment map
            title: Plot title
            save_path: Path to save the plot
        """
        action_names = ['LEFT', 'DOWN', 'RIGHT', 'UP']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for action in range(4):
            q_values = q_table[:, action].reshape(map_size, map_size)
            
            im = axes[action].imshow(q_values, cmap='RdYlBu_r')
            axes[action].set_title(f'{action_names[action]} Q-values')
            
            # Add values as text
            for i in range(map_size):
                for j in range(map_size):
                    text = axes[action].text(j, i, f'{q_values[i, j]:.2f}',
                                           ha="center", va="center", fontsize=8)
            
            plt.colorbar(im, ax=axes[action])
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        # Use default save path if none provided
        if save_path is None:
            save_path = self._get_default_save_path("q_values", f"q_value_heatmaps_{map_size}x{map_size}")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Q-value heatmaps saved to: {save_path}")
        
        plt.show()
    
    def plot_exploration_analysis(self, results: Dict[str, Any],
                                save_path: Optional[str] = None) -> None:
        """
        Analyze exploration strategies.
        
        Args:
            results: Dictionary of experiment results
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Final success rates
        ax1 = axes[0, 0]
        names = list(results.keys())
        success_rates = [results[name]['evaluation_stats']['success_rate'] for name in names]
        
        bars = ax1.bar(names, success_rates, color=self.colors[:len(names)], alpha=0.7)
        ax1.set_ylabel('Success Rate')
        ax1.set_title('Final Success Rate by Exploration Strategy')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{rate:.1%}', ha='center', va='bottom')
        
        # Plot 2: Learning efficiency (episodes to converge)
        ax2 = axes[0, 1]
        convergence_episodes = []
        for name, result in results.items():
            rewards = result['training_stats']['episode_rewards']
            # Find when success rate first exceeds 80%
            window = 100
            for i in range(window, len(rewards)):
                recent_rewards = rewards[i-window:i]
                if np.mean(np.array(recent_rewards) > 0) > 0.8:
                    convergence_episodes.append(i)
                    break
            else:
                convergence_episodes.append(len(rewards))  # Never converged
        
        bars = ax2.bar(names, convergence_episodes, color=self.colors[:len(names)], alpha=0.7)
        ax2.set_ylabel('Episodes to Convergence')
        ax2.set_title('Learning Speed Comparison')
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Exploration vs exploitation over time
        ax3 = axes[1, 0]
        for i, (name, result) in enumerate(results.items()):
            # Estimate exploration rate over time (simplified)
            rewards = result['training_stats']['episode_rewards']
            if 'agent' in result:
                agent = result['agent']
                # Create approximate epsilon decay curve
                episodes = np.arange(len(rewards))
                epsilon_curve = np.maximum(
                    agent.epsilon_min,
                    agent.epsilon * (agent.epsilon_decay ** episodes)
                )
                ax3.plot(episodes, epsilon_curve, label=name, color=self.colors[i])
        
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Exploration Rate (ε)')
        ax3.set_title('Exploration Rate Decay')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Variance in performance
        ax4 = axes[1, 1]
        avg_rewards = [results[name]['evaluation_stats']['avg_reward'] for name in names]
        std_rewards = [results[name]['evaluation_stats']['std_reward'] for name in names]
        
        ax4.errorbar(names, avg_rewards, yerr=std_rewards, fmt='o', capsize=5, capthick=2)
        ax4.set_ylabel('Average Reward ± Std')
        ax4.set_title('Performance Consistency')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Exploration Strategy Analysis', fontsize=16)
        plt.tight_layout()
        
        # Use default save path if none provided
        if save_path is None:
            save_path = self._get_default_save_path("analysis", "exploration_strategy_analysis")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Exploration analysis saved to: {save_path}")
        
        plt.show()
    
    def create_performance_dashboard(self, results: Dict[str, Any],
                                   save_path: Optional[str] = None) -> None:
        """
        Create a comprehensive performance dashboard.
        
        Args:
            results: Dictionary of experiment results
            save_path: Path to save the plot
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Main learning curves
        ax1 = fig.add_subplot(gs[0, :2])
        for i, (name, result) in enumerate(results.items()):
            rewards = result['training_stats']['episode_rewards']
            smoothed = self._smooth_curve(rewards, len(rewards)//50)
            ax1.plot(smoothed, label=name, color=self.colors[i], linewidth=2)
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Episode Reward')
        ax1.set_title('Learning Curves', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Performance metrics
        ax2 = fig.add_subplot(gs[0, 2:])
        metrics_df = pd.DataFrame({
            name: {
                'Success Rate': result['evaluation_stats']['success_rate'],
                'Avg Reward': result['evaluation_stats']['avg_reward'],
                'Training Time (s)': result.get('training_time', 0),
                'Avg Episode Length': result['evaluation_stats']['avg_length']
            }
            for name, result in results.items()
        }).T
        
        sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax2)
        ax2.set_title('Performance Metrics', fontsize=14, fontweight='bold')
        
        # Individual performance plots
        names = list(results.keys())
        for i, (name, result) in enumerate(results.items()):
            row = (i // 2) + 1
            col = (i % 2) * 2
            
            # Learning curve for this experiment
            ax = fig.add_subplot(gs[row, col:col+2])
            rewards = result['training_stats']['episode_rewards']
            ax.plot(rewards, alpha=0.3, color=self.colors[i])
            smoothed = self._smooth_curve(rewards, len(rewards)//20)
            ax.plot(smoothed, color=self.colors[i], linewidth=2)
            
            ax.set_title(f'{name}', fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Reward')
            ax.grid(True, alpha=0.3)
            
            # Add final performance text
            final_performance = result['evaluation_stats']['success_rate']
            ax.text(0.02, 0.98, f'Final Success Rate: {final_performance:.1%}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor=self.colors[i], alpha=0.2))
        
        plt.suptitle('Q-Learning Performance Dashboard', fontsize=18, fontweight='bold')
        
        # Use default save path if none provided
        if save_path is None:
            save_path = self._get_default_save_path("dashboards", "performance_dashboard")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Performance dashboard saved to: {save_path}")
        
        plt.show()
    
    def _smooth_curve(self, data: List[float], window_size: int) -> np.ndarray:
        """Smooth curve using moving average."""
        if window_size <= 1:
            return np.array(data)
        
        padded_data = np.pad(data, (window_size//2, window_size//2), mode='edge')
        smoothed = np.convolve(padded_data, np.ones(window_size)/window_size, mode='valid')
        return smoothed[:len(data)]
    
    def _calculate_success_rate(self, rewards: np.ndarray, window: int = 100) -> np.ndarray:
        """Calculate moving average success rate."""
        success_rate = []
        for i in range(len(rewards)):
            start = max(0, i - window + 1)
            window_rewards = rewards[start:i+1]
            success_rate.append(np.mean(window_rewards > 0))
        return np.array(success_rate)


def demonstrate_visualizations():
    """Demonstrate the visualization capabilities."""
    print("Q-Learning Visualization Demo")
    print("=============================")
    print("This demo creates dummy data and shows all visualization capabilities.")
    print("All plots will be automatically saved to the 'q_learning_plots' directory.")
    print()
    
    # Create dummy data for demonstration
    n_episodes = 1000
    n_states = 16
    n_actions = 4
    
    # Simulate different experiment results
    results = {}
    
    # High exploration experiment
    high_exp_rewards = np.random.normal(-0.5, 2.0, n_episodes)
    high_exp_rewards[500:] += np.linspace(0, 2, 500)  # Gradual improvement
    
    results['High Exploration'] = {
        'training_stats': {
            'episode_rewards': high_exp_rewards.tolist(),
            'episode_lengths': np.random.randint(10, 100, n_episodes).tolist()
        },
        'evaluation_stats': {
            'success_rate': 0.85,
            'avg_reward': 1.2,
            'std_reward': 0.5,
            'avg_length': 45
        },
        'training_time': 120.5,
        'q_table': np.random.random((n_states, n_actions))
    }
    
    # Low exploration experiment
    low_exp_rewards = np.random.normal(-1.0, 1.0, n_episodes)
    low_exp_rewards[300:] += np.linspace(0, 1.5, 700)  # Slower improvement
    
    results['Low Exploration'] = {
        'training_stats': {
            'episode_rewards': low_exp_rewards.tolist(),
            'episode_lengths': np.random.randint(15, 80, n_episodes).tolist()
        },
        'evaluation_stats': {
            'success_rate': 0.65,
            'avg_reward': 0.8,
            'std_reward': 0.7,
            'avg_length': 50
        },
        'training_time': 95.2,
        'q_table': np.random.random((n_states, n_actions))
    }
    
    # Create visualizer and demonstrate
    visualizer = QLearningVisualizer()
    
    # Show training curves
    visualizer.plot_training_curves(results)
    
    # Show exploration analysis
    visualizer.plot_exploration_analysis(results)
    
    # Show performance dashboard
    visualizer.create_performance_dashboard(results)
    
    # Show FrozenLake visualization
    dummy_q_table = np.random.random((16, 4)) * 2 - 1
    visualizer.visualize_frozenlake_policy(dummy_q_table, map_size=4)
    visualizer.plot_q_value_heatmaps(dummy_q_table, map_size=4)
    
    print("\n" + "="*50)
    print("✅ Visualization demonstration complete!")
    print(f"📁 All plots saved to: {visualizer.base_save_dir}/")
    print("📊 Created visualizations:")
    print("  • Training curves comparison")
    print("  • Exploration strategy analysis") 
    print("  • Performance dashboard")
    print("  • FrozenLake policy visualization")
    print("  • Q-value heatmaps")
    print("="*50)


if __name__ == "__main__":
    # Check for required packages
    try:
        import seaborn as sns
        import pandas as pd
    except ImportError:
        print("Installing required packages...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn", "pandas", "scipy"])
        import seaborn as sns
        import pandas as pd
    
    demonstrate_visualizations()
