import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Optional
import os

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_training_results(episode_rewards: List[float],
                         episode_lengths: List[float],
                         losses: List[float],
                         epsilon_history: List[float],
                         window_size: int = 100,
                         save_path: Optional[str] = None):
    """
    Plot comprehensive training results.
    
    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        losses: List of training losses
        epsilon_history: List of epsilon values over time
        window_size: Window size for moving averages
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    axes[0, 0].plot(episode_rewards, alpha=0.3, color='blue', label='Raw Rewards')
    
    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        axes[0, 0].plot(range(window_size-1, len(episode_rewards)), 
                       moving_avg, color='red', linewidth=2, label=f'Moving Avg ({window_size})')
    
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Episode lengths
    axes[0, 1].plot(episode_lengths, alpha=0.3, color='green', label='Raw Lengths')
    
    if len(episode_lengths) >= window_size:
        moving_avg = np.convolve(episode_lengths, np.ones(window_size)/window_size, mode='valid')
        axes[0, 1].plot(range(window_size-1, len(episode_lengths)), 
                       moving_avg, color='red', linewidth=2, label=f'Moving Avg ({window_size})')
    
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Training losses
    if losses:
        axes[1, 0].plot(losses, alpha=0.7, color='orange', label='Training Loss')
        
        if len(losses) >= window_size:
            moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            axes[1, 0].plot(range(window_size-1, len(losses)), 
                           moving_avg, color='red', linewidth=2, label=f'Moving Avg ({window_size})')
        
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No loss data available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Training Loss')
    
    # Epsilon decay
    axes[1, 1].plot(epsilon_history, color='purple', label='Epsilon')
    axes[1, 1].set_title('Exploration Rate Decay')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Epsilon')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('DQN Training Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_learning_curves(rewards_data: Dict[str, List[float]], 
                        window_size: int = 100,
                        save_path: Optional[str] = None):
    """
    Plot learning curves for multiple agents.
    
    Args:
        rewards_data: Dictionary mapping agent names to reward lists
        window_size: Window size for moving averages
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Raw rewards
    for name, rewards in rewards_data.items():
        axes[0].plot(rewards, alpha=0.3, label=f'{name} Raw')
    
    # Moving averages
    for name, rewards in rewards_data.items():
        if len(rewards) >= window_size:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            axes[0].plot(range(window_size-1, len(rewards)), 
                        moving_avg, linewidth=2, label=f'{name} MA')
    
    axes[0].set_title('Learning Curves Comparison')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Performance comparison (final performance)
    final_performances = []
    agent_names = []
    
    for name, rewards in rewards_data.items():
        if len(rewards) >= 100:
            final_perf = np.mean(rewards[-100:])
            final_performances.append(final_perf)
            agent_names.append(name)
    
    if final_performances:
        bars = axes[1].bar(agent_names, final_performances, alpha=0.7)
        axes[1].set_title('Final Performance Comparison')
        axes[1].set_xlabel('Agent')
        axes[1].set_ylabel('Mean Reward (last 100 episodes)')
        axes[1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, final_performances):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{value:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_hyperparameter_comparison(results: Dict, save_path: Optional[str] = None):
    """
    Plot comparison of different DQN variants.
    
    Args:
        results: Dictionary mapping variant names to results
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract data
    names = []
    final_rewards = []
    final_lengths = []
    convergence_episodes = []
    success_rates = []
    
    for variant, data in results.items():
        names.append(data['name'])
        stats = data['stats']
        eval_results = data['eval_results']
        
        final_rewards.append(eval_results['mean_reward'])
        final_lengths.append(eval_results['mean_length'])
        success_rates.append(eval_results['success_rate'])
        
        # Find convergence point
        rewards = stats['episode_rewards']
        window_size = 100
        convergence_ep = len(rewards)  # Default to end if no convergence
        
        if len(rewards) >= window_size:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            for i in range(window_size, len(moving_avg)):
                recent_std = np.std(moving_avg[i-window_size:i])
                if recent_std < 0.1:  # Convergence threshold
                    convergence_ep = i
                    break
        
        convergence_episodes.append(convergence_ep)
    
    # Final performance
    bars1 = axes[0, 0].bar(names, final_rewards, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Final Performance (Mean Reward)')
    axes[0, 0].set_xlabel('DQN Variant')
    axes[0, 0].set_ylabel('Mean Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars1, final_rewards):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{value:.1f}', ha='center', va='bottom')
    
    # Success rate
    bars2 = axes[0, 1].bar(names, success_rates, alpha=0.7, color='lightgreen')
    axes[0, 1].set_title('Success Rate')
    axes[0, 1].set_xlabel('DQN Variant')
    axes[0, 1].set_ylabel('Success Rate')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars2, success_rates):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.2%}', ha='center', va='bottom')
    
    # Episode length
    bars3 = axes[1, 0].bar(names, final_lengths, alpha=0.7, color='orange')
    axes[1, 0].set_title('Mean Episode Length')
    axes[1, 0].set_xlabel('DQN Variant')
    axes[1, 0].set_ylabel('Steps')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars3, final_lengths):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{value:.1f}', ha='center', va='bottom')
    
    # Convergence speed
    bars4 = axes[1, 1].bar(names, convergence_episodes, alpha=0.7, color='purple')
    axes[1, 1].set_title('Convergence Speed')
    axes[1, 1].set_xlabel('DQN Variant')
    axes[1, 1].set_ylabel('Convergence Episode')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars4, convergence_episodes):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                       f'{value}', ha='center', va='bottom')
    
    plt.suptitle('DQN Variants Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_hyperparameter_sweep_results(sweep_results: List[Dict], 
                                     save_path: Optional[str] = None):
    """
    Plot hyperparameter sweep results.
    
    Args:
        sweep_results: List of dictionaries containing hyperparameter results
        save_path: Path to save the plot
    """
    if not sweep_results:
        print("No sweep results to plot")
        return
    
    # Extract data
    learning_rates = [r['learning_rate'] for r in sweep_results]
    epsilon_decays = [r['epsilon_decay'] for r in sweep_results]
    target_freqs = [r['target_update_freq'] for r in sweep_results]
    mean_rewards = [r['mean_reward'] for r in sweep_results]
    std_rewards = [r['std_reward'] for r in sweep_results]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Learning rate vs performance
    lr_unique = sorted(set(learning_rates))
    lr_performance = []
    lr_std = []
    
    for lr in lr_unique:
        lr_rewards = [r['mean_reward'] for r in sweep_results if r['learning_rate'] == lr]
        lr_performance.append(np.mean(lr_rewards))
        lr_std.append(np.std(lr_rewards))
    
    axes[0, 0].errorbar(lr_unique, lr_performance, yerr=lr_std, 
                       marker='o', capsize=5, capthick=2)
    axes[0, 0].set_title('Learning Rate vs Performance')
    axes[0, 0].set_xlabel('Learning Rate')
    axes[0, 0].set_ylabel('Mean Reward')
    axes[0, 0].set_xscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Epsilon decay vs performance
    eps_unique = sorted(set(epsilon_decays))
    eps_performance = []
    eps_std = []
    
    for eps in eps_unique:
        eps_rewards = [r['mean_reward'] for r in sweep_results if r['epsilon_decay'] == eps]
        eps_performance.append(np.mean(eps_rewards))
        eps_std.append(np.std(eps_rewards))
    
    axes[0, 1].errorbar(eps_unique, eps_performance, yerr=eps_std, 
                       marker='o', capsize=5, capthick=2)
    axes[0, 1].set_title('Epsilon Decay vs Performance')
    axes[0, 1].set_xlabel('Epsilon Decay')
    axes[0, 1].set_ylabel('Mean Reward')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Target update frequency vs performance
    target_unique = sorted(set(target_freqs))
    target_performance = []
    target_std = []
    
    for target in target_unique:
        target_rewards = [r['mean_reward'] for r in sweep_results if r['target_update_freq'] == target]
        target_performance.append(np.mean(target_rewards))
        target_std.append(np.std(target_rewards))
    
    axes[1, 0].errorbar(target_unique, target_performance, yerr=target_std, 
                       marker='o', capsize=5, capthick=2)
    axes[1, 0].set_title('Target Update Frequency vs Performance')
    axes[1, 0].set_xlabel('Target Update Frequency')
    axes[1, 0].set_ylabel('Mean Reward')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 3D scatter plot of all parameters
    scatter = axes[1, 1].scatter(learning_rates, epsilon_decays, 
                                c=mean_rewards, s=100, alpha=0.7, cmap='viridis')
    axes[1, 1].set_title('Hyperparameter Space (colored by performance)')
    axes[1, 1].set_xlabel('Learning Rate')
    axes[1, 1].set_ylabel('Epsilon Decay')
    axes[1, 1].set_xscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label('Mean Reward')
    
    plt.suptitle('Hyperparameter Sweep Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_replay_buffer_analysis(buffer_stats: Dict, save_path: Optional[str] = None):
    """
    Plot replay buffer analysis.
    
    Args:
        buffer_stats: Dictionary containing buffer statistics
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Buffer utilization over time
    if 'utilization_history' in buffer_stats:
        axes[0, 0].plot(buffer_stats['utilization_history'])
        axes[0, 0].set_title('Buffer Utilization Over Time')
        axes[0, 0].set_xlabel('Training Step')
        axes[0, 0].set_ylabel('Utilization')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Reward distribution in buffer
    if 'reward_distribution' in buffer_stats:
        axes[0, 1].hist(buffer_stats['reward_distribution'], bins=50, alpha=0.7)
        axes[0, 1].set_title('Reward Distribution in Buffer')
        axes[0, 1].set_xlabel('Reward')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
    
    # Priority distribution (for prioritized replay)
    if 'priority_distribution' in buffer_stats:
        axes[1, 0].hist(buffer_stats['priority_distribution'], bins=50, alpha=0.7)
        axes[1, 0].set_title('Priority Distribution')
        axes[1, 0].set_xlabel('Priority')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Buffer statistics summary
    if 'current_stats' in buffer_stats:
        stats = buffer_stats['current_stats']
        stats_text = f"""
        Size: {stats.get('size', 'N/A')}
        Capacity: {stats.get('capacity', 'N/A')}
        Utilization: {stats.get('utilization', 0):.2%}
        Mean Reward: {stats.get('mean_reward', 0):.2f}
        Std Reward: {stats.get('std_reward', 0):.2f}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        axes[1, 1].set_title('Buffer Statistics')
        axes[1, 1].axis('off')
    
    plt.suptitle('Replay Buffer Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    # Test plotting functions
    print("Testing plotting functions...")
    
    # Generate dummy data
    episode_rewards = np.random.normal(100, 50, 1000)
    episode_lengths = np.random.normal(200, 50, 1000)
    losses = np.random.exponential(0.1, 1000)
    epsilon_history = np.exp(-np.linspace(0, 5, 1000))
    
    # Test training results plot
    plot_training_results(episode_rewards, episode_lengths, losses, epsilon_history)
    
    print("Plotting functions test completed!")
