#!/usr/bin/env python3
"""
Plotting utilities for Policy Gradient algorithms.

This script provides comprehensive plotting functions for visualizing
training progress, comparing different algorithms, and analyzing results.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import os
import json
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_learning_curves(rewards_data: Dict[str, List[float]], 
                        window_size: int = 100,
                        save_path: Optional[str] = None,
                        title: str = "Learning Curves Comparison"):
    """
    Plot learning curves for multiple algorithms.
    
    Args:
        rewards_data: Dictionary mapping algorithm names to reward lists
        window_size: Window size for moving averages
        save_path: Path to save plot
        title: Plot title
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
    
    axes[0].axhline(y=475, color='green', linestyle='--', alpha=0.7, label='Target (475)')
    axes[0].set_title('Learning Curves')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Final performance comparison
    final_performances = []
    algorithm_names = []
    
    for name, rewards in rewards_data.items():
        if len(rewards) >= 100:
            final_perf = np.mean(rewards[-100:])
            final_performances.append(final_perf)
            algorithm_names.append(name)
    
    if final_performances:
        bars = axes[1].bar(algorithm_names, final_performances, alpha=0.7)
        axes[1].axhline(y=475, color='green', linestyle='--', alpha=0.7, label='Target (475)')
        axes[1].set_title('Final Performance (last 100 episodes)')
        axes[1].set_xlabel('Algorithm')
        axes[1].set_ylabel('Mean Reward')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, final_performances):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        f'{value:.1f}', ha='center', va='bottom')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_analysis(training_stats: Dict, save_path: Optional[str] = None):
    """
    Plot comprehensive training analysis.
    
    Args:
        training_stats: Training statistics dictionary
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Episode rewards
    episode_rewards = training_stats.get('episode_rewards', [])
    if episode_rewards:
        axes[0, 0].plot(episode_rewards, alpha=0.3, color='blue', label='Raw Rewards')
        
        if len(episode_rewards) >= 100:
            moving_avg = np.convolve(episode_rewards, np.ones(100)/100, mode='valid')
            axes[0, 0].plot(range(99, len(episode_rewards)), 
                           moving_avg, color='red', linewidth=2, label='Moving Avg (100)')
        
        axes[0, 0].axhline(y=475, color='green', linestyle='--', label='Target (475)')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Episode lengths
    episode_lengths = training_stats.get('episode_lengths', [])
    if episode_lengths:
        axes[0, 1].plot(episode_lengths, alpha=0.3, color='green', label='Raw Lengths')
        
        if len(episode_lengths) >= 100:
            moving_avg = np.convolve(episode_lengths, np.ones(100)/100, mode='valid')
            axes[0, 1].plot(range(99, len(episode_lengths)), 
                           moving_avg, color='red', linewidth=2, label='Moving Avg (100)')
        
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Losses
    policy_losses = training_stats.get('policy_losses', [])
    value_losses = training_stats.get('value_losses', [])
    entropy_losses = training_stats.get('entropy_losses', [])
    
    if policy_losses:
        axes[0, 2].plot(policy_losses, alpha=0.7, color='orange', label='Policy Loss')
    if value_losses:
        axes[0, 2].plot(value_losses, alpha=0.7, color='purple', label='Value Loss')
    if entropy_losses:
        axes[0, 2].plot(entropy_losses, alpha=0.7, color='brown', label='Entropy Loss')
    
    axes[0, 2].set_title('Training Losses')
    axes[0, 2].set_xlabel('Update Step')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Advantages distribution
    advantages = training_stats.get('advantages', [])
    if advantages:
        axes[1, 0].hist(advantages, bins=50, alpha=0.7, color='cyan', label='Advantages')
        axes[1, 0].set_title('Advantages Distribution')
        axes[1, 0].set_xlabel('Advantage Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Returns distribution
    returns = training_stats.get('returns', [])
    if returns:
        axes[1, 1].hist(returns, bins=50, alpha=0.7, color='magenta', label='Returns')
        axes[1, 1].set_title('Returns Distribution')
        axes[1, 1].set_xlabel('Return Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Performance summary
    recent_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
    stats_text = f"""
    Total Episodes: {len(episode_rewards)}
    Final Performance: {np.mean(recent_rewards):.2f} ± {np.std(recent_rewards):.2f}
    Best Performance: {np.max(episode_rewards):.2f}
    Target Achieved: {np.mean(recent_rewards) >= 475}
    Mean Advantage: {np.mean(advantages):.2f}
    Mean Return: {np.mean(returns):.2f}
    """
    
    axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes,
                    fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    axes[1, 2].set_title('Performance Summary')
    axes[1, 2].axis('off')
    
    plt.suptitle('Policy Gradient Training Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_hyperparameter_comparison(results: Dict, save_path: Optional[str] = None):
    """
    Plot hyperparameter comparison results.
    
    Args:
        results: Dictionary containing hyperparameter results
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract data
    learning_rates = [r['learning_rate'] for r in results]
    entropy_coefs = [r['entropy_coef'] for r in results]
    final_rewards = [r['final_reward'] for r in results]
    convergence_episodes = [r['convergence_episode'] for r in results]
    
    # Learning rate vs performance
    lr_unique = sorted(set(learning_rates))
    lr_performance = []
    lr_std = []
    
    for lr in lr_unique:
        lr_rewards = [r['final_reward'] for r in results if r['learning_rate'] == lr]
        lr_performance.append(np.mean(lr_rewards))
        lr_std.append(np.std(lr_rewards))
    
    axes[0, 0].errorbar(lr_unique, lr_performance, yerr=lr_std, 
                       marker='o', capsize=5, capthick=2)
    axes[0, 0].set_title('Learning Rate vs Performance')
    axes[0, 0].set_xlabel('Learning Rate')
    axes[0, 0].set_ylabel('Final Reward')
    axes[0, 0].set_xscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Entropy coefficient vs performance
    entropy_unique = sorted(set(entropy_coefs))
    entropy_performance = []
    entropy_std = []
    
    for entropy in entropy_unique:
        entropy_rewards = [r['final_reward'] for r in results if r['entropy_coef'] == entropy]
        entropy_performance.append(np.mean(entropy_rewards))
        entropy_std.append(np.std(entropy_rewards))
    
    axes[0, 1].errorbar(entropy_unique, entropy_performance, yerr=entropy_std, 
                       marker='o', capsize=5, capthick=2)
    axes[0, 1].set_title('Entropy Coefficient vs Performance')
    axes[0, 1].set_xlabel('Entropy Coefficient')
    axes[0, 1].set_ylabel('Final Reward')
    axes[0, 1].set_xscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Performance scatter plot
    scatter = axes[1, 0].scatter(learning_rates, entropy_coefs, 
                                c=final_rewards, s=100, alpha=0.7, cmap='viridis')
    axes[1, 0].set_title('Hyperparameter Space (colored by performance)')
    axes[1, 0].set_xlabel('Learning Rate')
    axes[1, 0].set_ylabel('Entropy Coefficient')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[1, 0])
    cbar.set_label('Final Reward')
    
    # Convergence speed
    axes[1, 1].scatter(final_rewards, convergence_episodes, alpha=0.7)
    axes[1, 1].set_title('Performance vs Convergence Speed')
    axes[1, 1].set_xlabel('Final Reward')
    axes[1, 1].set_ylabel('Convergence Episode')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Hyperparameter Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_algorithm_comparison(algorithm_results: Dict, save_path: Optional[str] = None):
    """
    Plot comparison between different algorithms.
    
    Args:
        algorithm_results: Dictionary mapping algorithm names to results
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract data
    algorithm_names = list(algorithm_results.keys())
    final_rewards = []
    convergence_episodes = []
    success_rates = []
    mean_lengths = []
    
    for name, results in algorithm_results.items():
        final_rewards.append(results['final_reward'])
        convergence_episodes.append(results['convergence_episode'])
        success_rates.append(results['success_rate'])
        mean_lengths.append(results['mean_length'])
    
    # Final performance
    bars1 = axes[0, 0].bar(algorithm_names, final_rewards, alpha=0.7, color='skyblue')
    axes[0, 0].axhline(y=475, color='green', linestyle='--', alpha=0.7, label='Target (475)')
    axes[0, 0].set_title('Final Performance')
    axes[0, 0].set_xlabel('Algorithm')
    axes[0, 0].set_ylabel('Mean Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars1, final_rewards):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                       f'{value:.1f}', ha='center', va='bottom')
    
    # Success rate
    bars2 = axes[0, 1].bar(algorithm_names, success_rates, alpha=0.7, color='lightgreen')
    axes[0, 1].set_title('Success Rate (≥475)')
    axes[0, 1].set_xlabel('Algorithm')
    axes[0, 1].set_ylabel('Success Rate')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars2, success_rates):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.2%}', ha='center', va='bottom')
    
    # Convergence speed
    bars3 = axes[1, 0].bar(algorithm_names, convergence_episodes, alpha=0.7, color='orange')
    axes[1, 0].set_title('Convergence Speed')
    axes[1, 0].set_xlabel('Algorithm')
    axes[1, 0].set_ylabel('Convergence Episode')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars3, convergence_episodes):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                       f'{value}', ha='center', va='bottom')
    
    # Episode length
    bars4 = axes[1, 1].bar(algorithm_names, mean_lengths, alpha=0.7, color='purple')
    axes[1, 1].set_title('Mean Episode Length')
    axes[1, 1].set_xlabel('Algorithm')
    axes[1, 1].set_ylabel('Steps')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars4, mean_lengths):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{value:.1f}', ha='center', va='bottom')
    
    plt.suptitle('Algorithm Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_convergence_analysis(rewards: List[float], window_size: int = 100,
                            save_path: Optional[str] = None):
    """
    Plot convergence analysis for training rewards.
    
    Args:
        rewards: List of episode rewards
        window_size: Window size for moving average
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Raw rewards and moving average
    axes[0, 0].plot(rewards, alpha=0.3, color='blue', label='Raw Rewards')
    
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        axes[0, 0].plot(range(window_size-1, len(rewards)), 
                       moving_avg, color='red', linewidth=2, label=f'Moving Avg ({window_size})')
    
    axes[0, 0].axhline(y=475, color='green', linestyle='--', label='Target (475)')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Convergence speed analysis
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        
        # Find convergence point
        convergence_threshold = 0.1
        convergence_episode = len(rewards)
        
        for i in range(window_size, len(moving_avg)):
            recent_std = np.std(moving_avg[i-window_size:i])
            if recent_std < convergence_threshold:
                convergence_episode = i
                break
        
        axes[0, 1].axvline(x=convergence_episode, color='red', linestyle='--', 
                          label=f'Convergence: Episode {convergence_episode}')
        axes[0, 1].plot(range(window_size-1, len(rewards)), moving_avg, 
                       color='blue', linewidth=2, label='Moving Average')
        axes[0, 1].set_title('Convergence Analysis')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Moving Average Reward')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Reward distribution
    axes[1, 0].hist(rewards, bins=50, alpha=0.7, color='skyblue', label='All Episodes')
    
    if len(rewards) >= 100:
        recent_rewards = rewards[-100:]
        axes[1, 0].hist(recent_rewards, bins=30, alpha=0.7, color='orange', 
                       label='Last 100 Episodes')
    
    axes[1, 0].axvline(x=475, color='green', linestyle='--', label='Target (475)')
    axes[1, 0].set_title('Reward Distribution')
    axes[1, 0].set_xlabel('Reward')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Performance metrics
    recent_rewards = rewards[-100:] if len(rewards) >= 100 else rewards
    metrics_text = f"""
    Total Episodes: {len(rewards)}
    Final Performance: {np.mean(recent_rewards):.2f} ± {np.std(recent_rewards):.2f}
    Best Performance: {np.max(rewards):.2f}
    Target Achieved: {np.mean(recent_rewards) >= 475}
    Convergence Episode: {convergence_episode if 'convergence_episode' in locals() else 'N/A'}
    """
    
    axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes,
                    fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    axes[1, 1].set_title('Performance Metrics')
    axes[1, 1].axis('off')
    
    plt.suptitle('Convergence Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def save_training_plots(training_stats: Dict, algorithm_name: str, 
                       save_dir: str = "plots"):
    """
    Save all training plots for an algorithm.
    
    Args:
        training_stats: Training statistics dictionary
        algorithm_name: Name of the algorithm
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save individual plots
    plot_training_analysis(training_stats, 
                          save_path=os.path.join(save_dir, f"{algorithm_name}_analysis.png"))
    
    if 'episode_rewards' in training_stats:
        plot_convergence_analysis(training_stats['episode_rewards'],
                                 save_path=os.path.join(save_dir, f"{algorithm_name}_convergence.png"))
    
    print(f"Training plots saved to {save_dir}/")


if __name__ == "__main__":
    # Test plotting functions
    print("Testing plotting functions...")
    
    # Generate dummy data
    np.random.seed(42)
    episodes = 1000
    
    # Simulate REINFORCE training
    reinforce_rewards = []
    for i in range(episodes):
        base_reward = 200 + 0.3 * i + np.random.normal(0, 50)
        reinforce_rewards.append(max(0, min(500, base_reward)))
    
    # Simulate REINFORCE with baseline training
    baseline_rewards = []
    for i in range(episodes):
        base_reward = 250 + 0.4 * i + np.random.normal(0, 30)
        baseline_rewards.append(max(0, min(500, base_reward)))
    
    # Test learning curves comparison
    rewards_data = {
        'REINFORCE': reinforce_rewards,
        'REINFORCE + Baseline': baseline_rewards
    }
    
    plot_learning_curves(rewards_data, save_path='test_learning_curves.png')
    
    # Test training analysis
    training_stats = {
        'episode_rewards': reinforce_rewards,
        'episode_lengths': [min(500, max(10, int(r/10))) for r in reinforce_rewards],
        'policy_losses': np.random.exponential(0.1, len(reinforce_rewards)).tolist(),
        'entropy_losses': np.random.exponential(0.05, len(reinforce_rewards)).tolist(),
        'advantages': np.random.normal(0, 1, len(reinforce_rewards) * 10).tolist(),
        'returns': np.random.normal(200, 100, len(reinforce_rewards) * 10).tolist()
    }
    
    plot_training_analysis(training_stats, save_path='test_training_analysis.png')
    
    print("Plotting functions test completed!")
