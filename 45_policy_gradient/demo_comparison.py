#!/usr/bin/env python3
"""
Policy Gradient Algorithms Comparison Demo

This script demonstrates and compares different Policy Gradient algorithms:
- Basic REINFORCE
- REINFORCE with Value Baseline

The goal is to achieve ≥475/500 average reward on CartPole-v1.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import gym
import os
import warnings
warnings.filterwarnings('ignore')

from reinforce_cartpole import train_reinforce_cartpole
from reinforce_baseline import train_reinforce_baseline_cartpole
from plot_rewards import plot_learning_curves, plot_algorithm_comparison


def compare_algorithms(num_episodes: int = 800, max_steps: int = 500):
    """
    Compare different Policy Gradient algorithms.
    
    Args:
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        
    Returns:
        Dictionary containing results for each algorithm
    """
    print("Policy Gradient Algorithms Comparison")
    print("=" * 50)
    
    algorithms = {
        'REINFORCE': {
            'function': train_reinforce_cartpole,
            'params': {
                'num_episodes': num_episodes,
                'max_steps': max_steps,
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'entropy_coef': 0.01,
                'verbose': False,
                'save_model': False,
                'plot_results': False
            }
        },
        'REINFORCE + Baseline': {
            'function': train_reinforce_baseline_cartpole,
            'params': {
                'num_episodes': num_episodes,
                'max_steps': max_steps,
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'entropy_coef': 0.01,
                'value_coef': 0.5,
                'verbose': False,
                'save_model': False,
                'plot_results': False
            }
        }
    }
    
    results = {}
    
    for name, config in algorithms.items():
        print(f"\nTraining {name}...")
        print("-" * 30)
        
        # Train algorithm
        agent = config['function'](**config['params'])
        
        # Get training statistics
        training_stats = {
            'episode_rewards': agent.episode_rewards,
            'episode_lengths': agent.episode_lengths,
            'policy_losses': agent.policy_losses,
            'entropy_losses': agent.entropy_losses
        }
        
        if hasattr(agent, 'value_losses'):
            training_stats['value_losses'] = agent.value_losses
        if hasattr(agent, 'advantages'):
            training_stats['advantages'] = agent.advantages
        if hasattr(agent, 'returns'):
            training_stats['returns'] = agent.returns
        
        # Evaluate agent
        env = gym.make('CartPole-v1')
        eval_results = agent.evaluate(env, num_episodes=100, max_steps=max_steps)
        env.close()
        
        # Store results
        results[name] = {
            'agent': agent,
            'training_stats': training_stats,
            'eval_results': eval_results,
            'final_reward': eval_results['mean_reward'],
            'success_rate': eval_results['success_rate'],
            'mean_length': eval_results['mean_length'],
            'convergence_episode': find_convergence_episode(training_stats['episode_rewards'])
        }
        
        print(f"Final Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        print(f"Success Rate: {eval_results['success_rate']:.2%}")
        print(f"Mean Length: {eval_results['mean_length']:.2f}")
    
    return results


def find_convergence_episode(rewards: list, window_size: int = 100, threshold: float = 0.1) -> int:
    """
    Find the episode where the algorithm converges.
    
    Args:
        rewards: List of episode rewards
        window_size: Window size for moving average
        threshold: Convergence threshold
        
    Returns:
        Convergence episode number
    """
    if len(rewards) < window_size:
        return len(rewards)
    
    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    
    for i in range(window_size, len(moving_avg)):
        recent_std = np.std(moving_avg[i-window_size:i])
        if recent_std < threshold:
            return i
    
    return len(rewards)


def plot_comparison_results(results: dict, save_path: str = None):
    """
    Plot comparison results.
    
    Args:
        results: Dictionary containing results for each algorithm
        save_path: Path to save plot
    """
    # Prepare data for learning curves
    rewards_data = {}
    for name, result in results.items():
        rewards_data[name] = result['training_stats']['episode_rewards']
    
    # Plot learning curves
    plot_learning_curves(rewards_data, save_path='plots/algorithm_comparison.png')
    
    # Plot algorithm comparison
    algorithm_results = {}
    for name, result in results.items():
        algorithm_results[name] = {
            'final_reward': result['final_reward'],
            'success_rate': result['success_rate'],
            'mean_length': result['mean_length'],
            'convergence_episode': result['convergence_episode']
        }
    
    plot_algorithm_comparison(algorithm_results, save_path='plots/algorithm_metrics.png')


def analyze_performance(results: dict):
    """
    Analyze and compare algorithm performance.
    
    Args:
        results: Dictionary containing results for each algorithm
    """
    print("\n" + "=" * 50)
    print("PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Final Reward: {result['final_reward']:.2f}")
        print(f"  Success Rate: {result['success_rate']:.2%}")
        print(f"  Mean Length: {result['mean_length']:.2f}")
        print(f"  Convergence: Episode {result['convergence_episode']}")
        
        # Check if target achieved
        if result['final_reward'] >= 475:
            print(f"  ✅ Target achieved!")
        else:
            print(f"  ❌ Target not achieved")
    
    # Find best algorithm
    best_algorithm = max(results.keys(), key=lambda x: results[x]['final_reward'])
    print(f"\nBest Algorithm: {best_algorithm}")
    print(f"Best Performance: {results[best_algorithm]['final_reward']:.2f}")


def main():
    """Main comparison function."""
    print("Policy Gradient Algorithms Comparison Demo")
    print("=" * 50)
    
    # Create directories
    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Compare algorithms
    results = compare_algorithms(num_episodes=800, max_steps=500)
    
    # Plot results
    plot_comparison_results(results)
    
    # Analyze performance
    analyze_performance(results)
    
    print("\n" + "=" * 50)
    print("COMPARISON COMPLETE!")
    print("=" * 50)
    print("Key Insights:")
    print("1. REINFORCE with baseline typically performs better")
    print("2. Value baseline reduces variance in policy gradient")
    print("3. Advantage computation improves learning stability")
    print("4. Both algorithms can achieve target performance with proper tuning")
    print("5. Policy gradient methods are effective for discrete action spaces")
    print("\nCheck the 'plots' directory for detailed visualizations!")


if __name__ == "__main__":
    main()
