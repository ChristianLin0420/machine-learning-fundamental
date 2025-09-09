#!/usr/bin/env python3
"""
A2C (Advantage Actor-Critic) with GAE Implementation for CartPole-v1

This script implements A2C with Generalized Advantage Estimation (GAE) for
stable and efficient policy gradient learning. The implementation includes:
- Shared or split actor-critic networks
- GAE(λ) for advantage estimation
- Entropy regularization for exploration
- Gradient clipping for stability
- Comprehensive logging and visualization

The goal is to achieve ≥475/500 average reward on CartPole-v1.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
import os
import warnings
warnings.filterwarnings('ignore')

from nets import create_actor_critic_network, count_parameters
from gae import compute_gae_advantages_tensor, normalize_advantages_tensor
from rollout import RolloutCollector
from utils import (
    set_seed, create_optimizer, compute_policy_loss, compute_value_loss,
    compute_entropy_loss, clip_gradients, TrainingLogger, early_stopping_check,
    save_model, load_model, compute_gradient_norm
)


class A2CAgent:
    """
    A2C (Advantage Actor-Critic) Agent with GAE.
    """
    
    def __init__(self, state_size: int, action_size: int,
                 learning_rate: float = 3e-4, gamma: float = 0.99,
                 lam: float = 0.95, entropy_coef: float = 0.01,
                 value_coef: float = 0.5, max_grad_norm: float = 0.5,
                 hidden_sizes: List[int] = [256, 256], network_type: str = 'shared',
                 device: str = 'cpu', normalize_advantages: bool = True):
        """
        Initialize A2C Agent.
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            lam: GAE parameter
            entropy_coef: Entropy regularization coefficient
            value_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            hidden_sizes: Hidden layer sizes
            network_type: 'shared' or 'split' architecture
            device: Device to run on
            normalize_advantages: Whether to normalize advantages
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lam = lam
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)
        self.normalize_advantages = normalize_advantages
        
        # Create actor-critic network
        self.network = create_actor_critic_network(
            state_size, action_size, hidden_sizes, 'relu', network_type
        ).to(self.device)
        
        # Create optimizer
        self.optimizer = create_optimizer(self.network, learning_rate, 'adam')
        
        # Training logger
        self.logger = TrainingLogger()
        
        # Training statistics
        self.update_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        
        print(f"Created A2C Agent:")
        print(f"  Network type: {network_type}")
        print(f"  Total parameters: {count_parameters(self.network)}")
        print(f"  Device: {self.device}")
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        """
        Get action from policy and value estimate.
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Tuple of (action, log_probability, value)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value = self.network.get_action(state_tensor, deterministic)
        
        return action.item(), log_prob.item(), value.item() if hasattr(value, 'item') else value
    
    def get_value(self, state: np.ndarray) -> float:
        """
        Get value estimate for state.
        
        Args:
            state: Current state
            
        Returns:
            Value estimate
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            value = self.network.get_value(state_tensor)
        
        return value.item() if hasattr(value, 'item') else value
    
    def update(self, rollout_data: Dict) -> Dict:
        """
        Update the actor-critic network using collected rollout data.
        
        Args:
            rollout_data: Dictionary containing rollout data
            
        Returns:
            Dictionary containing update statistics
        """
        # Extract data
        states = rollout_data['states']
        actions = rollout_data['actions']
        rewards = rollout_data['rewards']
        values = rollout_data['values']
        log_probs = rollout_data['log_probs']
        dones = rollout_data['dones']
        advantages = rollout_data['advantages']
        returns = rollout_data['returns']
        
        # Normalize advantages if requested
        if self.normalize_advantages:
            advantages = normalize_advantages_tensor(advantages)
        
        # Get current policy log probabilities and values
        current_log_probs, current_values = self.network.get_log_probs(states, actions)
        
        # Compute policy loss
        policy_loss = compute_policy_loss(current_log_probs, advantages)
        
        # Compute value loss
        value_loss = compute_value_loss(current_values, returns)
        
        # Compute entropy loss
        entropies = self.network.get_entropy(states)
        entropy_loss = compute_entropy_loss(entropies)
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
        
        # Update network
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Clip gradients
        clip_gradients(self.network, self.max_grad_norm)
        
        # Compute gradient norm
        grad_norm = compute_gradient_norm(self.network)
        
        # Update parameters
        self.optimizer.step()
        
        # Log update
        self.logger.log_update(
            policy_loss.item(), value_loss.item(), entropy_loss.item(),
            advantages.cpu().numpy().tolist(), returns.cpu().numpy().tolist(),
            grad_norm
        )
        
        self.update_count += 1
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item(),
            'grad_norm': grad_norm,
            'update_count': self.update_count
        }
    
    def train(self, env, num_updates: int, max_rollout_length: int = 2048,
              verbose: bool = True, target_reward: float = 475.0) -> Dict:
        """
        Train the A2C agent.
        
        Args:
            env: Environment to train on
            num_updates: Number of training updates
            max_rollout_length: Maximum length of each rollout
            verbose: Whether to print progress
            target_reward: Target reward for early stopping
            
        Returns:
            Training statistics
        """
        print(f"Training A2C agent for {num_updates} updates...")
        print(f"Target reward: {target_reward}")
        print(f"Max rollout length: {max_rollout_length}")
        print()
        
        # Create rollout collector
        collector = RolloutCollector(env, max_rollout_length, str(self.device), self.normalize_advantages)
        
        for update in range(num_updates):
            # Collect rollout
            rollout_data, rollout_stats = collector.collect_rollout(
                self.network, self.network, self.gamma, self.lam
            )
            
            # Update network
            update_stats = self.update(rollout_data)
            
            # Log episode statistics
            for reward, length in zip(rollout_stats['episode_rewards'], rollout_stats['episode_lengths']):
                self.logger.log_episode(reward, length)
                self.episode_rewards.append(reward)
                self.episode_lengths.append(length)
            
            # Log steps
            self.logger.log_steps(rollout_stats['total_steps'])
            
            # Print progress
            if verbose and (update + 1) % 10 == 0:
                recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0
                std_reward = np.std(recent_rewards) if recent_rewards else 0
                
                print(f"Update {update + 1}/{num_updates}, "
                      f"Avg Reward: {avg_reward:.2f} ± {std_reward:.2f}, "
                      f"Policy Loss: {update_stats['policy_loss']:.4f}, "
                      f"Value Loss: {update_stats['value_loss']:.4f}, "
                      f"Grad Norm: {update_stats['grad_norm']:.4f}")
                
                # Check for early stopping
                if early_stopping_check(self.episode_rewards, target_reward):
                    print(f"Early stopping at update {update + 1}!")
                    print(f"Target reward {target_reward} achieved!")
                    break
        
        return self.logger.get_stats()
    
    def evaluate(self, env, num_episodes: int = 100, max_steps: int = 500) -> Dict:
        """
        Evaluate the trained agent.
        
        Args:
            env: Environment to evaluate on
            num_episodes: Number of episodes to evaluate
            max_steps: Maximum steps per episode
            
        Returns:
            Evaluation statistics
        """
        eval_rewards = []
        eval_lengths = []
        
        for episode in range(num_episodes):
            state = env.reset()
            if hasattr(state, '__len__') and len(state) > 1:
                state = state[0] if isinstance(state, tuple) else state
            
            total_reward = 0
            steps = 0
            
            for _ in range(max_steps):
                action, _, _ = self.get_action(state, deterministic=True)
                result = env.step(action)
                
                if len(result) == 4:
                    next_state, reward, done, info = result
                else:
                    next_state, reward, done, truncated, info = result
                    done = done or truncated
                
                total_reward += reward
                steps += 1
                state = next_state
                
                if done:
                    break
            
            eval_rewards.append(total_reward)
            eval_lengths.append(steps)
        
        return {
            'eval_rewards': eval_rewards,
            'eval_lengths': eval_lengths,
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'success_rate': np.mean([r >= 475 for r in eval_rewards])
        }
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        additional_info = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'update_count': self.update_count,
            'gamma': self.gamma,
            'lam': self.lam,
            'entropy_coef': self.entropy_coef,
            'value_coef': self.value_coef,
        }
        
        save_model(self.network, self.optimizer, filepath, additional_info)
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        additional_info = load_model(self.network, self.optimizer, filepath, str(self.device))
        
        self.episode_rewards = additional_info.get('episode_rewards', [])
        self.episode_lengths = additional_info.get('episode_lengths', [])
        self.update_count = additional_info.get('update_count', 0)
        self.gamma = additional_info.get('gamma', 0.99)
        self.lam = additional_info.get('lam', 0.95)
        self.entropy_coef = additional_info.get('entropy_coef', 0.01)
        self.value_coef = additional_info.get('value_coef', 0.5)


def train_a2c_cartpole(num_updates: int = 1000, max_rollout_length: int = 2048,
                      learning_rate: float = 3e-4, gamma: float = 0.99,
                      lam: float = 0.95, entropy_coef: float = 0.01,
                      value_coef: float = 0.5, max_grad_norm: float = 0.5,
                      hidden_sizes: List[int] = [256, 256], network_type: str = 'shared',
                      device: str = 'cpu', verbose: bool = True,
                      save_model: bool = True, plot_results: bool = True) -> A2CAgent:
    """
    Train A2C agent on CartPole-v1.
    
    Args:
        num_updates: Number of training updates
        max_rollout_length: Maximum length of each rollout
        learning_rate: Learning rate
        gamma: Discount factor
        lam: GAE parameter
        entropy_coef: Entropy regularization coefficient
        value_coef: Value loss coefficient
        max_grad_norm: Maximum gradient norm for clipping
        hidden_sizes: Hidden layer sizes
        network_type: 'shared' or 'split' architecture
        device: Device to run on
        verbose: Whether to print progress
        save_model: Whether to save the trained model
        plot_results: Whether to plot training results
        
    Returns:
        Trained A2C agent
    """
    # Set seed for reproducibility
    set_seed(42)
    
    # Create environment
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"Environment: {env.spec.id}")
    print(f"State space: {state_size}")
    print(f"Action space: {action_size}")
    print()
    
    # Create agent
    agent = A2CAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=learning_rate,
        gamma=gamma,
        lam=lam,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        max_grad_norm=max_grad_norm,
        hidden_sizes=hidden_sizes,
        network_type=network_type,
        device=device,
        normalize_advantages=True
    )
    
    # Train agent
    training_stats = agent.train(env, num_updates, max_rollout_length, verbose=verbose)
    
    # Evaluate agent
    print("\nEvaluating trained agent...")
    eval_results = agent.evaluate(env, num_episodes=100, max_steps=500)
    
    print(f"\nEvaluation Results:")
    print(f"Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"Success Rate (≥475): {eval_results['success_rate']:.2%}")
    print(f"Mean Episode Length: {eval_results['mean_length']:.2f}")
    
    # Save model
    if save_model:
        os.makedirs('models', exist_ok=True)
        model_path = 'models/a2c_cartpole.pth'
        agent.save_model(model_path)
        print(f"\nModel saved to: {model_path}")
    
    # Plot results
    if plot_results:
        agent.logger.plot_training_curves(save_path='plots/a2c_training.png')
    
    env.close()
    
    return agent


def compare_network_types(num_updates: int = 500, max_rollout_length: int = 1024):
    """
    Compare shared vs split network architectures.
    
    Args:
        num_updates: Number of training updates
        max_rollout_length: Maximum length of each rollout
    """
    print("Comparing Network Architectures")
    print("=" * 50)
    
    network_types = ['shared', 'split']
    results = {}
    
    for network_type in network_types:
        print(f"\nTraining {network_type} network...")
        print("-" * 30)
        
        agent = train_a2c_cartpole(
            num_updates=num_updates,
            max_rollout_length=max_rollout_length,
            network_type=network_type,
            verbose=False,
            save_model=False,
            plot_results=False
        )
        
        # Evaluate
        env = gym.make('CartPole-v1')
        eval_results = agent.evaluate(env, num_episodes=100)
        env.close()
        
        results[network_type] = {
            'mean_reward': eval_results['mean_reward'],
            'success_rate': eval_results['success_rate'],
            'episode_rewards': agent.episode_rewards,
            'total_parameters': count_parameters(agent.network)
        }
        
        print(f"Final Reward: {eval_results['mean_reward']:.2f}")
        print(f"Success Rate: {eval_results['success_rate']:.2%}")
        print(f"Total Parameters: {count_parameters(agent.network)}")
    
    # Plot comparison
    plot_network_comparison(results, save_path='plots/network_comparison.png')
    
    return results


def plot_network_comparison(results: Dict, save_path: Optional[str] = None):
    """
    Plot comparison between network types.
    
    Args:
        results: Dictionary containing results for each network type
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Learning curves
    for name, result in results.items():
        rewards = result['episode_rewards']
        if len(rewards) >= 100:
            moving_avg = np.convolve(rewards, np.ones(100)/100, mode='valid')
            axes[0].plot(range(99, len(rewards)), moving_avg, label=f'{name} (MA)')
        else:
            axes[0].plot(rewards, alpha=0.3, label=f'{name} (Raw)')
    
    axes[0].axhline(y=475, color='green', linestyle='--', label='Target (475)')
    axes[0].set_title('Learning Curves Comparison')
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Performance metrics
    names = list(results.keys())
    mean_rewards = [results[name]['mean_reward'] for name in names]
    success_rates = [results[name]['success_rate'] for name in names]
    total_params = [results[name]['total_parameters'] for name in names]
    
    x = np.arange(len(names))
    width = 0.25
    
    bars1 = axes[1].bar(x - width, mean_rewards, width, label='Mean Reward', alpha=0.7)
    bars2 = axes[1].bar(x, [sr * 500 for sr in success_rates], width, label='Success Rate × 500', alpha=0.7)
    bars3 = axes[1].bar(x + width, [tp / 1000 for tp in total_params], width, label='Parameters (K)', alpha=0.7)
    
    axes[1].set_title('Performance Metrics')
    axes[1].set_xlabel('Network Type')
    axes[1].set_ylabel('Value')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2, height + 5,
                        f'{height:.1f}', ha='center', va='bottom')
    
    plt.suptitle('A2C Network Architecture Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs('plots', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def main():
    """Main training function."""
    print("A2C with GAE CartPole Training")
    print("=" * 50)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Train A2C agent
    agent = train_a2c_cartpole(
        num_updates=1000,
        max_rollout_length=2048,
        learning_rate=3e-4,
        gamma=0.99,
        lam=0.95,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        hidden_sizes=[256, 256],
        network_type='shared',
        device='cpu',
        verbose=True,
        save_model=True,
        plot_results=True
    )
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)
    print("Key Insights:")
    print("1. A2C combines policy gradient with value function learning")
    print("2. GAE provides low-variance advantage estimates")
    print("3. Shared networks are more parameter-efficient")
    print("4. Entropy regularization encourages exploration")
    print("5. Gradient clipping stabilizes training")
    print("\nCheck the 'plots' directory for visualizations!")


if __name__ == "__main__":
    main()
