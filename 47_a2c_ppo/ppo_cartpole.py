#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) Implementation for CartPole-v1

This script implements PPO with clipped objective for stable and efficient
policy gradient learning. The implementation includes:
- Clipped objective with K epochs over shuffled mini-batches
- KL divergence tracking and adaptive learning rate
- Entropy bonus and value loss
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
    set_seed, create_optimizer, compute_value_loss,
    compute_entropy_loss, clip_gradients, TrainingLogger, early_stopping_check,
    save_model, load_model, compute_gradient_norm, moving_average
)


class PPOAgent:
    """
    PPO (Proximal Policy Optimization) Agent with clipped objective.
    """
    
    def __init__(self, state_size: int, action_size: int,
                 learning_rate: float = 3e-4, gamma: float = 0.99,
                 lam: float = 0.95, entropy_coef: float = 0.01,
                 value_coef: float = 0.5, max_grad_norm: float = 0.5,
                 clip_ratio: float = 0.2, ppo_epochs: int = 4,
                 mini_batch_size: int = 64, target_kl: float = 0.01,
                 hidden_sizes: List[int] = [256, 256], network_type: str = 'shared',
                 device: str = 'cpu', normalize_advantages: bool = True):
        """
        Initialize PPO Agent.
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            lam: GAE parameter
            entropy_coef: Entropy regularization coefficient
            value_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            clip_ratio: PPO clipping ratio
            ppo_epochs: Number of PPO epochs per update
            mini_batch_size: Size of mini-batches
            target_kl: Target KL divergence for early stopping
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
        self.clip_ratio = clip_ratio
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.target_kl = target_kl
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
        self.kl_divergences = []
        self.clip_fractions = []
        self.explained_variances = []
        
        print(f"Created PPO Agent:")
        print(f"  Network type: {network_type}")
        print(f"  Total parameters: {count_parameters(self.network)}")
        print(f"  Device: {self.device}")
        print(f"  PPO epochs: {ppo_epochs}")
        print(f"  Mini-batch size: {mini_batch_size}")
        print(f"  Clip ratio: {clip_ratio}")
    
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
    
    def compute_kl_divergence(self, old_log_probs: torch.Tensor, 
                             new_log_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between old and new policies.
        
        Args:
            old_log_probs: Old log probabilities
            new_log_probs: New log probabilities
            
        Returns:
            KL divergence
        """
        return (old_log_probs - new_log_probs).mean()
    
    def compute_clip_fraction(self, ratios: torch.Tensor) -> float:
        """
        Compute fraction of ratios that were clipped.
        
        Args:
            ratios: Policy ratios
            
        Returns:
            Clip fraction
        """
        return (torch.abs(ratios - 1.0) > self.clip_ratio).float().mean().item()
    
    def compute_explained_variance(self, values: torch.Tensor, returns: torch.Tensor) -> float:
        """
        Compute explained variance of value function.
        
        Args:
            values: Value estimates
            returns: Target returns
            
        Returns:
            Explained variance
        """
        var_y = torch.var(returns)
        return 1 - torch.var(returns - values) / var_y if var_y > 0 else 0
    
    def update(self, rollout_data: Dict) -> Dict:
        """
        Update the actor-critic network using PPO.
        
        Args:
            rollout_data: Dictionary containing rollout data
            
        Returns:
            Dictionary containing update statistics
        """
        # Extract data
        states = rollout_data['states']
        actions = rollout_data['actions']
        old_log_probs = rollout_data['log_probs']
        advantages = rollout_data['advantages']
        returns = rollout_data['returns']
        
        # Normalize advantages if requested
        if self.normalize_advantages:
            advantages = normalize_advantages_tensor(advantages)
        
        # Get buffer size
        buffer_size = states.size(0)
        
        # Training statistics
        policy_losses = []
        value_losses = []
        entropy_losses = []
        kl_divergences = []
        clip_fractions = []
        explained_variances = []
        
        # PPO epochs
        for epoch in range(self.ppo_epochs):
            # Shuffle data
            indices = torch.randperm(buffer_size, device=self.device)
            
            # Mini-batch training
            for start_idx in range(0, buffer_size, self.mini_batch_size):
                end_idx = min(start_idx + self.mini_batch_size, buffer_size)
                batch_indices = indices[start_idx:end_idx]
                
                # Get mini-batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Get current policy log probabilities and values
                current_log_probs, current_values = self.network.get_log_probs(batch_states, batch_actions)
                
                # Compute policy ratio
                ratio = torch.exp(current_log_probs - batch_old_log_probs)
                
                # Compute clipped objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                value_loss = compute_value_loss(current_values, batch_returns)
                
                # Compute entropy loss
                entropies = self.network.get_entropy(batch_states)
                entropy_loss = compute_entropy_loss(entropies)
                
                # Total loss
                total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
                
                # Update network
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Clip gradients
                clip_gradients(self.network, self.max_grad_norm)
                
                # Update parameters
                self.optimizer.step()
                
                # Compute statistics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                
                # Compute KL divergence
                kl_div = self.compute_kl_divergence(batch_old_log_probs, current_log_probs)
                kl_divergences.append(kl_div.item())
                
                # Compute clip fraction
                clip_frac = self.compute_clip_fraction(ratio)
                clip_fractions.append(clip_frac)
                
                # Compute explained variance
                exp_var = self.compute_explained_variance(current_values, batch_returns)
                explained_variances.append(exp_var.item())
                
                # Early stopping if KL divergence is too high
                if kl_div > 1.5 * self.target_kl:
                    print(f"Early stopping at epoch {epoch}, mini-batch {start_idx // self.mini_batch_size}")
                    break
            
            # Early stopping if KL divergence is too high
            if kl_divergences and np.mean(kl_divergences) > 1.5 * self.target_kl:
                break
        
        # Compute final statistics
        final_policy_loss = np.mean(policy_losses)
        final_value_loss = np.mean(value_losses)
        final_entropy_loss = np.mean(entropy_losses)
        final_kl_div = np.mean(kl_divergences)
        final_clip_frac = np.mean(clip_fractions)
        final_exp_var = np.mean(explained_variances)
        
        # Log update
        self.logger.log_update(
            final_policy_loss, final_value_loss, final_entropy_loss,
            advantages.cpu().numpy().tolist(), returns.cpu().numpy().tolist()
        )
        
        # Store additional statistics
        self.kl_divergences.append(final_kl_div)
        self.clip_fractions.append(final_clip_frac)
        self.explained_variances.append(final_exp_var)
        
        self.update_count += 1
        
        return {
            'policy_loss': final_policy_loss,
            'value_loss': final_value_loss,
            'entropy_loss': final_entropy_loss,
            'kl_divergence': final_kl_div,
            'clip_fraction': final_clip_frac,
            'explained_variance': final_exp_var,
            'update_count': self.update_count
        }
    
    def train(self, env, num_updates: int, max_rollout_length: int = 2048,
              verbose: bool = True, target_reward: float = 475.0) -> Dict:
        """
        Train the PPO agent.
        
        Args:
            env: Environment to train on
            num_updates: Number of training updates
            max_rollout_length: Maximum length of each rollout
            verbose: Whether to print progress
            target_reward: Target reward for early stopping
            
        Returns:
            Training statistics
        """
        print(f"Training PPO agent for {num_updates} updates...")
        print(f"Target reward: {target_reward}")
        print(f"Max rollout length: {max_rollout_length}")
        print()
        
        # Create rollout collector
        collector = RolloutCollector(env, max_rollout_length, str(self.device), self.gamma, self.lam)
        
        for update in range(num_updates):
            # Collect rollout
            rollout_data, rollout_stats = collector.collect_rollout(
                self.network, self.network
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
                      f"KL Div: {update_stats['kl_divergence']:.4f}, "
                      f"Clip Frac: {update_stats['clip_fraction']:.4f}")
                
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
    
    def plot_training_curves(self, save_path: Optional[str] = None, window_size: int = 100):
        """
        Plot training curves including PPO-specific metrics.
        
        Args:
            save_path: Path to save plot
            window_size: Window size for moving averages
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Episode rewards
        if self.episode_rewards:
            axes[0, 0].plot(self.episode_rewards, alpha=0.3, color='blue', label='Raw Rewards')
            
            if len(self.episode_rewards) >= window_size:
                moving_avg = moving_average(self.episode_rewards, window_size)
                axes[0, 0].plot(moving_avg, color='red', linewidth=2, label=f'Moving Avg ({window_size})')
            
            axes[0, 0].axhline(y=475, color='green', linestyle='--', label='Target (475)')
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Episode lengths
        if self.episode_lengths:
            axes[0, 1].plot(self.episode_lengths, alpha=0.3, color='green', label='Raw Lengths')
            
            if len(self.episode_lengths) >= window_size:
                moving_avg = moving_average(self.episode_lengths, window_size)
                axes[0, 1].plot(moving_avg, color='red', linewidth=2, label=f'Moving Avg ({window_size})')
            
            axes[0, 1].set_title('Episode Lengths')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Steps')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Losses
        if self.logger.policy_losses:
            axes[0, 2].plot(self.logger.policy_losses, alpha=0.7, color='orange', label='Policy Loss')
        if self.logger.value_losses:
            axes[0, 2].plot(self.logger.value_losses, alpha=0.7, color='purple', label='Value Loss')
        if self.logger.entropy_losses:
            axes[0, 2].plot(self.logger.entropy_losses, alpha=0.7, color='brown', label='Entropy Loss')
        
        axes[0, 2].set_title('Training Losses')
        axes[0, 2].set_xlabel('Update')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # KL Divergence
        if self.kl_divergences:
            axes[1, 0].plot(self.kl_divergences, alpha=0.7, color='red', label='KL Divergence')
            axes[1, 0].axhline(y=self.target_kl, color='orange', linestyle='--', label=f'Target KL ({self.target_kl})')
            axes[1, 0].set_title('KL Divergence')
            axes[1, 0].set_xlabel('Update')
            axes[1, 0].set_ylabel('KL Divergence')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Clip Fraction
        if self.clip_fractions:
            axes[1, 1].plot(self.clip_fractions, alpha=0.7, color='cyan', label='Clip Fraction')
            axes[1, 1].set_title('Clip Fraction')
            axes[1, 1].set_xlabel('Update')
            axes[1, 1].set_ylabel('Fraction Clipped')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Explained Variance
        if self.explained_variances:
            axes[1, 2].plot(self.explained_variances, alpha=0.7, color='magenta', label='Explained Variance')
            axes[1, 2].set_title('Explained Variance')
            axes[1, 2].set_xlabel('Update')
            axes[1, 2].set_ylabel('Explained Variance')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle('PPO Training Progress', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        additional_info = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'kl_divergences': self.kl_divergences,
            'clip_fractions': self.clip_fractions,
            'explained_variances': self.explained_variances,
            'update_count': self.update_count,
            'gamma': self.gamma,
            'lam': self.lam,
            'entropy_coef': self.entropy_coef,
            'value_coef': self.value_coef,
            'clip_ratio': self.clip_ratio,
            'ppo_epochs': self.ppo_epochs,
            'target_kl': self.target_kl
        }
        
        save_model(self.network, self.optimizer, filepath, additional_info)
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        additional_info = load_model(self.network, self.optimizer, filepath, str(self.device))
        
        self.episode_rewards = additional_info.get('episode_rewards', [])
        self.episode_lengths = additional_info.get('episode_lengths', [])
        self.kl_divergences = additional_info.get('kl_divergences', [])
        self.clip_fractions = additional_info.get('clip_fractions', [])
        self.explained_variances = additional_info.get('explained_variances', [])
        self.update_count = additional_info.get('update_count', 0)
        self.gamma = additional_info.get('gamma', 0.99)
        self.lam = additional_info.get('lam', 0.95)
        self.entropy_coef = additional_info.get('entropy_coef', 0.01)
        self.value_coef = additional_info.get('value_coef', 0.5)
        self.clip_ratio = additional_info.get('clip_ratio', 0.2)
        self.ppo_epochs = additional_info.get('ppo_epochs', 4)
        self.target_kl = additional_info.get('target_kl', 0.01)


def train_ppo_cartpole(num_updates: int = 1000, max_rollout_length: int = 2048,
                      learning_rate: float = 3e-4, gamma: float = 0.99,
                      lam: float = 0.95, entropy_coef: float = 0.01,
                      value_coef: float = 0.5, max_grad_norm: float = 0.5,
                      clip_ratio: float = 0.2, ppo_epochs: int = 4,
                      mini_batch_size: int = 64, target_kl: float = 0.01,
                      hidden_sizes: List[int] = [256, 256], network_type: str = 'shared',
                      device: str = 'cpu', verbose: bool = True,
                      save_model: bool = True, plot_results: bool = True) -> PPOAgent:
    """
    Train PPO agent on CartPole-v1.
    
    Args:
        num_updates: Number of training updates
        max_rollout_length: Maximum length of each rollout
        learning_rate: Learning rate
        gamma: Discount factor
        lam: GAE parameter
        entropy_coef: Entropy regularization coefficient
        value_coef: Value loss coefficient
        max_grad_norm: Maximum gradient norm for clipping
        clip_ratio: PPO clipping ratio
        ppo_epochs: Number of PPO epochs per update
        mini_batch_size: Size of mini-batches
        target_kl: Target KL divergence for early stopping
        hidden_sizes: Hidden layer sizes
        network_type: 'shared' or 'split' architecture
        device: Device to run on
        verbose: Whether to print progress
        save_model: Whether to save the trained model
        plot_results: Whether to plot training results
        
    Returns:
        Trained PPO agent
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
    agent = PPOAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=learning_rate,
        gamma=gamma,
        lam=lam,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        max_grad_norm=max_grad_norm,
        clip_ratio=clip_ratio,
        ppo_epochs=ppo_epochs,
        mini_batch_size=mini_batch_size,
        target_kl=target_kl,
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
        model_path = 'models/ppo_cartpole.pth'
        agent.save_model(model_path)
        print(f"\nModel saved to: {model_path}")
    
    # Plot results
    if plot_results:
        agent.plot_training_curves(save_path='plots/ppo_training.png')
    
    env.close()
    
    return agent


def main():
    """Main training function."""
    print("PPO CartPole Training")
    print("=" * 50)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Train PPO agent
    agent = train_ppo_cartpole(
        num_updates=1000,
        max_rollout_length=2048,
        learning_rate=3e-4,
        gamma=0.99,
        lam=0.95,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        clip_ratio=0.2,
        ppo_epochs=4,
        mini_batch_size=64,
        target_kl=0.01,
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
    print("1. PPO uses clipped objective for stable policy updates")
    print("2. Multiple epochs over mini-batches improve sample efficiency")
    print("3. KL divergence tracking prevents policy collapse")
    print("4. GAE provides low-variance advantage estimates")
    print("5. Entropy regularization encourages exploration")
    print("\nCheck the 'plots' directory for visualizations!")


if __name__ == "__main__":
    main()
