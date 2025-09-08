#!/usr/bin/env python3
"""
REINFORCE with Value Baseline Implementation for CartPole-v1

This script implements REINFORCE with a learned value baseline to reduce variance:
- Policy network for action selection
- Value network for baseline estimation
- Advantage computation (returns - baseline)
- Separate optimizers for policy and value networks

The goal is to achieve ≥475/500 average reward on CartPole-v1 with lower variance.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import os
import warnings
warnings.filterwarnings('ignore')

from nets import create_policy_network, create_value_network
from utils import (
    compute_returns, compute_advantages, normalize_advantages,
    compute_policy_loss, compute_value_loss, compute_entropy_loss,
    TrainingLogger, set_seed, create_optimizer, early_stopping_check
)


class REINFORCEBaselineAgent:
    """
    REINFORCE Agent with learned value baseline.
    """
    
    def __init__(self, state_size: int, action_size: int, 
                 learning_rate: float = 3e-4, gamma: float = 0.99,
                 entropy_coef: float = 0.01, value_coef: float = 0.5,
                 hidden_sizes: List[int] = [128, 128], device: str = 'cpu'):
        """
        Initialize REINFORCE Agent with Baseline.
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            learning_rate: Learning rate for optimizers
            gamma: Discount factor
            entropy_coef: Entropy regularization coefficient
            value_coef: Value loss coefficient
            hidden_sizes: Hidden layer sizes for networks
            device: Device to run on
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.device = torch.device(device)
        
        # Create policy network
        self.policy_net = create_policy_network(
            state_size, action_size, hidden_sizes
        ).to(self.device)
        
        # Create value network
        self.value_net = create_value_network(
            state_size, hidden_sizes
        ).to(self.device)
        
        # Create optimizers
        self.policy_optimizer = create_optimizer(self.policy_net, learning_rate, 'adam')
        self.value_optimizer = create_optimizer(self.value_net, learning_rate, 'adam')
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.advantages = []
        self.returns = []
        
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
            action, log_prob = self.policy_net.get_action(state_tensor, deterministic)
            value = self.value_net(state_tensor)
        
        return action.item(), log_prob.item(), value.item()
    
    def get_values(self, states: List[np.ndarray]) -> List[float]:
        """
        Get value estimates for a list of states.
        
        Args:
            states: List of states
            
        Returns:
            List of value estimates
        """
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        
        with torch.no_grad():
            values = self.value_net(states_tensor)
        
        return values.cpu().numpy().tolist()
    
    def update_networks(self, states: List[np.ndarray], actions: List[int], 
                       rewards: List[float], log_probs: List[float]):
        """
        Update both policy and value networks.
        
        Args:
            states: List of states from episode
            actions: List of actions taken
            rewards: List of rewards received
            log_probs: List of log probabilities of actions
        """
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        log_probs_tensor = torch.FloatTensor(log_probs).to(self.device)
        
        # Compute returns (reward-to-go)
        returns = compute_returns(rewards, self.gamma)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # Get value estimates
        values = self.value_net(states_tensor)
        
        # Compute advantages
        advantages = compute_advantages(returns, values.detach().cpu().numpy().tolist())
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages for stability
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # Get current log probabilities
        current_log_probs = self.policy_net.get_log_probs(states_tensor, actions_tensor)
        
        # Compute policy loss
        policy_loss = compute_policy_loss(current_log_probs, advantages_tensor)
        
        # Compute value loss
        value_loss = compute_value_loss(values, returns_tensor)
        
        # Compute entropy loss for regularization
        entropies = self.policy_net.get_entropy(states_tensor)
        entropy_loss = compute_entropy_loss(entropies)
        
        # Total policy loss
        total_policy_loss = policy_loss - self.entropy_coef * entropy_loss
        
        # Update policy network
        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=0.5)
        self.policy_optimizer.step()
        
        # Update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), max_norm=0.5)
        self.value_optimizer.step()
        
        # Store losses and statistics
        self.policy_losses.append(policy_loss.item())
        self.value_losses.append(value_loss.item())
        self.entropy_losses.append(entropy_loss.item())
        self.advantages.extend(advantages)
        self.returns.extend(returns)
    
    def train_episode(self, env, max_steps: int = 500) -> Tuple[float, int]:
        """
        Train for one episode.
        
        Args:
            env: Environment
            max_steps: Maximum steps per episode
            
        Returns:
            Tuple of (total_reward, episode_length)
        """
        state = env.reset()
        if hasattr(state, '__len__') and len(state) > 1:
            state = state[0] if isinstance(state, tuple) else state
        
        states = []
        actions = []
        rewards = []
        log_probs = []
        
        total_reward = 0
        
        for step in range(max_steps):
            # Get action from policy
            action, log_prob, value = self.get_action(state, deterministic=False)
            
            # Take action
            result = env.step(action)
            if len(result) == 4:
                next_state, reward, done, info = result
            else:
                next_state, reward, done, truncated, info = result
                done = done or truncated
            
            # Store experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        # Update networks
        self.update_networks(states, actions, rewards, log_probs)
        
        # Store episode statistics
        episode_length = len(states)
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(episode_length)
        
        return total_reward, episode_length
    
    def train(self, env, num_episodes: int, max_steps: int = 500, 
              verbose: bool = True, target_reward: float = 475.0) -> dict:
        """
        Train the agent for multiple episodes.
        
        Args:
            env: Environment
            num_episodes: Number of episodes to train
            max_steps: Maximum steps per episode
            verbose: Whether to print progress
            target_reward: Target reward for early stopping
            
        Returns:
            Training statistics
        """
        print(f"Training REINFORCE with Baseline agent for {num_episodes} episodes...")
        print(f"Target reward: {target_reward}")
        print(f"Device: {self.device}")
        print()
        
        for episode in range(num_episodes):
            reward, length = self.train_episode(env, max_steps)
            
            if verbose and (episode + 1) % 100 == 0:
                recent_rewards = self.episode_rewards[-100:]
                avg_reward = np.mean(recent_rewards)
                std_reward = np.std(recent_rewards)
                
                print(f"Episode {episode + 1}, "
                      f"Avg Reward: {avg_reward:.2f} ± {std_reward:.2f}, "
                      f"Length: {length}")
                
                # Check for early stopping
                if early_stopping_check(self.episode_rewards, target_reward):
                    print(f"Early stopping at episode {episode + 1}!")
                    print(f"Target reward {target_reward} achieved!")
                    break
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'entropy_losses': self.entropy_losses,
            'advantages': self.advantages,
            'returns': self.returns
        }
    
    def evaluate(self, env, num_episodes: int = 100, max_steps: int = 500) -> dict:
        """
        Evaluate the trained agent.
        
        Args:
            env: Environment
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
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'entropy_losses': self.entropy_losses
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        self.policy_losses = checkpoint['policy_losses']
        self.value_losses = checkpoint['value_losses']
        self.entropy_losses = checkpoint['entropy_losses']


def train_reinforce_baseline_cartpole(num_episodes: int = 1000, max_steps: int = 500,
                                    learning_rate: float = 3e-4, gamma: float = 0.99,
                                    entropy_coef: float = 0.01, value_coef: float = 0.5,
                                    hidden_sizes: List[int] = [128, 128], device: str = 'cpu',
                                    verbose: bool = True, save_model: bool = True,
                                    plot_results: bool = True) -> REINFORCEBaselineAgent:
    """
    Train REINFORCE with baseline agent on CartPole-v1.
    
    Args:
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        learning_rate: Learning rate
        gamma: Discount factor
        entropy_coef: Entropy regularization coefficient
        value_coef: Value loss coefficient
        hidden_sizes: Hidden layer sizes
        device: Device to run on
        verbose: Whether to print progress
        save_model: Whether to save the trained model
        plot_results: Whether to plot training results
        
    Returns:
        Trained REINFORCE with baseline agent
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
    agent = REINFORCEBaselineAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=learning_rate,
        gamma=gamma,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        hidden_sizes=hidden_sizes,
        device=device
    )
    
    # Train agent
    training_stats = agent.train(env, num_episodes, max_steps, verbose=verbose)
    
    # Evaluate agent
    print("\nEvaluating trained agent...")
    eval_results = agent.evaluate(env, num_episodes=100, max_steps=max_steps)
    
    print(f"\nEvaluation Results:")
    print(f"Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"Success Rate (≥475): {eval_results['success_rate']:.2%}")
    print(f"Mean Episode Length: {eval_results['mean_length']:.2f}")
    
    # Save model
    if save_model:
        os.makedirs('models', exist_ok=True)
        model_path = 'models/reinforce_baseline_cartpole.pth'
        agent.save_model(model_path)
        print(f"\nModel saved to: {model_path}")
    
    # Plot results
    if plot_results:
        plot_training_results(training_stats, save_path='plots/reinforce_baseline_training.png')
    
    env.close()
    
    return agent


def plot_training_results(training_stats: dict, save_path: Optional[str] = None):
    """
    Plot training results.
    
    Args:
        training_stats: Training statistics dictionary
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Episode rewards
    episode_rewards = training_stats['episode_rewards']
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
    episode_lengths = training_stats['episode_lengths']
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
    if training_stats['policy_losses']:
        axes[0, 2].plot(training_stats['policy_losses'], alpha=0.7, color='orange', label='Policy Loss')
    if training_stats['value_losses']:
        axes[0, 2].plot(training_stats['value_losses'], alpha=0.7, color='purple', label='Value Loss')
    if training_stats['entropy_losses']:
        axes[0, 2].plot(training_stats['entropy_losses'], alpha=0.7, color='brown', label='Entropy Loss')
    
    axes[0, 2].set_title('Training Losses')
    axes[0, 2].set_xlabel('Update Step')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Advantages distribution
    if training_stats['advantages']:
        axes[1, 0].hist(training_stats['advantages'], bins=50, alpha=0.7, color='cyan', label='Advantages')
        axes[1, 0].set_title('Advantages Distribution')
        axes[1, 0].set_xlabel('Advantage Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Returns distribution
    if training_stats['returns']:
        axes[1, 1].hist(training_stats['returns'], bins=50, alpha=0.7, color='magenta', label='Returns')
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
    Mean Advantage: {np.mean(training_stats['advantages']):.2f}
    Mean Return: {np.mean(training_stats['returns']):.2f}
    """
    
    axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes,
                    fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    axes[1, 2].set_title('Performance Summary')
    axes[1, 2].axis('off')
    
    plt.suptitle('REINFORCE with Baseline Training Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs('plots', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def main():
    """Main training function."""
    print("REINFORCE with Baseline CartPole Training")
    print("=" * 50)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Train REINFORCE with baseline agent
    agent = train_reinforce_baseline_cartpole(
        num_episodes=1000,
        max_steps=500,
        learning_rate=3e-4,
        gamma=0.99,
        entropy_coef=0.01,
        value_coef=0.5,
        hidden_sizes=[128, 128],
        device='cpu',
        verbose=True,
        save_model=True,
        plot_results=True
    )
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)
    print("Key Insights:")
    print("1. Value baseline reduces variance in policy gradient")
    print("2. Separate networks allow independent learning")
    print("3. Advantage computation improves learning stability")
    print("4. Baseline helps with credit assignment")
    print("5. Lower variance leads to more stable training")
    print("\nCheck the 'plots' directory for visualizations!")


if __name__ == "__main__":
    main()
