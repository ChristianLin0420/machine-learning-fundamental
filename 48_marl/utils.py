"""
Utilities for Multi-Agent Reinforcement Learning

This module provides utility functions for MARL training including
seeding, logging, minibatch creation, and other helper functions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Any
import os
import json
from datetime import datetime
import warnings


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_optimizer(network: torch.nn.Module, learning_rate: float = 3e-4,
                    optimizer_type: str = 'adam', weight_decay: float = 0.0) -> torch.optim.Optimizer:
    """
    Create optimizer for network.
    
    Args:
        network: PyTorch network
        learning_rate: Learning rate
        optimizer_type: Type of optimizer ('adam', 'sgd', 'rmsprop')
        weight_decay: Weight decay for regularization
        
    Returns:
        PyTorch optimizer
    """
    if optimizer_type.lower() == 'adam':
        return torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'sgd':
        return torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'rmsprop':
        return torch.optim.RMSprop(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def compute_policy_loss(log_probs: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
    """
    Compute policy gradient loss.
    
    Args:
        log_probs: Log probabilities of actions
        advantages: Advantage estimates
        
    Returns:
        Policy loss
    """
    return -(log_probs * advantages).mean()


def compute_value_loss(values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
    """
    Compute value function loss (MSE).
    
    Args:
        values: Value estimates
        returns: Target returns
        
    Returns:
        Value loss
    """
    return torch.nn.functional.mse_loss(values, returns)


def compute_entropy_loss(entropies: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy loss (negative entropy for regularization).
    
    Args:
        entropies: Entropy values
        
    Returns:
        Entropy loss
    """
    return -entropies.mean()


def clip_gradients(model: torch.nn.Module, max_norm: float = 0.5):
    """
    Clip gradients to prevent exploding gradients.
    
    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def moving_average(data: List[float], window_size: int = 100) -> List[float]:
    """
    Compute moving average of data.
    
    Args:
        data: List of values
        window_size: Window size for moving average
        
    Returns:
        List of moving averages
    """
    if len(data) < window_size:
        return data
    
    moving_avg = []
    for i in range(len(data)):
        start_idx = max(0, i - window_size + 1)
        window_data = data[start_idx:i+1]
        moving_avg.append(np.mean(window_data))
    
    return moving_avg


def compute_returns(rewards: List[float], gamma: float = 0.99) -> List[float]:
    """
    Compute discounted returns.
    
    Args:
        rewards: List of rewards
        gamma: Discount factor
        
    Returns:
        List of returns
    """
    returns = []
    running_return = 0
    
    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        returns.insert(0, running_return)
    
    return returns


def normalize_advantages(advantages: List[float], eps: float = 1e-8) -> List[float]:
    """
    Normalize advantages to have zero mean and unit variance.
    
    Args:
        advantages: List of advantages
        eps: Small constant for numerical stability
        
    Returns:
        List of normalized advantages
    """
    if len(advantages) == 0:
        return advantages
    
    mean_adv = np.mean(advantages)
    std_adv = np.std(advantages)
    
    if std_adv < eps:
        return [0.0] * len(advantages)
    
    return [(adv - mean_adv) / (std_adv + eps) for adv in advantages]


def normalize_advantages_tensor(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize advantages tensor to have zero mean and unit variance.
    
    Args:
        advantages: Advantage tensor
        eps: Small constant for numerical stability
        
    Returns:
        Normalized advantage tensor
    """
    if advantages.numel() == 0:
        return advantages
    
    mean_adv = advantages.mean()
    std_adv = advantages.std()
    
    if std_adv < eps:
        return torch.zeros_like(advantages)
    
    return (advantages - mean_adv) / (std_adv + eps)


class MARLLogger:
    """
    Logger for Multi-Agent RL training progress and metrics.
    """
    
    def __init__(self, log_dir: str = "logs", save_frequency: int = 100):
        """
        Initialize MARL Logger.
        
        Args:
            log_dir: Directory to save logs
            save_frequency: Frequency to save logs
        """
        self.log_dir = log_dir
        self.save_frequency = save_frequency
        os.makedirs(log_dir, exist_ok=True)
        
        # Training metrics
        self.episode_rewards = {}
        self.episode_lengths = {}
        self.policy_losses = {}
        self.value_losses = {}
        self.entropy_losses = {}
        self.advantages = {}
        self.returns = {}
        self.grad_norms = {}
        self.kl_divergences = {}
        self.clip_fractions = {}
        self.explained_variances = {}
        
        # Training statistics
        self.start_time = datetime.now()
        self.total_steps = 0
        self.total_episodes = 0
        self.update_count = 0
        
    def log_episode(self, agent_rewards: Dict[str, float], agent_lengths: Dict[str, int]):
        """
        Log episode statistics for all agents.
        
        Args:
            agent_rewards: Dictionary mapping agent names to episode rewards
            agent_lengths: Dictionary mapping agent names to episode lengths
        """
        for agent_name, reward in agent_rewards.items():
            if agent_name not in self.episode_rewards:
                self.episode_rewards[agent_name] = []
            self.episode_rewards[agent_name].append(reward)
        
        for agent_name, length in agent_lengths.items():
            if agent_name not in self.episode_lengths:
                self.episode_lengths[agent_name] = []
            self.episode_lengths[agent_name].append(length)
        
        self.total_episodes += 1
    
    def log_update(self, update_stats: Dict[str, Dict[str, float]], 
                   advantages: Dict[str, List[float]], returns: Dict[str, List[float]]):
        """
        Log update statistics for all agents.
        
        Args:
            update_stats: Dictionary mapping agent names to update statistics
            advantages: Dictionary mapping agent names to advantage lists
            returns: Dictionary mapping agent names to return lists
        """
        for agent_name, stats in update_stats.items():
            # Log losses
            if agent_name not in self.policy_losses:
                self.policy_losses[agent_name] = []
                self.value_losses[agent_name] = []
                self.entropy_losses[agent_name] = []
                self.kl_divergences[agent_name] = []
                self.clip_fractions[agent_name] = []
                self.explained_variances[agent_name] = []
            
            self.policy_losses[agent_name].append(stats['policy_loss'])
            self.value_losses[agent_name].append(stats['value_loss'])
            self.entropy_losses[agent_name].append(stats['entropy_loss'])
            self.kl_divergences[agent_name].append(stats['kl_divergence'])
            self.clip_fractions[agent_name].append(stats['clip_fraction'])
            self.explained_variances[agent_name].append(stats['explained_variance'])
            
            # Log advantages and returns
            if agent_name not in self.advantages:
                self.advantages[agent_name] = []
                self.returns[agent_name] = []
            
            self.advantages[agent_name].extend(advantages[agent_name])
            self.returns[agent_name].extend(returns[agent_name])
        
        self.update_count += 1
    
    def log_steps(self, num_steps: int):
        """
        Log number of steps.
        
        Args:
            num_steps: Number of steps
        """
        self.total_steps += num_steps
    
    def get_stats(self, window_size: int = 100) -> Dict:
        """
        Get training statistics.
        
        Args:
            window_size: Window size for moving averages
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'total_updates': self.update_count,
            'total_time': (datetime.now() - self.start_time).total_seconds(),
        }
        
        # Per-agent statistics
        for agent_name in self.episode_rewards.keys():
            if self.episode_rewards[agent_name]:
                stats[f'{agent_name}_mean_reward'] = np.mean(self.episode_rewards[agent_name])
                stats[f'{agent_name}_std_reward'] = np.std(self.episode_rewards[agent_name])
                stats[f'{agent_name}_max_reward'] = np.max(self.episode_rewards[agent_name])
                stats[f'{agent_name}_min_reward'] = np.min(self.episode_rewards[agent_name])
                
                if len(self.episode_rewards[agent_name]) >= window_size:
                    recent_rewards = self.episode_rewards[agent_name][-window_size:]
                    stats[f'{agent_name}_recent_mean_reward'] = np.mean(recent_rewards)
                    stats[f'{agent_name}_recent_std_reward'] = np.std(recent_rewards)
            
            if self.episode_lengths[agent_name]:
                stats[f'{agent_name}_mean_length'] = np.mean(self.episode_lengths[agent_name])
                stats[f'{agent_name}_std_length'] = np.std(self.episode_lengths[agent_name])
            
            if self.policy_losses[agent_name]:
                stats[f'{agent_name}_mean_policy_loss'] = np.mean(self.policy_losses[agent_name])
            if self.value_losses[agent_name]:
                stats[f'{agent_name}_mean_value_loss'] = np.mean(self.value_losses[agent_name])
            if self.entropy_losses[agent_name]:
                stats[f'{agent_name}_mean_entropy_loss'] = np.mean(self.entropy_losses[agent_name])
            if self.kl_divergences[agent_name]:
                stats[f'{agent_name}_mean_kl_divergence'] = np.mean(self.kl_divergences[agent_name])
            if self.clip_fractions[agent_name]:
                stats[f'{agent_name}_mean_clip_fraction'] = np.mean(self.clip_fractions[agent_name])
            if self.explained_variances[agent_name]:
                stats[f'{agent_name}_mean_explained_variance'] = np.mean(self.explained_variances[agent_name])
        
        # Overall statistics
        all_rewards = [reward for rewards in self.episode_rewards.values() for reward in rewards]
        all_lengths = [length for lengths in self.episode_lengths.values() for length in lengths]
        
        if all_rewards:
            stats['overall_mean_reward'] = np.mean(all_rewards)
            stats['overall_std_reward'] = np.std(all_rewards)
            stats['overall_max_reward'] = np.max(all_rewards)
            stats['overall_min_reward'] = np.min(all_rewards)
        
        if all_lengths:
            stats['overall_mean_length'] = np.mean(all_lengths)
            stats['overall_std_length'] = np.std(all_lengths)
        
        return stats
    
    def save_logs(self, filename: Optional[str] = None):
        """
        Save logs to file.
        
        Args:
            filename: Optional filename (defaults to timestamp)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"marl_training_log_{timestamp}.json"
        
        filepath = os.path.join(self.log_dir, filename)
        
        log_data = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'entropy_losses': self.entropy_losses,
            'advantages': self.advantages,
            'returns': self.returns,
            'kl_divergences': self.kl_divergences,
            'clip_fractions': self.clip_fractions,
            'explained_variances': self.explained_variances,
            'stats': self.get_stats()
        }
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"Logs saved to: {filepath}")
    
    def plot_training_curves(self, save_path: Optional[str] = None, window_size: int = 100):
        """
        Plot training curves for all agents.
        
        Args:
            save_path: Path to save plot
            window_size: Window size for moving averages
        """
        num_agents = len(self.episode_rewards)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Episode rewards
        if self.episode_rewards:
            for agent_name, rewards in self.episode_rewards.items():
                if rewards:
                    axes[0, 0].plot(rewards, alpha=0.3, label=f'{agent_name} (Raw)')
                    
                    if len(rewards) > window_size:
                        moving_avg = moving_average(rewards, window_size)
                        if len(moving_avg) > 0:
                            axes[0, 0].plot(range(window_size-1, len(rewards)), moving_avg, 
                                           linewidth=2, label=f'{agent_name} (MA)')
            
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Episode lengths
        if self.episode_lengths:
            for agent_name, lengths in self.episode_lengths.items():
                if lengths:
                    axes[0, 1].plot(lengths, alpha=0.3, label=f'{agent_name} (Raw)')
                    
                    if len(lengths) > window_size:
                        moving_avg = moving_average(lengths, window_size)
                        if len(moving_avg) > 0:
                            axes[0, 1].plot(range(window_size-1, len(lengths)), moving_avg, 
                                           linewidth=2, label=f'{agent_name} (MA)')
            
            axes[0, 1].set_title('Episode Lengths')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Steps')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Policy losses
        if self.policy_losses:
            for agent_name, losses in self.policy_losses.items():
                if losses:
                    axes[0, 2].plot(losses, alpha=0.7, label=f'{agent_name}')
            
            axes[0, 2].set_title('Policy Losses')
            axes[0, 2].set_xlabel('Update')
            axes[0, 2].set_ylabel('Loss')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # KL Divergences
        if self.kl_divergences:
            for agent_name, kl_divs in self.kl_divergences.items():
                if kl_divs:
                    axes[1, 0].plot(kl_divs, alpha=0.7, label=f'{agent_name}')
            
            axes[1, 0].set_title('KL Divergences')
            axes[1, 0].set_xlabel('Update')
            axes[1, 0].set_ylabel('KL Divergence')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Clip Fractions
        if self.clip_fractions:
            for agent_name, clip_fracs in self.clip_fractions.items():
                if clip_fracs:
                    axes[1, 1].plot(clip_fracs, alpha=0.7, label=f'{agent_name}')
            
            axes[1, 1].set_title('Clip Fractions')
            axes[1, 1].set_xlabel('Update')
            axes[1, 1].set_ylabel('Fraction Clipped')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Explained Variances
        if self.explained_variances:
            for agent_name, exp_vars in self.explained_variances.items():
                if exp_vars:
                    axes[1, 2].plot(exp_vars, alpha=0.7, label=f'{agent_name}')
            
            axes[1, 2].set_title('Explained Variances')
            axes[1, 2].set_xlabel('Update')
            axes[1, 2].set_ylabel('Explained Variance')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle('Multi-Agent RL Training Progress', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def early_stopping_check(rewards: Dict[str, List[float]], target_reward: float = 475.0,
                        window_size: int = 100, patience: int = 10) -> bool:
    """
    Check if early stopping criteria are met for multi-agent training.
    
    Args:
        rewards: Dictionary mapping agent names to reward lists
        target_reward: Target reward for success
        window_size: Window size for moving average
        patience: Number of consecutive successes required
        
    Returns:
        True if early stopping criteria are met
    """
    # Check if all agents have enough episodes
    min_episodes = min(len(agent_rewards) for agent_rewards in rewards.values())
    if min_episodes < window_size + patience:
        return False
    
    # Check if all agents meet the target reward
    for agent_name, agent_rewards in rewards.items():
        recent_rewards = agent_rewards[-window_size:]
        mean_reward = np.mean(recent_rewards)
        
        if mean_reward < target_reward:
            return False
    
    return True


def compute_gradient_norm(model: torch.nn.Module) -> float:
    """
    Compute the gradient norm of a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Gradient norm
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def save_model(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
               filepath: str, additional_info: Dict = None):
    """
    Save model and optimizer state.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        filepath: Path to save model
        additional_info: Additional information to save
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, filepath)
    print(f"Model saved to: {filepath}")


def load_model(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
               filepath: str, device: str = 'cpu') -> Dict:
    """
    Load model and optimizer state.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        filepath: Path to load model from
        device: Device to load on
        
    Returns:
        Additional information from checkpoint
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    additional_info = {k: v for k, v in checkpoint.items() 
                      if k not in ['model_state_dict', 'optimizer_state_dict']}
    
    print(f"Model loaded from: {filepath}")
    return additional_info


def create_lr_schedule(initial_lr: float, final_lr: float, total_steps: int,
                      schedule_type: str = 'linear') -> callable:
    """
    Create a learning rate schedule.
    
    Args:
        initial_lr: Initial learning rate
        final_lr: Final learning rate
        total_steps: Total number of training steps
        schedule_type: Type of schedule ('linear', 'cosine', 'exponential')
        
    Returns:
        Function that takes step number and returns learning rate
    """
    if schedule_type == 'linear':
        return lambda step: initial_lr + (final_lr - initial_lr) * min(step / total_steps, 1.0)
    elif schedule_type == 'cosine':
        return lambda step: final_lr + (initial_lr - final_lr) * 0.5 * (1 + np.cos(np.pi * min(step / total_steps, 1.0)))
    elif schedule_type == 'exponential':
        return lambda step: initial_lr * (final_lr / initial_lr) ** (step / total_steps)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


if __name__ == "__main__":
    # Test utility functions
    print("Testing MARL utility functions...")
    
    # Test moving average
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ma = moving_average(data, window_size=3)
    print(f"Moving average: {ma}")
    
    # Test logger
    logger = MARLLogger()
    logger.log_episode({'agent_0': 100.0, 'agent_1': 150.0, 'agent_2': 120.0}, 
                       {'agent_0': 200, 'agent_1': 250, 'agent_2': 220})
    
    update_stats = {
        'agent_0': {'policy_loss': 0.5, 'value_loss': 0.3, 'entropy_loss': 0.1, 
                   'kl_divergence': 0.01, 'clip_fraction': 0.1, 'explained_variance': 0.8},
        'agent_1': {'policy_loss': 0.4, 'value_loss': 0.2, 'entropy_loss': 0.15, 
                   'kl_divergence': 0.02, 'clip_fraction': 0.15, 'explained_variance': 0.75},
        'agent_2': {'policy_loss': 0.6, 'value_loss': 0.4, 'entropy_loss': 0.12, 
                   'kl_divergence': 0.015, 'clip_fraction': 0.12, 'explained_variance': 0.85}
    }
    
    advantages = {'agent_0': [1.0, 2.0, 3.0], 'agent_1': [1.5, 2.5, 3.5], 'agent_2': [0.8, 1.8, 2.8]}
    returns = {'agent_0': [4.0, 5.0, 6.0], 'agent_1': [4.5, 5.5, 6.5], 'agent_2': [3.8, 4.8, 5.8]}
    
    logger.log_update(update_stats, advantages, returns)
    
    stats = logger.get_stats()
    print(f"Logger stats: {stats}")
    
    print("MARL utility functions test completed!")
