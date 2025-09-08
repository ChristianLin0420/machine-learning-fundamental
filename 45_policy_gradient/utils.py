import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import os
import json
from datetime import datetime


def compute_returns(rewards: List[float], gamma: float = 0.99) -> List[float]:
    """
    Compute discounted returns (reward-to-go) for REINFORCE.
    
    Args:
        rewards: List of rewards for an episode
        gamma: Discount factor
        
    Returns:
        List of discounted returns
    """
    returns = []
    running_return = 0
    
    # Compute returns backwards
    for reward in reversed(rewards):
        running_return = reward + gamma * running_return
        returns.insert(0, running_return)
    
    return returns


def compute_advantages(returns: List[float], values: List[float]) -> List[float]:
    """
    Compute advantages as returns minus baseline values.
    
    Args:
        returns: List of returns
        values: List of value estimates
        
    Returns:
        List of advantages
    """
    return [ret - val for ret, val in zip(returns, values)]


def compute_gae_advantages(rewards: List[float], values: List[float], 
                          gamma: float = 0.99, lam: float = 0.95) -> List[float]:
    """
    Compute Generalized Advantage Estimation (GAE) advantages.
    
    Args:
        rewards: List of rewards
        values: List of value estimates
        gamma: Discount factor
        lam: GAE parameter
        
    Returns:
        List of GAE advantages
    """
    advantages = []
    running_advantage = 0
    
    # Add terminal value (0) to values
    values_extended = values + [0]
    
    # Compute advantages backwards
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values_extended[t + 1] - values[t]
        running_advantage = delta + gamma * lam * running_advantage
        advantages.insert(0, running_advantage)
    
    return advantages


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


class TrainingLogger:
    """
    Logger for tracking training progress and metrics.
    """
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize training logger.
        
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.advantages = []
        self.returns = []
        
        self.start_time = datetime.now()
    
    def log_episode(self, reward: float, length: int, 
                   policy_loss: Optional[float] = None,
                   value_loss: Optional[float] = None,
                   entropy_loss: Optional[float] = None,
                   advantages: Optional[List[float]] = None,
                   returns: Optional[List[float]] = None):
        """
        Log episode statistics.
        
        Args:
            reward: Episode reward
            length: Episode length
            policy_loss: Policy loss
            value_loss: Value loss
            entropy_loss: Entropy loss
            advantages: List of advantages
            returns: List of returns
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        
        if policy_loss is not None:
            self.policy_losses.append(policy_loss)
        if value_loss is not None:
            self.value_losses.append(value_loss)
        if entropy_loss is not None:
            self.entropy_losses.append(entropy_loss)
        if advantages is not None:
            self.advantages.extend(advantages)
        if returns is not None:
            self.returns.extend(returns)
    
    def get_stats(self, window_size: int = 100) -> dict:
        """
        Get training statistics.
        
        Args:
            window_size: Window size for moving averages
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            'total_episodes': len(self.episode_rewards),
            'total_time': (datetime.now() - self.start_time).total_seconds(),
            'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'std_reward': np.std(self.episode_rewards) if self.episode_rewards else 0,
            'mean_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'std_length': np.std(self.episode_lengths) if self.episode_lengths else 0,
        }
        
        if len(self.episode_rewards) >= window_size:
            recent_rewards = self.episode_rewards[-window_size:]
            stats.update({
                'recent_mean_reward': np.mean(recent_rewards),
                'recent_std_reward': np.std(recent_rewards),
                'recent_max_reward': np.max(recent_rewards),
                'recent_min_reward': np.min(recent_rewards)
            })
        
        if self.policy_losses:
            stats['mean_policy_loss'] = np.mean(self.policy_losses)
        if self.value_losses:
            stats['mean_value_loss'] = np.mean(self.value_losses)
        if self.entropy_losses:
            stats['mean_entropy_loss'] = np.mean(self.entropy_losses)
        
        return stats
    
    def save_logs(self, filename: Optional[str] = None):
        """
        Save logs to file.
        
        Args:
            filename: Optional filename (defaults to timestamp)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_log_{timestamp}.json"
        
        filepath = os.path.join(self.log_dir, filename)
        
        log_data = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'entropy_losses': self.entropy_losses,
            'advantages': self.advantages,
            'returns': self.returns,
            'stats': self.get_stats()
        }
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"Logs saved to: {filepath}")
    
    def plot_training_curves(self, save_path: Optional[str] = None, window_size: int = 100):
        """
        Plot training curves.
        
        Args:
            save_path: Path to save plot
            window_size: Window size for moving averages
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        if self.episode_rewards:
            axes[0, 0].plot(self.episode_rewards, alpha=0.3, color='blue', label='Raw Rewards')
            
            if len(self.episode_rewards) >= window_size:
                moving_avg = np.convolve(self.episode_rewards, np.ones(window_size)/window_size, mode='valid')
                axes[0, 0].plot(range(window_size-1, len(self.episode_rewards)), 
                               moving_avg, color='red', linewidth=2, label=f'Moving Avg ({window_size})')
            
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Episode lengths
        if self.episode_lengths:
            axes[0, 1].plot(self.episode_lengths, alpha=0.3, color='green', label='Raw Lengths')
            
            if len(self.episode_lengths) >= window_size:
                moving_avg = np.convolve(self.episode_lengths, np.ones(window_size)/window_size, mode='valid')
                axes[0, 1].plot(range(window_size-1, len(self.episode_lengths)), 
                               moving_avg, color='red', linewidth=2, label=f'Moving Avg ({window_size})')
            
            axes[0, 1].set_title('Episode Lengths')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Steps')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Losses
        if self.policy_losses:
            axes[1, 0].plot(self.policy_losses, alpha=0.7, color='orange', label='Policy Loss')
        if self.value_losses:
            axes[1, 0].plot(self.value_losses, alpha=0.7, color='purple', label='Value Loss')
        if self.entropy_losses:
            axes[1, 0].plot(self.entropy_losses, alpha=0.7, color='brown', label='Entropy Loss')
        
        axes[1, 0].set_title('Training Losses')
        axes[1, 0].set_xlabel('Update Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Advantages and Returns
        if self.advantages:
            axes[1, 1].hist(self.advantages, bins=50, alpha=0.7, color='cyan', label='Advantages')
        if self.returns:
            axes[1, 1].hist(self.returns, bins=50, alpha=0.7, color='magenta', label='Returns')
        
        axes[1, 1].set_title('Advantages and Returns Distribution')
        axes[1, 1].set_xlabel('Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('REINFORCE Training Progress', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


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
                    optimizer_type: str = 'adam') -> torch.optim.Optimizer:
    """
    Create optimizer for network.
    
    Args:
        network: PyTorch network
        learning_rate: Learning rate
        optimizer_type: Type of optimizer ('adam', 'sgd', 'rmsprop')
        
    Returns:
        PyTorch optimizer
    """
    if optimizer_type.lower() == 'adam':
        return torch.optim.Adam(network.parameters(), lr=learning_rate)
    elif optimizer_type.lower() == 'sgd':
        return torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_type.lower() == 'rmsprop':
        return torch.optim.RMSprop(network.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def compute_kl_divergence(old_log_probs: torch.Tensor, new_log_probs: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence between old and new policy distributions.
    
    Args:
        old_log_probs: Log probabilities from old policy
        new_log_probs: Log probabilities from new policy
        
    Returns:
        KL divergence
    """
    return (old_log_probs - new_log_probs).mean()


def compute_policy_entropy(log_probs: torch.Tensor) -> torch.Tensor:
    """
    Compute policy entropy.
    
    Args:
        log_probs: Log probabilities
        
    Returns:
        Entropy
    """
    return -log_probs.mean()


def early_stopping_check(rewards: List[float], target_reward: float = 475.0, 
                        window_size: int = 100, patience: int = 10) -> bool:
    """
    Check if early stopping criteria are met.
    
    Args:
        rewards: List of episode rewards
        target_reward: Target reward for success
        window_size: Window size for moving average
        patience: Number of consecutive successes required
        
    Returns:
        True if early stopping criteria are met
    """
    if len(rewards) < window_size + patience:
        return False
    
    recent_rewards = rewards[-window_size:]
    mean_reward = np.mean(recent_rewards)
    
    if mean_reward >= target_reward:
        # Check if we've maintained this performance for 'patience' episodes
        for i in range(patience):
            if len(rewards) < window_size + patience - i:
                return False
            recent_window = rewards[-(window_size + patience - i):-patience + i] if patience - i > 0 else rewards[-(window_size + patience - i):]
            if np.mean(recent_window) < target_reward:
                return False
        return True
    
    return False


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test returns computation
    rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
    returns = compute_returns(rewards, gamma=0.9)
    print(f"Returns: {returns}")
    
    # Test advantages computation
    values = [0.5, 1.0, 1.5, 2.0, 2.5]
    advantages = compute_advantages(returns, values)
    print(f"Advantages: {advantages}")
    
    # Test GAE advantages
    gae_advantages = compute_gae_advantages(rewards, values, gamma=0.9, lam=0.95)
    print(f"GAE Advantages: {gae_advantages}")
    
    # Test normalization
    normalized_advantages = normalize_advantages(advantages)
    print(f"Normalized Advantages: {normalized_advantages}")
    
    # Test logger
    logger = TrainingLogger()
    logger.log_episode(reward=100.0, length=200, policy_loss=0.5, value_loss=0.3)
    stats = logger.get_stats()
    print(f"Logger stats: {stats}")
    
    print("Utility functions test completed!")
