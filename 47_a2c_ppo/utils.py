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


class TrainingLogger:
    """
    Logger for tracking training progress and metrics.
    """
    
    def __init__(self, log_dir: str = "logs", save_frequency: int = 100):
        """
        Initialize training logger.
        
        Args:
            log_dir: Directory to save logs
            save_frequency: Frequency to save logs
        """
        self.log_dir = log_dir
        self.save_frequency = save_frequency
        os.makedirs(log_dir, exist_ok=True)
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.advantages = []
        self.returns = []
        self.grad_norms = []
        
        # Training statistics
        self.start_time = datetime.now()
        self.total_steps = 0
        self.total_episodes = 0
        self.update_count = 0
        
    def log_episode(self, reward: float, length: int):
        """
        Log episode statistics.
        
        Args:
            reward: Episode reward
            length: Episode length
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.total_episodes += 1
    
    def log_update(self, policy_loss: float, value_loss: float, entropy_loss: float,
                   advantages: List[float], returns: List[float], grad_norm: float = None):
        """
        Log update statistics.
        
        Args:
            policy_loss: Policy loss
            value_loss: Value loss
            entropy_loss: Entropy loss
            advantages: List of advantages
            returns: List of returns
            grad_norm: Gradient norm
        """
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.entropy_losses.append(entropy_loss)
        self.advantages.extend(advantages)
        self.returns.extend(returns)
        
        if grad_norm is not None:
            self.grad_norms.append(grad_norm)
        
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
        
        if self.episode_rewards:
            stats.update({
                'mean_reward': np.mean(self.episode_rewards),
                'std_reward': np.std(self.episode_rewards),
                'max_reward': np.max(self.episode_rewards),
                'min_reward': np.min(self.episode_rewards),
            })
            
            if len(self.episode_rewards) >= window_size:
                recent_rewards = self.episode_rewards[-window_size:]
                stats.update({
                    'recent_mean_reward': np.mean(recent_rewards),
                    'recent_std_reward': np.std(recent_rewards),
                    'recent_max_reward': np.max(recent_rewards),
                    'recent_min_reward': np.min(recent_rewards),
                })
        
        if self.episode_lengths:
            stats.update({
                'mean_length': np.mean(self.episode_lengths),
                'std_length': np.std(self.episode_lengths),
            })
        
        if self.policy_losses:
            stats['mean_policy_loss'] = np.mean(self.policy_losses)
        if self.value_losses:
            stats['mean_value_loss'] = np.mean(self.value_losses)
        if self.entropy_losses:
            stats['mean_entropy_loss'] = np.mean(self.entropy_losses)
        if self.grad_norms:
            stats['mean_grad_norm'] = np.mean(self.grad_norms)
        
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
            'grad_norms': self.grad_norms,
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
        if self.policy_losses:
            axes[0, 2].plot(self.policy_losses, alpha=0.7, color='orange', label='Policy Loss')
        if self.value_losses:
            axes[0, 2].plot(self.value_losses, alpha=0.7, color='purple', label='Value Loss')
        if self.entropy_losses:
            axes[0, 2].plot(self.entropy_losses, alpha=0.7, color='brown', label='Entropy Loss')
        
        axes[0, 2].set_title('Training Losses')
        axes[0, 2].set_xlabel('Update')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Advantages distribution
        if self.advantages:
            axes[1, 0].hist(self.advantages, bins=50, alpha=0.7, color='cyan', label='Advantages')
            axes[1, 0].set_title('Advantages Distribution')
            axes[1, 0].set_xlabel('Advantage Value')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Returns distribution
        if self.returns:
            axes[1, 1].hist(self.returns, bins=50, alpha=0.7, color='magenta', label='Returns')
            axes[1, 1].set_title('Returns Distribution')
            axes[1, 1].set_xlabel('Return Value')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Gradient norms
        if self.grad_norms:
            axes[1, 2].plot(self.grad_norms, alpha=0.7, color='red', label='Grad Norm')
            axes[1, 2].set_title('Gradient Norms')
            axes[1, 2].set_xlabel('Update')
            axes[1, 2].set_ylabel('Gradient Norm')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle('A2C Training Progress', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


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
    print("Testing utility functions...")
    
    # Test moving average
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ma = moving_average(data, window_size=3)
    print(f"Moving average: {ma}")
    
    # Test logger
    logger = TrainingLogger()
    logger.log_episode(reward=100.0, length=200)
    logger.log_update(policy_loss=0.5, value_loss=0.3, entropy_loss=0.1, 
                     advantages=[1.0, 2.0, 3.0], returns=[4.0, 5.0, 6.0])
    
    stats = logger.get_stats()
    print(f"Logger stats: {stats}")
    
    print("Utility functions test completed!")
