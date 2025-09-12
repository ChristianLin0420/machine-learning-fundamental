"""
Utilities for Model-Based RL

This module provides utility functions for logging, plotting, and other
helper functions for model-based RL training.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Any, Tuple
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


def create_optimizer(model: torch.nn.Module, learning_rate: float = 3e-4,
                    optimizer_type: str = 'adam', weight_decay: float = 0.0) -> torch.optim.Optimizer:
    """
    Create optimizer for model.
    
    Args:
        model: PyTorch model
        learning_rate: Learning rate
        optimizer_type: Type of optimizer ('adam', 'sgd', 'rmsprop')
        weight_decay: Weight decay for regularization
        
    Returns:
        PyTorch optimizer
    """
    if optimizer_type.lower() == 'adam':
        return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


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


def clip_gradients(model: torch.nn.Module, max_norm: float = 0.5):
    """
    Clip gradients to prevent exploding gradients.
    
    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


class MBRLLogger:
    """
    Logger for Model-Based RL training progress and metrics.
    """
    
    def __init__(self, log_dir: str = "logs", save_frequency: int = 100):
        """
        Initialize MBRL Logger.
        
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
        self.model_losses = []
        self.model_state_losses = []
        self.model_reward_losses = []
        self.grad_norms = []
        self.mpc_returns = []
        self.mpc_planning_times = []
        
        # Training statistics
        self.start_time = datetime.now()
        self.total_steps = 0
        self.total_episodes = 0
        self.model_updates = 0
        
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
    
    def log_model_update(self, total_loss: float, state_loss: float, reward_loss: float, grad_norm: float):
        """
        Log model update statistics.
        
        Args:
            total_loss: Total model loss
            state_loss: State prediction loss
            reward_loss: Reward prediction loss
            grad_norm: Gradient norm
        """
        self.model_losses.append(total_loss)
        self.model_state_losses.append(state_loss)
        self.model_reward_losses.append(reward_loss)
        self.grad_norms.append(grad_norm)
        self.model_updates += 1
    
    def log_mpc_planning(self, mpc_return: float, planning_time: float):
        """
        Log MPC planning statistics.
        
        Args:
            mpc_return: Return from MPC planning
            planning_time: Time taken for planning
        """
        self.mpc_returns.append(mpc_return)
        self.mpc_planning_times.append(planning_time)
    
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
            'total_model_updates': self.model_updates,
            'total_time': (datetime.now() - self.start_time).total_seconds(),
        }
        
        # Episode statistics
        if self.episode_rewards:
            stats['mean_episode_reward'] = np.mean(self.episode_rewards)
            stats['std_episode_reward'] = np.std(self.episode_rewards)
            stats['max_episode_reward'] = np.max(self.episode_rewards)
            stats['min_episode_reward'] = np.min(self.episode_rewards)
            
            if len(self.episode_rewards) >= window_size:
                recent_rewards = self.episode_rewards[-window_size:]
                stats['recent_mean_reward'] = np.mean(recent_rewards)
                stats['recent_std_reward'] = np.std(recent_rewards)
        
        if self.episode_lengths:
            stats['mean_episode_length'] = np.mean(self.episode_lengths)
            stats['std_episode_length'] = np.std(self.episode_lengths)
        
        # Model statistics
        if self.model_losses:
            stats['mean_model_loss'] = np.mean(self.model_losses)
            stats['std_model_loss'] = np.std(self.model_losses)
            stats['mean_state_loss'] = np.mean(self.model_state_losses)
            stats['mean_reward_loss'] = np.mean(self.model_reward_losses)
            stats['mean_grad_norm'] = np.mean(self.grad_norms)
        
        # MPC statistics
        if self.mpc_returns:
            stats['mean_mpc_return'] = np.mean(self.mpc_returns)
            stats['std_mpc_return'] = np.std(self.mpc_returns)
            stats['mean_planning_time'] = np.mean(self.mpc_planning_times)
        
        return stats
    
    def save_logs(self, filename: Optional[str] = None):
        """
        Save logs to file.
        
        Args:
            filename: Optional filename (defaults to timestamp)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mbrl_training_log_{timestamp}.json"
        
        filepath = os.path.join(self.log_dir, filename)
        
        log_data = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'model_losses': self.model_losses,
            'model_state_losses': self.model_state_losses,
            'model_reward_losses': self.model_reward_losses,
            'grad_norms': self.grad_norms,
            'mpc_returns': self.mpc_returns,
            'mpc_planning_times': self.mpc_planning_times,
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
            axes[0, 0].plot(self.episode_rewards, alpha=0.3, label='Raw')
            
            if len(self.episode_rewards) >= window_size:
                moving_avg = moving_average(self.episode_rewards, window_size)
                axes[0, 0].plot(range(window_size-1, len(self.episode_rewards)), moving_avg, 
                               linewidth=2, label=f'MA({window_size})')
            
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Episode lengths
        if self.episode_lengths:
            axes[0, 1].plot(self.episode_lengths, alpha=0.3, label='Raw')
            
            if len(self.episode_lengths) >= window_size:
                moving_avg = moving_average(self.episode_lengths, window_size)
                axes[0, 1].plot(range(window_size-1, len(self.episode_lengths)), moving_avg, 
                               linewidth=2, label=f'MA({window_size})')
            
            axes[0, 1].set_title('Episode Lengths')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Steps')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Model losses
        if self.model_losses:
            axes[0, 2].plot(self.model_losses, alpha=0.7, label='Total Loss')
            axes[0, 2].plot(self.model_state_losses, alpha=0.7, label='State Loss')
            axes[0, 2].plot(self.model_reward_losses, alpha=0.7, label='Reward Loss')
            
            axes[0, 2].set_title('Model Losses')
            axes[0, 2].set_xlabel('Update')
            axes[0, 2].set_ylabel('Loss')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Gradient norms
        if self.grad_norms:
            axes[1, 0].plot(self.grad_norms, alpha=0.7)
            
            axes[1, 0].set_title('Gradient Norms')
            axes[1, 0].set_xlabel('Update')
            axes[1, 0].set_ylabel('Gradient Norm')
            axes[1, 0].grid(True, alpha=0.3)
        
        # MPC returns
        if self.mpc_returns:
            axes[1, 1].plot(self.mpc_returns, alpha=0.7)
            
            axes[1, 1].set_title('MPC Returns')
            axes[1, 1].set_xlabel('Planning Step')
            axes[1, 1].set_ylabel('Return')
            axes[1, 1].grid(True, alpha=0.3)
        
        # MPC planning times
        if self.mpc_planning_times:
            axes[1, 2].plot(self.mpc_planning_times, alpha=0.7)
            
            axes[1, 2].set_title('MPC Planning Times')
            axes[1, 2].set_xlabel('Planning Step')
            axes[1, 2].set_ylabel('Time (s)')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle('Model-Based RL Training Progress', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def plot_model_predictions(model, states: torch.Tensor, actions: torch.Tensor,
                          next_states: torch.Tensor, rewards: torch.Tensor,
                          save_path: Optional[str] = None):
    """
    Plot model predictions vs actual values.
    
    Args:
        model: Trained dynamics model
        states: Input states
        actions: Input actions
        next_states: Target next states
        rewards: Target rewards
        save_path: Path to save plot
    """
    with torch.no_grad():
        if hasattr(model, 'probabilistic') and model.probabilistic:
            outputs = model.forward(states, actions)
            pred_next_states = outputs['state_mean']
            pred_rewards = outputs['reward_mean']
        else:
            outputs = model.forward(states, actions)
            pred_next_states = outputs['next_states']
            pred_rewards = outputs['rewards']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # State predictions
    for i in range(min(4, states.shape[1])):
        axes[0, 0].scatter(next_states[:, i].cpu().numpy(), pred_next_states[:, i].cpu().numpy(), alpha=0.5)
    
    axes[0, 0].plot([next_states.min().item(), next_states.max().item()], 
                   [next_states.min().item(), next_states.max().item()], 'r--', alpha=0.8)
    axes[0, 0].set_xlabel('Actual Next States')
    axes[0, 0].set_ylabel('Predicted Next States')
    axes[0, 0].set_title('State Predictions')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Reward predictions
    axes[0, 1].scatter(rewards.cpu().numpy(), pred_rewards.cpu().numpy(), alpha=0.5)
    axes[0, 1].plot([rewards.min().item(), rewards.max().item()], 
                   [rewards.min().item(), rewards.max().item()], 'r--', alpha=0.8)
    axes[0, 1].set_xlabel('Actual Rewards')
    axes[0, 1].set_ylabel('Predicted Rewards')
    axes[0, 1].set_title('Reward Predictions')
    axes[0, 1].grid(True, alpha=0.3)
    
    # State prediction errors
    state_errors = (next_states - pred_next_states).abs().mean(dim=1)
    axes[1, 0].hist(state_errors.cpu().numpy(), bins=50, alpha=0.7)
    axes[1, 0].set_xlabel('State Prediction Error')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('State Prediction Error Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Reward prediction errors
    reward_errors = (rewards - pred_rewards).abs()
    axes[1, 1].hist(reward_errors.cpu().numpy(), bins=50, alpha=0.7)
    axes[1, 1].set_xlabel('Reward Prediction Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Reward Prediction Error Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Model Prediction Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_mpc_performance(mpc_returns: List[float], save_path: Optional[str] = None):
    """
    Plot MPC performance over time.
    
    Args:
        mpc_returns: List of MPC returns
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(mpc_returns, alpha=0.7, label='MPC Returns')
    
    if len(mpc_returns) >= 100:
        moving_avg = moving_average(mpc_returns, 100)
        plt.plot(range(99, len(mpc_returns)), moving_avg, linewidth=2, label='MA(100)')
    
    plt.xlabel('Planning Step')
    plt.ylabel('Return')
    plt.title('MPC Performance Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


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
    print("Testing MBRL utility functions...")
    
    # Test moving average
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ma = moving_average(data, window_size=3)
    print(f"Moving average: {ma}")
    
    # Test logger
    logger = MBRLLogger()
    logger.log_episode(100.0, 200)
    logger.log_model_update(0.5, 0.3, 0.2, 0.1)
    logger.log_mpc_planning(50.0, 0.1)
    
    stats = logger.get_stats()
    print(f"Logger stats: {stats}")
    
    print("MBRL utility functions test completed!")
