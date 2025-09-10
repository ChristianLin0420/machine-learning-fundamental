import torch
import numpy as np
from typing import List, Tuple, Optional
import warnings


def compute_gae_advantages(rewards: List[float], values: List[float], 
                          next_values: List[float], dones: List[bool],
                          gamma: float = 0.99, lam: float = 0.95) -> Tuple[List[float], List[float]]:
    """
    Compute Generalized Advantage Estimation (GAE) advantages and returns.
    
    GAE provides a bias-variance trade-off in advantage estimation by using
    a parameter λ to control the amount of bootstrapping vs. Monte Carlo.
    
    Args:
        rewards: List of rewards for each timestep
        values: List of value estimates for each timestep
        next_values: List of value estimates for next timesteps
        dones: List of done flags for each timestep
        gamma: Discount factor
        lam: GAE parameter (0 = high bias, low variance; 1 = low bias, high variance)
        
    Returns:
        Tuple of (advantages, returns)
    """
    if len(rewards) != len(values) or len(rewards) != len(next_values) or len(rewards) != len(dones):
        raise ValueError("All input lists must have the same length")
    
    if len(rewards) == 0:
        return [], []
    
    advantages = []
    returns = []
    
    # Compute TD errors
    td_errors = []
    for t in range(len(rewards)):
        if dones[t]:
            # Terminal state: no bootstrap
            td_error = rewards[t] - values[t]
        else:
            # Non-terminal state: bootstrap with next value
            td_error = rewards[t] + gamma * next_values[t] - values[t]
        td_errors.append(td_error)
    
    # Compute GAE advantages (backwards)
    gae_advantage = 0.0
    for t in reversed(range(len(rewards))):
        if dones[t]:
            # Terminal state: reset GAE
            gae_advantage = td_errors[t]
        else:
            # Non-terminal state: accumulate GAE
            gae_advantage = td_errors[t] + gamma * lam * gae_advantage
        
        advantages.insert(0, gae_advantage)
    
    # Compute returns (advantages + values)
    for t in range(len(advantages)):
        return_t = advantages[t] + values[t]
        returns.append(return_t)
    
    return advantages, returns


def compute_gae_advantages_tensor(rewards: torch.Tensor, values: torch.Tensor,
                                 next_values: torch.Tensor, dones: torch.Tensor,
                                 gamma: float = 0.99, lam: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GAE advantages and returns using PyTorch tensors.
    
    Args:
        rewards: Reward tensor [T]
        values: Value tensor [T]
        next_values: Next value tensor [T]
        dones: Done tensor [T]
        gamma: Discount factor
        lam: GAE parameter
        
    Returns:
        Tuple of (advantages, returns)
    """
    # Ensure all tensors have the same shape
    if not (rewards.shape == values.shape == next_values.shape == dones.shape):
        raise ValueError("All input tensors must have the same shape")
    
    # Compute TD errors
    td_errors = rewards + gamma * next_values * (1 - dones.float()) - values
    
    # Compute GAE advantages (backwards)
    advantages = torch.zeros_like(rewards)
    gae_advantage = 0.0
    
    for t in reversed(range(len(rewards))):
        if dones[t]:
            gae_advantage = td_errors[t]
        else:
            gae_advantage = td_errors[t] + gamma * lam * gae_advantage
        
        advantages[t] = gae_advantage
    
    # Compute returns
    returns = advantages + values
    
    return advantages, returns


def compute_gae_advantages_batch(rewards: torch.Tensor, values: torch.Tensor,
                                next_values: torch.Tensor, dones: torch.Tensor,
                                gamma: float = 0.99, lam: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GAE advantages and returns for a batch of trajectories.
    
    Args:
        rewards: Reward tensor [B, T] where B is batch size, T is trajectory length
        values: Value tensor [B, T]
        next_values: Next value tensor [B, T]
        dones: Done tensor [B, T]
        gamma: Discount factor
        lam: GAE parameter
        
    Returns:
        Tuple of (advantages, returns)
    """
    batch_size, traj_length = rewards.shape
    
    # Compute TD errors
    td_errors = rewards + gamma * next_values * (1 - dones.float()) - values
    
    # Compute GAE advantages (backwards for each trajectory)
    advantages = torch.zeros_like(rewards)
    
    for b in range(batch_size):
        gae_advantage = 0.0
        for t in reversed(range(traj_length)):
            if dones[b, t]:
                gae_advantage = td_errors[b, t]
            else:
                gae_advantage = td_errors[b, t] + gamma * lam * gae_advantage
            
            advantages[b, t] = gae_advantage
    
    # Compute returns
    returns = advantages + values
    
    return advantages, returns


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


def compute_gae_advantages_episode(rewards: List[float], values: List[float],
                                  gamma: float = 0.99, lam: float = 0.95) -> Tuple[List[float], List[float]]:
    """
    Compute GAE advantages for a single episode.
    
    Args:
        rewards: List of rewards for the episode
        values: List of value estimates for the episode
        gamma: Discount factor
        lam: GAE parameter
        
    Returns:
        Tuple of (advantages, returns)
    """
    if len(rewards) != len(values):
        raise ValueError("Rewards and values must have the same length")
    
    if len(rewards) == 0:
        return [], []
    
    # Add terminal value (0) for the last timestep
    next_values = values[1:] + [0.0]
    dones = [False] * (len(rewards) - 1) + [True]
    
    return compute_gae_advantages(rewards, values, next_values, dones, gamma, lam)


def compute_gae_advantages_trajectory(rewards: torch.Tensor, values: torch.Tensor,
                                     gamma: float = 0.99, lam: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute GAE advantages for a single trajectory.
    
    Args:
        rewards: Reward tensor [T]
        values: Value tensor [T]
        gamma: Discount factor
        lam: GAE parameter
        
    Returns:
        Tuple of (advantages, returns)
    """
    if rewards.shape != values.shape:
        raise ValueError("Rewards and values must have the same shape")
    
    if len(rewards) == 0:
        return torch.tensor([]), torch.tensor([])
    
    # Add terminal value (0) for the last timestep
    next_values = torch.cat([values[1:], torch.zeros(1, device=values.device)])
    dones = torch.cat([torch.zeros(len(rewards) - 1, dtype=torch.bool, device=rewards.device),
                      torch.ones(1, dtype=torch.bool, device=rewards.device)])
    
    return compute_gae_advantages_tensor(rewards, values, next_values, dones, gamma, lam)


def compute_gae_advantages_parallel(rewards: List[List[float]], values: List[List[float]],
                                   gamma: float = 0.99, lam: float = 0.95) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Compute GAE advantages for multiple episodes in parallel.
    
    Args:
        rewards: List of reward lists for each episode
        values: List of value lists for each episode
        gamma: Discount factor
        lam: GAE parameter
        
    Returns:
        Tuple of (advantages_list, returns_list)
    """
    if len(rewards) != len(values):
        raise ValueError("Rewards and values must have the same number of episodes")
    
    advantages_list = []
    returns_list = []
    
    for ep_rewards, ep_values in zip(rewards, values):
        ep_advantages, ep_returns = compute_gae_advantages_episode(
            ep_rewards, ep_values, gamma, lam
        )
        advantages_list.append(ep_advantages)
        returns_list.append(ep_returns)
    
    return advantages_list, returns_list


def compute_gae_advantages_vectorized(rewards: torch.Tensor, values: torch.Tensor,
                                     gamma: float = 0.99, lam: float = 0.95) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized computation of GAE advantages for multiple trajectories.
    
    Args:
        rewards: Reward tensor [B, T] where B is batch size, T is max trajectory length
        values: Value tensor [B, T]
        gamma: Discount factor
        lam: GAE parameter
        
    Returns:
        Tuple of (advantages, returns)
    """
    batch_size, traj_length = rewards.shape
    
    # Pad with zeros for next values
    next_values = torch.cat([values[:, 1:], torch.zeros(batch_size, 1, device=values.device)], dim=1)
    
    # Compute TD errors
    td_errors = rewards + gamma * next_values - values
    
    # Compute GAE advantages using vectorized operations
    advantages = torch.zeros_like(rewards)
    
    # Process each trajectory
    for b in range(batch_size):
        gae_advantage = 0.0
        for t in reversed(range(traj_length)):
            gae_advantage = td_errors[b, t] + gamma * lam * gae_advantage
            advantages[b, t] = gae_advantage
    
    # Compute returns
    returns = advantages + values
    
    return advantages, returns


def get_gae_lambda_schedule(initial_lambda: float = 0.95, final_lambda: float = 0.95,
                           total_steps: int = 1000000, schedule_type: str = 'constant') -> callable:
    """
    Get a lambda schedule for GAE.
    
    Args:
        initial_lambda: Initial lambda value
        final_lambda: Final lambda value
        total_steps: Total number of training steps
        schedule_type: Type of schedule ('constant', 'linear', 'cosine')
        
    Returns:
        Function that takes step number and returns lambda value
    """
    if schedule_type == 'constant':
        return lambda step: initial_lambda
    elif schedule_type == 'linear':
        return lambda step: initial_lambda + (final_lambda - initial_lambda) * min(step / total_steps, 1.0)
    elif schedule_type == 'cosine':
        return lambda step: final_lambda + (initial_lambda - final_lambda) * 0.5 * (1 + np.cos(np.pi * min(step / total_steps, 1.0)))
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def compute_gae_advantages_with_schedule(rewards: List[float], values: List[float],
                                        next_values: List[float], dones: List[bool],
                                        gamma: float = 0.99, lam_schedule: callable = None,
                                        step: int = 0) -> Tuple[List[float], List[float]]:
    """
    Compute GAE advantages with a lambda schedule.
    
    Args:
        rewards: List of rewards for each timestep
        values: List of value estimates for each timestep
        next_values: List of value estimates for next timesteps
        dones: List of done flags for each timestep
        gamma: Discount factor
        lam_schedule: Lambda schedule function
        step: Current training step
        
    Returns:
        Tuple of (advantages, returns)
    """
    if lam_schedule is None:
        lam = 0.95
    else:
        lam = lam_schedule(step)
    
    return compute_gae_advantages(rewards, values, next_values, dones, gamma, lam)


if __name__ == "__main__":
    # Test GAE computation
    print("Testing GAE computation...")
    
    # Test with simple data
    rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
    values = [0.5, 1.0, 1.5, 2.0, 2.5]
    next_values = [1.0, 1.5, 2.0, 2.5, 0.0]
    dones = [False, False, False, False, True]
    
    advantages, returns = compute_gae_advantages(rewards, values, next_values, dones)
    
    print(f"Rewards: {rewards}")
    print(f"Values: {values}")
    print(f"Advantages: {advantages}")
    print(f"Returns: {returns}")
    
    # Test normalization
    normalized_advantages = normalize_advantages(advantages)
    print(f"Normalized advantages: {normalized_advantages}")
    
    # Test tensor version
    rewards_tensor = torch.tensor(rewards)
    values_tensor = torch.tensor(values)
    next_values_tensor = torch.tensor(next_values)
    dones_tensor = torch.tensor(dones)
    
    advantages_tensor, returns_tensor = compute_gae_advantages_tensor(
        rewards_tensor, values_tensor, next_values_tensor, dones_tensor
    )
    
    print(f"Tensor advantages: {advantages_tensor}")
    print(f"Tensor returns: {returns_tensor}")
    
    print("GAE tests completed!")
