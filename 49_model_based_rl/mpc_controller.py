"""
Model Predictive Control (MPC) Controller for Model-Based RL

This module implements MPC controllers using random shooting and
Cross-Entropy Method (CEM) for action selection.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any, Callable
import random
from scipy.optimize import minimize
from dynamics_model import ProbabilisticDynamicsModel, EnsembleDynamicsModel


class RandomShootingMPC:
    """
    Random Shooting MPC Controller.
    
    This controller samples random action sequences and selects the one
    with the highest expected return according to the learned model.
    """
    
    def __init__(self, dynamics_model: ProbabilisticDynamicsModel, horizon: int = 10,
                 num_samples: int = 1000, action_bounds: Tuple[float, float] = (-1.0, 1.0),
                 device: str = 'cpu'):
        """
        Initialize Random Shooting MPC Controller.
        
        Args:
            dynamics_model: Learned dynamics model
            horizon: Planning horizon
            num_samples: Number of action sequences to sample
            action_bounds: Bounds for action sampling
            device: Device to run on
        """
        self.dynamics_model = dynamics_model
        self.horizon = horizon
        self.num_samples = num_samples
        self.action_bounds = action_bounds
        self.device = torch.device(device)
        
        # Get action dimension from model
        self.action_dim = self.dynamics_model.action_dim
        
        # Statistics
        self.total_plans = 0
        self.best_returns = []
    
    def sample_action_sequences(self) -> torch.Tensor:
        """
        Sample random action sequences.
        
        Returns:
            Action sequences [num_samples, horizon, action_dim]
        """
        # Sample actions uniformly from action bounds
        actions = torch.rand(self.num_samples, self.horizon, self.action_dim, device=self.device)
        actions = actions * (self.action_bounds[1] - self.action_bounds[0]) + self.action_bounds[0]
        
        return actions
    
    def evaluate_action_sequence(self, initial_state: torch.Tensor, 
                                action_sequence: torch.Tensor) -> float:
        """
        Evaluate an action sequence using the learned model.
        
        Args:
            initial_state: Initial state [state_dim]
            action_sequence: Action sequence [horizon, action_dim]
            
        Returns:
            Expected return
        """
        # Roll out the action sequence
        state_sequence, reward_sequence = self.dynamics_model.rollout(
            initial_state, action_sequence, deterministic=True
        )
        
        # Compute return
        return reward_sequence.sum().item()
    
    def plan(self, initial_state: torch.Tensor) -> torch.Tensor:
        """
        Plan actions using random shooting.
        
        Args:
            initial_state: Current state [state_dim]
            
        Returns:
            Best action sequence [horizon, action_dim]
        """
        # Sample action sequences
        action_sequences = self.sample_action_sequences()
        
        # Evaluate each sequence
        returns = []
        for i in range(self.num_samples):
            return_val = self.evaluate_action_sequence(initial_state, action_sequences[i])
            returns.append(return_val)
        
        # Find best sequence
        best_idx = np.argmax(returns)
        best_sequence = action_sequences[best_idx]
        best_return = returns[best_idx]
        
        # Store statistics
        self.total_plans += 1
        self.best_returns.append(best_return)
        
        return best_sequence
    
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Get the first action from the planned sequence.
        
        Args:
            state: Current state
            
        Returns:
            First action from the planned sequence
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        action_sequence = self.plan(state_tensor)
        return action_sequence[0].cpu().numpy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get controller statistics.
        
        Returns:
            Dictionary of statistics
        """
        if not self.best_returns:
            return {
                'total_plans': self.total_plans,
                'mean_return': 0.0,
                'std_return': 0.0,
                'max_return': 0.0,
                'min_return': 0.0
            }
        
        return {
            'total_plans': self.total_plans,
            'mean_return': np.mean(self.best_returns),
            'std_return': np.std(self.best_returns),
            'max_return': np.max(self.best_returns),
            'min_return': np.min(self.best_returns)
        }


class CEM_MPC:
    """
    Cross-Entropy Method (CEM) MPC Controller.
    
    This controller uses CEM to iteratively improve action sequences
    by sampling from a distribution and updating it based on the best samples.
    """
    
    def __init__(self, dynamics_model: ProbabilisticDynamicsModel, horizon: int = 10,
                 num_samples: int = 1000, num_elite: int = 100, num_iterations: int = 5,
                 action_bounds: Tuple[float, float] = (-1.0, 1.0), device: str = 'cpu'):
        """
        Initialize CEM MPC Controller.
        
        Args:
            dynamics_model: Learned dynamics model
            horizon: Planning horizon
            num_samples: Number of action sequences to sample per iteration
            num_elite: Number of elite samples to keep
            num_iterations: Number of CEM iterations
            action_bounds: Bounds for action sampling
            device: Device to run on
        """
        self.dynamics_model = dynamics_model
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elite = num_elite
        self.num_iterations = num_iterations
        self.action_bounds = action_bounds
        self.device = torch.device(device)
        
        # Get action dimension from model
        self.action_dim = self.dynamics_model.action_dim
        
        # Initialize distribution parameters
        self.action_mean = torch.zeros(horizon, self.action_dim, device=self.device)
        self.action_std = torch.ones(horizon, self.action_dim, device=self.device)
        
        # Statistics
        self.total_plans = 0
        self.best_returns = []
        self.iteration_returns = []
    
    def sample_action_sequences(self) -> torch.Tensor:
        """
        Sample action sequences from current distribution.
        
        Returns:
            Action sequences [num_samples, horizon, action_dim]
        """
        # Sample from normal distribution
        actions = torch.randn(self.num_samples, self.horizon, self.action_dim, device=self.device)
        actions = actions * self.action_std.unsqueeze(0) + self.action_mean.unsqueeze(0)
        
        # Clip to action bounds
        actions = torch.clamp(actions, self.action_bounds[0], self.action_bounds[1])
        
        return actions
    
    def evaluate_action_sequence(self, initial_state: torch.Tensor, 
                                action_sequence: torch.Tensor) -> float:
        """
        Evaluate an action sequence using the learned model.
        
        Args:
            initial_state: Initial state [state_dim]
            action_sequence: Action sequence [horizon, action_dim]
            
        Returns:
            Expected return
        """
        # Roll out the action sequence
        state_sequence, reward_sequence = self.dynamics_model.rollout(
            initial_state, action_sequence, deterministic=True
        )
        
        # Compute return
        return reward_sequence.sum().item()
    
    def update_distribution(self, action_sequences: torch.Tensor, returns: np.ndarray):
        """
        Update the action distribution based on elite samples.
        
        Args:
            action_sequences: Action sequences [num_samples, horizon, action_dim]
            returns: Returns for each sequence [num_samples]
        """
        # Get elite samples
        elite_indices = np.argsort(returns)[-self.num_elite:]
        elite_actions = action_sequences[elite_indices]
        
        # Update distribution parameters
        self.action_mean = elite_actions.mean(dim=0)
        self.action_std = elite_actions.std(dim=0)
        
        # Add small noise to prevent collapse
        self.action_std = torch.clamp(self.action_std, min=0.01)
    
    def plan(self, initial_state: torch.Tensor) -> torch.Tensor:
        """
        Plan actions using CEM.
        
        Args:
            initial_state: Current state [state_dim]
            
        Returns:
            Best action sequence [horizon, action_dim]
        """
        iteration_returns = []
        
        for iteration in range(self.num_iterations):
            # Sample action sequences
            action_sequences = self.sample_action_sequences()
            
            # Evaluate each sequence
            returns = []
            for i in range(self.num_samples):
                return_val = self.evaluate_action_sequence(initial_state, action_sequences[i])
                returns.append(return_val)
            
            returns = np.array(returns)
            iteration_returns.append(returns)
            
            # Update distribution
            self.update_distribution(action_sequences, returns)
        
        # Get best sequence from final iteration
        best_idx = np.argmax(returns)
        best_sequence = action_sequences[best_idx]
        best_return = returns[best_idx]
        
        # Store statistics
        self.total_plans += 1
        self.best_returns.append(best_return)
        self.iteration_returns.append(iteration_returns)
        
        return best_sequence
    
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Get the first action from the planned sequence.
        
        Args:
            state: Current state
            
        Returns:
            First action from the planned sequence
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        action_sequence = self.plan(state_tensor)
        return action_sequence[0].cpu().numpy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get controller statistics.
        
        Returns:
            Dictionary of statistics
        """
        if not self.best_returns:
            return {
                'total_plans': self.total_plans,
                'mean_return': 0.0,
                'std_return': 0.0,
                'max_return': 0.0,
                'min_return': 0.0
            }
        
        return {
            'total_plans': self.total_plans,
            'mean_return': np.mean(self.best_returns),
            'std_return': np.std(self.best_returns),
            'max_return': np.max(self.best_returns),
            'min_return': np.min(self.best_returns),
            'mean_iteration_returns': [np.mean(returns) for returns in self.iteration_returns[-1]] if self.iteration_returns else []
        }


class MPPIMPC:
    """
    Model Predictive Path Integral (MPPI) MPC Controller.
    
    This controller uses MPPI to sample action sequences and weight them
    according to their expected returns.
    """
    
    def __init__(self, dynamics_model: ProbabilisticDynamicsModel, horizon: int = 10,
                 num_samples: int = 1000, temperature: float = 1.0,
                 action_bounds: Tuple[float, float] = (-1.0, 1.0), device: str = 'cpu'):
        """
        Initialize MPPI MPC Controller.
        
        Args:
            dynamics_model: Learned dynamics model
            horizon: Planning horizon
            num_samples: Number of action sequences to sample
            temperature: Temperature for weighting
            action_bounds: Bounds for action sampling
            device: Device to run on
        """
        self.dynamics_model = dynamics_model
        self.horizon = horizon
        self.num_samples = num_samples
        self.temperature = temperature
        self.action_bounds = action_bounds
        self.device = torch.device(device)
        
        # Get action dimension from model
        self.action_dim = self.dynamics_model.action_dim
        
        # Initialize action sequence
        self.action_sequence = torch.zeros(horizon, self.action_dim, device=self.device)
        
        # Statistics
        self.total_plans = 0
        self.best_returns = []
    
    def sample_action_sequences(self) -> torch.Tensor:
        """
        Sample action sequences around current sequence.
        
        Returns:
            Action sequences [num_samples, horizon, action_dim]
        """
        # Sample noise
        noise = torch.randn(self.num_samples, self.horizon, self.action_dim, device=self.device)
        
        # Add noise to current sequence
        action_sequences = self.action_sequence.unsqueeze(0) + noise
        
        # Clip to action bounds
        action_sequences = torch.clamp(action_sequences, self.action_bounds[0], self.action_bounds[1])
        
        return action_sequences
    
    def evaluate_action_sequence(self, initial_state: torch.Tensor, 
                                action_sequence: torch.Tensor) -> float:
        """
        Evaluate an action sequence using the learned model.
        
        Args:
            initial_state: Initial state [state_dim]
            action_sequence: Action sequence [horizon, action_dim]
            
        Returns:
            Expected return
        """
        # Roll out the action sequence
        state_sequence, reward_sequence = self.dynamics_model.rollout(
            initial_state, action_sequence, deterministic=True
        )
        
        # Compute return
        return reward_sequence.sum().item()
    
    def plan(self, initial_state: torch.Tensor) -> torch.Tensor:
        """
        Plan actions using MPPI.
        
        Args:
            initial_state: Current state [state_dim]
            
        Returns:
            Best action sequence [horizon, action_dim]
        """
        # Sample action sequences
        action_sequences = self.sample_action_sequences()
        
        # Evaluate each sequence
        returns = []
        for i in range(self.num_samples):
            return_val = self.evaluate_action_sequence(initial_state, action_sequences[i])
            returns.append(return_val)
        
        returns = np.array(returns)
        
        # Compute weights
        weights = np.exp(returns / self.temperature)
        weights = weights / weights.sum()
        
        # Update action sequence
        self.action_sequence = torch.sum(
            action_sequences * torch.FloatTensor(weights).unsqueeze(-1).unsqueeze(-1).to(self.device),
            dim=0
        )
        
        # Store statistics
        self.total_plans += 1
        self.best_returns.append(returns.max())
        
        return self.action_sequence
    
    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Get the first action from the planned sequence.
        
        Args:
            state: Current state
            
        Returns:
            First action from the planned sequence
        """
        state_tensor = torch.FloatTensor(state).to(self.device)
        action_sequence = self.plan(state_tensor)
        return action_sequence[0].cpu().numpy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get controller statistics.
        
        Returns:
            Dictionary of statistics
        """
        if not self.best_returns:
            return {
                'total_plans': self.total_plans,
                'mean_return': 0.0,
                'std_return': 0.0,
                'max_return': 0.0,
                'min_return': 0.0
            }
        
        return {
            'total_plans': self.total_plans,
            'mean_return': np.mean(self.best_returns),
            'std_return': np.std(self.best_returns),
            'max_return': np.max(self.best_returns),
            'min_return': np.min(self.best_returns)
        }


def create_mpc_controller(controller_type: str = 'random_shooting', 
                         dynamics_model: ProbabilisticDynamicsModel = None,
                         horizon: int = 10, num_samples: int = 1000,
                         action_bounds: Tuple[float, float] = (-1.0, 1.0),
                         device: str = 'cpu', **kwargs) -> Any:
    """
    Create an MPC controller.
    
    Args:
        controller_type: Type of controller ('random_shooting', 'cem', 'mppi')
        dynamics_model: Learned dynamics model
        horizon: Planning horizon
        num_samples: Number of action sequences to sample
        action_bounds: Bounds for action sampling
        device: Device to run on
        **kwargs: Additional arguments for specific controllers
        
    Returns:
        MPC controller
    """
    if controller_type == 'random_shooting':
        return RandomShootingMPC(
            dynamics_model, horizon, num_samples, action_bounds, device
        )
    elif controller_type == 'cem':
        return CEM_MPC(
            dynamics_model, horizon, num_samples, action_bounds, device, **kwargs
        )
    elif controller_type == 'mppi':
        return MPPIMPC(
            dynamics_model, horizon, num_samples, action_bounds, device, **kwargs
        )
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")


if __name__ == "__main__":
    # Test the MPC controllers
    print("Testing MPC Controllers...")
    
    from dynamics_model import create_dynamics_model
    
    # Create dynamics model
    model = create_dynamics_model('probabilistic', state_dim=4, action_dim=1)
    
    # Test random shooting MPC
    print("\nTesting Random Shooting MPC...")
    mpc = create_mpc_controller('random_shooting', model, horizon=5, num_samples=100)
    
    state = np.random.randn(4)
    action = mpc.get_action(state)
    print(f"Action: {action}")
    print(f"Statistics: {mpc.get_statistics()}")
    
    # Test CEM MPC
    print("\nTesting CEM MPC...")
    mpc_cem = create_mpc_controller('cem', model, horizon=5, num_samples=100, num_elite=20)
    
    action = mpc_cem.get_action(state)
    print(f"Action: {action}")
    print(f"Statistics: {mpc_cem.get_statistics()}")
    
    # Test MPPI MPC
    print("\nTesting MPPI MPC...")
    mpc_mppi = create_mpc_controller('mppi', model, horizon=5, num_samples=100)
    
    action = mpc_mppi.get_action(state)
    print(f"Action: {action}")
    print(f"Statistics: {mpc_mppi.get_statistics()}")
    
    print("MPC controller tests completed!")
