"""
Probabilistic Dynamics Model for Model-Based RL

This module implements a neural network-based dynamics model that learns
to predict next states and rewards from current states and actions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
import math


class ProbabilisticDynamicsModel(nn.Module):
    """
    Probabilistic Dynamics Model for Model-Based RL.
    
    This model learns to predict next states and rewards from current
    states and actions. It can be either deterministic or probabilistic.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: list = [256, 256],
                 activation: str = 'relu', probabilistic: bool = True, 
                 use_orthogonal_init: bool = True):
        """
        Initialize Probabilistic Dynamics Model.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_sizes: List of hidden layer sizes
            activation: Activation function ('relu', 'tanh', 'elu')
            probabilistic: Whether to use probabilistic predictions
            use_orthogonal_init: Whether to use orthogonal initialization
        """
        super(ProbabilisticDynamicsModel, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes
        self.probabilistic = probabilistic
        self.use_orthogonal_init = use_orthogonal_init
        
        # Activation function
        if activation.lower() == 'relu':
            self.activation = nn.ReLU
        elif activation.lower() == 'tanh':
            self.activation = nn.Tanh
        elif activation.lower() == 'elu':
            self.activation = nn.ELU
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Input layer (state + action)
        input_dim = state_dim + action_dim
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        prev_size = input_dim
        
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        # Output layers
        if probabilistic:
            # Probabilistic: predict mean and log_std for next state and reward
            self.state_mean = nn.Linear(prev_size, state_dim)
            self.state_log_std = nn.Linear(prev_size, state_dim)
            self.reward_mean = nn.Linear(prev_size, 1)
            self.reward_log_std = nn.Linear(prev_size, 1)
        else:
            # Deterministic: predict next state and reward directly
            self.state_output = nn.Linear(prev_size, state_dim)
            self.reward_output = nn.Linear(prev_size, 1)
        
        # Initialize weights
        if use_orthogonal_init:
            self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using orthogonal initialization."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the dynamics model.
        
        Args:
            states: Current states [batch_size, state_dim]
            actions: Current actions [batch_size, action_dim]
            
        Returns:
            Dictionary containing predictions
        """
        # Concatenate state and action
        x = torch.cat([states, actions], dim=-1)
        
        # Forward through hidden layers
        for layer in self.hidden_layers:
            x = self.activation()(layer(x))
        
        if self.probabilistic:
            # Probabilistic predictions
            state_mean = self.state_mean(x)
            state_log_std = self.state_log_std(x)
            reward_mean = self.reward_mean(x)
            reward_log_std = self.reward_log_std(x)
            
            # Clamp log_std for numerical stability
            state_log_std = torch.clamp(state_log_std, min=-10, max=2)
            reward_log_std = torch.clamp(reward_log_std, min=-10, max=2)
            
            return {
                'state_mean': state_mean,
                'state_log_std': state_log_std,
                'reward_mean': reward_mean,
                'reward_log_std': reward_log_std
            }
        else:
            # Deterministic predictions
            next_states = self.state_output(x)
            rewards = self.reward_output(x)
            
            return {
                'next_states': next_states,
                'rewards': rewards
            }
    
    def predict(self, states: torch.Tensor, actions: torch.Tensor, 
                deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next states and rewards.
        
        Args:
            states: Current states
            actions: Current actions
            deterministic: Whether to use deterministic predictions
            
        Returns:
            Tuple of (next_states, rewards)
        """
        with torch.no_grad():
            if self.probabilistic and not deterministic:
                # Sample from probabilistic model
                outputs = self.forward(states, actions)
                state_std = torch.exp(outputs['state_log_std'])
                reward_std = torch.exp(outputs['reward_log_std'])
                
                next_states = outputs['state_mean'] + state_std * torch.randn_like(state_std)
                rewards = outputs['reward_mean'] + reward_std * torch.randn_like(reward_std)
            else:
                # Use mean predictions (deterministic)
                outputs = self.forward(states, actions)
                if self.probabilistic:
                    next_states = outputs['state_mean']
                    rewards = outputs['reward_mean']
                else:
                    next_states = outputs['next_states']
                    rewards = outputs['rewards']
        
        return next_states, rewards
    
    def compute_loss(self, states: torch.Tensor, actions: torch.Tensor,
                     next_states: torch.Tensor, rewards: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute loss for training the dynamics model.
        
        Args:
            states: Current states
            actions: Current actions
            next_states: Next states (targets)
            rewards: Rewards (targets)
            
        Returns:
            Dictionary containing loss components
        """
        if self.probabilistic:
            # Probabilistic loss (negative log-likelihood)
            outputs = self.forward(states, actions)
            
            # State prediction loss
            state_mean = outputs['state_mean']
            state_log_std = outputs['state_log_std']
            state_std = torch.exp(state_log_std)
            
            state_loss = 0.5 * torch.sum(
                ((next_states - state_mean) / state_std) ** 2 + 2 * state_log_std,
                dim=-1
            ).mean()
            
            # Reward prediction loss
            reward_mean = outputs['reward_mean']
            reward_log_std = outputs['reward_log_std']
            reward_std = torch.exp(reward_log_std)
            
            reward_loss = 0.5 * torch.sum(
                ((rewards - reward_mean) / reward_std) ** 2 + 2 * reward_log_std,
                dim=-1
            ).mean()
            
            total_loss = state_loss + reward_loss
            
            return {
                'total_loss': total_loss,
                'state_loss': state_loss,
                'reward_loss': reward_loss
            }
        else:
            # Deterministic loss (MSE)
            outputs = self.forward(states, actions)
            
            state_loss = F.mse_loss(outputs['next_states'], next_states)
            reward_loss = F.mse_loss(outputs['rewards'], rewards)
            
            total_loss = state_loss + reward_loss
            
            return {
                'total_loss': total_loss,
                'state_loss': state_loss,
                'reward_loss': reward_loss
            }
    
    def rollout(self, initial_state: torch.Tensor, action_sequence: torch.Tensor,
                deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Roll out a sequence of actions in the learned model.
        
        Args:
            initial_state: Initial state [state_dim]
            action_sequence: Sequence of actions [horizon, action_dim]
            deterministic: Whether to use deterministic predictions
            
        Returns:
            Tuple of (state_sequence, reward_sequence)
        """
        horizon = action_sequence.shape[0]
        state_dim = initial_state.shape[0]
        
        # Initialize sequences
        state_sequence = torch.zeros(horizon + 1, state_dim)
        reward_sequence = torch.zeros(horizon)
        
        state_sequence[0] = initial_state
        
        # Roll out actions
        for t in range(horizon):
            current_state = state_sequence[t].unsqueeze(0)
            current_action = action_sequence[t].unsqueeze(0)
            
            next_state, reward = self.predict(current_state, current_action, deterministic)
            
            state_sequence[t + 1] = next_state.squeeze(0)
            reward_sequence[t] = reward.squeeze(0)
        
        return state_sequence, reward_sequence


class EnsembleDynamicsModel(nn.Module):
    """
    Ensemble of Dynamics Models for better uncertainty estimation.
    
    This model uses multiple dynamics models to provide better uncertainty
    estimates and more robust predictions.
    """
    
    def __init__(self, state_dim: int, action_dim: int, num_models: int = 5,
                 hidden_sizes: list = [256, 256], activation: str = 'relu',
                 probabilistic: bool = True, use_orthogonal_init: bool = True):
        """
        Initialize Ensemble Dynamics Model.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            num_models: Number of models in the ensemble
            hidden_sizes: List of hidden layer sizes
            activation: Activation function
            probabilistic: Whether to use probabilistic predictions
            use_orthogonal_init: Whether to use orthogonal initialization
        """
        super(EnsembleDynamicsModel, self).__init__()
        
        self.num_models = num_models
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Create ensemble of models
        self.models = nn.ModuleList([
            ProbabilisticDynamicsModel(
                state_dim, action_dim, hidden_sizes, activation,
                probabilistic, use_orthogonal_init
            ) for _ in range(num_models)
        ])
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all models in the ensemble.
        
        Args:
            states: Current states [batch_size, state_dim]
            actions: Current actions [batch_size, action_dim]
            
        Returns:
            Dictionary containing ensemble predictions
        """
        outputs = []
        for model in self.models:
            outputs.append(model.forward(states, actions))
        
        # Average predictions across ensemble
        if self.models[0].probabilistic:
            state_means = torch.stack([out['state_mean'] for out in outputs])
            state_log_stds = torch.stack([out['state_log_std'] for out in outputs])
            reward_means = torch.stack([out['reward_mean'] for out in outputs])
            reward_log_stds = torch.stack([out['reward_log_std'] for out in outputs])
            
            return {
                'state_mean': state_means.mean(dim=0),
                'state_log_std': state_log_stds.mean(dim=0),
                'reward_mean': reward_means.mean(dim=0),
                'reward_log_std': reward_log_stds.mean(dim=0),
                'state_std': torch.exp(state_log_stds).mean(dim=0),
                'reward_std': torch.exp(reward_log_stds).mean(dim=0)
            }
        else:
            next_states = torch.stack([out['next_states'] for out in outputs])
            rewards = torch.stack([out['rewards'] for out in outputs])
            
            return {
                'next_states': next_states.mean(dim=0),
                'rewards': rewards.mean(dim=0),
                'next_states_std': next_states.std(dim=0),
                'rewards_std': rewards.std(dim=0)
            }
    
    def predict(self, states: torch.Tensor, actions: torch.Tensor,
                deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next states and rewards using ensemble.
        
        Args:
            states: Current states
            actions: Current actions
            deterministic: Whether to use deterministic predictions
            
        Returns:
            Tuple of (next_states, rewards)
        """
        with torch.no_grad():
            if self.models[0].probabilistic and not deterministic:
                # Sample from each model and average
                next_states_list = []
                rewards_list = []
                
                for model in self.models:
                    next_state, reward = model.predict(states, actions, deterministic=False)
                    next_states_list.append(next_state)
                    rewards_list.append(reward)
                
                next_states = torch.stack(next_states_list).mean(dim=0)
                rewards = torch.stack(rewards_list).mean(dim=0)
            else:
                # Use mean predictions
                outputs = self.forward(states, actions)
                if self.models[0].probabilistic:
                    next_states = outputs['state_mean']
                    rewards = outputs['reward_mean']
                else:
                    next_states = outputs['next_states']
                    rewards = outputs['rewards']
        
        return next_states, rewards
    
    def compute_loss(self, states: torch.Tensor, actions: torch.Tensor,
                     next_states: torch.Tensor, rewards: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute loss for training the ensemble.
        
        Args:
            states: Current states
            actions: Current actions
            next_states: Next states (targets)
            rewards: Rewards (targets)
            
        Returns:
            Dictionary containing loss components
        """
        total_loss = 0.0
        state_loss = 0.0
        reward_loss = 0.0
        
        # Compute loss for each model
        for model in self.models:
            model_losses = model.compute_loss(states, actions, next_states, rewards)
            total_loss += model_losses['total_loss']
            state_loss += model_losses['state_loss']
            reward_loss += model_losses['reward_loss']
        
        # Average losses across ensemble
        total_loss /= self.num_models
        state_loss /= self.num_models
        reward_loss /= self.num_models
        
        return {
            'total_loss': total_loss,
            'state_loss': state_loss,
            'reward_loss': reward_loss
        }
    
    def rollout(self, initial_state: torch.Tensor, action_sequence: torch.Tensor,
                deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Roll out a sequence of actions using the ensemble.
        
        Args:
            initial_state: Initial state
            action_sequence: Sequence of actions
            deterministic: Whether to use deterministic predictions
            
        Returns:
            Tuple of (state_sequence, reward_sequence)
        """
        # Use the first model for rollout (can be extended to use ensemble)
        return self.models[0].rollout(initial_state, action_sequence, deterministic)


def create_dynamics_model(model_type: str = 'probabilistic', state_dim: int = 4,
                         action_dim: int = 1, hidden_sizes: list = [256, 256],
                         activation: str = 'relu', num_models: int = 5,
                         use_orthogonal_init: bool = True) -> nn.Module:
    """
    Create a dynamics model.
    
    Args:
        model_type: Type of model ('probabilistic', 'deterministic', 'ensemble')
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        hidden_sizes: List of hidden layer sizes
        activation: Activation function
        num_models: Number of models in ensemble
        use_orthogonal_init: Whether to use orthogonal initialization
        
    Returns:
        Dynamics model
    """
    if model_type == 'probabilistic':
        return ProbabilisticDynamicsModel(
            state_dim, action_dim, hidden_sizes, activation,
            probabilistic=True, use_orthogonal_init=use_orthogonal_init
        )
    elif model_type == 'deterministic':
        return ProbabilisticDynamicsModel(
            state_dim, action_dim, hidden_sizes, activation,
            probabilistic=False, use_orthogonal_init=use_orthogonal_init
        )
    elif model_type == 'ensemble':
        return EnsembleDynamicsModel(
            state_dim, action_dim, num_models, hidden_sizes, activation,
            probabilistic=True, use_orthogonal_init=use_orthogonal_init
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test the dynamics model
    print("Testing Probabilistic Dynamics Model...")
    
    state_dim = 4
    action_dim = 1
    batch_size = 32
    
    # Create model
    model = create_dynamics_model('probabilistic', state_dim, action_dim)
    
    # Test forward pass
    states = torch.randn(batch_size, state_dim)
    actions = torch.randn(batch_size, action_dim)
    
    outputs = model.forward(states, actions)
    print(f"Output keys: {list(outputs.keys())}")
    print(f"State mean shape: {outputs['state_mean'].shape}")
    print(f"Reward mean shape: {outputs['reward_mean'].shape}")
    
    # Test prediction
    next_states, rewards = model.predict(states, actions)
    print(f"Next states shape: {next_states.shape}")
    print(f"Rewards shape: {rewards.shape}")
    
    # Test loss computation
    next_states_target = torch.randn(batch_size, state_dim)
    rewards_target = torch.randn(batch_size, 1)
    
    losses = model.compute_loss(states, actions, next_states_target, rewards_target)
    print(f"Losses: {losses}")
    
    # Test rollout
    initial_state = torch.randn(state_dim)
    action_sequence = torch.randn(10, action_dim)
    
    state_seq, reward_seq = model.rollout(initial_state, action_sequence)
    print(f"State sequence shape: {state_seq.shape}")
    print(f"Reward sequence shape: {reward_seq.shape}")
    
    print("Dynamics model test completed!")
