"""
Actor-Critic Networks for Multi-Agent Reinforcement Learning

This module implements actor-critic networks specifically designed for
multi-agent environments, supporting both IPPO and MAPPO architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import math


class SharedActorCritic(nn.Module):
    """
    Shared Actor-Critic Network for Multi-Agent RL.
    
    This network can be used by multiple agents in IPPO, where each agent
    has its own instance but they share the same architecture and parameters.
    """
    
    def __init__(self, obs_dim: int, action_dim: int, 
                 hidden_sizes: List[int] = [256, 256],
                 activation: str = 'relu', use_orthogonal_init: bool = True):
        """
        Initialize Shared Actor-Critic Network.
        
        Args:
            obs_dim: Dimension of observation space
            action_dim: Number of possible actions
            hidden_sizes: List of hidden layer sizes
            activation: Activation function ('relu', 'tanh', 'elu')
            use_orthogonal_init: Whether to use orthogonal initialization
        """
        super(SharedActorCritic, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes
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
        
        # Shared feature extractor
        self.shared_layers = nn.ModuleList()
        prev_size = obs_dim
        
        for hidden_size in hidden_sizes:
            self.shared_layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        # Actor head (policy)
        self.actor_head = nn.Linear(prev_size, action_dim)
        
        # Critic head (value)
        self.critic_head = nn.Linear(prev_size, 1)
        
        # Initialize weights
        if use_orthogonal_init:
            self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using orthogonal initialization."""
        if isinstance(module, nn.Linear):
            # Orthogonal initialization for shared layers
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
            
            # Special initialization for output heads
            if module == self.actor_head:
                # Small gain for policy head
                nn.init.orthogonal_(module.weight, gain=0.01)
            elif module == self.critic_head:
                # Standard gain for value head
                nn.init.orthogonal_(module.weight, gain=1.0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            obs: Input observation tensor [batch_size, obs_dim]
            
        Returns:
            Tuple of (policy_logits, value)
        """
        # Shared feature extraction
        x = obs
        for layer in self.shared_layers:
            x = self.activation()(layer(x))
        
        # Actor and critic heads
        policy_logits = self.actor_head(x)
        value = self.critic_head(x).squeeze(-1)
        
        return policy_logits, value
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy and get value estimate.
        
        Args:
            obs: Input observation tensor
            deterministic: If True, return action with highest probability
            
        Returns:
            Tuple of (action, log_probability, value)
        """
        policy_logits, value = self.forward(obs)
        
        if deterministic:
            action = torch.argmax(policy_logits, dim=-1)
            log_prob = F.log_softmax(policy_logits, dim=-1)
            log_prob = torch.gather(log_prob, 1, action.unsqueeze(1)).squeeze(1)
        else:
            # Sample from categorical distribution
            dist = torch.distributions.Categorical(logits=policy_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob, value
    
    def get_log_probs(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get log probabilities and value for given observation-action pairs.
        
        Args:
            obs: Input observation tensor
            action: Action tensor
            
        Returns:
            Tuple of (log_probabilities, values)
        """
        policy_logits, value = self.forward(obs)
        log_probs = F.log_softmax(policy_logits, dim=-1)
        log_prob = torch.gather(log_probs, 1, action.unsqueeze(1)).squeeze(1)
        
        return log_prob, value
    
    def get_entropy(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Calculate entropy of the policy distribution.
        
        Args:
            obs: Input observation tensor
            
        Returns:
            Entropy values
        """
        policy_logits, _ = self.forward(obs)
        probs = F.softmax(policy_logits, dim=-1)
        log_probs = F.log_softmax(policy_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy
    
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate for observations.
        
        Args:
            obs: Input observation tensor
            
        Returns:
            Value estimates
        """
        _, value = self.forward(obs)
        return value


class CentralizedCritic(nn.Module):
    """
    Centralized Critic Network for MAPPO.
    
    This network uses global state information (all agents' observations)
    to provide value estimates during training, while policies still use
    local observations (CTDE - Centralized Training, Decentralized Execution).
    """
    
    def __init__(self, global_obs_dim: int, hidden_sizes: List[int] = [256, 256],
                 activation: str = 'relu', use_orthogonal_init: bool = True):
        """
        Initialize Centralized Critic Network.
        
        Args:
            global_obs_dim: Dimension of global observation (concatenated)
            hidden_sizes: List of hidden layer sizes
            activation: Activation function
            use_orthogonal_init: Whether to use orthogonal initialization
        """
        super(CentralizedCritic, self).__init__()
        
        self.global_obs_dim = global_obs_dim
        self.hidden_sizes = hidden_sizes
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
        
        # Critic layers
        self.critic_layers = nn.ModuleList()
        prev_size = global_obs_dim
        
        for hidden_size in hidden_sizes:
            self.critic_layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        # Value head
        self.value_head = nn.Linear(prev_size, 1)
        
        # Initialize weights
        if use_orthogonal_init:
            self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using orthogonal initialization."""
        if isinstance(module, nn.Linear):
            # Orthogonal initialization
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
            
            # Special initialization for value head
            if module == self.value_head:
                nn.init.orthogonal_(module.weight, gain=1.0)
    
    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the centralized critic.
        
        Args:
            global_obs: Global observation tensor [batch_size, global_obs_dim]
            
        Returns:
            Value estimates
        """
        x = global_obs
        for layer in self.critic_layers:
            x = self.activation()(layer(x))
        
        value = self.value_head(x).squeeze(-1)
        return value


class MultiAgentActorCritic(nn.Module):
    """
    Multi-Agent Actor-Critic Network for IPPO.
    
    This network manages multiple actor-critic networks, one for each agent.
    All agents share the same network architecture but have separate parameters.
    """
    
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int,
                 hidden_sizes: List[int] = [256, 256], activation: str = 'relu',
                 use_orthogonal_init: bool = True, shared_parameters: bool = False):
        """
        Initialize Multi-Agent Actor-Critic Network.
        
        Args:
            num_agents: Number of agents
            obs_dim: Dimension of observation space
            action_dim: Number of possible actions
            hidden_sizes: List of hidden layer sizes
            activation: Activation function
            use_orthogonal_init: Whether to use orthogonal initialization
            shared_parameters: Whether to share parameters across agents
        """
        super(MultiAgentActorCritic, self).__init__()
        
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.shared_parameters = shared_parameters
        
        if shared_parameters:
            # All agents share the same network
            self.shared_network = SharedActorCritic(
                obs_dim, action_dim, hidden_sizes, activation, use_orthogonal_init
            )
            self.agent_networks = [self.shared_network] * num_agents
        else:
            # Each agent has its own network
            self.agent_networks = nn.ModuleList([
                SharedActorCritic(obs_dim, action_dim, hidden_sizes, activation, use_orthogonal_init)
                for _ in range(num_agents)
            ])
    
    def get_action(self, obs: torch.Tensor, agent_id: int, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action for a specific agent.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim]
            agent_id: ID of the agent
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Tuple of (action, log_probability, value)
        """
        return self.agent_networks[agent_id].get_action(obs, deterministic)
    
    def get_log_probs(self, obs: torch.Tensor, action: torch.Tensor, agent_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get log probabilities and value for a specific agent.
        
        Args:
            obs: Observation tensor
            action: Action tensor
            agent_id: ID of the agent
            
        Returns:
            Tuple of (log_probabilities, values)
        """
        return self.agent_networks[agent_id].get_log_probs(obs, action)
    
    def get_entropy(self, obs: torch.Tensor, agent_id: int) -> torch.Tensor:
        """
        Get entropy for a specific agent.
        
        Args:
            obs: Observation tensor
            agent_id: ID of the agent
            
        Returns:
            Entropy values
        """
        return self.agent_networks[agent_id].get_entropy(obs)
    
    def get_value(self, obs: torch.Tensor, agent_id: int) -> torch.Tensor:
        """
        Get value estimate for a specific agent.
        
        Args:
            obs: Observation tensor
            agent_id: ID of the agent
            
        Returns:
            Value estimates
        """
        return self.agent_networks[agent_id].get_value(obs)
    
    def get_all_actions(self, obs_list: List[torch.Tensor], deterministic: bool = False) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Get actions for all agents.
        
        Args:
            obs_list: List of observation tensors for each agent
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Tuple of (actions, log_probs, values) for each agent
        """
        actions = []
        log_probs = []
        values = []
        
        for i, obs in enumerate(obs_list):
            action, log_prob, value = self.get_action(obs, i, deterministic)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
        
        return actions, log_probs, values


class MAPPOActorCritic(nn.Module):
    """
    MAPPO Actor-Critic Network with Centralized Critic.
    
    This network combines individual actor networks (one per agent) with
    a centralized critic that uses global state information.
    """
    
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int,
                 global_obs_dim: int, hidden_sizes: List[int] = [256, 256],
                 activation: str = 'relu', use_orthogonal_init: bool = True):
        """
        Initialize MAPPO Actor-Critic Network.
        
        Args:
            num_agents: Number of agents
            obs_dim: Dimension of local observation space
            action_dim: Number of possible actions
            global_obs_dim: Dimension of global observation space
            hidden_sizes: List of hidden layer sizes
            activation: Activation function
            use_orthogonal_init: Whether to use orthogonal initialization
        """
        super(MAPPOActorCritic, self).__init__()
        
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.global_obs_dim = global_obs_dim
        
        # Individual actor networks (one per agent)
        self.actor_networks = nn.ModuleList([
            SharedActorCritic(obs_dim, action_dim, hidden_sizes, activation, use_orthogonal_init)
            for _ in range(num_agents)
        ])
        
        # Centralized critic network
        self.centralized_critic = CentralizedCritic(
            global_obs_dim, hidden_sizes, activation, use_orthogonal_init
        )
    
    def get_action(self, obs: torch.Tensor, agent_id: int, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action for a specific agent using local observation.
        
        Args:
            obs: Local observation tensor [batch_size, obs_dim]
            agent_id: ID of the agent
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Tuple of (action, log_probability, value)
        """
        return self.actor_networks[agent_id].get_action(obs, deterministic)
    
    def get_log_probs(self, obs: torch.Tensor, action: torch.Tensor, agent_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get log probabilities and value for a specific agent.
        
        Args:
            obs: Local observation tensor
            action: Action tensor
            agent_id: ID of the agent
            
        Returns:
            Tuple of (log_probabilities, values)
        """
        return self.actor_networks[agent_id].get_log_probs(obs, action)
    
    def get_entropy(self, obs: torch.Tensor, agent_id: int) -> torch.Tensor:
        """
        Get entropy for a specific agent.
        
        Args:
            obs: Local observation tensor
            agent_id: ID of the agent
            
        Returns:
            Entropy values
        """
        return self.actor_networks[agent_id].get_entropy(obs)
    
    def get_centralized_value(self, global_obs: torch.Tensor) -> torch.Tensor:
        """
        Get centralized value estimate using global observation.
        
        Args:
            global_obs: Global observation tensor [batch_size, global_obs_dim]
            
        Returns:
            Centralized value estimates
        """
        return self.centralized_critic(global_obs)
    
    def get_all_actions(self, obs_list: List[torch.Tensor], deterministic: bool = False) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Get actions for all agents using local observations.
        
        Args:
            obs_list: List of local observation tensors for each agent
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Tuple of (actions, log_probs, values) for each agent
        """
        actions = []
        log_probs = []
        values = []
        
        for i, obs in enumerate(obs_list):
            action, log_prob, value = self.get_action(obs, i, deterministic)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
        
        return actions, log_probs, values


def create_actor_critic_network(network_type: str, num_agents: int, obs_dim: int, 
                               action_dim: int, global_obs_dim: Optional[int] = None,
                               hidden_sizes: List[int] = [256, 256], 
                               activation: str = 'relu',
                               use_orthogonal_init: bool = True,
                               shared_parameters: bool = False) -> nn.Module:
    """
    Create an actor-critic network for multi-agent RL.
    
    Args:
        network_type: Type of network ('ippo', 'mappo')
        num_agents: Number of agents
        obs_dim: Dimension of observation space
        action_dim: Number of possible actions
        global_obs_dim: Dimension of global observation space (for MAPPO)
        hidden_sizes: List of hidden layer sizes
        activation: Activation function
        use_orthogonal_init: Whether to use orthogonal initialization
        shared_parameters: Whether to share parameters across agents (for IPPO)
        
    Returns:
        Actor-critic network
    """
    if network_type.lower() == 'ippo':
        return MultiAgentActorCritic(
            num_agents, obs_dim, action_dim, hidden_sizes, 
            activation, use_orthogonal_init, shared_parameters
        )
    elif network_type.lower() == 'mappo':
        if global_obs_dim is None:
            raise ValueError("global_obs_dim must be provided for MAPPO")
        return MAPPOActorCritic(
            num_agents, obs_dim, action_dim, global_obs_dim,
            hidden_sizes, activation, use_orthogonal_init
        )
    else:
        raise ValueError(f"Unknown network type: {network_type}")


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the networks
    print("Testing Multi-Agent Actor-Critic Networks...")
    
    num_agents = 3
    obs_dim = 18  # Typical MPE observation dimension
    action_dim = 5  # Typical MPE action dimension
    global_obs_dim = obs_dim * num_agents
    
    # Test IPPO network
    print("\nTesting IPPO Network:")
    ippo_net = create_actor_critic_network(
        'ippo', num_agents, obs_dim, action_dim, 
        shared_parameters=True
    )
    
    print(f"IPPO Parameters: {count_parameters(ippo_net)}")
    
    # Test MAPPO network
    print("\nTesting MAPPO Network:")
    mappo_net = create_actor_critic_network(
        'mappo', num_agents, obs_dim, action_dim, global_obs_dim
    )
    
    print(f"MAPPO Parameters: {count_parameters(mappo_net)}")
    
    # Test forward pass
    batch_size = 32
    obs = torch.randn(batch_size, obs_dim)
    global_obs = torch.randn(batch_size, global_obs_dim)
    
    # Test IPPO
    action, log_prob, value = ippo_net.get_action(obs, agent_id=0)
    print(f"IPPO Action shape: {action.shape}")
    print(f"IPPO Log prob shape: {log_prob.shape}")
    print(f"IPPO Value shape: {value.shape}")
    
    # Test MAPPO
    action, log_prob, value = mappo_net.get_action(obs, agent_id=0)
    centralized_value = mappo_net.get_centralized_value(global_obs)
    print(f"MAPPO Action shape: {action.shape}")
    print(f"MAPPO Log prob shape: {log_prob.shape}")
    print(f"MAPPO Value shape: {value.shape}")
    print(f"MAPPO Centralized value shape: {centralized_value.shape}")
    
    print("\nNetwork tests completed!")
