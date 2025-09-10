import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
import math


class SharedActorCritic(nn.Module):
    """
    Shared Actor-Critic Network with common feature extractor.
    
    This architecture shares the feature extraction layers between
    the actor (policy) and critic (value) networks, which is more
    parameter-efficient and often performs better.
    """
    
    def __init__(self, state_size: int, action_size: int, 
                 hidden_sizes: List[int] = [256, 256],
                 activation: str = 'relu', use_orthogonal_init: bool = True):
        """
        Initialize Shared Actor-Critic Network.
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            hidden_sizes: List of hidden layer sizes
            activation: Activation function ('relu', 'tanh', 'elu')
            use_orthogonal_init: Whether to use orthogonal initialization
        """
        super(SharedActorCritic, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
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
        prev_size = state_size
        
        for hidden_size in hidden_sizes:
            self.shared_layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        # Actor head (policy)
        self.actor_head = nn.Linear(prev_size, action_size)
        
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
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Tuple of (policy_logits, value)
        """
        # Shared feature extraction
        x = state
        for layer in self.shared_layers:
            x = self.activation()(layer(x))
        
        # Actor and critic heads
        policy_logits = self.actor_head(x)
        value = self.critic_head(x).squeeze(-1)
        
        return policy_logits, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy and get value estimate.
        
        Args:
            state: Input state tensor
            deterministic: If True, return action with highest probability
            
        Returns:
            Tuple of (action, log_probability, value)
        """
        policy_logits, value = self.forward(state)
        
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
    
    def get_log_probs(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get log probabilities and value for given state-action pairs.
        
        Args:
            state: Input state tensor
            action: Action tensor
            
        Returns:
            Tuple of (log_probabilities, values)
        """
        policy_logits, value = self.forward(state)
        log_probs = F.log_softmax(policy_logits, dim=-1)
        log_prob = torch.gather(log_probs, 1, action.unsqueeze(1)).squeeze(1)
        
        return log_prob, value
    
    def get_entropy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Calculate entropy of the policy distribution.
        
        Args:
            state: Input state tensor
            
        Returns:
            Entropy values
        """
        policy_logits, _ = self.forward(state)
        probs = F.softmax(policy_logits, dim=-1)
        log_probs = F.log_softmax(policy_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy


class SplitActorCritic(nn.Module):
    """
    Split Actor-Critic Network with separate networks.
    
    This architecture uses completely separate networks for the
    actor and critic, which can be more stable in some cases.
    """
    
    def __init__(self, state_size: int, action_size: int,
                 actor_hidden_sizes: List[int] = [256, 256],
                 critic_hidden_sizes: List[int] = [256, 256],
                 activation: str = 'relu', use_orthogonal_init: bool = True):
        """
        Initialize Split Actor-Critic Network.
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            actor_hidden_sizes: Hidden layer sizes for actor network
            critic_hidden_sizes: Hidden layer sizes for critic network
            activation: Activation function
            use_orthogonal_init: Whether to use orthogonal initialization
        """
        super(SplitActorCritic, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
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
        
        # Actor network
        self.actor_layers = nn.ModuleList()
        prev_size = state_size
        
        for hidden_size in actor_hidden_sizes:
            self.actor_layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        self.actor_head = nn.Linear(prev_size, action_size)
        
        # Critic network
        self.critic_layers = nn.ModuleList()
        prev_size = state_size
        
        for hidden_size in critic_hidden_sizes:
            self.critic_layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        self.critic_head = nn.Linear(prev_size, 1)
        
        # Initialize weights
        if use_orthogonal_init:
            self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using orthogonal initialization."""
        if isinstance(module, nn.Linear):
            # Orthogonal initialization
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
            
            # Special initialization for output heads
            if module == self.actor_head:
                nn.init.orthogonal_(module.weight, gain=0.01)
            elif module == self.critic_head:
                nn.init.orthogonal_(module.weight, gain=1.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both networks.
        
        Args:
            state: Input state tensor
            
        Returns:
            Tuple of (policy_logits, value)
        """
        # Actor forward pass
        actor_x = state
        for layer in self.actor_layers:
            actor_x = self.activation()(layer(actor_x))
        policy_logits = self.actor_head(actor_x)
        
        # Critic forward pass
        critic_x = state
        for layer in self.critic_layers:
            critic_x = self.activation()(layer(critic_x))
        value = self.critic_head(critic_x).squeeze(-1)
        
        return policy_logits, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy and get value estimate.
        
        Args:
            state: Input state tensor
            deterministic: If True, return action with highest probability
            
        Returns:
            Tuple of (action, log_probability, value)
        """
        policy_logits, value = self.forward(state)
        
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
    
    def get_log_probs(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get log probabilities and value for given state-action pairs.
        
        Args:
            state: Input state tensor
            action: Action tensor
            
        Returns:
            Tuple of (log_probabilities, values)
        """
        policy_logits, value = self.forward(state)
        log_probs = F.log_softmax(policy_logits, dim=-1)
        log_prob = torch.gather(log_probs, 1, action.unsqueeze(1)).squeeze(1)
        
        return log_prob, value
    
    def get_entropy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Calculate entropy of the policy distribution.
        
        Args:
            state: Input state tensor
            
        Returns:
            Entropy values
        """
        policy_logits, _ = self.forward(state)
        probs = F.softmax(policy_logits, dim=-1)
        log_probs = F.log_softmax(policy_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy


class ActorCriticWithGAE(nn.Module):
    """
    Actor-Critic Network optimized for GAE (Generalized Advantage Estimation).
    
    This is a specialized version that includes methods for computing
    advantages and returns using GAE.
    """
    
    def __init__(self, state_size: int, action_size: int,
                 hidden_sizes: List[int] = [256, 256],
                 activation: str = 'relu', use_orthogonal_init: bool = True,
                 network_type: str = 'shared'):
        """
        Initialize Actor-Critic Network with GAE support.
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            hidden_sizes: Hidden layer sizes
            activation: Activation function
            use_orthogonal_init: Whether to use orthogonal initialization
            network_type: 'shared' or 'split' architecture
        """
        super(ActorCriticWithGAE, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.network_type = network_type
        
        if network_type == 'shared':
            self.network = SharedActorCritic(
                state_size, action_size, hidden_sizes, activation, use_orthogonal_init
            )
        elif network_type == 'split':
            self.network = SplitActorCritic(
                state_size, action_size, hidden_sizes, hidden_sizes, activation, use_orthogonal_init
            )
        else:
            raise ValueError(f"Unknown network type: {network_type}")
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network."""
        return self.network.forward(state)
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action from policy and value estimate."""
        return self.network.get_action(state, deterministic)
    
    def get_log_probs(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get log probabilities and value for given state-action pairs."""
        return self.network.get_log_probs(state, action)
    
    def get_entropy(self, state: torch.Tensor) -> torch.Tensor:
        """Calculate entropy of the policy distribution."""
        return self.network.get_entropy(state)
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate for states.
        
        Args:
            state: Input state tensor
            
        Returns:
            Value estimates
        """
        _, value = self.forward(state)
        return value


def create_actor_critic_network(state_size: int, action_size: int,
                               hidden_sizes: List[int] = [256, 256],
                               activation: str = 'relu',
                               network_type: str = 'shared',
                               use_orthogonal_init: bool = True) -> ActorCriticWithGAE:
    """
    Create an Actor-Critic network.
    
    Args:
        state_size: Dimension of state space
        action_size: Number of possible actions
        hidden_sizes: Hidden layer sizes
        activation: Activation function
        network_type: 'shared' or 'split' architecture
        use_orthogonal_init: Whether to use orthogonal initialization
        
    Returns:
        Actor-Critic network
    """
    return ActorCriticWithGAE(
        state_size=state_size,
        action_size=action_size,
        hidden_sizes=hidden_sizes,
        activation=activation,
        use_orthogonal_init=use_orthogonal_init,
        network_type=network_type
    )


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model: nn.Module) -> dict:
    """
    Get information about the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model information
    """
    total_params = count_parameters(model)
    
    info = {
        'total_parameters': total_params,
        'model_type': type(model).__name__,
        'device': next(model.parameters()).device,
    }
    
    if hasattr(model, 'network_type'):
        info['network_type'] = model.network_type
    
    return info


if __name__ == "__main__":
    # Test the networks
    print("Testing Actor-Critic Networks...")
    
    state_size = 4
    action_size = 2
    batch_size = 32
    
    # Test shared network
    print("\nTesting Shared Actor-Critic:")
    shared_net = create_actor_critic_network(
        state_size, action_size, [128, 128], 'relu', 'shared'
    )
    
    state = torch.randn(batch_size, state_size)
    action, log_prob, value = shared_net.get_action(state)
    entropy = shared_net.get_entropy(state)
    
    print(f"Action shape: {action.shape}")
    print(f"Log prob shape: {log_prob.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Entropy shape: {entropy.shape}")
    print(f"Total parameters: {count_parameters(shared_net)}")
    
    # Test split network
    print("\nTesting Split Actor-Critic:")
    split_net = create_actor_critic_network(
        state_size, action_size, [128, 128], 'relu', 'split'
    )
    
    action, log_prob, value = split_net.get_action(state)
    entropy = split_net.get_entropy(state)
    
    print(f"Action shape: {action.shape}")
    print(f"Log prob shape: {log_prob.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Entropy shape: {entropy.shape}")
    print(f"Total parameters: {count_parameters(split_net)}")
    
    print("\nNetwork tests completed!")
