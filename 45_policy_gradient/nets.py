import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class PolicyNet(nn.Module):
    """
    Policy Network for REINFORCE algorithm.
    
    Outputs a categorical distribution over discrete actions.
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: list = [128, 128]):
        """
        Initialize Policy Network.
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            hidden_sizes: List of hidden layer sizes
        """
        super(PolicyNet, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Build network layers
        layers = []
        prev_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size
        
        # Output layer (logits for categorical distribution)
        layers.append(nn.Linear(prev_size, action_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Logits for action probabilities
        """
        return self.network(state)
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            state: Input state tensor
            deterministic: If True, return action with highest probability
            
        Returns:
            Tuple of (action, log_probability)
        """
        logits = self.forward(state)
        
        if deterministic:
            action = torch.argmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            log_prob = torch.gather(log_prob, 1, action.unsqueeze(1)).squeeze(1)
        else:
            # Sample from categorical distribution
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob
    
    def get_log_probs(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Get log probabilities for given state-action pairs.
        
        Args:
            state: Input state tensor
            action: Action tensor
            
        Returns:
            Log probabilities
        """
        logits = self.forward(state)
        log_probs = F.log_softmax(logits, dim=-1)
        return torch.gather(log_probs, 1, action.unsqueeze(1)).squeeze(1)
    
    def get_entropy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Calculate entropy of the policy distribution.
        
        Args:
            state: Input state tensor
            
        Returns:
            Entropy values
        """
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy


class ValueNet(nn.Module):
    """
    Value Network for baseline estimation in REINFORCE.
    
    Estimates state values V(s) to reduce variance in policy gradient.
    """
    
    def __init__(self, state_size: int, hidden_sizes: list = [128, 128]):
        """
        Initialize Value Network.
        
        Args:
            state_size: Dimension of state space
            hidden_sizes: List of hidden layer sizes
        """
        super(ValueNet, self).__init__()
        
        self.state_size = state_size
        
        # Build network layers
        layers = []
        prev_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size
        
        # Output layer (single value)
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor
            
        Returns:
            State values V(s)
        """
        return self.network(state).squeeze(-1)


class ActorCriticNet(nn.Module):
    """
    Combined Actor-Critic Network.
    
    Shares features between policy and value networks for efficiency.
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: list = [128, 128]):
        """
        Initialize Actor-Critic Network.
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            hidden_sizes: List of hidden layer sizes
        """
        super(ActorCriticNet, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Shared feature layers
        shared_layers = []
        prev_size = state_size
        
        for hidden_size in hidden_sizes[:-1]:
            shared_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size
        
        self.shared_network = nn.Sequential(*shared_layers)
        
        # Policy head
        self.policy_head = nn.Linear(prev_size, action_size)
        
        # Value head
        self.value_head = nn.Linear(prev_size, 1)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Tuple of (policy_logits, value)
        """
        features = self.shared_network(state)
        policy_logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        
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


class GaussianPolicyNet(nn.Module):
    """
    Gaussian Policy Network for continuous action spaces.
    
    Outputs mean and log_std for diagonal Gaussian distribution.
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: list = [128, 128]):
        """
        Initialize Gaussian Policy Network.
        
        Args:
            state_size: Dimension of state space
            action_size: Number of action dimensions
            hidden_sizes: List of hidden layer sizes
        """
        super(GaussianPolicyNet, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Build network layers
        layers = []
        prev_size = state_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size
        
        # Mean and log_std heads
        self.mean_head = nn.Linear(prev_size, action_size)
        self.log_std_head = nn.Linear(prev_size, action_size)
        
        self.shared_network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Tuple of (mean, log_std)
        """
        features = self.shared_network(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        
        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        return mean, log_std
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            state: Input state tensor
            deterministic: If True, return mean action
            
        Returns:
            Tuple of (action, log_probability)
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        if deterministic:
            action = mean
            log_prob = torch.zeros_like(action)
        else:
            # Sample from Gaussian distribution
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob
    
    def get_log_probs(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Get log probabilities for given state-action pairs.
        
        Args:
            state: Input state tensor
            action: Action tensor
            
        Returns:
            Log probabilities
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(action).sum(dim=-1)
    
    def get_entropy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Calculate entropy of the policy distribution.
        
        Args:
            state: Input state tensor
            
        Returns:
            Entropy values
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        dist = torch.distributions.Normal(mean, std)
        return dist.entropy().sum(dim=-1)


def initialize_weights(module: nn.Module, gain: float = 1.0):
    """
    Initialize network weights using orthogonal initialization.
    
    Args:
        module: PyTorch module
        gain: Gain factor for orthogonal initialization
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.0)


def create_policy_network(state_size: int, action_size: int, 
                         hidden_sizes: list = [128, 128],
                         use_orthogonal_init: bool = True) -> PolicyNet:
    """
    Create and initialize a policy network.
    
    Args:
        state_size: Dimension of state space
        action_size: Number of possible actions
        hidden_sizes: List of hidden layer sizes
        use_orthogonal_init: Whether to use orthogonal initialization
        
    Returns:
        Initialized PolicyNet
    """
    net = PolicyNet(state_size, action_size, hidden_sizes)
    
    if use_orthogonal_init:
        net.apply(lambda m: initialize_weights(m, gain=0.01))
    
    return net


def create_value_network(state_size: int, 
                        hidden_sizes: list = [128, 128],
                        use_orthogonal_init: bool = True) -> ValueNet:
    """
    Create and initialize a value network.
    
    Args:
        state_size: Dimension of state space
        hidden_sizes: List of hidden layer sizes
        use_orthogonal_init: Whether to use orthogonal initialization
        
    Returns:
        Initialized ValueNet
    """
    net = ValueNet(state_size, hidden_sizes)
    
    if use_orthogonal_init:
        net.apply(lambda m: initialize_weights(m, gain=1.0))
    
    return net


def create_actor_critic_network(state_size: int, action_size: int,
                               hidden_sizes: list = [128, 128],
                               use_orthogonal_init: bool = True) -> ActorCriticNet:
    """
    Create and initialize an actor-critic network.
    
    Args:
        state_size: Dimension of state space
        action_size: Number of possible actions
        hidden_sizes: List of hidden layer sizes
        use_orthogonal_init: Whether to use orthogonal initialization
        
    Returns:
        Initialized ActorCriticNet
    """
    net = ActorCriticNet(state_size, action_size, hidden_sizes)
    
    if use_orthogonal_init:
        # Initialize policy head with small gain
        net.policy_head.apply(lambda m: initialize_weights(m, gain=0.01))
        # Initialize value head with standard gain
        net.value_head.apply(lambda m: initialize_weights(m, gain=1.0))
        # Initialize shared layers with standard gain
        net.shared_network.apply(lambda m: initialize_weights(m, gain=1.0))
    
    return net


if __name__ == "__main__":
    # Test the networks
    print("Testing Policy Networks...")
    
    state_size = 4
    action_size = 2
    batch_size = 32
    
    # Test PolicyNet
    policy_net = create_policy_network(state_size, action_size)
    state = torch.randn(batch_size, state_size)
    
    action, log_prob = policy_net.get_action(state)
    entropy = policy_net.get_entropy(state)
    
    print(f"PolicyNet - Action shape: {action.shape}, Log prob shape: {log_prob.shape}")
    print(f"PolicyNet - Entropy shape: {entropy.shape}")
    
    # Test ValueNet
    value_net = create_value_network(state_size)
    value = value_net(state)
    
    print(f"ValueNet - Value shape: {value.shape}")
    
    # Test ActorCriticNet
    ac_net = create_actor_critic_network(state_size, action_size)
    action, log_prob, value = ac_net.get_action(state)
    entropy = ac_net.get_entropy(state)
    
    print(f"ActorCriticNet - Action shape: {action.shape}, Log prob shape: {log_prob.shape}")
    print(f"ActorCriticNet - Value shape: {value.shape}, Entropy shape: {entropy.shape}")
    
    print("Network tests completed!")
