import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


class Actor(nn.Module):
    """
    Actor network that outputs action distribution given latent state.
    
    Implements: π_θ(a|z)
    """
    
    def __init__(self, latent_dim, action_dim, hidden_dim=128, max_action=1.0):
        super(Actor, self).__init__()
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * 2)  # Mean and log_std
        )
        
    def forward(self, latent):
        """
        Get action distribution for given latent state.
        
        Args:
            latent: (batch_size, latent_dim) latent state
            
        Returns:
            action: (batch_size, action_dim) sampled action
            mean: (batch_size, action_dim) action mean
            log_std: (batch_size, action_dim) action log std
        """
        h = self.actor(latent)
        mean, log_std = torch.chunk(h, 2, dim=-1)
        
        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, -20, 2)
        
        # Sample action
        std = torch.exp(log_std)
        normal = dist.Normal(mean, std)
        action = normal.rsample()
        
        # Clamp action to valid range
        action = torch.tanh(action) * self.max_action
        
        return action, mean, log_std
    
    def get_action_deterministic(self, latent):
        """
        Get deterministic action (mean) for given latent state.
        
        Args:
            latent: (batch_size, latent_dim) latent state
            
        Returns:
            action: (batch_size, action_dim) deterministic action
        """
        h = self.actor(latent)
        mean, _ = torch.chunk(h, 2, dim=-1)
        action = torch.tanh(mean) * self.max_action
        return action
    
    def log_prob(self, latent, action):
        """
        Compute log probability of action given latent state.
        
        Args:
            latent: (batch_size, latent_dim) latent state
            action: (batch_size, action_dim) action
            
        Returns:
            log_prob: (batch_size,) log probability
        """
        h = self.actor(latent)
        mean, log_std = torch.chunk(h, 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        
        std = torch.exp(log_std)
        normal = dist.Normal(mean, std)
        
        # Apply tanh transformation
        action_tanh = torch.atanh(torch.clamp(action / self.max_action, -0.999, 0.999))
        log_prob = normal.log_prob(action_tanh)
        
        # Apply tanh correction
        log_prob -= torch.log(1 - torch.tanh(action_tanh) ** 2 + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        
        return log_prob


class Critic(nn.Module):
    """
    Critic network that estimates value function given latent state.
    
    Implements: V_ψ(z)
    """
    
    def __init__(self, latent_dim, hidden_dim=128):
        super(Critic, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Single value
        )
        
    def forward(self, latent):
        """
        Get value estimate for given latent state.
        
        Args:
            latent: (batch_size, latent_dim) latent state
            
        Returns:
            value: (batch_size, 1) estimated value
        """
        return self.critic(latent)




