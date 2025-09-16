import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encoder network that maps observations to latent representations.
    
    For CartPole: 4-dim state -> 32-dim latent
    For Pendulum: 3-dim state -> 32-dim latent
    """
    
    def __init__(self, state_dim, latent_dim=32, hidden_dim=128):
        super(Encoder, self).__init__()
        
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # Mean and log_std
        )
        
    def forward(self, state):
        """
        Encode state to latent representation.
        
        Args:
            state: (batch_size, state_dim) observation tensor
            
        Returns:
            latent: (batch_size, latent_dim) latent representation
            mean: (batch_size, latent_dim) mean of latent distribution
            log_std: (batch_size, latent_dim) log standard deviation
        """
        h = self.encoder(state)
        mean, log_std = torch.chunk(h, 2, dim=-1)
        
        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, -10, 2)
        
        # Sample from latent distribution
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        latent = mean + eps * std
        
        return latent, mean, log_std
    
    def encode_deterministic(self, state):
        """
        Encode state to latent representation without sampling (deterministic).
        
        Args:
            state: (batch_size, state_dim) observation tensor
            
        Returns:
            mean: (batch_size, latent_dim) mean of latent distribution
        """
        h = self.encoder(state)
        mean, _ = torch.chunk(h, 2, dim=-1)
        return mean


