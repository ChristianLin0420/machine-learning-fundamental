import torch
import torch.nn as nn
import torch.nn.functional as F


class TransitionModel(nn.Module):
    """
    Transition model that predicts next latent state given current latent and action.
    
    Implements: z_{t+1} ~ g_φ(z_t, a_t)
    """
    
    def __init__(self, latent_dim, action_dim, hidden_dim=128):
        super(TransitionModel, self).__init__()
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        # Transition network
        self.transition = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # Mean and log_std
        )
        
    def forward(self, latent, action):
        """
        Predict next latent state.
        
        Args:
            latent: (batch_size, latent_dim) current latent state
            action: (batch_size, action_dim) action taken
            
        Returns:
            next_latent: (batch_size, latent_dim) predicted next latent
            mean: (batch_size, latent_dim) mean of next latent distribution
            log_std: (batch_size, latent_dim) log standard deviation
        """
        # Concatenate latent and action
        x = torch.cat([latent, action], dim=-1)
        h = self.transition(x)
        mean, log_std = torch.chunk(h, 2, dim=-1)
        
        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, -10, 2)
        
        # Sample from predicted distribution
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        next_latent = mean + eps * std
        
        return next_latent, mean, log_std
    
    def predict_deterministic(self, latent, action):
        """
        Predict next latent state deterministically (without sampling).
        
        Args:
            latent: (batch_size, latent_dim) current latent state
            action: (batch_size, action_dim) action taken
            
        Returns:
            mean: (batch_size, latent_dim) predicted next latent mean
        """
        x = torch.cat([latent, action], dim=-1)
        h = self.transition(x)
        mean, _ = torch.chunk(h, 2, dim=-1)
        return mean





