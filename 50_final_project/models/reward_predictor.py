import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardPredictor(nn.Module):
    """
    Reward predictor that estimates reward given latent state and action.
    
    Implements: r_t ~ p_φ(r|z_t, a_t)
    """
    
    def __init__(self, latent_dim, action_dim, hidden_dim=128):
        super(RewardPredictor, self).__init__()
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        
        # Reward prediction network
        self.reward_net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Single reward value
        )
        
    def forward(self, latent, action):
        """
        Predict reward for given latent state and action.
        
        Args:
            latent: (batch_size, latent_dim) latent state
            action: (batch_size, action_dim) action taken
            
        Returns:
            reward: (batch_size, 1) predicted reward
        """
        # Concatenate latent and action
        x = torch.cat([latent, action], dim=-1)
        reward = self.reward_net(x)
        return reward




