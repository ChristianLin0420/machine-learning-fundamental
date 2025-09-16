import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.encoder import Encoder
from models.transition import TransitionModel
from models.reward_predictor import RewardPredictor


class WorldModel(nn.Module):
    """
    Combined world model consisting of encoder, transition model, and reward predictor.
    """
    
    def __init__(self, state_dim, action_dim, latent_dim=32, hidden_dim=128, device='cpu'):
        super(WorldModel, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.device = device
        
        # World model components
        self.encoder = Encoder(state_dim, latent_dim, hidden_dim).to(device)
        self.transition = TransitionModel(latent_dim, action_dim, hidden_dim).to(device)
        self.reward_predictor = RewardPredictor(latent_dim, action_dim, hidden_dim).to(device)
        
        # Optimizers
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=3e-4)
        self.transition_optimizer = optim.Adam(self.transition.parameters(), lr=3e-4)
        self.reward_optimizer = optim.Adam(self.reward_predictor.parameters(), lr=3e-4)
        
    def forward(self, state, action):
        """
        Forward pass through the world model.
        
        Args:
            state: (batch_size, state_dim) current state
            action: (batch_size, action_dim) action taken
            
        Returns:
            next_latent: (batch_size, latent_dim) predicted next latent
            reward: (batch_size, 1) predicted reward
            latent: (batch_size, latent_dim) current latent
        """
        # Encode current state
        latent, _, _ = self.encoder(state)
        
        # Predict next latent and reward
        next_latent, _, _ = self.transition(latent, action)
        reward = self.reward_predictor(latent, action)
        
        return next_latent, reward, latent
    
    def encode(self, state):
        """
        Encode state to latent representation.
        
        Args:
            state: (batch_size, state_dim) state tensor
            
        Returns:
            latent: (batch_size, latent_dim) latent representation
        """
        latent, _, _ = self.encoder(state)
        return latent
    
    def predict_next_latent(self, latent, action):
        """
        Predict next latent state.
        
        Args:
            latent: (batch_size, latent_dim) current latent
            action: (batch_size, action_dim) action taken
            
        Returns:
            next_latent: (batch_size, latent_dim) predicted next latent
        """
        next_latent, _, _ = self.transition(latent, action)
        return next_latent
    
    def predict_reward(self, latent, action):
        """
        Predict reward for given latent and action.
        
        Args:
            latent: (batch_size, latent_dim) latent state
            action: (batch_size, action_dim) action taken
            
        Returns:
            reward: (batch_size, 1) predicted reward
        """
        return self.reward_predictor(latent, action)
    
    def train_step(self, batch):
        """
        Perform one training step on the world model.
        
        Args:
            batch: dict containing transitions from replay buffer
            
        Returns:
            losses: dict containing individual losses
        """
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        
        # Encode states
        latent, latent_mean, latent_log_std = self.encoder(states)
        next_latent, next_latent_mean, next_latent_log_std = self.encoder(next_states)
        
        # Predict next latent
        pred_next_latent, pred_next_latent_mean, pred_next_latent_log_std = self.transition(latent, actions)
        
        # Predict reward
        pred_reward = self.reward_predictor(latent, actions)
        
        # Compute losses
        # KL divergence loss for encoder (regularization)
        kl_loss = 0.5 * torch.mean(torch.sum(latent_log_std**2 + latent_mean**2 - 1 - 2*latent_log_std, dim=-1))
        
        # Transition loss (MSE between predicted and actual next latent)
        transition_loss = F.mse_loss(pred_next_latent_mean, next_latent_mean.detach())
        
        # Reward loss (MSE between predicted and actual reward)
        reward_loss = F.mse_loss(pred_reward, rewards)
        
        # Total loss
        total_loss = kl_loss + transition_loss + reward_loss
        
        # Backward pass
        self.encoder_optimizer.zero_grad()
        self.transition_optimizer.zero_grad()
        self.reward_optimizer.zero_grad()
        
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.transition.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.reward_predictor.parameters(), 1.0)
        
        self.encoder_optimizer.step()
        self.transition_optimizer.step()
        self.reward_optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'kl_loss': kl_loss.item(),
            'transition_loss': transition_loss.item(),
            'reward_loss': reward_loss.item()
        }
    
    def save(self, filepath):
        """Save the world model."""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'transition_state_dict': self.transition.state_dict(),
            'reward_predictor_state_dict': self.reward_predictor.state_dict(),
            'encoder_optimizer_state_dict': self.encoder_optimizer.state_dict(),
            'transition_optimizer_state_dict': self.transition_optimizer.state_dict(),
            'reward_optimizer_state_dict': self.reward_optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath):
        """Load the world model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.transition.load_state_dict(checkpoint['transition_state_dict'])
        self.reward_predictor.load_state_dict(checkpoint['reward_predictor_state_dict'])
        self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer_state_dict'])
        self.transition_optimizer.load_state_dict(checkpoint['transition_optimizer_state_dict'])
        self.reward_optimizer.load_state_dict(checkpoint['reward_optimizer_state_dict'])
