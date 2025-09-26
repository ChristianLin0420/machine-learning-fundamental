import torch
import numpy as np
from collections import deque


class ReplayBuffer:
    """
    Replay buffer for storing and sampling transitions.
    """
    
    def __init__(self, capacity, state_dim, action_dim, device='cpu'):
        self.capacity = capacity
        self.device = device
        
        # Storage for transitions
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.bool, device=device)
        
        self.ptr = 0
        self.size = 0
        
    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.
        
        Args:
            state: (state_dim,) current state
            action: (action_dim,) action taken
            reward: (1,) reward received
            next_state: (state_dim,) next state
            done: (1,) whether episode ended
        """
        self.states[self.ptr] = torch.tensor(state, dtype=torch.float32, device=self.device)
        self.actions[self.ptr] = torch.tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[self.ptr] = torch.tensor(reward, dtype=torch.float32, device=self.device)
        self.next_states[self.ptr] = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        self.dones[self.ptr] = torch.tensor(done, dtype=torch.bool, device=self.device)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size):
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: number of transitions to sample
            
        Returns:
            batch: dict containing batched transitions
        """
        if self.size < batch_size:
            batch_size = self.size
            
        indices = np.random.randint(0, self.size, size=batch_size)
        
        batch = {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_states': self.next_states[indices],
            'dones': self.dones[indices]
        }
        
        return batch
    
    def __len__(self):
        return self.size


class ImaginaryBuffer:
    """
    Buffer for storing imaginary rollouts from the world model.
    """
    
    def __init__(self, capacity, latent_dim, action_dim, device='cpu'):
        self.capacity = capacity
        self.device = device
        
        # Storage for imaginary transitions
        self.latents = torch.zeros((capacity, latent_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_latents = torch.zeros((capacity, latent_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.bool, device=device)
        
        self.ptr = 0
        self.size = 0
        
    def add(self, latent, action, reward, next_latent, done):
        """
        Add an imaginary transition to the buffer.
        
        Args:
            latent: (latent_dim,) current latent state
            action: (action_dim,) action taken
            reward: (1,) predicted reward
            next_latent: (latent_dim,) next latent state
            done: (1,) whether episode ended
        """
        self.latents[self.ptr] = latent
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_latents[self.ptr] = next_latent
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size):
        """
        Sample a batch of imaginary transitions.
        
        Args:
            batch_size: number of transitions to sample
            
        Returns:
            batch: dict containing batched imaginary transitions
        """
        if self.size < batch_size:
            batch_size = self.size
            
        indices = np.random.randint(0, self.size, size=batch_size)
        
        batch = {
            'latents': self.latents[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_latents': self.next_latents[indices],
            'dones': self.dones[indices]
        }
        
        return batch
    
    def __len__(self):
        return self.size



