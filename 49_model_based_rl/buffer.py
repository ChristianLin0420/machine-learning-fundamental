"""
Replay Buffer for Model-Based RL

This module implements a replay buffer for storing and sampling transitions
for training the dynamics model.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import random
from collections import deque


class ReplayBuffer:
    """
    Replay Buffer for storing transitions.
    
    This buffer stores (state, action, next_state, reward, done) tuples
    and provides methods for sampling batches for training the dynamics model.
    """
    
    def __init__(self, capacity: int = 1000000, device: str = 'cpu'):
        """
        Initialize Replay Buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            device: Device to store tensors on
        """
        self.capacity = capacity
        self.device = torch.device(device)
        
        # Storage for transitions
        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.next_states = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)
        
        # Statistics
        self.size = 0
        self.total_transitions = 0
    
    def add(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray,
            reward: float, done: bool):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            reward: Reward received
            done: Whether episode is done
        """
        self.states.append(state.copy())
        self.actions.append(action.copy())
        self.next_states.append(next_state.copy())
        self.rewards.append(reward)
        self.dones.append(done)
        
        self.size = min(self.size + 1, self.capacity)
        self.total_transitions += 1
    
    def add_batch(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray,
                  rewards: np.ndarray, dones: np.ndarray):
        """
        Add a batch of transitions to the buffer.
        
        Args:
            states: Batch of current states
            actions: Batch of actions
            next_states: Batch of next states
            rewards: Batch of rewards
            dones: Batch of done flags
        """
        for i in range(len(states)):
            self.add(states[i], actions[i], next_states[i], rewards[i], dones[i])
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, next_states, rewards, dones)
        """
        if self.size < batch_size:
            batch_size = self.size
        
        # Sample random indices
        indices = random.sample(range(self.size), batch_size)
        
        # Get transitions
        states = torch.FloatTensor([self.states[i] for i in indices]).to(self.device)
        actions = torch.FloatTensor([self.actions[i] for i in indices]).to(self.device)
        next_states = torch.FloatTensor([self.next_states[i] for i in indices]).to(self.device)
        rewards = torch.FloatTensor([self.rewards[i] for i in indices]).to(self.device)
        dones = torch.BoolTensor([self.dones[i] for i in indices]).to(self.device)
        
        return states, actions, next_states, rewards, dones
    
    def sample_sequential(self, batch_size: int, sequence_length: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample sequential transitions for training on sequences.
        
        Args:
            batch_size: Number of sequences to sample
            sequence_length: Length of each sequence
            
        Returns:
            Tuple of (states, actions, next_states, rewards, dones)
        """
        if self.size < sequence_length:
            sequence_length = self.size
        
        # Sample random starting indices
        max_start_idx = self.size - sequence_length
        start_indices = [random.randint(0, max_start_idx) for _ in range(batch_size)]
        
        # Get sequences
        states = torch.zeros(batch_size, sequence_length, self.states[0].shape[0]).to(self.device)
        actions = torch.zeros(batch_size, sequence_length, self.actions[0].shape[0]).to(self.device)
        next_states = torch.zeros(batch_size, sequence_length, self.next_states[0].shape[0]).to(self.device)
        rewards = torch.zeros(batch_size, sequence_length).to(self.device)
        dones = torch.zeros(batch_size, sequence_length, dtype=torch.bool).to(self.device)
        
        for i, start_idx in enumerate(start_indices):
            for j in range(sequence_length):
                idx = start_idx + j
                states[i, j] = torch.FloatTensor(self.states[idx]).to(self.device)
                actions[i, j] = torch.FloatTensor(self.actions[idx]).to(self.device)
                next_states[i, j] = torch.FloatTensor(self.next_states[idx]).to(self.device)
                rewards[i, j] = self.rewards[idx]
                dones[i, j] = self.dones[idx]
        
        return states, actions, next_states, rewards, dones
    
    def get_all_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get all data in the buffer.
        
        Returns:
            Tuple of (states, actions, next_states, rewards, dones)
        """
        states = torch.FloatTensor(list(self.states)).to(self.device)
        actions = torch.FloatTensor(list(self.actions)).to(self.device)
        next_states = torch.FloatTensor(list(self.next_states)).to(self.device)
        rewards = torch.FloatTensor(list(self.rewards)).to(self.device)
        dones = torch.BoolTensor(list(self.dones)).to(self.device)
        
        return states, actions, next_states, rewards, dones
    
    def clear(self):
        """Clear the buffer."""
        self.states.clear()
        self.actions.clear()
        self.next_states.clear()
        self.rewards.clear()
        self.dones.clear()
        self.size = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get buffer statistics.
        
        Returns:
            Dictionary of statistics
        """
        if self.size == 0:
            return {
                'size': 0,
                'total_transitions': self.total_transitions,
                'mean_reward': 0.0,
                'std_reward': 0.0,
                'mean_done': 0.0
            }
        
        rewards = list(self.rewards)
        dones = list(self.dones)
        
        return {
            'size': self.size,
            'total_transitions': self.total_transitions,
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_done': np.mean(dones),
            'capacity': self.capacity,
            'utilization': self.size / self.capacity
        }


class PrioritizedReplayBuffer:
    """
    Prioritized Replay Buffer for storing transitions with priorities.
    
    This buffer stores transitions with priorities and samples them
    according to their priorities for more efficient learning.
    """
    
    def __init__(self, capacity: int = 1000000, alpha: float = 0.6, beta: float = 0.4,
                 device: str = 'cpu'):
        """
        Initialize Prioritized Replay Buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            alpha: Priority exponent
            beta: Importance sampling exponent
            device: Device to store tensors on
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.device = torch.device(device)
        
        # Storage for transitions
        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.next_states = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        
        # Statistics
        self.size = 0
        self.total_transitions = 0
        self.max_priority = 1.0
    
    def add(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray,
            reward: float, done: bool, priority: Optional[float] = None):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            reward: Reward received
            done: Whether episode is done
            priority: Priority of the transition (if None, use max priority)
        """
        if priority is None:
            priority = self.max_priority
        
        self.states.append(state.copy())
        self.actions.append(action.copy())
        self.next_states.append(next_state.copy())
        self.rewards.append(reward)
        self.dones.append(done)
        self.priorities.append(priority)
        
        self.size = min(self.size + 1, self.capacity)
        self.total_transitions += 1
        self.max_priority = max(self.max_priority, priority)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of transitions according to their priorities.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, next_states, rewards, dones, indices, weights)
        """
        if self.size < batch_size:
            batch_size = self.size
        
        # Compute sampling probabilities
        priorities = np.array(list(self.priorities))
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        
        # Compute importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Get transitions
        states = torch.FloatTensor([self.states[i] for i in indices]).to(self.device)
        actions = torch.FloatTensor([self.actions[i] for i in indices]).to(self.device)
        next_states = torch.FloatTensor([self.next_states[i] for i in indices]).to(self.device)
        rewards = torch.FloatTensor([self.rewards[i] for i in indices]).to(self.device)
        dones = torch.BoolTensor([self.dones[i] for i in indices]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        indices = torch.LongTensor(indices).to(self.device)
        
        return states, actions, next_states, rewards, dones, indices, weights
    
    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor):
        """
        Update priorities of transitions.
        
        Args:
            indices: Indices of transitions to update
            priorities: New priorities
        """
        for idx, priority in zip(indices.cpu().numpy(), priorities.cpu().numpy()):
            if idx < self.size:
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get buffer statistics.
        
        Returns:
            Dictionary of statistics
        """
        if self.size == 0:
            return {
                'size': 0,
                'total_transitions': self.total_transitions,
                'mean_reward': 0.0,
                'std_reward': 0.0,
                'mean_done': 0.0,
                'mean_priority': 0.0,
                'max_priority': 0.0
            }
        
        rewards = list(self.rewards)
        dones = list(self.dones)
        priorities = list(self.priorities)
        
        return {
            'size': self.size,
            'total_transitions': self.total_transitions,
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_done': np.mean(dones),
            'mean_priority': np.mean(priorities),
            'max_priority': self.max_priority,
            'capacity': self.capacity,
            'utilization': self.size / self.capacity
        }


def create_replay_buffer(buffer_type: str = 'standard', capacity: int = 1000000,
                        device: str = 'cpu', **kwargs) -> ReplayBuffer:
    """
    Create a replay buffer.
    
    Args:
        buffer_type: Type of buffer ('standard', 'prioritized')
        capacity: Maximum number of transitions to store
        device: Device to store tensors on
        **kwargs: Additional arguments for specific buffer types
        
    Returns:
        Replay buffer
    """
    if buffer_type == 'standard':
        return ReplayBuffer(capacity, device)
    elif buffer_type == 'prioritized':
        return PrioritizedReplayBuffer(capacity, device=device, **kwargs)
    else:
        raise ValueError(f"Unknown buffer type: {buffer_type}")


if __name__ == "__main__":
    # Test the replay buffer
    print("Testing Replay Buffer...")
    
    # Create buffer
    buffer = ReplayBuffer(capacity=1000, device='cpu')
    
    # Add some transitions
    for i in range(100):
        state = np.random.randn(4)
        action = np.random.randn(1)
        next_state = np.random.randn(4)
        reward = np.random.randn()
        done = np.random.rand() < 0.1
        
        buffer.add(state, action, next_state, reward, done)
    
    print(f"Buffer size: {buffer.size}")
    print(f"Buffer statistics: {buffer.get_statistics()}")
    
    # Test sampling
    states, actions, next_states, rewards, dones = buffer.sample(32)
    print(f"Sampled states shape: {states.shape}")
    print(f"Sampled actions shape: {actions.shape}")
    print(f"Sampled next_states shape: {next_states.shape}")
    print(f"Sampled rewards shape: {rewards.shape}")
    print(f"Sampled dones shape: {dones.shape}")
    
    # Test sequential sampling
    states_seq, actions_seq, next_states_seq, rewards_seq, dones_seq = buffer.sample_sequential(8, 10)
    print(f"Sequential states shape: {states_seq.shape}")
    print(f"Sequential actions shape: {actions_seq.shape}")
    
    print("Replay buffer test completed!")
