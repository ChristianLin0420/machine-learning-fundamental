import numpy as np
import random
from collections import deque
from typing import Tuple, Optional, List
import torch


class ReplayBuffer:
    """
    Experience Replay Buffer for DQN.
    
    Stores transitions (state, action, reward, next_state, done) and provides
    random sampling for training. This breaks the correlation between consecutive
    experiences and stabilizes training.
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is finished
        """
        # Convert to tensors if they aren't already
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.FloatTensor(next_state)
        
        transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough transitions in buffer. Have {len(self.buffer)}, need {batch_size}")
        
        batch = random.sample(self.buffer, batch_size)
        
        # Unpack and stack tensors
        states = torch.stack([transition[0] for transition in batch])
        actions = torch.LongTensor([transition[1] for transition in batch])
        rewards = torch.FloatTensor([transition[2] for transition in batch])
        next_states = torch.stack([transition[3] for transition in batch])
        dones = torch.BoolTensor([transition[4] for transition in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return current number of transitions in buffer."""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough transitions for sampling."""
        return len(self.buffer) >= batch_size
    
    def clear(self):
        """Clear all transitions from buffer."""
        self.buffer.clear()
    
    def get_stats(self) -> dict:
        """Get statistics about the buffer."""
        if len(self.buffer) == 0:
            return {"size": 0, "capacity": self.capacity, "utilization": 0.0}
        
        rewards = [transition[2] for transition in self.buffer]
        return {
            "size": len(self.buffer),
            "capacity": self.capacity,
            "utilization": len(self.buffer) / self.capacity,
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards)
        }


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    
    Samples transitions with probability proportional to their TD error,
    giving higher priority to more "surprising" transitions.
    """
    
    def __init__(self, capacity: int = 10000, alpha: float = 0.6, beta: float = 0.4):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling correction exponent
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def push(self, state, action, reward, next_state, done, priority: Optional[float] = None):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is finished
            priority: Priority of this transition (if None, uses max priority)
        """
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.FloatTensor(next_state)
        
        transition = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        
        # Set priority
        if priority is None:
            priority = self.max_priority
        
        self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of transitions with prioritization.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights)
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough transitions in buffer. Have {len(self.buffer)}, need {batch_size}")
        
        # Calculate sampling probabilities
        valid_priorities = self.priorities[:len(self.buffer)]
        probabilities = valid_priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        
        # Get transitions
        batch = [self.buffer[idx] for idx in indices]
        
        # Unpack and stack tensors
        states = torch.stack([transition[0] for transition in batch])
        actions = torch.LongTensor([transition[1] for transition in batch])
        rewards = torch.FloatTensor([transition[2] for transition in batch])
        next_states = torch.stack([transition[3] for transition in batch])
        dones = torch.BoolTensor([transition[4] for transition in batch])
        weights = torch.FloatTensor(weights)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        Update priorities for given indices.
        
        Args:
            indices: Indices to update
            priorities: New priorities
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        """Return current number of transitions in buffer."""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough transitions for sampling."""
        return len(self.buffer) >= batch_size
    
    def clear(self):
        """Clear all transitions from buffer."""
        self.buffer.clear()
        self.priorities.fill(0)
        self.max_priority = 1.0


class NStepReplayBuffer:
    """
    N-Step Experience Replay Buffer.
    
    Stores n-step transitions for more stable learning, especially useful
    for environments with sparse rewards.
    """
    
    def __init__(self, capacity: int = 10000, n_steps: int = 3, gamma: float = 0.99):
        """
        Initialize n-step replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            n_steps: Number of steps to look ahead
            gamma: Discount factor
        """
        self.capacity = capacity
        self.n_steps = n_steps
        self.gamma = gamma
        self.buffer = deque(maxlen=capacity)
        self.n_step_buffer = deque(maxlen=n_steps)
    
    def push(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is finished
        """
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.FloatTensor(next_state)
        
        # Add to n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # If buffer is full or episode is done, compute n-step return
        if len(self.n_step_buffer) == self.n_steps or done:
            n_step_transition = self._compute_n_step_return()
            if n_step_transition is not None:
                self.buffer.append(n_step_transition)
            
            # Remove oldest transition
            if len(self.n_step_buffer) == self.n_steps:
                self.n_step_buffer.popleft()
    
    def _compute_n_step_return(self) -> Optional[Tuple]:
        """Compute n-step return for the oldest transition."""
        if len(self.n_step_buffer) == 0:
            return None
        
        # Get the first transition
        first_state, first_action, first_reward, _, _ = self.n_step_buffer[0]
        
        # Compute n-step return
        n_step_reward = 0
        n_step_next_state = None
        n_step_done = False
        
        for i, (_, _, reward, next_state, done) in enumerate(self.n_step_buffer):
            n_step_reward += (self.gamma ** i) * reward
            if done:
                n_step_done = True
                break
            n_step_next_state = next_state
        
        return (first_state, first_action, n_step_reward, n_step_next_state, n_step_done)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of n-step transitions."""
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough transitions in buffer. Have {len(self.buffer)}, need {batch_size}")
        
        batch = random.sample(self.buffer, batch_size)
        
        # Unpack and stack tensors
        states = torch.stack([transition[0] for transition in batch])
        actions = torch.LongTensor([transition[1] for transition in batch])
        rewards = torch.FloatTensor([transition[2] for transition in batch])
        next_states = torch.stack([transition[3] for transition in batch])
        dones = torch.BoolTensor([transition[4] for transition in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return current number of transitions in buffer."""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough transitions for sampling."""
        return len(self.buffer) >= batch_size
    
    def clear(self):
        """Clear all transitions from buffer."""
        self.buffer.clear()
        self.n_step_buffer.clear()


if __name__ == "__main__":
    # Test the replay buffer
    print("Testing ReplayBuffer...")
    
    buffer = ReplayBuffer(capacity=1000)
    
    # Add some dummy transitions
    for i in range(100):
        state = np.random.randn(4)
        action = np.random.randint(0, 2)
        reward = np.random.randn()
        next_state = np.random.randn(4)
        done = np.random.choice([True, False])
        
        buffer.push(state, action, reward, next_state, done)
    
    print(f"Buffer size: {len(buffer)}")
    print(f"Buffer stats: {buffer.get_stats()}")
    
    # Test sampling
    batch = buffer.sample(32)
    print(f"Batch shapes: {[tensor.shape for tensor in batch]}")
    
    print("ReplayBuffer test completed!")
