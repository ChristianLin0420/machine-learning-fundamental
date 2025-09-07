import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from typing import Tuple, Optional, List
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class DQN(nn.Module):
    """
    Deep Q-Network architecture.
    
    A neural network that approximates the Q-function Q(s,a) for discrete action spaces.
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = [128, 128]):
        """
        Initialize DQN.
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            hidden_sizes: List of hidden layer sizes
        """
        super(DQN, self).__init__()
        
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
        
        # Output layer
        layers.append(nn.Linear(prev_size, action_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture.
    
    Separates the Q-value into value V(s) and advantage A(s,a) components:
    Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_sizes: List[int] = [128, 128]):
        """
        Initialize Dueling DQN.
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            hidden_sizes: List of hidden layer sizes
        """
        super(DuelingDQN, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Shared feature layers
        shared_layers = []
        prev_size = state_size
        
        for hidden_size in hidden_sizes:
            shared_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size
        
        self.shared_network = nn.Sequential(*shared_layers)
        
        # Value stream
        self.value_stream = nn.Linear(prev_size, 1)
        
        # Advantage stream
        self.advantage_stream = nn.Linear(prev_size, action_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the dueling network."""
        features = self.shared_network(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values


class DQNAgent:
    """
    Deep Q-Network Agent with Experience Replay and Target Network.
    
    Implements the DQN algorithm with the following key components:
    - Neural network function approximation
    - Experience replay buffer
    - Target network for stability
    - Epsilon-greedy exploration
    """
    
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 buffer_size: int = 10000,
                 batch_size: int = 64,
                 target_update_freq: int = 100,
                 device: str = 'cpu',
                 use_dueling: bool = False,
                 use_double: bool = False,
                 use_prioritized: bool = False):
        """
        Initialize DQN Agent.
        
        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Rate of epsilon decay
            epsilon_min: Minimum epsilon value
            buffer_size: Size of experience replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates
            device: Device to run on ('cpu' or 'cuda')
            use_dueling: Whether to use Dueling DQN
            use_double: Whether to use Double DQN
            use_prioritized: Whether to use prioritized experience replay
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)
        self.use_dueling = use_dueling
        self.use_double = use_double
        self.use_prioritized = use_prioritized
        
        # Initialize networks
        if use_dueling:
            self.q_network = DuelingDQN(state_size, action_size).to(self.device)
            self.target_network = DuelingDQN(state_size, action_size).to(self.device)
        else:
            self.q_network = DQN(state_size, action_size).to(self.device)
            self.target_network = DQN(state_size, action_size).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        if use_prioritized:
            self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        else:
            self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training statistics
        self.training_step = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.epsilon_history = []
        
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode (affects exploration)
            
        Returns:
            Selected action
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def step(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """
        Take a step and potentially train the agent.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is finished
        """
        # Store experience in replay buffer
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        # Train if buffer has enough samples
        if self.replay_buffer.is_ready(self.batch_size):
            self._train()
    
    def _train(self):
        """Train the agent on a batch of experiences."""
        # Sample batch from replay buffer
        if self.use_prioritized:
            batch = self.replay_buffer.sample(self.batch_size)
            states, actions, rewards, next_states, dones, indices, weights = batch
        else:
            batch = self.replay_buffer.sample(self.batch_size)
            states, actions, rewards, next_states, dones = batch
            weights = None
            indices = None
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        if weights is not None:
            weights = weights.to(self.device)
        
        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            if self.use_double:
                # Double DQN: use online network to select actions, target network to evaluate
                next_actions = self.q_network(next_states).argmax(1)
                next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            else:
                # Standard DQN: use target network for both selection and evaluation
                next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))
        
        # Compute loss
        if self.use_prioritized and weights is not None:
            td_errors = (current_q_values - target_q_values).squeeze()
            loss = (weights * F.mse_loss(current_q_values, target_q_values, reduction='none').squeeze()).mean()
        else:
            loss = F.mse_loss(current_q_values, target_q_values)
        
        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update priorities if using prioritized replay
        if self.use_prioritized and indices is not None:
            priorities = (torch.abs(td_errors) + 1e-6).detach().cpu().numpy()
            self.replay_buffer.update_priorities(indices, priorities)
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Store loss
        self.losses.append(loss.item())
    
    def train_episode(self, env, max_steps: int = 1000) -> Tuple[float, int]:
        """
        Train for one episode.
        
        Args:
            env: Environment to train on
            max_steps: Maximum steps per episode
            
        Returns:
            Total reward and episode length
        """
        state = env.reset()
        if hasattr(state, '__len__') and len(state) > 1:
            state = state[0] if isinstance(state, tuple) else state
        
        total_reward = 0
        steps = 0
        
        for _ in range(max_steps):
            # Select action
            action = self.act(state, training=True)
            
            # Take action
            result = env.step(action)
            if len(result) == 4:
                next_state, reward, done, info = result
            else:
                next_state, reward, done, truncated, info = result
                done = done or truncated
            
            # Store experience and train
            self.step(state, action, reward, next_state, done)
            
            # Update statistics
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        # Record episode statistics
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(steps)
        self.epsilon_history.append(self.epsilon)
        
        return total_reward, steps
    
    def train(self, env, num_episodes: int, max_steps: int = 1000, 
              verbose: bool = True) -> dict:
        """
        Train the agent for multiple episodes.
        
        Args:
            env: Environment to train on
            num_episodes: Number of episodes to train
            max_steps: Maximum steps per episode
            verbose: Whether to print progress
            
        Returns:
            Training statistics
        """
        print(f"Training DQN agent for {num_episodes} episodes...")
        print(f"Device: {self.device}")
        print(f"Network: {'Dueling' if self.use_dueling else 'Standard'} DQN")
        print(f"Double DQN: {self.use_double}")
        print(f"Prioritized Replay: {self.use_prioritized}")
        print()
        
        for episode in range(num_episodes):
            reward, steps = self.train_episode(env, max_steps)
            
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_length = np.mean(self.episode_lengths[-100:])
                avg_loss = np.mean(self.losses[-100:]) if self.losses else 0
                
                print(f"Episode {episode + 1}, "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Avg Length: {avg_length:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}, "
                      f"Avg Loss: {avg_loss:.4f}")
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses,
            'epsilon_history': self.epsilon_history
        }
    
    def evaluate(self, env, num_episodes: int = 100, max_steps: int = 1000) -> dict:
        """
        Evaluate the trained agent.
        
        Args:
            env: Environment to evaluate on
            num_episodes: Number of episodes to evaluate
            max_steps: Maximum steps per episode
            
        Returns:
            Evaluation statistics
        """
        eval_rewards = []
        eval_lengths = []
        
        for episode in range(num_episodes):
            state = env.reset()
            if hasattr(state, '__len__') and len(state) > 1:
                state = state[0] if isinstance(state, tuple) else state
            
            total_reward = 0
            steps = 0
            
            for _ in range(max_steps):
                action = self.act(state, training=False)
                result = env.step(action)
                
                if len(result) == 4:
                    next_state, reward, done, info = result
                else:
                    next_state, reward, done, truncated, info = result
                    done = done or truncated
                
                total_reward += reward
                steps += 1
                state = next_state
                
                if done:
                    break
            
            eval_rewards.append(total_reward)
            eval_lengths.append(steps)
        
        return {
            'eval_rewards': eval_rewards,
            'eval_lengths': eval_lengths,
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'success_rate': np.mean([r > 0 for r in eval_rewards])
        }
    
    def save(self, filepath: str):
        """Save the agent's state."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses,
            'epsilon_history': self.epsilon_history
        }, filepath)
    
    def load(self, filepath: str):
        """Load the agent's state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        self.losses = checkpoint['losses']
        self.epsilon_history = checkpoint['epsilon_history']


if __name__ == "__main__":
    # Test DQN agent
    print("Testing DQN Agent...")
    
    # Create a simple test environment
    state_size = 4
    action_size = 2
    
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=1e-3,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        buffer_size=1000,
        batch_size=32
    )
    
    print(f"Agent created with {sum(p.numel() for p in agent.q_network.parameters())} parameters")
    print(f"Q-network: {agent.q_network}")
    print(f"Target network: {agent.target_network}")
    
    # Test action selection
    state = np.random.randn(state_size)
    action = agent.act(state, training=True)
    print(f"Action selected: {action}")
    
    print("DQN Agent test completed!")
