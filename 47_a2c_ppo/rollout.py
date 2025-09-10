import torch
import numpy as np
import gym
from typing import List, Tuple, Dict, Optional, Union
from collections import deque
import warnings


class PPORolloutBuffer:
    """
    Rollout buffer for PPO training that stores experiences and computes advantages.
    
    This buffer is specifically designed for PPO and stores all necessary data
    for the clipped objective, including old log probabilities for ratio computation.
    """
    
    def __init__(self, buffer_size: int, state_size: int, action_size: int,
                 device: str = 'cpu', gamma: float = 0.99, lam: float = 0.95):
        """
        Initialize PPO Rollout Buffer.
        
        Args:
            buffer_size: Maximum size of the buffer
            state_size: Dimension of state space
            action_size: Number of possible actions
            device: Device to run on
            gamma: Discount factor
            lam: GAE parameter
        """
        self.buffer_size = buffer_size
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device(device)
        self.gamma = gamma
        self.lam = lam
        
        # Storage tensors
        self.states = torch.zeros((buffer_size, state_size), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((buffer_size,), dtype=torch.long, device=self.device)
        self.rewards = torch.zeros((buffer_size,), dtype=torch.float32, device=self.device)
        self.values = torch.zeros((buffer_size,), dtype=torch.float32, device=self.device)
        self.log_probs = torch.zeros((buffer_size,), dtype=torch.float32, device=self.device)
        self.advantages = torch.zeros((buffer_size,), dtype=torch.float32, device=self.device)
        self.returns = torch.zeros((buffer_size,), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((buffer_size,), dtype=torch.bool, device=self.device)
        
        # Buffer management
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = buffer_size
        
        # Episode tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
        # Statistics
        self.total_steps = 0
        self.total_episodes = 0
    
    def add(self, state: np.ndarray, action: int, reward: float, value: float,
            log_prob: float, done: bool):
        """
        Add a single experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            value: Value estimate
            log_prob: Log probability of action
            done: Whether episode is done
        """
        assert self.ptr < self.max_size, "Buffer overflow"
        
        # Store experience
        self.states[self.ptr] = torch.FloatTensor(state).to(self.device)
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        # Update episode tracking
        self.current_episode_reward += reward
        self.current_episode_length += 1
        self.total_steps += 1
        
        # Check if episode is done
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.total_episodes += 1
            
            # Reset for next episode
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        self.ptr += 1
    
    def finish_path(self, last_value: float = 0.0):
        """
        Finish the current path and compute advantages.
        
        Args:
            last_value: Value estimate for the last state (0 if terminal)
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = torch.cat([self.rewards[path_slice], torch.tensor([last_value], device=self.device)])
        values = torch.cat([self.values[path_slice], torch.tensor([last_value], device=self.device)])
        dones = torch.cat([self.dones[path_slice], torch.tensor([True], device=self.device)])
        
        # Compute GAE advantages
        advantages, returns = self._compute_gae(rewards, values, dones)
        
        # Store advantages and returns
        self.advantages[path_slice] = advantages
        self.returns[path_slice] = returns
        
        # Update path start index
        self.path_start_idx = self.ptr
    
    def _compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, 
                    dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and returns.
        
        Args:
            rewards: Reward tensor
            values: Value tensor
            dones: Done tensor
            
        Returns:
            Tuple of (advantages, returns)
        """
        # Compute TD errors
        td_errors = rewards[:-1] + self.gamma * values[1:] * (1 - dones[:-1].float()) - values[:-1]
        
        # Compute GAE advantages (backwards)
        advantages = torch.zeros_like(td_errors)
        gae_advantage = 0.0
        
        for t in reversed(range(len(td_errors))):
            if dones[t]:
                gae_advantage = td_errors[t]
            else:
                gae_advantage = td_errors[t] + self.gamma * self.lam * gae_advantage
            
            advantages[t] = gae_advantage
        
        # Compute returns
        returns = advantages + values[:-1]
        
        return advantages, returns
    
    def get(self) -> Dict[str, torch.Tensor]:
        """
        Get all data from the buffer.
        
        Returns:
            Dictionary containing all buffer data
        """
        assert self.ptr == self.max_size, "Buffer not full"
        
        # Normalize advantages
        advantages = self.advantages.clone()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return {
            'states': self.states,
            'actions': self.actions,
            'rewards': self.rewards,
            'values': self.values,
            'log_probs': self.log_probs,
            'advantages': advantages,
            'returns': self.returns,
            'dones': self.dones
        }
    
    def get_minibatch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Get a random minibatch from the buffer.
        
        Args:
            batch_size: Size of the minibatch
            
        Returns:
            Dictionary containing minibatch data
        """
        indices = torch.randperm(self.ptr, device=self.device)[:batch_size]
        
        # Normalize advantages
        advantages = self.advantages.clone()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'values': self.values[indices],
            'log_probs': self.log_probs[indices],
            'advantages': advantages[indices],
            'returns': self.returns[indices],
            'dones': self.dones[indices]
        }
    
    def clear(self):
        """Clear the buffer."""
        self.ptr = 0
        self.path_start_idx = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.total_steps = 0
        self.total_episodes = 0
    
    def get_statistics(self) -> Dict:
        """Get buffer statistics."""
        return {
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'episode_rewards': self.episode_rewards.copy(),
            'episode_lengths': self.episode_lengths.copy(),
            'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'std_reward': np.std(self.episode_rewards) if self.episode_rewards else 0,
            'mean_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'std_length': np.std(self.episode_lengths) if self.episode_lengths else 0,
            'buffer_size': self.ptr,
            'buffer_capacity': self.max_size
        }


class RolloutCollector:
    """
    Collects rollouts for PPO training using the PPORolloutBuffer.
    
    This class handles the collection of trajectories from the environment
    using the current policy, and prepares the data for PPO training.
    """
    
    def __init__(self, env, buffer_size: int = 2048, device: str = 'cpu',
                 gamma: float = 0.99, lam: float = 0.95):
        """
        Initialize Rollout Collector.
        
        Args:
            env: Environment to collect rollouts from
            buffer_size: Size of the rollout buffer
            device: Device to run on
            gamma: Discount factor
            lam: GAE parameter
        """
        self.env = env
        self.device = torch.device(device)
        
        # Get environment dimensions
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        
        # Create rollout buffer
        self.buffer = PPORolloutBuffer(
            buffer_size, state_size, action_size, device, gamma, lam
        )
        
        # Statistics
        self.total_steps = 0
        self.total_episodes = 0
    
    def collect_rollout(self, policy_net, value_net, max_steps: Optional[int] = None) -> Tuple[Dict, Dict]:
        """
        Collect a single rollout from the environment.
        
        Args:
            policy_net: Policy network for action selection
            value_net: Value network for value estimation
            max_steps: Maximum steps to collect
            
        Returns:
            Tuple of (rollout_data, statistics)
        """
        if max_steps is None:
            max_steps = self.buffer.max_size
        
        # Reset environment
        state = self.env.reset()
        if hasattr(state, '__len__') and len(state) > 1:
            state = state[0] if isinstance(state, tuple) else state
        
        # Reset buffer
        self.buffer.clear()
        
        # Collect rollout
        for step in range(max_steps):
            # Get action and value from networks
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action, log_prob, value = policy_net.get_action(state_tensor, deterministic=False)
                if value_net is not None and value_net != policy_net:
                    value = value_net(state_tensor)
                elif hasattr(value, 'item'):
                    value = value
                else:
                    value = value.item() if hasattr(value, 'item') else value
            
            # Take action
            result = self.env.step(action.item())
            if len(result) == 4:
                next_state, reward, done, info = result
            else:
                next_state, reward, done, truncated, info = result
                done = done or truncated
            
            # Add experience to buffer
            self.buffer.add(state, action.item(), reward, value.item(), log_prob.item(), done)
            
            # Update statistics
            self.total_steps += 1
            if done:
                self.total_episodes += 1
            
            # Check if episode is done
            if done:
                # Finish the path
                self.buffer.finish_path(last_value=0.0)
                
                # Reset environment
                state = self.env.reset()
                if hasattr(state, '__len__') and len(state) > 1:
                    state = state[0] if isinstance(state, tuple) else state
            else:
                state = next_state
        
        # Finish the last path if not done
        if not done:
            # Get value for the last state
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, _, last_value = policy_net.get_action(state_tensor, deterministic=False)
                if value_net is not None and value_net != policy_net:
                    last_value = value_net(state_tensor)
                elif hasattr(last_value, 'item'):
                    last_value = last_value
                else:
                    last_value = last_value.item() if hasattr(last_value, 'item') else last_value
            
            self.buffer.finish_path(last_value=last_value)
        
        # Get rollout data
        rollout_data = self.buffer.get()
        
        # Get statistics
        stats = self.buffer.get_statistics()
        stats.update({
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes
        })
        
        return rollout_data, stats
    
    def get_minibatch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Get a random minibatch from the buffer.
        
        Args:
            batch_size: Size of the minibatch
            
        Returns:
            Dictionary containing minibatch data
        """
        return self.buffer.get_minibatch(batch_size)
    
    def get_statistics(self) -> Dict:
        """Get collector statistics."""
        return self.buffer.get_statistics()


def create_rollout_collector(env, buffer_size: int = 2048, device: str = 'cpu',
                           gamma: float = 0.99, lam: float = 0.95) -> RolloutCollector:
    """
    Create a rollout collector.
    
    Args:
        env: Environment to collect rollouts from
        buffer_size: Size of the rollout buffer
        device: Device to run on
        gamma: Discount factor
        lam: GAE parameter
        
    Returns:
        Rollout collector instance
    """
    return RolloutCollector(env, buffer_size, device, gamma, lam)


if __name__ == "__main__":
    # Test rollout collector
    print("Testing PPO Rollout Collector...")
    
    # Create test environment
    env = gym.make('CartPole-v1')
    
    # Create test networks
    from nets import create_actor_critic_network
    policy_net = create_actor_critic_network(4, 2, [64, 64])
    value_net = policy_net  # Shared network
    
    # Create collector
    collector = RolloutCollector(env, buffer_size=100, device='cpu')
    
    # Collect rollout
    rollout_data, stats = collector.collect_rollout(policy_net, value_net)
    
    print(f"Collected {len(rollout_data['states'])} steps")
    print(f"Mean reward: {stats['mean_reward']:.2f}")
    print(f"Mean length: {stats['mean_length']:.2f}")
    
    # Test minibatch
    minibatch = collector.get_minibatch(32)
    print(f"Minibatch states shape: {minibatch['states'].shape}")
    print(f"Minibatch actions shape: {minibatch['actions'].shape}")
    
    env.close()
    print("PPO Rollout collector test completed!")
