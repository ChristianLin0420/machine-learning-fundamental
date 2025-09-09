import torch
import numpy as np
import gym
from typing import List, Tuple, Dict, Optional, Union
from collections import deque
import warnings


class RolloutCollector:
    """
    Collects rollouts for Actor-Critic training.
    
    This class handles the collection of trajectories from the environment
    using the current policy, and prepares the data for training.
    """
    
    def __init__(self, env, max_rollout_length: int = 2048,
                 device: str = 'cpu', normalize_advantages: bool = True):
        """
        Initialize Rollout Collector.
        
        Args:
            env: Environment to collect rollouts from
            max_rollout_length: Maximum length of each rollout
            device: Device to run on
            normalize_advantages: Whether to normalize advantages
        """
        self.env = env
        self.max_rollout_length = max_rollout_length
        self.device = torch.device(device)
        self.normalize_advantages = normalize_advantages
        
        # Storage for rollout data
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = []
        self.returns = []
        
        # Episode tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
        # Statistics
        self.total_steps = 0
        self.total_episodes = 0
        
    def reset(self):
        """Reset the collector for new rollouts."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = []
        self.returns = []
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
        self.total_steps = 0
        self.total_episodes = 0
    
    def collect_rollout(self, policy_net, value_net, gamma: float = 0.99,
                       lam: float = 0.95, max_steps: Optional[int] = None) -> Dict:
        """
        Collect a single rollout from the environment.
        
        Args:
            policy_net: Policy network for action selection
            value_net: Value network for value estimation
            gamma: Discount factor
            lam: GAE parameter
            max_steps: Maximum steps to collect (overrides max_rollout_length)
            
        Returns:
            Dictionary containing rollout data and statistics
        """
        if max_steps is None:
            max_steps = self.max_rollout_length
        
        # Reset environment
        state = self.env.reset()
        if hasattr(state, '__len__') and len(state) > 1:
            state = state[0] if isinstance(state, tuple) else state
        
        # Reset episode tracking
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
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
            
            # Store experience
            self.states.append(state)
            self.actions.append(action.item())
            self.rewards.append(reward)
            self.values.append(value.item())
            self.log_probs.append(log_prob.item())
            self.dones.append(done)
            
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
                
                # Reset environment
                state = self.env.reset()
                if hasattr(state, '__len__') and len(state) > 1:
                    state = state[0] if isinstance(state, tuple) else state
            else:
                state = next_state
        
        # Compute advantages and returns
        self._compute_advantages_and_returns(gamma, lam)
        
        # Prepare rollout data
        rollout_data = {
            'states': torch.FloatTensor(np.array(self.states)).to(self.device),
            'actions': torch.LongTensor(self.actions).to(self.device),
            'rewards': torch.FloatTensor(self.rewards).to(self.device),
            'values': torch.FloatTensor(self.values).to(self.device),
            'log_probs': torch.FloatTensor(self.log_probs).to(self.device),
            'dones': torch.BoolTensor(self.dones).to(self.device),
            'advantages': torch.FloatTensor(self.advantages).to(self.device),
            'returns': torch.FloatTensor(self.returns).to(self.device),
        }
        
        # Statistics
        stats = {
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'episode_rewards': self.episode_rewards.copy(),
            'episode_lengths': self.episode_lengths.copy(),
            'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'std_reward': np.std(self.episode_rewards) if self.episode_rewards else 0,
            'mean_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'std_length': np.std(self.episode_lengths) if self.episode_lengths else 0,
        }
        
        return rollout_data, stats
    
    def _compute_advantages_and_returns(self, gamma: float, lam: float):
        """Compute GAE advantages and returns."""
        from gae import compute_gae_advantages, normalize_advantages
        
        # Compute advantages and returns
        advantages, returns = compute_gae_advantages(
            self.rewards, self.values, self.values[1:] + [0.0], self.dones, gamma, lam
        )
        
        # Normalize advantages if requested
        if self.normalize_advantages:
            advantages = normalize_advantages(advantages)
        
        self.advantages = advantages
        self.returns = returns
    
    def get_rollout_data(self) -> Dict:
        """Get the collected rollout data."""
        return {
            'states': torch.FloatTensor(np.array(self.states)).to(self.device),
            'actions': torch.LongTensor(self.actions).to(self.device),
            'rewards': torch.FloatTensor(self.rewards).to(self.device),
            'values': torch.FloatTensor(self.values).to(self.device),
            'log_probs': torch.FloatTensor(self.log_probs).to(self.device),
            'dones': torch.BoolTensor(self.dones).to(self.device),
            'advantages': torch.FloatTensor(self.advantages).to(self.device),
            'returns': torch.FloatTensor(self.returns).to(self.device),
        }
    
    def get_statistics(self) -> Dict:
        """Get rollout statistics."""
        return {
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'episode_rewards': self.episode_rewards.copy(),
            'episode_lengths': self.episode_lengths.copy(),
            'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'std_reward': np.std(self.episode_rewards) if self.episode_rewards else 0,
            'mean_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'std_length': np.std(self.episode_lengths) if self.episode_lengths else 0,
        }


class ParallelRolloutCollector:
    """
    Collects rollouts from multiple environments in parallel.
    
    This class handles parallel rollout collection for more efficient
    data gathering, especially useful for A2C training.
    """
    
    def __init__(self, envs, max_rollout_length: int = 2048,
                 device: str = 'cpu', normalize_advantages: bool = True):
        """
        Initialize Parallel Rollout Collector.
        
        Args:
            envs: List of environments
            max_rollout_length: Maximum length of each rollout
            device: Device to run on
            normalize_advantages: Whether to normalize advantages
        """
        self.envs = envs
        self.num_envs = len(envs)
        self.max_rollout_length = max_rollout_length
        self.device = torch.device(device)
        self.normalize_advantages = normalize_advantages
        
        # Storage for rollout data
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = []
        self.returns = []
        
        # Episode tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_rewards = [0.0] * self.num_envs
        self.current_episode_lengths = [0] * self.num_envs
        
        # Statistics
        self.total_steps = 0
        self.total_episodes = 0
        
        # Environment states
        self.env_states = [None] * self.num_envs
        self.env_dones = [True] * self.num_envs
        
    def reset(self):
        """Reset the collector for new rollouts."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = []
        self.returns = []
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_rewards = [0.0] * self.num_envs
        self.current_episode_lengths = [0] * self.num_envs
        
        self.total_steps = 0
        self.total_episodes = 0
        
        self.env_states = [None] * self.num_envs
        self.env_dones = [True] * self.num_envs
    
    def collect_rollout(self, policy_net, value_net, gamma: float = 0.99,
                       lam: float = 0.95, max_steps: Optional[int] = None) -> Dict:
        """
        Collect rollouts from multiple environments in parallel.
        
        Args:
            policy_net: Policy network for action selection
            value_net: Value network for value estimation
            gamma: Discount factor
            lam: GAE parameter
            max_steps: Maximum steps to collect
            
        Returns:
            Dictionary containing rollout data and statistics
        """
        if max_steps is None:
            max_steps = self.max_rollout_length
        
        # Reset environments if needed
        for i in range(self.num_envs):
            if self.env_dones[i]:
                state = self.envs[i].reset()
                if hasattr(state, '__len__') and len(state) > 1:
                    state = state[0] if isinstance(state, tuple) else state
                self.env_states[i] = state
                self.env_dones[i] = False
                self.current_episode_rewards[i] = 0.0
                self.current_episode_lengths[i] = 0
        
        # Collect rollouts
        for step in range(max_steps):
            # Prepare batch of states
            states_batch = []
            for i in range(self.num_envs):
                if not self.env_dones[i]:
                    states_batch.append(self.env_states[i])
                else:
                    states_batch.append(np.zeros_like(self.env_states[i]))
            
            states_tensor = torch.FloatTensor(np.array(states_batch)).to(self.device)
            
            # Get actions and values
            with torch.no_grad():
                actions, log_probs, values = policy_net.get_action(states_tensor, deterministic=False)
                if value_net is not None:
                    values = value_net(states_tensor)
            
            # Take actions in all environments
            for i in range(self.num_envs):
                if not self.env_dones[i]:
                    result = self.envs[i].step(actions[i].item())
                    if len(result) == 4:
                        next_state, reward, done, info = result
                    else:
                        next_state, reward, done, truncated, info = result
                        done = done or truncated
                    
                    # Store experience
                    self.states.append(self.env_states[i])
                    self.actions.append(actions[i].item())
                    self.rewards.append(reward)
                    self.values.append(values[i].item())
                    self.log_probs.append(log_probs[i].item())
                    self.dones.append(done)
                    
                    # Update episode tracking
                    self.current_episode_rewards[i] += reward
                    self.current_episode_lengths[i] += 1
                    self.total_steps += 1
                    
                    # Check if episode is done
                    if done:
                        self.episode_rewards.append(self.current_episode_rewards[i])
                        self.episode_lengths.append(self.current_episode_lengths[i])
                        self.total_episodes += 1
                        
                        # Reset for next episode
                        self.current_episode_rewards[i] = 0.0
                        self.current_episode_lengths[i] = 0
                        
                        # Reset environment
                        state = self.envs[i].reset()
                        if hasattr(state, '__len__') and len(state) > 1:
                            state = state[0] if isinstance(state, tuple) else state
                        self.env_states[i] = state
                        self.env_dones[i] = False
                    else:
                        self.env_states[i] = next_state
                        self.env_dones[i] = done
        
        # Compute advantages and returns
        self._compute_advantages_and_returns(gamma, lam)
        
        # Prepare rollout data
        rollout_data = {
            'states': torch.FloatTensor(np.array(self.states)).to(self.device),
            'actions': torch.LongTensor(self.actions).to(self.device),
            'rewards': torch.FloatTensor(self.rewards).to(self.device),
            'values': torch.FloatTensor(self.values).to(self.device),
            'log_probs': torch.FloatTensor(self.log_probs).to(self.device),
            'dones': torch.BoolTensor(self.dones).to(self.device),
            'advantages': torch.FloatTensor(self.advantages).to(self.device),
            'returns': torch.FloatTensor(self.returns).to(self.device),
        }
        
        # Statistics
        stats = {
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'episode_rewards': self.episode_rewards.copy(),
            'episode_lengths': self.episode_lengths.copy(),
            'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'std_reward': np.std(self.episode_rewards) if self.episode_rewards else 0,
            'mean_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'std_length': np.std(self.episode_lengths) if self.episode_lengths else 0,
        }
        
        return rollout_data, stats
    
    def _compute_advantages_and_returns(self, gamma: float, lam: float):
        """Compute GAE advantages and returns."""
        from gae import compute_gae_advantages, normalize_advantages
        
        # Compute advantages and returns
        advantages, returns = compute_gae_advantages(
            self.rewards, self.values, self.values[1:] + [0.0], self.dones, gamma, lam
        )
        
        # Normalize advantages if requested
        if self.normalize_advantages:
            advantages = normalize_advantages(advantages)
        
        self.advantages = advantages
        self.returns = returns
    
    def get_rollout_data(self) -> Dict:
        """Get the collected rollout data."""
        return {
            'states': torch.FloatTensor(np.array(self.states)).to(self.device),
            'actions': torch.LongTensor(self.actions).to(self.device),
            'rewards': torch.FloatTensor(self.rewards).to(self.device),
            'values': torch.FloatTensor(self.values).to(self.device),
            'log_probs': torch.FloatTensor(self.log_probs).to(self.device),
            'dones': torch.BoolTensor(self.dones).to(self.device),
            'advantages': torch.FloatTensor(self.advantages).to(self.device),
            'returns': torch.FloatTensor(self.returns).to(self.device),
        }
    
    def get_statistics(self) -> Dict:
        """Get rollout statistics."""
        return {
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'episode_rewards': self.episode_rewards.copy(),
            'episode_lengths': self.episode_lengths.copy(),
            'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'std_reward': np.std(self.episode_rewards) if self.episode_rewards else 0,
            'mean_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'std_length': np.std(self.episode_lengths) if self.episode_lengths else 0,
        }


def create_rollout_collector(env, max_rollout_length: int = 2048,
                           device: str = 'cpu', normalize_advantages: bool = True,
                           parallel: bool = False) -> Union[RolloutCollector, ParallelRolloutCollector]:
    """
    Create a rollout collector.
    
    Args:
        env: Environment or list of environments
        max_rollout_length: Maximum length of each rollout
        device: Device to run on
        normalize_advantages: Whether to normalize advantages
        parallel: Whether to use parallel collection
        
    Returns:
        Rollout collector instance
    """
    if parallel and isinstance(env, list):
        return ParallelRolloutCollector(env, max_rollout_length, device, normalize_advantages)
    else:
        return RolloutCollector(env, max_rollout_length, device, normalize_advantages)


if __name__ == "__main__":
    # Test rollout collector
    print("Testing Rollout Collector...")
    
    # Create test environment
    env = gym.make('CartPole-v1')
    
    # Create test networks
    from nets import create_actor_critic_network
    policy_net = create_actor_critic_network(4, 2, [64, 64])
    value_net = policy_net  # Shared network
    
    # Create collector
    collector = RolloutCollector(env, max_rollout_length=100, device='cpu')
    
    # Collect rollout
    rollout_data, stats = collector.collect_rollout(policy_net, value_net)
    
    print(f"Collected {len(rollout_data['states'])} steps")
    print(f"Mean reward: {stats['mean_reward']:.2f}")
    print(f"Mean length: {stats['mean_length']:.2f}")
    
    env.close()
    print("Rollout collector test completed!")
