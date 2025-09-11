"""
PettingZoo MPE Environment Wrapper for IPPO/MAPPO

This module provides wrappers for PettingZoo MPE environments to work with
IPPO and MAPPO algorithms. It handles vectorized rollouts and provides
proper interfaces for multi-agent training.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
import gymnasium as gym
from pettingzoo.mpe import simple_spread_v3, simple_tag_v3, simple_adversary_v3
from supersuit import pad_observations_v0, pad_action_space_v0
import warnings
warnings.filterwarnings('ignore')


class MPEEnvWrapper:
    """
    Wrapper for PettingZoo MPE environments to work with IPPO/MAPPO.
    
    This wrapper handles:
    - Environment reset and step
    - Action and observation space management
    - Episode tracking and statistics
    - Vectorized rollout support
    """
    
    def __init__(self, env_name: str = "simple_spread_v3", 
                 num_agents: int = 3, max_cycles: int = 25,
                 continuous_actions: bool = False, render_mode: Optional[str] = None):
        """
        Initialize MPE Environment Wrapper.
        
        Args:
            env_name: Name of the MPE environment
            num_agents: Number of agents in the environment
            max_cycles: Maximum number of cycles per episode
            continuous_actions: Whether to use continuous actions
            render_mode: Rendering mode ('human', 'rgb_array', None)
        """
        self.env_name = env_name
        self.num_agents = num_agents
        self.max_cycles = max_cycles
        self.continuous_actions = continuous_actions
        self.render_mode = render_mode
        
        # Create environment
        self._create_env()
        
        # Environment state
        self.episode_rewards = {agent: 0.0 for agent in self.env.possible_agents}
        self.episode_lengths = {agent: 0 for agent in self.env.possible_agents}
        self.total_episodes = 0
        self.total_steps = 0
        
        # Statistics
        self.episode_reward_history = []
        self.episode_length_history = []
        self.success_history = []
        
    def _create_env(self):
        """Create the PettingZoo environment."""
        if self.env_name == "simple_spread_v3":
            # For simple_spread_v3, N is the number of agents
            self.env = simple_spread_v3.parallel_env(
                N=self.num_agents,
                local_ratio=0.5,
                max_cycles=self.max_cycles,
                continuous_actions=self.continuous_actions,
                render_mode=self.render_mode
            )
            # Debug: check what we actually got
            print(f"Environment possible agents after creation: {list(self.env.possible_agents)}")
            print(f"Expected agents: {self.num_agents}")
        elif self.env_name == "simple_tag_v3":
            self.env = simple_tag_v3.parallel_env(
                num_good=self.num_agents - 1,
                num_adversaries=1,
                num_obstacles=2,
                max_cycles=self.max_cycles,
                continuous_actions=self.continuous_actions,
                render_mode=self.render_mode
            )
        elif self.env_name == "simple_adversary_v3":
            self.env = simple_adversary_v3.parallel_env(
                N=self.num_agents,
                max_cycles=self.max_cycles,
                continuous_actions=self.continuous_actions,
                render_mode=self.render_mode
            )
        else:
            raise ValueError(f"Unknown environment: {self.env_name}")
        
        # Apply wrappers (skip normalization for infinite bounds)
        # self.env = normalize_obs_v0(self.env)  # Skip due to infinite bounds
        self.env = pad_observations_v0(self.env)
        self.env = pad_action_space_v0(self.env)
        
        # Get agent names
        self.agent_names = list(self.env.possible_agents)
        self.num_agents = len(self.agent_names)
        
        # Get observation and action spaces
        self.obs_space = self.env.observation_space(self.agent_names[0])
        self.action_space = self.env.action_space(self.agent_names[0])
        
        print(f"Environment: {self.env_name}")
        print(f"Number of agents: {self.num_agents}")
        print(f"Observation space: {self.obs_space}")
        print(f"Action space: {self.action_space}")
        print(f"Agent names: {self.agent_names}")
    
    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset the environment.
        
        Returns:
            Dictionary mapping agent names to initial observations
        """
        observations = self.env.reset()
        # Convert tuple to dictionary if needed
        if isinstance(observations, tuple):
            # Get the actual agent names from the environment
            actual_agent_names = list(self.env.possible_agents)
            # Handle case where tuple has fewer elements than agents
            if len(observations) < len(actual_agent_names):
                print(f"Warning: Environment returned {len(observations)} observations for {len(actual_agent_names)} agents")
                # Pad with zeros if needed
                padded_observations = list(observations)
                while len(padded_observations) < len(actual_agent_names):
                    padded_observations.append(np.zeros(self.obs_space.shape[0]))
                observations = {agent: obs for agent, obs in zip(actual_agent_names, padded_observations)}
            else:
                observations = {agent: obs for agent, obs in zip(actual_agent_names, observations)}
        self.episode_rewards = {agent: 0.0 for agent in self.agent_names}
        self.episode_lengths = {agent: 0 for agent in self.agent_names}
        return observations
    
    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], 
                                                     Dict[str, float], 
                                                     Dict[str, bool], 
                                                     Dict[str, bool], 
                                                     Dict[str, Dict]]:
        """
        Take a step in the environment.
        
        Args:
            actions: Dictionary mapping agent names to actions
            
        Returns:
            Tuple of (observations, rewards, dones, truncateds, infos)
        """
        observations, rewards, dones, truncateds, infos = self.env.step(actions)
        
        # Convert tuple to dictionary if needed
        if isinstance(observations, tuple):
            # Get the actual agent names from the environment
            actual_agent_names = list(self.env.possible_agents)
            observations = {agent: obs for agent, obs in zip(actual_agent_names, observations)}
        if isinstance(rewards, tuple):
            actual_agent_names = list(self.env.possible_agents)
            rewards = {agent: rew for agent, rew in zip(actual_agent_names, rewards)}
        if isinstance(dones, tuple):
            actual_agent_names = list(self.env.possible_agents)
            dones = {agent: done for agent, done in zip(actual_agent_names, dones)}
        if isinstance(truncateds, tuple):
            actual_agent_names = list(self.env.possible_agents)
            truncateds = {agent: trunc for agent, trunc in zip(actual_agent_names, truncateds)}
        if isinstance(infos, tuple):
            actual_agent_names = list(self.env.possible_agents)
            infos = {agent: info for agent, info in zip(actual_agent_names, infos)}
        
        # Update episode statistics
        for agent in self.agent_names:
            self.episode_rewards[agent] += rewards[agent]
            self.episode_lengths[agent] += 1
            self.total_steps += 1
        
        # Check if episode is done
        if all(dones.values()) or all(truncateds.values()):
            self._finish_episode()
        
        return observations, rewards, dones, truncateds, infos
    
    def _finish_episode(self):
        """Finish the current episode and update statistics."""
        # Record episode statistics
        episode_reward = sum(self.episode_rewards.values())
        episode_length = max(self.episode_lengths.values())
        
        self.episode_reward_history.append(episode_reward)
        self.episode_length_history.append(episode_length)
        
        # Compute success metric (environment-specific)
        success = self._compute_success()
        self.success_history.append(success)
        
        self.total_episodes += 1
        
        # Reset episode statistics
        self.episode_rewards = {agent: 0.0 for agent in self.agent_names}
        self.episode_lengths = {agent: 0 for agent in self.agent_names}
    
    def _compute_success(self) -> float:
        """
        Compute success metric for the episode.
        
        Returns:
            Success rate (0.0 to 1.0)
        """
        if self.env_name == "simple_spread_v3":
            # Success: all agents reach different landmarks
            # This is a simplified success metric
            return 1.0 if max(self.episode_rewards.values()) > 0 else 0.0
        elif self.env_name == "simple_tag_v3":
            # Success: good agents avoid being tagged
            return 1.0 if max(self.episode_rewards.values()) > 0 else 0.0
        elif self.env_name == "simple_adversary_v3":
            # Success: good agents reach the target
            return 1.0 if max(self.episode_rewards.values()) > 0 else 0.0
        else:
            return 0.0
    
    def get_agent_observations(self, observations: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """
        Get observations as a list ordered by agent names.
        
        Args:
            observations: Dictionary of observations
            
        Returns:
            List of observations ordered by agent names
        """
        # Handle both dictionary and tuple formats
        if isinstance(observations, tuple):
            return list(observations)
        else:
            # Use the stored agent names to maintain order
            obs_list = []
            for agent in self.agent_names:
                if agent in observations:
                    obs = observations[agent]
                    # Ensure observation is a numpy array
                    if isinstance(obs, dict):
                        # If it's a dict, try to extract the observation
                        if 'observation' in obs:
                            obs = obs['observation']
                        else:
                            # If no 'observation' key, convert the dict to array
                            obs = np.array(list(obs.values())).flatten()
                    elif not isinstance(obs, np.ndarray):
                        obs = np.array(obs)
                    
                    # Ensure observation is numeric and can be converted to float32
                    try:
                        obs = obs.astype(np.float32)
                    except (TypeError, ValueError):
                        # If conversion fails, create a zero array
                        print(f"Warning: Could not convert observation to float32, using zeros")
                        obs = np.zeros(self.obs_space.shape[0], dtype=np.float32)
                    
                    # Ensure observation has the correct shape
                    if obs.shape[0] != self.obs_space.shape[0]:
                        # If observation is too large, take the first part
                        if obs.shape[0] > self.obs_space.shape[0]:
                            obs = obs[:self.obs_space.shape[0]]
                        # If observation is too small, pad with zeros
                        elif obs.shape[0] < self.obs_space.shape[0]:
                            padding = np.zeros(self.obs_space.shape[0] - obs.shape[0], dtype=np.float32)
                            obs = np.concatenate([obs, padding])
                    
                    obs_list.append(obs)
                else:
                    print(f"Warning: Agent {agent} not found in observations")
                    obs_list.append(np.zeros(self.obs_space.shape[0]))
            return obs_list
    
    def get_agent_actions(self, actions: List[int]) -> Dict[str, int]:
        """
        Convert list of actions to dictionary format.
        
        Args:
            actions: List of actions
            
        Returns:
            Dictionary mapping agent names to actions
        """
        return {agent: actions[i] for i, agent in enumerate(self.agent_names)}
    
    def get_global_state(self, observations: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Get global state by concatenating all agent observations.
        
        Args:
            observations: Dictionary of observations
            
        Returns:
            Concatenated global state
        """
        # Handle both dictionary and tuple formats
        if isinstance(observations, tuple):
            return np.concatenate(list(observations))
        else:
            # Ensure all observations are 1D arrays
            obs_list = []
            for agent in self.agent_names:
                if agent in observations:
                    obs = observations[agent]
                    # Ensure observation is a numpy array
                    if isinstance(obs, dict):
                        if 'observation' in obs:
                            obs = obs['observation']
                        else:
                            obs = np.array(list(obs.values())).flatten()
                    elif not isinstance(obs, np.ndarray):
                        obs = np.array(obs)
                    
                    # Ensure observation is 1D
                    if obs.ndim == 0:
                        obs = np.array([obs])
                    elif obs.ndim > 1:
                        obs = obs.flatten()
                    
                    # Ensure observation has the correct data type
                    try:
                        obs = obs.astype(np.float32)
                    except (TypeError, ValueError):
                        obs = np.zeros(self.obs_space.shape[0], dtype=np.float32)
                    
                    # Ensure observation has the correct shape
                    if obs.shape[0] != self.obs_space.shape[0]:
                        if obs.shape[0] > self.obs_space.shape[0]:
                            obs = obs[:self.obs_space.shape[0]]
                        elif obs.shape[0] < self.obs_space.shape[0]:
                            padding = np.zeros(self.obs_space.shape[0] - obs.shape[0], dtype=np.float32)
                            obs = np.concatenate([obs, padding])
                    
                    obs_list.append(obs)
                else:
                    obs_list.append(np.zeros(self.obs_space.shape[0], dtype=np.float32))
            
            return np.concatenate(obs_list)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get environment statistics.
        
        Returns:
            Dictionary of statistics
        """
        if not self.episode_reward_history:
            return {
                'total_episodes': 0,
                'total_steps': 0,
                'mean_episode_reward': 0.0,
                'std_episode_reward': 0.0,
                'mean_episode_length': 0.0,
                'mean_success_rate': 0.0
            }
        
        return {
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'mean_episode_reward': np.mean(self.episode_reward_history),
            'std_episode_reward': np.std(self.episode_reward_history),
            'mean_episode_length': np.mean(self.episode_length_history),
            'mean_success_rate': np.mean(self.success_history),
            'recent_mean_reward': np.mean(self.episode_reward_history[-100:]) if len(self.episode_reward_history) >= 100 else np.mean(self.episode_reward_history),
            'recent_success_rate': np.mean(self.success_history[-100:]) if len(self.success_history) >= 100 else np.mean(self.success_history)
        }
    
    def render(self, mode: str = 'human'):
        """Render the environment."""
        return self.env.render(mode)
    
    def close(self):
        """Close the environment."""
        self.env.close()


class VectorizedMPEEnv:
    """
    Vectorized MPE environment for parallel rollout collection.
    
    This class manages multiple MPE environments for efficient
    parallel data collection in IPPO/MAPPO training.
    """
    
    def __init__(self, env_name: str = "simple_spread_v3", 
                 num_envs: int = 4, num_agents: int = 3,
                 max_cycles: int = 25, continuous_actions: bool = False):
        """
        Initialize Vectorized MPE Environment.
        
        Args:
            env_name: Name of the MPE environment
            num_envs: Number of parallel environments
            num_agents: Number of agents per environment
            max_cycles: Maximum cycles per episode
            continuous_actions: Whether to use continuous actions
        """
        self.env_name = env_name
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.max_cycles = max_cycles
        self.continuous_actions = continuous_actions
        
        # Create environments
        self.envs = []
        for i in range(num_envs):
            env = MPEEnvWrapper(
                env_name=env_name,
                num_agents=num_agents,
                max_cycles=max_cycles,
                continuous_actions=continuous_actions
            )
            self.envs.append(env)
        
        # Get observation and action spaces from first environment
        self.obs_space = self.envs[0].obs_space
        self.action_space = self.envs[0].action_space
        self.agent_names = self.envs[0].agent_names
        
        # Environment states
        self.observations = [None] * num_envs
        self.episode_rewards = [{} for _ in range(num_envs)]
        self.episode_lengths = [{} for _ in range(num_envs)]
        self.dones = [False] * num_envs
        
        print(f"Created {num_envs} parallel {env_name} environments")
        print(f"Total agents: {num_envs * num_agents}")
    
    def reset(self) -> List[Dict[str, np.ndarray]]:
        """
        Reset all environments.
        
        Returns:
            List of observation dictionaries for each environment
        """
        self.observations = []
        for i, env in enumerate(self.envs):
            obs = env.reset()
            self.observations.append(obs)
            self.episode_rewards[i] = {agent: 0.0 for agent in self.agent_names}
            self.episode_lengths[i] = {agent: 0 for agent in self.agent_names}
            self.dones[i] = False
        
        return self.observations
    
    def step(self, actions: List[Dict[str, int]]) -> Tuple[List[Dict[str, np.ndarray]], 
                                                          List[Dict[str, float]], 
                                                          List[Dict[str, bool]], 
                                                          List[Dict[str, bool]], 
                                                          List[Dict[str, Dict]]]:
        """
        Take a step in all environments.
        
        Args:
            actions: List of action dictionaries for each environment
            
        Returns:
            Tuple of (observations, rewards, dones, truncateds, infos) for each environment
        """
        all_observations = []
        all_rewards = []
        all_dones = []
        all_truncateds = []
        all_infos = []
        
        for i, (env, action_dict) in enumerate(zip(self.envs, actions)):
            if not self.dones[i]:
                obs, rewards, dones, truncateds, infos = env.step(action_dict)
                all_observations.append(obs)
                all_rewards.append(rewards)
                all_dones.append(dones)
                all_truncateds.append(truncateds)
                all_infos.append(infos)
                
                # Update episode statistics
                for agent in self.agent_names:
                    self.episode_rewards[i][agent] += rewards[agent]
                    self.episode_lengths[i][agent] += 1
                
                # Check if episode is done
                if all(dones.values()) or all(truncateds.values()):
                    self.dones[i] = True
            else:
                # Environment is done, return dummy data
                all_observations.append(self.observations[i])
                all_rewards.append({agent: 0.0 for agent in self.agent_names})
                all_dones.append({agent: True for agent in self.agent_names})
                all_truncateds.append({agent: False for agent in self.agent_names})
                all_infos.append({agent: {} for agent in self.agent_names})
        
        self.observations = all_observations
        return all_observations, all_rewards, all_dones, all_truncateds, all_infos
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from all environments.
        
        Returns:
            Dictionary of aggregated statistics
        """
        all_stats = [env.get_statistics() for env in self.envs]
        
        return {
            'total_episodes': sum(stats['total_episodes'] for stats in all_stats),
            'total_steps': sum(stats['total_steps'] for stats in all_stats),
            'mean_episode_reward': np.mean([stats['mean_episode_reward'] for stats in all_stats]),
            'std_episode_reward': np.std([stats['mean_episode_reward'] for stats in all_stats]),
            'mean_episode_length': np.mean([stats['mean_episode_length'] for stats in all_stats]),
            'mean_success_rate': np.mean([stats['mean_success_rate'] for stats in all_stats]),
            'recent_mean_reward': np.mean([stats['recent_mean_reward'] for stats in all_stats]),
            'recent_success_rate': np.mean([stats['recent_success_rate'] for stats in all_stats])
        }
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()


def create_mpe_env(env_name: str = "simple_spread_v3", 
                   num_agents: int = 3, max_cycles: int = 25,
                   continuous_actions: bool = False,
                   vectorized: bool = False, num_envs: int = 4) -> MPEEnvWrapper:
    """
    Create an MPE environment.
    
    Args:
        env_name: Name of the MPE environment
        num_agents: Number of agents
        max_cycles: Maximum cycles per episode
        continuous_actions: Whether to use continuous actions
        vectorized: Whether to create vectorized environment
        num_envs: Number of parallel environments (if vectorized)
        
    Returns:
        MPE environment wrapper
    """
    if vectorized:
        return VectorizedMPEEnv(env_name, num_envs, num_agents, max_cycles, continuous_actions)
    else:
        return MPEEnvWrapper(env_name, num_agents, max_cycles, continuous_actions)


if __name__ == "__main__":
    # Test MPE environment wrapper
    print("Testing MPE Environment Wrapper...")
    
    # Test single environment
    env = create_mpe_env("simple_spread_v3", num_agents=3, max_cycles=25)
    
    # Test reset
    observations = env.reset()
    print(f"Initial observations shape: {[obs.shape for obs in env.get_agent_observations(observations)]}")
    
    # Test step
    actions = {agent: 0 for agent in env.agent_names}
    observations, rewards, dones, truncateds, infos = env.step(actions)
    print(f"Rewards: {rewards}")
    print(f"Dones: {dones}")
    
    # Test statistics
    stats = env.get_statistics()
    print(f"Statistics: {stats}")
    
    env.close()
    print("MPE Environment test completed!")
