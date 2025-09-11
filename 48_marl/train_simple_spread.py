#!/usr/bin/env python3
"""
Multi-Agent RL Training Script for simple_spread_v3

This script implements IPPO and MAPPO-lite training on the PettingZoo
simple_spread_v3 environment. It demonstrates both independent learning
and centralized training with decentralized execution.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
import os
import warnings
warnings.filterwarnings('ignore')

from envs.mpe_ippo_env import create_mpe_env
from algs.nets import create_actor_critic_network, count_parameters
from algs.gae import compute_gae_advantages_tensor, normalize_advantages_tensor
from algs.ippo import IPPOTrainer
from algs.mappo_critic import MAPPOTrainer
from utils import set_seed, MARLLogger, early_stopping_check, moving_average


class MARLRolloutCollector:
    """
    Rollout collector for Multi-Agent RL training.
    
    This class handles the collection of trajectories from multi-agent
    environments and prepares the data for IPPO/MAPPO training.
    """
    
    def __init__(self, env, buffer_size: int = 2048, device: str = 'cpu',
                 gamma: float = 0.99, lam: float = 0.95):
        """
        Initialize MARL Rollout Collector.
        
        Args:
            env: Multi-agent environment
            buffer_size: Size of the rollout buffer
            device: Device to run on
            gamma: Discount factor
            lam: GAE parameter
        """
        self.env = env
        self.device = torch.device(device)
        self.gamma = gamma
        self.lam = lam
        self.buffer_size = buffer_size
        
        # Get environment dimensions
        self.num_agents = env.num_agents
        self.obs_dim = env.obs_space.shape[0]
        self.action_dim = env.action_space.n
        self.agent_names = env.agent_names
        
        # Update agent names based on actual environment
        self.agent_names = list(env.env.possible_agents)
        self.num_agents = len(self.agent_names)
        
        # Debug: print collector info
        print(f"Collector agent names: {self.agent_names}")
        print(f"Collector num agents: {self.num_agents}")
        
        # Storage for rollout data
        self.states = {agent: [] for agent in self.agent_names}
        self.actions = {agent: [] for agent in self.agent_names}
        self.rewards = {agent: [] for agent in self.agent_names}
        self.values = {agent: [] for agent in self.agent_names}
        self.log_probs = {agent: [] for agent in self.agent_names}
        self.dones = {agent: [] for agent in self.agent_names}
        self.advantages = {agent: [] for agent in self.agent_names}
        self.returns = {agent: [] for agent in self.agent_names}
        
        # Global state storage (for MAPPO)
        self.global_states = []
        self.global_advantages = []
        self.global_returns = []
        
        # Episode tracking
        self.episode_rewards = {agent: 0.0 for agent in self.agent_names}
        self.episode_lengths = {agent: 0 for agent in self.agent_names}
        
        # Statistics
        self.total_steps = 0
        self.total_episodes = 0
    
    def collect_rollout(self, trainer, max_steps: Optional[int] = None) -> Tuple[Dict, Dict]:
        """
        Collect a single rollout from the environment.
        
        Args:
            trainer: IPPO or MAPPO trainer
            max_steps: Maximum steps to collect
            
        Returns:
            Tuple of (rollout_data, statistics)
        """
        if max_steps is None:
            max_steps = self.buffer_size
        
        # Reset environment
        observations = self.env.reset()
        
        # Debug: check what we got
        print(f"Reset observations type: {type(observations)}")
        if isinstance(observations, dict):
            print(f"Reset observation keys: {list(observations.keys())}")
        print(f"Environment agent names: {self.env.agent_names}")
        
        # Reset episode tracking
        self.episode_rewards = {agent: 0.0 for agent in self.agent_names}
        self.episode_lengths = {agent: 0 for agent in self.agent_names}
        
        # Collect rollout
        for step in range(max_steps):
            # Get observations as list
            obs_list = self.env.get_agent_observations(observations)
            
            
            # Get actions from trainer
            if isinstance(trainer, IPPOTrainer):
                actions, log_probs, values = trainer.get_actions(obs_list, deterministic=False)
            elif isinstance(trainer, MAPPOTrainer):
                actions, log_probs, values = trainer.get_actions(obs_list, deterministic=False)
            else:
                raise ValueError(f"Unknown trainer type: {type(trainer)}")
            
            # Convert actions to dictionary format
            action_dict = self.env.get_agent_actions(actions)
            
            # Take action
            next_observations, rewards, dones, truncateds, infos = self.env.step(action_dict)
            
            # Store experience for each agent
            for i, agent in enumerate(self.agent_names):
                self.states[agent].append(obs_list[i])
                self.actions[agent].append(actions[i])
                self.rewards[agent].append(rewards[agent])
                self.values[agent].append(values[i])
                self.log_probs[agent].append(log_probs[i])
                self.dones[agent].append(dones[agent])
                
                # Update episode tracking
                self.episode_rewards[agent] += rewards[agent]
                self.episode_lengths[agent] += 1
                self.total_steps += 1
            
            # Store global state (for MAPPO)
            if isinstance(trainer, MAPPOTrainer):
                global_state = self.env.get_global_state(observations)
                self.global_states.append(global_state)
            
            # Check if episode is done
            if all(dones.values()) or all(truncateds.values()):
                self._finish_episode()
                
                # Reset environment
                observations = self.env.reset()
            else:
                observations = next_observations
        
        # Finish the last path if not done
        if not all(dones.values()) and not all(truncateds.values()):
            # Get final values
            obs_list = self.env.get_agent_observations(observations)
            if isinstance(trainer, IPPOTrainer):
                _, _, final_values = trainer.get_actions(obs_list, deterministic=False)
            elif isinstance(trainer, MAPPOTrainer):
                _, _, final_values = trainer.get_actions(obs_list, deterministic=False)
            
            # Finish paths for each agent
            for i, agent in enumerate(self.agent_names):
                self._finish_agent_path(agent, final_values[i])
        
        # Prepare rollout data
        rollout_data = self._prepare_rollout_data()
        
        # Get statistics
        stats = self._get_statistics()
        
        return rollout_data, stats
    
    def _finish_episode(self):
        """Finish the current episode and update statistics."""
        # Finish paths for each agent
        for agent in self.agent_names:
            self._finish_agent_path(agent, 0.0)  # Terminal value is 0
        
        # Compute global advantages and returns for MAPPO
        if self.global_states:
            self._compute_global_advantages()
        
        # Record episode statistics
        self.total_episodes += 1
        
        # Reset episode tracking
        self.episode_rewards = {agent: 0.0 for agent in self.agent_names}
        self.episode_lengths = {agent: 0 for agent in self.agent_names}
    
    def _finish_agent_path(self, agent: str, last_value: float = 0.0):
        """Finish the path for a specific agent."""
        if not self.states[agent]:
            return
        
        # Compute GAE advantages
        rewards = torch.tensor(self.rewards[agent], dtype=torch.float32, device=self.device)
        values = torch.tensor(self.values[agent] + [last_value], dtype=torch.float32, device=self.device)
        dones = torch.tensor(self.dones[agent] + [True], dtype=torch.bool, device=self.device)
        
        # Compute advantages and returns
        advantages, returns = compute_gae_advantages_tensor(
            rewards, values[:-1], values[1:], dones[:-1], self.gamma, self.lam
        )
        
        # Store advantages and returns
        self.advantages[agent] = advantages.cpu().numpy().tolist()
        self.returns[agent] = returns.cpu().numpy().tolist()
    
    def _compute_global_advantages(self):
        """Compute global advantages and returns for MAPPO."""
        if not self.global_states:
            return
        
        # Find the minimum length across all agent data
        min_length = min(len(self.rewards[agent]) for agent in self.agent_names)
        
        # Compute global rewards (sum of all agent rewards)
        global_rewards = []
        for i in range(min_length):
            step_rewards = [self.rewards[agent][i] for agent in self.agent_names]
            global_rewards.append(sum(step_rewards))
        
        # Compute global values (sum of all agent values)
        global_values = []
        for i in range(min_length):
            step_values = [self.values[agent][i] for agent in self.agent_names]
            global_values.append(sum(step_values))
        
        # Compute global dones (any agent is done)
        global_dones = []
        for i in range(min_length):
            step_dones = [self.dones[agent][i] for agent in self.agent_names]
            global_dones.append(any(step_dones))
        
        # Convert to tensors
        rewards_tensor = torch.tensor(global_rewards, dtype=torch.float32, device=self.device)
        values_tensor = torch.tensor(global_values + [0.0], dtype=torch.float32, device=self.device)  # Add terminal value
        dones_tensor = torch.tensor(global_dones + [True], dtype=torch.bool, device=self.device)  # Add terminal done
        
        # Compute GAE advantages
        advantages, returns = compute_gae_advantages_tensor(
            rewards_tensor, values_tensor[:-1], values_tensor[1:], dones_tensor[:-1], self.gamma, self.lam
        )
        
        # Store global advantages and returns
        self.global_advantages = advantages.cpu().numpy().tolist()
        self.global_returns = returns.cpu().numpy().tolist()
    
    def _prepare_rollout_data(self) -> Dict:
        """Prepare rollout data for training."""
        rollout_data = {}
        
        # Prepare data for each agent
        for agent in self.agent_names:
            rollout_data[agent] = {
                'states': torch.FloatTensor(np.array(self.states[agent])).to(self.device),
                'actions': torch.LongTensor(self.actions[agent]).to(self.device),
                'rewards': torch.FloatTensor(self.rewards[agent]).to(self.device),
                'values': torch.FloatTensor(self.values[agent]).to(self.device),
                'log_probs': torch.FloatTensor(self.log_probs[agent]).to(self.device),
                'dones': torch.BoolTensor(self.dones[agent]).to(self.device),
                'advantages': torch.FloatTensor(self.advantages[agent]).to(self.device),
                'returns': torch.FloatTensor(self.returns[agent]).to(self.device),
            }
        
        # Prepare global data for MAPPO
        if self.global_states:
            # Ensure all arrays have the same length
            min_length = min(len(self.global_states), len(self.global_advantages), len(self.global_returns))
            rollout_data['global'] = {
                'global_states': torch.FloatTensor(np.array(self.global_states[:min_length])).to(self.device),
                'global_advantages': torch.FloatTensor(np.array(self.global_advantages[:min_length])).to(self.device),
                'global_returns': torch.FloatTensor(np.array(self.global_returns[:min_length])).to(self.device),
            }
        
        return rollout_data
    
    def _get_statistics(self) -> Dict:
        """Get rollout statistics."""
        return {
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'episode_rewards': {agent: self.episode_rewards[agent] for agent in self.agent_names},
            'episode_lengths': {agent: self.episode_lengths[agent] for agent in self.agent_names},
            'mean_reward': np.mean([self.episode_rewards[agent] for agent in self.agent_names]),
            'std_reward': np.std([self.episode_rewards[agent] for agent in self.agent_names]),
            'mean_length': np.mean([self.episode_lengths[agent] for agent in self.agent_names]),
            'std_length': np.std([self.episode_lengths[agent] for agent in self.agent_names]),
        }
    
    def clear(self):
        """Clear the rollout buffer."""
        for agent in self.agent_names:
            self.states[agent] = []
            self.actions[agent] = []
            self.rewards[agent] = []
            self.values[agent] = []
            self.log_probs[agent] = []
            self.dones[agent] = []
            self.advantages[agent] = []
            self.returns[agent] = []
        
        self.global_states = []
        self.global_advantages = []
        self.global_returns = []
        
        self.episode_rewards = {agent: 0.0 for agent in self.agent_names}
        self.episode_lengths = {agent: 0 for agent in self.agent_names}
        self.total_steps = 0
        self.total_episodes = 0


def train_ippo_simple_spread(num_updates: int = 1000, max_rollout_length: int = 2048,
                            learning_rate: float = 3e-4, gamma: float = 0.99,
                            lam: float = 0.95, entropy_coef: float = 0.01,
                            value_coef: float = 0.5, max_grad_norm: float = 0.5,
                            clip_ratio: float = 0.2, ppo_epochs: int = 4,
                            mini_batch_size: int = 64, target_kl: float = 0.01,
                            hidden_sizes: List[int] = [256, 256], device: str = 'cpu',
                            verbose: bool = True, save_model: bool = True,
                            plot_results: bool = True) -> IPPOTrainer:
    """
    Train IPPO agent on simple_spread_v3.
    
    Args:
        num_updates: Number of training updates
        max_rollout_length: Maximum length of each rollout
        learning_rate: Learning rate
        gamma: Discount factor
        lam: GAE parameter
        entropy_coef: Entropy regularization coefficient
        value_coef: Value loss coefficient
        max_grad_norm: Maximum gradient norm for clipping
        clip_ratio: PPO clipping ratio
        ppo_epochs: Number of PPO epochs per update
        mini_batch_size: Size of mini-batches
        target_kl: Target KL divergence for early stopping
        hidden_sizes: Hidden layer sizes
        device: Device to run on
        verbose: Whether to print progress
        save_model: Whether to save the trained model
        plot_results: Whether to plot training results
        
    Returns:
        Trained IPPO agent
    """
    # Set seed for reproducibility
    set_seed(42)
    
    # Create environment
    env = create_mpe_env("simple_spread_v3", num_agents=2, max_cycles=25)
    
    print(f"Environment: simple_spread_v3")
    print(f"Number of agents: {env.num_agents}")
    print(f"Observation space: {env.obs_space}")
    print(f"Action space: {env.action_space}")
    print()
    
    # Create IPPO trainer
    trainer = IPPOTrainer(
        num_agents=len(env.agent_names),
        obs_dim=env.obs_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=learning_rate,
        gamma=gamma,
        lam=lam,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        max_grad_norm=max_grad_norm,
        clip_ratio=clip_ratio,
        ppo_epochs=ppo_epochs,
        mini_batch_size=mini_batch_size,
        target_kl=target_kl,
        hidden_sizes=hidden_sizes,
        device=device,
        shared_parameters=True
    )
    
    # Create rollout collector
    collector = MARLRolloutCollector(env, max_rollout_length, device, gamma, lam)
    
    # Create logger
    logger = MARLLogger()
    
    # Training loop
    print(f"Training IPPO agent for {num_updates} updates...")
    print()
    
    for update in range(num_updates):
        # Collect rollout
        rollout_data, rollout_stats = collector.collect_rollout(trainer)
        
        # Update network
        update_stats = trainer.update(rollout_data)
        
        # Log episode statistics
        trainer.log_episode(rollout_stats['episode_rewards'], rollout_stats['episode_lengths'])
        logger.log_episode(rollout_stats['episode_rewards'], rollout_stats['episode_lengths'])
        
        # Log update statistics
        advantages = {agent: rollout_data[agent]['advantages'].cpu().numpy().tolist() 
                     for agent in env.agent_names}
        returns = {agent: rollout_data[agent]['returns'].cpu().numpy().tolist() 
                  for agent in env.agent_names}
        logger.log_update(update_stats, advantages, returns)
        
        # Print progress
        if verbose and (update + 1) % 10 == 0:
            stats = trainer.get_statistics()
            print(f"Update {update + 1}/{num_updates}")
            print(f"  Overall Mean Reward: {stats['overall_mean_reward']:.2f} ± {stats['overall_std_reward']:.2f}")
            print(f"  Overall Mean Length: {stats['overall_mean_length']:.2f}")
            
            for agent_name in env.agent_names:
                agent_stats = update_stats[agent_name]
                print(f"  {agent_name}: Policy Loss: {agent_stats['policy_loss']:.4f}, "
                      f"Value Loss: {agent_stats['value_loss']:.4f}, "
                      f"KL Div: {agent_stats['kl_divergence']:.4f}")
            
            print()
            
            # Check for early stopping
            if early_stopping_check(trainer.episode_rewards, target_reward=0.0, window_size=100):
                print(f"Early stopping at update {update + 1}!")
                break
        
        # Clear collector
        collector.clear()
    
    # Evaluate agent
    print("\nEvaluating trained agent...")
    eval_results = evaluate_agent(trainer, env, num_episodes=100)
    
    print(f"\nEvaluation Results:")
    print(f"Overall Mean Reward: {eval_results['overall_mean_reward']:.2f} ± {eval_results['overall_std_reward']:.2f}")
    print(f"Overall Mean Length: {eval_results['overall_mean_length']:.2f}")
    
    for agent_name in env.agent_names:
        print(f"{agent_name} Mean Reward: {eval_results[f'{agent_name}_mean_reward']:.2f} ± {eval_results[f'{agent_name}_std_reward']:.2f}")
    
    # Save model
    if save_model:
        os.makedirs('models', exist_ok=True)
        model_path = 'models/ippo_simple_spread.pth'
        trainer.save_model(model_path)
        print(f"\nModel saved to: {model_path}")
    
    # Plot results
    if plot_results:
        logger.plot_training_curves(save_path='plots/ippo_training.png')
    
    env.close()
    
    return trainer


def train_mappo_simple_spread(num_updates: int = 1000, max_rollout_length: int = 2048,
                             learning_rate: float = 3e-4, gamma: float = 0.99,
                             lam: float = 0.95, entropy_coef: float = 0.01,
                             value_coef: float = 0.5, max_grad_norm: float = 0.5,
                             clip_ratio: float = 0.2, ppo_epochs: int = 4,
                             mini_batch_size: int = 64, target_kl: float = 0.01,
                             hidden_sizes: List[int] = [256, 256], device: str = 'cpu',
                             verbose: bool = True, save_model: bool = True,
                             plot_results: bool = True) -> MAPPOTrainer:
    """
    Train MAPPO agent on simple_spread_v3.
    
    Args:
        num_updates: Number of training updates
        max_rollout_length: Maximum length of each rollout
        learning_rate: Learning rate
        gamma: Discount factor
        lam: GAE parameter
        entropy_coef: Entropy regularization coefficient
        value_coef: Value loss coefficient
        max_grad_norm: Maximum gradient norm for clipping
        clip_ratio: PPO clipping ratio
        ppo_epochs: Number of PPO epochs per update
        mini_batch_size: Size of mini-batches
        target_kl: Target KL divergence for early stopping
        hidden_sizes: Hidden layer sizes
        device: Device to run on
        verbose: Whether to print progress
        save_model: Whether to save the trained model
        plot_results: Whether to plot training results
        
    Returns:
        Trained MAPPO agent
    """
    # Set seed for reproducibility
    set_seed(42)
    
    # Create environment
    env = create_mpe_env("simple_spread_v3", num_agents=2, max_cycles=25)
    
    print(f"Environment: simple_spread_v3")
    print(f"Number of agents: {env.num_agents}")
    print(f"Observation space: {env.obs_space}")
    print(f"Action space: {env.action_space}")
    print(f"Global obs dim: {env.obs_space.shape[0] * env.num_agents}")
    print()
    
    # Create MAPPO trainer
    trainer = MAPPOTrainer(
        num_agents=len(env.agent_names),
        obs_dim=env.obs_space.shape[0],
        action_dim=env.action_space.n,
        global_obs_dim=env.obs_space.shape[0] * len(env.agent_names),
        learning_rate=learning_rate,
        gamma=gamma,
        lam=lam,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        max_grad_norm=max_grad_norm,
        clip_ratio=clip_ratio,
        ppo_epochs=ppo_epochs,
        mini_batch_size=mini_batch_size,
        target_kl=target_kl,
        hidden_sizes=hidden_sizes,
        device=device
    )
    
    # Create rollout collector
    collector = MARLRolloutCollector(env, max_rollout_length, device, gamma, lam)
    
    # Create logger
    logger = MARLLogger()
    
    # Training loop
    print(f"Training MAPPO agent for {num_updates} updates...")
    print()
    
    for update in range(num_updates):
        # Collect rollout
        rollout_data, rollout_stats = collector.collect_rollout(trainer)
        
        # Update network
        update_stats = trainer.update(rollout_data, rollout_data['global'])
        
        # Log episode statistics
        trainer.log_episode(rollout_stats['episode_rewards'], rollout_stats['episode_lengths'])
        logger.log_episode(rollout_stats['episode_rewards'], rollout_stats['episode_lengths'])
        
        # Log update statistics
        advantages = {agent: rollout_data[agent]['advantages'].cpu().numpy().tolist() 
                     for agent in env.agent_names}
        returns = {agent: rollout_data[agent]['returns'].cpu().numpy().tolist() 
                  for agent in env.agent_names}
        logger.log_update(update_stats, advantages, returns)
        
        # Print progress
        if verbose and (update + 1) % 10 == 0:
            stats = trainer.get_statistics()
            print(f"Update {update + 1}/{num_updates}")
            print(f"  Overall Mean Reward: {stats['overall_mean_reward']:.2f} ± {stats['overall_std_reward']:.2f}")
            print(f"  Overall Mean Length: {stats['overall_mean_length']:.2f}")
            
            for agent_name in env.agent_names:
                agent_stats = update_stats[agent_name]
                print(f"  {agent_name}: Policy Loss: {agent_stats['policy_loss']:.4f}, "
                      f"Value Loss: {agent_stats['value_loss']:.4f}, "
                      f"KL Div: {agent_stats['kl_divergence']:.4f}")
            
            print()
            
            # Check for early stopping
            if early_stopping_check(trainer.episode_rewards, target_reward=0.0, window_size=100):
                print(f"Early stopping at update {update + 1}!")
                break
        
        # Clear collector
        collector.clear()
    
    # Evaluate agent
    print("\nEvaluating trained agent...")
    eval_results = evaluate_agent(trainer, env, num_episodes=100)
    
    print(f"\nEvaluation Results:")
    print(f"Overall Mean Reward: {eval_results['overall_mean_reward']:.2f} ± {eval_results['overall_std_reward']:.2f}")
    print(f"Overall Mean Length: {eval_results['overall_mean_length']:.2f}")
    
    for agent_name in env.agent_names:
        print(f"{agent_name} Mean Reward: {eval_results[f'{agent_name}_mean_reward']:.2f} ± {eval_results[f'{agent_name}_std_reward']:.2f}")
    
    # Save model
    if save_model:
        os.makedirs('models', exist_ok=True)
        model_path = 'models/mappo_simple_spread.pth'
        trainer.save_model(model_path)
        print(f"\nModel saved to: {model_path}")
    
    # Plot results
    if plot_results:
        logger.plot_training_curves(save_path='plots/mappo_training.png')
    
    env.close()
    
    return trainer


def evaluate_agent(trainer, env, num_episodes: int = 100) -> Dict[str, float]:
    """
    Evaluate the trained agent.
    
    Args:
        trainer: Trained IPPO or MAPPO agent
        env: Environment to evaluate on
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Evaluation statistics
    """
    eval_rewards = {agent: [] for agent in env.agent_names}
    eval_lengths = {agent: [] for agent in env.agent_names}
    
    for episode in range(num_episodes):
        observations = env.reset()
        episode_rewards = {agent: 0.0 for agent in env.agent_names}
        episode_lengths = {agent: 0 for agent in env.agent_names}
        
        for step in range(env.max_cycles):
            # Get observations as list
            obs_list = env.get_agent_observations(observations)
            
            # Get actions from trainer
            if isinstance(trainer, IPPOTrainer):
                actions, _, _ = trainer.get_actions(obs_list, deterministic=True)
            elif isinstance(trainer, MAPPOTrainer):
                actions, _, _ = trainer.get_actions(obs_list, deterministic=True)
            else:
                raise ValueError(f"Unknown trainer type: {type(trainer)}")
            
            # Convert actions to dictionary format
            action_dict = env.get_agent_actions(actions)
            
            # Take action
            next_observations, rewards, dones, truncateds, infos = env.step(action_dict)
            
            # Update episode statistics
            for i, agent in enumerate(env.agent_names):
                episode_rewards[agent] += rewards[agent]
                episode_lengths[agent] += 1
            
            # Check if episode is done
            if all(dones.values()) or all(truncateds.values()):
                break
            
            observations = next_observations
        
        # Store episode results
        for agent in env.agent_names:
            eval_rewards[agent].append(episode_rewards[agent])
            eval_lengths[agent].append(episode_lengths[agent])
    
    # Compute statistics
    results = {}
    
    # Overall statistics
    all_rewards = [reward for rewards in eval_rewards.values() for reward in rewards]
    all_lengths = [length for lengths in eval_lengths.values() for length in lengths]
    
    results['overall_mean_reward'] = np.mean(all_rewards)
    results['overall_std_reward'] = np.std(all_rewards)
    results['overall_mean_length'] = np.mean(all_lengths)
    results['overall_std_length'] = np.std(all_lengths)
    
    # Per-agent statistics
    for agent in env.agent_names:
        results[f'{agent}_mean_reward'] = np.mean(eval_rewards[agent])
        results[f'{agent}_std_reward'] = np.std(eval_rewards[agent])
        results[f'{agent}_mean_length'] = np.mean(eval_lengths[agent])
        results[f'{agent}_std_length'] = np.std(eval_lengths[agent])
    
    return results


def main():
    """Main training function."""
    print("Multi-Agent RL Training on simple_spread_v3")
    print("=" * 60)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Choose algorithm
    algorithm = input("Choose algorithm (ippo/mappo): ").lower().strip()
    
    if algorithm == 'ippo':
        print("\nTraining IPPO...")
        trainer = train_ippo_simple_spread(
            num_updates=100,
            max_rollout_length=2048,
            learning_rate=3e-4,
            gamma=0.99,
            lam=0.95,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.5,
            clip_ratio=0.2,
            ppo_epochs=4,
            mini_batch_size=64,
            target_kl=0.01,
            hidden_sizes=[256, 256],
            device='cpu',
            verbose=True,
            save_model=True,
            plot_results=True
        )
    elif algorithm == 'mappo':
        print("\nTraining MAPPO...")
        trainer = train_mappo_simple_spread(
            num_updates=100,
            max_rollout_length=2048,
            learning_rate=3e-4,
            gamma=0.99,
            lam=0.95,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.5,
            clip_ratio=0.2,
            ppo_epochs=4,
            mini_batch_size=64,
            target_kl=0.01,
            hidden_sizes=[256, 256],
            device='cpu',
            verbose=True,
            save_model=True,
            plot_results=True
        )
    else:
        print("Invalid algorithm choice. Please choose 'ippo' or 'mappo'.")
        return
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("Key Insights:")
    print("1. Multi-agent RL requires coordination between agents")
    print("2. IPPO: Independent learning with shared parameters")
    print("3. MAPPO: Centralized training with decentralized execution")
    print("4. GAE provides low-variance advantage estimates")
    print("5. PPO ensures stable policy updates")
    print("\nCheck the 'plots' directory for visualizations!")


if __name__ == "__main__":
    main()
