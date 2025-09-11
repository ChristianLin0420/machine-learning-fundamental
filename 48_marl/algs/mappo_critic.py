"""
MAPPO-lite Centralized Critic Implementation

This module implements MAPPO with a centralized critic that uses global state
information during training but still allows decentralized execution.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

from .nets import MAPPOActorCritic, count_parameters
from .gae import compute_gae_advantages_tensor, normalize_advantages_tensor


class MAPPOTrainer:
    """
    MAPPO (Multi-Agent PPO) Trainer with Centralized Critic.
    
    This implementation uses individual actor networks for each agent but
    a centralized critic that has access to global state information during training.
    """
    
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int, global_obs_dim: int,
                 learning_rate: float = 3e-4, gamma: float = 0.99,
                 lam: float = 0.95, entropy_coef: float = 0.01,
                 value_coef: float = 0.5, max_grad_norm: float = 0.5,
                 clip_ratio: float = 0.2, ppo_epochs: int = 4,
                 mini_batch_size: int = 64, target_kl: float = 0.01,
                 hidden_sizes: List[int] = [256, 256], activation: str = 'relu',
                 device: str = 'cpu', normalize_advantages: bool = True):
        """
        Initialize MAPPO Trainer.
        
        Args:
            num_agents: Number of agents
            obs_dim: Dimension of local observation space
            action_dim: Number of possible actions
            global_obs_dim: Dimension of global observation space
            learning_rate: Learning rate for optimizer
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
            activation: Activation function
            device: Device to run on
            normalize_advantages: Whether to normalize advantages
        """
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.global_obs_dim = global_obs_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lam = lam
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.clip_ratio = clip_ratio
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.target_kl = target_kl
        self.device = torch.device(device)
        self.normalize_advantages = normalize_advantages
        
        # Create MAPPO actor-critic network
        self.network = MAPPOActorCritic(
            num_agents=num_agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
            global_obs_dim=global_obs_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            use_orthogonal_init=True
        ).to(self.device)
        
        # Create optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Training statistics
        self.update_count = 0
        self.episode_rewards = {f'agent_{i}': [] for i in range(num_agents)}
        self.episode_lengths = {f'agent_{i}': [] for i in range(num_agents)}
        self.kl_divergences = {f'agent_{i}': [] for i in range(num_agents)}
        self.clip_fractions = {f'agent_{i}': [] for i in range(num_agents)}
        self.explained_variances = {f'agent_{i}': [] for i in range(num_agents)}
        
        print(f"Created MAPPO Trainer:")
        print(f"  Number of agents: {num_agents}")
        print(f"  Local obs dim: {obs_dim}")
        print(f"  Global obs dim: {global_obs_dim}")
        print(f"  Total parameters: {count_parameters(self.network)}")
        print(f"  Device: {self.device}")
        print(f"  PPO epochs: {ppo_epochs}")
        print(f"  Mini-batch size: {mini_batch_size}")
    
    def get_actions(self, observations: List[np.ndarray], deterministic: bool = False) -> Tuple[List[int], List[float], List[float]]:
        """
        Get actions for all agents using local observations.
        
        Args:
            observations: List of local observations for each agent
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Tuple of (actions, log_probs, values) for each agent
        """
        actions = []
        log_probs = []
        values = []
        
        for i, obs in enumerate(observations):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action, log_prob, value = self.network.get_action(obs_tensor, i, deterministic)
            
            actions.append(action.item())
            log_probs.append(log_prob.item())
            values.append(value.item())
        
        return actions, log_probs, values
    
    def get_centralized_values(self, global_observations: np.ndarray) -> List[float]:
        """
        Get centralized value estimates using global observations.
        
        Args:
            global_observations: Global observation (concatenated)
            
        Returns:
            List of centralized value estimates for each agent
        """
        global_obs_tensor = torch.FloatTensor(global_observations).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            centralized_value = self.network.get_centralized_value(global_obs_tensor)
        
        # Return the same centralized value for all agents
        return [centralized_value.item()] * self.num_agents
    
    def compute_kl_divergence(self, old_log_probs: torch.Tensor, 
                             new_log_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between old and new policies.
        
        Args:
            old_log_probs: Old log probabilities
            new_log_probs: New log probabilities
            
        Returns:
            KL divergence
        """
        return (old_log_probs - new_log_probs).mean()
    
    def compute_clip_fraction(self, ratios: torch.Tensor) -> float:
        """
        Compute fraction of ratios that were clipped.
        
        Args:
            ratios: Policy ratios
            
        Returns:
            Clip fraction
        """
        return (torch.abs(ratios - 1.0) > self.clip_ratio).float().mean().item()
    
    def compute_explained_variance(self, values: torch.Tensor, returns: torch.Tensor) -> float:
        """
        Compute explained variance of value function.
        
        Args:
            values: Value estimates
            returns: Target returns
            
        Returns:
            Explained variance
        """
        var_y = torch.var(returns)
        return 1 - torch.var(returns - values) / var_y if var_y > 0 else 0
    
    def update(self, rollout_data: Dict[str, Dict[str, torch.Tensor]], 
               global_rollout_data: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
        """
        Update all agent networks using MAPPO.
        
        Args:
            rollout_data: Dictionary mapping agent names to local rollout data
            global_rollout_data: Dictionary containing global rollout data
            
        Returns:
            Dictionary containing update statistics for each agent
        """
        update_stats = {}
        
        # Extract global data
        global_states = global_rollout_data['global_states']
        global_advantages = global_rollout_data['global_advantages']
        global_returns = global_rollout_data['global_returns']
        
        # Normalize global advantages if requested
        if self.normalize_advantages:
            global_advantages = normalize_advantages_tensor(global_advantages)
        
        # Get buffer size
        buffer_size = global_states.size(0)
        
        # Training statistics
        policy_losses = {f'agent_{i}': [] for i in range(self.num_agents)}
        value_losses = {f'agent_{i}': [] for i in range(self.num_agents)}
        entropy_losses = {f'agent_{i}': [] for i in range(self.num_agents)}
        kl_divergences = {f'agent_{i}': [] for i in range(self.num_agents)}
        clip_fractions = {f'agent_{i}': [] for i in range(self.num_agents)}
        explained_variances = {f'agent_{i}': [] for i in range(self.num_agents)}
        
        # PPO epochs
        for epoch in range(self.ppo_epochs):
            # Shuffle data
            indices = torch.randperm(buffer_size, device=self.device)
            
            # Mini-batch training
            for start_idx in range(0, buffer_size, self.mini_batch_size):
                end_idx = min(start_idx + self.mini_batch_size, buffer_size)
                batch_indices = indices[start_idx:end_idx]
                
                # Get global mini-batch
                batch_global_states = global_states[batch_indices]
                batch_global_advantages = global_advantages[batch_indices]
                batch_global_returns = global_returns[batch_indices]
                
                # Get centralized values
                centralized_values = self.network.get_centralized_value(batch_global_states)
                
                # Compute centralized value loss
                centralized_value_loss = nn.functional.mse_loss(centralized_values, batch_global_returns)
                
                # Update each agent's policy
                total_policy_loss = 0.0
                total_entropy_loss = 0.0
                
                for agent_id in range(self.num_agents):
                    agent_name = f'agent_{agent_id}'
                    agent_data = rollout_data[agent_name]
                    
                    # Get agent mini-batch
                    batch_states = agent_data['states'][batch_indices]
                    batch_actions = agent_data['actions'][batch_indices]
                    batch_old_log_probs = agent_data['log_probs'][batch_indices]
                    
                    # Get current policy log probabilities
                    current_log_probs, _ = self.network.get_log_probs(
                        batch_states, batch_actions, agent_id
                    )
                    
                    # Compute policy ratio
                    ratio = torch.exp(current_log_probs - batch_old_log_probs)
                    
                    # Compute clipped objective
                    surr1 = ratio * batch_global_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_global_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Compute entropy loss
                    entropies = self.network.get_entropy(batch_states, agent_id)
                    entropy_loss = -entropies.mean()
                    
                    total_policy_loss += policy_loss
                    total_entropy_loss += entropy_loss
                    
                    # Store statistics
                    policy_losses[agent_name].append(policy_loss.item())
                    entropy_losses[agent_name].append(entropy_loss.item())
                    
                    # Compute KL divergence
                    kl_div = self.compute_kl_divergence(batch_old_log_probs, current_log_probs)
                    kl_divergences[agent_name].append(kl_div.item())
                    
                    # Compute clip fraction
                    clip_frac = self.compute_clip_fraction(ratio)
                    clip_fractions[agent_name].append(clip_frac)
                    
                    # Compute explained variance
                    exp_var = self.compute_explained_variance(centralized_values, batch_global_returns)
                    explained_variances[agent_name].append(exp_var.item())
                
                # Average policy and entropy losses
                total_policy_loss = total_policy_loss / self.num_agents
                total_entropy_loss = total_entropy_loss / self.num_agents
                
                # Total loss
                total_loss = total_policy_loss + self.value_coef * centralized_value_loss - self.entropy_coef * total_entropy_loss
                
                # Update network
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                
                # Update parameters
                self.optimizer.step()
                
                # Store value loss for all agents
                for agent_id in range(self.num_agents):
                    agent_name = f'agent_{agent_id}'
                    value_losses[agent_name].append(centralized_value_loss.item())
                
                # Early stopping if KL divergence is too high
                avg_kl_div = np.mean([np.mean(kl_divs) for kl_divs in kl_divergences.values()])
                if avg_kl_div > 1.5 * self.target_kl:
                    break
            
            # Early stopping if KL divergence is too high
            avg_kl_div = np.mean([np.mean(kl_divs) for kl_divs in kl_divergences.values()])
            if avg_kl_div > 1.5 * self.target_kl:
                break
        
        # Compute final statistics for each agent
        for agent_id in range(self.num_agents):
            agent_name = f'agent_{agent_id}'
            
            final_policy_loss = np.mean(policy_losses[agent_name])
            final_value_loss = np.mean(value_losses[agent_name])
            final_entropy_loss = np.mean(entropy_losses[agent_name])
            final_kl_div = np.mean(kl_divergences[agent_name])
            final_clip_frac = np.mean(clip_fractions[agent_name])
            final_exp_var = np.mean(explained_variances[agent_name])
            
            # Store statistics
            self.kl_divergences[agent_name].append(final_kl_div)
            self.clip_fractions[agent_name].append(final_clip_frac)
            self.explained_variances[agent_name].append(final_exp_var)
            
            update_stats[agent_name] = {
                'policy_loss': final_policy_loss,
                'value_loss': final_value_loss,
                'entropy_loss': final_entropy_loss,
                'kl_divergence': final_kl_div,
                'clip_fraction': final_clip_frac,
                'explained_variance': final_exp_var
            }
        
        self.update_count += 1
        return update_stats
    
    def log_episode(self, episode_rewards: Dict[str, float], episode_lengths: Dict[str, int]):
        """
        Log episode statistics.
        
        Args:
            episode_rewards: Dictionary mapping agent names to episode rewards
            episode_lengths: Dictionary mapping agent names to episode lengths
        """
        for agent_name, reward in episode_rewards.items():
            self.episode_rewards[agent_name].append(reward)
        
        for agent_name, length in episode_lengths.items():
            self.episode_lengths[agent_name].append(length)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get training statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'update_count': self.update_count,
            'total_episodes': {agent: len(rewards) for agent, rewards in self.episode_rewards.items()},
            'mean_episode_rewards': {agent: np.mean(rewards) if rewards else 0.0 for agent, rewards in self.episode_rewards.items()},
            'std_episode_rewards': {agent: np.std(rewards) if rewards else 0.0 for agent, rewards in self.episode_rewards.items()},
            'mean_episode_lengths': {agent: np.mean(lengths) if lengths else 0.0 for agent, lengths in self.episode_lengths.items()},
            'mean_kl_divergences': {agent: np.mean(kl_divs) if kl_divs else 0.0 for agent, kl_divs in self.kl_divergences.items()},
            'mean_clip_fractions': {agent: np.mean(clip_fracs) if clip_fracs else 0.0 for agent, clip_fracs in self.clip_fractions.items()},
            'mean_explained_variances': {agent: np.mean(exp_vars) if exp_vars else 0.0 for agent, exp_vars in self.explained_variances.items()}
        }
        
        # Overall statistics
        all_rewards = [reward for rewards in self.episode_rewards.values() for reward in rewards]
        all_lengths = [length for lengths in self.episode_lengths.values() for length in lengths]
        
        stats['overall_mean_reward'] = np.mean(all_rewards) if all_rewards else 0.0
        stats['overall_std_reward'] = np.std(all_rewards) if all_rewards else 0.0
        stats['overall_mean_length'] = np.mean(all_lengths) if all_lengths else 0.0
        
        return stats
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update_count': self.update_count,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'kl_divergences': self.kl_divergences,
            'clip_fractions': self.clip_fractions,
            'explained_variances': self.explained_variances
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.update_count = checkpoint['update_count']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        self.kl_divergences = checkpoint['kl_divergences']
        self.clip_fractions = checkpoint['clip_fractions']
        self.explained_variances = checkpoint['explained_variances']


if __name__ == "__main__":
    # Test MAPPO trainer
    print("Testing MAPPO Trainer...")
    
    num_agents = 3
    obs_dim = 18
    action_dim = 5
    global_obs_dim = obs_dim * num_agents
    
    # Create trainer
    trainer = MAPPOTrainer(
        num_agents=num_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        global_obs_dim=global_obs_dim
    )
    
    # Test action selection
    observations = [np.random.randn(obs_dim) for _ in range(num_agents)]
    actions, log_probs, values = trainer.get_actions(observations)
    
    print(f"Actions: {actions}")
    print(f"Log probs: {log_probs}")
    print(f"Values: {values}")
    
    # Test centralized value estimation
    global_obs = np.random.randn(global_obs_dim)
    centralized_values = trainer.get_centralized_values(global_obs)
    print(f"Centralized values: {centralized_values}")
    
    # Test statistics
    stats = trainer.get_statistics()
    print(f"Statistics: {stats}")
    
    print("MAPPO Trainer test completed!")
