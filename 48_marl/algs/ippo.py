"""
IPPO (Independent PPO) Implementation for Multi-Agent RL

This module implements IPPO where each agent has its own actor-critic network
but they are trained independently. This is a strong baseline for multi-agent RL.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')

from .nets import MultiAgentActorCritic, count_parameters
from .gae import compute_gae_advantages_tensor, normalize_advantages_tensor


class IPPOTrainer:
    """
    IPPO (Independent PPO) Trainer for Multi-Agent RL.
    
    Each agent has its own actor-critic network and is trained independently
    using PPO with clipped objective. This is a strong baseline for MARL.
    """
    
    def __init__(self, num_agents: int, obs_dim: int, action_dim: int,
                 learning_rate: float = 3e-4, gamma: float = 0.99,
                 lam: float = 0.95, entropy_coef: float = 0.01,
                 value_coef: float = 0.5, max_grad_norm: float = 0.5,
                 clip_ratio: float = 0.2, ppo_epochs: int = 4,
                 mini_batch_size: int = 64, target_kl: float = 0.01,
                 hidden_sizes: List[int] = [256, 256], activation: str = 'relu',
                 device: str = 'cpu', normalize_advantages: bool = True,
                 shared_parameters: bool = False):
        """
        Initialize IPPO Trainer.
        
        Args:
            num_agents: Number of agents
            obs_dim: Dimension of observation space
            action_dim: Number of possible actions
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
            shared_parameters: Whether to share parameters across agents
        """
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
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
        self.shared_parameters = shared_parameters
        
        # Create multi-agent actor-critic network
        self.network = MultiAgentActorCritic(
            num_agents=num_agents,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            use_orthogonal_init=True,
            shared_parameters=shared_parameters
        ).to(self.device)
        
        # Create optimizers for each agent
        if shared_parameters:
            # All agents share the same optimizer
            self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        else:
            # Each agent has its own optimizer
            self.optimizers = []
            for i in range(num_agents):
                optimizer = optim.Adam(self.network.agent_networks[i].parameters(), lr=learning_rate)
                self.optimizers.append(optimizer)
        
        # Training statistics
        self.update_count = 0
        self.episode_rewards = {f'agent_{i}': [] for i in range(num_agents)}
        self.episode_lengths = {f'agent_{i}': [] for i in range(num_agents)}
        self.kl_divergences = {f'agent_{i}': [] for i in range(num_agents)}
        self.clip_fractions = {f'agent_{i}': [] for i in range(num_agents)}
        self.explained_variances = {f'agent_{i}': [] for i in range(num_agents)}
        
        print(f"Created IPPO Trainer:")
        print(f"  Number of agents: {num_agents}")
        print(f"  Total parameters: {count_parameters(self.network)}")
        print(f"  Device: {self.device}")
        print(f"  Shared parameters: {shared_parameters}")
        print(f"  PPO epochs: {ppo_epochs}")
        print(f"  Mini-batch size: {mini_batch_size}")
    
    def get_actions(self, observations: List[np.ndarray], deterministic: bool = False) -> Tuple[List[int], List[float], List[float]]:
        """
        Get actions for all agents.
        
        Args:
            observations: List of observations for each agent
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
    
    def get_values(self, observations: List[np.ndarray]) -> List[float]:
        """
        Get value estimates for all agents.
        
        Args:
            observations: List of observations for each agent
            
        Returns:
            List of value estimates for each agent
        """
        values = []
        
        for i, obs in enumerate(observations):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                value = self.network.get_value(obs_tensor, i)
            
            values.append(value.item())
        
        return values
    
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
    
    def update(self, rollout_data: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, float]]:
        """
        Update all agent networks using IPPO.
        
        Args:
            rollout_data: Dictionary mapping agent names to rollout data
            
        Returns:
            Dictionary containing update statistics for each agent
        """
        update_stats = {}
        
        # Update each agent independently
        for agent_id in range(self.num_agents):
            agent_name = f'agent_{agent_id}'
            agent_data = rollout_data[agent_name]
            
            # Extract data
            states = agent_data['states']
            actions = agent_data['actions']
            old_log_probs = agent_data['log_probs']
            advantages = agent_data['advantages']
            returns = agent_data['returns']
            
            # Normalize advantages if requested
            if self.normalize_advantages:
                advantages = normalize_advantages_tensor(advantages)
            
            # Get buffer size
            buffer_size = states.size(0)
            
            # Training statistics
            policy_losses = []
            value_losses = []
            entropy_losses = []
            kl_divergences = []
            clip_fractions = []
            explained_variances = []
            
            # PPO epochs
            for epoch in range(self.ppo_epochs):
                # Shuffle data
                indices = torch.randperm(buffer_size, device=self.device)
                
                # Mini-batch training
                for start_idx in range(0, buffer_size, self.mini_batch_size):
                    end_idx = min(start_idx + self.mini_batch_size, buffer_size)
                    batch_indices = indices[start_idx:end_idx]
                    
                    # Get mini-batch
                    batch_states = states[batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_advantages = advantages[batch_indices]
                    batch_returns = returns[batch_indices]
                    
                    # Get current policy log probabilities and values
                    current_log_probs, current_values = self.network.get_log_probs(
                        batch_states, batch_actions, agent_id
                    )
                    
                    # Compute policy ratio
                    ratio = torch.exp(current_log_probs - batch_old_log_probs)
                    
                    # Compute clipped objective
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Compute value loss
                    value_loss = nn.functional.mse_loss(current_values, batch_returns)
                    
                    # Compute entropy loss
                    entropies = self.network.get_entropy(batch_states, agent_id)
                    entropy_loss = -entropies.mean()
                    
                    # Total loss
                    total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
                    
                    # Update network
                    if self.shared_parameters:
                        self.optimizer.zero_grad()
                    else:
                        self.optimizers[agent_id].zero_grad()
                    
                    total_loss.backward()
                    
                    # Clip gradients
                    if self.shared_parameters:
                        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.network.agent_networks[agent_id].parameters(), self.max_grad_norm)
                    
                    # Update parameters
                    if self.shared_parameters:
                        self.optimizer.step()
                    else:
                        self.optimizers[agent_id].step()
                    
                    # Compute statistics
                    policy_losses.append(policy_loss.item())
                    value_losses.append(value_loss.item())
                    entropy_losses.append(entropy_loss.item())
                    
                    # Compute KL divergence
                    kl_div = self.compute_kl_divergence(batch_old_log_probs, current_log_probs)
                    kl_divergences.append(kl_div.item())
                    
                    # Compute clip fraction
                    clip_frac = self.compute_clip_fraction(ratio)
                    clip_fractions.append(clip_frac)
                    
                    # Compute explained variance
                    exp_var = self.compute_explained_variance(current_values, batch_returns)
                    explained_variances.append(exp_var.item())
                    
                    # Early stopping if KL divergence is too high
                    if kl_div > 1.5 * self.target_kl:
                        break
                
                # Early stopping if KL divergence is too high
                if kl_divergences and np.mean(kl_divergences) > 1.5 * self.target_kl:
                    break
            
            # Compute final statistics
            final_policy_loss = np.mean(policy_losses)
            final_value_loss = np.mean(value_losses)
            final_entropy_loss = np.mean(entropy_losses)
            final_kl_div = np.mean(kl_divergences)
            final_clip_frac = np.mean(clip_fractions)
            final_exp_var = np.mean(explained_variances)
            
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
            'optimizer_state_dict': self.optimizer.state_dict() if self.shared_parameters else [opt.state_dict() for opt in self.optimizers],
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
        
        if self.shared_parameters:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            for i, opt_state in enumerate(checkpoint['optimizer_state_dict']):
                self.optimizers[i].load_state_dict(opt_state)
        
        self.update_count = checkpoint['update_count']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        self.kl_divergences = checkpoint['kl_divergences']
        self.clip_fractions = checkpoint['clip_fractions']
        self.explained_variances = checkpoint['explained_variances']


if __name__ == "__main__":
    # Test IPPO trainer
    print("Testing IPPO Trainer...")
    
    num_agents = 3
    obs_dim = 18
    action_dim = 5
    
    # Create trainer
    trainer = IPPOTrainer(
        num_agents=num_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        shared_parameters=True
    )
    
    # Test action selection
    observations = [np.random.randn(obs_dim) for _ in range(num_agents)]
    actions, log_probs, values = trainer.get_actions(observations)
    
    print(f"Actions: {actions}")
    print(f"Log probs: {log_probs}")
    print(f"Values: {values}")
    
    # Test statistics
    stats = trainer.get_statistics()
    print(f"Statistics: {stats}")
    
    print("IPPO Trainer test completed!")
