#!/usr/bin/env python3
"""
Model-Based RL Training Script for CartPole-v1

This script implements model-based RL using a learned dynamics model
and MPC controller on the CartPole-v1 environment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
import os
import time
import warnings
warnings.filterwarnings('ignore')

from dynamics_model import create_dynamics_model
from buffer import create_replay_buffer
from mpc_controller import create_mpc_controller
from utils import set_seed, MBRLLogger, plot_model_predictions, plot_mpc_performance, save_model, load_model


class ModelBasedRLAgent:
    """
    Model-Based RL Agent for CartPole.
    
    This agent learns a dynamics model from experience and uses MPC
    to plan actions based on the learned model.
    """
    
    def __init__(self, state_dim: int, action_dim: int, model_type: str = 'probabilistic',
                 mpc_type: str = 'random_shooting', horizon: int = 10, num_samples: int = 1000,
                 learning_rate: float = 3e-4, device: str = 'cpu'):
        """
        Initialize Model-Based RL Agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            model_type: Type of dynamics model ('probabilistic', 'deterministic', 'ensemble')
            mpc_type: Type of MPC controller ('random_shooting', 'cem', 'mppi')
            horizon: Planning horizon
            num_samples: Number of action sequences to sample
            learning_rate: Learning rate for model training
            device: Device to run on
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        
        # Create dynamics model
        self.dynamics_model = create_dynamics_model(
            model_type=model_type,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=[256, 256],
            activation='relu',
            use_orthogonal_init=True
        ).to(self.device)
        
        # Create optimizer
        self.optimizer = optim.Adam(self.dynamics_model.parameters(), lr=learning_rate)
        
        # Create MPC controller
        self.mpc_controller = create_mpc_controller(
            controller_type=mpc_type,
            dynamics_model=self.dynamics_model,
            horizon=horizon,
            num_samples=num_samples,
            action_bounds=(-1.0, 1.0),  # CartPole actions are discrete, but we'll use continuous for planning
            device=device
        )
        
        # Training statistics
        self.model_losses = []
        self.episode_rewards = []
        self.episode_lengths = []
        
    def train_model(self, states: torch.Tensor, actions: torch.Tensor,
                   next_states: torch.Tensor, rewards: torch.Tensor) -> Dict[str, float]:
        """
        Train the dynamics model on a batch of transitions.
        
        Args:
            states: Current states
            actions: Actions taken
            next_states: Next states
            rewards: Rewards received
            
        Returns:
            Dictionary containing loss components
        """
        self.optimizer.zero_grad()
        
        # Compute loss
        losses = self.dynamics_model.compute_loss(states, actions, next_states, rewards)
        
        # Backward pass
        losses['total_loss'].backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.dynamics_model.parameters(), 0.5)
        
        # Update parameters
        self.optimizer.step()
        
        # Store losses
        self.model_losses.append(losses['total_loss'].item())
        
        return {
            'total_loss': losses['total_loss'].item(),
            'state_loss': losses['state_loss'].item(),
            'reward_loss': losses['reward_loss'].item()
        }
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Get action using MPC controller.
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic predictions
            
        Returns:
            Action to take
        """
        # Get continuous action from MPC
        continuous_action = self.mpc_controller.get_action(state)
        
        # Convert to discrete action for CartPole
        if continuous_action[0] < 0:
            return 0  # Push cart to the left
        else:
            return 1  # Push cart to the right
    
    def evaluate_model(self, states: torch.Tensor, actions: torch.Tensor,
                      next_states: torch.Tensor, rewards: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate the dynamics model on a batch of transitions.
        
        Args:
            states: Current states
            actions: Actions taken
            next_states: Next states
            rewards: Rewards received
            
        Returns:
            Dictionary containing evaluation metrics
        """
        with torch.no_grad():
            losses = self.dynamics_model.compute_loss(states, actions, next_states, rewards)
            
            # Compute prediction accuracy
            if hasattr(self.dynamics_model, 'probabilistic') and self.dynamics_model.probabilistic:
                outputs = self.dynamics_model.forward(states, actions)
                pred_next_states = outputs['state_mean']
                pred_rewards = outputs['reward_mean']
            else:
                outputs = self.dynamics_model.forward(states, actions)
                pred_next_states = outputs['next_states']
                pred_rewards = outputs['rewards']
            
            state_mse = torch.nn.functional.mse_loss(pred_next_states, next_states).item()
            reward_mse = torch.nn.functional.mse_loss(pred_rewards, rewards).item()
            
            return {
                'total_loss': losses['total_loss'].item(),
                'state_loss': losses['state_loss'].item(),
                'reward_loss': losses['reward_loss'].item(),
                'state_mse': state_mse,
                'reward_mse': reward_mse
            }


def collect_random_data(env, buffer, num_episodes: int = 100, max_steps: int = 500):
    """
    Collect random data for initial model training.
    
    Args:
        env: Environment
        buffer: Replay buffer
        num_episodes: Number of episodes to collect
        max_steps: Maximum steps per episode
    """
    print(f"Collecting {num_episodes} episodes of random data...")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            # Random action
            action = env.action_space.sample()
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            buffer.add(state, np.array([action]), next_state, reward, done)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}")
    
    print(f"Collected {buffer.size} transitions")


def train_dynamics_model(agent: ModelBasedRLAgent, buffer, num_epochs: int = 100,
                        batch_size: int = 256, verbose: bool = True) -> List[float]:
    """
    Train the dynamics model on collected data.
    
    Args:
        agent: Model-based RL agent
        buffer: Replay buffer
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        verbose: Whether to print progress
        
    Returns:
        List of training losses
    """
    print(f"Training dynamics model for {num_epochs} epochs...")
    
    losses = []
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        # Sample batches
        num_batches = max(1, buffer.size // batch_size)
        
        for batch_idx in range(num_batches):
            states, actions, next_states, rewards, dones = buffer.sample(batch_size)
            
            # Train model
            loss_dict = agent.train_model(states, actions, next_states, rewards)
            epoch_losses.append(loss_dict['total_loss'])
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    print(f"Model training completed. Final loss: {losses[-1]:.4f}")
    return losses


def evaluate_agent(agent: ModelBasedRLAgent, env, num_episodes: int = 10,
                  max_steps: int = 500, deterministic: bool = True) -> Dict[str, float]:
    """
    Evaluate the agent on the environment.
    
    Args:
        agent: Model-based RL agent
        env: Environment
        num_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode
        deterministic: Whether to use deterministic predictions
        
    Returns:
        Dictionary containing evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            action = agent.get_action(state, deterministic)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'max_reward': np.max(episode_rewards),
        'min_reward': np.min(episode_rewards)
    }


def train_model_based_rl(env_name: str = 'CartPole-v1', num_episodes: int = 1000,
                        model_type: str = 'probabilistic', mpc_type: str = 'random_shooting',
                        horizon: int = 10, num_samples: int = 1000, learning_rate: float = 3e-4,
                        buffer_capacity: int = 100000, batch_size: int = 256,
                        model_epochs: int = 100, eval_frequency: int = 50,
                        device: str = 'cpu', verbose: bool = True,
                        save_model_flag: bool = True, plot_results: bool = True) -> ModelBasedRLAgent:
    """
    Train a model-based RL agent.
    
    Args:
        env_name: Name of the environment
        num_episodes: Number of training episodes
        model_type: Type of dynamics model
        mpc_type: Type of MPC controller
        horizon: Planning horizon
        num_samples: Number of action sequences to sample
        learning_rate: Learning rate for model training
        buffer_capacity: Capacity of replay buffer
        batch_size: Batch size for training
        model_epochs: Number of epochs to train model
        eval_frequency: Frequency of evaluation
        device: Device to run on
        verbose: Whether to print progress
        save_model_flag: Whether to save the trained model
        plot_results: Whether to plot results
        
    Returns:
        Trained model-based RL agent
    """
    # Set seed for reproducibility
    set_seed(42)
    
    # Create environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = 1  # We'll use continuous actions for planning
    
    print(f"Environment: {env_name}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print()
    
    # Create agent
    agent = ModelBasedRLAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        model_type=model_type,
        mpc_type=mpc_type,
        horizon=horizon,
        num_samples=num_samples,
        learning_rate=learning_rate,
        device=device
    )
    
    # Create replay buffer
    buffer = create_replay_buffer('standard', buffer_capacity, device)
    
    # Create logger
    logger = MBRLLogger()
    
    # Collect initial random data
    collect_random_data(env, buffer, num_episodes=100, max_steps=500)
    
    # Train initial model
    train_dynamics_model(agent, buffer, num_epochs=model_epochs, batch_size=batch_size, verbose=verbose)
    
    # Training loop
    print(f"Starting model-based RL training for {num_episodes} episodes...")
    print()
    
    for episode in range(num_episodes):
        # Reset environment
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        # Run episode
        for step in range(500):  # CartPole max steps
            # Get action from MPC
            action = agent.get_action(state)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            buffer.add(state, np.array([action]), next_state, reward, done)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        # Log episode
        logger.log_episode(episode_reward, episode_length)
        
        # Retrain model periodically
        if (episode + 1) % 10 == 0:
            # Sample recent data for retraining
            recent_states, recent_actions, recent_next_states, recent_rewards, recent_dones = buffer.sample(min(1000, buffer.size))
            
            # Train model
            loss_dict = agent.train_model(recent_states, recent_actions, recent_next_states, recent_rewards)
            logger.log_model_update(
                loss_dict['total_loss'],
                loss_dict['state_loss'],
                loss_dict['reward_loss'],
                0.1  # Placeholder for gradient norm
            )
        
        # Evaluate agent
        if (episode + 1) % eval_frequency == 0:
            eval_results = evaluate_agent(agent, env, num_episodes=10)
            
            if verbose:
                print(f"Episode {episode + 1}/{num_episodes}")
                print(f"  Episode Reward: {episode_reward:.2f}")
                print(f"  Episode Length: {episode_length}")
                print(f"  Eval Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
                print(f"  Eval Mean Length: {eval_results['mean_length']:.2f}")
                print(f"  Buffer Size: {buffer.size}")
                print()
    
    # Final evaluation
    print("Final evaluation...")
    final_eval = evaluate_agent(agent, env, num_episodes=100)
    
    print(f"Final Results:")
    print(f"  Mean Reward: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}")
    print(f"  Mean Length: {final_eval['mean_length']:.2f}")
    print(f"  Max Reward: {final_eval['max_reward']:.2f}")
    print(f"  Min Reward: {final_eval['min_reward']:.2f}")
    
    # Save model
    if save_model_flag:
        os.makedirs('models', exist_ok=True)
        model_path = f'models/mbrl_{env_name.lower().replace("-", "_")}.pth'
        save_model(agent.dynamics_model, agent.optimizer, model_path)
        print(f"Model saved to: {model_path}")
    
    # Plot results
    if plot_results:
        logger.plot_training_curves(save_path='plots/mbrl_training.png')
        
        # Plot model predictions
        states, actions, next_states, rewards, dones = buffer.sample(1000)
        plot_model_predictions(agent.dynamics_model, states, actions, next_states, rewards, 
                              save_path='plots/model_predictions.png')
        
        # Plot MPC performance
        mpc_stats = agent.mpc_controller.get_statistics()
        if mpc_stats['total_plans'] > 0:
            plot_mpc_performance(agent.mpc_controller.best_returns, save_path='plots/mpc_performance.png')
    
    env.close()
    
    return agent


def compare_model_based_vs_model_free(env_name: str = 'CartPole-v1', num_episodes: int = 500,
                                     device: str = 'cpu', verbose: bool = True):
    """
    Compare model-based RL vs model-free RL on CartPole.
    
    Args:
        env_name: Name of the environment
        num_episodes: Number of training episodes
        device: Device to run on
        verbose: Whether to print progress
    """
    print("Comparing Model-Based RL vs Model-Free RL...")
    print("=" * 60)
    
    # Model-based RL
    print("Training Model-Based RL...")
    mbrl_agent = train_model_based_rl(
        env_name=env_name,
        num_episodes=num_episodes,
        model_type='probabilistic',
        mpc_type='random_shooting',
        horizon=10,
        num_samples=1000,
        device=device,
        verbose=verbose,
        save_model_flag=False,
        plot_results=False
    )
    
    # Evaluate model-based RL
    mbrl_eval = evaluate_agent(mbrl_agent, gym.make(env_name), num_episodes=100)
    
    print(f"Model-Based RL Results:")
    print(f"  Mean Reward: {mbrl_eval['mean_reward']:.2f} ± {mbrl_eval['std_reward']:.2f}")
    print(f"  Mean Length: {mbrl_eval['mean_length']:.2f}")
    
    # Model-free RL (simple random policy for comparison)
    print("\nEvaluating Random Policy (Model-Free baseline)...")
    env = gym.make(env_name)
    random_rewards = []
    random_lengths = []
    
    for episode in range(100):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(500):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        random_rewards.append(episode_reward)
        random_lengths.append(episode_length)
    
    random_eval = {
        'mean_reward': np.mean(random_rewards),
        'std_reward': np.std(random_rewards),
        'mean_length': np.mean(random_lengths)
    }
    
    print(f"Random Policy Results:")
    print(f"  Mean Reward: {random_eval['mean_reward']:.2f} ± {random_eval['std_reward']:.2f}")
    print(f"  Mean Length: {random_eval['mean_length']:.2f}")
    
    # Compare results
    print(f"\nComparison:")
    print(f"  Model-Based RL: {mbrl_eval['mean_reward']:.2f} ± {mbrl_eval['std_reward']:.2f}")
    print(f"  Random Policy:  {random_eval['mean_reward']:.2f} ± {random_eval['std_reward']:.2f}")
    print(f"  Improvement:    {mbrl_eval['mean_reward'] - random_eval['mean_reward']:.2f}")
    
    env.close()


def main():
    """Main training function."""
    print("Model-Based RL Training on CartPole-v1")
    print("=" * 60)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Choose training mode
    mode = input("Choose mode (train/compare): ").lower().strip()
    
    if mode == 'train':
        print("\nTraining Model-Based RL...")
        agent = train_model_based_rl(
            env_name='CartPole-v1',
            num_episodes=500,
            model_type='probabilistic',
            mpc_type='random_shooting',
            horizon=10,
            num_samples=1000,
            learning_rate=3e-4,
            buffer_capacity=100000,
            batch_size=256,
            model_epochs=100,
            eval_frequency=50,
            device='cpu',
            verbose=True,
            save_model_flag=True,
            plot_results=True
        )
    elif mode == 'compare':
        compare_model_based_vs_model_free(
            env_name='CartPole-v1',
            num_episodes=500,
            device='cpu',
            verbose=True
        )
    else:
        print("Invalid mode. Please choose 'train' or 'compare'.")
        return
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("Key Insights:")
    print("1. Model-based RL learns environment dynamics")
    print("2. MPC uses learned model for planning")
    print("3. Sample efficiency depends on model accuracy")
    print("4. Planning horizon affects performance")
    print("5. Model uncertainty can guide exploration")
    print("\nCheck the 'plots' directory for visualizations!")


if __name__ == "__main__":
    main()
