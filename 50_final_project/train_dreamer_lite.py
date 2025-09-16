import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from collections import deque
import os
import time

from world_model import WorldModel
from models.actor_critic import Actor, Critic
from buffer import ReplayBuffer, ImaginaryBuffer
from utils import plot_training_curves, save_gif


class DreamerLite:
    """
    Dreamer-Lite implementation with hybrid RL (MBPO-style).
    """
    
    def __init__(self, env_name='CartPole-v1', latent_dim=32, hidden_dim=128, 
                 buffer_size=100000, imaginary_buffer_size=50000, 
                 batch_size=64, horizon=15, device='cpu'):
        
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.device = device
        
        # Environment dimensions
        self.state_dim = self.env.observation_space.shape[0]
        if hasattr(self.env.action_space, 'n'):
            # Discrete action space
            self.action_dim = self.env.action_space.n
            self.is_discrete = True
        else:
            # Continuous action space
            self.action_dim = self.env.action_space.shape[0]
            self.is_discrete = False
        
        # Hyperparameters
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.buffer_size = buffer_size
        self.imaginary_buffer_size = imaginary_buffer_size
        self.batch_size = batch_size
        self.horizon = horizon
        
        # Initialize models
        self.world_model = WorldModel(
            self.state_dim, self.action_dim, latent_dim, hidden_dim, device
        ).to(device)
        
        self.actor = Actor(latent_dim, self.action_dim, hidden_dim).to(device)
        self.critic = Critic(latent_dim, hidden_dim).to(device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        
        # Buffers
        self.replay_buffer = ReplayBuffer(buffer_size, self.state_dim, self.action_dim, device)
        self.imaginary_buffer = ImaginaryBuffer(imaginary_buffer_size, latent_dim, self.action_dim, device)
        
        # Training tracking
        self.episode_rewards = deque(maxlen=100)
        self.world_model_losses = []
        self.actor_losses = []
        self.critic_losses = []
        
    def select_action(self, state, deterministic=False):
        """
        Select action using the actor network.
        
        Args:
            state: current state
            deterministic: whether to use deterministic action selection
            
        Returns:
            action: selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Encode state to latent
            latent = self.world_model.encode(state_tensor)
            
            if deterministic:
                action = self.actor.get_action_deterministic(latent)
            else:
                action, _, _ = self.actor(latent)
            
            # Convert to numpy and handle discrete actions
            if self.is_discrete:
                # Discrete action space - use argmax for action selection
                action = action.argmax(dim=-1).cpu().numpy()[0]
            else:
                # Continuous action space
                action = action.cpu().numpy()[0]
                
        return action
    
    def collect_real_data(self, num_episodes=1):
        """
        Collect real data from the environment.
        
        Args:
            num_episodes: number of episodes to collect
            
        Returns:
            total_reward: total reward collected
        """
        total_reward = 0
        
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Store transition
                self.replay_buffer.add(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward
                
            total_reward += episode_reward
            self.episode_rewards.append(episode_reward)
            
        return total_reward / num_episodes
    
    def generate_imaginary_rollouts(self, num_rollouts=10):
        """
        Generate imaginary rollouts using the world model.
        
        Args:
            num_rollouts: number of imaginary rollouts to generate
        """
        if len(self.replay_buffer) < self.batch_size:
            return
            
        # Sample initial states from replay buffer
        batch = self.replay_buffer.sample(min(num_rollouts, len(self.replay_buffer)))
        initial_states = batch['states']
        
        # Encode initial states
        with torch.no_grad():
            initial_latents = self.world_model.encode(initial_states)
        
        # Generate rollouts
        for i in range(num_rollouts):
            latent = initial_latents[i:i+1]
            done = False
            step = 0
            
            while not done and step < self.horizon:
                # Select action
                action, _, _ = self.actor(latent)
                
                # Predict next latent and reward
                next_latent = self.world_model.predict_next_latent(latent, action)
                reward = self.world_model.predict_reward(latent, action)
                
                # Store imaginary transition
                self.imaginary_buffer.add(
                    latent.squeeze(0), action.squeeze(0), reward.squeeze(0),
                    next_latent.squeeze(0), torch.tensor(done, device=self.device)
                )
                
                latent = next_latent
                step += 1
                
                # Random termination for imagination rollouts
                if np.random.random() < 0.1:
                    done = True
    
    def train_world_model(self):
        """
        Train the world model on real data.
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}
            
        batch = self.replay_buffer.sample(self.batch_size)
        losses = self.world_model.train_step(batch)
        self.world_model_losses.append(losses)
        return losses
    
    def train_actor_critic(self, use_imaginary=True):
        """
        Train actor and critic networks.
        
        Args:
            use_imaginary: whether to use imaginary data for training
        """
        if use_imaginary and len(self.imaginary_buffer) >= self.batch_size:
            # Train on imaginary data
            batch = self.imaginary_buffer.sample(self.batch_size)
            latents = batch['latents'].detach()
            actions = batch['actions'].detach()
            rewards = batch['rewards'].detach()
            next_latents = batch['next_latents'].detach()
            dones = batch['dones'].detach()
        else:
            # Train on real data
            if len(self.replay_buffer) < self.batch_size:
                return {}
                
            batch = self.replay_buffer.sample(self.batch_size)
            states = batch['states']
            actions = batch['actions']
            rewards = batch['rewards']
            next_states = batch['next_states']
            dones = batch['dones']
            
            # Encode states to latents
            with torch.no_grad():
                latents = self.world_model.encode(states).detach()
                next_latents = self.world_model.encode(next_states).detach()
        
        # Train critic
        values = self.critic(latents)
        next_values = self.critic(next_latents)
        
        # Compute target values
        targets = rewards + 0.99 * next_values * (~dones)
        critic_loss = F.mse_loss(values, targets.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Train actor
        with torch.no_grad():
            old_values = self.critic(latents)
            advantages = targets - old_values
        
        new_actions, _, _ = self.actor(latents)
        
        # Actor loss (policy gradient)
        actor_loss = -(advantages * self.actor.log_prob(latents, new_actions)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        losses = {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item()
        }
        
        self.actor_losses.append(losses['actor_loss'])
        self.critic_losses.append(losses['critic_loss'])
        
        return losses
    
    def train(self, num_episodes=1000, world_model_update_freq=1, 
              actor_critic_update_freq=1, imaginary_update_freq=5):
        """
        Main training loop.
        
        Args:
            num_episodes: total number of episodes to train
            world_model_update_freq: frequency of world model updates
            actor_critic_update_freq: frequency of actor-critic updates
            imaginary_update_freq: frequency of imaginary rollout generation
        """
        print(f"Starting training on {self.env_name}")
        print(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            # Collect real data
            avg_reward = self.collect_real_data(num_episodes=1)
            
            # Train world model
            if episode % world_model_update_freq == 0:
                world_model_losses = self.train_world_model()
            
            # Generate imaginary rollouts
            if episode % imaginary_update_freq == 0 and len(self.replay_buffer) >= self.batch_size:
                self.generate_imaginary_rollouts(num_rollouts=10)
            
            # Train actor-critic
            if episode % actor_critic_update_freq == 0:
                actor_critic_losses = self.train_actor_critic(use_imaginary=True)
            
            # Logging
            if episode % 100 == 0:
                avg_reward_100 = np.mean(list(self.episode_rewards)[-100:])
                print(f"Episode {episode}, Avg Reward (100): {avg_reward_100:.2f}")
                
                if len(self.world_model_losses) > 0:
                    print(f"World Model Loss: {self.world_model_losses[-1]['total_loss']:.4f}")
                if len(self.actor_losses) > 0:
                    print(f"Actor Loss: {self.actor_losses[-1]:.4f}")
                if len(self.critic_losses) > 0:
                    print(f"Critic Loss: {self.critic_losses[-1]:.4f}")
                print("-" * 50)
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Plot training curves
        plot_training_curves(self.episode_rewards, self.world_model_losses, 
                           self.actor_losses, self.critic_losses)
    
    def evaluate(self, num_episodes=10, render=False):
        """
        Evaluate the trained agent.
        
        Args:
            num_episodes: number of episodes to evaluate
            render: whether to render the environment
            
        Returns:
            avg_reward: average reward over episodes
        """
        total_reward = 0
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                if render:
                    self.env.render()
                    
                action = self.select_action(state, deterministic=True)
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                
            total_reward += episode_reward
            print(f"Episode {episode + 1}: {episode_reward:.2f}")
            
        avg_reward = total_reward / num_episodes
        print(f"Average reward over {num_episodes} episodes: {avg_reward:.2f}")
        
        return avg_reward
    
    def save_models(self, filepath):
        """Save all models."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath + '_actor_critic.pth')
        
        self.world_model.save(filepath + '_world_model.pth')
    
    def load_models(self, filepath):
        """Load all models."""
        # Load actor-critic
        checkpoint = torch.load(filepath + '_actor_critic.pth', map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        # Load world model
        self.world_model.load(filepath + '_world_model.pth')


def main():
    """Main training function."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create agent
    agent = DreamerLite(
        env_name='CartPole-v1',
        latent_dim=32,
        hidden_dim=128,
        device=device
    )
    
    # Train agent
    agent.train(num_episodes=1000)
    
    # Evaluate agent
    print("\nEvaluating trained agent...")
    agent.evaluate(num_episodes=10)
    
    # Save models
    agent.save_models('models/dreamer_lite')


if __name__ == "__main__":
    main()
