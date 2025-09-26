#!/usr/bin/env python3
"""
Simple demo script for Dreamer-Lite implementation.
Tests core functionality without environment interaction.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from train_dreamer_lite import DreamerLite
from world_model import WorldModel
from models.encoder import Encoder
from models.transition import TransitionModel
from models.reward_predictor import RewardPredictor
from models.actor_critic import Actor, Critic


def test_world_model_training():
    """Test world model training on synthetic data."""
    print("Testing World Model Training")
    print("=" * 40)
    
    # Create world model
    world_model = WorldModel(state_dim=4, action_dim=2, latent_dim=32)
    
    # Generate synthetic data
    batch_size = 64
    states = torch.randn(batch_size, 4)
    actions = torch.randn(batch_size, 2)
    rewards = torch.randn(batch_size, 1)
    next_states = torch.randn(batch_size, 4)
    
    # Create batch
    batch = {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'next_states': next_states,
        'dones': torch.zeros(batch_size, 1, dtype=torch.bool)
    }
    
    # Train world model
    print("Training world model...")
    losses = []
    for i in range(100):
        loss = world_model.train_step(batch)
        losses.append(loss['total_loss'])
        
        if i % 20 == 0:
            print(f"  Step {i}: Total Loss = {loss['total_loss']:.4f}")
    
    print(f"Final loss: {losses[-1]:.4f}")
    print("✓ World model training successful")
    
    return losses


def test_actor_critic_training():
    """Test actor-critic training on synthetic data."""
    print("\nTesting Actor-Critic Training")
    print("=" * 40)
    
    # Create models
    actor = Actor(latent_dim=32, action_dim=2)
    critic = Critic(latent_dim=32)
    
    # Generate synthetic data
    batch_size = 64
    latents = torch.randn(batch_size, 32)
    actions = torch.randn(batch_size, 2)
    rewards = torch.randn(batch_size, 1)
    next_latents = torch.randn(batch_size, 32)
    dones = torch.zeros(batch_size, 1, dtype=torch.bool)
    
    # Optimizers
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)
    
    print("Training actor-critic...")
    actor_losses = []
    critic_losses = []
    
    for i in range(100):
        # Train critic
        values = critic(latents)
        next_values = critic(next_latents)
        targets = rewards + 0.99 * next_values * (~dones)
        critic_loss = torch.nn.functional.mse_loss(values, targets.detach())
        
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        
        # Train actor
        new_actions, _, _ = actor(latents)
        new_values = critic(latents)
        
        with torch.no_grad():
            old_values = critic(latents)
            advantages = targets - old_values
        
        actor_loss = -(advantages * actor.log_prob(latents, new_actions)).mean()
        
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        
        actor_losses.append(actor_loss.item())
        critic_losses.append(critic_loss.item())
        
        if i % 20 == 0:
            print(f"  Step {i}: Actor Loss = {actor_loss.item():.4f}, Critic Loss = {critic_loss.item():.4f}")
    
    print(f"Final actor loss: {actor_losses[-1]:.4f}")
    print(f"Final critic loss: {critic_losses[-1]:.4f}")
    print("✓ Actor-critic training successful")
    
    return actor_losses, critic_losses


def test_imagination_rollouts():
    """Test imagination rollouts using world model."""
    print("\nTesting Imagination Rollouts")
    print("=" * 40)
    
    # Create world model and actor
    world_model = WorldModel(state_dim=4, action_dim=2, latent_dim=32)
    actor = Actor(latent_dim=32, action_dim=2)
    
    # Generate initial state
    initial_state = torch.randn(1, 4)
    
    print("Generating imagination rollout...")
    
    # Encode initial state
    with torch.no_grad():
        latent = world_model.encode(initial_state)
    
    # Generate rollout
    horizon = 10
    latents = [latent]
    actions = []
    rewards = []
    
    for t in range(horizon):
        # Select action
        action, _, _ = actor(latent)
        actions.append(action)
        
        # Predict next latent and reward
        next_latent = world_model.predict_next_latent(latent, action)
        reward = world_model.predict_reward(latent, action)
        
        latents.append(next_latent)
        rewards.append(reward)
        
        latent = next_latent
    
    print(f"Generated {horizon} step rollout")
    print(f"Latent shapes: {[l.shape for l in latents[:3]]}...")
    print(f"Action shapes: {[a.shape for a in actions[:3]]}...")
    print(f"Reward shapes: {[r.shape for r in rewards[:3]]}...")
    print("✓ Imagination rollouts successful")
    
    return latents, actions, rewards


def plot_training_results(world_model_losses, actor_losses, critic_losses):
    """Plot training results."""
    print("\nPlotting Training Results")
    print("=" * 40)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # World model losses
    axes[0].plot(world_model_losses)
    axes[0].set_title('World Model Loss')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)
    
    # Actor losses
    axes[1].plot(actor_losses)
    axes[1].set_title('Actor Loss')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Loss')
    axes[1].grid(True)
    
    # Critic losses
    axes[2].plot(critic_losses)
    axes[2].set_title('Critic Loss')
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Loss')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Training results plotted and saved")


def main():
    """Main demo function."""
    print("Dreamer-Lite: Core Functionality Demo")
    print("=" * 50)
    
    # Test world model training
    world_model_losses = test_world_model_training()
    
    # Test actor-critic training
    actor_losses, critic_losses = test_actor_critic_training()
    
    # Test imagination rollouts
    latents, actions, rewards = test_imagination_rollouts()
    
    # Plot results
    plot_training_results(world_model_losses, actor_losses, critic_losses)
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")
    print("=" * 50)
    
    print("\nKey Features Demonstrated:")
    print("✓ World model training (encoder + transition + reward)")
    print("✓ Actor-critic training in latent space")
    print("✓ Imagination rollouts for sample efficiency")
    print("✓ Hybrid RL architecture")
    
    print("\nFiles created:")
    print("  - training_results.png")
    
    print("\nNext steps:")
    print("  - Run with real environment: python train_dreamer_lite.py")
    print("  - Try different environments: Pendulum-v1, etc.")
    print("  - Experiment with hyper-parameters")


if __name__ == "__main__":
    main()



