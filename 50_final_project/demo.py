#!/usr/bin/env python3
"""
Demo script for Dreamer-Lite implementation.
Shows how to train and evaluate the agent.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from train_dreamer_lite import DreamerLite


def demo_cartpole():
    """Demo with CartPole-v1 environment."""
    print("=" * 60)
    print("Dreamer-Lite Demo: CartPole-v1")
    print("=" * 60)
    
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
    
    print(f"Environment: {agent.env_name}")
    print(f"State dimension: {agent.state_dim}")
    print(f"Action dimension: {agent.action_dim}")
    print(f"Latent dimension: {agent.latent_dim}")
    
    # Train for a short period
    print("\nStarting training...")
    agent.train(num_episodes=200)
    
    # Evaluate
    print("\nEvaluating trained agent...")
    avg_reward = agent.evaluate(num_episodes=10)
    
    print(f"\nFinal average reward: {avg_reward:.2f}")
    
    # Save models
    agent.save_models('models/dreamer_lite_cartpole')
    print("Models saved to 'models/dreamer_lite_cartpole'")
    
    return agent


def demo_pendulum():
    """Demo with Pendulum-v1 environment."""
    print("=" * 60)
    print("Dreamer-Lite Demo: Pendulum-v1")
    print("=" * 60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create agent
    agent = DreamerLite(
        env_name='Pendulum-v1',
        latent_dim=32,
        hidden_dim=128,
        device=device
    )
    
    print(f"Environment: {agent.env_name}")
    print(f"State dimension: {agent.state_dim}")
    print(f"Action dimension: {agent.action_dim}")
    print(f"Latent dimension: {agent.latent_dim}")
    
    # Train for a short period
    print("\nStarting training...")
    agent.train(num_episodes=200)
    
    # Evaluate
    print("\nEvaluating trained agent...")
    avg_reward = agent.evaluate(num_episodes=10)
    
    print(f"\nFinal average reward: {avg_reward:.2f}")
    
    # Save models
    agent.save_models('models/dreamer_lite_pendulum')
    print("Models saved to 'models/dreamer_lite_pendulum'")
    
    return agent


def compare_environments():
    """Compare performance on different environments."""
    print("=" * 60)
    print("Environment Comparison")
    print("=" * 60)
    
    environments = ['CartPole-v1', 'Pendulum-v1']
    results = {}
    
    for env_name in environments:
        print(f"\nTraining on {env_name}...")
        
        agent = DreamerLite(
            env_name=env_name,
            latent_dim=32,
            hidden_dim=128
        )
        
        # Train
        agent.train(num_episodes=100)
        
        # Evaluate
        avg_reward = agent.evaluate(num_episodes=10)
        results[env_name] = avg_reward
        
        print(f"{env_name} average reward: {avg_reward:.2f}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    env_names = list(results.keys())
    rewards = list(results.values())
    
    plt.bar(env_names, rewards)
    plt.title('Dreamer-Lite Performance Comparison')
    plt.xlabel('Environment')
    plt.ylabel('Average Reward')
    plt.grid(True, alpha=0.3)
    
    for i, v in enumerate(rewards):
        plt.text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('environment_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results


def test_world_model():
    """Test world model components individually."""
    print("=" * 60)
    print("World Model Component Testing")
    print("=" * 60)
    
    from world_model import WorldModel
    
    # Create world model
    world_model = WorldModel(
        state_dim=4,  # CartPole
        action_dim=1,
        latent_dim=32,
        hidden_dim=128
    )
    
    # Test with random data
    batch_size = 32
    states = torch.randn(batch_size, 4)
    actions = torch.randn(batch_size, 1)
    
    print("Testing encoder...")
    latent, mean, log_std = world_model.encoder(states)
    print(f"  Input shape: {states.shape}")
    print(f"  Latent shape: {latent.shape}")
    print(f"  Mean shape: {mean.shape}")
    print(f"  Log std shape: {log_std.shape}")
    
    print("\nTesting transition model...")
    next_latent, next_mean, next_log_std = world_model.transition(latent, actions)
    print(f"  Next latent shape: {next_latent.shape}")
    print(f"  Next mean shape: {next_mean.shape}")
    print(f"  Next log std shape: {next_log_std.shape}")
    
    print("\nTesting reward predictor...")
    rewards = world_model.reward_predictor(latent, actions)
    print(f"  Reward shape: {rewards.shape}")
    
    print("\nTesting full forward pass...")
    next_latent_full, reward_full, latent_full = world_model(states, actions)
    print(f"  Full forward pass successful!")
    print(f"  Next latent shape: {next_latent_full.shape}")
    print(f"  Reward shape: {reward_full.shape}")
    print(f"  Latent shape: {latent_full.shape}")


def main():
    """Main demo function."""
    print("Dreamer-Lite: World Models + Hybrid RL Demo")
    print("=" * 60)
    
    # Test world model components
    test_world_model()
    
    # Demo CartPole
    cartpole_agent = demo_cartpole()
    
    # Demo Pendulum
    pendulum_agent = demo_pendulum()
    
    # Compare environments
    results = compare_environments()
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)
    
    print("\nResults Summary:")
    for env, reward in results.items():
        print(f"  {env}: {reward:.2f} average reward")
    
    print("\nFiles created:")
    print("  - training_curves.png")
    print("  - environment_comparison.png")
    print("  - models/dreamer_lite_cartpole_*.pth")
    print("  - models/dreamer_lite_pendulum_*.pth")


if __name__ == "__main__":
    main()


