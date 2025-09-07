#!/usr/bin/env python3
"""
CartPole-v1 DQN Training Script

This script trains a DQN agent on the CartPole-v1 environment and demonstrates
the key components of Deep Q-Networks including experience replay and target networks.
"""

import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import os
from dqn_agent import DQNAgent
from plot_results import plot_training_results, plot_learning_curves, plot_hyperparameter_comparison
import warnings
warnings.filterwarnings('ignore')


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_dqn_cartpole(agent_type: str = 'standard', 
                      num_episodes: int = 1000,
                      max_steps: int = 500,
                      learning_rate: float = 1e-3,
                      epsilon_decay: float = 0.995,
                      target_update_freq: int = 100,
                      buffer_size: int = 10000,
                      batch_size: int = 64,
                      use_double: bool = False,
                      use_prioritized: bool = False,
                      save_model: bool = True,
                      verbose: bool = True):
    """
    Train DQN agent on CartPole-v1.
    
    Args:
        agent_type: Type of DQN ('standard', 'dueling', 'double', 'prioritized')
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        learning_rate: Learning rate for optimizer
        epsilon_decay: Epsilon decay rate
        target_update_freq: Target network update frequency
        buffer_size: Size of replay buffer
        batch_size: Batch size for training
        use_double: Whether to use Double DQN
        use_prioritized: Whether to use prioritized experience replay
        save_model: Whether to save the trained model
        verbose: Whether to print training progress
        
    Returns:
        Trained agent and training statistics
    """
    print(f"Training {agent_type.upper()} DQN on CartPole-v1")
    print("=" * 50)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create environment
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"State space: {state_size}")
    print(f"Action space: {action_size}")
    print(f"Environment: {env.spec.id}")
    print()
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create agent based on type
    if agent_type == 'dueling':
        agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=learning_rate,
            epsilon_decay=epsilon_decay,
            target_update_freq=target_update_freq,
            buffer_size=buffer_size,
            batch_size=batch_size,
            device=device,
            use_dueling=True,
            use_double=use_double,
            use_prioritized=use_prioritized
        )
    elif agent_type == 'double':
        agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=learning_rate,
            epsilon_decay=epsilon_decay,
            target_update_freq=target_update_freq,
            buffer_size=buffer_size,
            batch_size=batch_size,
            device=device,
            use_double=True,
            use_prioritized=use_prioritized
        )
    elif agent_type == 'prioritized':
        agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=learning_rate,
            epsilon_decay=epsilon_decay,
            target_update_freq=target_update_freq,
            buffer_size=buffer_size,
            batch_size=batch_size,
            device=device,
            use_prioritized=True
        )
    else:  # standard
        agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=learning_rate,
            epsilon_decay=epsilon_decay,
            target_update_freq=target_update_freq,
            buffer_size=buffer_size,
            batch_size=batch_size,
            device=device
        )
    
    # Train the agent
    training_stats = agent.train(env, num_episodes, max_steps, verbose=verbose)
    
    # Evaluate the trained agent
    print("\nEvaluating trained agent...")
    eval_results = agent.evaluate(env, num_episodes=100, max_steps=max_steps)
    
    print(f"\nEvaluation Results:")
    print(f"Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"Success Rate: {eval_results['success_rate']:.2%}")
    print(f"Mean Episode Length: {eval_results['mean_length']:.2f}")
    
    # Save model if requested
    if save_model:
        model_path = f"models/cartpole_{agent_type}_dqn.pth"
        os.makedirs("models", exist_ok=True)
        agent.save(model_path)
        print(f"Model saved to: {model_path}")
    
    env.close()
    
    return agent, training_stats, eval_results


def compare_dqn_variants(num_episodes: int = 1000, max_steps: int = 500):
    """Compare different DQN variants on CartPole-v1."""
    print("Comparing DQN Variants on CartPole-v1")
    print("=" * 50)
    
    variants = [
        ('standard', 'Standard DQN'),
        ('dueling', 'Dueling DQN'),
        ('double', 'Double DQN'),
        ('prioritized', 'Prioritized DQN')
    ]
    
    results = {}
    
    for variant, name in variants:
        print(f"\nTraining {name}...")
        agent, stats, eval_results = train_dqn_cartpole(
            agent_type=variant,
            num_episodes=num_episodes,
            max_steps=max_steps,
            verbose=False
        )
        
        results[variant] = {
            'name': name,
            'agent': agent,
            'stats': stats,
            'eval_results': eval_results
        }
    
    # Plot comparison
    plot_hyperparameter_comparison(results, save_path='plots/dqn_variants_comparison.png')
    
    return results


def hyperparameter_sweep():
    """Perform hyperparameter sweep to find optimal settings."""
    print("Hyperparameter Sweep for DQN")
    print("=" * 50)
    
    # Define hyperparameter ranges
    learning_rates = [1e-4, 5e-4, 1e-3, 5e-3]
    epsilon_decays = [0.99, 0.995, 0.999]
    target_update_freqs = [50, 100, 200]
    
    best_score = -np.inf
    best_params = None
    all_results = []
    
    for lr in learning_rates:
        for eps_decay in epsilon_decays:
            for target_freq in target_update_freqs:
                print(f"\nTesting: LR={lr}, EpsDecay={eps_decay}, TargetFreq={target_freq}")
                
                try:
                    agent, stats, eval_results = train_dqn_cartpole(
                        agent_type='standard',
                        num_episodes=500,  # Shorter episodes for sweep
                        max_steps=500,
                        learning_rate=lr,
                        epsilon_decay=eps_decay,
                        target_update_freq=target_freq,
                        verbose=False
                    )
                    
                    score = eval_results['mean_reward']
                    all_results.append({
                        'learning_rate': lr,
                        'epsilon_decay': eps_decay,
                        'target_update_freq': target_freq,
                        'mean_reward': score,
                        'std_reward': eval_results['std_reward']
                    })
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'learning_rate': lr,
                            'epsilon_decay': eps_decay,
                            'target_update_freq': target_freq
                        }
                    
                    print(f"Score: {score:.2f}")
                    
                except Exception as e:
                    print(f"Failed: {e}")
                    continue
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best score: {best_score:.2f}")
    
    return all_results, best_params


def analyze_learning_curves(agent, training_stats):
    """Analyze and plot learning curves."""
    print("\nAnalyzing Learning Curves...")
    
    # Plot training results
    plot_training_results(
        episode_rewards=training_stats['episode_rewards'],
        episode_lengths=training_stats['episode_lengths'],
        losses=training_stats['losses'],
        epsilon_history=training_stats['epsilon_history'],
        save_path='plots/cartpole_training_results.png'
    )
    
    # Analyze convergence
    rewards = training_stats['episode_rewards']
    window_size = 100
    
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        
        # Find convergence point
        convergence_threshold = 0.1
        for i in range(window_size, len(moving_avg)):
            recent_std = np.std(moving_avg[i-window_size:i])
            if recent_std < convergence_threshold:
                print(f"Convergence point: Episode {i}")
                break
        else:
            print("No clear convergence detected")
    
    # Performance statistics
    final_performance = np.mean(rewards[-100:])
    max_performance = np.max(rewards)
    
    print(f"Final performance (last 100 episodes): {final_performance:.2f}")
    print(f"Best performance: {max_performance:.2f}")
    print(f"Total episodes: {len(rewards)}")


def demo_trained_agent(agent, num_episodes: int = 5):
    """Demonstrate the trained agent."""
    print(f"\nDemonstrating trained agent for {num_episodes} episodes...")
    
    env = gym.make('CartPole-v1')
    
    for episode in range(num_episodes):
        state = env.reset()
        if hasattr(state, '__len__') and len(state) > 1:
            state = state[0] if isinstance(state, tuple) else state
        
        total_reward = 0
        steps = 0
        
        for step in range(500):
            action = agent.act(state, training=False)
            result = env.step(action)
            
            if len(result) == 4:
                next_state, reward, done, info = result
            else:
                next_state, reward, done, truncated, info = result
                done = done or truncated
            
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        print(f"Episode {episode + 1}: Reward = {total_reward}, Steps = {steps}")
    
    env.close()


def main():
    """Main training function."""
    print("DQN CartPole Training")
    print("=" * 50)
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Train standard DQN
    print("1. Training Standard DQN...")
    agent, stats, eval_results = train_dqn_cartpole(
        agent_type='standard',
        num_episodes=1000,
        max_steps=500,
        learning_rate=1e-3,
        epsilon_decay=0.995,
        target_update_freq=100,
        verbose=True
    )
    
    # Analyze learning curves
    analyze_learning_curves(agent, stats)
    
    # Demonstrate trained agent
    demo_trained_agent(agent, num_episodes=5)
    
    # Compare DQN variants
    print("\n2. Comparing DQN Variants...")
    variant_results = compare_dqn_variants(num_episodes=800, max_steps=500)
    
    # Hyperparameter sweep
    print("\n3. Hyperparameter Sweep...")
    sweep_results, best_params = hyperparameter_sweep()
    
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)
    print("Key Insights:")
    print("1. DQN successfully learns to balance the CartPole")
    print("2. Experience replay stabilizes training")
    print("3. Target network prevents instability")
    print("4. Different DQN variants have different strengths")
    print("5. Hyperparameters significantly affect performance")
    print("\nCheck the 'plots' directory for visualizations!")


if __name__ == "__main__":
    main()
