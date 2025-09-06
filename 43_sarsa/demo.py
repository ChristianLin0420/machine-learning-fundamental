#!/usr/bin/env python3
"""
SARSA Algorithm Demo

This script demonstrates the SARSA algorithm implementation and compares it
with Q-Learning on a simple GridWorld environment.
"""

import numpy as np
import matplotlib.pyplot as plt
from sarsa import SARSAAgent, QLearningAgent, plot_learning_curves
from gridworld_env import create_simple_gridworld, create_obstacle_gridworld, create_stochastic_gridworld
import warnings
warnings.filterwarnings('ignore')


def demo_sarsa_basics():
    """Demonstrate basic SARSA functionality."""
    print("=" * 60)
    print("SARSA ALGORITHM DEMO")
    print("=" * 60)
    
    # Create a simple GridWorld environment
    env = create_simple_gridworld(size=4)
    print(f"Environment: {env.size}x{env.size} GridWorld")
    print(f"State space: {env.observation_space.n}")
    print(f"Action space: {env.action_space.n}")
    print()
    
    # Create SARSA agent
    agent = SARSAAgent(
        state_size=env.observation_space.n,
        action_size=env.action_space.n,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    print("Training SARSA agent...")
    stats = agent.train(env, num_episodes=500, max_steps=50, verbose=True)
    
    # Evaluate the trained agent
    print("\nEvaluating trained agent...")
    eval_results = agent.evaluate(env, num_episodes=100, max_steps=50)
    
    print(f"Success rate: {eval_results['success_rate']:.2%}")
    print(f"Mean reward: {eval_results['mean_reward']:.2f}")
    print(f"Mean episode length: {eval_results['mean_length']:.2f}")
    
    # Show learned policy
    policy = agent.get_policy()
    print(f"\nLearned policy (first 10 states): {policy[:10]}")
    
    return agent, stats


def demo_sarsa_vs_qlearning():
    """Compare SARSA with Q-Learning."""
    print("\n" + "=" * 60)
    print("SARSA vs Q-LEARNING COMPARISON")
    print("=" * 60)
    
    # Create environment
    env = create_obstacle_gridworld(size=5)
    print(f"Environment: {env.size}x{env.size} GridWorld with obstacles")
    
    # Create agents
    sarsa_agent = SARSAAgent(
        state_size=env.observation_space.n,
        action_size=env.action_space.n,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    qlearning_agent = QLearningAgent(
        state_size=env.observation_space.n,
        action_size=env.action_space.n,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    # Train both agents
    print("\nTraining SARSA agent...")
    sarsa_stats = sarsa_agent.train(env, num_episodes=1000, max_steps=100, verbose=True)
    
    print("\nTraining Q-Learning agent...")
    qlearning_stats = qlearning_agent.train(env, num_episodes=1000, max_steps=100, verbose=True)
    
    # Evaluate both agents
    print("\nEvaluating agents...")
    sarsa_eval = sarsa_agent.evaluate(env, num_episodes=100, max_steps=100)
    qlearning_eval = qlearning_agent.evaluate(env, num_episodes=100, max_steps=100)
    
    print(f"\nResults:")
    print(f"SARSA - Success rate: {sarsa_eval['success_rate']:.2%}, Mean reward: {sarsa_eval['mean_reward']:.2f}")
    print(f"Q-Learning - Success rate: {qlearning_eval['success_rate']:.2%}, Mean reward: {qlearning_eval['mean_reward']:.2f}")
    
    # Plot comparison
    plot_learning_curves(sarsa_stats, qlearning_stats, window_size=50)
    
    return sarsa_agent, qlearning_agent, sarsa_stats, qlearning_stats


def demo_stochastic_environment():
    """Demonstrate SARSA's behavior in stochastic environments."""
    print("\n" + "=" * 60)
    print("STOCHASTIC ENVIRONMENT DEMO")
    print("=" * 60)
    
    # Create stochastic environment
    env = create_stochastic_gridworld(size=5)
    print(f"Environment: {env.size}x{env.size} Stochastic GridWorld (10% random actions)")
    
    # Create agents
    sarsa_agent = SARSAAgent(
        state_size=env.observation_space.n,
        action_size=env.action_space.n,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    qlearning_agent = QLearningAgent(
        state_size=env.observation_space.n,
        action_size=env.action_space.n,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    # Train both agents
    print("\nTraining SARSA agent...")
    sarsa_stats = sarsa_agent.train(env, num_episodes=1000, max_steps=100, verbose=True)
    
    print("\nTraining Q-Learning agent...")
    qlearning_stats = qlearning_agent.train(env, num_episodes=1000, max_steps=100, verbose=True)
    
    # Evaluate both agents
    print("\nEvaluating agents...")
    sarsa_eval = sarsa_agent.evaluate(env, num_episodes=100, max_steps=100)
    qlearning_eval = qlearning_agent.evaluate(env, num_episodes=100, max_steps=100)
    
    print(f"\nResults in Stochastic Environment:")
    print(f"SARSA - Success rate: {sarsa_eval['success_rate']:.2%}, Mean reward: {sarsa_eval['mean_reward']:.2f}, Mean length: {sarsa_eval['mean_length']:.2f}")
    print(f"Q-Learning - Success rate: {qlearning_eval['success_rate']:.2%}, Mean reward: {qlearning_eval['mean_reward']:.2f}, Mean length: {qlearning_eval['mean_length']:.2f}")
    
    # Show policy differences
    sarsa_policy = sarsa_agent.get_policy()
    qlearning_policy = qlearning_agent.get_policy()
    
    policy_diff = np.sum(sarsa_policy != qlearning_policy)
    print(f"\nPolicy differences: {policy_diff} out of {len(sarsa_policy)} states")
    
    return sarsa_agent, qlearning_agent, sarsa_stats, qlearning_stats


def demo_hyperparameter_sensitivity():
    """Demonstrate hyperparameter sensitivity."""
    print("\n" + "=" * 60)
    print("HYPERPARAMETER SENSITIVITY DEMO")
    print("=" * 60)
    
    env = create_simple_gridworld(size=4)
    
    # Test different learning rates
    learning_rates = [0.01, 0.1, 0.5]
    results = {}
    
    for lr in learning_rates:
        print(f"\nTesting learning rate: {lr}")
        agent = SARSAAgent(
            state_size=env.observation_space.n,
            action_size=env.action_space.n,
            learning_rate=lr,
            discount_factor=0.95,
            epsilon=1.0,
            epsilon_decay=0.995,
            epsilon_min=0.01
        )
        
        stats = agent.train(env, num_episodes=500, max_steps=50, verbose=False)
        eval_results = agent.evaluate(env, num_episodes=50, max_steps=50)
        
        results[lr] = {
            'final_reward': np.mean(stats['episode_rewards'][-50:]),
            'success_rate': eval_results['success_rate'],
            'convergence_episode': find_convergence_point(stats['episode_rewards'])
        }
        
        print(f"  Final reward: {results[lr]['final_reward']:.2f}")
        print(f"  Success rate: {results[lr]['success_rate']:.2%}")
        print(f"  Convergence: Episode {results[lr]['convergence_episode']}")
    
    return results


def find_convergence_point(rewards, threshold=0.1, window=50):
    """Find convergence point in learning curve."""
    if len(rewards) < window:
        return len(rewards)
    
    for i in range(window, len(rewards)):
        recent_std = np.std(rewards[i-window:i])
        if recent_std < threshold:
            return i
    return len(rewards)


def main():
    """Run all demos."""
    print("Starting SARSA Algorithm Demonstration")
    print("This demo will show:")
    print("1. Basic SARSA functionality")
    print("2. SARSA vs Q-Learning comparison")
    print("3. Behavior in stochastic environments")
    print("4. Hyperparameter sensitivity")
    print()
    
    try:
        # Demo 1: Basic SARSA
        agent, stats = demo_sarsa_basics()
        
        # Demo 2: SARSA vs Q-Learning
        sarsa_agent, qlearning_agent, sarsa_stats, qlearning_stats = demo_sarsa_vs_qlearning()
        
        # Demo 3: Stochastic environment
        sarsa_stoch, qlearning_stoch, sarsa_stoch_stats, qlearning_stoch_stats = demo_stochastic_environment()
        
        # Demo 4: Hyperparameter sensitivity
        hyperparam_results = demo_hyperparameter_sensitivity()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETE!")
        print("=" * 60)
        print("Key Takeaways:")
        print("1. SARSA learns to navigate GridWorld environments effectively")
        print("2. SARSA and Q-Learning perform similarly in deterministic environments")
        print("3. SARSA may be more conservative in stochastic environments")
        print("4. Learning rate significantly affects convergence speed and stability")
        print("5. Both algorithms benefit from proper exploration scheduling")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        print("Please ensure all dependencies are installed: pip install gym matplotlib seaborn numpy scipy")


if __name__ == "__main__":
    main()
