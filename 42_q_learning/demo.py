"""
Q-Learning Demo Script
======================

Complete demonstration of Q-learning implementation with GridWorld and FrozenLake.
This script ties together all components for easy testing and evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from q_learning import QLearningAgent
from gridworld_env import create_simple_gridworld, create_cliff_walking, create_maze_gridworld
from plot_results import QLearningVisualizer

def demo_gridworld():
    """Demonstrate Q-learning on GridWorld environments."""
    
    print("=" * 60)
    print("Q-LEARNING GRIDWORLD DEMONSTRATION")
    print("=" * 60)
    
    # Create visualizer
    visualizer = QLearningVisualizer()
    
    # Test different GridWorld environments
    environments = {
        "Simple GridWorld": create_simple_gridworld(),
        "Cliff Walking": create_cliff_walking(), 
        "Maze GridWorld": create_maze_gridworld()
    }
    
    results = {}
    
    for env_name, env in environments.items():
        print(f"\n--- Testing {env_name} ---")
        print(f"Environment size: {env.height}×{env.width}")
        print(f"Start: {env.start_pos}, Goal: {env.goal_pos}")
        print(f"Obstacles: {len(env.obstacles)}, Cliffs: {len(env.cliffs)}")
        
        # Create Q-learning agent
        agent = QLearningAgent(
            n_states=env.observation_space.n,
            n_actions=env.action_space.n,
            learning_rate=0.1,
            discount_factor=0.95,
            epsilon=0.1,
            epsilon_decay=0.995,
            epsilon_min=0.01
        )
        
        # Train agent
        print("Training agent...")
        start_time = time.time()
        stats = agent.train(env, n_episodes=1000, verbose=False)
        training_time = time.time() - start_time
        
        # Evaluate agent
        print("Evaluating agent...")
        eval_stats = agent.evaluate(env, n_episodes=100)
        
        # Store results
        results[env_name] = {
            'agent': agent,
            'training_stats': stats,
            'evaluation_stats': eval_stats,
            'training_time': training_time,
            'env': env
        }
        
        # Print summary
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Final average reward: {eval_stats['avg_reward']:.3f}")
        print(f"Success rate: {eval_stats['success_rate']:.1%}")
        print(f"Average episode length: {eval_stats['avg_length']:.1f}")
        
        # Show optimal path
        policy = agent.get_policy()
        path = env.get_optimal_path(policy)
        print(f"Optimal path length: {len(path)} steps")
        print(f"Path: {path[:5]}{'...' if len(path) > 5 else ''}")
    
    # Visualize results
    print("\n--- Creating Visualizations ---")
    
    # Plot learning curves comparison
    visualizer.plot_training_curves(results)
    
    # Visualize best performing environment
    best_env = max(results.keys(), key=lambda x: results[x]['evaluation_stats']['success_rate'])
    best_result = results[best_env]
    
    print(f"\nBest performing environment: {best_env}")
    visualizer.visualize_gridworld_policy(
        best_result['env'], 
        best_result['agent'].get_policy(),
        best_result['agent'].q_table,
        title=f"Best Policy: {best_env}"
    )
    
    return results

def demo_exploration_strategies():
    """Demonstrate different exploration strategies."""
    
    print("\n" + "=" * 60)
    print("EXPLORATION STRATEGY COMPARISON")
    print("=" * 60)
    
    # Create environment
    env = create_simple_gridworld()
    
    # Define different exploration strategies
    strategies = {
        "High Exploration (ε=0.3)": {
            "epsilon": 0.3,
            "epsilon_decay": 0.999,
            "epsilon_min": 0.01,
            "learning_rate": 0.1,
            "discount_factor": 0.95
        },
        "Medium Exploration (ε=0.1)": {
            "epsilon": 0.1,
            "epsilon_decay": 0.995,
            "epsilon_min": 0.01,
            "learning_rate": 0.1,
            "discount_factor": 0.95
        },
        "Low Exploration (ε=0.05)": {
            "epsilon": 0.05,
            "epsilon_decay": 0.99,
            "epsilon_min": 0.01,
            "learning_rate": 0.1,
            "discount_factor": 0.95
        },
        "Greedy (ε=0)": {
            "epsilon": 0.0,
            "epsilon_decay": 1.0,
            "epsilon_min": 0.0,
            "learning_rate": 0.1,
            "discount_factor": 0.95
        }
    }
    
    results = {}
    
    for strategy_name, params in strategies.items():
        print(f"\n--- Testing {strategy_name} ---")
        
        # Create agent
        agent = QLearningAgent(
            n_states=env.observation_space.n,
            n_actions=env.action_space.n,
            **params
        )
        
        # Train agent
        print("Training...")
        start_time = time.time()
        stats = agent.train(env, n_episodes=1000, verbose=False)
        training_time = time.time() - start_time
        
        # Evaluate agent
        eval_stats = agent.evaluate(env, n_episodes=100)
        
        # Store results
        results[strategy_name] = {
            'agent': agent,
            'training_stats': stats,
            'evaluation_stats': eval_stats,
            'training_time': training_time
        }
        
        print(f"Success rate: {eval_stats['success_rate']:.1%}")
        print(f"Average reward: {eval_stats['avg_reward']:.3f}")
        print(f"Training time: {training_time:.2f}s")
    
    # Create visualizations
    visualizer = QLearningVisualizer()
    visualizer.plot_exploration_analysis(results)
    
    # Show policies for comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (strategy_name, result) in enumerate(results.items()):
        ax = axes[i]
        policy = result['agent'].get_policy()
        state_values = result['agent'].get_state_values()
        
        # Create policy grid
        value_grid = state_values.reshape(env.height, env.width)
        im = ax.imshow(value_grid, cmap='RdYlBu_r', alpha=0.3)
        
        # Action arrows
        action_directions = {0: (0, 0.3), 1: (0, -0.3), 2: (-0.3, 0), 3: (0.3, 0)}
        
        for row in range(env.height):
            for col in range(env.width):
                state = row * env.width + col
                pos = (row, col)
                
                if pos not in env.obstacles and pos != env.goal_pos and pos not in env.cliffs:
                    action = policy[state]
                    dx, dy = action_directions[action]
                    ax.arrow(col, row, dx, dy, head_width=0.1, head_length=0.1,
                           fc='black', ec='black')
        
        ax.set_title(f"{strategy_name}\nSuccess: {result['evaluation_stats']['success_rate']:.1%}")
        ax.set_xticks(range(env.width))
        ax.set_yticks(range(env.height))
    
    plt.suptitle("Policy Comparison: Different Exploration Strategies")
    plt.tight_layout()
    plt.show()
    
    return results

def demo_hyperparameter_study():
    """Demonstrate hyperparameter sensitivity."""
    
    print("\n" + "=" * 60)
    print("HYPERPARAMETER SENSITIVITY STUDY")
    print("=" * 60)
    
    env = create_simple_gridworld()
    
    # Learning rate study
    print("\n--- Learning Rate Study ---")
    learning_rates = [0.01, 0.05, 0.1, 0.3, 0.5]
    lr_results = {}
    
    base_params = {
        "discount_factor": 0.95,
        "epsilon": 0.1,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.01
    }
    
    for lr in learning_rates:
        print(f"Testing learning rate: {lr}")
        
        agent = QLearningAgent(
            n_states=env.observation_space.n,
            n_actions=env.action_space.n,
            learning_rate=lr,
            **base_params
        )
        
        stats = agent.train(env, n_episodes=500, verbose=False)
        eval_stats = agent.evaluate(env, n_episodes=100)
        
        lr_results[f"α={lr}"] = {
            'training_stats': stats,
            'evaluation_stats': eval_stats,
            'learning_rate': lr
        }
        
        print(f"  Success rate: {eval_stats['success_rate']:.1%}")
    
    # Discount factor study
    print("\n--- Discount Factor Study ---")
    discount_factors = [0.8, 0.9, 0.95, 0.99]
    gamma_results = {}
    
    base_params['learning_rate'] = 0.1
    
    for gamma in discount_factors:
        print(f"Testing discount factor: {gamma}")
        
        agent = QLearningAgent(
            n_states=env.observation_space.n,
            n_actions=env.action_space.n,
            discount_factor=gamma,
            **{k: v for k, v in base_params.items() if k != 'discount_factor'}
        )
        
        stats = agent.train(env, n_episodes=500, verbose=False)
        eval_stats = agent.evaluate(env, n_episodes=100)
        
        gamma_results[f"γ={gamma}"] = {
            'training_stats': stats,
            'evaluation_stats': eval_stats,
            'discount_factor': gamma
        }
        
        print(f"  Success rate: {eval_stats['success_rate']:.1%}")
    
    # Visualize hyperparameter effects
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Learning rate effects
    lrs = [r['learning_rate'] for r in lr_results.values()]
    lr_success = [r['evaluation_stats']['success_rate'] for r in lr_results.values()]
    lr_rewards = [r['evaluation_stats']['avg_reward'] for r in lr_results.values()]
    
    ax1.plot(lrs, lr_success, 'bo-', label='Success Rate')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(lrs, lr_rewards, 'ro-', label='Avg Reward')
    
    ax1.set_xlabel('Learning Rate (α)')
    ax1.set_ylabel('Success Rate', color='b')
    ax1_twin.set_ylabel('Average Reward', color='r')
    ax1.set_title('Learning Rate Sensitivity')
    ax1.grid(True, alpha=0.3)
    
    # Discount factor effects
    gammas = [r['discount_factor'] for r in gamma_results.values()]
    gamma_success = [r['evaluation_stats']['success_rate'] for r in gamma_results.values()]
    gamma_rewards = [r['evaluation_stats']['avg_reward'] for r in gamma_results.values()]
    
    ax2.plot(gammas, gamma_success, 'bo-', label='Success Rate')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(gammas, gamma_rewards, 'ro-', label='Avg Reward')
    
    ax2.set_xlabel('Discount Factor (γ)')
    ax2.set_ylabel('Success Rate', color='b')
    ax2_twin.set_ylabel('Average Reward', color='r')
    ax2.set_title('Discount Factor Sensitivity')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Find optimal parameters
    best_lr = max(lr_results.keys(), key=lambda x: lr_results[x]['evaluation_stats']['success_rate'])
    best_gamma = max(gamma_results.keys(), key=lambda x: gamma_results[x]['evaluation_stats']['success_rate'])
    
    print(f"\nOptimal parameters:")
    print(f"  Best learning rate: {best_lr}")
    print(f"  Best discount factor: {best_gamma}")
    
    return lr_results, gamma_results

def main():
    """Main demonstration function."""
    
    print("🚀 Q-LEARNING COMPREHENSIVE DEMONSTRATION")
    print("==========================================")
    print("This demo showcases the complete Q-learning implementation")
    print("with GridWorld environments, exploration strategies, and analysis.")
    print()
    
    try:
        # Demo 1: GridWorld environments
        print("📍 DEMO 1: GridWorld Environments")
        gridworld_results = demo_gridworld()
        
        # Demo 2: Exploration strategies
        print("\n🔍 DEMO 2: Exploration Strategies")
        exploration_results = demo_exploration_strategies()
        
        # Demo 3: Hyperparameter study
        print("\n⚙️ DEMO 3: Hyperparameter Study")
        lr_results, gamma_results = demo_hyperparameter_study()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("✅ Successfully demonstrated:")
        print("  • Q-learning on multiple GridWorld environments")
        print("  • Exploration strategy comparison")
        print("  • Hyperparameter sensitivity analysis")
        print("  • Policy visualization and performance analysis")
        print()
        print("📊 Key Findings:")
        print("  • Q-learning successfully learns optimal policies")
        print("  • Exploration strategy significantly affects performance")
        print("  • Hyperparameter tuning is crucial for optimization")
        print("  • Off-policy learning enables safe exploration")
        print()
        print("🎯 Next Steps:")
        print("  • Try FrozenLake demo: python frozenlake_demo.py")
        print("  • Explore visualization tools: python plot_results.py")
        print("  • Experiment with custom environments")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        print("Please ensure all required packages are installed:")
        print("  pip install numpy matplotlib seaborn pandas")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
