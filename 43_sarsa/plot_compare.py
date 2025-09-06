import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sarsa import SARSAAgent, QLearningAgent, plot_learning_curves
from gridworld_env import create_simple_gridworld, create_obstacle_gridworld, create_stochastic_gridworld
import gym
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def compare_algorithms_on_gridworld(env_type: str = 'simple', 
                                  num_episodes: int = 1000,
                                  max_steps: int = 100,
                                  grid_size: int = 5):
    """
    Compare SARSA and Q-Learning on different GridWorld configurations.
    
    Args:
        env_type: Type of GridWorld ('simple', 'obstacle', 'stochastic')
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        grid_size: Size of the grid
    """
    print(f"Comparing algorithms on {env_type} GridWorld ({grid_size}x{grid_size})")
    print("=" * 60)
    
    # Create environment
    if env_type == 'simple':
        env = create_simple_gridworld(grid_size)
    elif env_type == 'obstacle':
        env = create_obstacle_gridworld(grid_size)
    elif env_type == 'stochastic':
        env = create_stochastic_gridworld(grid_size)
    else:
        raise ValueError("env_type must be 'simple', 'obstacle', or 'stochastic'")
    
    state_size = env.observation_space.n
    action_size = env.action_space.n
    
    print(f"State space size: {state_size}")
    print(f"Action space size: {action_size}")
    print(f"Environment type: {env_type}")
    print()
    
    # Create agents
    sarsa_agent = SARSAAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    qlearning_agent = QLearningAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )
    
    # Train agents
    print("Training SARSA agent...")
    sarsa_stats = sarsa_agent.train(env, num_episodes, max_steps, verbose=True)
    
    # Reset environment for Q-Learning
    env.reset()
    
    print("\nTraining Q-Learning agent...")
    qlearning_stats = qlearning_agent.train(env, num_episodes, max_steps, verbose=True)
    
    # Evaluate agents
    print("\nEvaluating agents...")
    sarsa_eval = sarsa_agent.evaluate(env, num_episodes=100, max_steps=max_steps)
    qlearning_eval = qlearning_agent.evaluate(env, num_episodes=100, max_steps=max_steps)
    
    # Print results
    print(f"\nEvaluation Results ({env_type} GridWorld):")
    print("-" * 50)
    print(f"SARSA - Mean Reward: {sarsa_eval['mean_reward']:.2f} ± {sarsa_eval['std_reward']:.2f}")
    print(f"SARSA - Success Rate: {sarsa_eval['success_rate']:.2%}")
    print(f"SARSA - Mean Episode Length: {sarsa_eval['mean_length']:.2f}")
    print()
    print(f"Q-Learning - Mean Reward: {qlearning_eval['mean_reward']:.2f} ± {qlearning_eval['std_reward']:.2f}")
    print(f"Q-Learning - Success Rate: {qlearning_eval['success_rate']:.2%}")
    print(f"Q-Learning - Mean Episode Length: {qlearning_eval['mean_length']:.2f}")
    
    # Plot learning curves
    plot_learning_curves(sarsa_stats, qlearning_stats, 
                        window_size=50, 
                        save_path=f'plots/gridworld_{env_type}_comparison.png')
    
    # Plot policy comparison
    plot_policy_comparison(sarsa_agent, qlearning_agent, env, 
                          save_path=f'plots/gridworld_{env_type}_policies.png')
    
    # Plot Q-value heatmaps
    plot_q_value_heatmaps(sarsa_agent, qlearning_agent, env, 
                         save_path=f'plots/gridworld_{env_type}_q_values.png')
    
    return sarsa_agent, qlearning_agent, sarsa_stats, qlearning_stats


def plot_policy_comparison(sarsa_agent, qlearning_agent, env, save_path: str = None):
    """Plot policy comparison for GridWorld."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Get policies
    sarsa_policy = sarsa_agent.get_policy()
    qlearning_policy = qlearning_agent.get_policy()
    
    # Get environment dimensions
    n_states = env.observation_space.n
    grid_size = int(np.sqrt(n_states))
    
    # Action symbols
    action_symbols = ['↑', '↓', '←', '→']
    
    # Reshape policies to grid
    sarsa_policy_grid = sarsa_policy.reshape(grid_size, grid_size)
    qlearning_policy_grid = qlearning_policy.reshape(grid_size, grid_size)
    
    # Plot SARSA policy
    im1 = axes[0].imshow(sarsa_policy_grid, cmap='tab10', aspect='equal')
    axes[0].set_title('SARSA Policy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Column')
    axes[0].set_ylabel('Row')
    
    # Add action symbols
    for i in range(grid_size):
        for j in range(grid_size):
            action = sarsa_policy_grid[i, j]
            axes[0].text(j, i, action_symbols[action], ha="center", va="center", 
                        color="white", fontsize=16, weight='bold')
    
    # Plot Q-Learning policy
    im2 = axes[1].imshow(qlearning_policy_grid, cmap='tab10', aspect='equal')
    axes[1].set_title('Q-Learning Policy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Column')
    axes[1].set_ylabel('Row')
    
    # Add action symbols
    for i in range(grid_size):
        for j in range(grid_size):
            action = qlearning_policy_grid[i, j]
            axes[1].text(j, i, action_symbols[action], ha="center", va="center", 
                        color="white", fontsize=16, weight='bold')
    
    # Add legend
    action_names = ['Up', 'Down', 'Left', 'Right']
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=plt.cm.tab10(i), label=action_names[i]) 
                      for i in range(4)]
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=4)
    
    plt.suptitle('Policy Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_q_value_heatmaps(sarsa_agent, qlearning_agent, env, save_path: str = None):
    """Plot Q-value heatmaps for GridWorld."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Get Q-values
    sarsa_q = sarsa_agent.get_q_values()
    qlearning_q = qlearning_agent.get_q_values()
    
    # Get environment dimensions
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    # Calculate grid dimensions
    grid_size = int(np.sqrt(n_states))
    
    # Plot Q-values for each action
    action_names = ['Up', 'Down', 'Left', 'Right']
    
    for action in range(n_actions):
        row = action // 2
        col = action % 2
        
        # Reshape Q-values to grid
        sarsa_q_grid = sarsa_q[:, action].reshape(grid_size, grid_size)
        qlearning_q_grid = qlearning_q[:, action].reshape(grid_size, grid_size)
        
        # Plot SARSA Q-values
        im1 = axes[row, col].imshow(sarsa_q_grid, cmap='RdYlBu', aspect='equal')
        axes[row, col].set_title(f'SARSA Q-values - {action_names[action]}', fontweight='bold')
        axes[row, col].set_xlabel('Column')
        axes[row, col].set_ylabel('Row')
        
        # Add colorbar
        plt.colorbar(im1, ax=axes[row, col])
        
        # Add text annotations
        for i in range(grid_size):
            for j in range(grid_size):
                text = axes[row, col].text(j, i, f'{sarsa_q_grid[i, j]:.1f}',
                                         ha="center", va="center", color="black", fontsize=8)
    
    plt.suptitle('Q-Value Heatmaps Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_convergence_analysis(sarsa_stats, qlearning_stats, window_size: int = 50):
    """Plot detailed convergence analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Calculate moving averages
    sarsa_ma = np.convolve(sarsa_stats['episode_rewards'], 
                          np.ones(window_size)/window_size, mode='valid')
    qlearning_ma = np.convolve(qlearning_stats['episode_rewards'], 
                              np.ones(window_size)/window_size, mode='valid')
    
    # Plot 1: Learning curves with moving averages
    axes[0, 0].plot(sarsa_stats['episode_rewards'], alpha=0.3, color='blue', label='SARSA Raw')
    axes[0, 0].plot(qlearning_stats['episode_rewards'], alpha=0.3, color='red', label='Q-Learning Raw')
    axes[0, 0].plot(range(window_size-1, len(sarsa_stats['episode_rewards'])), 
                   sarsa_ma, color='blue', linewidth=2, label='SARSA MA')
    axes[0, 0].plot(range(window_size-1, len(qlearning_stats['episode_rewards'])), 
                   qlearning_ma, color='red', linewidth=2, label='Q-Learning MA')
    axes[0, 0].set_title('Learning Curves')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Convergence speed
    sarsa_std = np.array([np.std(sarsa_stats['episode_rewards'][i:i+window_size]) 
                         for i in range(len(sarsa_stats['episode_rewards'])-window_size+1)])
    qlearning_std = np.array([np.std(qlearning_stats['episode_rewards'][i:i+window_size]) 
                             for i in range(len(qlearning_stats['episode_rewards'])-window_size+1)])
    
    axes[0, 1].plot(sarsa_std, color='blue', label='SARSA')
    axes[0, 1].plot(qlearning_std, color='red', label='Q-Learning')
    axes[0, 1].set_title('Reward Standard Deviation (Convergence)')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Std Dev')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Epsilon decay
    axes[1, 0].plot(sarsa_stats['epsilon_history'], color='blue', label='SARSA')
    axes[1, 0].plot(qlearning_stats['epsilon_history'], color='red', label='Q-Learning')
    axes[1, 0].set_title('Exploration Rate Decay')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Epsilon')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Learning rate decay
    axes[1, 1].plot(sarsa_stats['learning_rate_history'], color='blue', label='SARSA')
    axes[1, 1].plot(qlearning_stats['learning_rate_history'], color='red', label='Q-Learning')
    axes[1, 1].set_title('Learning Rate Decay')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Convergence Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def run_comprehensive_comparison():
    """Run comprehensive comparison across different environments."""
    print("Running Comprehensive SARSA vs Q-Learning Comparison")
    print("=" * 60)
    
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Test different GridWorld configurations
    environments = [
        ('simple', 'Simple GridWorld (no obstacles)'),
        ('obstacle', 'GridWorld with obstacles'),
        ('stochastic', 'Stochastic GridWorld')
    ]
    
    all_results = {}
    
    for env_type, description in environments:
        print(f"\n{description}")
        print("-" * 40)
        
        sarsa_agent, qlearning_agent, sarsa_stats, qlearning_stats = compare_algorithms_on_gridworld(
            env_type=env_type,
            num_episodes=1000,
            max_steps=100,
            grid_size=5
        )
        
        # Store results
        all_results[env_type] = {
            'sarsa_agent': sarsa_agent,
            'qlearning_agent': qlearning_agent,
            'sarsa_stats': sarsa_stats,
            'qlearning_stats': qlearning_stats
        }
        
        # Plot convergence analysis
        plot_convergence_analysis(sarsa_stats, qlearning_stats)
    
    # Summary comparison
    plot_summary_comparison(all_results)
    
    return all_results


def plot_summary_comparison(all_results):
    """Plot summary comparison across all environments."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    env_names = list(all_results.keys())
    sarsa_final_rewards = []
    qlearning_final_rewards = []
    sarsa_success_rates = []
    qlearning_success_rates = []
    
    for env_type in env_names:
        sarsa_stats = all_results[env_type]['sarsa_stats']
        qlearning_stats = all_results[env_type]['qlearning_stats']
        
        # Calculate final performance (last 100 episodes)
        sarsa_final = np.mean(sarsa_stats['episode_rewards'][-100:])
        qlearning_final = np.mean(qlearning_stats['episode_rewards'][-100:])
        
        sarsa_final_rewards.append(sarsa_final)
        qlearning_final_rewards.append(qlearning_final)
        
        # Calculate success rates
        sarsa_success = np.mean([r > 0 for r in sarsa_stats['episode_rewards'][-100:]])
        qlearning_success = np.mean([r > 0 for r in qlearning_stats['episode_rewards'][-100:]])
        
        sarsa_success_rates.append(sarsa_success)
        qlearning_success_rates.append(qlearning_success)
    
    # Plot 1: Final performance comparison
    x = np.arange(len(env_names))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, sarsa_final_rewards, width, label='SARSA', alpha=0.8)
    axes[0, 0].bar(x + width/2, qlearning_final_rewards, width, label='Q-Learning', alpha=0.8)
    axes[0, 0].set_title('Final Performance Comparison')
    axes[0, 0].set_xlabel('Environment')
    axes[0, 0].set_ylabel('Mean Reward (last 100 episodes)')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(env_names)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Success rate comparison
    axes[0, 1].bar(x - width/2, sarsa_success_rates, width, label='SARSA', alpha=0.8)
    axes[0, 1].bar(x + width/2, qlearning_success_rates, width, label='Q-Learning', alpha=0.8)
    axes[0, 1].set_title('Success Rate Comparison')
    axes[0, 1].set_xlabel('Environment')
    axes[0, 1].set_ylabel('Success Rate')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(env_names)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Performance difference
    performance_diff = np.array(sarsa_final_rewards) - np.array(qlearning_final_rewards)
    colors = ['green' if diff > 0 else 'red' for diff in performance_diff]
    axes[1, 0].bar(x, performance_diff, color=colors, alpha=0.8)
    axes[1, 0].set_title('Performance Difference (SARSA - Q-Learning)')
    axes[1, 0].set_xlabel('Environment')
    axes[1, 0].set_ylabel('Reward Difference')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(env_names)
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Convergence speed comparison
    convergence_episodes = []
    for env_type in env_names:
        sarsa_stats = all_results[env_type]['sarsa_stats']
        qlearning_stats = all_results[env_type]['qlearning_stats']
        
        # Find convergence point (when moving average stabilizes)
        sarsa_ma = np.convolve(sarsa_stats['episode_rewards'], 
                              np.ones(50)/50, mode='valid')
        qlearning_ma = np.convolve(qlearning_stats['episode_rewards'], 
                                  np.ones(50)/50, mode='valid')
        
        sarsa_conv = find_convergence_point(sarsa_ma)
        qlearning_conv = find_convergence_point(qlearning_ma)
        
        convergence_episodes.append([sarsa_conv, qlearning_conv])
    
    convergence_episodes = np.array(convergence_episodes)
    axes[1, 1].bar(x - width/2, convergence_episodes[:, 0], width, label='SARSA', alpha=0.8)
    axes[1, 1].bar(x + width/2, convergence_episodes[:, 1], width, label='Q-Learning', alpha=0.8)
    axes[1, 1].set_title('Convergence Speed')
    axes[1, 1].set_xlabel('Environment')
    axes[1, 1].set_ylabel('Convergence Episode')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(env_names)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Comprehensive Algorithm Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/comprehensive_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def find_convergence_point(ma_rewards, threshold: float = 0.1, window: int = 50):
    """Find the episode where the algorithm converges."""
    for i in range(window, len(ma_rewards)):
        recent_std = np.std(ma_rewards[i-window:i])
        if recent_std < threshold:
            return i
    return len(ma_rewards)


if __name__ == "__main__":
    # Run comprehensive comparison
    results = run_comprehensive_comparison()
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE!")
    print("="*60)
    print("Key Insights:")
    print("1. SARSA tends to learn safer policies in stochastic environments")
    print("2. Q-Learning may converge faster but can be more risky")
    print("3. The choice between algorithms depends on the environment characteristics")
    print("4. Check the 'plots' directory for detailed visualizations")
