import numpy as np
import matplotlib.pyplot as plt
import gym
from sarsa import SARSAAgent, QLearningAgent, plot_learning_curves
import warnings
warnings.filterwarnings('ignore')


def run_frozenlake_comparison(env_name: str = 'FrozenLake-v1', 
                            num_episodes: int = 2000,
                            max_steps: int = 100,
                            learning_rate: float = 0.1,
                            discount_factor: float = 0.95,
                            epsilon: float = 1.0,
                            epsilon_decay: float = 0.995,
                            epsilon_min: float = 0.01):
    """
    Compare SARSA and Q-Learning on FrozenLake environment.
    
    Args:
        env_name: Name of the FrozenLake environment
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        learning_rate: Learning rate for both agents
        discount_factor: Discount factor for both agents
        epsilon: Initial exploration rate
        epsilon_decay: Epsilon decay rate
        epsilon_min: Minimum epsilon value
    """
    print(f"Running FrozenLake comparison: {env_name}")
    print("=" * 50)
    
    # Create environment
    env = gym.make(env_name)
    state_size = env.observation_space.n
    action_size = env.action_space.n
    
    print(f"State space size: {state_size}")
    print(f"Action space size: {action_size}")
    print(f"Environment description: {env.desc}")
    print()
    
    # Create agents
    sarsa_agent = SARSAAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min
    )
    
    qlearning_agent = QLearningAgent(
        state_size=state_size,
        action_size=action_size,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min
    )
    
    # Train SARSA agent
    print("Training SARSA agent...")
    sarsa_stats = sarsa_agent.train(env, num_episodes, max_steps, verbose=True)
    
    # Reset environment for Q-Learning
    env.reset()
    
    # Train Q-Learning agent
    print("\nTraining Q-Learning agent...")
    qlearning_stats = qlearning_agent.train(env, num_episodes, max_steps, verbose=True)
    
    # Evaluate both agents
    print("\nEvaluating agents...")
    sarsa_eval = sarsa_agent.evaluate(env, num_episodes=100, max_steps=max_steps)
    qlearning_eval = qlearning_agent.evaluate(env, num_episodes=100, max_steps=max_steps)
    
    # Print evaluation results
    print("\nEvaluation Results (100 episodes):")
    print("-" * 40)
    print(f"SARSA - Mean Reward: {sarsa_eval['mean_reward']:.2f} ± {sarsa_eval['std_reward']:.2f}")
    print(f"SARSA - Success Rate: {sarsa_eval['success_rate']:.2%}")
    print(f"SARSA - Mean Episode Length: {sarsa_eval['mean_length']:.2f}")
    print()
    print(f"Q-Learning - Mean Reward: {qlearning_eval['mean_reward']:.2f} ± {qlearning_eval['std_reward']:.2f}")
    print(f"Q-Learning - Success Rate: {qlearning_eval['success_rate']:.2%}")
    print(f"Q-Learning - Mean Episode Length: {qlearning_eval['mean_length']:.2f}")
    
    # Plot learning curves
    plot_learning_curves(sarsa_stats, qlearning_stats, 
                        window_size=100, 
                        save_path='plots/frozenlake_comparison.png')
    
    # Plot Q-value heatmaps
    plot_q_value_heatmaps(sarsa_agent, qlearning_agent, env, 
                         save_path='plots/frozenlake_q_values.png')
    
    # Plot policy comparison
    plot_policy_comparison(sarsa_agent, qlearning_agent, env, 
                          save_path='plots/frozenlake_policies.png')
    
    return sarsa_agent, qlearning_agent, sarsa_stats, qlearning_stats


def plot_q_value_heatmaps(sarsa_agent, qlearning_agent, env, save_path: str = None):
    """Plot Q-value heatmaps for both agents."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Get Q-values
    sarsa_q = sarsa_agent.get_q_values()
    qlearning_q = qlearning_agent.get_q_values()
    
    # Get environment dimensions
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    # Calculate grid dimensions (assuming square grid)
    grid_size = int(np.sqrt(n_states))
    
    # Plot Q-values for each action
    action_names = ['Left', 'Down', 'Right', 'Up']
    
    for action in range(n_actions):
        row = action // 2
        col = action % 2
        
        # Reshape Q-values to grid
        sarsa_q_grid = sarsa_q[:, action].reshape(grid_size, grid_size)
        qlearning_q_grid = qlearning_q[:, action].reshape(grid_size, grid_size)
        
        # Plot SARSA Q-values
        im1 = axes[row, col].imshow(sarsa_q_grid, cmap='RdYlBu', aspect='equal')
        axes[row, col].set_title(f'SARSA Q-values - {action_names[action]}')
        axes[row, col].set_xlabel('Column')
        axes[row, col].set_ylabel('Row')
        
        # Add colorbar
        plt.colorbar(im1, ax=axes[row, col])
        
        # Add text annotations
        for i in range(grid_size):
            for j in range(grid_size):
                text = axes[row, col].text(j, i, f'{sarsa_q_grid[i, j]:.1f}',
                                         ha="center", va="center", color="black", fontsize=8)
    
    plt.suptitle('Q-Value Heatmaps Comparison', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_policy_comparison(sarsa_agent, qlearning_agent, env, save_path: str = None):
    """Plot policy comparison between SARSA and Q-Learning."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Get policies
    sarsa_policy = sarsa_agent.get_policy()
    qlearning_policy = qlearning_agent.get_policy()
    
    # Get environment dimensions
    n_states = env.observation_space.n
    grid_size = int(np.sqrt(n_states))
    
    # Action symbols
    action_symbols = ['←', '↓', '→', '↑']
    
    # Reshape policies to grid
    sarsa_policy_grid = sarsa_policy.reshape(grid_size, grid_size)
    qlearning_policy_grid = qlearning_policy.reshape(grid_size, grid_size)
    
    # Plot SARSA policy
    im1 = axes[0].imshow(sarsa_policy_grid, cmap='tab10', aspect='equal')
    axes[0].set_title('SARSA Policy')
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
    axes[1].set_title('Q-Learning Policy')
    axes[1].set_xlabel('Column')
    axes[1].set_ylabel('Row')
    
    # Add action symbols
    for i in range(grid_size):
        for j in range(grid_size):
            action = qlearning_policy_grid[i, j]
            axes[1].text(j, i, action_symbols[action], ha="center", va="center", 
                        color="white", fontsize=16, weight='bold')
    
    # Add legend
    action_names = ['Left', 'Down', 'Right', 'Up']
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=plt.cm.tab10(i), label=action_names[i]) 
                      for i in range(4)]
    fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=4)
    
    plt.suptitle('Policy Comparison', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def analyze_convergence(sarsa_stats, qlearning_stats, window_size: int = 100):
    """Analyze convergence behavior of both algorithms."""
    print("\nConvergence Analysis:")
    print("=" * 30)
    
    # Calculate moving averages
    sarsa_ma = np.convolve(sarsa_stats['episode_rewards'], 
                          np.ones(window_size)/window_size, mode='valid')
    qlearning_ma = np.convolve(qlearning_stats['episode_rewards'], 
                              np.ones(window_size)/window_size, mode='valid')
    
    # Find convergence points (when moving average stabilizes)
    sarsa_convergence = find_convergence_point(sarsa_ma, threshold=0.1)
    qlearning_convergence = find_convergence_point(qlearning_ma, threshold=0.1)
    
    print(f"SARSA convergence point: Episode {sarsa_convergence}")
    print(f"Q-Learning convergence point: Episode {qlearning_convergence}")
    
    # Final performance
    sarsa_final = np.mean(sarsa_stats['episode_rewards'][-100:])
    qlearning_final = np.mean(qlearning_stats['episode_rewards'][-100:])
    
    print(f"SARSA final performance: {sarsa_final:.2f}")
    print(f"Q-Learning final performance: {qlearning_final:.2f}")
    
    # Stability analysis
    sarsa_std = np.std(sarsa_stats['episode_rewards'][-100:])
    qlearning_std = np.std(qlearning_stats['episode_rewards'][-100:])
    
    print(f"SARSA stability (std): {sarsa_std:.2f}")
    print(f"Q-Learning stability (std): {qlearning_std:.2f}")


def find_convergence_point(ma_rewards, threshold: float = 0.1, window: int = 50):
    """Find the episode where the algorithm converges."""
    for i in range(window, len(ma_rewards)):
        recent_std = np.std(ma_rewards[i-window:i])
        if recent_std < threshold:
            return i
    return len(ma_rewards)


def run_multiple_seeds(env_name: str = 'FrozenLake-v1', 
                      num_episodes: int = 1000,
                      num_seeds: int = 5):
    """Run multiple seeds to get statistical significance."""
    print(f"Running {num_seeds} seeds for statistical analysis...")
    
    all_sarsa_rewards = []
    all_qlearning_rewards = []
    
    for seed in range(num_seeds):
        print(f"\nSeed {seed + 1}/{num_seeds}")
        
        # Set random seed
        np.random.seed(seed)
        
        # Create environment
        env = gym.make(env_name)
        state_size = env.observation_space.n
        action_size = env.action_space.n
        
        # Create agents
        sarsa_agent = SARSAAgent(state_size, action_size)
        qlearning_agent = QLearningAgent(state_size, action_size)
        
        # Train agents
        sarsa_stats = sarsa_agent.train(env, num_episodes, verbose=False)
        qlearning_stats = qlearning_agent.train(env, num_episodes, verbose=False)
        
        # Store final performance
        all_sarsa_rewards.append(sarsa_stats['episode_rewards'][-100:])
        all_qlearning_rewards.append(qlearning_stats['episode_rewards'][-100:])
    
    # Calculate statistics
    sarsa_mean = np.mean([np.mean(rewards) for rewards in all_sarsa_rewards])
    sarsa_std = np.std([np.mean(rewards) for rewards in all_sarsa_rewards])
    qlearning_mean = np.mean([np.mean(rewards) for rewards in all_qlearning_rewards])
    qlearning_std = np.std([np.mean(rewards) for rewards in all_qlearning_rewards])
    
    print(f"\nStatistical Results ({num_seeds} seeds):")
    print("-" * 40)
    print(f"SARSA: {sarsa_mean:.2f} ± {sarsa_std:.2f}")
    print(f"Q-Learning: {qlearning_mean:.2f} ± {qlearning_std:.2f}")
    
    # Statistical significance test
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(
        [np.mean(rewards) for rewards in all_sarsa_rewards],
        [np.mean(rewards) for rewards in all_qlearning_rewards]
    )
    
    print(f"T-test p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Difference is statistically significant (p < 0.05)")
    else:
        print("Difference is not statistically significant (p >= 0.05)")


if __name__ == "__main__":
    # Create plots directory
    import os
    os.makedirs('plots', exist_ok=True)
    
    # Run FrozenLake comparison
    sarsa_agent, qlearning_agent, sarsa_stats, qlearning_stats = run_frozenlake_comparison(
        env_name='FrozenLake-v1',
        num_episodes=2000,
        max_steps=100
    )
    
    # Analyze convergence
    analyze_convergence(sarsa_stats, qlearning_stats)
    
    # Run multiple seeds for statistical analysis
    run_multiple_seeds(num_episodes=1000, num_seeds=5)
