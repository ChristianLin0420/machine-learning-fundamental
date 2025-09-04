"""
FrozenLake Q-Learning Demo
==========================

Implementation of Q-learning on OpenAI Gym's FrozenLake environment.
Compares different exploration strategies and analyzes performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
import time

try:
    import gymnasium as gym
except ImportError:
    try:
        import gym
    except ImportError:
        print("Neither gymnasium nor gym found. Please install with: pip install gymnasium")
        exit(1)

from q_learning import QLearningAgent, compare_exploration_strategies


class FrozenLakeExperiment:
    """
    Comprehensive Q-learning experiment on FrozenLake environment.
    """
    
    def __init__(self, map_name: str = "4x4", is_slippery: bool = True, render_mode: str = None):
        """
        Initialize FrozenLake experiment.
        
        Args:
            map_name: Size of the map ("4x4" or "8x8")
            is_slippery: Whether the lake is slippery (stochastic transitions)
            render_mode: Rendering mode for visualization
        """
        self.map_name = map_name
        self.is_slippery = is_slippery
        self.render_mode = render_mode
        
        # Create environment
        try:
            # Try gymnasium first
            self.env = gym.make(
                f"FrozenLake-v1",
                map_name=map_name,
                is_slippery=is_slippery,
                render_mode=render_mode
            )
        except:
            # Fall back to old gym
            self.env = gym.make(
                f"FrozenLake-v1",
                map_name=map_name,
                is_slippery=is_slippery
            )
        
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        
        print(f"Environment: FrozenLake-{map_name}")
        print(f"States: {self.n_states}, Actions: {self.n_actions}")
        print(f"Slippery: {is_slippery}")
        
    def run_single_experiment(self, agent_params: Dict, n_episodes: int = 2000,
                            experiment_name: str = "Q-Learning") -> Dict[str, Any]:
        """
        Run a single Q-learning experiment.
        
        Args:
            agent_params: Parameters for the Q-learning agent
            n_episodes: Number of training episodes
            experiment_name: Name for the experiment
            
        Returns:
            Experiment results dictionary
        """
        print(f"\n--- Running {experiment_name} ---")
        
        # Create agent
        agent = QLearningAgent(self.n_states, self.n_actions, **agent_params)
        
        # Train agent
        start_time = time.time()
        training_stats = agent.train(
            self.env, 
            n_episodes=n_episodes, 
            verbose=True, 
            log_interval=n_episodes // 10
        )
        training_time = time.time() - start_time
        
        # Evaluate agent
        eval_stats = agent.evaluate(self.env, n_episodes=1000)
        
        # Get learned policy and values
        policy = agent.get_policy()
        state_values = agent.get_state_values()
        
        results = {
            'agent': agent,
            'training_stats': training_stats,
            'evaluation_stats': eval_stats,
            'training_time': training_time,
            'policy': policy,
            'state_values': state_values,
            'q_table': agent.q_table.copy(),
            'experiment_name': experiment_name
        }
        
        # Print results summary
        print(f"\n{experiment_name} Results:")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Final average reward: {eval_stats['avg_reward']:.3f} ± {eval_stats['std_reward']:.3f}")
        print(f"Success rate: {eval_stats['success_rate']:.1%}")
        print(f"Average episode length: {eval_stats['avg_length']:.1f}")
        
        return results
    
    def compare_exploration_strategies(self, n_episodes: int = 2000) -> Dict[str, Any]:
        """
        Compare different exploration strategies.
        
        Args:
            n_episodes: Number of episodes for each strategy
            
        Returns:
            Comparison results
        """
        strategies = {
            "High Exploration (ε=0.3)": {
                "epsilon": 0.3,
                "epsilon_decay": 0.999,
                "epsilon_min": 0.01,
                "learning_rate": 0.1
            },
            "Medium Exploration (ε=0.1)": {
                "epsilon": 0.1,
                "epsilon_decay": 0.995,
                "epsilon_min": 0.01,
                "learning_rate": 0.1
            },
            "Low Exploration (ε=0.05)": {
                "epsilon": 0.05,
                "epsilon_decay": 0.99,
                "epsilon_min": 0.01,
                "learning_rate": 0.1
            },
            "Greedy (ε=0)": {
                "epsilon": 0.0,
                "epsilon_decay": 1.0,
                "epsilon_min": 0.0,
                "learning_rate": 0.1
            }
        }
        
        results = {}
        
        for strategy_name, params in strategies.items():
            results[strategy_name] = self.run_single_experiment(
                params, n_episodes, strategy_name
            )
        
        return results
    
    def hyperparameter_study(self, n_episodes: int = 1000) -> Dict[str, Any]:
        """
        Study the effect of different hyperparameters.
        
        Args:
            n_episodes: Number of episodes for each configuration
            
        Returns:
            Hyperparameter study results
        """
        # Different learning rates
        learning_rates = [0.01, 0.05, 0.1, 0.3, 0.5]
        
        # Different discount factors
        discount_factors = [0.8, 0.9, 0.95, 0.99]
        
        results = {'learning_rate_study': {}, 'discount_factor_study': {}}
        
        print("\n=== Hyperparameter Study ===")
        
        # Learning rate study
        print("\n--- Learning Rate Study ---")
        base_params = {
            "learning_rate": 0.1,  # Will be overridden
            "discount_factor": 0.95,
            "epsilon": 0.1,
            "epsilon_decay": 0.995,
            "epsilon_min": 0.01
        }
        
        for lr in learning_rates:
            params = base_params.copy()
            params["learning_rate"] = lr
            
            result = self.run_single_experiment(
                params, n_episodes, f"Learning Rate {lr}"
            )
            results['learning_rate_study'][lr] = result
        
        # Discount factor study
        print("\n--- Discount Factor Study ---")
        base_params["learning_rate"] = 0.1  # Reset to optimal
        
        for gamma in discount_factors:
            params = base_params.copy()
            params["discount_factor"] = gamma
            
            result = self.run_single_experiment(
                params, n_episodes, f"Gamma {gamma}"
            )
            results['discount_factor_study'][gamma] = result
        
        return results
    
    def visualize_policy(self, policy: np.ndarray, title: str = "Learned Policy") -> None:
        """
        Visualize the learned policy.
        
        Args:
            policy: Policy array
            title: Title for the plot
        """
        if self.map_name == "4x4":
            grid_size = 4
        elif self.map_name == "8x8":
            grid_size = 8
        else:
            grid_size = int(np.sqrt(self.n_states))
        
        policy_grid = policy.reshape(grid_size, grid_size)
        
        # Action names and symbols
        action_names = ['↑', '↓', '←', '→']
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create a color map for the policy
        im = ax.imshow(policy_grid, cmap='tab10', alpha=0.3)
        
        # Add action arrows
        for i in range(grid_size):
            for j in range(grid_size):
                action = policy_grid[i, j]
                ax.text(j, i, action_names[int(action)], ha='center', va='center',
                       fontsize=16, fontweight='bold')
        
        ax.set_title(title, fontsize=16)
        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        ax.grid(True, linewidth=2, color='black', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_q_values(self, q_table: np.ndarray, title: str = "Q-Values Heatmap") -> None:
        """
        Visualize Q-values as heatmaps.
        
        Args:
            q_table: Q-table array
            title: Title for the plot
        """
        action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        if self.map_name == "4x4":
            grid_size = 4
        elif self.map_name == "8x8":
            grid_size = 8
        else:
            grid_size = int(np.sqrt(self.n_states))
        
        for action in range(4):
            q_values = q_table[:, action].reshape(grid_size, grid_size)
            
            im = axes[action].imshow(q_values, cmap='RdYlBu_r')
            axes[action].set_title(f'{action_names[action]} Q-values')
            
            # Add values as text
            for i in range(grid_size):
                for j in range(grid_size):
                    text = axes[action].text(j, i, f'{q_values[i, j]:.2f}',
                                           ha="center", va="center", color="black", fontsize=8)
            
            plt.colorbar(im, ax=axes[action])
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_learning_curves(self, results: Dict[str, Any]) -> None:
        """
        Plot learning curves for comparison.
        
        Args:
            results: Dictionary of experiment results
        """
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Episode rewards
        plt.subplot(1, 3, 1)
        for name, result in results.items():
            rewards = result['training_stats']['episode_rewards']
            # Smooth the curve
            window_size = len(rewards) // 50
            if window_size > 1:
                smoothed = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                plt.plot(smoothed, label=name, alpha=0.8)
            else:
                plt.plot(rewards, label=name, alpha=0.8)
        
        plt.xlabel('Episode')
        plt.ylabel('Episode Reward')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Success rates (moving average)
        plt.subplot(1, 3, 2)
        for name, result in results.items():
            rewards = np.array(result['training_stats']['episode_rewards'])
            # Calculate moving average success rate
            window = 100
            success_rate = []
            for i in range(window, len(rewards)):
                recent_rewards = rewards[i-window:i]
                success_rate.append(np.mean(recent_rewards > 0))
            
            if success_rate:
                plt.plot(range(window, len(rewards)), success_rate, label=name, alpha=0.8)
        
        plt.xlabel('Episode')
        plt.ylabel('Success Rate')
        plt.title('Success Rate (Moving Average)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Final evaluation comparison
        plt.subplot(1, 3, 3)
        names = list(results.keys())
        success_rates = [results[name]['evaluation_stats']['success_rate'] for name in names]
        avg_rewards = [results[name]['evaluation_stats']['avg_reward'] for name in names]
        
        x = np.arange(len(names))
        width = 0.35
        
        plt.bar(x - width/2, success_rates, width, label='Success Rate', alpha=0.7)
        plt.bar(x + width/2, avg_rewards, width, label='Avg Reward', alpha=0.7)
        
        plt.xlabel('Method')
        plt.ylabel('Performance')
        plt.title('Final Performance Comparison')
        plt.xticks(x, names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def main():
    """Main demonstration of Q-learning on FrozenLake."""
    
    print("FrozenLake Q-Learning Demo")
    print("==========================")
    
    # Experiment 1: Basic Q-learning on 4x4 FrozenLake
    print("\n=== Experiment 1: Basic Q-Learning (4x4) ===")
    experiment = FrozenLakeExperiment(map_name="4x4", is_slippery=True)
    
    basic_params = {
        "learning_rate": 0.1,
        "discount_factor": 0.95,
        "epsilon": 0.1,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.01
    }
    
    basic_result = experiment.run_single_experiment(basic_params, n_episodes=2000)
    
    # Visualize results
    experiment.visualize_policy(basic_result['policy'], "Basic Q-Learning Policy")
    experiment.visualize_q_values(basic_result['q_table'], "Basic Q-Learning Q-Values")
    
    # Experiment 2: Compare exploration strategies
    print("\n=== Experiment 2: Exploration Strategy Comparison ===")
    exploration_results = experiment.compare_exploration_strategies(n_episodes=2000)
    experiment.plot_learning_curves(exploration_results)
    
    # Show best strategy's policy
    best_strategy = max(exploration_results.keys(), 
                       key=lambda x: exploration_results[x]['evaluation_stats']['success_rate'])
    print(f"\nBest strategy: {best_strategy}")
    experiment.visualize_policy(exploration_results[best_strategy]['policy'], 
                              f"Policy: {best_strategy}")
    
    # Experiment 3: 8x8 FrozenLake (more challenging)
    print("\n=== Experiment 3: Challenging 8x8 FrozenLake ===")
    large_experiment = FrozenLakeExperiment(map_name="8x8", is_slippery=True)
    
    # Use parameters optimized for larger state space
    large_params = {
        "learning_rate": 0.1,
        "discount_factor": 0.95,
        "epsilon": 0.3,  # Higher exploration for larger space
        "epsilon_decay": 0.999,
        "epsilon_min": 0.01
    }
    
    large_result = large_experiment.run_single_experiment(large_params, n_episodes=5000)
    large_experiment.visualize_policy(large_result['policy'], "8x8 FrozenLake Policy")
    
    # Experiment 4: Deterministic vs Stochastic
    print("\n=== Experiment 4: Deterministic vs Stochastic Environment ===")
    
    # Deterministic environment
    det_experiment = FrozenLakeExperiment(map_name="4x4", is_slippery=False)
    det_result = det_experiment.run_single_experiment(basic_params, n_episodes=1000, 
                                                    experiment_name="Deterministic")
    
    # Compare results
    comparison_results = {
        "Stochastic": basic_result,
        "Deterministic": det_result
    }
    
    experiment.plot_learning_curves(comparison_results)
    
    print("\n=== Summary ===")
    print("Q-Learning successfully learned policies for FrozenLake environments!")
    print("Key observations:")
    print("- Higher exploration helps in stochastic environments")
    print("- Larger state spaces require more episodes and exploration")
    print("- Deterministic environments converge faster but may not generalize")
    print("- Q-learning is robust across different environment configurations")


if __name__ == "__main__":
    # Check if required packages are installed
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Visualization packages not found. Installing...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "seaborn"])
        import matplotlib.pyplot as plt
        import seaborn as sns
    
    # Run the main demo
    main()
