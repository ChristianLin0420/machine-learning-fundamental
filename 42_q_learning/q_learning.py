"""
Q-Learning Agent Implementation
==============================

Core Q-learning algorithm with epsilon-greedy exploration strategy.
Implements off-policy temporal difference learning for optimal control.
"""

import numpy as np
from typing import Tuple, Dict, Any
import random
from collections import defaultdict


class QLearningAgent:
    """
    Q-Learning Agent implementing off-policy TD control.
    
    Q-Learning Update Rule:
    Q(s,a) ← Q(s,a) + α(r + γ max_a' Q(s',a') - Q(s,a))
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        """
        Initialize Q-Learning Agent.
        
        Args:
            n_states: Number of states in the environment
            n_actions: Number of actions available
            learning_rate: Learning rate α (0 < α ≤ 1)
            discount_factor: Discount factor γ (0 ≤ γ < 1)
            epsilon: Initial exploration rate for ε-greedy
            epsilon_decay: Decay rate for epsilon
            epsilon_min: Minimum epsilon value
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((n_states, n_actions))
        
        # Statistics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.q_value_history = []
        
    def get_action(self, state: int, greedy: bool = False) -> int:
        """
        Select action using ε-greedy policy.
        
        Args:
            state: Current state
            greedy: If True, use greedy policy (no exploration)
            
        Returns:
            Selected action
        """
        if greedy or np.random.random() > self.epsilon:
            # Greedy action: choose action with highest Q-value
            return np.argmax(self.q_table[state])
        else:
            # Random action for exploration
            return np.random.randint(self.n_actions)
    
    def update_q_value(self, state: int, action: int, reward: float, 
                      next_state: int, done: bool) -> None:
        """
        Update Q-value using Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is terminated
        """
        current_q = self.q_table[state, action]
        
        if done:
            # Terminal state: no future rewards
            td_target = reward
        else:
            # Non-terminal: max Q-value of next state
            td_target = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        # Q-learning update
        td_error = td_target - current_q
        self.q_table[state, action] += self.learning_rate * td_error
    
    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train(self, env, n_episodes: int = 1000, max_steps: int = 200,
              verbose: bool = True, log_interval: int = 100) -> Dict[str, Any]:
        """
        Train the Q-learning agent.
        
        Args:
            env: Environment to train on
            n_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            verbose: Whether to print progress
            log_interval: Episodes between progress logs
            
        Returns:
            Training statistics dictionary
        """
        self.episode_rewards = []
        self.episode_lengths = []
        
        for episode in range(n_episodes):
            state = env.reset()
            if isinstance(state, tuple):  # Handle OpenAI Gym environments
                state = state[0]
                
            total_reward = 0
            steps = 0
            
            for step in range(max_steps):
                # Select and execute action
                action = self.get_action(state)
                result = env.step(action)
                
                # Handle different environment return formats
                if len(result) == 4:
                    next_state, reward, done, _ = result
                elif len(result) == 5:
                    next_state, reward, terminated, truncated, _ = result
                    done = terminated or truncated
                else:
                    raise ValueError(f"Unexpected environment step result: {result}")
                
                # Update Q-value
                self.update_q_value(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            # Decay exploration rate
            self.decay_epsilon()
            
            # Record episode statistics
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            
            # Log progress
            if verbose and (episode + 1) % log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-log_interval:])
                avg_length = np.mean(self.episode_lengths[-log_interval:])
                print(f"Episode {episode + 1:4d} | "
                      f"Avg Reward: {avg_reward:6.2f} | "
                      f"Avg Length: {avg_length:5.1f} | "
                      f"Epsilon: {self.epsilon:.3f}")
        
        # Calculate final statistics
        final_100_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
        
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'final_avg_reward': np.mean(final_100_rewards),
            'final_epsilon': self.epsilon,
            'q_table': self.q_table.copy()
        }
        
        return stats
    
    def evaluate(self, env, n_episodes: int = 100, render: bool = False) -> Dict[str, float]:
        """
        Evaluate the learned policy.
        
        Args:
            env: Environment for evaluation
            n_episodes: Number of evaluation episodes
            render: Whether to render the environment
            
        Returns:
            Evaluation statistics
        """
        eval_rewards = []
        eval_lengths = []
        success_rate = 0
        
        for episode in range(n_episodes):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]
                
            total_reward = 0
            steps = 0
            
            while steps < 200:  # Max steps limit
                action = self.get_action(state, greedy=True)  # Use greedy policy
                result = env.step(action)
                
                if len(result) == 4:
                    next_state, reward, done, _ = result
                elif len(result) == 5:
                    next_state, reward, terminated, truncated, _ = result
                    done = terminated or truncated
                
                if render:
                    env.render()
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    if reward > 0:  # Assume positive reward indicates success
                        success_rate += 1
                    break
            
            eval_rewards.append(total_reward)
            eval_lengths.append(steps)
        
        return {
            'avg_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'avg_length': np.mean(eval_lengths),
            'success_rate': success_rate / n_episodes
        }
    
    def get_policy(self) -> np.ndarray:
        """
        Extract the learned policy.
        
        Returns:
            Policy array where policy[s] is the best action for state s
        """
        return np.argmax(self.q_table, axis=1)
    
    def get_state_values(self) -> np.ndarray:
        """
        Calculate state values V(s) = max_a Q(s,a).
        
        Returns:
            State value array
        """
        return np.max(self.q_table, axis=1)


class QLearningAgentDict:
    """
    Q-Learning Agent using dictionary for environments with complex state spaces.
    """
    
    def __init__(
        self,
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Use defaultdict for Q-table
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        
    def get_action(self, state, greedy: bool = False) -> int:
        """Get action using ε-greedy policy."""
        if greedy or np.random.random() > self.epsilon:
            return np.argmax(self.q_table[state])
        else:
            return np.random.randint(self.n_actions)
    
    def update_q_value(self, state, action: int, reward: float, 
                      next_state, done: bool) -> None:
        """Update Q-value using Q-learning rule."""
        current_q = self.q_table[state][action]
        
        if done:
            td_target = reward
        else:
            td_target = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        td_error = td_target - current_q
        self.q_table[state][action] += self.learning_rate * td_error
    
    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def compare_exploration_strategies(env, strategies: Dict[str, Dict], 
                                 n_episodes: int = 1000) -> Dict[str, Any]:
    """
    Compare different exploration strategies.
    
    Args:
        env: Environment to test on
        strategies: Dictionary of strategy names and parameters
        n_episodes: Number of episodes to run
        
    Returns:
        Comparison results
    """
    results = {}
    
    for strategy_name, params in strategies.items():
        print(f"\n--- Training with {strategy_name} ---")
        
        # Get environment dimensions
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        
        # Create agent with specified parameters
        agent = QLearningAgent(n_states, n_actions, **params)
        
        # Train agent
        stats = agent.train(env, n_episodes=n_episodes, verbose=True)
        
        # Evaluate performance
        eval_stats = agent.evaluate(env, n_episodes=100)
        
        results[strategy_name] = {
            'training_stats': stats,
            'evaluation_stats': eval_stats,
            'agent': agent
        }
        
        print(f"Final Performance - Avg Reward: {eval_stats['avg_reward']:.3f}, "
              f"Success Rate: {eval_stats['success_rate']:.1%}")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Q-Learning Agent Implementation")
    print("==============================")
    
    # Check if we can import the environment
    try:
        from gridworld_env import create_simple_gridworld
        
        print("Running Q-Learning demonstration...")
        print()
        
        # Create a simple environment
        env = create_simple_gridworld()
        print(f"Environment: {env.height}×{env.width} GridWorld")
        print(f"Start: {env.start_pos}, Goal: {env.goal_pos}")
        print(f"Obstacles: {env.obstacles}")
        print(f"Cliffs: {env.cliffs}")
        print()
        
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
        
        print(f"Agent initialized with {agent.n_states} states and {agent.n_actions} actions")
        print()
        
        # Train the agent
        print("Training Q-learning agent...")
        stats = agent.train(env, n_episodes=500, verbose=True, log_interval=100)
        
        print()
        print("Training completed!")
        print(f"Final average reward (last 100 episodes): {stats['final_avg_reward']:.3f}")
        print(f"Final exploration rate: {stats['final_epsilon']:.3f}")
        print()
        
        # Evaluate the learned policy
        print("Evaluating learned policy...")
        eval_stats = agent.evaluate(env, n_episodes=100)
        
        print("Evaluation Results:")
        print(f"  Success Rate: {eval_stats['success_rate']:.1%}")
        print(f"  Average Reward: {eval_stats['avg_reward']:.3f} ± {eval_stats['std_reward']:.3f}")
        print(f"  Average Episode Length: {eval_stats['avg_length']:.1f} steps")
        print()
        
        # Show the learned policy
        policy = agent.get_policy()
        state_values = agent.get_state_values()
        
        print("Learned Policy (actions for each state):")
        action_names = ['↑', '↓', '←', '→']
        for i in range(env.height):
            row_actions = []
            for j in range(env.width):
                state = i * env.width + j
                pos = (i, j)
                if pos in env.obstacles:
                    row_actions.append('#')
                elif pos in env.cliffs:
                    row_actions.append('X')
                elif pos == env.goal_pos:
                    row_actions.append('G')
                else:
                    action = policy[state]
                    row_actions.append(action_names[action])
            print(' '.join(f'{a:>2}' for a in row_actions))
        print()
        
        print("State Values (max Q-value for each state):")
        for i in range(env.height):
            row_values = []
            for j in range(env.width):
                state = i * env.width + j
                pos = (i, j)
                if pos in env.obstacles:
                    row_values.append('#####')
                elif pos in env.cliffs:
                    row_values.append(' XXX ')
                elif pos == env.goal_pos:
                    row_values.append(' GOL ')
                else:
                    value = state_values[state]
                    row_values.append(f'{value:5.1f}')
            print(' '.join(row_values))
        print()
        
        # Show optimal path
        path = env.get_optimal_path(policy)
        print(f"Optimal path from start to goal ({len(path)} steps):")
        print(f"Path: {' → '.join(str(pos) for pos in path[:10])}")
        if len(path) > 10:
            print(f"... (showing first 10 steps of {len(path)} total)")
        print()
        
        # Show Q-table statistics
        print("Q-table Statistics:")
        print(f"  Q-values range: [{np.min(agent.q_table):.3f}, {np.max(agent.q_table):.3f}]")
        print(f"  Average Q-value: {np.mean(agent.q_table):.3f}")
        print(f"  Non-zero Q-values: {np.count_nonzero(agent.q_table)}/{agent.q_table.size}")
        print()
        
        print("✅ Q-learning demonstration completed successfully!")
        print("For more advanced demonstrations, run:")
        print("  • python demo.py - Complete multi-environment demo")
        print("  • python frozenlake_demo.py - FrozenLake experiments")
        print("  • python gridworld_env.py - GridWorld environment demo")
        print("  • python plot_results.py - Visualization demo")
        
    except ImportError as e:
        print(f"Could not import required modules: {e}")
        print("This module provides Q-learning agents for tabular reinforcement learning.")
        print("To see it in action, make sure gridworld_env.py is in the same directory.")
        print()
        
        # Show basic agent creation and usage
        print("Basic Usage Example:")
        print("```python")
        print("from q_learning import QLearningAgent")
        print("")
        print("# Create agent")
        print("agent = QLearningAgent(")
        print("    n_states=25,     # 5x5 grid = 25 states")
        print("    n_actions=4,     # up, down, left, right")
        print("    learning_rate=0.1,")
        print("    discount_factor=0.95,")
        print("    epsilon=0.1      # exploration rate")
        print(")")
        print("")
        print("# Train on your environment")
        print("stats = agent.train(env, n_episodes=1000)")
        print("")
        print("# Evaluate performance")
        print("eval_stats = agent.evaluate(env, n_episodes=100)")
        print("print(f'Success rate: {eval_stats[\"success_rate\"]:.1%}')")
        print("```")
        print()
        
        # Demonstrate basic Q-learning components
        print("Demonstrating core Q-learning components:")
        print()
        
        # Create a dummy agent
        agent = QLearningAgent(n_states=10, n_actions=4)
        print(f"✓ Created Q-learning agent with Q-table shape: {agent.q_table.shape}")
        
        # Show action selection
        state = 0
        action = agent.get_action(state)
        print(f"✓ Action selection (ε-greedy): state {state} → action {action}")
        
        # Show Q-value update
        agent.update_q_value(state=0, action=1, reward=1.0, next_state=1, done=False)
        print(f"✓ Q-value update: Q(0,1) = {agent.q_table[0, 1]:.3f}")
        
        # Show epsilon decay
        initial_epsilon = agent.epsilon
        agent.decay_epsilon()
        print(f"✓ Epsilon decay: {initial_epsilon:.3f} → {agent.epsilon:.3f}")
        
        print()
        print("✅ Core Q-learning functionality verified!")
