import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from typing import Tuple, Dict, List, Optional
import gym
from gym import spaces


class SARSAAgent:
    """
    SARSA (State-Action-Reward-State-Action) agent implementation.
    
    SARSA is an on-policy temporal difference learning algorithm that learns
    the action-value function Q(s,a) by following the current policy.
    """
    
    def __init__(self, 
                 state_size: int, 
                 action_size: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 learning_rate_decay: float = 0.999,
                 learning_rate_min: float = 0.01):
        """
        Initialize SARSA agent.
        
        Args:
            state_size: Number of states in the environment
            action_size: Number of actions available
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Initial exploration rate
            epsilon_decay: Rate of epsilon decay
            epsilon_min: Minimum epsilon value
            learning_rate_decay: Rate of learning rate decay
            learning_rate_min: Minimum learning rate
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_min = learning_rate_min
        
        # Initialize Q-table
        self.q_table = np.zeros((state_size, action_size))
        
        # Track learning statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.epsilon_history = []
        self.learning_rate_history = []
        
    def get_action(self, state: int, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode (affects exploration)
            
        Returns:
            Selected action
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.q_table[state])
    
    def update(self, state: int, action: int, reward: float, 
               next_state: int, next_action: int, done: bool):
        """
        Update Q-table using SARSA update rule.
        
        Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_action: Next action (selected by current policy)
            done: Whether episode is finished
        """
        # Current Q-value
        current_q = self.q_table[state, action]
        
        # Next Q-value (from action selected by current policy)
        if done:
            next_q = 0
        else:
            next_q = self.q_table[next_state, next_action]
        
        # TD target
        td_target = reward + self.discount_factor * next_q
        
        # TD error
        td_error = td_target - current_q
        
        # Update Q-value
        self.q_table[state, action] += self.learning_rate * td_error
    
    def decay_parameters(self):
        """Decay exploration and learning rate parameters."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        if self.learning_rate > self.learning_rate_min:
            self.learning_rate *= self.learning_rate_decay
    
    def train_episode(self, env, max_steps: int = 1000) -> Tuple[float, int]:
        """
        Train for one episode.
        
        Args:
            env: Environment to train on
            max_steps: Maximum steps per episode
            
        Returns:
            Total reward and episode length
        """
        state = env.reset()
        if hasattr(state, '__len__') and len(state) > 1:
            state = state[0] if isinstance(state, tuple) else state
        
        total_reward = 0
        steps = 0
        
        # Select initial action
        action = self.get_action(state, training=True)
        
        for _ in range(max_steps):
            # Take action and observe result
            result = env.step(action)
            if len(result) == 4:
                next_state, reward, done, info = result
            else:
                next_state, reward, done, truncated, info = result
                done = done or truncated
            
            # Select next action using current policy
            next_action = self.get_action(next_state, training=True)
            
            # Update Q-values
            self.update(state, action, reward, next_state, next_action, done)
            
            # Update statistics
            total_reward += reward
            steps += 1
            
            # Move to next state
            state = next_state
            action = next_action
            
            if done:
                break
        
        # Decay parameters
        self.decay_parameters()
        
        # Record statistics
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(steps)
        self.epsilon_history.append(self.epsilon)
        self.learning_rate_history.append(self.learning_rate)
        
        return total_reward, steps
    
    def train(self, env, num_episodes: int, max_steps: int = 1000, 
              verbose: bool = True) -> Dict:
        """
        Train the agent for multiple episodes.
        
        Args:
            env: Environment to train on
            num_episodes: Number of episodes to train
            max_steps: Maximum steps per episode
            verbose: Whether to print progress
            
        Returns:
            Training statistics
        """
        print(f"Training SARSA agent for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            reward, steps = self.train_episode(env, max_steps)
            
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}, LR: {self.learning_rate:.4f}")
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'epsilon_history': self.epsilon_history,
            'learning_rate_history': self.learning_rate_history
        }
    
    def evaluate(self, env, num_episodes: int = 100, max_steps: int = 1000) -> Dict:
        """
        Evaluate the trained agent.
        
        Args:
            env: Environment to evaluate on
            num_episodes: Number of episodes to evaluate
            max_steps: Maximum steps per episode
            
        Returns:
            Evaluation statistics
        """
        eval_rewards = []
        eval_lengths = []
        
        for episode in range(num_episodes):
            state = env.reset()
            if hasattr(state, '__len__') and len(state) > 1:
                state = state[0] if isinstance(state, tuple) else state
            
            total_reward = 0
            steps = 0
            
            for _ in range(max_steps):
                action = self.get_action(state, training=False)
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
            
            eval_rewards.append(total_reward)
            eval_lengths.append(steps)
        
        return {
            'eval_rewards': eval_rewards,
            'eval_lengths': eval_lengths,
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'success_rate': np.mean([r > 0 for r in eval_rewards])
        }
    
    def get_policy(self) -> np.ndarray:
        """Get the current policy (greedy)."""
        return np.argmax(self.q_table, axis=1)
    
    def get_q_values(self) -> np.ndarray:
        """Get the current Q-values."""
        return self.q_table.copy()


class QLearningAgent:
    """
    Q-Learning agent for comparison with SARSA.
    
    Q-Learning is an off-policy algorithm that learns the optimal action-value
    function by using the maximum Q-value of the next state.
    """
    
    def __init__(self, 
                 state_size: int, 
                 action_size: int,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 learning_rate_decay: float = 0.999,
                 learning_rate_min: float = 0.01):
        """Initialize Q-Learning agent with same parameters as SARSA."""
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.learning_rate_decay = learning_rate_decay
        self.learning_rate_min = learning_rate_min
        
        # Initialize Q-table
        self.q_table = np.zeros((state_size, action_size))
        
        # Track learning statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.epsilon_history = []
        self.learning_rate_history = []
    
    def get_action(self, state: int, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.q_table[state])
    
    def update(self, state: int, action: int, reward: float, 
               next_state: int, done: bool):
        """
        Update Q-table using Q-Learning update rule.
        
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        """
        # Current Q-value
        current_q = self.q_table[state, action]
        
        # Next Q-value (maximum over all actions)
        if done:
            next_q = 0
        else:
            next_q = np.max(self.q_table[next_state])
        
        # TD target
        td_target = reward + self.discount_factor * next_q
        
        # TD error
        td_error = td_target - current_q
        
        # Update Q-value
        self.q_table[state, action] += self.learning_rate * td_error
    
    def decay_parameters(self):
        """Decay exploration and learning rate parameters."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        if self.learning_rate > self.learning_rate_min:
            self.learning_rate *= self.learning_rate_decay
    
    def train_episode(self, env, max_steps: int = 1000) -> Tuple[float, int]:
        """Train for one episode."""
        state = env.reset()
        if hasattr(state, '__len__') and len(state) > 1:
            state = state[0] if isinstance(state, tuple) else state
        
        total_reward = 0
        steps = 0
        
        for _ in range(max_steps):
            # Select action
            action = self.get_action(state, training=True)
            
            # Take action and observe result
            result = env.step(action)
            if len(result) == 4:
                next_state, reward, done, info = result
            else:
                next_state, reward, done, truncated, info = result
                done = done or truncated
            
            # Update Q-values
            self.update(state, action, reward, next_state, done)
            
            # Update statistics
            total_reward += reward
            steps += 1
            state = next_state
            
            if done:
                break
        
        # Decay parameters
        self.decay_parameters()
        
        # Record statistics
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(steps)
        self.epsilon_history.append(self.epsilon)
        self.learning_rate_history.append(self.learning_rate)
        
        return total_reward, steps
    
    def train(self, env, num_episodes: int, max_steps: int = 1000, 
              verbose: bool = True) -> Dict:
        """Train the agent for multiple episodes."""
        print(f"Training Q-Learning agent for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            reward, steps = self.train_episode(env, max_steps)
            
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}, LR: {self.learning_rate:.4f}")
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'epsilon_history': self.epsilon_history,
            'learning_rate_history': self.learning_rate_history
        }
    
    def evaluate(self, env, num_episodes: int = 100, max_steps: int = 1000) -> Dict:
        """Evaluate the trained agent."""
        eval_rewards = []
        eval_lengths = []
        
        for episode in range(num_episodes):
            state = env.reset()
            if hasattr(state, '__len__') and len(state) > 1:
                state = state[0] if isinstance(state, tuple) else state
            
            total_reward = 0
            steps = 0
            
            for _ in range(max_steps):
                action = self.get_action(state, training=False)
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
            
            eval_rewards.append(total_reward)
            eval_lengths.append(steps)
        
        return {
            'eval_rewards': eval_rewards,
            'eval_lengths': eval_lengths,
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_length': np.mean(eval_lengths),
            'success_rate': np.mean([r > 0 for r in eval_rewards])
        }
    
    def get_policy(self) -> np.ndarray:
        """Get the current policy (greedy)."""
        return np.argmax(self.q_table, axis=1)
    
    def get_q_values(self) -> np.ndarray:
        """Get the current Q-values."""
        return self.q_table.copy()


def plot_learning_curves(sarsa_stats: Dict, qlearning_stats: Dict, 
                        window_size: int = 100, save_path: str = None):
    """
    Plot learning curves comparing SARSA and Q-Learning.
    
    Args:
        sarsa_stats: SARSA training statistics
        qlearning_stats: Q-Learning training statistics
        window_size: Window size for moving average
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    axes[0, 0].plot(sarsa_stats['episode_rewards'], alpha=0.3, color='blue', label='SARSA Raw')
    axes[0, 0].plot(qlearning_stats['episode_rewards'], alpha=0.3, color='red', label='Q-Learning Raw')
    
    # Moving averages
    sarsa_ma = np.convolve(sarsa_stats['episode_rewards'], 
                          np.ones(window_size)/window_size, mode='valid')
    qlearning_ma = np.convolve(qlearning_stats['episode_rewards'], 
                              np.ones(window_size)/window_size, mode='valid')
    
    axes[0, 0].plot(range(window_size-1, len(sarsa_stats['episode_rewards'])), 
                   sarsa_ma, color='blue', linewidth=2, label='SARSA MA')
    axes[0, 0].plot(range(window_size-1, len(qlearning_stats['episode_rewards'])), 
                   qlearning_ma, color='red', linewidth=2, label='Q-Learning MA')
    
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Episode lengths
    axes[0, 1].plot(sarsa_stats['episode_lengths'], alpha=0.3, color='blue', label='SARSA Raw')
    axes[0, 1].plot(qlearning_stats['episode_lengths'], alpha=0.3, color='red', label='Q-Learning Raw')
    
    sarsa_length_ma = np.convolve(sarsa_stats['episode_lengths'], 
                                 np.ones(window_size)/window_size, mode='valid')
    qlearning_length_ma = np.convolve(qlearning_stats['episode_lengths'], 
                                     np.ones(window_size)/window_size, mode='valid')
    
    axes[0, 1].plot(range(window_size-1, len(sarsa_stats['episode_lengths'])), 
                   sarsa_length_ma, color='blue', linewidth=2, label='SARSA MA')
    axes[0, 1].plot(range(window_size-1, len(qlearning_stats['episode_lengths'])), 
                   qlearning_length_ma, color='red', linewidth=2, label='Q-Learning MA')
    
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Epsilon decay
    axes[1, 0].plot(sarsa_stats['epsilon_history'], color='blue', label='SARSA')
    axes[1, 0].plot(qlearning_stats['epsilon_history'], color='red', label='Q-Learning')
    axes[1, 0].set_title('Epsilon Decay')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Epsilon')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning rate decay
    axes[1, 1].plot(sarsa_stats['learning_rate_history'], color='blue', label='SARSA')
    axes[1, 1].plot(qlearning_stats['learning_rate_history'], color='red', label='Q-Learning')
    axes[1, 1].set_title('Learning Rate Decay')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("SARSA and Q-Learning agents implemented!")
    print("Use the plot_learning_curves function to visualize training progress.")
