import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import gym
from gym import spaces


class GridWorldEnv(gym.Env):
    """
    A simple GridWorld environment for testing SARSA and Q-Learning algorithms.
    
    The agent starts at (0,0) and must reach the goal at (n-1, n-1).
    There are obstacles and a penalty for hitting walls.
    """
    
    def __init__(self, size: int = 5, obstacles: Optional[list] = None, 
                 goal_reward: float = 100, step_penalty: float = -1, 
                 wall_penalty: float = -10, stochastic: bool = False):
        """
        Initialize GridWorld environment.
        
        Args:
            size: Size of the grid (size x size)
            obstacles: List of (row, col) tuples for obstacles
            goal_reward: Reward for reaching the goal
            step_penalty: Penalty for each step
            wall_penalty: Penalty for hitting walls
            stochastic: Whether to add stochasticity to movements
        """
        super().__init__()
        
        self.size = size
        self.obstacles = obstacles or []
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.wall_penalty = wall_penalty
        self.stochastic = stochastic
        
        # Define action space (up, down, left, right)
        self.action_space = spaces.Discrete(4)
        
        # Define observation space (flattened grid position)
        self.observation_space = spaces.Discrete(size * size)
        
        # Action mappings
        self.actions = {
            0: (-1, 0),  # Up
            1: (1, 0),   # Down
            2: (0, -1),  # Left
            3: (0, 1)    # Right
        }
        
        # Initialize state
        self.state = None
        self.goal_pos = (size - 1, size - 1)
        self.start_pos = (0, 0)
        
    def reset(self, seed: Optional[int] = None) -> int:
        """Reset the environment to initial state."""
        if seed is not None:
            np.random.seed(seed)
        
        self.state = self.start_pos
        return self._state_to_obs(self.state)
    
    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to take (0-3)
            
        Returns:
            (observation, reward, done, info)
        """
        if self.state is None:
            raise ValueError("Environment not reset. Call reset() first.")
        
        # Get action direction
        direction = self.actions[action]
        
        # Calculate next position
        next_row = self.state[0] + direction[0]
        next_col = self.state[1] + direction[1]
        
        # Add stochasticity if enabled
        if self.stochastic and np.random.random() < 0.1:  # 10% chance of random action
            random_action = np.random.randint(0, 4)
            direction = self.actions[random_action]
            next_row = self.state[0] + direction[0]
            next_col = self.state[1] + direction[1]
        
        # Check bounds
        if (next_row < 0 or next_row >= self.size or 
            next_col < 0 or next_col >= self.size):
            # Hit wall
            reward = self.wall_penalty
            done = False
            info = {'hit_wall': True}
        elif (next_row, next_col) in self.obstacles:
            # Hit obstacle
            reward = self.wall_penalty
            done = False
            info = {'hit_obstacle': True}
        elif (next_row, next_col) == self.goal_pos:
            # Reached goal
            self.state = (next_row, next_col)
            reward = self.goal_reward
            done = True
            info = {'reached_goal': True}
        else:
            # Normal step
            self.state = (next_row, next_col)
            reward = self.step_penalty
            done = False
            info = {'normal_step': True}
        
        return self._state_to_obs(self.state), reward, done, info
    
    def _state_to_obs(self, state: Tuple[int, int]) -> int:
        """Convert (row, col) state to observation index."""
        return state[0] * self.size + state[1]
    
    def _obs_to_state(self, obs: int) -> Tuple[int, int]:
        """Convert observation index to (row, col) state."""
        return (obs // self.size, obs % self.size)
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Args:
            mode: Rendering mode ('human' or 'rgb_array')
            
        Returns:
            Rendered image if mode is 'rgb_array', None otherwise
        """
        if self.state is None:
            return None
        
        # Create grid visualization
        grid = np.zeros((self.size, self.size))
        
        # Mark obstacles
        for obs_row, obs_col in self.obstacles:
            grid[obs_row, obs_col] = -1
        
        # Mark goal
        grid[self.goal_pos[0], self.goal_pos[1]] = 2
        
        # Mark current position
        if self.state is not None:
            grid[self.state[0], self.state[1]] = 1
        
        if mode == 'human':
            plt.figure(figsize=(6, 6))
            plt.imshow(grid, cmap='RdYlBu', vmin=-1, vmax=2)
            
            # Add text annotations
            for i in range(self.size):
                for j in range(self.size):
                    if grid[i, j] == -1:
                        plt.text(j, i, 'X', ha='center', va='center', fontsize=20, color='black')
                    elif grid[i, j] == 1:
                        plt.text(j, i, 'A', ha='center', va='center', fontsize=20, color='white')
                    elif grid[i, j] == 2:
                        plt.text(j, i, 'G', ha='center', va='center', fontsize=20, color='white')
            
            plt.title('GridWorld Environment')
            plt.xlabel('Column')
            plt.ylabel('Row')
            plt.xticks(range(self.size))
            plt.yticks(range(self.size))
            plt.grid(True, alpha=0.3)
            plt.show()
            
        elif mode == 'rgb_array':
            return grid
        
        return None
    
    def get_state(self) -> Tuple[int, int]:
        """Get current state as (row, col) tuple."""
        return self.state
    
    def is_terminal(self, state: Tuple[int, int]) -> bool:
        """Check if a state is terminal."""
        return state == self.goal_pos
    
    def get_valid_actions(self, state: Tuple[int, int]) -> list:
        """Get list of valid actions from a state."""
        valid_actions = []
        
        for action, direction in self.actions.items():
            next_row = state[0] + direction[0]
            next_col = state[1] + direction[1]
            
            # Check if action is valid (not out of bounds or hitting obstacle)
            if (0 <= next_row < self.size and 
                0 <= next_col < self.size and 
                (next_row, next_col) not in self.obstacles):
                valid_actions.append(action)
        
        return valid_actions


def create_simple_gridworld(size: int = 5) -> GridWorldEnv:
    """Create a simple GridWorld with no obstacles."""
    return GridWorldEnv(size=size, obstacles=[])


def create_obstacle_gridworld(size: int = 5) -> GridWorldEnv:
    """Create a GridWorld with obstacles."""
    obstacles = [(1, 1), (1, 3), (2, 2), (3, 1)]
    return GridWorldEnv(size=size, obstacles=obstacles)


def create_stochastic_gridworld(size: int = 5) -> GridWorldEnv:
    """Create a stochastic GridWorld with obstacles."""
    obstacles = [(1, 1), (1, 3), (2, 2), (3, 1)]
    return GridWorldEnv(size=size, obstacles=obstacles, stochastic=True)


if __name__ == "__main__":
    # Test the environment
    env = create_obstacle_gridworld()
    
    print("GridWorld Environment Test")
    print("=" * 30)
    
    # Reset environment
    obs = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Initial state: {env.get_state()}")
    
    # Take some random actions
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step {i+1}: Action {action}, Reward {reward}, Done {done}, Info {info}")
        
        if done:
            print("Episode finished!")
            break
    
    # Render the environment
    env.render()
