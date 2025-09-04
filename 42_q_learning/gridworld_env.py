"""
GridWorld Environment for Q-Learning
=====================================

Simple grid world environment for testing Q-learning algorithms.
Features obstacles, rewards, and terminal states.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Tuple, Optional, List
from enum import IntEnum


class Actions(IntEnum):
    """Action space for GridWorld."""
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class GridWorld:
    """
    Simple GridWorld Environment.
    
    The agent starts at a specified position and needs to reach the goal
    while avoiding obstacles and cliff/penalty states.
    """
    
    def __init__(
        self, 
        grid_size: Tuple[int, int] = (5, 5),
        start_pos: Tuple[int, int] = (0, 0),
        goal_pos: Tuple[int, int] = (4, 4),
        obstacles: Optional[List[Tuple[int, int]]] = None,
        cliffs: Optional[List[Tuple[int, int]]] = None,
        step_reward: float = -0.1,
        goal_reward: float = 10.0,
        cliff_reward: float = -10.0,
        obstacle_reward: float = -1.0
    ):
        """
        Initialize GridWorld environment.
        
        Args:
            grid_size: (height, width) of the grid
            start_pos: Starting position (row, col)
            goal_pos: Goal position (row, col)
            obstacles: List of obstacle positions
            cliffs: List of cliff/penalty positions
            step_reward: Reward for each step
            goal_reward: Reward for reaching goal
            cliff_reward: Penalty for hitting cliff
            obstacle_reward: Penalty for hitting obstacle
        """
        self.height, self.width = grid_size
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.obstacles = obstacles or []
        self.cliffs = cliffs or []
        
        # Rewards
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.cliff_reward = cliff_reward
        self.obstacle_reward = obstacle_reward
        
        # Environment properties
        self.observation_space = type('Space', (), {'n': self.height * self.width})()
        self.action_space = type('Space', (), {'n': len(Actions)})()
        
        # Action mappings
        self.action_map = {
            Actions.UP: (-1, 0),
            Actions.DOWN: (1, 0),
            Actions.LEFT: (0, -1),
            Actions.RIGHT: (0, 1)
        }
        
        # Current state
        self.current_pos = None
        self.episode_step = 0
        
    def reset(self) -> int:
        """
        Reset environment to initial state.
        
        Returns:
            Initial state (encoded as integer)
        """
        self.current_pos = self.start_pos
        self.episode_step = 0
        return self._pos_to_state(self.current_pos)
    
    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """
        Execute action and return results.
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        self.episode_step += 1
        
        # Get action direction
        if action not in range(len(Actions)):
            raise ValueError(f"Invalid action: {action}")
        
        dr, dc = self.action_map[Actions(action)]
        
        # Calculate new position
        new_row = max(0, min(self.height - 1, self.current_pos[0] + dr))
        new_col = max(0, min(self.width - 1, self.current_pos[1] + dc))
        new_pos = (new_row, new_col)
        
        # Check if move is valid (not into obstacle)
        if new_pos in self.obstacles:
            # Stay in current position, get penalty
            reward = self.obstacle_reward
            done = False
        else:
            # Move to new position
            self.current_pos = new_pos
            
            # Calculate reward and check for terminal states
            if self.current_pos == self.goal_pos:
                reward = self.goal_reward
                done = True
            elif self.current_pos in self.cliffs:
                reward = self.cliff_reward
                done = True  # Episode ends when hitting cliff
            else:
                reward = self.step_reward
                done = False
        
        # Additional termination condition (max steps)
        if self.episode_step >= 200:
            done = True
        
        next_state = self._pos_to_state(self.current_pos)
        info = {'pos': self.current_pos, 'episode_step': self.episode_step}
        
        return next_state, reward, done, info
    
    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        """Convert position to state index."""
        return pos[0] * self.width + pos[1]
    
    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        """Convert state index to position."""
        return (state // self.width, state % self.width)
    
    def render(self, q_table: Optional[np.ndarray] = None, 
               policy: Optional[np.ndarray] = None, mode: str = 'human') -> None:
        """
        Render the environment.
        
        Args:
            q_table: Optional Q-table for value visualization
            policy: Optional policy for action visualization
            mode: Rendering mode
        """
        if mode == 'human':
            self._render_text()
        elif mode == 'rgb_array' or q_table is not None or policy is not None:
            return self._render_matplotlib(q_table, policy)
    
    def _render_text(self) -> None:
        """Simple text rendering."""
        grid = [['.' for _ in range(self.width)] for _ in range(self.height)]
        
        # Mark special positions
        for obs_pos in self.obstacles:
            grid[obs_pos[0]][obs_pos[1]] = '#'
        
        for cliff_pos in self.cliffs:
            grid[cliff_pos[0]][cliff_pos[1]] = 'X'
        
        grid[self.goal_pos[0]][self.goal_pos[1]] = 'G'
        grid[self.current_pos[0]][self.current_pos[1]] = 'A'
        
        print("\n" + "="*20)
        for row in grid:
            print(" ".join(row))
        print("="*20)
        print(f"Position: {self.current_pos}, Step: {self.episode_step}")
    
    def _render_matplotlib(self, q_table: Optional[np.ndarray] = None,
                          policy: Optional[np.ndarray] = None) -> np.ndarray:
        """Render using matplotlib for better visualization."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create base grid
        if q_table is not None:
            # Show state values as heatmap
            values = np.max(q_table, axis=1).reshape(self.height, self.width)
            im = ax.imshow(values, cmap='RdYlBu_r', alpha=0.7)
            plt.colorbar(im, ax=ax, label='State Value')
        else:
            # Just show grid
            ax.set_xlim(-0.5, self.width - 0.5)
            ax.set_ylim(-0.5, self.height - 0.5)
            ax.set_aspect('equal')
        
        # Add grid lines
        for i in range(self.height + 1):
            ax.axhline(i - 0.5, color='black', linewidth=0.5)
        for j in range(self.width + 1):
            ax.axvline(j - 0.5, color='black', linewidth=0.5)
        
        # Mark special positions
        for obs_pos in self.obstacles:
            rect = patches.Rectangle((obs_pos[1] - 0.5, obs_pos[0] - 0.5), 1, 1,
                                   linewidth=2, edgecolor='black', facecolor='black')
            ax.add_patch(rect)
            ax.text(obs_pos[1], obs_pos[0], '#', ha='center', va='center',
                   color='white', fontsize=16, fontweight='bold')
        
        for cliff_pos in self.cliffs:
            rect = patches.Rectangle((cliff_pos[1] - 0.5, cliff_pos[0] - 0.5), 1, 1,
                                   linewidth=2, edgecolor='red', facecolor='red', alpha=0.7)
            ax.add_patch(rect)
            ax.text(cliff_pos[1], cliff_pos[0], 'X', ha='center', va='center',
                   color='white', fontsize=16, fontweight='bold')
        
        # Mark goal
        goal_circle = patches.Circle((self.goal_pos[1], self.goal_pos[0]), 0.3,
                                   color='green', alpha=0.8)
        ax.add_patch(goal_circle)
        ax.text(self.goal_pos[1], self.goal_pos[0], 'G', ha='center', va='center',
               color='white', fontsize=12, fontweight='bold')
        
        # Mark current position
        agent_circle = patches.Circle((self.current_pos[1], self.current_pos[0]), 0.2,
                                    color='blue', alpha=0.8)
        ax.add_patch(agent_circle)
        ax.text(self.current_pos[1], self.current_pos[0], 'A', ha='center', va='center',
               color='white', fontsize=10, fontweight='bold')
        
        # Show policy arrows if provided
        if policy is not None:
            arrow_map = {
                Actions.UP: (0, 0.3),
                Actions.DOWN: (0, -0.3),
                Actions.LEFT: (-0.3, 0),
                Actions.RIGHT: (0.3, 0)
            }
            
            for state in range(len(policy)):
                pos = self._state_to_pos(state)
                if pos not in self.obstacles and pos != self.goal_pos and pos not in self.cliffs:
                    action = policy[state]
                    dx, dy = arrow_map[Actions(action)]
                    ax.arrow(pos[1], pos[0], dx, dy, head_width=0.1, head_length=0.1,
                           fc='black', ec='black', alpha=0.6)
        
        ax.set_title(f"GridWorld Environment (Step: {self.episode_step})")
        ax.set_xticks(range(self.width))
        ax.set_yticks(range(self.height))
        ax.invert_yaxis()  # Invert y-axis to match grid coordinates
        
        # Convert to rgb array if needed
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        ncols, nrows = fig.canvas.get_width_height()
        rgb_array = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
        plt.close(fig)
        
        return rgb_array
    
    def get_optimal_path(self, policy: np.ndarray, max_steps: int = 100) -> List[Tuple[int, int]]:
        """
        Get the path that the policy would take from start to goal.
        
        Args:
            policy: Policy array (action for each state)
            max_steps: Maximum steps to prevent infinite loops
            
        Returns:
            List of positions in the path
        """
        path = [self.start_pos]
        current_pos = self.start_pos
        
        for _ in range(max_steps):
            if current_pos == self.goal_pos:
                break
            
            state = self._pos_to_state(current_pos)
            action = policy[state]
            dr, dc = self.action_map[Actions(action)]
            
            new_row = max(0, min(self.height - 1, current_pos[0] + dr))
            new_col = max(0, min(self.width - 1, current_pos[1] + dc))
            new_pos = (new_row, new_col)
            
            # Check if move is valid
            if new_pos not in self.obstacles:
                current_pos = new_pos
                path.append(current_pos)
            else:
                # Stuck at obstacle, break
                break
        
        return path


def create_simple_gridworld() -> GridWorld:
    """Create a simple 5x5 GridWorld with some obstacles."""
    return GridWorld(
        grid_size=(5, 5),
        start_pos=(0, 0),
        goal_pos=(4, 4),
        obstacles=[(1, 1), (2, 2), (3, 1)],
        cliffs=[(1, 3), (3, 3)]
    )


def create_cliff_walking() -> GridWorld:
    """Create the classic cliff walking problem."""
    cliffs = [(3, i) for i in range(1, 11)]  # Cliff along bottom row
    
    return GridWorld(
        grid_size=(4, 12),
        start_pos=(3, 0),
        goal_pos=(3, 11),
        obstacles=[],
        cliffs=cliffs,
        step_reward=-1,
        goal_reward=0,
        cliff_reward=-100
    )


def create_maze_gridworld() -> GridWorld:
    """Create a more complex maze-like GridWorld."""
    obstacles = [
        (1, 1), (1, 2), (1, 4), (1, 5),
        (2, 2), (2, 5), (2, 7), 
        (3, 0), (3, 2), (3, 4), (3, 7),
        (4, 4), (4, 5), (4, 6),
        (5, 1), (5, 2), (5, 6),
        (6, 3), (6, 4)
    ]
    
    cliffs = [(2, 0), (4, 7), (6, 0)]
    
    return GridWorld(
        grid_size=(8, 8),
        start_pos=(0, 0),
        goal_pos=(7, 7),
        obstacles=obstacles,
        cliffs=cliffs,
        step_reward=-0.1,
        goal_reward=10.0,
        cliff_reward=-5.0
    )


if __name__ == "__main__":
    # Demo the GridWorld environment
    print("GridWorld Environment Demo")
    print("==========================")
    
    # Create and test simple gridworld
    env = create_simple_gridworld()
    print(f"Environment created: {env.height}x{env.width} grid")
    print(f"Start: {env.start_pos}, Goal: {env.goal_pos}")
    print(f"Obstacles: {env.obstacles}")
    print(f"Cliffs: {env.cliffs}")
    
    # Test basic functionality
    state = env.reset()
    print(f"\nInitial state: {state}")
    env.render()
    
    # Take a few random steps
    for i in range(5):
        action = np.random.choice(4)
        next_state, reward, done, info = env.step(action)
        print(f"Step {i+1}: Action {action} -> State {next_state}, Reward {reward:.1f}, Done {done}")
        env.render()
        
        if done:
            print("Episode finished!")
            break
    
    print("\nEnvironment ready for Q-learning!")
