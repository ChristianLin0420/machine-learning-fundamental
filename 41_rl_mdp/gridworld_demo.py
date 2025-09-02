"""
GridWorld MDP Demonstration.

Implements a classic GridWorld environment to demonstrate MDP concepts and algorithms.
The agent moves in a grid world with obstacles, rewards, and terminal states.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Optional
from mdp import FiniteMDP
from policy_evaluation import policy_evaluation, create_random_policy
from policy_iteration import policy_iteration, extract_deterministic_policy
from value_iteration import value_iteration


class GridWorld:
    """
    A GridWorld environment for MDP demonstrations.
    
    The grid contains:
    - Normal cells (small negative reward for each step)
    - Obstacles (cannot enter)
    - Terminal states (goal with positive reward, pit with negative reward)
    """
    
    def __init__(self, 
                 grid_size: Tuple[int, int] = (4, 4),
                 obstacles: List[Tuple[int, int]] = None,
                 terminals: Dict[Tuple[int, int], float] = None,
                 step_reward: float = -0.04,
                 discount: float = 0.9):
        """
        Initialize GridWorld.
        
        Args:
            grid_size: (height, width) of the grid
            obstacles: List of (row, col) obstacle positions
            terminals: Dict mapping (row, col) to terminal rewards
            step_reward: Reward for each non-terminal step
            discount: Discount factor
        """
        self.height, self.width = grid_size
        self.step_reward = step_reward
        self.discount = discount
        
        # Default obstacles and terminals for 4x4 grid
        if obstacles is None:
            obstacles = [(1, 1)] if grid_size == (4, 4) else []
        if terminals is None:
            terminals = {(0, 3): 1.0, (1, 3): -1.0} if grid_size == (4, 4) else {}
        
        self.obstacles = set(obstacles)
        self.terminals = terminals
        
        # Define actions
        self.actions = ['up', 'down', 'left', 'right']
        self.action_effects = {
            'up': (-1, 0),
            'down': (1, 0), 
            'left': (0, -1),
            'right': (0, 1)
        }
        
        # Create state space (exclude obstacles)
        self.states = []
        self.state_coords = {}  # state_id -> (row, col)
        self.coord_to_state = {}  # (row, col) -> state_id
        
        state_id = 0
        for row in range(self.height):
            for col in range(self.width):
                if (row, col) not in self.obstacles:
                    state_name = f"({row},{col})"
                    self.states.append(state_name)
                    self.state_coords[state_name] = (row, col)
                    self.coord_to_state[(row, col)] = state_name
                    state_id += 1
        
        # Build MDP
        self.mdp = self._build_mdp()
    
    def _build_mdp(self) -> FiniteMDP:
        """Build the MDP representation of the GridWorld."""
        P = {}  # Transition probabilities
        
        for state in self.states:
            row, col = self.state_coords[state]
            
            # Terminal states have self-loops with zero reward
            if (row, col) in self.terminals:
                for action in self.actions:
                    P[(state, action)] = [(1.0, state, 0.0)]
            else:
                # Non-terminal states
                for action in self.actions:
                    P[(state, action)] = self._get_transitions(state, action)
        
        return FiniteMDP(self.states, self.actions, P, gamma=self.discount)
    
    def _get_transitions(self, state: str, action: str) -> List[Tuple[float, str, float]]:
        """
        Get transition probabilities for a state-action pair.
        
        Returns list of (probability, next_state, reward) tuples.
        """
        row, col = self.state_coords[state]
        
        # Deterministic transitions (can be made stochastic for more interesting examples)
        dr, dc = self.action_effects[action]
        new_row, new_col = row + dr, col + dc
        
        # Check boundaries and obstacles
        if (0 <= new_row < self.height and 
            0 <= new_col < self.width and 
            (new_row, new_col) not in self.obstacles):
            
            next_state = self.coord_to_state[(new_row, new_col)]
            
            # Reward depends on destination
            if (new_row, new_col) in self.terminals:
                reward = self.terminals[(new_row, new_col)]
            else:
                reward = self.step_reward
                
            return [(1.0, next_state, reward)]
        
        else:
            # Stay in same state if move is invalid
            return [(1.0, state, self.step_reward)]
    
    def visualize_grid(self, ax=None, title="GridWorld"):
        """Visualize the grid layout."""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.set_title(title)
        
        # Draw grid
        for i in range(self.height + 1):
            ax.axhline(y=i, color='black', linewidth=0.5)
        for i in range(self.width + 1):
            ax.axvline(x=i, color='black', linewidth=0.5)
        
        # Draw obstacles
        for row, col in self.obstacles:
            rect = patches.Rectangle((col, self.height - row - 1), 1, 1, 
                                   facecolor='black', alpha=0.8)
            ax.add_patch(rect)
            ax.text(col + 0.5, self.height - row - 0.5, 'X', 
                   ha='center', va='center', color='white', fontsize=12, fontweight='bold')
        
        # Draw terminals
        for (row, col), reward in self.terminals.items():
            color = 'green' if reward > 0 else 'red'
            rect = patches.Rectangle((col, self.height - row - 1), 1, 1,
                                   facecolor=color, alpha=0.6)
            ax.add_patch(rect)
            ax.text(col + 0.5, self.height - row - 0.5, f'{reward:+.1f}',
                   ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Label axes
        ax.set_xticks(range(self.width + 1))
        ax.set_yticks(range(self.height + 1))
        ax.set_xlabel('Column')
        ax.set_ylabel('Row (flipped)')
        ax.invert_yaxis()
        
        return ax
    
    def visualize_values(self, V: Dict[str, float], ax=None, title="Value Function"):
        """Visualize state values on the grid."""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Start with grid
        self.visualize_grid(ax, title)
        
        # Add value annotations
        v_min = min(V.values())
        v_max = max(V.values())
        
        for state in self.states:
            row, col = self.state_coords[state]
            value = V[state]
            
            # Color coding
            if (row, col) not in self.terminals:
                # Normalize value for color intensity
                if v_max != v_min:
                    intensity = (value - v_min) / (v_max - v_min)
                else:
                    intensity = 0.5
                
                color = plt.cm.RdYlBu_r(intensity)
                rect = patches.Rectangle((col, self.height - row - 1), 1, 1,
                                       facecolor=color, alpha=0.3)
                ax.add_patch(rect)
            
            # Value text
            ax.text(col + 0.5, self.height - row - 0.5, f'{value:.3f}',
                   ha='center', va='center', fontsize=8)
        
        return ax
    
    def visualize_policy(self, policy: Dict[str, Dict[str, float]], 
                        ax=None, title="Policy"):
        """Visualize policy with arrows."""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Start with grid
        self.visualize_grid(ax, title)
        
        # Arrow directions
        arrow_dirs = {
            'up': (0, 0.3),
            'down': (0, -0.3),
            'left': (-0.3, 0),
            'right': (0.3, 0)
        }
        
        for state in self.states:
            row, col = self.state_coords[state]
            
            # Skip terminals
            if (row, col) in self.terminals:
                continue
            
            # Find best action(s)
            best_actions = [a for a, p in policy[state].items() if p > 0]
            
            for action in best_actions:
                if policy[state][action] > 0.1:  # Only show significant probabilities
                    dx, dy = arrow_dirs[action]
                    ax.arrow(col + 0.5, self.height - row - 0.5, dx, dy,
                           head_width=0.1, head_length=0.1, fc='blue', ec='blue',
                           alpha=min(1.0, policy[state][action] * 2))
        
        return ax
    
    def create_stochastic_gridworld(self, slip_prob: float = 0.1):
        """
        Create a stochastic version where actions sometimes fail.
        
        Args:
            slip_prob: Probability of slipping to a perpendicular direction
        """
        P = {}
        
        # Perpendicular actions
        perpendiculars = {
            'up': ['left', 'right'],
            'down': ['left', 'right'], 
            'left': ['up', 'down'],
            'right': ['up', 'down']
        }
        
        for state in self.states:
            row, col = self.state_coords[state]
            
            if (row, col) in self.terminals:
                # Terminal states still have self-loops
                for action in self.actions:
                    P[(state, action)] = [(1.0, state, 0.0)]
            else:
                for action in self.actions:
                    transitions = []
                    
                    # Main action (with reduced probability)
                    main_prob = 1.0 - slip_prob
                    main_transitions = self._get_transitions(state, action)
                    transitions.extend([(main_prob * p, s, r) for p, s, r in main_transitions])
                    
                    # Slip actions
                    slip_prob_each = slip_prob / 2
                    for slip_action in perpendiculars[action]:
                        slip_transitions = self._get_transitions(state, slip_action)
                        transitions.extend([(slip_prob_each * p, s, r) for p, s, r in slip_transitions])
                    
                    P[(state, action)] = self._merge_transitions(transitions)
        
        return FiniteMDP(self.states, self.actions, P, gamma=self.discount)
    
    def _merge_transitions(self, transitions: List[Tuple[float, str, float]]) -> List[Tuple[float, str, float]]:
        """Merge transitions to the same state."""
        merged = {}
        
        for prob, next_state, reward in transitions:
            if next_state not in merged:
                merged[next_state] = [0.0, reward]
            merged[next_state][0] += prob
        
        return [(prob, state, reward) for state, (prob, reward) in merged.items()]


def run_gridworld_demo():
    """Run a comprehensive GridWorld demonstration."""
    print("GridWorld MDP Demonstration")
    print("=" * 50)
    
    # Create standard 4x4 GridWorld
    gridworld = GridWorld()
    mdp = gridworld.mdp
    
    print(f"GridWorld Setup:")
    print(f"  Size: {gridworld.height}x{gridworld.width}")
    print(f"  States: {len(gridworld.states)}")
    print(f"  Actions: {gridworld.actions}")
    print(f"  Obstacles: {gridworld.obstacles}")
    print(f"  Terminals: {gridworld.terminals}")
    print(f"  Step reward: {gridworld.step_reward}")
    print(f"  Discount: {gridworld.discount}")
    
    # Test 1: Policy Evaluation
    print(f"\n" + "="*50)
    print("TEST 1: Policy Evaluation")
    print("="*50)
    
    # Random policy
    random_policy = create_random_policy(mdp)
    V_random = policy_evaluation(mdp, random_policy, verbose=True)
    
    print("Random policy values:")
    for state in gridworld.states:
        row, col = gridworld.state_coords[state]
        print(f"  V({row},{col}) = {V_random[state]:.4f}")
    
    # Test 2: Policy Iteration
    print(f"\n" + "="*50)
    print("TEST 2: Policy Iteration")
    print("="*50)
    
    policy_opt_pi, V_opt_pi, info_pi = policy_iteration(mdp, verbose=True)
    det_policy_pi = extract_deterministic_policy(policy_opt_pi)
    
    print("Optimal policy (Policy Iteration):")
    for state in gridworld.states:
        row, col = gridworld.state_coords[state]
        if (row, col) not in gridworld.terminals:
            print(f"  π*({row},{col}) = {det_policy_pi[state]}")
    
    print("Optimal values (Policy Iteration):")
    for state in gridworld.states:
        row, col = gridworld.state_coords[state]
        print(f"  V*({row},{col}) = {V_opt_pi[state]:.4f}")
    
    # Test 3: Value Iteration
    print(f"\n" + "="*50)
    print("TEST 3: Value Iteration")
    print("="*50)
    
    V_opt_vi, policy_opt_vi, info_vi = value_iteration(mdp, verbose=True)
    det_policy_vi = extract_deterministic_policy(policy_opt_vi)
    
    print("Optimal policy (Value Iteration):")
    for state in gridworld.states:
        row, col = gridworld.state_coords[state]
        if (row, col) not in gridworld.terminals:
            print(f"  π*({row},{col}) = {det_policy_vi[state]}")
    
    # Test 4: Compare methods
    print(f"\n" + "="*50)
    print("TEST 4: Method Comparison")
    print("="*50)
    
    print(f"Convergence:")
    print(f"  Policy Iteration: {info_pi['iterations']} iterations")
    print(f"  Value Iteration:  {info_vi['iterations']} iterations")
    
    print(f"\nValue function differences:")
    max_diff = 0
    for state in gridworld.states:
        diff = abs(V_opt_pi[state] - V_opt_vi[state])
        max_diff = max(max_diff, diff)
        row, col = gridworld.state_coords[state]
        print(f"  |V_PI({row},{col}) - V_VI({row},{col})| = {diff:.8f}")
    print(f"Maximum difference: {max_diff:.8f}")
    
    print(f"\nPolicy comparison:")
    policies_match = True
    for state in gridworld.states:
        row, col = gridworld.state_coords[state]
        if (row, col) not in gridworld.terminals:
            match = det_policy_pi[state] == det_policy_vi[state]
            policies_match = policies_match and match
            status = "✓" if match else "✗"
            print(f"  ({row},{col}): PI={det_policy_pi[state]}, VI={det_policy_vi[state]} {status}")
    
    print(f"Policies match: {'Yes' if policies_match else 'No'}")
    
    # Visualization
    print(f"\n" + "="*50)
    print("VISUALIZATION")
    print("="*50)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # 1. Grid layout
    gridworld.visualize_grid(axes[0], "GridWorld Layout")
    
    # 2. Random policy values
    gridworld.visualize_values(V_random, axes[1], "Random Policy Values")
    
    # 3. Optimal values
    gridworld.visualize_values(V_opt_pi, axes[2], "Optimal Values")
    
    # 4. Random policy
    gridworld.visualize_policy(random_policy, axes[3], "Random Policy")
    
    # 5. Optimal policy
    gridworld.visualize_policy(policy_opt_pi, axes[4], "Optimal Policy")
    
    # 6. Value evolution (if available)
    if 'value_history' in info_vi:
        evolution_ax = axes[5]
        history = info_vi['value_history']
        
        # Plot value evolution for a few states
        interesting_states = [state for state in gridworld.states 
                            if gridworld.state_coords[state] not in gridworld.terminals][:3]
        
        for state in interesting_states:
            row, col = gridworld.state_coords[state]
            values = [V[state] for V in history]
            evolution_ax.plot(values, label=f'({row},{col})', marker='o')
        
        evolution_ax.set_xlabel('Iteration')
        evolution_ax.set_ylabel('Value')
        evolution_ax.set_title('Value Evolution (VI)')
        evolution_ax.legend()
        evolution_ax.grid(True)
    else:
        axes[5].text(0.5, 0.5, 'Value history\nnot available', 
                    ha='center', va='center', transform=axes[5].transAxes)
    
    plt.tight_layout()
    plt.savefig('/home/chrislin/machine-learning-fundamental/41_rl_mdp/gridworld_demo.png', 
                dpi=150, bbox_inches='tight')
    print("Visualization saved to 'gridworld_demo.png'")
    
    # Test 5: Stochastic GridWorld
    print(f"\n" + "="*50)
    print("TEST 5: Stochastic GridWorld")
    print("="*50)
    
    stochastic_mdp = gridworld.create_stochastic_gridworld(slip_prob=0.2)
    print("Created stochastic GridWorld with 20% slip probability")
    
    V_stoch, policy_stoch, info_stoch = value_iteration(stochastic_mdp, verbose=True)
    det_policy_stoch = extract_deterministic_policy(policy_stoch)
    
    print("Stochastic GridWorld optimal policy:")
    for state in gridworld.states:
        row, col = gridworld.state_coords[state]
        if (row, col) not in gridworld.terminals:
            print(f"  π*({row},{col}) = {det_policy_stoch[state]}")
    
    # Compare deterministic vs stochastic policies
    print(f"\nDeterministic vs Stochastic policy comparison:")
    for state in gridworld.states:
        row, col = gridworld.state_coords[state]
        if (row, col) not in gridworld.terminals:
            det_action = det_policy_pi[state]
            stoch_action = det_policy_stoch[state]
            match = det_action == stoch_action
            status = "same" if match else f"changed: {det_action} → {stoch_action}"
            print(f"  ({row},{col}): {status}")
    
    return {
        'gridworld': gridworld,
        'mdp_deterministic': mdp,
        'mdp_stochastic': stochastic_mdp,
        'policies': {
            'random': random_policy,
            'optimal_pi': policy_opt_pi,
            'optimal_vi': policy_opt_vi,
            'optimal_stochastic': policy_stoch
        },
        'values': {
            'random': V_random,
            'optimal_pi': V_opt_pi,
            'optimal_vi': V_opt_vi,
            'optimal_stochastic': V_stoch
        },
        'info': {
            'pi': info_pi,
            'vi': info_vi,
            'stoch': info_stoch
        }
    }


if __name__ == "__main__":
    # Ensure matplotlib works in headless mode
    plt.switch_backend('Agg')
    
    # Run the demo
    results = run_gridworld_demo()
    
    print(f"\n" + "="*50)
    print("DEMO COMPLETED")
    print("="*50)
    print("All algorithms successfully demonstrated on GridWorld!")
    print("Check 'gridworld_demo.png' for visualizations.")
