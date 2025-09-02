"""
Finite Markov Decision Process (MDP) implementation.

An MDP is defined by the tuple (S, A, P, R, γ) where:
- S: finite state space
- A: finite action space  
- P(s'|s,a): transition probability from state s to s' given action a
- R(s,a): immediate reward for taking action a in state s
- γ: discount factor [0,1]
"""

import numpy as np
from typing import Dict, List, Tuple, Union


class FiniteMDP:
    """
    A finite Markov Decision Process.
    
    The transition dynamics can be specified in two formats:
    1. P: dict {(s,a): [(prob, s_next, reward), ...]} - full transition specification
    2. P: dict {(s,a): [(prob, s_next), ...]} with separate R dict for rewards
    """
    
    def __init__(self, 
                 states: List, 
                 actions: List, 
                 P: Dict[Tuple, List[Tuple]], 
                 R: Union[Dict[Tuple, float], None] = None,
                 gamma: float = 0.9,
                 initial_state_dist: Union[Dict, None] = None):
        """
        Initialize the MDP.
        
        Args:
            states: List of state identifiers
            actions: List of action identifiers  
            P: Transition probabilities. Format:
               - {(s,a): [(prob, s_next, reward), ...]} OR
               - {(s,a): [(prob, s_next), ...]} (requires R dict)
            R: Reward function {(s,a): reward} (optional if rewards in P)
            gamma: Discount factor ∈ [0,1]
            initial_state_dist: Initial state distribution (uniform if None)
        """
        self.states = states
        self.actions = actions
        self.gamma = gamma
        self.n_states = len(states)
        self.n_actions = len(actions)
        
        # Create state/action to index mappings for efficiency
        self.state_to_idx = {s: i for i, s in enumerate(states)}
        self.action_to_idx = {a: i for i, a in enumerate(actions)}
        self.idx_to_state = {i: s for i, s in enumerate(states)}
        self.idx_to_action = {i: a for i, a in enumerate(actions)}
        
        # Process transition probabilities and rewards
        self.P = {}
        self.R = {}
        
        for (s, a), transitions in P.items():
            self.P[(s, a)] = []
            
            if len(transitions[0]) == 3:  # Format: (prob, s_next, reward)
                for prob, s_next, reward in transitions:
                    self.P[(s, a)].append((prob, s_next, reward))
                    if (s, a) not in self.R:
                        # Use first reward if multiple transitions from same (s,a)
                        self.R[(s, a)] = reward
            else:  # Format: (prob, s_next) - requires separate R dict
                if R is None:
                    raise ValueError("R dict required when P doesn't include rewards")
                for prob, s_next in transitions:
                    reward = R.get((s, a), 0.0)
                    self.P[(s, a)].append((prob, s_next, reward))
                    self.R[(s, a)] = reward
        
        # If R was provided separately, use it
        if R is not None:
            self.R.update(R)
            
        # Validate MDP
        self._validate_mdp()
        
        # Set initial state distribution
        if initial_state_dist is None:
            self.initial_state_dist = {s: 1.0/len(states) for s in states}
        else:
            self.initial_state_dist = initial_state_dist
    
    def _validate_mdp(self):
        """Validate that the MDP is properly defined."""
        for s in self.states:
            for a in self.actions:
                if (s, a) not in self.P:
                    raise ValueError(f"Missing transition probabilities for state {s}, action {a}")
                
                # Check probabilities sum to 1
                total_prob = sum(prob for prob, _, _ in self.P[(s, a)])
                if not np.isclose(total_prob, 1.0, rtol=1e-6):
                    raise ValueError(f"Probabilities for ({s}, {a}) sum to {total_prob}, not 1.0")
    
    def get_reward(self, state, action):
        """Get immediate reward for (state, action) pair."""
        return self.R.get((state, action), 0.0)
    
    def get_transitions(self, state, action):
        """Get all possible transitions from (state, action)."""
        return self.P.get((state, action), [])
    
    def sample_transition(self, state, action):
        """Sample next state and reward from transition probabilities."""
        transitions = self.get_transitions(state, action)
        if not transitions:
            raise ValueError(f"No transitions defined for state {state}, action {action}")
        
        probs = [prob for prob, _, _ in transitions]
        idx = np.random.choice(len(transitions), p=probs)
        _, next_state, reward = transitions[idx]
        
        return next_state, reward
    
    def get_transition_matrix(self):
        """
        Get transition matrices as numpy arrays for vectorized operations.
        
        Returns:
            P_matrices: dict {action: (n_states, n_states) transition matrix}
            R_matrices: dict {action: (n_states,) reward vector}
        """
        P_matrices = {}
        R_matrices = {}
        
        for a in self.actions:
            P_a = np.zeros((self.n_states, self.n_states))
            R_a = np.zeros(self.n_states)
            
            for s in self.states:
                s_idx = self.state_to_idx[s]
                R_a[s_idx] = self.get_reward(s, a)
                
                for prob, s_next, _ in self.get_transitions(s, a):
                    s_next_idx = self.state_to_idx[s_next]
                    P_a[s_idx, s_next_idx] = prob
            
            P_matrices[a] = P_a
            R_matrices[a] = R_a
        
        return P_matrices, R_matrices
    
    def reset(self):
        """Reset to initial state according to initial state distribution."""
        states = list(self.initial_state_dist.keys())
        probs = list(self.initial_state_dist.values())
        return np.random.choice(states, p=probs)
    
    def step(self, state, action):
        """
        Take a step in the MDP.
        
        Args:
            state: current state
            action: action to take
            
        Returns:
            next_state: resulting state
            reward: immediate reward
            done: always False (infinite horizon)
            info: additional information
        """
        next_state, reward = self.sample_transition(state, action)
        
        return next_state, reward, False, {'transition_probs': self.get_transitions(state, action)}
    
    def __repr__(self):
        return f"FiniteMDP(|S|={self.n_states}, |A|={self.n_actions}, γ={self.gamma})"


def create_simple_mdp_example():
    """Create a simple 3-state MDP example for testing."""
    states = ['s0', 's1', 's2']
    actions = ['a0', 'a1']
    
    # Define transitions: {(state, action): [(prob, next_state, reward), ...]}
    P = {
        ('s0', 'a0'): [(0.7, 's0', 0), (0.3, 's1', 1)],
        ('s0', 'a1'): [(0.2, 's0', 0), (0.8, 's2', -1)],
        ('s1', 'a0'): [(1.0, 's2', 2)],
        ('s1', 'a1'): [(0.5, 's0', 0), (0.5, 's2', 1)],
        ('s2', 'a0'): [(0.6, 's0', -1), (0.4, 's1', 0)],
        ('s2', 'a1'): [(1.0, 's2', 0)]
    }
    
    return FiniteMDP(states, actions, P, gamma=0.9)


if __name__ == "__main__":
    # Test the MDP implementation
    print("Creating simple MDP example...")
    mdp = create_simple_mdp_example()
    print(f"MDP: {mdp}")
    
    print(f"\nStates: {mdp.states}")
    print(f"Actions: {mdp.actions}")
    print(f"Discount factor: {mdp.gamma}")
    
    print("\nTesting transitions:")
    for state in mdp.states:
        for action in mdp.actions:
            transitions = mdp.get_transitions(state, action)
            reward = mdp.get_reward(state, action)
            print(f"({state}, {action}): {transitions}, R = {reward}")
    
    print("\nTesting sampling:")
    for _ in range(5):
        state = 's0'
        action = 'a0'
        next_state, reward = mdp.sample_transition(state, action)
        print(f"From {state} with {action} -> {next_state}, reward = {reward}")
    
    print("\nTesting step function:")
    state = mdp.reset()
    print(f"Initial state: {state}")
    
    for step in range(3):
        action = np.random.choice(mdp.actions)
        next_state, reward, done, info = mdp.step(state, action)
        print(f"Step {step}: {state} --{action}--> {next_state}, reward = {reward}")
        state = next_state
