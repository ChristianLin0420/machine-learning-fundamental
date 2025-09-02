"""
Policy Evaluation for Markov Decision Processes.

Implements iterative policy evaluation to compute the state-value function V^π(s)
for a given policy π using the Bellman expectation equation:

V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a)[R(s,a) + γV^π(s')]
"""

import numpy as np
from typing import Dict, List, Tuple
from mdp import FiniteMDP


def policy_evaluation(mdp: FiniteMDP, 
                     policy: Dict[str, Dict[str, float]], 
                     theta: float = 1e-6,
                     max_iterations: int = 1000,
                     verbose: bool = False) -> Dict[str, float]:
    """
    Evaluate a policy using iterative policy evaluation.
    
    Args:
        mdp: The MDP environment
        policy: Policy π(a|s) as nested dict {state: {action: probability}}
        theta: Convergence threshold for value function changes
        max_iterations: Maximum number of iterations
        verbose: Print convergence information
        
    Returns:
        V: State-value function {state: value}
    """
    # Initialize value function to zero
    V = {s: 0.0 for s in mdp.states}
    
    for iteration in range(max_iterations):
        delta = 0.0  # Track maximum change in value function
        
        # Update value for each state
        for s in mdp.states:
            v_old = V[s]
            
            # Bellman expectation equation
            V[s] = 0.0
            for a in mdp.actions:
                if a in policy[s] and policy[s][a] > 0:
                    # Expected value for taking action a from state s
                    action_value = 0.0
                    for prob, s_next, reward in mdp.get_transitions(s, a):
                        action_value += prob * (reward + mdp.gamma * V[s_next])
                    
                    # Weight by policy probability
                    V[s] += policy[s][a] * action_value
            
            # Track convergence
            delta = max(delta, abs(v_old - V[s]))
        
        if verbose and (iteration % 10 == 0 or iteration < 5):
            print(f"Iteration {iteration:3d}: max|ΔV| = {delta:.8f}")
        
        # Check for convergence
        if delta < theta:
            if verbose:
                print(f"Policy evaluation converged after {iteration + 1} iterations")
                print(f"Final max|ΔV| = {delta:.8f}")
            break
    else:
        if verbose:
            print(f"Warning: Policy evaluation did not converge after {max_iterations} iterations")
            print(f"Final max|ΔV| = {delta:.8f}")
    
    return V


def policy_evaluation_matrix(mdp: FiniteMDP, 
                           policy: Dict[str, Dict[str, float]],
                           theta: float = 1e-6,
                           verbose: bool = False) -> Dict[str, float]:
    """
    Policy evaluation using matrix operations (faster for large state spaces).
    
    Solves: V = (I - γP^π)^(-1) R^π
    where P^π and R^π are the policy-weighted transition and reward matrices.
    """
    n_states = mdp.n_states
    
    # Build policy-weighted transition matrix P^π
    P_pi = np.zeros((n_states, n_states))
    R_pi = np.zeros(n_states)
    
    for s in mdp.states:
        s_idx = mdp.state_to_idx[s]
        
        for a in mdp.actions:
            if a in policy[s] and policy[s][a] > 0:
                pi_a = policy[s][a]
                
                # Add weighted transitions and rewards
                for prob, s_next, reward in mdp.get_transitions(s, a):
                    s_next_idx = mdp.state_to_idx[s_next]
                    P_pi[s_idx, s_next_idx] += pi_a * prob
                    R_pi[s_idx] += pi_a * prob * reward
    
    # Solve linear system: V = R^π + γP^π V
    # Rearrange to: (I - γP^π)V = R^π
    I = np.eye(n_states)
    A = I - mdp.gamma * P_pi
    
    try:
        V_array = np.linalg.solve(A, R_pi)
    except np.linalg.LinAlgError:
        if verbose:
            print("Matrix is singular, using iterative method instead")
        return policy_evaluation(mdp, policy, theta, verbose=verbose)
    
    # Convert back to dictionary
    V = {mdp.idx_to_state[i]: V_array[i] for i in range(n_states)}
    
    if verbose:
        print("Policy evaluation completed using matrix solution")
    
    return V


def compute_action_values(mdp: FiniteMDP, 
                         V: Dict[str, float]) -> Dict[Tuple[str, str], float]:
    """
    Compute action-value function Q(s,a) from state-value function V(s).
    
    Q^π(s,a) = Σ_{s'} P(s'|s,a)[R(s,a) + γV^π(s')]
    
    Args:
        mdp: The MDP environment
        V: State-value function
        
    Returns:
        Q: Action-value function {(state, action): value}
    """
    Q = {}
    
    for s in mdp.states:
        for a in mdp.actions:
            Q[(s, a)] = 0.0
            for prob, s_next, reward in mdp.get_transitions(s, a):
                Q[(s, a)] += prob * (reward + mdp.gamma * V[s_next])
    
    return Q


def evaluate_policy_performance(mdp: FiniteMDP,
                               policy: Dict[str, Dict[str, float]],
                               n_episodes: int = 1000,
                               max_steps: int = 100) -> Dict[str, float]:
    """
    Evaluate policy performance through simulation.
    
    Args:
        mdp: The MDP environment
        policy: Policy to evaluate
        n_episodes: Number of episodes to simulate
        max_steps: Maximum steps per episode
        
    Returns:
        performance: Dictionary with performance metrics
    """
    total_returns = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        state = mdp.reset()
        episode_return = 0.0
        discount = 1.0
        
        for step in range(max_steps):
            # Sample action from policy
            action_probs = policy[state]
            actions = list(action_probs.keys())
            probs = list(action_probs.values())
            
            if not actions or sum(probs) == 0:
                break
                
            action = np.random.choice(actions, p=probs)
            
            # Take action
            next_state, reward, done, info = mdp.step(state, action)
            
            # Accumulate discounted return
            episode_return += discount * reward
            discount *= mdp.gamma
            
            state = next_state
            
            # Early termination for demonstration
            if done or discount < 1e-6:
                break
        
        total_returns.append(episode_return)
        episode_lengths.append(step + 1)
    
    return {
        'mean_return': np.mean(total_returns),
        'std_return': np.std(total_returns),
        'mean_length': np.mean(episode_lengths),
        'total_episodes': n_episodes
    }


def create_random_policy(mdp: FiniteMDP) -> Dict[str, Dict[str, float]]:
    """Create a random uniform policy."""
    policy = {}
    for s in mdp.states:
        policy[s] = {a: 1.0/len(mdp.actions) for a in mdp.actions}
    return policy


def create_deterministic_policy(mdp: FiniteMDP, 
                               action_map: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """
    Create a deterministic policy from state->action mapping.
    
    Args:
        mdp: The MDP environment
        action_map: Dictionary mapping states to actions
        
    Returns:
        policy: Deterministic policy
    """
    policy = {}
    for s in mdp.states:
        if s in action_map:
            chosen_action = action_map[s]
            policy[s] = {a: 1.0 if a == chosen_action else 0.0 for a in mdp.actions}
        else:
            # Default to uniform random
            policy[s] = {a: 1.0/len(mdp.actions) for a in mdp.actions}
    return policy


if __name__ == "__main__":
    # Test policy evaluation
    from mdp import create_simple_mdp_example
    
    print("Testing Policy Evaluation")
    print("=" * 40)
    
    # Create test MDP
    mdp = create_simple_mdp_example()
    print(f"MDP: {mdp}")
    
    # Test 1: Random policy
    print(f"\n1. Random Policy Evaluation")
    print("-" * 30)
    random_policy = create_random_policy(mdp)
    print("Policy:")
    for s in mdp.states:
        print(f"  π({s}): {random_policy[s]}")
    
    V_random = policy_evaluation(mdp, random_policy, verbose=True)
    print(f"\nValue function:")
    for s in mdp.states:
        print(f"  V({s}) = {V_random[s]:.4f}")
    
    # Test 2: Deterministic policy
    print(f"\n2. Deterministic Policy Evaluation")
    print("-" * 30)
    det_policy = create_deterministic_policy(mdp, {'s0': 'a0', 's1': 'a0', 's2': 'a1'})
    print("Policy:")
    for s in mdp.states:
        actions = [a for a, p in det_policy[s].items() if p > 0]
        print(f"  π({s}): {actions[0] if actions else 'None'}")
    
    V_det = policy_evaluation(mdp, det_policy, verbose=True)
    print(f"\nValue function:")
    for s in mdp.states:
        print(f"  V({s}) = {V_det[s]:.4f}")
    
    # Test 3: Matrix method comparison
    print(f"\n3. Matrix Method Comparison")
    print("-" * 30)
    V_matrix = policy_evaluation_matrix(mdp, random_policy, verbose=True)
    print("Difference between iterative and matrix methods:")
    for s in mdp.states:
        diff = abs(V_random[s] - V_matrix[s])
        print(f"  |V_iter({s}) - V_matrix({s})| = {diff:.10f}")
    
    # Test 4: Action values
    print(f"\n4. Action Values")
    print("-" * 30)
    Q = compute_action_values(mdp, V_random)
    for s in mdp.states:
        print(f"Q({s}, ·):")
        for a in mdp.actions:
            print(f"  Q({s}, {a}) = {Q[(s, a)]:.4f}")
    
    # Test 5: Performance evaluation
    print(f"\n5. Policy Performance Simulation")
    print("-" * 30)
    perf_random = evaluate_policy_performance(mdp, random_policy, n_episodes=1000)
    perf_det = evaluate_policy_performance(mdp, det_policy, n_episodes=1000)
    
    print("Random policy performance:")
    for k, v in perf_random.items():
        print(f"  {k}: {v:.4f}")
    
    print("Deterministic policy performance:")
    for k, v in perf_det.items():
        print(f"  {k}: {v:.4f}")
