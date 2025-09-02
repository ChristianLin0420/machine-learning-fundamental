"""
Value Iteration for Markov Decision Processes.

Implements value iteration algorithm that directly computes the optimal value function V*
using the Bellman optimality equation:

V*(s) = max_a Σ_{s'} P(s'|s,a)[R(s,a) + γV*(s')]

The optimal policy is then extracted as:
π*(s) = argmax_a Σ_{s'} P(s'|s,a)[R(s,a) + γV*(s')]
"""

import numpy as np
from typing import Dict, List, Tuple
from mdp import FiniteMDP
from policy_evaluation import compute_action_values


def value_iteration(mdp: FiniteMDP,
                   theta: float = 1e-6,
                   max_iterations: int = 1000,
                   verbose: bool = False) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict]:
    """
    Value iteration algorithm to find optimal value function and policy.
    
    Args:
        mdp: The MDP environment
        theta: Convergence threshold for value function changes
        max_iterations: Maximum number of iterations
        verbose: Print convergence information
        
    Returns:
        V_optimal: Optimal value function
        policy_optimal: Optimal policy
        info: Convergence information
    """
    # Initialize value function
    V = {s: 0.0 for s in mdp.states}
    
    value_history = []
    
    if verbose:
        print("Starting Value Iteration")
        print("=" * 40)
    
    for iteration in range(max_iterations):
        delta = 0.0  # Track maximum change in value function
        V_old = dict(V)  # Store old values
        
        # Update value for each state using Bellman optimality equation
        for s in mdp.states:
            # Compute value for all possible actions
            action_values = []
            for a in mdp.actions:
                action_value = 0.0
                for prob, s_next, reward in mdp.get_transitions(s, a):
                    action_value += prob * (reward + mdp.gamma * V[s_next])
                action_values.append(action_value)
            
            # Take maximum over actions
            V[s] = max(action_values)
            
            # Track convergence
            delta = max(delta, abs(V_old[s] - V[s]))
        
        # Store history
        value_history.append(dict(V))
        
        if verbose and (iteration % 10 == 0 or iteration < 5):
            print(f"Iteration {iteration:3d}: max|ΔV| = {delta:.8f}")
            if iteration < 5:  # Show values for first few iterations
                print(f"  Values: {[f'{V[s]:.4f}' for s in mdp.states]}")
        
        # Check for convergence
        if delta < theta:
            if verbose:
                print(f"Value iteration converged after {iteration + 1} iterations")
                print(f"Final max|ΔV| = {delta:.8f}")
            break
    else:
        if verbose:
            print(f"Warning: Value iteration did not converge after {max_iterations} iterations")
            print(f"Final max|ΔV| = {delta:.8f}")
    
    # Extract optimal policy
    policy_optimal = extract_policy_from_values(mdp, V)
    
    info = {
        'iterations': iteration + 1,
        'converged': delta < theta,
        'final_delta': delta,
        'value_history': value_history
    }
    
    return V, policy_optimal, info


def extract_policy_from_values(mdp: FiniteMDP, 
                              V: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """
    Extract greedy policy from value function.
    
    π*(s) = argmax_a Σ_{s'} P(s'|s,a)[R(s,a) + γV*(s')]
    
    Args:
        mdp: The MDP environment
        V: Value function
        
    Returns:
        policy: Greedy policy (deterministic)
    """
    policy = {}
    
    for s in mdp.states:
        # Compute action values
        action_values = {}
        for a in mdp.actions:
            action_values[a] = 0.0
            for prob, s_next, reward in mdp.get_transitions(s, a):
                action_values[a] += prob * (reward + mdp.gamma * V[s_next])
        
        # Find best action(s)
        max_value = max(action_values.values())
        best_actions = [a for a, v in action_values.items() if np.isclose(v, max_value)]
        
        # Create deterministic policy (break ties randomly)
        policy[s] = {}
        for a in mdp.actions:
            if a in best_actions:
                policy[s][a] = 1.0 / len(best_actions)
            else:
                policy[s][a] = 0.0
    
    return policy


def modified_value_iteration(mdp: FiniteMDP,
                           theta: float = 1e-6,
                           max_iterations: int = 1000,
                           stopping_criterion: str = 'max_change',
                           verbose: bool = False) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict]:
    """
    Modified value iteration with different stopping criteria.
    
    Args:
        mdp: The MDP environment
        theta: Convergence threshold
        max_iterations: Maximum iterations
        stopping_criterion: 'max_change', 'mean_change', or 'policy_stable'
        verbose: Print information
        
    Returns:
        V_optimal: Optimal value function
        policy_optimal: Optimal policy  
        info: Convergence information
    """
    V = {s: 0.0 for s in mdp.states}
    policy_history = []
    
    for iteration in range(max_iterations):
        V_old = dict(V)
        old_policy = extract_policy_from_values(mdp, V) if stopping_criterion == 'policy_stable' else None
        
        # Value iteration update
        for s in mdp.states:
            action_values = []
            for a in mdp.actions:
                action_value = sum(prob * (reward + mdp.gamma * V[s_next])
                                 for prob, s_next, reward in mdp.get_transitions(s, a))
                action_values.append(action_value)
            V[s] = max(action_values)
        
        # Check stopping criterion
        if stopping_criterion == 'max_change':
            delta = max(abs(V[s] - V_old[s]) for s in mdp.states)
            converged = delta < theta
        elif stopping_criterion == 'mean_change':
            delta = np.mean([abs(V[s] - V_old[s]) for s in mdp.states])
            converged = delta < theta
        elif stopping_criterion == 'policy_stable':
            new_policy = extract_policy_from_values(mdp, V)
            policy_history.append(new_policy)
            
            if old_policy is not None:
                policy_stable = True
                for s in mdp.states:
                    for a in mdp.actions:
                        if not np.isclose(old_policy[s][a], new_policy[s][a], atol=1e-10):
                            policy_stable = False
                            break
                    if not policy_stable:
                        break
                converged = policy_stable
            else:
                converged = False
            delta = 0.0  # Not used for policy stability
        else:
            raise ValueError(f"Unknown stopping criterion: {stopping_criterion}")
        
        if verbose and iteration % 10 == 0:
            print(f"Iteration {iteration}: criterion = {delta:.8f}")
        
        if converged:
            if verbose:
                print(f"Converged after {iteration + 1} iterations ({stopping_criterion})")
            break
    
    policy_optimal = extract_policy_from_values(mdp, V)
    
    info = {
        'iterations': iteration + 1,
        'converged': converged,
        'stopping_criterion': stopping_criterion,
        'policy_history': policy_history if stopping_criterion == 'policy_stable' else None
    }
    
    return V, policy_optimal, info


def value_iteration_with_analysis(mdp: FiniteMDP, **kwargs) -> Tuple:
    """
    Value iteration with detailed analysis and visualization.
    """
    V_opt, policy_opt, info = value_iteration(mdp, verbose=True, **kwargs)
    
    print("\n" + "="*50)
    print("VALUE ITERATION SUMMARY")
    print("="*50)
    
    print(f"Converged: {info['converged']}")
    print(f"Iterations: {info['iterations']}")
    print(f"Final Δ: {info['final_delta']:.8f}")
    
    print(f"\nOptimal Value Function V*:")
    for s in mdp.states:
        print(f"  V*({s}) = {V_opt[s]:.6f}")
    
    # Extract deterministic policy representation
    det_policy = {}
    for s in mdp.states:
        best_actions = [a for a, p in policy_opt[s].items() if p > 0]
        det_policy[s] = best_actions[0] if len(best_actions) == 1 else best_actions
    
    print(f"\nOptimal Policy π*:")
    for s in mdp.states:
        print(f"  π*({s}) = {det_policy[s]}")
    
    # Compute and display action values
    Q_opt = compute_action_values(mdp, V_opt)
    print(f"\nOptimal Action Values Q*:")
    for s in mdp.states:
        print(f"  Q*({s}, ·):")
        for a in mdp.actions:
            marker = " ← optimal" if (isinstance(det_policy[s], str) and det_policy[s] == a) or \
                                   (isinstance(det_policy[s], list) and a in det_policy[s]) else ""
            print(f"    Q*({s}, {a}) = {Q_opt[(s, a)]:.6f}{marker}")
    
    return V_opt, policy_opt, info


def compare_value_iteration_variants(mdp: FiniteMDP, theta: float = 1e-6) -> Dict:
    """
    Compare different variants of value iteration.
    """
    print("Comparing Value Iteration Variants")
    print("=" * 50)
    
    variants = {
        'max_change': 'Maximum value change',
        'mean_change': 'Mean value change', 
        'policy_stable': 'Policy stability'
    }
    
    results = {}
    
    for criterion, description in variants.items():
        print(f"\n{description}:")
        print("-" * 30)
        
        V, policy, info = modified_value_iteration(
            mdp, theta=theta, stopping_criterion=criterion, verbose=True
        )
        
        results[criterion] = {
            'V': V,
            'policy': policy,
            'info': info
        }
        
        det_policy = {s: max(policy[s].items(), key=lambda x: x[1])[0] for s in mdp.states}
        print(f"Final policy: {det_policy}")
        print(f"Iterations: {info['iterations']}")
    
    # Compare convergence
    print(f"\nConvergence Comparison:")
    print("-" * 30)
    for criterion in variants:
        info = results[criterion]['info']
        print(f"{criterion:12}: {info['iterations']:3d} iterations, "
              f"converged: {info['converged']}")
    
    # Check if all methods found same optimal policy
    reference_policy = results['max_change']['policy']
    print(f"\nPolicy Consistency Check:")
    print("-" * 30)
    
    for criterion in variants:
        if criterion == 'max_change':
            continue
        
        policy = results[criterion]['policy']
        same_policy = True
        
        for s in mdp.states:
            for a in mdp.actions:
                if not np.isclose(reference_policy[s][a], policy[s][a], atol=1e-10):
                    same_policy = False
                    break
            if not same_policy:
                break
        
        print(f"{criterion:12} vs max_change: {'Same' if same_policy else 'Different'}")
    
    return results


if __name__ == "__main__":
    # Test value iteration
    from mdp import create_simple_mdp_example
    from policy_evaluation import evaluate_policy_performance
    from policy_iteration import policy_iteration
    
    print("Testing Value Iteration")
    print("=" * 50)
    
    # Create test MDP
    mdp = create_simple_mdp_example()
    print(f"MDP: {mdp}")
    print(f"States: {mdp.states}")
    print(f"Actions: {mdp.actions}")
    print(f"Discount factor: γ = {mdp.gamma}")
    
    # Test 1: Basic value iteration
    print(f"\n" + "="*60)
    print("TEST 1: Basic Value Iteration")
    print("="*60)
    
    V_opt, policy_opt, info = value_iteration_with_analysis(mdp)
    
    # Test 2: Compare with policy iteration
    print(f"\n" + "="*60)
    print("TEST 2: Comparison with Policy Iteration")
    print("="*60)
    
    # Run policy iteration for comparison
    policy_pi, V_pi, info_pi = policy_iteration(mdp, verbose=False)
    
    print("Value Iteration vs Policy Iteration:")
    print("Iterations:")
    print(f"  Value Iteration:  {info['iterations']:3d}")
    print(f"  Policy Iteration: {info_pi['iterations']:3d}")
    
    print("Value function differences:")
    max_diff = 0
    for s in mdp.states:
        diff = abs(V_opt[s] - V_pi[s])
        max_diff = max(max_diff, diff)
        print(f"  |V_VI({s}) - V_PI({s})| = {diff:.10f}")
    print(f"Maximum difference: {max_diff:.10f}")
    
    print("Policy comparison:")
    for s in mdp.states:
        vi_action = max(policy_opt[s].items(), key=lambda x: x[1])[0]
        pi_action = max(policy_pi[s].items(), key=lambda x: x[1])[0]
        match = "✓" if vi_action == pi_action else "✗"
        print(f"  {s}: VI={vi_action}, PI={pi_action} {match}")
    
    # Test 3: Different stopping criteria
    print(f"\n" + "="*60)
    print("TEST 3: Different Stopping Criteria")
    print("="*60)
    
    comparison_results = compare_value_iteration_variants(mdp, theta=1e-6)
    
    # Test 4: Convergence analysis
    print(f"\n" + "="*60)
    print("TEST 4: Convergence Analysis")
    print("="*60)
    
    # Test different theta values
    theta_values = [1e-3, 1e-6, 1e-9]
    
    for theta in theta_values:
        V, policy, info = value_iteration(mdp, theta=theta, verbose=False)
        det_policy = {s: max(policy[s].items(), key=lambda x: x[1])[0] for s in mdp.states}
        
        print(f"θ = {theta:.0e}: {info['iterations']:3d} iterations, "
              f"final Δ = {info['final_delta']:.2e}")
        print(f"  Policy: {det_policy}")
        print(f"  Values: {[f'{V[s]:.6f}' for s in mdp.states]}")
    
    # Test 5: Performance evaluation
    print(f"\n" + "="*60)
    print("TEST 5: Optimal Policy Performance")
    print("="*60)
    
    # Evaluate optimal policy performance
    performance = evaluate_policy_performance(mdp, policy_opt, n_episodes=1000)
    
    print("Optimal policy performance (1000 episodes):")
    for k, v in performance.items():
        print(f"  {k}: {v:.4f}")
    
    # Compare with random policy
    from policy_evaluation import create_random_policy
    random_policy = create_random_policy(mdp)
    random_performance = evaluate_policy_performance(mdp, random_policy, n_episodes=1000)
    
    print("Random policy performance (1000 episodes):")
    for k, v in random_performance.items():
        print(f"  {k}: {v:.4f}")
    
    improvement = performance['mean_return'] - random_performance['mean_return']
    print(f"\nImprovement over random policy: {improvement:.4f}")
    
    if random_performance['mean_return'] != 0:
        relative_improvement = improvement / abs(random_performance['mean_return']) * 100
        print(f"Relative improvement: {relative_improvement:.1f}%")
