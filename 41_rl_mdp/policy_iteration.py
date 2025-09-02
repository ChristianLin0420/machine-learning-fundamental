"""
Policy Iteration for Markov Decision Processes.

Implements policy iteration algorithm that alternates between:
1. Policy Evaluation: Compute V^π for current policy π
2. Policy Improvement: Update policy to be greedy w.r.t. current value function

The algorithm is guaranteed to converge to the optimal policy π* and value function V*.
"""

import numpy as np
from typing import Dict, List, Tuple
from mdp import FiniteMDP
from policy_evaluation import policy_evaluation, policy_evaluation_matrix, compute_action_values


def policy_improvement(mdp: FiniteMDP, 
                      V: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """
    Policy improvement step: create greedy policy w.r.t. value function V.
    
    π_new(s) = argmax_a Σ_{s'} P(s'|s,a)[R(s,a) + γV(s')]
    
    Args:
        mdp: The MDP environment
        V: Current value function
        
    Returns:
        policy: Improved policy (deterministic)
    """
    policy = {}
    
    for s in mdp.states:
        # Compute action values for all actions
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
                policy[s][a] = 1.0 / len(best_actions)  # Equal probability for ties
            else:
                policy[s][a] = 0.0
    
    return policy


def policy_iteration(mdp: FiniteMDP,
                    initial_policy: Dict[str, Dict[str, float]] = None,
                    theta: float = 1e-6,
                    max_iterations: int = 100,
                    use_matrix_eval: bool = False,
                    verbose: bool = False) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], Dict]:
    """
    Policy iteration algorithm.
    
    Args:
        mdp: The MDP environment
        initial_policy: Starting policy (random if None)
        theta: Convergence threshold for policy evaluation
        max_iterations: Maximum number of policy iterations
        use_matrix_eval: Use matrix-based policy evaluation
        verbose: Print iteration details
        
    Returns:
        policy: Optimal policy
        V: Optimal value function
        info: Convergence information
    """
    # Initialize policy
    if initial_policy is None:
        policy = {}
        for s in mdp.states:
            policy[s] = {a: 1.0/len(mdp.actions) for a in mdp.actions}
    else:
        policy = initial_policy.copy()
    
    policy_history = []
    value_history = []
    
    if verbose:
        print("Starting Policy Iteration")
        print("=" * 40)
    
    for iteration in range(max_iterations):
        if verbose:
            print(f"\nIteration {iteration + 1}")
            print("-" * 20)
        
        # Policy Evaluation
        if use_matrix_eval:
            V = policy_evaluation_matrix(mdp, policy, theta, verbose=verbose)
        else:
            V = policy_evaluation(mdp, policy, theta, verbose=verbose)
        
        # Store history
        policy_history.append({s: dict(policy[s]) for s in mdp.states})
        value_history.append(dict(V))
        
        if verbose:
            print("Current value function:")
            for s in mdp.states:
                print(f"  V({s}) = {V[s]:.6f}")
        
        # Policy Improvement
        new_policy = policy_improvement(mdp, V)
        
        if verbose:
            print("Policy improvement:")
            for s in mdp.states:
                old_actions = [a for a, p in policy[s].items() if p > 0]
                new_actions = [a for a, p in new_policy[s].items() if p > 0]
                if old_actions != new_actions:
                    print(f"  π({s}): {old_actions} -> {new_actions}")
        
        # Check for policy stability
        policy_stable = True
        for s in mdp.states:
            for a in mdp.actions:
                if not np.isclose(policy[s][a], new_policy[s][a], atol=1e-10):
                    policy_stable = False
                    break
            if not policy_stable:
                break
        
        # Update policy
        policy = new_policy
        
        if policy_stable:
            if verbose:
                print(f"\nPolicy iteration converged after {iteration + 1} iterations")
            break
    else:
        if verbose:
            print(f"\nWarning: Policy iteration did not converge after {max_iterations} iterations")
    
    # Final evaluation with the converged policy
    if use_matrix_eval:
        V_final = policy_evaluation_matrix(mdp, policy, theta, verbose=False)
    else:
        V_final = policy_evaluation(mdp, policy, theta, verbose=False)
    
    info = {
        'iterations': iteration + 1,
        'converged': policy_stable,
        'policy_history': policy_history,
        'value_history': value_history
    }
    
    return policy, V_final, info


def extract_deterministic_policy(policy: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    """
    Extract deterministic policy from probabilistic representation.
    
    Args:
        policy: Probabilistic policy
        
    Returns:
        det_policy: Deterministic policy mapping states to actions
    """
    det_policy = {}
    for s, action_probs in policy.items():
        best_action = max(action_probs.items(), key=lambda x: x[1])[0]
        det_policy[s] = best_action
    return det_policy


def compare_policies(mdp: FiniteMDP, 
                    policy1: Dict[str, Dict[str, float]], 
                    policy2: Dict[str, Dict[str, float]]) -> Dict[str, bool]:
    """
    Compare two policies to see if they are equivalent.
    
    Args:
        mdp: The MDP environment
        policy1: First policy
        policy2: Second policy
        
    Returns:
        comparison: Dictionary indicating differences per state
    """
    comparison = {}
    
    for s in mdp.states:
        same = True
        for a in mdp.actions:
            if not np.isclose(policy1[s][a], policy2[s][a], atol=1e-10):
                same = False
                break
        comparison[s] = same
    
    return comparison


def policy_iteration_with_visualization(mdp: FiniteMDP, **kwargs) -> Tuple:
    """
    Policy iteration with detailed tracking for visualization.
    """
    policy, V, info = policy_iteration(mdp, verbose=True, **kwargs)
    
    print("\n" + "="*50)
    print("POLICY ITERATION SUMMARY")
    print("="*50)
    
    print(f"Converged: {info['converged']}")
    print(f"Iterations: {info['iterations']}")
    
    print(f"\nFinal Policy:")
    det_policy = extract_deterministic_policy(policy)
    for s in mdp.states:
        print(f"  π*({s}) = {det_policy[s]}")
    
    print(f"\nFinal Value Function:")
    for s in mdp.states:
        print(f"  V*({s}) = {V[s]:.6f}")
    
    # Compute action values for final policy
    Q = compute_action_values(mdp, V)
    print(f"\nFinal Action Values:")
    for s in mdp.states:
        print(f"  Q*({s}, ·):")
        for a in mdp.actions:
            marker = " ← optimal" if det_policy[s] == a else ""
            print(f"    Q*({s}, {a}) = {Q[(s, a)]:.6f}{marker}")
    
    return policy, V, info


if __name__ == "__main__":
    # Test policy iteration
    from mdp import create_simple_mdp_example
    from policy_evaluation import create_random_policy, evaluate_policy_performance
    
    print("Testing Policy Iteration")
    print("=" * 50)
    
    # Create test MDP
    mdp = create_simple_mdp_example()
    print(f"MDP: {mdp}")
    print(f"States: {mdp.states}")
    print(f"Actions: {mdp.actions}")
    print(f"Discount factor: γ = {mdp.gamma}")
    
    # Test 1: Policy iteration with random initial policy
    print(f"\n" + "="*50)
    print("TEST 1: Policy Iteration (Random Initial Policy)")
    print("="*50)
    
    initial_policy = create_random_policy(mdp)
    policy_opt, V_opt, info = policy_iteration_with_visualization(
        mdp, initial_policy=initial_policy, use_matrix_eval=False
    )
    
    # Test 2: Compare with matrix-based evaluation
    print(f"\n" + "="*50)
    print("TEST 2: Policy Iteration (Matrix-based Evaluation)")
    print("="*50)
    
    policy_opt2, V_opt2, info2 = policy_iteration(
        mdp, initial_policy=initial_policy, use_matrix_eval=True, verbose=True
    )
    
    print("Comparison of iterative vs matrix methods:")
    for s in mdp.states:
        diff = abs(V_opt[s] - V_opt2[s])
        print(f"  |V_iter({s}) - V_matrix({s})| = {diff:.10f}")
    
    # Test 3: Different initial policies
    print(f"\n" + "="*50)
    print("TEST 3: Different Initial Policies")
    print("="*50)
    
    from policy_evaluation import create_deterministic_policy
    
    # Test multiple deterministic initial policies
    test_policies = [
        {'s0': 'a0', 's1': 'a0', 's2': 'a0'},
        {'s0': 'a1', 's1': 'a1', 's2': 'a1'},
        {'s0': 'a0', 's1': 'a1', 's2': 'a0'}
    ]
    
    results = []
    for i, policy_map in enumerate(test_policies):
        print(f"\nInitial policy {i+1}: {policy_map}")
        initial = create_deterministic_policy(mdp, policy_map)
        policy_result, V_result, info_result = policy_iteration(
            mdp, initial_policy=initial, verbose=False
        )
        results.append((policy_result, V_result, info_result))
        
        det_policy = extract_deterministic_policy(policy_result)
        print(f"  Converged to: {det_policy}")
        print(f"  Iterations: {info_result['iterations']}")
        print(f"  Values: {[f'{V_result[s]:.4f}' for s in mdp.states]}")
    
    # Verify all converge to same solution
    print("\nVerifying convergence to same optimal policy:")
    reference_policy = results[0][0]
    for i, (policy, V, info) in enumerate(results[1:], 1):
        comparison = compare_policies(mdp, reference_policy, policy)
        all_same = all(comparison.values())
        print(f"  Policy {i+1} vs Policy 1: {'Same' if all_same else 'Different'}")
    
    # Test 4: Performance comparison
    print(f"\n" + "="*50)
    print("TEST 4: Policy Performance Comparison")
    print("="*50)
    
    # Compare random vs optimal policy performance
    random_policy = create_random_policy(mdp)
    
    perf_random = evaluate_policy_performance(mdp, random_policy, n_episodes=1000)
    perf_optimal = evaluate_policy_performance(mdp, policy_opt, n_episodes=1000)
    
    print("Random policy performance:")
    for k, v in perf_random.items():
        print(f"  {k}: {v:.4f}")
    
    print("Optimal policy performance:")
    for k, v in perf_optimal.items():
        print(f"  {k}: {v:.4f}")
    
    improvement = perf_optimal['mean_return'] - perf_random['mean_return']
    print(f"\nImprovement: {improvement:.4f} ({improvement/abs(perf_random['mean_return'])*100:.1f}% relative)")
