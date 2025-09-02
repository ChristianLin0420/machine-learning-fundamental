# Day 41 — Reinforcement Learning: Markov Decision Processes (MDP)

## 📌 Overview
A comprehensive implementation of Markov Decision Processes (MDPs) and fundamental dynamic programming algorithms for reinforcement learning. This project demonstrates the theoretical foundations of RL through practical implementations of Policy Evaluation, Policy Iteration, and Value Iteration.

## 🧠 Theory

### Markov Decision Process Definition
An MDP is formally defined by the tuple **(S, A, P, R, γ)** where:

- **S**: Finite state space
- **A**: Finite action space  
- **P(s'|s,a)**: Transition probability from state s to s' given action a
- **R(s,a)**: Immediate reward for taking action a in state s
- **γ ∈ [0,1]**: Discount factor for future rewards

### Key Equations

#### Bellman Expectation Equation
The value of a state under policy π:
```
V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a)[R(s,a) + γV^π(s')]
```

#### Bellman Optimality Equation  
The optimal value function:
```
V*(s) = max_a Σ_{s'} P(s'|s,a)[R(s,a) + γV*(s')]
```

#### Optimal Policy
```
π*(s) = argmax_a Σ_{s'} P(s'|s,a)[R(s,a) + γV*(s')]
```

## 🛠️ Implementation

### Core Components

#### 1. FiniteMDP Class (`mdp.py`)
- Complete MDP representation with validation
- Support for both probabilistic and deterministic transitions
- Matrix-based operations for efficient computation
- State/action space management and indexing

#### 2. Policy Evaluation (`policy_evaluation.py`)
- Iterative policy evaluation algorithm
- Matrix-based policy evaluation (faster for large state spaces)
- Action-value function computation
- Policy performance simulation

#### 3. Policy Iteration (`policy_iteration.py`) 
- Complete policy iteration algorithm
- Policy improvement step
- Convergence tracking and history
- Deterministic policy extraction

#### 4. Value Iteration (`value_iteration.py`)
- Value iteration with Bellman optimality updates
- Multiple stopping criteria (max change, mean change, policy stability)
- Convergence analysis and comparison tools
- Direct optimal policy extraction

#### 5. GridWorld Demo (`gridworld_demo.py`)
- Classic 4x4 GridWorld environment
- Visualization of values, policies, and convergence
- Stochastic GridWorld with slip probability
- Comprehensive algorithm comparison

## 📁 Project Structure

```
41_rl_mdp/
├── mdp.py                 # Core MDP implementation
├── policy_evaluation.py   # Policy evaluation algorithms
├── policy_iteration.py    # Policy iteration algorithm
├── value_iteration.py     # Value iteration algorithm  
├── gridworld_demo.py      # GridWorld demonstration
├── README.md              # This file
└── gridworld_demo.png     # Generated visualization
```

## 🚀 Usage

### Quick Start
```bash
# Navigate to the project directory
cd 41_rl_mdp

# Test individual components
python mdp.py                    # Test MDP implementation
python policy_evaluation.py     # Test policy evaluation
python policy_iteration.py      # Test policy iteration
python value_iteration.py       # Test value iteration

# Run comprehensive GridWorld demonstration
python gridworld_demo.py
```

### Basic Example
```python
from mdp import FiniteMDP
from policy_iteration import policy_iteration
from value_iteration import value_iteration

# Create a simple MDP
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

mdp = FiniteMDP(states, actions, P, gamma=0.9)

# Solve using Policy Iteration
policy_pi, V_pi, info_pi = policy_iteration(mdp, verbose=True)

# Solve using Value Iteration
V_vi, policy_vi, info_vi = value_iteration(mdp, verbose=True)

print(f"Policy Iteration converged in {info_pi['iterations']} iterations")
print(f"Value Iteration converged in {info_vi['iterations']} iterations")
```

### GridWorld Example
```python
from gridworld_demo import GridWorld
from policy_iteration import policy_iteration
from value_iteration import value_iteration

# Create 4x4 GridWorld
gridworld = GridWorld(
    grid_size=(4, 4),
    obstacles=[(1, 1)],
    terminals={(0, 3): 1.0, (1, 3): -1.0},
    step_reward=-0.04
)

# Solve with both algorithms
policy_pi, V_pi, _ = policy_iteration(gridworld.mdp)
V_vi, policy_vi, _ = value_iteration(gridworld.mdp)

# Visualize results
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
gridworld.visualize_grid(axes[0], "GridWorld")
gridworld.visualize_values(V_pi, axes[1], "Optimal Values") 
gridworld.visualize_policy(policy_pi, axes[2], "Optimal Policy")
plt.show()
```

## 📊 Results & Analysis

### Algorithm Comparison

The implementation demonstrates several key insights:

#### 1. **Convergence Properties**
- **Policy Iteration**: Typically requires fewer iterations but more computation per iteration
- **Value Iteration**: More iterations but simpler updates
- Both algorithms are guaranteed to converge to the optimal solution

#### 2. **Computational Complexity**
- **Policy Evaluation**: O(|S|³) for matrix method, O(|S|²) per iteration for iterative
- **Policy Iteration**: O(k₁|S|²|A| + k₂|S|³) where k₁, k₂ are iteration counts
- **Value Iteration**: O(k|S|²|A|) where k is number of iterations

#### 3. **GridWorld Insights**
```
Standard 4x4 GridWorld Results:
- Optimal policy avoids obstacles and reaches positive terminal
- Value function shows clear gradient toward goal
- Stochastic version leads to more conservative policies
```

### Performance Metrics

| Algorithm | Iterations | Time Complexity | Space Complexity |
|-----------|------------|-----------------|------------------|
| Policy Evaluation | ~20-50 | O(|S|²) per iter | O(|S|) |
| Policy Iteration | ~5-10 | O(|S|²|A|) per iter | O(|S||A|) |
| Value Iteration | ~10-30 | O(|S||A|) per iter | O(|S|) |

## 🔬 Advanced Features

### 1. **Matrix-Based Policy Evaluation**
```python
# Solve linear system directly: V = (I - γP^π)^(-1)R^π
from policy_evaluation import policy_evaluation_matrix

V = policy_evaluation_matrix(mdp, policy, verbose=True)
```

### 2. **Stochastic Environments**
```python
# Create environment with slip probability
stochastic_mdp = gridworld.create_stochastic_gridworld(slip_prob=0.2)
```

### 3. **Convergence Analysis**
```python
# Different stopping criteria for value iteration
from value_iteration import modified_value_iteration

V, policy, info = modified_value_iteration(
    mdp, stopping_criterion='policy_stable'
)
```

### 4. **Visualization Tools**
- Grid layout with obstacles and terminals
- Value function heatmaps
- Policy visualization with arrows
- Convergence tracking plots

## 🎯 Key Insights

### 1. **Markov Property**
The future depends only on the current state, not the history:
```
P(s_{t+1}|s_t, a_t, s_{t-1}, ..., s_0) = P(s_{t+1}|s_t, a_t)
```

### 2. **Optimal Substructure**
Optimal policies have the property that whatever the initial state and decision, the remaining decisions must constitute an optimal policy with regard to the state resulting from the first decision.

### 3. **Policy vs Value Iteration Trade-offs**
- **Policy Iteration**: Better when good initial policy available
- **Value Iteration**: Better for cold starts and when policy not explicitly needed

### 4. **Discount Factor Impact**
- γ → 0: Only immediate rewards matter (myopic)
- γ → 1: All future rewards equally important (far-sighted)
- γ < 1: Ensures convergence in infinite horizon problems

## 📚 Educational Value

This implementation serves as:

1. **Theoretical Foundation**: Clear connection between mathematical formulations and code
2. **Algorithm Comparison**: Side-by-side analysis of different approaches  
3. **Visualization**: Intuitive understanding through GridWorld graphics
4. **Extensibility**: Framework for more complex RL algorithms

## 🔍 Extensions & Future Work

### Possible Extensions
1. **Approximate Methods**: Function approximation for large state spaces
2. **Temporal Difference Learning**: TD(0), SARSA, Q-Learning
3. **Monte Carlo Methods**: Model-free policy evaluation  
4. **Continuous Spaces**: Discretization and interpolation techniques
5. **Multi-Agent MDPs**: Game-theoretic extensions

### Advanced Algorithms
- **Modified Policy Iteration**: Truncated policy evaluation
- **Asynchronous Methods**: In-place value updates
- **Prioritized Sweeping**: Focus on important state updates
- **Real-Time Dynamic Programming**: Anytime algorithms

## 📖 References

1. **Sutton, R. S., & Barto, A. G.** (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT Press.
2. **Bellman, R.** (1957). *Dynamic Programming*. Princeton University Press.
3. **Puterman, M. L.** (2014). *Markov Decision Processes: Discrete Stochastic Dynamic Programming*. John Wiley & Sons.
4. **Bertsekas, D. P.** (2012). *Dynamic Programming and Optimal Control* (4th ed.). Athena Scientific.

## 🏃‍♂️ Quick Test

Run the complete demonstration:
```bash
python gridworld_demo.py
```

Expected output:
- Algorithm convergence statistics
- Policy comparisons
- Value function analysis  
- Visualization generation (`gridworld_demo.png`)

This implementation provides a solid foundation for understanding MDPs and serves as a stepping stone to more advanced reinforcement learning algorithms!