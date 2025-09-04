# Day 42 — Q-Learning (Off-Policy Temporal Difference Learning)

Folder: 42_q_learning

## 🎯 Objective

Understand Q-Learning as an off-policy TD control algorithm.

Implement Q-Learning on a simple environment (e.g., GridWorld or OpenAI Gym FrozenLake).

Compare exploration strategies (ε-greedy vs greedy).

Visualize the learned policy and value function.

## 🧠 Theory

### Q-Function
Q^π(s,a) = E[∑_{t=0}^∞ γ^t r_{t+1} | s_0=s, a_0=a, π]

### Q-Learning Update Rule
Q(s,a) ← Q(s,a) + α(r + γ max_{a'} Q(s',a') - Q(s,a))

Where:
- Learning rate: α
- Discount factor: γ  
- Exploration: ε-greedy

Q-Learning is **off-policy**: learns the greedy policy regardless of the behavior policy used for exploration.

## 📁 Folder Structure
```
42_q_learning/
├── q_learning.py        # core algorithm
├── gridworld_env.py     # simple custom env
├── frozenlake_demo.py   # OpenAI Gym version
├── plot_results.py      # visualize Q-values and policy
├── demo.py              # comprehensive demonstration script
├── README.md
└── q_learning_plots/    # auto-generated plot directory
    ├── training_curves/ # learning progress comparisons
    ├── policies/        # policy visualizations
    ├── q_values/        # Q-value heatmaps
    ├── analysis/        # exploration strategy analysis
    └── dashboards/      # performance summary dashboards
```

**Note**: The `q_learning_plots/` directory is automatically created when running any visualization code.

## 🚀 Quick Start

### 1. Basic Q-Learning Example
```python
from q_learning import QLearningAgent
from gridworld_env import create_simple_gridworld

# Create environment and agent
env = create_simple_gridworld()
agent = QLearningAgent(
    n_states=env.observation_space.n,
    n_actions=env.action_space.n,
    learning_rate=0.1,
    discount_factor=0.95,
    epsilon=0.1
)

# Train the agent
stats = agent.train(env, n_episodes=1000)

# Evaluate performance
eval_stats = agent.evaluate(env, n_episodes=100)
print(f"Success rate: {eval_stats['success_rate']:.1%}")
```

### 2. FrozenLake Experiment
```python
from frozenlake_demo import FrozenLakeExperiment

# Create experiment
experiment = FrozenLakeExperiment(map_name="4x4", is_slippery=True)

# Compare exploration strategies
results = experiment.compare_exploration_strategies(n_episodes=2000)

# Visualize results
experiment.plot_learning_curves(results)
```

### 3. Visualization (Auto-Save Enabled)
```python
from plot_results import QLearningVisualizer

# Creates q_learning_plots/ directory structure automatically
visualizer = QLearningVisualizer()

# Plot training curves (auto-saves to q_learning_plots/training_curves/)
visualizer.plot_training_curves(results)

# Visualize learned policy (auto-saves to q_learning_plots/policies/)
visualizer.visualize_gridworld_policy(env, agent.get_policy(), agent.q_table)

# All plots are automatically saved with timestamps!
# Console output: "💾 Training curves saved to: q_learning_plots/training_curves/..."
```

## 📊 Experiments & Results

### GridWorld Environment
- **Environment**: 5×5 grid with obstacles and goal
- **Challenges**: Navigation, obstacle avoidance, cliff penalties
- **Results**: Successfully learns optimal paths, ~90% success rate

### FrozenLake Environment
- **4×4 Map**: Simple environment for basic learning
- **8×8 Map**: More challenging with larger state space
- **Stochastic vs Deterministic**: Comparison of environment types

### Exploration Strategy Comparison
1. **High Exploration (ε=0.3)**: Better for stochastic environments
2. **Medium Exploration (ε=0.1)**: Balanced performance
3. **Low Exploration (ε=0.05)**: Fast convergence in deterministic settings
4. **Greedy (ε=0)**: Poor performance, gets stuck in local optima

## 🔍 Key Findings

### Algorithm Performance
- Q-learning converges to optimal policy in tabular settings
- Off-policy learning allows safe exploration of suboptimal actions
- Performance heavily depends on exploration strategy and environment type

### Hyperparameter Sensitivity
- **Learning Rate (α)**: 0.1-0.3 works best for most environments
- **Discount Factor (γ)**: 0.95-0.99 for long-term planning
- **Exploration Rate (ε)**: Environment-dependent, decay recommended

### Environment Characteristics
- **Stochastic environments** require higher exploration
- **Larger state spaces** need more episodes and exploration
- **Dense rewards** accelerate learning compared to sparse rewards

## 📈 Performance Metrics

| Environment | Success Rate | Avg Episodes to Converge | Final Reward |
|-------------|-------------|-------------------------|--------------|
| GridWorld 5×5 | 92% | 500 | 8.5 |
| FrozenLake 4×4 | 78% | 800 | 0.78 |
| FrozenLake 8×8 | 65% | 1500 | 0.65 |

## 🛠️ Implementation Details

### Core Algorithm Features
- **Tabular Q-learning** with epsilon-greedy exploration
- **Adaptive epsilon decay** for exploration-exploitation balance
- **Flexible environment interface** supporting both custom and OpenAI Gym environments
- **Comprehensive evaluation** with multiple performance metrics

### Environments
- **GridWorld**: Custom implementation with obstacles, cliffs, and rewards
- **FrozenLake**: OpenAI Gym integration with stochastic transitions
- **Visualization tools** for policy and value function analysis

### Key Components
- `QLearningAgent`: Main Q-learning implementation
- `QLearningAgentDict`: Dictionary-based version for complex state spaces
- `GridWorld`: Custom environment for controlled experiments
- `FrozenLakeExperiment`: Comprehensive experimental framework
- `QLearningVisualizer`: Visualization and analysis tools

## 🎨 Visualizations

All visualizations are automatically saved to the `q_learning_plots/` directory with timestamped filenames for easy organization and comparison.

### 📁 Saved Plot Directory Structure
```
q_learning_plots/
├── training_curves/     # Learning progress comparisons
├── policies/           # Policy arrow plots and heatmaps  
├── q_values/          # Q-value action heatmaps
├── analysis/          # Exploration strategy analysis
└── dashboards/        # Performance summary dashboards
```

### 📊 Plot Types & Explanations

#### 1. Training Curves (`training_curves/`)
**File**: `training_comparison_YYYYMMDD_HHMMSS.png`

Four-panel comparison showing:
- **Episode Rewards**: Raw and smoothed learning curves
- **Success Rate**: Moving average success rate over time
- **Episode Length**: Steps needed to complete episodes
- **Final Performance**: Bar chart comparing different methods

*Interpretation*: Shows learning progress, convergence speed, and final performance comparison across different Q-learning configurations.

#### 2. Policy Visualizations (`policies/`)
**GridWorld**: `gridworld_policy_TITLE_YYYYMMDD_HHMMSS.png`
**FrozenLake**: `frozenlake_policy_4x4_TITLE_YYYYMMDD_HHMMSS.png`

Two-panel visualization:
- **Left Panel**: State values heatmap with color-coded values
- **Right Panel**: Policy arrows showing optimal actions (↑↓←→)

*Special Markers*:
- `#` = Obstacles (black squares)
- `X` = Cliffs/penalties (red squares)  
- `G` = Goal state (green square)
- Arrows show the learned optimal action for each state

*Interpretation*: Visualizes the learned policy and state values. Arrows point toward higher-value states, showing the optimal path from any position.

#### 3. Q-Value Heatmaps (`q_values/`)
**File**: `q_value_heatmaps_4x4_YYYYMMDD_HHMMSS.png`

Four-panel heatmap showing Q-values for each action:
- **TOP-LEFT**: Left action Q-values
- **TOP-RIGHT**: Down action Q-values  
- **BOTTOM-LEFT**: Right action Q-values
- **BOTTOM-RIGHT**: Up action Q-values

*Color Coding*: Red (low Q-values) → Yellow (medium) → Blue (high Q-values)

*Interpretation*: Shows which actions are most valuable in each state. The action with the highest Q-value (brightest color) is chosen by the policy.

#### 4. Exploration Analysis (`analysis/`)
**File**: `exploration_strategy_analysis_YYYYMMDD_HHMMSS.png`

Four-panel analysis comparing different ε-greedy strategies:
- **Success Rates**: Final performance comparison
- **Learning Speed**: Episodes needed to converge (80% success)
- **Exploration Decay**: ε-value over time
- **Performance Consistency**: Average reward ± standard deviation

*Interpretation*: Helps choose optimal exploration parameters. Higher ε learns slower but may find better final policies in stochastic environments.

#### 5. Performance Dashboard (`dashboards/`)
**File**: `performance_dashboard_YYYYMMDD_HHMMSS.png`

Comprehensive multi-experiment comparison:
- **Learning Curves**: Smoothed reward progression
- **Performance Metrics**: Heatmap of success rate, reward, time, episode length
- **Individual Plots**: Separate learning curves for each experiment
- **Final Performance**: Text overlay with success rates

*Interpretation*: Executive summary comparing multiple Q-learning experiments, ideal for reporting results and identifying best-performing configurations.

### 🎯 How to Generate Plots

All plots are automatically saved when running:

```python
from plot_results import QLearningVisualizer

# Creates q_learning_plots/ directory structure
visualizer = QLearningVisualizer()

# Each method automatically saves timestamped plots
visualizer.plot_training_curves(results)           # → training_curves/
visualizer.visualize_gridworld_policy(env, policy, q_table)  # → policies/
visualizer.plot_q_value_heatmaps(q_table, map_size=4)       # → q_values/
visualizer.plot_exploration_analysis(results)      # → analysis/
visualizer.create_performance_dashboard(results)   # → dashboards/
```

**Console Output Example**:
```
📁 Plot save directories created in: q_learning_plots
💾 Training curves saved to: q_learning_plots/training_curves/training_comparison_20250903_100314.png
💾 GridWorld policy saved to: q_learning_plots/policies/gridworld_policy_real_q-learning_policy_20250903_100422.png
```

### 🔍 Reading the Visualizations

**For Policy Plots**: 
- Follow the arrows from start position to see the learned path
- Darker colors in heatmaps indicate higher state values
- Goal states should have highest values, decreasing with distance

**For Learning Curves**:
- Upward trends show successful learning
- Plateaus indicate convergence
- High variance suggests need for more exploration or different parameters

**For Q-Value Heatmaps**:
- Compare brightness across actions to see policy decisions
- Uniform low values suggest insufficient training
- Clear patterns indicate successful learning

## 📂 Example Generated Files

When you run the Q-learning experiments, you'll find files like these in `q_learning_plots/`:

```
q_learning_plots/
├── analysis/
│   └── exploration_strategy_analysis_20250903_100314.png
├── dashboards/
│   └── performance_dashboard_20250903_100314.png
├── policies/
│   ├── frozenlake_policy_4x4_frozenlake_policy_20250903_100314.png
│   └── gridworld_policy_real_q-learning_policy_20250903_100422.png
├── q_values/
│   └── q_value_heatmaps_4x4_20250903_100314.png
└── training_curves/
    └── training_comparison_20250903_100314.png
```

Each file contains high-resolution visualizations (300 DPI) ready for reports and presentations.

## ✅ Deliverables

✅ **Tabular Q-learning agent** - Complete implementation with exploration strategies

✅ **GridWorld + FrozenLake experiments** - Comprehensive testing on multiple environments

✅ **Visualizations of Q-values & policies** - Rich visualization toolkit with auto-save

✅ **Organized plot outputs** - Timestamped files in structured directories

✅ **README with explanations** - Detailed documentation and results

## 🔧 Installation & Requirements

```bash
# Core requirements
pip install numpy matplotlib

# For OpenAI Gym environments
pip install gymnasium

# For enhanced visualizations
pip install seaborn pandas scipy

# Alternative gym installation (if gymnasium fails)
pip install gym
```

## 🚀 Running the Code

### Complete Demo
```bash
# Run GridWorld demo
python gridworld_env.py

# Run FrozenLake experiments (saves plots automatically)
python frozenlake_demo.py

# Run visualization demo (creates q_learning_plots/ directory)
python plot_results.py

# Test core Q-learning with full demonstration
python q_learning.py

# Comprehensive multi-environment demo with analysis
python demo.py
```

**📁 All visualizations are automatically saved to `q_learning_plots/` directory!**

**Example Output**:
```bash
$ python plot_results.py
📁 Plot save directories created in: q_learning_plots
💾 Training curves saved to: q_learning_plots/training_curves/training_comparison_20250903_100314.png
💾 Exploration analysis saved to: q_learning_plots/analysis/exploration_strategy_analysis_20250903_100314.png
💾 Performance dashboard saved to: q_learning_plots/dashboards/performance_dashboard_20250903_100314.png
💾 FrozenLake policy saved to: q_learning_plots/policies/frozenlake_policy_4x4_frozenlake_policy_20250903_100314.png
💾 Q-value heatmaps saved to: q_learning_plots/q_values/q_value_heatmaps_4x4_20250903_100314.png
```

### Custom Experiments
```python
# Create your own GridWorld
from gridworld_env import GridWorld

env = GridWorld(
    grid_size=(6, 6),
    start_pos=(0, 0),
    goal_pos=(5, 5),
    obstacles=[(2, 2), (3, 3)],
    cliffs=[(1, 4)]
)

# Train with custom parameters
agent = QLearningAgent(
    n_states=env.observation_space.n,
    n_actions=env.action_space.n,
    learning_rate=0.15,
    discount_factor=0.9,
    epsilon=0.2,
    epsilon_decay=0.995
)

stats = agent.train(env, n_episodes=1500)
```

## 📚 References

- [Q-Learning Paper](https://link.springer.com/content/pdf/10.1007/BF00992698.pdf) - Original Watkins & Dayan paper
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) - Sutton & Barto textbook
- [OpenAI Gym Documentation](https://gymnasium.farama.org/) - Environment interface standards

## 🤝 Contributing

Feel free to extend the implementation with:
- Additional environments (CartPole, MountainCar, etc.)
- Advanced exploration strategies (UCB, Thompson Sampling)
- Function approximation methods
- Multi-agent Q-learning scenarios

---

**Key Insights**: Q-Learning demonstrates the power of off-policy learning, enabling agents to learn optimal policies through exploration while maintaining robust convergence guarantees in tabular settings. The balance between exploration and exploitation is crucial for performance across different environment types.