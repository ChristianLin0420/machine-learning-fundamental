# SARSA Algorithm

## 📌 Overview
Implement SARSA (State-Action-Reward-State-Action) as an on-policy temporal difference learning algorithm for reinforcement learning. This implementation includes comprehensive comparison with Q-Learning across multiple environments.

## 🧠 Key Concepts

### On-policy vs Off-policy Learning
- **SARSA (On-policy)**: Learns the value of the policy being followed, using the action actually taken by the current policy
- **Q-Learning (Off-policy)**: Learns the optimal action-value function regardless of the policy being followed

### SARSA Update Rule
```
Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]
```
Where:
- `α` is the learning rate
- `γ` is the discount factor
- `r` is the immediate reward
- `s'` is the next state
- `a'` is the next action (selected by current policy)

### Key Differences from Q-Learning
- **SARSA**: Uses `Q(s',a')` where `a'` is the action selected by the current policy
- **Q-Learning**: Uses `max_a' Q(s',a')` (maximum over all possible actions)

## 🛠️ Implementation

### Files Structure
```
43_sarsa/
├── sarsa.py              # Core SARSA and Q-Learning implementations
├── gridworld_env.py      # Custom GridWorld environment
├── frozenlake_demo.py    # FrozenLake environment comparison
├── plot_compare.py       # Comprehensive comparison and visualization
├── plots/                # Generated visualizations
└── README.md
```

### Core Features
- **SARSAAgent**: Complete SARSA implementation with epsilon-greedy exploration
- **QLearningAgent**: Q-Learning implementation for comparison
- **GridWorld Environment**: Custom environment with obstacles and stochasticity
- **Comprehensive Evaluation**: Multiple environments and statistical analysis
- **Visualization Tools**: Learning curves, policy comparison, Q-value heatmaps

### Key Parameters
- `learning_rate`: Controls how much the Q-values are updated (default: 0.1)
- `discount_factor`: Importance of future rewards (default: 0.95)
- `epsilon`: Exploration rate (default: 1.0, decays to 0.01)
- `epsilon_decay`: Rate of exploration decay (default: 0.995)

## 📊 Results

### GridWorld Comparison Results

#### Simple GridWorld (5x5, no obstacles)
- **SARSA**: 100% success rate, 8.0 mean episode length
- **Q-Learning**: 100% success rate, 8.0 mean episode length
- **Insight**: Both algorithms perform equally well in deterministic environments

#### GridWorld with Obstacles
- **SARSA**: 100% success rate, 8.0 mean episode length
- **Q-Learning**: 100% success rate, 8.0 mean episode length
- **Insight**: Both algorithms learn to navigate around obstacles effectively

#### Stochastic GridWorld (10% random actions)
- **SARSA**: 100% success rate, 9.08 mean episode length
- **Q-Learning**: 100% success rate, 8.93 mean episode length
- **Insight**: SARSA shows slightly more conservative behavior (longer episodes) in stochastic environments

### FrozenLake Results
- **SARSA**: 0% success rate (very challenging environment)
- **Q-Learning**: 0% success rate (very challenging environment)
- **Insight**: Both algorithms struggle with the high stochasticity of FrozenLake

### Key Insights

1. **Policy Safety**: SARSA tends to learn safer policies in stochastic environments
2. **Convergence Speed**: Q-Learning may converge faster in some cases
3. **Environment Dependency**: Performance varies significantly based on environment characteristics
4. **Exploration Strategy**: Both algorithms benefit from proper exploration scheduling

## 🎯 Usage Examples

### Basic SARSA Training
```python
from sarsa import SARSAAgent
from gridworld_env import create_simple_gridworld

# Create environment and agent
env = create_simple_gridworld(size=5)
agent = SARSAAgent(state_size=25, action_size=4)

# Train the agent
stats = agent.train(env, num_episodes=1000)

# Evaluate performance
eval_results = agent.evaluate(env, num_episodes=100)
print(f"Success rate: {eval_results['success_rate']:.2%}")
```

### Comparison with Q-Learning
```python
from sarsa import SARSAAgent, QLearningAgent, plot_learning_curves

# Train both agents
sarsa_agent = SARSAAgent(state_size=25, action_size=4)
qlearning_agent = QLearningAgent(state_size=25, action_size=4)

sarsa_stats = sarsa_agent.train(env, 1000)
qlearning_stats = qlearning_agent.train(env, 1000)

# Compare learning curves
plot_learning_curves(sarsa_stats, qlearning_stats)
```

### Run Comprehensive Comparison
```python
# Run all comparisons and generate visualizations
python plot_compare.py
```

## 📈 Generated Visualizations

The implementation generates comprehensive visualizations:

1. **Learning Curves**: Episode rewards and lengths over time
2. **Policy Comparison**: Visual representation of learned policies
3. **Q-Value Heatmaps**: Q-values for each action across states
4. **Convergence Analysis**: Detailed convergence behavior analysis
5. **Statistical Comparison**: Performance across multiple seeds

## 🔬 Technical Details

### Algorithm Complexity
- **Time Complexity**: O(|S| × |A|) per episode
- **Space Complexity**: O(|S| × |A|) for Q-table storage
- **Convergence**: Guaranteed under certain conditions (sufficient exploration)

### Hyperparameter Sensitivity
- **Learning Rate**: Too high causes instability, too low slows learning
- **Epsilon Decay**: Affects exploration-exploitation balance
- **Discount Factor**: Higher values prioritize long-term rewards

### Environment Considerations
- **Deterministic**: Both algorithms perform similarly
- **Stochastic**: SARSA may be more conservative
- **Sparse Rewards**: Both algorithms may struggle without proper reward shaping

## 📚 References
- [SARSA Algorithm](https://en.wikipedia.org/wiki/State%E2%80%93action%E2%80%93reward%E2%80%93state%E2%80%93action)
- [Sutton & Barto Chapter 6](http://incompleteideas.net/book/the-book-2nd.html)
- [SARSA vs Q-Learning](https://towardsdatascience.com/sarsa-vs-q-learning-7267a85c47bd) 
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)

## 🚀 Future Extensions
- Expected SARSA implementation
- Function approximation for large state spaces
- Multi-agent SARSA
- Continuous action spaces
- Deep SARSA (DQN with SARSA updates) 