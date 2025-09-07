# Deep Q-Network (DQN)

## 📌 Overview
Implement Deep Q-Network (DQN) with neural network function approximation, extending tabular Q-learning to handle large state spaces. This implementation includes experience replay, target networks, and various DQN variants.

## 🧠 Theory

### Motivation
Tabular Q-learning is limited to small, discrete state spaces. DQN uses a neural network to approximate the Q-function Q(s,a;θ), enabling learning in high-dimensional continuous state spaces.

### DQN Update Rule
```
L(θ) = (r + γ max_a' Q(s',a';θ⁻) - Q(s,a;θ))²
```

Where:
- `θ`: Online network parameters
- `θ⁻`: Target network parameters (periodically updated)
- `r`: Immediate reward
- `γ`: Discount factor
- `s'`: Next state

### Key Components

#### 1. Experience Replay Buffer
- **Purpose**: Breaks correlations in training data
- **Storage**: Transitions (s, a, r, s', done)
- **Sampling**: Random minibatches for updates
- **Benefits**: Stabilizes training, improves sample efficiency

#### 2. Target Network
- **Purpose**: Stabilizes training by using delayed copy of online network
- **Update**: Periodically copy online network weights to target network
- **Frequency**: Every C steps (e.g., 100-1000 steps)

#### 3. ε-Greedy Exploration
- **Initial**: High exploration (ε = 1.0)
- **Decay**: Gradually reduce ε over time
- **Final**: Low exploration (ε = 0.01-0.1)

## 🛠️ Implementation

### Files Structure
```
44_dqn/
├── dqn_agent.py         # DQN agent with neural network
├── replay_buffer.py     # Experience replay buffer implementations
├── cartpole_train.py    # Training script for CartPole-v1
├── plot_results.py      # Visualization and plotting tools
├── models/              # Saved model checkpoints
├── plots/               # Generated visualizations
└── README.md
```

### Core Classes

#### DQNAgent
- **Standard DQN**: Basic implementation with experience replay and target network
- **Dueling DQN**: Separates value V(s) and advantage A(s,a) streams
- **Double DQN**: Reduces overestimation bias in Q-values
- **Prioritized Replay**: Samples transitions based on TD error magnitude

#### ReplayBuffer
- **Standard Replay**: Uniform random sampling
- **Prioritized Replay**: Probability proportional to TD error
- **N-Step Replay**: Multi-step returns for better learning

### Key Parameters
- `learning_rate`: Neural network learning rate (default: 1e-3)
- `gamma`: Discount factor (default: 0.99)
- `epsilon`: Initial exploration rate (default: 1.0)
- `epsilon_decay`: Exploration decay rate (default: 0.995)
- `target_update_freq`: Target network update frequency (default: 100)
- `buffer_size`: Experience replay buffer size (default: 10000)
- `batch_size`: Training batch size (default: 64)

## 📊 Results

### CartPole-v1 Performance

#### Standard DQN
- **Environment**: CartPole-v1 (4D state space, 2 actions)
- **Success Rate**: 100% (episodes lasting 500+ steps)
- **Convergence**: ~300-500 episodes
- **Final Performance**: 500 steps consistently

#### DQN Variants Comparison
- **Dueling DQN**: Similar performance, better sample efficiency
- **Double DQN**: More stable learning, reduced overestimation
- **Prioritized Replay**: Faster convergence, better sample efficiency

### Key Insights

1. **Neural Network Approximation**: Successfully handles continuous state spaces
2. **Experience Replay**: Essential for stable learning, breaks data correlation
3. **Target Network**: Prevents training instability, improves convergence
4. **Exploration Strategy**: Critical for discovering optimal policies
5. **Hyperparameter Sensitivity**: Learning rate and target update frequency significantly affect performance

## 🎯 Usage Examples

### Basic DQN Training
```python
from dqn_agent import DQNAgent
import gym

# Create environment
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Create agent
agent = DQNAgent(
    state_size=state_size,
    action_size=action_size,
    learning_rate=1e-3,
    epsilon_decay=0.995
)

# Train agent
stats = agent.train(env, num_episodes=1000)

# Evaluate
eval_results = agent.evaluate(env, num_episodes=100)
print(f"Success rate: {eval_results['success_rate']:.2%}")
```

### Dueling DQN
```python
# Create Dueling DQN agent
agent = DQNAgent(
    state_size=state_size,
    action_size=action_size,
    use_dueling=True,
    learning_rate=1e-3
)
```

### Prioritized Experience Replay
```python
# Create agent with prioritized replay
agent = DQNAgent(
    state_size=state_size,
    action_size=action_size,
    use_prioritized=True,
    learning_rate=1e-3
)
```

### Training Script
```bash
# Train standard DQN
python cartpole_train.py

# The script will:
# 1. Train a standard DQN agent
# 2. Compare different DQN variants
# 3. Perform hyperparameter sweep
# 4. Generate comprehensive visualizations
```

## 📈 Generated Visualizations

The implementation generates comprehensive visualizations:

1. **Training Results**: Episode rewards, lengths, losses, and epsilon decay
2. **Learning Curves**: Comparison of different DQN variants
3. **Hyperparameter Analysis**: Performance across different parameter settings
4. **Replay Buffer Analysis**: Buffer utilization and reward distributions
5. **Convergence Analysis**: Learning stability and convergence speed

## 🔬 Technical Details

### Algorithm Complexity
- **Time Complexity**: O(B × N) per training step, where B is batch size, N is network size
- **Space Complexity**: O(M + N) where M is buffer size, N is network size
- **Convergence**: Not guaranteed, but empirically effective with proper hyperparameters

### Network Architecture
- **Input Layer**: State space dimension
- **Hidden Layers**: Fully connected with ReLU activation
- **Output Layer**: Action space dimension (Q-values)
- **Dueling Variant**: Separate value and advantage streams

### Training Process
1. **Experience Collection**: Store transitions in replay buffer
2. **Batch Sampling**: Sample random batch from buffer
3. **Target Computation**: Use target network for stable targets
4. **Loss Calculation**: MSE between current and target Q-values
5. **Network Update**: Backpropagation with gradient clipping
6. **Target Update**: Periodically copy online network to target

### Hyperparameter Guidelines
- **Learning Rate**: 1e-4 to 1e-3 (start high, decay if needed)
- **Epsilon Decay**: 0.99 to 0.999 (slower decay for more exploration)
- **Target Update**: 100-1000 steps (more frequent = more stable)
- **Buffer Size**: 10K-100K (larger = more stable, more memory)
- **Batch Size**: 32-128 (larger = more stable, more computation)

## 🚀 Advanced Features

### DQN Variants
1. **Dueling DQN**: Separates value and advantage estimation
2. **Double DQN**: Reduces overestimation bias
3. **Prioritized Replay**: Samples important transitions more frequently
4. **N-Step Returns**: Uses multi-step returns for better learning

### Training Strategies
1. **Curriculum Learning**: Start with easier tasks
2. **Reward Shaping**: Design rewards to guide learning
3. **Experience Replay Variants**: Prioritized, n-step, etc.
4. **Network Regularization**: Dropout, batch normalization

### Evaluation Metrics
1. **Success Rate**: Percentage of successful episodes
2. **Mean Reward**: Average reward per episode
3. **Episode Length**: Average steps per episode
4. **Convergence Speed**: Episodes to reach stable performance
5. **Sample Efficiency**: Episodes needed to learn optimal policy

## 🔧 Troubleshooting

### Common Issues
1. **Unstable Training**: Reduce learning rate, increase target update frequency
2. **Slow Convergence**: Increase exploration, adjust network architecture
3. **Poor Performance**: Check reward function, increase training time
4. **Memory Issues**: Reduce buffer size, use smaller batch size

### Debugging Tips
1. **Monitor Loss**: Should decrease over time
2. **Check Exploration**: Epsilon should decay appropriately
3. **Buffer Analysis**: Ensure diverse experiences
4. **Target Updates**: Verify target network is updating

## 📚 References
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
- [Deep Reinforcement Learning: An Overview](https://arxiv.org/abs/1701.07274)
- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- [Dueling Network Architectures](https://arxiv.org/abs/1511.06581)
- [Double DQN](https://arxiv.org/abs/1509.06461)

## 🚀 Future Extensions
- Rainbow DQN (combining all improvements)
- Distributional DQN (learning value distributions)
- Noisy Networks (parameter space exploration)
- Multi-step learning
- Continuous action spaces (DDPG, TD3)
- Multi-agent DQN
- Transfer learning and domain adaptation