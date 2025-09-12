# Day 49 — Model-Based Reinforcement Learning (MBRL)

This implementation demonstrates Model-Based Reinforcement Learning using learned dynamics models and Model Predictive Control (MPC) on CartPole-v1.

## 🎯 Goals

- ✅ Understand model-based RL: separate environment model + policy/plan
- ✅ Implement a probabilistic dynamics model (NN predicting next state + reward)
- ✅ Use Model Predictive Control (MPC) with random shooting / CEM to select actions
- ✅ Demonstrate on CartPole-v1 (low-dim, easy to learn dynamics)
- ✅ Compare model-based vs model-free sample efficiency

## 🧠 Theory

### Decomposition

An MDP with transitions P(s'|s,a) is replaced by a learned model P̂_φ(s'|s,a), R̂_φ(s,a).

### Dynamics Model

(s_{t+1}, r_t) ~ f̂_φ(s_t, a_t)

Usually parameterized by neural nets (deterministic or probabilistic).

Train with supervised learning on transition tuples (s, a, s', r).

### Planning (MPC)

At each step, sample action sequences for horizon H.

Roll out sequences in the learned model.

Pick first action of best sequence.

## 📁 Folder Structure

```
49_model_based_rl/
├── dynamics_model.py        # NN dynamics model
├── buffer.py                # Replay buffer for transitions
├── mpc_controller.py        # Random shooting / CEM planner
├── train_cartpole.py        # Collect data, train model, run MPC
├── utils.py                 # Logging, plotting
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
pip install torch numpy matplotlib seaborn gymnasium
```

### Basic Training

```bash
python train_cartpole.py
```

Choose between 'train' or 'compare' when prompted.

### Programmatic Usage

```python
from train_cartpole import train_model_based_rl

# Train model-based RL agent
agent = train_model_based_rl(
    env_name='CartPole-v1',
    num_episodes=500,
    model_type='probabilistic',
    mpc_type='random_shooting',
    horizon=10,
    num_samples=1000
)
```

## 🔧 Key Components

### 1. Dynamics Model (`dynamics_model.py`)

#### Probabilistic Dynamics Model
- **Neural Network**: Predicts next states and rewards
- **Probabilistic**: Outputs mean and log_std for uncertainty
- **Training**: Supervised learning on transition tuples

```python
model = ProbabilisticDynamicsModel(
    state_dim=4,
    action_dim=1,
    hidden_sizes=[256, 256],
    probabilistic=True
)
```

#### Ensemble Dynamics Model
- **Multiple Models**: Ensemble for better uncertainty estimation
- **Robust Predictions**: Average across ensemble members
- **Uncertainty**: Variance across ensemble predictions

```python
ensemble = EnsembleDynamicsModel(
    state_dim=4,
    action_dim=1,
    num_models=5
)
```

### 2. Replay Buffer (`buffer.py`)

#### Standard Replay Buffer
- **Transition Storage**: (state, action, next_state, reward, done)
- **Random Sampling**: Uniform sampling for training
- **Sequential Sampling**: For sequence-based training

```python
buffer = ReplayBuffer(capacity=100000, device='cpu')
buffer.add(state, action, next_state, reward, done)
states, actions, next_states, rewards, dones = buffer.sample(256)
```

#### Prioritized Replay Buffer
- **Priority-based Sampling**: Sample important transitions more often
- **Importance Sampling**: Correct for sampling bias
- **Dynamic Priorities**: Update priorities based on prediction errors

### 3. MPC Controller (`mpc_controller.py`)

#### Random Shooting MPC
- **Random Sampling**: Sample random action sequences
- **Evaluation**: Roll out sequences in learned model
- **Selection**: Choose best sequence

```python
mpc = RandomShootingMPC(
    dynamics_model=model,
    horizon=10,
    num_samples=1000
)
action = mpc.get_action(state)
```

#### CEM MPC
- **Iterative Improvement**: Update distribution based on elite samples
- **Cross-Entropy Method**: Optimize action distribution
- **Better Exploration**: More focused search over time

```python
mpc = CEM_MPC(
    dynamics_model=model,
    horizon=10,
    num_samples=1000,
    num_elite=100,
    num_iterations=5
)
```

#### MPPI MPC
- **Path Integral**: Weight action sequences by returns
- **Temperature Control**: Balance exploration vs exploitation
- **Smooth Updates**: Gradual improvement of action sequence

### 4. Training Script (`train_cartpole.py`)

#### Model-Based RL Agent
- **Dynamics Learning**: Train model on collected data
- **MPC Planning**: Use learned model for action selection
- **Data Collection**: Gather experience for model training

```python
agent = ModelBasedRLAgent(
    state_dim=4,
    action_dim=1,
    model_type='probabilistic',
    mpc_type='random_shooting'
)
```

## 📊 Training Process

### 1. Data Collection
- Collect random data for initial model training
- Store transitions in replay buffer
- Ensure sufficient data for model learning

### 2. Model Training
- Train dynamics model on collected data
- Use supervised learning (MSE for deterministic, NLL for probabilistic)
- Regularize with gradient clipping

### 3. MPC Planning
- Sample action sequences for planning horizon
- Roll out sequences in learned model
- Select first action of best sequence

### 4. Online Learning
- Collect new data using MPC controller
- Retrain model periodically
- Improve model accuracy over time

## 🎛️ Hyperparameters

### Dynamics Model
- `state_dim`: 4 (CartPole state dimension)
- `action_dim`: 1 (continuous action for planning)
- `hidden_sizes`: [256, 256] (network architecture)
- `probabilistic`: True (uncertainty estimation)

### MPC Controller
- `horizon`: 10 (planning horizon)
- `num_samples`: 1000 (action sequences to sample)
- `action_bounds`: (-1.0, 1.0) (action limits)

### Training
- `learning_rate`: 3e-4 (model learning rate)
- `batch_size`: 256 (training batch size)
- `buffer_capacity`: 100000 (replay buffer size)

## 📈 Expected Results

### Training Performance
- **Model Accuracy**: Low prediction error on held-out data
- **MPC Performance**: Increasing returns with better models
- **Sample Efficiency**: Better than random policy

### Key Metrics
- **Model Loss**: State and reward prediction errors
- **Episode Rewards**: Performance on environment
- **Planning Time**: Time for MPC computation
- **Model Uncertainty**: Confidence in predictions

## 🔍 Model-Based vs Model-Free Comparison

| Aspect | Model-Based RL | Model-Free RL |
|--------|----------------|---------------|
| **Sample Efficiency** | High | Low |
| **Planning** | Explicit | Implicit |
| **Uncertainty** | Explicit | Implicit |
| **Interpretability** | High | Low |
| **Computational Cost** | High | Low |
| **Model Accuracy** | Critical | Not needed |

## 🛠️ Advanced Features

### 1. Probabilistic Models
```python
# Uncertainty estimation
outputs = model.forward(states, actions)
state_mean = outputs['state_mean']
state_std = torch.exp(outputs['state_log_std'])
```

### 2. Ensemble Methods
```python
# Multiple models for robustness
ensemble = EnsembleDynamicsModel(num_models=5)
predictions = ensemble.predict(states, actions)
```

### 3. Different MPC Algorithms
```python
# CEM for better optimization
mpc = CEM_MPC(
    num_iterations=5,
    num_elite=100
)

# MPPI for smooth updates
mpc = MPPIMPC(
    temperature=1.0
)
```

### 4. Model Evaluation
```python
# Evaluate model accuracy
eval_results = agent.evaluate_model(states, actions, next_states, rewards)
print(f"State MSE: {eval_results['state_mse']:.4f}")
print(f"Reward MSE: {eval_results['reward_mse']:.4f}")
```

## 📊 Visualization

The training script generates comprehensive plots:

1. **Training Curves**: Episode rewards, lengths, model losses
2. **Model Predictions**: Predicted vs actual states and rewards
3. **MPC Performance**: Planning returns over time
4. **Error Distributions**: Prediction error analysis

## 🔧 Troubleshooting

### Common Issues

1. **Poor Model Accuracy**
   - Increase model capacity
   - Collect more data
   - Adjust learning rate

2. **MPC Performance Issues**
   - Increase planning horizon
   - Sample more action sequences
   - Improve model accuracy

3. **Training Instability**
   - Reduce learning rate
   - Increase batch size
   - Add gradient clipping

4. **Memory Issues**
   - Reduce buffer capacity
   - Decrease batch size
   - Use smaller models

### Debugging Tips

- Monitor model prediction accuracy
- Check MPC planning statistics
- Visualize model predictions
- Compare with random policy

## 🎓 Key Learnings

### Model-Based RL Advantages
1. **Sample Efficiency**: Learn from fewer interactions
2. **Planning**: Explicit reasoning about future
3. **Uncertainty**: Know what you don't know
4. **Interpretability**: Understand learned dynamics

### Challenges
1. **Model Bias**: Learned model may be inaccurate
2. **Computational Cost**: Planning is expensive
3. **Model Complexity**: Hard to learn complex dynamics
4. **Distribution Shift**: Model may not generalize

### Best Practices
1. **Start Simple**: Use deterministic models first
2. **Validate Models**: Check prediction accuracy
3. **Tune Planning**: Adjust horizon and samples
4. **Monitor Performance**: Track key metrics

## 🚀 Extensions

### 1. Different Environments
```python
# Try other environments
agent = train_model_based_rl(env_name='MountainCar-v0')
agent = train_model_based_rl(env_name='Pendulum-v1')
```

### 2. Advanced Models
- **Recurrent Models**: For sequential dependencies
- **Graph Neural Networks**: For structured environments
- **Transformer Models**: For long sequences

### 3. Better Planning
- **Value Functions**: Combine with value-based methods
- **Policy Networks**: Learn action distributions
- **Hierarchical Planning**: Multi-level planning

### 4. Uncertainty Quantification
- **Bayesian Neural Networks**: True uncertainty
- **Ensemble Methods**: Multiple model predictions
- **Calibration**: Ensure uncertainty is well-calibrated

## 📚 References

- [Model-Based RL Survey](https://arxiv.org/abs/1906.05253)
- [MPC for RL](https://arxiv.org/abs/1802.10592)
- [CEM for RL](https://arxiv.org/abs/1906.02425)
- [MPPI for RL](https://arxiv.org/abs/1707.02087)

## 🎉 Success Criteria

- ✅ **Dynamics model trained supervised from replay buffer**
- ✅ **MPC controller with random shooting**
- ✅ **CartPole evaluation showing increasing returns with more data**
- ✅ **README with explanation and extensions**

---

**Happy Learning!** 🎯

This implementation demonstrates the power of Model-Based RL for sample-efficient learning. The combination of learned dynamics models and MPC provides a strong foundation for understanding how to leverage environment models for better decision-making.