# Day 47 — PPO (Proximal Policy Optimization)

This implementation demonstrates PPO (Proximal Policy Optimization) with clipped objective for stable and efficient policy gradient learning on CartPole-v1.

## 🎯 Goals

- ✅ Reuse A2C with GAE(λ) advantages
- ✅ Add PPO clipped objective with K epochs over shuffled mini-batches
- ✅ Track approximate KL divergence with early stopping
- ✅ Keep entropy bonus, value loss, and gradient clipping
- ✅ Achieve solved CartPole (≥475/500 average reward)

## 📁 Folder Structure

```
47_a2c_ppo/
├── ppo_cartpole.py        # Main PPO training script (discrete)
├── nets.py                # Shared actor-critic network (reused from Day 46)
├── gae.py                 # GAE(λ) computation (reused from Day 46)
├── rollout.py             # PPO rollout buffer (stores logp, actions, values)
├── utils.py               # Utilities (reused from Day 46)
└── README.md              # This file
```

## 🚀 Quick Start

### Basic Training

```python
from ppo_cartpole import train_ppo_cartpole

# Train PPO agent
agent = train_ppo_cartpole(
    num_updates=1000,
    max_rollout_length=2048,
    learning_rate=3e-4,
    gamma=0.99,
    lam=0.95,
    entropy_coef=0.01,
    value_coef=0.5,
    clip_ratio=0.2,
    ppo_epochs=4,
    mini_batch_size=64,
    target_kl=0.01,
    verbose=True
)
```

### Run Training

```bash
python ppo_cartpole.py
```

## 🔧 Key Components

### 1. PPO Agent (`PPOAgent`)

The main PPO agent that implements:
- **Clipped Objective**: Prevents large policy updates
- **Multiple Epochs**: K epochs over shuffled mini-batches
- **KL Divergence Tracking**: Early stopping on high KL
- **Adaptive Learning**: Optional learning rate adjustment

```python
agent = PPOAgent(
    state_size=4,
    action_size=2,
    learning_rate=3e-4,
    clip_ratio=0.2,
    ppo_epochs=4,
    mini_batch_size=64,
    target_kl=0.01
)
```

### 2. PPO Rollout Buffer (`PPORolloutBuffer`)

Specialized buffer for PPO that stores:
- States, actions, rewards, values
- **Old log probabilities** (for ratio computation)
- Advantages and returns (computed via GAE)
- Episode statistics

```python
buffer = PPORolloutBuffer(
    buffer_size=2048,
    state_size=4,
    action_size=2,
    gamma=0.99,
    lam=0.95
)
```

### 3. Clipped Objective

The core PPO innovation - prevents destructive policy updates:

```python
# Compute policy ratio
ratio = torch.exp(current_log_probs - old_log_probs)

# Compute clipped objective
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
policy_loss = -torch.min(surr1, surr2).mean()
```

## 📊 Training Process

### 1. Data Collection
- Collect rollout using current policy
- Store experiences in PPO buffer
- Compute GAE advantages and returns

### 2. PPO Updates
- **K epochs** over shuffled mini-batches
- **Clipped objective** for stable updates
- **KL divergence tracking** for early stopping
- **Gradient clipping** for stability

### 3. Monitoring
- Episode rewards and lengths
- Policy, value, and entropy losses
- KL divergence and clip fractions
- Explained variance of value function

## 🎛️ Hyperparameters

### Core PPO Parameters
- `clip_ratio`: 0.2 (clipping range)
- `ppo_epochs`: 4 (number of epochs per update)
- `mini_batch_size`: 64 (mini-batch size)
- `target_kl`: 0.01 (target KL divergence)

### Training Parameters
- `learning_rate`: 3e-4
- `gamma`: 0.99 (discount factor)
- `lam`: 0.95 (GAE parameter)
- `entropy_coef`: 0.01 (entropy bonus)
- `value_coef`: 0.5 (value loss weight)

### Network Parameters
- `hidden_sizes`: [256, 256]
- `network_type`: 'shared' (or 'split')
- `max_grad_norm`: 0.5

## 📈 Expected Results

### Training Performance
- **Target**: ≥475/500 average reward
- **Convergence**: ~200-500 updates
- **Sample Efficiency**: Better than A2C
- **Stability**: More stable than vanilla policy gradient

### Key Metrics
- **Episode Rewards**: Trending upward to 475+
- **KL Divergence**: Staying below target_kl
- **Clip Fraction**: ~10-30% (indicates clipping is active)
- **Explained Variance**: High (>0.8) indicates good value function

## 🔍 PPO vs A2C Comparison

| Aspect | A2C | PPO |
|--------|-----|-----|
| **Updates** | Single update per rollout | Multiple epochs per rollout |
| **Objective** | Policy gradient | Clipped objective |
| **Stability** | Moderate | High |
| **Sample Efficiency** | Good | Better |
| **KL Control** | None | Explicit tracking |
| **Hyperparameter Sensitivity** | Medium | Low |

## 🛠️ Advanced Features

### 1. KL Divergence Tracking
```python
# Compute KL divergence
kl_div = compute_kl_divergence(old_log_probs, current_log_probs)

# Early stopping if KL is too high
if kl_div > 1.5 * target_kl:
    break
```

### 2. Clip Fraction Monitoring
```python
# Compute fraction of ratios that were clipped
clip_frac = compute_clip_fraction(ratio)
```

### 3. Explained Variance
```python
# Measure how well value function explains returns
exp_var = compute_explained_variance(values, returns)
```

## 📊 Visualization

The training script generates comprehensive plots:

1. **Episode Rewards**: Learning curve with moving average
2. **Episode Lengths**: Episode duration over time
3. **Training Losses**: Policy, value, and entropy losses
4. **KL Divergence**: Policy change tracking
5. **Clip Fraction**: Clipping activity monitoring
6. **Explained Variance**: Value function quality

## 🔧 Troubleshooting

### Common Issues

1. **High KL Divergence**
   - Reduce learning rate
   - Increase target_kl
   - Reduce ppo_epochs

2. **Low Clip Fraction**
   - Increase clip_ratio
   - Reduce learning rate
   - Check advantage normalization

3. **Poor Performance**
   - Increase entropy_coef
   - Adjust network architecture
   - Check hyperparameters

4. **Training Instability**
   - Reduce learning rate
   - Increase max_grad_norm
   - Check gradient clipping

### Debugging Tips

- Monitor KL divergence - should stay below target_kl
- Watch clip fraction - should be 10-30%
- Check explained variance - should be >0.8
- Verify advantage normalization is working

## 🎓 Key Learnings

### PPO Advantages
1. **Stability**: Clipped objective prevents destructive updates
2. **Sample Efficiency**: Multiple epochs improve data utilization
3. **Robustness**: Less sensitive to hyperparameters than A2C
4. **KL Control**: Explicit tracking prevents policy collapse

### Implementation Insights
1. **Buffer Design**: Store old log probabilities for ratio computation
2. **Mini-batching**: Shuffle data for better gradient estimates
3. **Early Stopping**: Prevent over-optimization with KL tracking
4. **Monitoring**: Track multiple metrics for debugging

## 🚀 Next Steps

1. **Experiment with hyperparameters**:
   - Try different clip ratios (0.1, 0.3)
   - Vary PPO epochs (2, 8)
   - Adjust mini-batch sizes

2. **Compare architectures**:
   - Shared vs split networks
   - Different hidden layer sizes
   - Various activation functions

3. **Advanced techniques**:
   - Learning rate scheduling
   - Adaptive clip ratio
   - Value function clipping

4. **Other environments**:
   - LunarLander-v2
   - Atari games
   - Continuous control tasks

## 📚 References

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [OpenAI Spinning Up PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

## 🎉 Success Criteria

- ✅ **Solved CartPole**: ≥475/500 average reward
- ✅ **Clean Logs**: Clear progress tracking
- ✅ **Modular Code**: Reuses Day 46 components
- ✅ **Comprehensive Monitoring**: KL, clip fraction, explained variance
- ✅ **Stable Training**: No catastrophic policy updates

---

**Happy Learning!** 🎯

This implementation demonstrates the power of PPO for stable and efficient reinforcement learning. The clipped objective and multiple epochs make it one of the most robust policy gradient methods available.