# Day 48 — Multi-Agent RL (MARL)

This implementation demonstrates Multi-Agent Reinforcement Learning using IPPO and MAPPO-lite on PettingZoo MPE environments, specifically `simple_spread_v3`.

## 🎯 Goals

- ✅ Understand Markov games and credit assignment issues
- ✅ Implement IPPO (shared actor-critic per agent) on PettingZoo MPE simple_spread_v3
- ✅ Switch to MAPPO-lite (centralized critic with global info)
- ✅ Log per-agent returns and success metrics
- ✅ Demonstrate CTDE (Centralized Training, Decentralized Execution)

## 📁 Folder Structure

```
48_marl/
├── envs/
│   └── mpe_ippo_env.py          # PettingZoo MPE wrappers (vectorized rollout)
├── algs/
│   ├── nets.py                  # ActorCritic for discrete actions (shared)
│   ├── gae.py                   # GAE(λ) per-agent
│   ├── ippo.py                  # IPPO trainer (shared policy, per-agent rollouts)
│   └── mappo_critic.py          # Centralized critic (MAPPO-lite, optional)
├── train_simple_spread.py       # Run IPPO/MAPPO-lite on simple_spread_v3
├── utils.py                     # Seeding, logging, minibatches
├── README.md
└── requirements.txt
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Training

```bash
python train_simple_spread.py
```

Choose between IPPO or MAPPO when prompted.

### Programmatic Usage

```python
from train_simple_spread import train_ippo_simple_spread, train_mappo_simple_spread

# Train IPPO
ippo_trainer = train_ippo_simple_spread(
    num_updates=1000,
    learning_rate=3e-4,
    verbose=True
)

# Train MAPPO
mappo_trainer = train_mappo_simple_spread(
    num_updates=1000,
    learning_rate=3e-4,
    verbose=True
)
```

## 🧠 Key Concepts

### Markov Game

A Markov Game is defined as:
- **N**: Number of agents
- **S**: State space
- **{A_i}**: Action spaces for each agent i
- **{R_i}**: Reward functions for each agent i
- **P**: Transition probability function
- **γ**: Discount factor

### Decentralized Policies

Each agent i has its own policy:
- **π_θ_i(a_i | o_i)**: Policy for agent i
- **o_i**: Local observation for agent i
- **a_i**: Action for agent i

### CTDE (Centralized Training, Decentralized Execution)

- **Training**: Critics can use global state/all observations
- **Execution**: Policies only use local observations
- **Benefit**: Better coordination during training, practical deployment

## 🔧 Key Components

### 1. Environment Wrapper (`envs/mpe_ippo_env.py`)

Handles PettingZoo MPE environments with:
- **Multi-agent support**: 3 agents in simple_spread_v3
- **Vectorized rollouts**: Parallel environment collection
- **Global state**: Concatenated observations for MAPPO
- **Episode tracking**: Per-agent statistics

```python
env = create_mpe_env("simple_spread_v3", num_agents=3, max_cycles=25)
observations = env.reset()
actions, rewards, dones, truncateds, infos = env.step(action_dict)
```

### 2. Actor-Critic Networks (`algs/nets.py`)

#### IPPO Network
- **Shared architecture**: All agents use same network structure
- **Independent parameters**: Each agent has its own parameters
- **Local observations**: Each agent sees only its own observation

#### MAPPO Network
- **Individual actors**: Each agent has its own policy network
- **Centralized critic**: Uses global state information
- **CTDE**: Centralized training, decentralized execution

```python
# IPPO
ippo_net = create_actor_critic_network(
    'ippo', num_agents=3, obs_dim=18, action_dim=5,
    shared_parameters=True
)

# MAPPO
mappo_net = create_actor_critic_network(
    'mappo', num_agents=3, obs_dim=18, action_dim=5,
    global_obs_dim=54
)
```

### 3. IPPO Trainer (`algs/ippo.py`)

Independent PPO for each agent:
- **Independent learning**: Each agent learns separately
- **Shared parameters**: All agents use same network architecture
- **Per-agent updates**: Individual PPO updates for each agent

```python
trainer = IPPOTrainer(
    num_agents=3,
    obs_dim=18,
    action_dim=5,
    learning_rate=3e-4,
    shared_parameters=True
)
```

### 4. MAPPO Trainer (`algs/mappo_critic.py`)

Centralized critic with decentralized actors:
- **Centralized critic**: Uses global state information
- **Decentralized actors**: Each agent has its own policy
- **CTDE**: Better coordination during training

```python
trainer = MAPPOTrainer(
    num_agents=3,
    obs_dim=18,
    action_dim=5,
    global_obs_dim=54,
    learning_rate=3e-4
)
```

## 📊 Training Process

### 1. Data Collection
- Collect rollout using current policies
- Store experiences for each agent
- Compute GAE advantages and returns
- Prepare global state for MAPPO

### 2. IPPO Updates
- **Independent updates**: Each agent updated separately
- **Shared parameters**: All agents use same network
- **Per-agent statistics**: Individual metrics for each agent

### 3. MAPPO Updates
- **Centralized critic**: Uses global state information
- **Decentralized actors**: Individual policy networks
- **Coordinated training**: Better cooperation learning

### 4. Monitoring
- **Per-agent rewards**: Individual agent performance
- **Overall performance**: Team coordination metrics
- **Training statistics**: Losses, KL divergence, clip fractions

## 🎛️ Hyperparameters

### Core MARL Parameters
- `num_agents`: 3 (number of agents)
- `obs_dim`: 18 (observation dimension)
- `action_dim`: 5 (action dimension)
- `global_obs_dim`: 54 (3 × 18 for MAPPO)

### Training Parameters
- `learning_rate`: 3e-4
- `gamma`: 0.99 (discount factor)
- `lam`: 0.95 (GAE parameter)
- `entropy_coef`: 0.01 (entropy bonus)
- `value_coef`: 0.5 (value loss weight)

### PPO Parameters
- `clip_ratio`: 0.2 (clipping range)
- `ppo_epochs`: 4 (number of epochs per update)
- `mini_batch_size`: 64 (mini-batch size)
- `target_kl`: 0.01 (target KL divergence)

## 📈 Expected Results

### Training Performance
- **Coordination**: Agents learn to spread out and cover landmarks
- **Convergence**: ~500-1000 updates for good performance
- **Sample Efficiency**: MAPPO typically more efficient than IPPO
- **Stability**: PPO ensures stable multi-agent learning

### Key Metrics
- **Per-agent rewards**: Individual agent performance
- **Overall coordination**: Team success metrics
- **KL divergence**: Policy change tracking
- **Clip fraction**: Clipping activity monitoring

## 🔍 IPPO vs MAPPO Comparison

| Aspect | IPPO | MAPPO |
|--------|------|-------|
| **Learning** | Independent | Centralized |
| **Information** | Local observations | Global state (critic) |
| **Coordination** | Limited | Better |
| **Sample Efficiency** | Lower | Higher |
| **Implementation** | Simpler | More complex |
| **Scalability** | Good | Better |

## 🛠️ Advanced Features

### 1. Per-Agent Logging
```python
# Track individual agent performance
for agent_name in env.agent_names:
    print(f"{agent_name} Mean Reward: {agent_mean_reward:.2f}")
```

### 2. Global State for MAPPO
```python
# Concatenate all agent observations
global_state = env.get_global_state(observations)
centralized_value = mappo_trainer.get_centralized_values(global_state)
```

### 3. Vectorized Rollouts
```python
# Parallel environment collection
vectorized_env = create_mpe_env(
    "simple_spread_v3", 
    vectorized=True, 
    num_envs=4
)
```

## 📊 Visualization

The training script generates comprehensive plots:

1. **Per-Agent Rewards**: Individual learning curves
2. **Overall Performance**: Team coordination metrics
3. **Training Losses**: Policy, value, and entropy losses
4. **KL Divergence**: Policy change tracking
5. **Clip Fraction**: Clipping activity monitoring
6. **Explained Variance**: Value function quality

## 🔧 Troubleshooting

### Common Issues

1. **Poor Coordination**
   - Try MAPPO instead of IPPO
   - Increase entropy coefficient
   - Adjust reward shaping

2. **Training Instability**
   - Reduce learning rate
   - Increase gradient clipping
   - Check advantage normalization

3. **Slow Convergence**
   - Increase PPO epochs
   - Adjust mini-batch size
   - Try different network architecture

4. **Environment Issues**
   - Check PettingZoo installation
   - Verify environment compatibility
   - Check observation/action spaces

### Debugging Tips

- Monitor per-agent rewards separately
- Check global state construction for MAPPO
- Verify advantage normalization
- Watch KL divergence trends

## 🎓 Key Learnings

### Multi-Agent Challenges
1. **Credit Assignment**: Who gets credit for team success?
2. **Non-stationarity**: Other agents change during training
3. **Coordination**: Agents must learn to work together
4. **Scalability**: Performance with more agents

### Algorithm Insights
1. **IPPO**: Simple baseline, independent learning
2. **MAPPO**: Better coordination, centralized training
3. **CTDE**: Best of both worlds
4. **GAE**: Low-variance advantage estimation

### Implementation Insights
1. **Environment Wrapping**: Handle multi-agent interfaces
2. **Data Collection**: Per-agent rollout management
3. **Training Loops**: Independent vs centralized updates
4. **Monitoring**: Per-agent and overall metrics

## 🚀 Extensions

### 1. Other MPE Tasks
```python
# Try different environments
env = create_mpe_env("simple_tag_v3", num_agents=4)
env = create_mpe_env("simple_adversary_v3", num_agents=3)
```

### 2. Curriculum Learning
```python
# Vary number of landmarks/agents
for num_landmarks in [3, 5, 7]:
    env = create_mpe_env("simple_spread_v3", num_agents=num_landmarks)
```

### 3. Advanced Algorithms
- **VDN/QMIX**: Value decomposition methods
- **MADDPG**: Multi-agent DDPG
- **FACMAC**: Multi-agent actor-critic

### 4. Communication
- **Message passing**: Agents can communicate
- **Attention mechanisms**: Focus on relevant agents
- **Graph neural networks**: Structured communication

## 📚 References

- [PettingZoo: Multi-Agent Gym API](https://pettingzoo.farama.org/)
- [MAPPO: Yu et al., 2021](https://arxiv.org/abs/2103.01955)
- [PPO: Schulman et al., 2017](https://arxiv.org/abs/1707.06347)
- [Multi-Agent RL Survey](https://arxiv.org/abs/1911.10635)

## 🎉 Success Criteria

- ✅ **IPPO training script solving/learning on simple_spread_v3**
- ✅ **Clear logs (per-agent reward trend)**
- ✅ **Optional: MAPPO-lite centralized critic for CTDE comparison**
- ✅ **Per-agent logging and success metrics**
- ✅ **Comprehensive documentation and examples**

---

**Happy Learning!** 🎯

This implementation demonstrates the power of Multi-Agent RL for coordination and cooperation. The combination of IPPO and MAPPO provides a strong foundation for understanding both independent learning and centralized training approaches in multi-agent environments.