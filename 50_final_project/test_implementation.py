#!/usr/bin/env python3
"""
Test script to verify the Dreamer-Lite implementation works correctly.
"""

import torch
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from models.encoder import Encoder
        from models.transition import TransitionModel
        from models.reward_predictor import RewardPredictor
        from models.actor_critic import Actor, Critic
        from buffer import ReplayBuffer, ImaginaryBuffer
        from world_model import WorldModel
        from train_dreamer_lite import DreamerLite
        from utils import plot_training_curves
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_encoder():
    """Test encoder model."""
    print("\nTesting encoder...")
    
    try:
        from models.encoder import Encoder
        
        # Test with CartPole dimensions
        encoder = Encoder(state_dim=4, latent_dim=32)
        
        # Test forward pass
        state = torch.randn(32, 4)
        latent, mean, log_std = encoder(state)
        
        assert latent.shape == (32, 32), f"Expected (32, 32), got {latent.shape}"
        assert mean.shape == (32, 32), f"Expected (32, 32), got {mean.shape}"
        assert log_std.shape == (32, 32), f"Expected (32, 32), got {log_std.shape}"
        
        # Test deterministic encoding
        mean_det = encoder.encode_deterministic(state)
        assert mean_det.shape == (32, 32), f"Expected (32, 32), got {mean_det.shape}"
        
        print("✓ Encoder test passed")
        return True
    except Exception as e:
        print(f"✗ Encoder test failed: {e}")
        return False


def test_transition():
    """Test transition model."""
    print("\nTesting transition model...")
    
    try:
        from models.transition import TransitionModel
        
        # Test with CartPole dimensions
        transition = TransitionModel(latent_dim=32, action_dim=1)
        
        # Test forward pass
        latent = torch.randn(32, 32)
        action = torch.randn(32, 1)
        next_latent, mean, log_std = transition(latent, action)
        
        assert next_latent.shape == (32, 32), f"Expected (32, 32), got {next_latent.shape}"
        assert mean.shape == (32, 32), f"Expected (32, 32), got {mean.shape}"
        assert log_std.shape == (32, 32), f"Expected (32, 32), got {log_std.shape}"
        
        # Test deterministic prediction
        mean_det = transition.predict_deterministic(latent, action)
        assert mean_det.shape == (32, 32), f"Expected (32, 32), got {mean_det.shape}"
        
        print("✓ Transition model test passed")
        return True
    except Exception as e:
        print(f"✗ Transition model test failed: {e}")
        return False


def test_reward_predictor():
    """Test reward predictor."""
    print("\nTesting reward predictor...")
    
    try:
        from models.reward_predictor import RewardPredictor
        
        # Test with CartPole dimensions
        reward_predictor = RewardPredictor(latent_dim=32, action_dim=1)
        
        # Test forward pass
        latent = torch.randn(32, 32)
        action = torch.randn(32, 1)
        reward = reward_predictor(latent, action)
        
        assert reward.shape == (32, 1), f"Expected (32, 1), got {reward.shape}"
        
        print("✓ Reward predictor test passed")
        return True
    except Exception as e:
        print(f"✗ Reward predictor test failed: {e}")
        return False


def test_actor_critic():
    """Test actor and critic models."""
    print("\nTesting actor-critic...")
    
    try:
        from models.actor_critic import Actor, Critic
        
        # Test actor
        actor = Actor(latent_dim=32, action_dim=1)
        latent = torch.randn(32, 32)
        action, mean, log_std = actor(latent)
        
        assert action.shape == (32, 1), f"Expected (32, 1), got {action.shape}"
        assert mean.shape == (32, 1), f"Expected (32, 1), got {mean.shape}"
        assert log_std.shape == (32, 1), f"Expected (32, 1), got {log_std.shape}"
        
        # Test deterministic action
        action_det = actor.get_action_deterministic(latent)
        assert action_det.shape == (32, 1), f"Expected (32, 1), got {action_det.shape}"
        
        # Test log probability
        log_prob = actor.log_prob(latent, action)
        assert log_prob.shape == (32,), f"Expected (32,), got {log_prob.shape}"
        
        # Test critic
        critic = Critic(latent_dim=32)
        value = critic(latent)
        assert value.shape == (32, 1), f"Expected (32, 1), got {value.shape}"
        
        print("✓ Actor-critic test passed")
        return True
    except Exception as e:
        print(f"✗ Actor-critic test failed: {e}")
        return False


def test_buffers():
    """Test replay buffers."""
    print("\nTesting buffers...")
    
    try:
        from buffer import ReplayBuffer, ImaginaryBuffer
        
        # Test replay buffer
        replay_buffer = ReplayBuffer(capacity=1000, state_dim=4, action_dim=1)
        
        # Add some transitions
        for i in range(10):
            state = np.random.randn(4)
            action = np.random.randn(1)
            reward = np.random.randn(1)
            next_state = np.random.randn(4)
            done = np.random.choice([True, False])
            replay_buffer.add(state, action, reward, next_state, done)
        
        assert len(replay_buffer) == 10, f"Expected 10, got {len(replay_buffer)}"
        
        # Test sampling
        batch = replay_buffer.sample(5)
        assert batch['states'].shape == (5, 4), f"Expected (5, 4), got {batch['states'].shape}"
        assert batch['actions'].shape == (5, 1), f"Expected (5, 1), got {batch['actions'].shape}"
        assert batch['rewards'].shape == (5, 1), f"Expected (5, 1), got {batch['rewards'].shape}"
        assert batch['next_states'].shape == (5, 4), f"Expected (5, 4), got {batch['next_states'].shape}"
        assert batch['dones'].shape == (5, 1), f"Expected (5, 1), got {batch['dones'].shape}"
        
        # Test imaginary buffer
        imaginary_buffer = ImaginaryBuffer(capacity=1000, latent_dim=32, action_dim=1)
        
        # Add some imaginary transitions
        for i in range(10):
            latent = torch.randn(32)
            action = torch.randn(1)
            reward = torch.randn(1)
            next_latent = torch.randn(32)
            done = torch.tensor(np.random.choice([True, False]))
            imaginary_buffer.add(latent, action, reward, next_latent, done)
        
        assert len(imaginary_buffer) == 10, f"Expected 10, got {len(imaginary_buffer)}"
        
        # Test sampling
        batch = imaginary_buffer.sample(5)
        assert batch['latents'].shape == (5, 32), f"Expected (5, 32), got {batch['latents'].shape}"
        assert batch['actions'].shape == (5, 1), f"Expected (5, 1), got {batch['actions'].shape}"
        assert batch['rewards'].shape == (5, 1), f"Expected (5, 1), got {batch['rewards'].shape}"
        assert batch['next_latents'].shape == (5, 32), f"Expected (5, 32), got {batch['next_latents'].shape}"
        assert batch['dones'].shape == (5, 1), f"Expected (5, 1), got {batch['dones'].shape}"
        
        print("✓ Buffers test passed")
        return True
    except Exception as e:
        print(f"✗ Buffers test failed: {e}")
        return False


def test_world_model():
    """Test world model."""
    print("\nTesting world model...")
    
    try:
        from world_model import WorldModel
        
        # Create world model
        world_model = WorldModel(state_dim=4, action_dim=1, latent_dim=32)
        
        # Test forward pass
        states = torch.randn(32, 4)
        actions = torch.randn(32, 1)
        next_latent, reward, latent = world_model(states, actions)
        
        assert next_latent.shape == (32, 32), f"Expected (32, 32), got {next_latent.shape}"
        assert reward.shape == (32, 1), f"Expected (32, 1), got {reward.shape}"
        assert latent.shape == (32, 32), f"Expected (32, 32), got {latent.shape}"
        
        # Test individual components
        latent = world_model.encode(states)
        assert latent.shape == (32, 32), f"Expected (32, 32), got {latent.shape}"
        
        next_latent = world_model.predict_next_latent(latent, actions)
        assert next_latent.shape == (32, 32), f"Expected (32, 32), got {next_latent.shape}"
        
        reward = world_model.predict_reward(latent, actions)
        assert reward.shape == (32, 1), f"Expected (32, 1), got {reward.shape}"
        
        print("✓ World model test passed")
        return True
    except Exception as e:
        print(f"✗ World model test failed: {e}")
        return False


def test_dreamer_lite():
    """Test DreamerLite class."""
    print("\nTesting DreamerLite...")
    
    try:
        from train_dreamer_lite import DreamerLite
        
        # Create agent
        agent = DreamerLite(env_name='CartPole-v1', latent_dim=32, hidden_dim=128)
        
        # Test action selection
        state = np.random.randn(4)
        action = agent.select_action(state)
        assert isinstance(action, (int, np.integer)), f"Expected int, got {type(action)}"
        
        # Test deterministic action selection
        action_det = agent.select_action(state, deterministic=True)
        assert isinstance(action_det, (int, np.integer)), f"Expected int, got {type(action_det)}"
        
        print("✓ DreamerLite test passed")
        return True
    except Exception as e:
        print(f"✗ DreamerLite test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Dreamer-Lite Implementation Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_encoder,
        test_transition,
        test_reward_predictor,
        test_actor_critic,
        test_buffers,
        test_world_model,
        test_dreamer_lite
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Implementation is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


