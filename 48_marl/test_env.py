#!/usr/bin/env python3
"""
Simple test script for the environment
"""

from envs.mpe_ippo_env import create_mpe_env

# Test environment creation
print("Testing environment creation...")
env = create_mpe_env('simple_spread_v3', num_agents=3)
print("Environment created successfully!")

# Test reset
print("Testing reset...")
obs = env.reset()
print("Reset successful!")
print("Observation type:", type(obs))
print("Agent names:", env.agent_names)

# Test step
print("Testing step...")
actions = [0, 1, 2]  # Simple actions
action_dict = env.get_agent_actions(actions)
print("Action dict:", action_dict)

obs, rewards, dones, truncateds, infos = env.step(action_dict)
print("Step successful!")
print("Rewards:", rewards)
print("Dones:", dones)

env.close()
print("Test completed successfully!")
