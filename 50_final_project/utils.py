import matplotlib.pyplot as plt
import numpy as np
import os
from collections import deque


def plot_training_curves(episode_rewards, world_model_losses, actor_losses, critic_losses):
    """
    Plot training curves for rewards and losses.
    
    Args:
        episode_rewards: list of episode rewards
        world_model_losses: list of world model loss dictionaries
        actor_losses: list of actor losses
        critic_losses: list of critic losses
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    axes[0, 0].plot(episode_rewards)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    
    # World model losses
    if world_model_losses:
        total_losses = [loss['total_loss'] for loss in world_model_losses]
        kl_losses = [loss['kl_loss'] for loss in world_model_losses]
        transition_losses = [loss['transition_loss'] for loss in world_model_losses]
        reward_losses = [loss['reward_loss'] for loss in world_model_losses]
        
        axes[0, 1].plot(total_losses, label='Total')
        axes[0, 1].plot(kl_losses, label='KL')
        axes[0, 1].plot(transition_losses, label='Transition')
        axes[0, 1].plot(reward_losses, label='Reward')
        axes[0, 1].set_title('World Model Losses')
        axes[0, 1].set_xlabel('Update')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Actor losses
    if actor_losses:
        axes[1, 0].plot(actor_losses)
        axes[1, 0].set_title('Actor Losses')
        axes[1, 0].set_xlabel('Update')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
    
    # Critic losses
    if critic_losses:
        axes[1, 1].plot(critic_losses)
        axes[1, 1].set_title('Critic Losses')
        axes[1, 1].set_xlabel('Update')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_reward_curve(episode_rewards, window=100):
    """
    Plot episode rewards with moving average.
    
    Args:
        episode_rewards: list of episode rewards
        window: window size for moving average
    """
    plt.figure(figsize=(12, 6))
    
    # Raw rewards
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.3, color='blue')
    plt.title('Episode Rewards (Raw)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    
    # Moving average
    plt.subplot(1, 2, 2)
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(episode_rewards)), moving_avg, color='red', linewidth=2)
    plt.title(f'Episode Rewards (Moving Average, window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('reward_curve.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_gif(env, agent, filepath, num_episodes=1, max_steps=500):
    """
    Save a GIF of the agent playing the environment.
    
    Args:
        env: gym environment
        agent: trained agent
        filepath: path to save GIF
        num_episodes: number of episodes to record
        max_steps: maximum steps per episode
    """
    try:
        import imageio
    except ImportError:
        print("imageio not available, skipping GIF creation")
        return
    
    frames = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        step = 0
        
        while not done and step < max_steps:
            # Render environment
            frame = env.render(mode='rgb_array')
            frames.append(frame)
            
            # Select action
            action = agent.select_action(state, deterministic=True)
            state, reward, done, _ = env.step(action)
            step += 1
    
    # Save GIF
    if frames:
        imageio.mimsave(filepath, frames, fps=30)
        print(f"GIF saved to {filepath}")


def print_training_info(episode, avg_reward, losses, start_time):
    """
    Print training information.
    
    Args:
        episode: current episode number
        avg_reward: average reward over recent episodes
        losses: dictionary of current losses
        start_time: training start time
    """
    elapsed_time = time.time() - start_time
    
    print(f"Episode {episode}")
    print(f"  Average Reward (100): {avg_reward:.2f}")
    print(f"  Elapsed Time: {elapsed_time:.2f}s")
    
    if losses:
        for key, value in losses.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"    {sub_key}: {sub_value:.4f}")
            else:
                print(f"  {key}: {value:.4f}")
    
    print("-" * 50)


def create_experiment_log(experiment_name, config):
    """
    Create a log file for the experiment.
    
    Args:
        experiment_name: name of the experiment
        config: configuration dictionary
    """
    log_dir = f"experiments/{experiment_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "config.txt")
    
    with open(log_file, 'w') as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write("=" * 50 + "\n\n")
        
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Experiment log created at {log_file}")


def load_experiment_config(experiment_name):
    """
    Load experiment configuration from log file.
    
    Args:
        experiment_name: name of the experiment
        
    Returns:
        config: configuration dictionary
    """
    log_file = f"experiments/{experiment_name}/config.txt"
    
    if not os.path.exists(log_file):
        print(f"Experiment log not found: {log_file}")
        return None
    
    config = {}
    with open(log_file, 'r') as f:
        for line in f:
            if ':' in line and not line.startswith('='):
                key, value = line.strip().split(':', 1)
                config[key.strip()] = value.strip()
    
    return config


def compare_experiments(experiment_names, metric='reward'):
    """
    Compare multiple experiments.
    
    Args:
        experiment_names: list of experiment names
        metric: metric to compare ('reward', 'loss', etc.)
    """
    plt.figure(figsize=(12, 8))
    
    for exp_name in experiment_names:
        # Load experiment data (this would need to be implemented based on your data storage)
        # For now, just plot placeholder data
        pass
    
    plt.title(f'Comparison of Experiments: {metric}')
    plt.xlabel('Episode')
    plt.ylabel(metric.title())
    plt.legend()
    plt.grid(True)
    plt.show()


def setup_logging(experiment_name, log_level='INFO'):
    """
    Setup logging for the experiment.
    
    Args:
        experiment_name: name of the experiment
        log_level: logging level
    """
    import logging
    
    log_dir = f"experiments/{experiment_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "training.log")
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(experiment_name)


