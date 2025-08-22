"""
Run ResNet experiments with different configurations

This script automates running multiple ResNet experiments with different
configurations to compare performance and demonstrate the impact of various techniques.
"""

import os
import sys
import yaml
import subprocess
import argparse
from pathlib import Path
import time


def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def dict_to_args(config_dict):
    """Convert config dictionary to command line arguments"""
    args = []
    
    for key, value in config_dict.items():
        if isinstance(value, bool):
            if value:
                args.append(f'--{key.replace("_", "-")}')
        elif value is not None and value != "":
            args.extend([f'--{key.replace("_", "-")}', str(value)])
    
    return args


def run_experiment(config_path, gpu_id=0, dry_run=False):
    """Run a single experiment with given configuration"""
    config_name = Path(config_path).stem
    print(f"Running experiment: {config_name}")
    print(f"Config file: {config_path}")
    
    # Load configuration
    config = load_config(config_path)
    
    # Convert to command line arguments
    args = dict_to_args(config)
    
    # Add experiment-specific arguments
    experiment_name = f"{config_name}_{int(time.time())}"
    args.extend(['--experiment-name', experiment_name])
    args.extend(['--gpu', str(gpu_id)])
    
    # Build command
    cmd = ['python', 'train_cifar10.py'] + args
    
    print(f"Command: {' '.join(cmd)}")
    
    if dry_run:
        print("Dry run - not executing command")
        return True
    
    # Run experiment
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"Experiment {config_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Experiment {config_name} failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"Experiment {config_name} interrupted by user")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run ResNet experiments')
    parser.add_argument('--configs', nargs='+', help='Configuration files to run')
    parser.add_argument('--config-dir', default='configs', help='Directory containing config files')
    parser.add_argument('--gpu', default=0, type=int, help='GPU to use')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without executing')
    parser.add_argument('--sequential', action='store_true', help='Run experiments sequentially')
    
    args = parser.parse_args()
    
    # Find configuration files
    config_dir = Path(args.config_dir)
    if args.configs:
        config_files = [config_dir / f"{config}.yml" if not config.endswith('.yml') else config_dir / config 
                       for config in args.configs]
    else:
        config_files = list(config_dir.glob('*.yml'))
    
    # Validate config files
    valid_configs = []
    for config_file in config_files:
        if config_file.exists():
            valid_configs.append(config_file)
        else:
            print(f"Warning: Config file not found: {config_file}")
    
    if not valid_configs:
        print("No valid configuration files found")
        return
    
    print(f"Found {len(valid_configs)} configuration files:")
    for config in valid_configs:
        print(f"  - {config}")
    
    print("\nStarting experiments...")
    print("=" * 60)
    
    # Run experiments
    results = {}
    for i, config_file in enumerate(valid_configs):
        print(f"\n[{i+1}/{len(valid_configs)}] Running {config_file.name}")
        print("-" * 40)
        
        success = run_experiment(config_file, gpu_id=args.gpu, dry_run=args.dry_run)
        results[config_file.name] = success
        
        if not success and not args.dry_run:
            response = input("Experiment failed. Continue with next? (y/n): ")
            if response.lower() != 'y':
                break
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    
    for config_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{config_name:30} {status}")


def create_sample_configs():
    """Create sample configuration files for demonstration"""
    configs_dir = Path('configs')
    configs_dir.mkdir(exist_ok=True)
    
    # ResNet-34 configuration
    resnet34_config = {
        'arch': 'resnet34',
        'pre_activation': True,
        'epochs': 250,
        'batch_size': 128,
        'lr': 0.1,
        'weight_decay': 0.0005,
        'momentum': 0.9,
        'scheduler': 'warmup_cosine',
        'warmup_epochs': 8,
        'min_lr': 0.00001,
        'mixup_alpha': 0.2,
        'mixup_prob': 1.0,
        'cutmix': True,
        'label_smoothing': 0.1,
        'grad_clip': 0.0,
        'randaugment': True,
        'num_workers': 4,
        'print_freq': 50,
        'save_freq': 50
    }
    
    with open(configs_dir / 'resnet34_cifar.yml', 'w') as f:
        yaml.dump(resnet34_config, f, default_flow_style=False)
    
    # ResNet-101 configuration  
    resnet101_config = {
        'arch': 'resnet101',
        'pre_activation': True,
        'epochs': 400,
        'batch_size': 64,  # Smaller batch size for memory
        'lr': 0.05,  # Lower learning rate for stability
        'weight_decay': 0.001,
        'momentum': 0.9,
        'scheduler': 'warmup_cosine',
        'warmup_epochs': 15,
        'min_lr': 0.000001,
        'mixup_alpha': 0.3,
        'mixup_prob': 1.0,
        'cutmix': True,
        'label_smoothing': 0.15,
        'grad_clip': 2.0,  # More aggressive clipping for very deep network
        'randaugment': True,
        'num_workers': 4,
        'print_freq': 100,
        'save_freq': 100
    }
    
    with open(configs_dir / 'resnet101_cifar.yml', 'w') as f:
        yaml.dump(resnet101_config, f, default_flow_style=False)
    
    print("Created additional sample configurations:")
    print("  - resnet34_cifar.yml")
    print("  - resnet101_cifar.yml")


if __name__ == '__main__':
    # Create sample configs if they don't exist
    if not Path('configs').exists() or len(list(Path('configs').glob('*.yml'))) < 3:
        print("Creating sample configuration files...")
        create_sample_configs()
        print()
    
    main()