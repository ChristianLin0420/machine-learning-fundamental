"""
Weight Initialization Utilities

Implements various weight initialization strategies for ResNet training,
with focus on He (Kaiming) initialization that works well with ReLU activations.
"""

import torch
import torch.nn as nn
import math


def he_normal_init(m):
    """
    He/Kaiming normal initialization for conv layers
    
    Designed for ReLU activations. Initializes weights from normal distribution
    with std = sqrt(2/fan_in) where fan_in is number of input connections.
    
    Reference: He et al. (2015) "Delving Deep into Rectifiers"
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def he_uniform_init(m):
    """He/Kaiming uniform initialization"""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def xavier_normal_init(m):
    """Xavier/Glorot normal initialization"""
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def xavier_uniform_init(m):
    """Xavier/Glorot uniform initialization"""
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def bn_init(m):
    """
    Batch normalization initialization
    
    Initialize BN weights to 1 and biases to 0.
    For ResNet, we often initialize the last BN in each residual branch to 0
    for better training stability.
    """
    if isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def zero_gamma_init(model):
    """
    Zero-initialize the last BN in each residual branch
    
    This technique from "Bag of Tricks for Image Classification" (2019)
    helps with training stability by making residual branches start as identity.
    """
    for m in model.modules():
        # For BasicBlock, zero-init the last BN (bn2)
        if hasattr(m, 'bn2') and isinstance(m.bn2, nn.BatchNorm2d):
            nn.init.constant_(m.bn2.weight, 0)
        # For Bottleneck, zero-init the last BN (bn3)  
        elif hasattr(m, 'bn3') and isinstance(m.bn3, nn.BatchNorm2d):
            nn.init.constant_(m.bn3.weight, 0)


def init_weights(model, init_type='he_normal', zero_gamma=True):
    """
    Initialize model weights
    
    Args:
        model: PyTorch model
        init_type: Initialization strategy ('he_normal', 'he_uniform', 'xavier_normal', 'xavier_uniform')
        zero_gamma: Whether to zero-initialize last BN in residual branches
    """
    init_functions = {
        'he_normal': he_normal_init,
        'he_uniform': he_uniform_init,
        'xavier_normal': xavier_normal_init,
        'xavier_uniform': xavier_uniform_init
    }
    
    if init_type not in init_functions:
        raise ValueError(f"Unknown init_type: {init_type}")
    
    # Apply weight initialization
    model.apply(init_functions[init_type])
    
    # Initialize batch normalization
    model.apply(bn_init)
    
    # Zero-gamma initialization for residual branches
    if zero_gamma:
        zero_gamma_init(model)
    
    print(f"Initialized model with {init_type}" + (" + zero_gamma" if zero_gamma else ""))


def lecun_normal_init(m):
    """LeCun normal initialization (for SELU activations)"""
    if isinstance(m, nn.Conv2d):
        fan_in = m.weight.size(1) * m.weight.size(2) * m.weight.size(3)
        std = math.sqrt(1.0 / fan_in)
        nn.init.normal_(m.weight, 0, std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        fan_in = m.weight.size(1)
        std = math.sqrt(1.0 / fan_in)
        nn.init.normal_(m.weight, 0, std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def orthogonal_init(m):
    """Orthogonal initialization"""
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def analyze_initialization(model, input_size=(1, 3, 32, 32)):
    """
    Analyze weight initialization by computing statistics
    
    Returns statistics about weight distributions and activations
    """
    model.eval()
    stats = {}
    
    # Weight statistics
    for name, param in model.named_parameters():
        if param.dim() >= 2:  # Only analyze conv and linear weights
            stats[f'{name}_mean'] = param.data.mean().item()
            stats[f'{name}_std'] = param.data.std().item()
            stats[f'{name}_min'] = param.data.min().item()
            stats[f'{name}_max'] = param.data.max().item()
    
    # Forward pass to analyze activations
    with torch.no_grad():
        x = torch.randn(input_size)
        
        # Hook to collect activation statistics
        activation_stats = {}
        hooks = []
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activation_stats[f'{name}_mean'] = output.mean().item()
                    activation_stats[f'{name}_std'] = output.std().item()
                    activation_stats[f'{name}_dead_neurons'] = (output == 0).float().mean().item()
            return hook
        
        # Register hooks for key layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU)):
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Forward pass
        try:
            _ = model(x)
            stats.update(activation_stats)
        except Exception as e:
            print(f"Forward pass failed during analysis: {e}")
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
    
    return stats


def print_initialization_analysis(model, input_size=(1, 3, 32, 32)):
    """Print analysis of weight initialization"""
    stats = analyze_initialization(model, input_size)
    
    print("Weight Initialization Analysis")
    print("=" * 50)
    
    # Print weight statistics
    weight_keys = [k for k in stats.keys() if 'weight' in k and 'mean' in k]
    if weight_keys:
        print("\nWeight Statistics:")
        for key in sorted(weight_keys):
            layer_name = key.replace('_mean', '')
            if f'{layer_name}_std' in stats:
                print(f"  {layer_name}: mean={stats[key]:.4f}, std={stats[f'{layer_name}_std']:.4f}")
    
    # Print activation statistics
    activation_keys = [k for k in stats.keys() if 'mean' in k and 'weight' not in k]
    if activation_keys:
        print("\nActivation Statistics:")
        for key in sorted(activation_keys):
            layer_name = key.replace('_mean', '')
            if f'{layer_name}_std' in stats and f'{layer_name}_dead_neurons' in stats:
                print(f"  {layer_name}: mean={stats[key]:.4f}, "
                      f"std={stats[f'{layer_name}_std']:.4f}, "
                      f"dead={stats[f'{layer_name}_dead_neurons']:.2%}")


if __name__ == "__main__":
    # Test initialization strategies
    from .resnet import resnet18
    
    print("Testing ResNet-18 initialization strategies...")
    print("=" * 60)
    
    strategies = ['he_normal', 'he_uniform', 'xavier_normal', 'xavier_uniform']
    
    for strategy in strategies:
        print(f"\n{strategy.upper()} INITIALIZATION")
        print("-" * 40)
        
        model = resnet18()
        init_weights(model, init_type=strategy, zero_gamma=True)
        print_initialization_analysis(model)