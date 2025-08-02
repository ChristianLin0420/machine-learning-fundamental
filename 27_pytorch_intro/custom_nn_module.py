"""
Custom PyTorch nn.Module Examples
=================================

This module demonstrates custom PyTorch nn.Module implementations:
- Basic custom modules
- Multi-layer perceptron (MLP)
- Custom layers and activation functions
- Parameter initialization
- Forward hooks and debugging
- Model composition and modularity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Callable, Tuple
import math

class SimpleLinear(nn.Module):
    """Simple linear layer implementation from scratch"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights properly
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Xavier uniform"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: y = xW^T + b"""
        output = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            output += self.bias
        return output
    
    def extra_repr(self) -> str:
        """String representation for printing"""
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

class CustomActivation(nn.Module):
    """Custom activation function: Swish"""
    
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Swish activation: x * sigmoid(beta * x)"""
        return x * torch.sigmoid(self.beta * x)

class DropoutCustom(nn.Module):
    """Custom dropout implementation"""
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"Dropout probability must be between 0 and 1, got {p}")
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Generate random mask
            mask = torch.bernoulli(torch.full_like(x, 1 - self.p))
            # Scale by 1/(1-p) to maintain expected value
            return x * mask / (1 - self.p)
        else:
            return x

class MyMLP(nn.Module):
    """Custom Multi-Layer Perceptron"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_sizes: List[int],
                 output_size: int,
                 activation: str = 'relu',
                 dropout_rate: float = 0.0,
                 use_batch_norm: bool = False):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Build layers
        layers = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            # Linear layer
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            
            # Add batch normalization (except for output layer)
            if use_batch_norm and i < len(layer_sizes) - 2:
                layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
            
            # Add activation (except for output layer)
            if i < len(layer_sizes) - 2:
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                elif activation == 'sigmoid':
                    layers.append(nn.Sigmoid())
                elif activation == 'swish':
                    layers.append(CustomActivation())
                else:
                    raise ValueError(f"Unknown activation: {activation}")
                
                # Add dropout
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using He initialization for ReLU"""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class ResidualBlock(nn.Module):
    """Simple residual block"""
    
    def __init__(self, size: int, dropout_rate: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(size, size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)  # Residual connection

class ResNet(nn.Module):
    """Simple ResNet-style network"""
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 num_blocks: int = 2,
                 dropout_rate: float = 0.0):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_size, dropout_rate) 
            for _ in range(num_blocks)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_proj(x))
        
        for block in self.blocks:
            x = block(x)
        
        return self.output_proj(x)

class AttentionLayer(nn.Module):
    """Simple self-attention layer"""
    
    def __init__(self, embed_dim: int, num_heads: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape
        
        # Compute Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(attention_weights, V)
        
        # Reshape and project
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.output(attended)
        
        return output

def demonstrate_basic_modules():
    """Demonstrate basic custom modules"""
    print("=" * 60)
    print("BASIC CUSTOM MODULES")
    print("=" * 60)
    
    # Test SimpleLinear
    print("1. Testing SimpleLinear layer:")
    linear = SimpleLinear(3, 2)
    x = torch.randn(5, 3)
    output = linear(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Weight shape: {linear.weight.shape}")
    print(f"Bias shape: {linear.bias.shape if linear.bias is not None else None}")
    print(f"Linear layer: {linear}")
    
    # Test CustomActivation
    print("\n2. Testing CustomActivation (Swish):")
    activation = CustomActivation(beta=1.0)
    x = torch.linspace(-5, 5, 11)
    y = activation(x)
    
    print(f"Input: {x}")
    print(f"Swish output: {y}")
    
    # Compare with standard activations
    relu_out = F.relu(x)
    sigmoid_out = torch.sigmoid(x)
    
    print(f"ReLU output: {relu_out}")
    print(f"Sigmoid output: {sigmoid_out}")
    
    # Test DropoutCustom
    print("\n3. Testing CustomDropout:")
    dropout = DropoutCustom(p=0.5)
    x = torch.ones(10)
    
    # Training mode
    dropout.train()
    y_train = dropout(x)
    print(f"Training mode output (p=0.5): {y_train}")
    
    # Evaluation mode
    dropout.eval()
    y_eval = dropout(x)
    print(f"Evaluation mode output: {y_eval}")
    
    return linear, activation, dropout

def demonstrate_mlp():
    """Demonstrate custom MLP implementation"""
    print("\n" + "=" * 60)
    print("CUSTOM MLP DEMONSTRATION")
    print("=" * 60)
    
    # Create different MLP configurations
    configs = [
        {"name": "Simple MLP", "hidden_sizes": [64, 32], "activation": "relu"},
        {"name": "Deep MLP", "hidden_sizes": [128, 64, 32, 16], "activation": "relu"},
        {"name": "MLP with Dropout", "hidden_sizes": [64, 32], "activation": "relu", "dropout_rate": 0.2},
        {"name": "MLP with BatchNorm", "hidden_sizes": [64, 32], "activation": "relu", "use_batch_norm": True},
        {"name": "Swish MLP", "hidden_sizes": [64, 32], "activation": "swish"}
    ]
    
    models = {}
    
    for config in configs:
        print(f"\n{config['name']}:")
        
        # Create model
        model = MyMLP(
            input_size=10,
            hidden_sizes=config["hidden_sizes"],
            output_size=3,
            activation=config.get("activation", "relu"),
            dropout_rate=config.get("dropout_rate", 0.0),
            use_batch_norm=config.get("use_batch_norm", False)
        )
        
        models[config["name"]] = model
        
        # Test forward pass
        x = torch.randn(8, 10)  # Batch of 8 samples
        output = model(x)
        
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {model.count_parameters():,}")
        print(f"  Architecture: {model.network}")
    
    return models

def demonstrate_advanced_modules():
    """Demonstrate advanced module concepts"""
    print("\n" + "=" * 60)
    print("ADVANCED MODULE CONCEPTS")
    print("=" * 60)
    
    # 1. ResNet-style network
    print("1. ResNet-style network:")
    resnet = ResNet(input_size=20, hidden_size=64, output_size=5, num_blocks=3)
    x = torch.randn(10, 20)
    output = resnet(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Number of residual blocks: {len(resnet.blocks)}")
    
    # 2. Attention mechanism
    print("\n2. Self-attention layer:")
    attention = AttentionLayer(embed_dim=64, num_heads=4)
    x = torch.randn(2, 10, 64)  # (batch, sequence, embedding)
    output = attention(x)
    
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Number of heads: {attention.num_heads}")
    print(f"  Head dimension: {attention.head_dim}")
    
    return resnet, attention

def demonstrate_parameter_access():
    """Demonstrate parameter access and manipulation"""
    print("\n" + "=" * 60)
    print("PARAMETER ACCESS AND MANIPULATION")
    print("=" * 60)
    
    # Create a simple model
    model = MyMLP(input_size=5, hidden_sizes=[10, 8], output_size=3)
    
    print("1. Named parameters:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}")
    
    print("\n2. Parameter groups:")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    print("\n3. Freezing parameters:")
    # Freeze first layer
    for param in model.network[0].parameters():
        param.requires_grad = False
    
    trainable_after_freeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable after freezing first layer: {trainable_after_freeze:,}")
    
    print("\n4. Parameter statistics:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: mean={param.mean().item():.4f}, std={param.std().item():.4f}")
    
    return model

def demonstrate_hooks():
    """Demonstrate forward and backward hooks"""
    print("\n" + "=" * 60)
    print("HOOKS DEMONSTRATION")
    print("=" * 60)
    
    # Create model
    model = MyMLP(input_size=4, hidden_sizes=[6, 4], output_size=2)
    
    # Storage for activations
    activations = {}
    gradients = {}
    
    def forward_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    def backward_hook(name):
        def hook(module, grad_input, grad_output):
            gradients[name] = grad_output[0].detach()
        return hook
    
    # Register hooks
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(forward_hook(name)))
            handles.append(module.register_backward_hook(backward_hook(name)))
    
    # Forward and backward pass
    x = torch.randn(3, 4, requires_grad=True)
    target = torch.randn(3, 2)
    
    output = model(x)
    loss = F.mse_loss(output, target)
    loss.backward()
    
    print("Forward activations:")
    for name, activation in activations.items():
        print(f"  {name}: shape={activation.shape}, mean={activation.mean().item():.4f}")
    
    print("\nBackward gradients:")
    for name, gradient in gradients.items():
        print(f"  {name}: shape={gradient.shape}, mean={gradient.mean().item():.4f}")
    
    # Clean up hooks
    for handle in handles:
        handle.remove()
    
    return model, activations, gradients

def visualize_model_architectures(models):
    """Visualize model architectures and properties"""
    print("\n" + "=" * 60)
    print("MODEL VISUALIZATION")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Custom PyTorch Module Analysis', fontsize=16)
    
    # 1. Parameter count comparison
    ax1 = axes[0, 0]
    model_names = list(models.keys())
    param_counts = [models[name].count_parameters() for name in model_names]
    
    bars = ax1.bar(range(len(model_names)), param_counts, color='skyblue')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Parameter Count')
    ax1.set_title('Parameter Count Comparison')
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    
    for bar, count in zip(bars, param_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{count:,}', ha='center', va='bottom')
    
    # 2. Activation function comparison
    ax2 = axes[0, 1]
    x = torch.linspace(-3, 3, 100)
    
    activations = {
        'ReLU': F.relu(x),
        'Sigmoid': torch.sigmoid(x),
        'Tanh': torch.tanh(x),
        'Swish': x * torch.sigmoid(x)
    }
    
    for name, y in activations.items():
        ax2.plot(x.numpy(), y.numpy(), label=name, linewidth=2)
    
    ax2.set_xlabel('Input')
    ax2.set_ylabel('Output')
    ax2.set_title('Activation Functions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Weight distribution (example model)
    ax3 = axes[0, 2]
    model = models['Simple MLP']
    
    all_weights = []
    for param in model.parameters():
        if param.dim() > 1:  # Only weight matrices, not biases
            all_weights.extend(param.detach().flatten().numpy())
    
    ax3.hist(all_weights, bins=50, alpha=0.7, color='green')
    ax3.set_xlabel('Weight Value')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Weight Distribution (Simple MLP)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Forward pass timing
    ax4 = axes[1, 0]
    x_test = torch.randn(1000, 10)
    
    times = {}
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            if torch.cuda.is_available():
                start_time.record()
                _ = model(x_test)
                end_time.record()
                torch.cuda.synchronize()
                elapsed_time = start_time.elapsed_time(end_time)
            else:
                import time
                start = time.time()
                _ = model(x_test)
                elapsed_time = (time.time() - start) * 1000  # Convert to ms
            
            times[name] = elapsed_time
    
    names = list(times.keys())
    timing_values = list(times.values())
    
    bars = ax4.bar(range(len(names)), timing_values, color='orange')
    ax4.set_xlabel('Model')
    ax4.set_ylabel('Time (ms)')
    ax4.set_title('Forward Pass Timing (1000 samples)')
    ax4.set_xticks(range(len(names)))
    ax4.set_xticklabels(names, rotation=45, ha='right')
    
    # 5. Gradient flow simulation
    ax5 = axes[1, 1]
    
    # Simulate gradient magnitudes through layers
    layer_names = ['Input', 'Hidden1', 'Hidden2', 'Output']
    grad_magnitudes = [1.0, 0.8, 0.6, 0.3]  # Simulated gradient decay
    
    ax5.plot(layer_names, grad_magnitudes, 'o-', linewidth=3, markersize=8, color='red')
    ax5.set_ylabel('Gradient Magnitude')
    ax5.set_title('Gradient Flow Through Layers')
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, 1.2)
    
    # 6. Model complexity vs accuracy (simulated)
    ax6 = axes[1, 2]
    
    complexities = [models[name].count_parameters() for name in model_names]
    # Simulate accuracy (more complex models generally perform better up to a point)
    accuracies = [85 + 5 * np.log(c/1000) + np.random.normal(0, 2) for c in complexities]
    
    ax6.scatter(complexities, accuracies, s=100, alpha=0.7, color='purple')
    ax6.set_xlabel('Model Complexity (Parameters)')
    ax6.set_ylabel('Accuracy (%)')
    ax6.set_title('Model Complexity vs Performance')
    ax6.grid(True, alpha=0.3)
    
    for i, name in enumerate(model_names):
        ax6.annotate(name, (complexities[i], accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('27_pytorch_intro/plots/custom_modules_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def comprehensive_module_demo():
    """Run comprehensive custom module demonstration"""
    print("PyTorch Custom Modules Comprehensive Demo")
    print("========================================")
    
    # Run all demonstrations
    basic_modules = demonstrate_basic_modules()
    mlp_models = demonstrate_mlp()
    advanced_modules = demonstrate_advanced_modules()
    param_model = demonstrate_parameter_access()
    hooks_results = demonstrate_hooks()
    
    # Create visualizations
    fig = visualize_model_architectures(mlp_models)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✅ Basic custom modules (Linear, Activation, Dropout)")
    print("✅ Custom MLP with various configurations")
    print("✅ Advanced modules (ResNet, Attention)")
    print("✅ Parameter access and manipulation")
    print("✅ Forward and backward hooks")
    print("✅ Model architecture visualization")
    print("✅ Performance analysis and comparison")
    
    return {
        'basic_modules': basic_modules,
        'mlp_models': mlp_models,
        'advanced_modules': advanced_modules,
        'parameter_model': param_model,
        'hooks_results': hooks_results,
        'visualization': fig
    }

if __name__ == "__main__":
    results = comprehensive_module_demo() 