"""
ResNet Architecture Implementation

Implements ResNet-18, ResNet-34, ResNet-50, and ResNet-101 architectures
with CIFAR-10 specific modifications.

Key adaptations for CIFAR-10 (32x32 images):
- Use 3x3 conv stem instead of 7x7 + maxpool
- Start with stride=1 to preserve small image resolution
- 4 stages with [64, 128, 256, 512] channels
"""

import torch
import torch.nn as nn
from .blocks import BasicBlock, Bottleneck, PreActBasicBlock, PreActBottleneck


class ResNet(nn.Module):
    """
    ResNet Architecture for CIFAR-10
    
    Args:
        block: BasicBlock or Bottleneck
        layers: List of number of blocks in each stage
        num_classes: Number of output classes
        pre_activation: Whether to use pre-activation variant
    """
    
    def __init__(self, block, layers, num_classes=10, pre_activation=False):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.pre_activation = pre_activation
        
        # CIFAR-10 specific stem (3x3 conv, no maxpool)
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        
        if not pre_activation:
            self.bn1 = nn.BatchNorm2d(64)
        
        # 4 ResNet stages
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Final layers for pre-activation
        if pre_activation:
            self.bn_final = nn.BatchNorm2d(512 * block.expansion)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        """Create a layer with specified number of blocks"""
        downsample = None
        
        # Create downsample layer if dimensions change
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            if self.pre_activation:
                # Pre-activation downsample: conv only (BN+ReLU applied before)
                downsample = nn.Conv2d(
                    self.in_channels, out_channels * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                )
            else:
                # Post-activation downsample: conv + BN
                downsample = nn.Sequential(
                    nn.Conv2d(
                        self.in_channels, out_channels * block.expansion,
                        kernel_size=1, stride=stride, bias=False
                    ),
                    nn.BatchNorm2d(out_channels * block.expansion)
                )
        
        layers = []
        # First block (may have downsample)
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization for conv layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # Initialize BN weights
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        # Zero-initialize last BN in each residual branch
        # This helps with training stability for deep networks
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)
            elif isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, PreActBasicBlock):
                nn.init.constant_(m.bn2.weight, 0)
            elif isinstance(m, PreActBottleneck):
                nn.init.constant_(m.bn3.weight, 0)
    
    def forward(self, x):
        # Stem
        x = self.conv1(x)
        
        if not self.pre_activation:
            x = self.bn1(x)
            x = torch.relu(x)
            
        # ResNet stages  
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Final activation for pre-activation
        if self.pre_activation:
            x = self.bn_final(x)
            x = torch.relu(x)
            
        # Global average pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def get_feature_maps(self, x):
        """Return intermediate feature maps for analysis"""
        features = {}
        
        # Stem
        x = self.conv1(x)
        if not self.pre_activation:
            x = self.bn1(x)
            x = torch.relu(x)
        features['stem'] = x.clone()
        
        # Stages
        x = self.layer1(x)
        features['stage1'] = x.clone()
        
        x = self.layer2(x)
        features['stage2'] = x.clone()
        
        x = self.layer3(x)
        features['stage3'] = x.clone()
        
        x = self.layer4(x)
        features['stage4'] = x.clone()
        
        return features


def resnet18(num_classes=10, pre_activation=False):
    """ResNet-18 for CIFAR-10"""
    block = PreActBasicBlock if pre_activation else BasicBlock
    model = ResNet(block, [2, 2, 2, 2], num_classes, pre_activation)
    return model


def resnet34(num_classes=10, pre_activation=False):
    """ResNet-34 for CIFAR-10"""
    block = PreActBasicBlock if pre_activation else BasicBlock
    model = ResNet(block, [3, 4, 6, 3], num_classes, pre_activation)
    return model


def resnet50(num_classes=10, pre_activation=False):
    """ResNet-50 for CIFAR-10"""
    block = PreActBottleneck if pre_activation else Bottleneck
    model = ResNet(block, [3, 4, 6, 3], num_classes, pre_activation)
    return model


def resnet101(num_classes=10, pre_activation=False):
    """ResNet-101 for CIFAR-10"""
    block = PreActBottleneck if pre_activation else Bottleneck
    model = ResNet(block, [3, 4, 23, 3], num_classes, pre_activation)
    return model


def count_parameters(model):
    """Count the number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model, input_size=(1, 3, 32, 32)):
    """Print model summary"""
    total_params = count_parameters(model)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Total trainable parameters: {total_params:,}")
    
    # Test forward pass
    with torch.no_grad():
        x = torch.randn(input_size)
        try:
            output = model(x)
            print(f"Input shape: {x.shape}")
            print(f"Output shape: {output.shape}")
            print(f"Memory usage: {torch.cuda.memory_allocated() / 1024**2:.1f}MB" 
                  if torch.cuda.is_available() else "CPU mode")
        except Exception as e:
            print(f"Forward pass failed: {e}")


if __name__ == "__main__":
    # Test all ResNet variants
    models = {
        'ResNet-18': resnet18(),
        'ResNet-18 (Pre-act)': resnet18(pre_activation=True),
        'ResNet-34': resnet34(),
        'ResNet-34 (Pre-act)': resnet34(pre_activation=True),
        'ResNet-50': resnet50(),
        'ResNet-50 (Pre-act)': resnet50(pre_activation=True),
        'ResNet-101': resnet101(),
        'ResNet-101 (Pre-act)': resnet101(pre_activation=True),
    }
    
    print("ResNet Architecture Summary")
    print("=" * 50)
    
    for name, model in models.items():
        print(f"\n{name}:")
        print(f"  Parameters: {count_parameters(model):,}")
        
    # Test one model in detail
    print(f"\nDetailed summary for ResNet-18:")
    print("-" * 30)
    model_summary(resnet18())