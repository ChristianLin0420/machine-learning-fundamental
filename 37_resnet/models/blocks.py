"""
ResNet Building Blocks

Implements BasicBlock and Bottleneck blocks with both post-activation (original ResNet)
and pre-activation variants (improved ResNet v2).

References:
- He et al. (2015): "Deep Residual Learning for Image Recognition" (post-activation)
- He et al. (2016): "Identity Mappings in Deep Residual Networks" (pre-activation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    Basic ResNet block with post-activation (original ResNet).
    
    Structure: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+residual) -> ReLU
    """
    expansion = 1  # Output channels multiplier
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        # First convolution
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution  
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Downsample for residual connection
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        residual = x
        
        # First conv-bn-relu
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        # Second conv-bn
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add residual connection
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        
        # Final activation
        out = F.relu(out, inplace=True)
        
        return out


class Bottleneck(nn.Module):
    """
    Bottleneck ResNet block with post-activation (original ResNet).
    
    Structure: Conv1x1 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU -> Conv1x1 -> BN -> (+residual) -> ReLU
    """
    expansion = 4  # Output channels = in_channels * expansion
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        
        # First 1x1 conv (dimension reduction)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3x3 conv (spatial processing)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Second 1x1 conv (dimension expansion)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion,
            kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        # Downsample for residual connection
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        residual = x
        
        # First 1x1 conv-bn-relu
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        
        # 3x3 conv-bn-relu
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        
        # Second 1x1 conv-bn
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Add residual connection
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        
        # Final activation
        out = F.relu(out, inplace=True)
        
        return out


class PreActBasicBlock(nn.Module):
    """
    Basic ResNet block with pre-activation (ResNet v2).
    
    Structure: BN -> ReLU -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> (+residual)
    """
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(PreActBasicBlock, self).__init__()
        
        # Pre-activation batch normalization
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1,
            padding=1, bias=False
        )
        
        # Downsample for residual connection
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        residual = x
        
        # First bn-relu-conv
        out = self.bn1(x)
        out = F.relu(out, inplace=True)
        
        if self.downsample is not None:
            residual = self.downsample(out)
            
        out = self.conv1(out)
        
        # Second bn-relu-conv
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        
        # Add residual connection (no final activation in pre-act)
        out += residual
        
        return out


class PreActBottleneck(nn.Module):
    """
    Bottleneck ResNet block with pre-activation (ResNet v2).
    
    Structure: BN -> ReLU -> Conv1x1 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU -> Conv1x1 -> (+residual)
    """
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(PreActBottleneck, self).__init__()
        
        # Pre-activation batch normalization
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion,
            kernel_size=1, bias=False
        )
        
        # Downsample for residual connection
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        residual = x
        
        # First bn-relu-conv (1x1)
        out = self.bn1(x)
        out = F.relu(out, inplace=True)
        
        if self.downsample is not None:
            residual = self.downsample(out)
            
        out = self.conv1(out)
        
        # Second bn-relu-conv (3x3)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        
        # Third bn-relu-conv (1x1)
        out = self.bn3(out)
        out = F.relu(out, inplace=True)
        out = self.conv3(out)
        
        # Add residual connection (no final activation in pre-act)
        out += residual
        
        return out


def conv3x3(in_channels, out_channels, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride,
        padding=1, bias=False
    )


def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=stride,
        bias=False
    )