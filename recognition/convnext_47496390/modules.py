"""
Contains neural network architectures for ADNI MRI classifications.
Implements:
    1. BaselineCNN: a standard 3-stage CNN for establishing an empirical baseline
    2. ConvNeXt: a modernized convolutional backbone for classifying ADNI brain scans
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/Bengal1/CNN-Architecture-Guide/blob/main/SimpleCNN.py
# https://share.gemini.google/ylSwoghQk0f9
class BaselineCNN(nn.Module):
    """Baseline Convolutional Neural Network for ADNI AD-vs-CN classification.
Architecture Overview:
    - 3 Convolutional Blocks:
        Block 1: Conv2d(1 -> 32, 3x3, pad 1) -> BatchNorm2d(32) -> ReLU -> MaxPool2d(2x2)
        Block 2: Conv2d(32 -> 64, 3x3, pad 1) -> BatchNorm2d(64) -> ReLU -> MaxPool2d(2x2)
        Block 3: Conv2d(64 -> 128, 3x3, pad 1) -> BatchNorm2d(128) -> ReLU -> AdaptiveAvgPool2d(8x8)
    - Fully Connected Classifier:
        Flatten (128 * 8 * 8 = 8192) -> Linear(8192 -> 128) -> ReLU -> Dropout(0.3) -> Linear(128 -> num_classes)
    Input:
        x: Tensor of shape (B, 1, H, W) - Grayscale brain MRI slice.
    Output:
        logits: Tensor of shape (B, num_classes) - Unnormalized class scores."""

    def __init__(self, num_classes=2, in_channels = 1, dropout_rate=0.3):
        super().__init__()

        # convolutional layers - feature extraction stage
        # block 1: (B, 1, H, W) -> (B, 32, H/2, W/2) <- 1 channel for gray scale image to 32 feature maps
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), # num features is the out_channels of conv layer
            nn.ReLU(inplace=True), #  updates the existing tensor values in place --> ower your peak memory usage
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # block 2: (B, 32, H/2, W/2) -> (B, 64, H/4, W/4)
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # block 3: (B, 64, H/4, W/4) -> (B, 128, H/8, W/8)
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # reduces feature maps:  (B, 128, 8, 8)
            # gives classifier a fixed size input
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # classifier
        self.classifier = nn.Sequential(
            nn.Flatten(), # # (B, 128, 8, 8) -> (B, 8192)
            nn.Linear(128 * 8 * 8, 128), # 128 feature maps from conv, each map contributes 8 values -- 128 outputs
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes) # feature map outputs to class scores: (B, 2)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)


# https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py
# https://share.gemini.google/n9s7qtpk7eQD
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # match dimensions: (batch_size, 1, 1, 1) for 4D tensor
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class LayerNorm(nn.Module):
    """
LayerNorm supporting both channels_last (N, H, W, C) and channels_first (N, C, H, W). Reflects the ConvNeXt official architecture (Liu et al., 2022)
    - 'channels_last': Uses PyTorch's native F.layer_norm over the trailing channel dimension.
        • Expects input tensor to already arrive in shape (N, H, W, C)
        • Permutes the tensor from (N, C, H, W) to (N, H, W, C) before the normalization step, applies standard PyTorch LayerNorm or Linear layers, and then permutes it back. Permutations done by ConvNeXt block.
        • This approach is often slightly faster in PyTorch hardware kernels.
    - 'channels_first': Computes mean and variance across the channel axis while preserving (N, C, H, W). 
	• Keeps tensors in the standard convolutional format.
	• Performs LayerNorm across channels for each spatial location.
    """
    
    def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_last",):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"Unsupported data_format '{data_format}'. Choose 'channels_last' or 'channels_first'.")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # Normalization over channel dimension C: shape (N, C, H, W)
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block.

    Structure:
        1. 7x7 Depthwise Conv (groups=dim, padding=3): Large receptive field with decoupled spatial mixing.
        2. Permute to channels-last: (N, C, H, W) -> (N, H, W, C).
        3. LayerNorm: Applied across channel dimension.
        4. Pointwise Linear 1 (Inverted Bottleneck): dim -> 4 * dim.
        5. GELU Activation: Single non-linear activation per block (Transformers design).
        6. Pointwise Linear 2: 4 * dim -> dim.
        7. LayerScale: Elementwise scaling by learnable parameter vector gamma (init 1e-6).
        8. Permute back to channels-first: (N, H, W, C) -> (N, C, H, W).
        9. Residual Connection with Stochastic Depth (DropPath): shortcut + drop_path(x).
    """
    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init_value: float = 1e-6):
        super().__init__()
        # 7x7 Depthwise convolution
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        # Normalization, channels-last layour
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")

        # Pointwise inverted bottleneck MLP (dim -> 4*dim -> dim)
        # (1x1 convs implemented efficiently as linear layers)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        # LayerScale
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim,)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        # Spatial depthwise filtering in channels-first
        x = self.dwconv(x)

        # Linear layers run faster over channels-last format
        # Permute to channels-last for LayerNorm and pointwise linear operations
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # LayerScale
        if self.gamma is not None:
            x = self.gamma * x

        # Permute back to channels-first
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        # Residual shortcut + stochastic depth
        x = shortcut + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    """
    ConvNeXt Backbone and Classifier.

    The standard ImageNet configuration in Liu et al. (2022) specifies in_chans=3 for RGB photography.
    For structural T1-weighted brain MRI, scans are single-channel grayscale scalar intensity fields.
    Setting in_chans=1 is a task adaptation that processes native MRI intensities directly,
    eliminating redundant 3x channel replication, saving memory bandwidth, and preventing wasted compute.

    Input:
        x: Tensor of shape (B, in_chans, H, W).
    Output:
        logits: Tensor of shape (B, num_classes).
    """
    def __init__(
        self,
        in_chans: int = 1,
        num_classes: int = 2,
        depths: list[int] = (3, 3, 9, 3),
        dims: list[int] = (96, 192, 384, 768),
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        self.downsample_layers = nn.ModuleList()

        # Stem: 4x4 non-overlapping patchify convolution (stride 4) + channels-first LayerNorm
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)

        # 3 Downsampling layers between stages (2x2 stride 2 conv + LayerNorm)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # 4 Feature stages with progressive DropPath rates
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    ConvNeXtBlock(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        # Classification Head: Global Avg Pool -> LayerNorm (channels_last / 1D) -> Linear
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts spatial representations across all four stages."""
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # Global average pool (N, C, H, W) -> (N, C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        return self.head(x)

# Factory Functions

def convnext_base(num_classes: int = 1000, in_chans: int = 3, **kwargs):
    """
    ConvNeXt-Base variant:
    Depths: [3, 3, 27, 3], Channels: [128, 256, 512, 1024] (~88M parameters).
    """
    return ConvNeXt(
        in_chans=in_chans,
        num_classes=num_classes,
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
        drop_path_rate=kwargs.get("drop_path_rate", 0.5),
        layer_scale_init_value=kwargs.get("layer_scale_init_value", 1e-6),
    )

def convnext_tiny(num_classes: int = 2, in_chans: int = 3, **kwargs):
    """
    ConvNeXt-Tiny variant:
    Depths: [3, 3, 9, 3], Channels: [96, 192, 384, 768] (~28M parameters).
    """
    return ConvNeXt(
        in_chans=in_chans,
        num_classes=num_classes,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=kwargs.get("drop_path_rate", 0.1),
        layer_scale_init_value=kwargs.get("layer_scale_init_value", 1e-6),
    )
