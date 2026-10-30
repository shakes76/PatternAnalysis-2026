"""ConvNeXt model components for image classification."""

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Randomly drops residual branches during training for regularization."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError("drop_prob must be in the range [0, 1)")
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x * mask / keep_prob


class ConvNeXtBlock(nn.Module):
    """Applies ConvNeXt feature processing with a residual connection."""

    def __init__(self, dim, layer_scale_init_value=1e-6, drop_path_rate=0.0):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pointwise_conv1 = nn.Linear(dim, 4 * dim)
        self.activation = nn.GELU()
        self.pointwise_conv2 = nn.Linear(4 * dim, dim)
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x):
        residual = x
        x = self.depthwise_conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pointwise_conv1(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.layer_scale * x
        x = x.permute(0, 3, 1, 2)
        return residual + self.drop_path(x)


class LayerNorm2d(nn.Module):
    """Applies channel-wise LayerNorm to a channels-first 2D feature map."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class ConvNeXtClassifier(nn.Module):
    """Builds a four-stage ConvNeXt model that outputs class logits."""

    def __init__(
        self,
        in_channels=1,
        num_classes=2,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        layer_scale_init_value=1e-6,
        drop_path_rate=0.1,
    ):
        super().__init__()
        if len(depths) != 4 or len(dims) != 4:
            raise ValueError("depths and dims must each contain four values")
        if any(depth <= 0 for depth in depths) or any(dim <= 0 for dim in dims):
            raise ValueError("all stage depths and dimensions must be positive")
        if not 0.0 <= drop_path_rate < 1.0:
            raise ValueError("drop_path_rate must be in the range [0, 1)")

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0]),
        )
        drop_path_rates = torch.linspace(0.0, drop_path_rate, sum(depths)).tolist()
        self.stages = nn.ModuleList()
        block_index = 0
        for dim, depth in zip(dims, depths):
            blocks = []
            for _ in range(depth):
                blocks.append(
                    ConvNeXtBlock(
                        dim, layer_scale_init_value, drop_path_rates[block_index]
                    )
                )
                block_index += 1
            self.stages.append(nn.Sequential(*blocks))
        self.downsample_layers = nn.ModuleList(
            [
                nn.Sequential(
                    LayerNorm2d(dims[index]),
                    nn.Conv2d(dims[index], dims[index + 1], kernel_size=2, stride=2),
                )
                for index in range(3)
            ]
        )
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        x = self.stem(x)
        for index, stage in enumerate(self.stages):
            x = stage(x)
            if index < len(self.downsample_layers):
                x = self.downsample_layers[index](x)
        x = x.mean(dim=(2, 3))
        x = self.norm(x)
        return self.head(x)
