"""
Narrow ConvNeXt for AD/NC classification.

Single-channel 224x224 input; widths 32/64/128/256; depths 3/3/9/3.
DropPath increases from 0 to 0.2 across 18 blocks.
Default model: 3,178,018 parameters, trained from scratch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Helper layers
# --------------------------------------------------------------------------- #
class LayerNorm(nn.Module):
    """LayerNorm for channels-first or channels-last tensors."""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ("channels_last", "channels_first"):
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        # channels_first: Normalize along the channel dimension.
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[None, :, None, None] * x + self.bias[None, :, None, None]


class DropPath(nn.Module):
    """Drop entire residual branches randomly during training."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x.div(keep_prob) * mask


# --------------------------------------------------------------------------- #
# ConvNeXt Block
# --------------------------------------------------------------------------- #
class ConvNeXtBlock(nn.Module):
    def __init__(self, channels, drop_path=0.0):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, 7, padding=3, groups=channels
        )
        self.norm = LayerNorm(channels)
        self.expand = nn.Linear(channels, 4 * channels)
        self.act = nn.GELU()
        self.project = nn.Linear(4 * channels, channels)
        self.gamma = nn.Parameter(torch.full((channels,), 1e-6))
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        residual = x
        x = self.depthwise(x).permute(0, 2, 3, 1)      # NCHW -> NHWC
        x = self.project(self.act(self.expand(self.norm(x)))) * self.gamma
        return residual + self.drop_path(x.permute(0, 3, 1, 2))  # NHWC -> NCHW


# --------------------------------------------------------------------------- #
# Narrow ConvNeXt: Tiny block counts with reduced channel widths
# --------------------------------------------------------------------------- #
class ConvNeXtNarrow(nn.Module):
    """Narrow ConvNeXt for single-slice AD/NC classification."""

    def __init__(
        self,
        in_channels=1,
        num_classes=2,
        base_channels=32,
        depths=(3, 3, 9, 3),
        drop_path_rate=0.2,
    ):
        super().__init__()
        dims = [base_channels * (2 ** i) for i in range(4)]
        self.dims = dims
        self.depths = list(depths)

        # Stem: 1 -> 32 channels, 224x224 -> 56x56
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
                LayerNorm(dims[0], data_format="channels_first"),
            )
        ])
        # Three downsamplers: double channels and halve spatial size.
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(dims[i], data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            ))

        # Linearly increase DropPath across all 18 blocks.
        rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.stages = nn.ModuleList()
        offset = 0
        for dim, depth in zip(dims, depths):
            self.stages.append(nn.Sequential(*[
                ConvNeXtBlock(dim, rates[offset + j]) for j in range(depth)
            ]))
            offset += depth

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward_features(self, x):
        for downsample, stage in zip(self.downsample_layers, self.stages):
            x = stage(downsample(x))
        return x

    def forward(self, x):
        x = self.forward_features(x)
        # Global average pooling -> LayerNorm -> classification
        return self.head(self.norm(x.mean(dim=(-2, -1))))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------- #
# Model factory
# --------------------------------------------------------------------------- #
CLASS_TO_INDEX = {"NC": 0, "AD": 1}
INDEX_TO_CLASS = {0: "NC", 1: "AD"}


def build_model(**kwargs):
    """Build the default model with approximately 3.18M parameters."""
    model = ConvNeXtNarrow(**kwargs)
    return model


if __name__ == "__main__":
    m = build_model()
    n = m.num_parameters()
    print(f"Parameters: {n:,} ({n / 1e6:.2f}M)")
    x = torch.randn(2, 1, 224, 224)
    y = m(x)
    print(f"Input {tuple(x.shape)} -> output {tuple(y.shape)}")
