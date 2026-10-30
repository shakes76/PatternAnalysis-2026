"""Small CNN baseline for the COMP3710 ADNI classification investigation.

This is the simpler comparison model, NOT the Hard-Difficulty ConvNeXt model.
Input: preprocessed [B, 1, 224, 224] MRI JPEG slices. Output: two raw logits,
ordered NC=0, AD=1. Use cross-entropy on logits, not on softmax probabilities.

Four Conv/BatchNorm/ReLU blocks downsample spatially. A 4x4 pooled feature map
retains coarse anatomical location before a small fully connected classifier.
The network uses only PyTorch (no NumPy, torchvision, or external model package).

API reference: https://docs.pytorch.org/docs/stable/nn.html
Development assistance: ChatGPT drafted code/tests. The student must review,
validate, explain, and disclose this assistance in the assessment report.
"""
from __future__ import annotations

import torch
from torch import nn


class BaselineCNN(nn.Module):
    """A compact CNN trained from scratch; no downloaded/pretrained weights."""

    def __init__(self, dropout: float = 0.25) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        blocks: list[nn.Module] = []
        channels = (1, 16, 32, 64, 128)
        for incoming, outgoing in zip(channels[:-1], channels[1:]):
            blocks.extend([
                nn.Conv2d(incoming, outgoing, kernel_size=3, stride=2,
                          padding=1, bias=False),
                nn.BatchNorm2d(outgoing),
                nn.ReLU(inplace=True),
            ])
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            # Fixed pooling maps 14x14 to 4x4. Unlike CUDA adaptive-pool
            # backward, this supports the strict deterministic-mode check.
            nn.AvgPool2d(kernel_size=5, stride=3),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 2),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1:] != (1, 224, 224):
            raise ValueError("Expected image tensor shape [B, 1, 224, 224].")
        return self.classifier(self.features(images))


def parameter_count(model: nn.Module) -> int:
    """Count learnable parameters, not BatchNorm running-statistic buffers."""
    return sum(parameter.numel() for parameter in model.parameters()
               if parameter.requires_grad)


# ConvNeXt implementation added for Step 8. The BaselineCNN above is unchanged.
# Architectural reference: Liu et al., "A ConvNet for the 2020s", CVPR 2022,
# https://arxiv.org/abs/2201.03545 (assessment reference [7]).
# This implementation follows the paper's Tiny stage widths/depths. The input
# stem and classification head are adapted to one-channel MRI and two classes.
# No torchvision/timm model or pretrained weights are used.

class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels independently at each NCHW spatial location."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class StochasticDepth(nn.Module):
    """Drop a residual branch per sample in training; identity in evaluation."""

    def __init__(self, probability: float) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("Stochastic-depth probability must be in [0, 1).")
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0.0:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = (torch.rand(shape, dtype=x.dtype, device=x.device) < keep).to(x.dtype)
        return x * mask / keep


class ConvNeXtBlock(nn.Module):
    """7x7 depthwise convolution, channel LN, inverted bottleneck and residual."""

    def __init__(self, channels: int, drop_path: float = 0.0,
                 layer_scale: float = 1e-6) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.expand = nn.Linear(channels, 4 * channels)
        self.activation = nn.GELU()
        self.contract = nn.Linear(4 * channels, channels)
        self.gamma = nn.Parameter(torch.full((channels,), float(layer_scale)))
        self.drop_path = StochasticDepth(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.depthwise(x).permute(0, 2, 3, 1)
        x = self.contract(self.activation(self.expand(self.norm(x))))
        x = (self.gamma * x).permute(0, 3, 1, 2)
        return shortcut + self.drop_path(x)


class ConvNeXtTiny(nn.Module):
    """ConvNeXt-Tiny: depths (3,3,9,3), widths (96,192,384,768), MRI -> AD/NC.

    Stage/depth arguments exist for small synthetic tests only. The experiment
    builder always instantiates the unmodified Tiny widths and depths.
    """

    def __init__(self, drop_path_rate: float = 0.1, *,
                 depths: tuple[int, ...] = (3, 3, 9, 3),
                 dims: tuple[int, ...] = (96, 192, 384, 768)) -> None:
        super().__init__()
        if (len(depths) != 4 or len(dims) != 4 or
                any(v <= 0 for v in (*depths, *dims)) or
                not 0.0 <= drop_path_rate < 1.0):
            raise ValueError("Expected four positive stage sizes and drop-path in [0,1).")
        self.depths, self.dims = tuple(depths), tuple(dims)
        self.downsample = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1, dims[0], kernel_size=4, stride=4),
                          ChannelLayerNorm(dims[0]))
        ])
        for i in range(3):
            self.downsample.append(nn.Sequential(
                ChannelLayerNorm(dims[i]),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)))
        rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.stages = nn.ModuleList()
        offset = 0
        for channels, count in zip(dims, depths):
            self.stages.append(nn.Sequential(*[
                ConvNeXtBlock(channels, rates[offset + j]) for j in range(count)]))
            offset += count
        self.final_norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], 2)
        self.apply(self._initialise)

    @staticmethod
    def _initialise(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1:] != (1, 224, 224):
            raise ValueError("Expected image tensor shape [B, 1, 224, 224].")
        x = images
        for downsample, stage in zip(self.downsample, self.stages):
            x = stage(downsample(x))
        return self.head(self.final_norm(x.mean(dim=(-2, -1))))
