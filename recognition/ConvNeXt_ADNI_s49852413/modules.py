"""Pure PyTorch neural network modules for ADNI Brain MRI classification.

Author: Lingxiao Li (s49852413)
Course: COMP3710 / COMP7710 Pattern Recognition and Analysis (2026)

Strict Compliance Note (Section 4.1):
    This file depends EXCLUSIVELY on PyTorch (`torch`, `torch.nn`, `torch.nn.functional`)
    and Python standard library typing annotations. It contains zero dependency on NumPy
    or any external third-party libraries.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-first 2D Layer Normalization for `[B, C, H, W]` feature maps.

    In standard Vision Transformers, LayerNorm operates on channel-last `[B, H, W, C]`
    tensors. ConvNeXt (Liu et al., CVPR 2022) applies LayerNorm across the channel
    dimension `C` at each spatial location `(h, w)`.

    Mathematical formulation:
        y_{b,c,h,w} = gamma_c * (x_{b,c,h,w} - mu_{b,h,w}) / sqrt(sigma^2_{b,h,w} + eps) + beta_c
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class DropPath(nn.Module):
    """Stochastic Depth (DropPath) regularization per sample in residual branches.

    During training, randomly drops entire residual block outputs with probability `drop_prob`
    and scales surviving paths by `1 / (1 - drop_prob)` to preserve expected activation magnitude.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)
        return x.div(keep_prob) * random_tensor


class SimpleCNNBaseline(nn.Module):
    """Standard 3-stage Convolutional Neural Network Baseline for ADNI AD vs. NC screening.

    Serves as the mandatory Section 2.1 baseline benchmark. Uses conventional 3x3 convolutions,
    BatchNorm2d, ReLU activations, 2x2 MaxPooling, global average pooling, and a 2-layer
    classifier head. Because its effective receptive field is local (3x3 kernels without
    large-kernel depthwise mixing or inverted bottlenecks), it provides a clear reference
    against which modernized ConvNeXt architectural gains and confidence calibration are measured.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 24,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.features = nn.Sequential(
            # Stage 1
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Stage 2
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Stage 3
            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(c3, c2),
            nn.ReLU(inplace=True),
            nn.Linear(c2, num_classes),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pre-logit pooled feature vector `[B, C]`."""
        feat_map = self.features(x)
        return self.pool(feat_map).flatten(1)

    def get_last_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Return final spatial feature map `[B, C, H', W']` for spatial attribution."""
        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute unnormalized class logits `[B, num_classes]`."""
        feat_map = self.features(x)
        pooled = self.pool(feat_map)
        return self.classifier(pooled)
