"""The small CNN baseline for AD/NC classification."""

import torch
from torch import nn


class SmallCNN(nn.Module):
    """Four Conv-GroupNorm-ReLU-MaxPool blocks, global average pooling and one AD logit.

    GroupNorm keeps no running statistics, so evaluation never depends on batch
    composition. Global pooling accepts any input size of at least 16 pixels.
    """

    def __init__(self, input_channels: int = 1) -> None:
        super().__init__()
        if input_channels not in (1, 3):
            raise ValueError("input_channels must be 1 or 3.")
        self.input_channels = input_channels
        layers = []
        in_channels = input_channels
        for out_channels in (16, 32, 64, 128):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(4, out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ])
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Map [batch, channels, height, width] images to [batch] AD logits."""
        if images.ndim != 4 or images.shape[1] != self.input_channels or min(images.shape[-2:]) < 16:
            raise ValueError(f"Expected [batch, {self.input_channels}, height, width] with sides >= 16.")
        features = self.features(images).mean(dim=(2, 3))
        return self.classifier(features).squeeze(1)
