"""
modules.py — models for ADNI AD vs NC classification.

"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU, twice, then 2x2 max-pool (halves H and W)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleCNN(nn.Module):
    """Four conv blocks (32 -> 64 -> 128 -> 256 channels) + global pool + classifier.

    Spatial size for a 224x224 input: 224 -> 112 -> 56 -> 28 -> 14 -> pooled to 1x1.
    About 1.2M parameters — trains fine on an Apple GPU or CPU.
    """

    def __init__(self, in_chans: int = 1, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_chans, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)          # [B, 256, h, w] -> [B, 256, 1, 1]
        self.classifier = nn.Sequential(
            nn.Flatten(),                            # -> [B, 256]
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),             # -> [B, num_classes]
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Quick shape check: python modules.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    model = SimpleCNN(in_chans=1, num_classes=2)
    print(f"SimpleCNN parameters: {count_parameters(model):,}")
    for size in (224, 256):
        x = torch.randn(4, 1, size, size)
        print(f"input {tuple(x.shape)} -> output {tuple(model(x).shape)}")