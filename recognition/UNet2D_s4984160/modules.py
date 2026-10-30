"""Standard PyTorch 2D U-Net for six-class HipMRI segmentation.

Adapted from the author's model draft. The default width is reduced to 16
for a CPU training candidate; use base_features=64 for the original width.
"""

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    """Two Conv2d -> BatchNorm2d -> ReLU blocks."""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Max pooling followed by two convolutions."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Learned upsampling, skip concatenation and two convolutions."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Restore pixels lost by pooling odd spatial dimensions.
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    """Map features to unnormalized class scores."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)


class UNet2D(nn.Module):
    """Four-level U-Net: [B,1,H,W] -> logits [B,6,H,W].

    No softmax is applied: CrossEntropyLoss expects logits. Both spatial
    dimensions must be at least 16; training BatchNorm also requires more
    than one value per channel at the deepest level.
    """

    def __init__(self, in_channels=1, num_classes=6, base_features=16):
        super().__init__()
        for name, value in (("in_channels", in_channels),
                            ("num_classes", num_classes),
                            ("base_features", base_features)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_features = base_features
        self.inc = DoubleConv(in_channels, base_features)
        self.down1 = Down(base_features, base_features * 2)
        self.down2 = Down(base_features * 2, base_features * 4)
        self.down3 = Down(base_features * 4, base_features * 8)
        self.down4 = Down(base_features * 8, base_features * 16)
        self.up1 = Up(base_features * 16, base_features * 8)
        self.up2 = Up(base_features * 8, base_features * 4)
        self.up3 = Up(base_features * 4, base_features * 2)
        self.up4 = Up(base_features * 2, base_features)
        self.outc = OutConv(base_features, num_classes)

    def forward(self, x):
        if x.ndim != 4 or x.size(1) != self.in_channels:
            raise ValueError(f"Expected [B,{self.in_channels},H,W], got {tuple(x.shape)}")
        if min(x.shape[-2:]) < 16:
            raise ValueError("Both spatial dimensions must be at least 16")
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
