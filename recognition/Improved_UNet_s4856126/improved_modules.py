
"""Improved 2D U-Net with Residual Blocks for HipMRI segmentation."""

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    """Two convolutions with a residual shortcut."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


class ResidualDown(nn.Module):
    """Downsampling block."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            ResidualBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.block(x)


class ResidualUp(nn.Module):
    """Upsampling block with U-Net skip connection."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2
        )

        self.block = ResidualBlock(
            out_channels + skip_channels,
            out_channels
        )

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        x = torch.cat([skip, x], dim=1)

        return self.block(x)


class ImprovedUNet2D(nn.Module):
    """Residual U-Net for 6-class segmentation."""

    def __init__(
        self,
        in_channels=1,
        out_channels=6,
        base_channels=16
    ):
        super().__init__()

        c = base_channels

        # Encoder
        self.inc = ResidualBlock(in_channels, c)
        self.down1 = ResidualDown(c, c * 2)
        self.down2 = ResidualDown(c * 2, c * 4)
        self.down3 = ResidualDown(c * 4, c * 8)
        self.down4 = ResidualDown(c * 8, c * 16)

        # Decoder
        self.up1 = ResidualUp(c * 16, c * 8, c * 8)
        self.up2 = ResidualUp(c * 8, c * 4, c * 4)
        self.up3 = ResidualUp(c * 4, c * 2, c * 2)
        self.up4 = ResidualUp(c * 2, c, c)

        # Output layer
        self.outc = nn.Conv2d(
            c, out_channels, kernel_size=1
        )

    def forward(self, x):

        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)


if __name__ == "__main__":

    model = ImprovedUNet2D(
        in_channels=1,
        out_channels=6,
        base_channels=16
    )

    model.eval()

    x = torch.randn(1, 1, 256, 128)

    with torch.no_grad():
        y = model(x)

    parameters = sum(
        p.numel() for p in model.parameters()
    )

    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(y.shape))
    print("Parameters:", parameters)

    assert y.shape == (1, 6, 256, 128)

    print("Forward test: PASSED")
