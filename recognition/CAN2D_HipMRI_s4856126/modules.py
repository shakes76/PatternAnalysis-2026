"""COMP3710 HipMRI segmentation architectures.

Contains the original Standard 2D U-Net baseline and a separate 2D
Context Aggregation Network (CAN). PyTorch only; no dataset I/O.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two 3x3 convolutions, each followed by BatchNorm and ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class Down(nn.Module):
    """Halve resolution and extract more features."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                    DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.layers(x)


class Up(nn.Module):
    """Upsample, concatenate corresponding encoder features, then convolve."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet2D(nn.Module):
    """Original 4-level Standard U-Net baseline; outputs per-class logits."""

    def __init__(self, in_channels=1, out_channels=6, base_channels=16):
        super().__init__()
        c = base_channels
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 16)
        self.up1 = Up(c * 16, c * 8, c * 8)
        self.up2 = Up(c * 8, c * 4, c * 4)
        self.up3 = Up(c * 4, c * 2, c * 2)
        self.up4 = Up(c * 2, c, c)
        self.out_conv = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out_conv(x)


class CANBlock(nn.Module):
    """A 3x3 dilated convolution which preserves H and W."""

    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, stride=1, padding=dilation,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class CAN2D(nn.Module):
    """2D context aggregation network for six-class HipMRI segmentation.

    Stacked dilated convolutions with rates 1, 1, 2, 4, 8, 16, 1, 1.
    No pooling or decoder: full input resolution is retained throughout.
    A final 1x1 convolution produces class logits (no Softmax).

    This is a compact educational CAN implementation, not a claimed
    reproduction of a specific published network's exact architecture.

    base_channels=16 => hidden_channels=64. The multiplier is a CAN-specific
    design decision; parameters and resources are measured rather than matched.
    """

    def __init__(self, in_channels=1, out_channels=6, base_channels=16):
        super().__init__()
        hidden = base_channels * 4
        dilations = (1, 1, 2, 4, 8, 16, 1, 1)
        layers = []
        current_channels = in_channels
        for rate in dilations:
            layers.append(CANBlock(current_channels, hidden, dilation=rate))
            current_channels = hidden
        self.context_layers = nn.Sequential(*layers)
        self.out_conv = nn.Conv2d(hidden, out_channels, kernel_size=1)

    def forward(self, x):
        return self.out_conv(self.context_layers(x))


if __name__ == "__main__":
    torch.set_num_threads(1)
    x = torch.randn(1, 1, 64, 64)
    y = torch.randint(0, 6, (1, 64, 64))
    for name, model in (("Basic U-Net", UNet2D(base_channels=8)),
                        ("CAN", CAN2D(base_channels=8))):
        model.train()
        predictions = model(x)
        assert predictions.shape == (1, 6, 64, 64)
        loss = nn.CrossEntropyLoss()(predictions, y)
        loss.backward()
        nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name}: output={tuple(predictions.shape)}, "
              f"loss={loss.item():.4f}, params={nparams}, backward=OK")
