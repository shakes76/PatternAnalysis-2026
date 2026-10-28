"""
Model components for HipMRI prostate segmentation.

Contains:
    - UNet2D: standard 2D U-Net (Ronneberger et al., 2015), adapted from my own COMP3710 demo 2 (part4q2.py).
    - DiceLoss / CombinedLoss: losses for the heavily imbalanced prostate class.

Only PyTorch is used here (no NumPy), as required by the assessment spec.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# UNet Architecture
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Fundamental building block: Two consecutive convolutions to extract features, each followed by Batch Normalization and ReLU
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Encoder step: Halves the spatial dimensions using MaxPool, then applies the DoubleConv to extract deeper features
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Decoder step: Upsamples the feature map using transposed convolution to double the spatial dimensions
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Calculates padding needed in case the dimensions of the upsampled map and skip connection don't match perfectly
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Skip connection: Concatenates high-resolution spatial details from the encoder (x2) with the upsampled features (x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet2D(nn.Module):
    def __init__(self, n_channels=1, n_classes=6):
        super().__init__()
        # Defines the "U" shape: downsampling path (encoder) followed by upsampling path (decoder)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        # Output layer: Maps the final feature map to one logit per segmentation class
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Passes data down the encoder, saving intermediate outputs (x1 through x4) to use later for skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Passes data up the decoder, merging it with the saved encoder outputs at each step to recover spatial details
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


# ==========================================
# Losses
# ==========================================
class DiceLoss(nn.Module):
    """Soft multi-class Dice loss on softmax probabilities, averaged over classes.

    Every class counts equally regardless of size, so the tiny prostate class (<2% of pixels)
    is not drowned out by the background the way it is with cross-entropy alone.
    """

    def __init__(self, n_classes, smooth=1.0):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        # (N, H, W) integer labels -> (N, C, H, W) one-hot to match the probabilities
        one_hot = F.one_hot(targets, self.n_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)  # sum over batch and pixels, keep classes separate
        intersection = (probs * one_hot).sum(dims)
        cardinality = probs.sum(dims) + one_hot.sum(dims)
        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1. - dice.mean()


class CombinedLoss(nn.Module):
    """Cross-entropy (stable pixel-wise gradients) + Dice (class-balanced overlap)."""

    def __init__(self, n_classes, dice_weight=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(n_classes)
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        return self.ce(logits, targets) + self.dice_weight * self.dice(logits, targets)
