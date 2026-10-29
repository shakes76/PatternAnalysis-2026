"""Course-level 2D U-Net baseline and multiclass soft Dice loss.

Architecture basis: Ronneberger et al. (2015), arXiv:1505.04597, Fig. 1.
Adaptations: padded convolutions retain resolution; initial width 16 instead
of 64 for a controlled pilot. No pretrained weights or extra encoder.
The one-downsampling CAN3D adaptation is implemented in can3d.py and
re-exported here so both architectures have one public module entry point.
"""

import torch
from torch import nn
from can3d import CAN3D


class DoubleConv2D(nn.Module):
    """Two 3x3 convolution/ReLU operations with unchanged spatial size."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(),
        )

    def forward(self, image):
        return self.layers(image)


class UNet2D(nn.Module):
    """Four-level encoder/decoder with concatenated same-scale skip features.

    Input Bx1xHxW, raw output logits Bx6xHxW. H and W must be multiples
    of 16 for the four pooling operations. Apply softmax in the loss or
    prediction code, not twice in the model and loss.
    """

    def __init__(self, num_classes=6, base_channels=16):
        super().__init__()
        if num_classes < 2 or base_channels < 1:
            raise ValueError("Need at least two classes and a positive channel width.")
        widths = [base_channels * 2**i for i in range(5)]
        self.encoders = nn.ModuleList()
        in_channels = 1
        for width in widths[:-1]:
            self.encoders.append(DoubleConv2D(in_channels, width))
            in_channels = width
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv2D(widths[-2], widths[-1])
        self.upsamplers, self.decoders = nn.ModuleList(), nn.ModuleList()
        for high, low in zip(widths[::-1], widths[-2::-1]):
            self.upsamplers.append(nn.ConvTranspose2d(high, low, 2, stride=2))
            self.decoders.append(DoubleConv2D(low * 2, low))
        self.head = nn.Conv2d(base_channels, num_classes, 1)

    def forward(self, image):
        if (image.ndim != 4 or image.shape[1] != 1
                or any(size < 16 or size % 16 for size in image.shape[-2:])):
            raise ValueError("U-Net expects Bx1xHxW with H,W positive multiples of 16.")
        skips = []
        features = image
        for encoder in self.encoders:
            features = encoder(features)
            skips.append(features)
            features = self.pool(features)
        features = self.bottleneck(features)
        for up, decoder, skip in zip(self.upsamplers, self.decoders, reversed(skips)):
            features = decoder(torch.cat((up(features), skip), dim=1))
        return self.head(features)


class SoftDiceLoss(nn.Module):
    """Lecture-based 1 - mean class Dice, including background by default.

    For each class: (2*sum(p*y)+eps)/(sum(p)+sum(y)+eps).
    Sum over batch and spatial axes; then give each class equal weight.
    Inputs are raw logits and aligned one-hot targets of the same shape.
    This is not the CAN3D paper's specialised Dice Squared Focal loss.
    """

    def __init__(self, include_background=True, epsilon=1e-6):
        super().__init__()
        if epsilon <= 0:
            raise ValueError("Dice smoothing must be positive.")
        self.include_background = include_background
        self.epsilon = epsilon

    def forward(self, logits, target):
        if logits.shape != target.shape or logits.ndim not in (4, 5) or logits.shape[1] < 2:
            raise ValueError("Expected equal BxKxHxW or BxKxDxHxW tensors, K>=2.")
        probabilities = logits.softmax(dim=1)
        axes = (0, *range(2, logits.ndim))
        intersection = (probabilities * target).sum(dim=axes)
        total = probabilities.sum(dim=axes) + target.sum(dim=axes)
        dice = (2 * intersection + self.epsilon) / (total + self.epsilon)
        if not self.include_background:
            dice = dice[1:]
        return 1 - dice.mean()


class DiceCrossEntropyLoss(nn.Module):
    """One soft Dice loss plus one mean per-pixel cross-entropy loss.

    Both ingredients are covered in Contact 6/7. Combining them with unit
    coefficients is this project's controlled experiment, not a claim that
    the lecturer prescribed this exact recipe. Targets are aligned one-hot
    labels. CrossEntropyLoss receives raw logits, never softmax probabilities.
    No class weights or new network components are introduced.
    """

    def __init__(self):
        super().__init__()
        self.dice = SoftDiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        dice = self.dice(logits, target)
        return dice + self.cross_entropy(logits, target.argmax(dim=1))


def segmentation_loss(kind='soft-dice'):
    """Keep the original objective as the default for reproducible old runs."""
    if kind == 'soft-dice':
        return SoftDiceLoss()
    if kind == 'dice-ce':
        return DiceCrossEntropyLoss()
    raise ValueError(f'Unknown segmentation loss: {kind}')
