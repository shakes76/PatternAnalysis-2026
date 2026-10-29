"""One-downsampling CAN3D adaptation, based on Dai et al. (2021), Sec.3.1.3.

Source: https://arxiv.org/abs/2109.05443 (v2, Figs.1/2, PDF pp.5-7).
The methods text specifies 8 -> 36 -> 48 channels; Fig.2 illustrates a
different two-downsampling, 3 -> 6 -> 36 -> 48 variant. This project uses
the text's one-downsampling option for resized 64x128x128 whole volumes.
It does not claim an exact reproduction of the Fig.2 channel layout.
"""

import torch
from torch import nn


class AdaptiveInstanceNorm3D(nn.Module):
    """Paper Eq.3: learned scalar a*x + b*IN(x), then activation elsewhere.

    Initialization a=1,b=0 and non-affine IN (epsilon 1e-5) are explicit
    project choices, because these details are unspecified in the paper.
    """

    def __init__(self, channels):
        super().__init__()
        self.a = nn.Parameter(torch.ones(()))
        self.b = nn.Parameter(torch.zeros(()))
        self.instance = nn.InstanceNorm3d(channels, affine=False, eps=1e-5)

    def forward(self, value):
        return self.a*value + self.b*self.instance(value)


class ConvBlock3D(nn.Module):
    """3D convolution -> adaptive instance normalization -> LeakyReLU(0.1)."""

    def __init__(self, in_channels, out_channels, kernel=3, stride=1,
                 dilation=1, identity=False, transpose=False):
        super().__init__()
        if transpose:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel,
                                           stride=2, padding=1, output_padding=1)
        else:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel,
                                  stride=stride, padding=dilation*(kernel//2),
                                  dilation=dilation)
        if identity:
            # Center taps copy shared channels; additional output channels
            # start at zero. This also defines the rectangular 36 -> 48 case.
            nn.init.dirac_(self.conv.weight)
        else:
            nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        self.norm = AdaptiveInstanceNorm3D(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, value):
        return self.activation(self.norm(self.conv(value)))


class CAN3D(nn.Module):
    """Whole-volume Bx1xDxHxW -> Bx6xDxHxW raw class logits.

    One stride-2 downsample; context dilation 2,4,8,1; transposed
    upsampling; element-wise shortcut sum, then 1x1x1 segmentation head.
    The 48->8 upsampling width and 8-channel shortcut are adaptation choices
    making the methods-text variant's sum explicit. No patch/crop inference.
    """

    def __init__(self, num_classes=6):
        super().__init__()
        if num_classes < 2:
            raise ValueError('Need at least two classes.')
        self.stem = ConvBlock3D(1, 8)
        self.shortcut = ConvBlock3D(8, 8, kernel=1)
        self.down = ConvBlock3D(8, 36, stride=2)
        self.context = nn.Sequential(
            ConvBlock3D(36, 36, dilation=2, identity=True),
            ConvBlock3D(36, 36, dilation=4, identity=True),
            ConvBlock3D(36, 36, dilation=8, identity=True),
            ConvBlock3D(36, 48, identity=True),
        )
        self.up = ConvBlock3D(48, 8, transpose=True)
        self.head = nn.Conv3d(8, num_classes, 1)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, image):
        if image.ndim != 5 or image.shape[1] != 1 or any(n < 4 or n % 2 for n in image.shape[2:]):
            raise ValueError('CAN3D expects Bx1xDxHxW with even spatial sizes >=4.')
        stem = self.stem(image)
        return self.head(self.up(self.context(self.down(stem))) + self.shortcut(stem))
