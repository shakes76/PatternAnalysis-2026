"""
Model components for 2D prostate segmentation on HipMRI slices.

UNet: baseline 2D U-Net (Ronneberger et al., 2015, "U-Net: Convolutional
Networks for Biomedical Image Segmentation"). Used as the benchmark that the
CAN is compared against on the same splits.

CAN: 2D Context Aggregation Network (Yu and Koltun, 2016, "Multi-scale
context aggregation by dilated convolutions"). Instead of shrinking and
re-growing the image like a U-Net, it stays at full resolution and widens
its view with dilated 3x3 convolutions whose dilation doubles each layer.

ResidualCAN: CAN variant with hybrid (non-doubling) dilations and residual
context layers that start as identities, to make the dilated context more
even (less gridding) and easier to train.

Only PyTorch is used here (no NumPy).
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two 3x3 convolutions, each followed by batch norm and ReLU."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Encoder step: 2x2 max-pool to halve resolution, then DoubleConv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """
    Decoder step: transposed conv to double resolution, concatenate the
    matching encoder features (skip connection), then DoubleConv.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """
    Baseline 2D U-Net.

    Four down/up levels, so input height and width must be divisible by 16
    (HipMRI slices are 256x128). Outputs raw logits (B, num_classes, H, W);
    softmax is applied in the loss and at inference.

    `base_channels` sets the width of the first level and doubles each level.
    """

    def __init__(self, in_channels=1, num_classes=6, base_channels=32):
        super().__init__()
        c = base_channels
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 16)
        self.up1 = Up(c * 16, c * 8)
        self.up2 = Up(c * 8, c * 4)
        self.up3 = Up(c * 4, c * 2)
        self.up4 = Up(c * 2, c)
        self.head = nn.Conv2d(c, num_classes, kernel_size=1)

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
        return self.head(x)


class DilatedConv(nn.Module):
    """3x3 convolution with the given dilation, then batch norm and ReLU."""

    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CAN(nn.Module):
    """
    2D Context Aggregation Network (Yu and Koltun, 2016).

    A front-end of two ordinary 3x3 convolutions extracts local features,
    then the context module stacks 3x3 convolutions with dilations
    (1, 1, 2, 4, 8, 16, 32, 1). Doubling the dilation grows the receptive
    field exponentially while the feature maps stay at full 256x128
    resolution, so no detail is lost to downsampling. The paper's context
    module ends at dilation 16 for its image size; dilation 32 is added so
    the receptive field (about 135 px) spans the prostate's surroundings in
    these slices. A 1x1 convolution gives per-pixel class logits.

    Batch norm and ReLU match the baseline U-Net, so the comparison isolates
    dilated full-resolution context versus encoder-decoder downsampling.
    """

    DILATIONS = (1, 1, 2, 4, 8, 16, 32, 1)

    def __init__(self, in_channels=1, num_classes=6, base_channels=64):
        super().__init__()
        c = base_channels
        self.front_end = nn.Sequential(
            DilatedConv(in_channels, c, dilation=1),
            DilatedConv(c, c, dilation=1),
        )
        self.context = nn.Sequential(*[DilatedConv(c, c, d) for d in self.DILATIONS])
        self.head = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x):
        return self.head(self.context(self.front_end(x)))


class ResidualDilatedConv(nn.Module):
    """
    Dilated 3x3 conv and batch norm added back to the input, then ReLU.

    The batch norm scale starts at zero (Goyal et al., 2017), so the block
    is an exact identity at initialisation. This plays the role of the
    identity initialisation that Yu and Koltun (2016) found necessary for
    training the context module.
    """

    def __init__(self, channels, dilation):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3,
                              padding=dilation, dilation=dilation, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        nn.init.zeros_(self.norm.weight)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.norm(self.conv(x)))


class ResidualCAN(nn.Module):
    """
    CAN variant aimed at the gridding artefact of doubling dilations.

    With dilations 1, 2, 4, 8, ... every pixel in the receptive field is
    reached, but very unevenly: some pixels are reached through far more
    paths than their neighbours (34 to 88 near the centre for the CAN), so
    the effective weighting is chequered. Following Hybrid Dilated
    Convolution (Wang et al., 2018), the dilations (1, 2, 5, 9, 17, 33) do
    not share a common spacing, which evens this out (44 to 76 paths). The
    last two layers step back down (2, 1) to smooth any remaining gridding,
    as in Dilated Residual Networks (Yu et al., 2017). Receptive field is about 145 px, close to the CAN's
    135 px, with the same number of layers and parameters.

    Context layers are residual and start as identities (see
    ResidualDilatedConv). Front-end and head match the CAN.
    """

    DILATIONS = (1, 2, 5, 9, 17, 33, 2, 1)

    def __init__(self, in_channels=1, num_classes=6, base_channels=64):
        super().__init__()
        c = base_channels
        self.front_end = nn.Sequential(
            DilatedConv(in_channels, c, dilation=1),
            DilatedConv(c, c, dilation=1),
        )
        self.context = nn.Sequential(*[ResidualDilatedConv(c, d) for d in self.DILATIONS])
        self.head = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x):
        return self.head(self.context(self.front_end(x)))


def count_parameters(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Shape check: one forward and backward pass on a dummy batch.
    x = torch.randn(2, 1, 256, 128)
    for model in (UNet(), CAN(), ResidualCAN()):
        out = model(x)
        out.mean().backward()
        print(f"{type(model).__name__} output {tuple(out.shape)}, "
              f"{count_parameters(model):,} parameters")
