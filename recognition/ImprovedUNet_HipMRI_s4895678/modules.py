"""Model components for HipMRI prostate segmentation.

Only PyTorch is imported here (no NumPy), as required by the task sheet.
Baseline: the 2D U-Net of Ronneberger et al. (2015), U-Net: Convolutional
Networks for Biomedical Image Segmentation, MICCAI.
Also the Improved U-Net (ImprovedUNet, dim=2 or 3): Isensee et al. (2018), with 3D
convolutions in the manner of Cicek et al. (2016), 3D U-Net.
"""
import torch
import torch.nn as nn


class DoubleConv2D(nn.Module):
    """Two 3x3 convolutions, each followed by batch norm and ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Apply both convolutions: [B, in_ch, H, W] -> [B, out_ch, H, W]."""
        return self.block(x)


class UNet2D(nn.Module):
    """Standard 2D U-Net. Input [B,1,H,W] -> logits [B,num_classes,H,W].

    H and W must be divisible by 2**depth (256 is fine for depth=4).

    Args: in_channels = image channels (1 for MRI), num_classes = labels (6), base = filters
    in the first level (doubled at every level), depth = number of downsampling steps.
    """

    def __init__(self, in_channels=1, num_classes=6, base=32, depth=4):
        super().__init__()
        chs = [base * 2 ** i for i in range(depth + 1)]   # e.g. 32,64,128,256,512

        # Encoder: DoubleConv blocks, halving the image between them
        self.enc = nn.ModuleList()
        prev = in_channels
        for c in chs[:-1]:
            self.enc.append(DoubleConv2D(prev, c))
            prev = c
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv2D(chs[-2], chs[-1])

        # Decoder: upsample, concatenate the skip connection, DoubleConv
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for c in reversed(chs[:-1]):
            self.up.append(nn.ConvTranspose2d(c * 2, c, 2, stride=2))
            self.dec.append(DoubleConv2D(c * 2, c))

        self.head = nn.Conv2d(chs[0], num_classes, 1)     # 1x1 conv -> class scores

    def forward(self, x):
        """Encoder, bottleneck, decoder with skip connections; returns class logits per pixel."""
        skips = []
        for block in self.enc:
            x = block(x)
            skips.append(x)          # saved for the skip connection
            x = self.pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([skip, x], dim=1))
        return self.head(x)


# ---------------------------------------------------------------------------
# Improved U-Net (Isensee et al., 2018, "Brain Tumor Segmentation and Radiomics
# Survival Prediction: Contribution to the BRATS 2017 Challenge").
# One class builds the 2D or 3D version through the `dim` argument.
# Simplification: our context module is post-activation (conv-norm-act), where
# the paper uses a pre-activation residual block.
# ---------------------------------------------------------------------------
def _layers(dim):
    """Return the (Conv, InstanceNorm, Dropout) layer classes for dim 2 or 3."""
    if dim == 2:
        return nn.Conv2d, nn.InstanceNorm2d, nn.Dropout2d
    if dim == 3:
        return nn.Conv3d, nn.InstanceNorm3d, nn.Dropout3d
    raise ValueError("dim must be 2 or 3")


class ConvNormAct(nn.Module):
    """Convolution -> instance norm -> leaky ReLU."""

    def __init__(self, in_ch, out_ch, dim, kernel=3, stride=1):
        super().__init__()
        Conv, Norm, _ = _layers(dim)
        self.block = nn.Sequential(
            Conv(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2, bias=False),
            Norm(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        """Convolution, instance norm and leaky ReLU; stride 2 halves each spatial size."""
        return self.block(x)


class ContextModule(nn.Module):
    """Two convolutions with dropout between them, plus a residual connection."""

    def __init__(self, ch, dim, p_drop=0.3):
        super().__init__()
        _, _, Drop = _layers(dim)
        self.body = nn.Sequential(
            ConvNormAct(ch, ch, dim), Drop(p_drop), ConvNormAct(ch, ch, dim))

    def forward(self, x):
        """Return the input plus the output of the two convolutions (same shape as the input)."""
        return x + self.body(x)          # residual: the input is added back


class ImprovedUNet(nn.Module):
    """Improved U-Net with stride-2 downsampling, context modules, localization
    modules and deep supervision (segmentation maps from the last decoder levels
    are upsampled and summed). Returns logits [B, num_classes, *spatial].

    Spatial sizes must be divisible by 2**(depth-1).

    Args: dim = 2 (slices) or 3 (volumes), in_channels = image channels, num_classes = labels,
    base = filters in the first level (doubled per level), depth = number of levels,
    p_drop = dropout probability inside the context modules.
    """

    def __init__(self, dim=2, in_channels=1, num_classes=6, base=32, depth=5, p_drop=0.3):
        super().__init__()
        Conv, _, _ = _layers(dim)
        self.depth = depth
        chs = [base * 2 ** i for i in range(depth)]

        # Encoder: stem, then (stride-2 conv + context module) per level
        self.stem = ConvNormAct(in_channels, chs[0], dim)
        self.ctx = nn.ModuleList([ContextModule(c, dim, p_drop) for c in chs])
        self.down = nn.ModuleList(
            [ConvNormAct(chs[i - 1], chs[i], dim, stride=2) for i in range(1, depth)])

        # Decoder: upsample + conv, concatenate skip, localization module
        self.up, self.loc = nn.ModuleList(), nn.ModuleList()
        for i in range(depth - 2, -1, -1):
            self.up.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                ConvNormAct(chs[i + 1], chs[i], dim)))
            self.loc.append(nn.Sequential(
                ConvNormAct(2 * chs[i], chs[i], dim),
                ConvNormAct(chs[i], chs[i], dim, kernel=1)))

        # Deep supervision: a 1x1 segmentation head on the last decoder levels
        self.n_seg = min(3, depth - 1)
        self.seg = nn.ModuleList([Conv(chs[i], num_classes, 1) for i in range(self.n_seg)])
        self.up_seg = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        """Encode with skips, decode with localization modules, and sum the segmentation
        maps of the last levels (deep supervision) into logits [B, num_classes, *spatial]."""
        x = self.ctx[0](self.stem(x))
        skips = [x]
        for i in range(1, self.depth):
            x = self.ctx[i](self.down[i - 1](x))
            skips.append(x)

        total = None
        for j, (up, loc) in enumerate(zip(self.up, self.loc)):
            level = self.depth - 2 - j
            x = loc(torch.cat([skips[level], up(x)], dim=1))
            if level < self.n_seg:
                s = self.seg[level](x)
                total = s if total is None else self.up_seg(total) + s
        return total
