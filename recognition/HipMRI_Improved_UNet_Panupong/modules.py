"""Model components for HipMRI 2D prostate segmentation (PyTorch only).

Two networks are compared on the same data, loss and training budget:

* ``UNet`` - the standard 2D U-Net baseline (Ronneberger et al., 2015),
  using padded 3x3 convolutions so the output has the input size.
* ``ImprovedUNet`` - a 2D adaptation of the Improved U-Net of Isensee et al.
  (2018), "Brain Tumor Segmentation and Radiomics Survival Prediction:
  Contribution to the BRATS 2017 Challenge", Section 2.1 and Fig. 1.
  Every 3x3x3 / 1x1x1 convolution of the paper becomes a 3x3 / 1x1
  convolution. Levels, filter counts, module types and the deep-supervision
  wiring follow the paper. Details the paper does not specify are marked
  "Assumption" below: the norm/activation order outside the context
  modules, the 1x1 kernel of the segmentation layers and the width of the
  3x3 convolution inside the localization module.

``MulticlassSoftDiceLoss`` implements Eq. 1 of Isensee et al. and is used
for both networks so that only the architecture differs.

Both networks take a (B, 1, H, W) image with H and W divisible by 16 and
return raw class scores (logits) of shape (B, num_classes, H, W); softmax is
applied inside the loss and argmax gives the predicted label map.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Baseline: standard 2D U-Net
# ---------------------------------------------------------------------------

class DoubleConv(nn.Sequential):
    """Two 3x3 convolutions, each followed by batch normalisation and ReLU.

    The original U-Net uses unpadded convolutions and no normalisation. Here
    ``padding=1`` keeps the spatial size (as recommended for this course),
    and batch normalisation is added, as in common modern U-Net baselines,
    to make Adam training stable. Both deviations are listed in the README.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    """Standard 2D U-Net with 4 down/up-sampling steps.

    Encoder: DoubleConv, then (2x2 max-pool + DoubleConv) four times, doubling
    the channels each time (64, 128, 256, 512, 1024 by default).
    Dropout is applied to the bottleneck features, following "drop-out layers
    at the end of the contracting path" in the original paper (rate 0.5).
    Decoder: (2x2 transposed convolution that halves the channels, concatenate
    the encoder features of the same size, DoubleConv) four times.
    Head: 1x1 convolution to ``num_classes`` scores.
    """

    def __init__(self, in_channels=1, num_classes=6, base_channels=64, depth=4, p_drop=0.5):
        super().__init__()
        self.dropout = nn.Dropout(p_drop)
        channels = [base_channels * 2 ** level for level in range(depth + 1)]

        self.encoders = nn.ModuleList([DoubleConv(in_channels, channels[0])])
        for level in range(1, depth + 1):
            self.encoders.append(DoubleConv(channels[level - 1], channels[level]))
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for level in range(depth, 0, -1):
            self.upsamplers.append(
                nn.ConvTranspose2d(channels[level], channels[level - 1], kernel_size=2, stride=2))
            # After concatenation the decoder sees 2 x channels[level - 1].
            self.decoders.append(DoubleConv(2 * channels[level - 1], channels[level - 1]))

        self.head = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        skips = []
        for level, encoder in enumerate(self.encoders):
            if level > 0:
                x = self.pool(x)
            x = encoder(x)
            skips.append(x)

        x = self.dropout(skips.pop())  # bottleneck features
        for upsample, decoder in zip(self.upsamplers, self.decoders):
            x = upsample(x)
            x = decoder(torch.cat([skips.pop(), x], dim=1))
        return self.head(x)


# ---------------------------------------------------------------------------
# Improved U-Net (Isensee et al., 2018), 2D adaptation
# ---------------------------------------------------------------------------

NEGATIVE_SLOPE = 1e-2  # "leaky ReLU nonlinearities with a negative slope of 10^-2"


class ConvNormAct(nn.Sequential):
    """Convolution -> instance normalisation -> leaky ReLU.

    Used for every feature-computing convolution outside the context modules.
    The paper uses instance normalisation instead of batch normalisation and
    leaky ReLU (slope 0.01) for all such convolutions. Assumption: the paper
    does not say where the normalisation sits, so the common
    convolution -> norm -> activation order is used here.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                      padding=kernel_size // 2),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(NEGATIVE_SLOPE, inplace=True),
        )


class ContextModule(nn.Module):
    """Pre-activation residual block with two 3x3 convolutions and dropout.

    "Each context module is in fact a pre-activation residual block with two
    3x3x3 convolutional layers and a dropout layer (p_drop = 0.3) in between."
    Pre-activation (He et al., 2016) means norm -> activation -> convolution.
    The block output is added to its input (the identity shortcut), which is
    the element-wise sum shown in Fig. 1 of the paper.
    """

    def __init__(self, channels, p_drop=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm2d(channels, affine=True),
            nn.LeakyReLU(NEGATIVE_SLOPE),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Dropout(p_drop),
            nn.InstanceNorm2d(channels, affine=True),
            nn.LeakyReLU(NEGATIVE_SLOPE),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class UpsamplingModule(nn.Sequential):
    """Nearest-neighbour x2 upscale followed by a 3x3 convolution that halves the channels.

    "a simple upscale that repeats the feature voxels twice in each spatial
    dimension, followed by a 3x3x3 convolution that halves the number of
    feature maps" - chosen by the authors instead of transposed convolution.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvNormAct(in_channels, out_channels, kernel_size=3),
        )


class LocalizationModule(nn.Sequential):
    """3x3 convolution followed by a 1x1 convolution that halves the channels.

    "A localization module consists of a 3x3x3 convolution followed by a
    1x1x1 convolution that halves the number of feature maps."
    Assumption: the 3x3 convolution keeps the concatenated width and only the
    1x1 convolution halves it; the module outputs match Fig. 1.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__(
            ConvNormAct(in_channels, in_channels, kernel_size=3),
            ConvNormAct(in_channels, out_channels, kernel_size=1),
        )


class ImprovedUNet(nn.Module):
    """2D Improved U-Net with deep supervision (Isensee et al., 2018, Fig. 1).

    Context (encoder) pathway, 5 levels with 16, 32, 64, 128, 256 filters:
      level 1: 3x3 convolution (16) + context module;
      levels 2-5: 3x3 stride-2 convolution (doubles the filters, halves the
      size) + context module.
    Localization (decoder) pathway, levels 4 to 1:
      upsampling module, concatenation with the encoder features of the same
      level, then a localization module (levels 4-2) or a 3x3 convolution
      with 32 filters (level 1).
    Deep supervision: segmentation layers at decoder levels 3, 2 and 1
    (assumption: 1x1 convolutions; the paper does not give the kernel size).
    The lower-resolution map is upscaled (nearest) and added element-wise to
    the next one; the final sum is the network output (softmax in the loss).
    """

    def __init__(self, in_channels=1, num_classes=6, base_channels=16, p_drop=0.3):
        super().__init__()
        c = [base_channels * 2 ** level for level in range(5)]  # 16, 32, 64, 128, 256

        # Context pathway.
        self.stem = ConvNormAct(in_channels, c[0], kernel_size=3)
        self.downsamplers = nn.ModuleList(
            [ConvNormAct(c[level - 1], c[level], kernel_size=3, stride=2) for level in range(1, 5)])
        self.contexts = nn.ModuleList([ContextModule(c[level], p_drop) for level in range(5)])

        # Localization pathway (from the bottleneck at level 5 up to level 1).
        self.upsamplers = nn.ModuleList(
            [UpsamplingModule(c[level], c[level - 1]) for level in range(4, 0, -1)])
        self.localizations = nn.ModuleList(
            [LocalizationModule(2 * c[level], c[level]) for level in range(3, 0, -1)])
        self.final_conv = ConvNormAct(2 * c[0], 2 * c[0], kernel_size=3)  # 32 filters

        # Segmentation layers for deep supervision (decoder levels 3, 2 and 1).
        self.seg_level3 = nn.Conv2d(c[2], num_classes, kernel_size=1)
        self.seg_level2 = nn.Conv2d(c[1], num_classes, kernel_size=1)
        self.seg_level1 = nn.Conv2d(2 * c[0], num_classes, kernel_size=1)
        self.upscale = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        # Context pathway: keep each level's output for the skip connections.
        skips = []
        x = self.contexts[0](self.stem(x))
        skips.append(x)
        for downsample, context in zip(self.downsamplers, self.contexts[1:]):
            x = context(downsample(x))
            skips.append(x)
        skips.pop()  # the level-5 output is the bottleneck, not a skip

        # Localization pathway; collect the features used for segmentation.
        seg_features = []
        for step, upsample in enumerate(self.upsamplers):
            x = torch.cat([skips.pop(), upsample(x)], dim=1)
            if step < 3:
                x = self.localizations[step](x)   # levels 4, 3, 2
                if step >= 1:
                    seg_features.append(x)        # levels 3 and 2
            else:
                x = self.final_conv(x)            # level 1
                seg_features.append(x)

        # Deep supervision: upscale and sum the segmentation maps.
        level3, level2, level1 = seg_features
        out = self.upscale(self.seg_level3(level3)) + self.seg_level2(level2)
        out = self.upscale(out) + self.seg_level1(level1)
        return out


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class MulticlassSoftDiceLoss(nn.Module):
    """Multiclass soft Dice loss, Eq. 1 of Isensee et al. (2018).

        L = -(2 / |K|) * sum_k [ sum_i u_ik v_ik / (sum_i u_ik + sum_i v_ik) ]

    ``u`` is the softmax of the logits and ``v`` the one-hot target. All
    classes, including background, are included. The sums over ``i`` run
    over every pixel of every slice in the batch (the paper sums over the
    voxels of one training patch): on the many slices without prostate a
    per-slice prostate term would be a constant 0 with no gradient. ``eps``
    (not specified in the paper) avoids division by zero. The loss lies in
    [-1, 0]; -1 is a perfect overlap for every class.
    """

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target_one_hot):
        # Compute in float32 even under mixed precision for a stable softmax.
        probs = torch.softmax(logits.float(), dim=1)
        target = target_one_hot.float()
        dims = (0, 2, 3)  # batch and spatial dimensions; keep the class dimension
        intersection = (probs * target).sum(dims)
        denominator = probs.sum(dims) + target.sum(dims)
        return -2.0 * (intersection / (denominator + self.eps)).mean()


def build_model(name, num_classes=6, in_channels=1):
    """Create a model by name: ``"unet"`` (baseline) or ``"improved"``.

    ``in_channels`` is 1 for 2D slices and 2K + 1 for 2.5D input with K
    neighbouring slices on each side; only the first convolution changes.
    """
    if name == "unet":
        return UNet(in_channels=in_channels, num_classes=num_classes)
    if name == "improved":
        return ImprovedUNet(in_channels=in_channels, num_classes=num_classes)
    raise ValueError(f"Unknown model {name!r}; use 'unet' or 'improved'")


def count_parameters(model):
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
