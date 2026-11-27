import torch
import torch.nn as nn
import torch.nn.functional as F


class ImpConvBlock(nn.Module):
    """Convolution -> InstanceNorm -> LeakyReLU.

    Padding is set to ``kernel_size // 2`` so that, with ``stride=1``, the
    spatial size is preserved (for odd kernel sizes). A stride of 2 halves the
    spatial resolution and is used for downsampling in the encoder.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Convolution kernel size. Default: 3.
        stride (int): Convolution stride. Default: 1.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, stride=stride, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        return x

class ContextModule(nn.Module):
    """Upsample by 2x (bilinear) then apply a 3x3 ``ImpConvBlock``.

    Using interpolation followed by a convolution (instead of a transposed
    convolution) avoids checkerboard artefacts.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, channels, p_drop=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm2d(channels, affine=True),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.Dropout2d(p=p_drop),
            nn.InstanceNorm2d(channels, affine=True),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        x = self.block(x)
        return x

class UpscaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ImpConvBlock(in_channels, out_channels),
        )

    def forward(self, x):
        x = self.block(x)
        return x

class LocalisationBlock(nn.Module):
    """Localisation module: a 3x3 conv followed by a 1x1 conv.

    Merges the concatenated skip / upsampled features and reduces the channel
    count. The 1x1 conv mixes channels without changing the spatial size.

    Args:
        in_channels (int): Number of input channels (skip + upsampled).
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            ImpConvBlock(in_channels, out_channels, kernel_size=3),
            ImpConvBlock(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        x = self.block(x)
        return x

class ImpDownBlock(nn.Module):
    """Encoder stage: (strided) conv followed by a residual context module.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride of the first conv; use 2 to halve the resolution
            and 1 to keep it. Default: 2.
        p_drop (float): Dropout probability inside the context module.
            Default: 0.3.
    """
    def __init__(self, in_channels, out_channels, stride=2, p_drop=0.3):
        super().__init__()
        self.imp_conv = ImpConvBlock(in_channels, out_channels, stride=stride)
        self.context = ContextModule(out_channels, p_drop)

    def forward(self, x):
        x = self.imp_conv(x)
        x = x + self.context(x)
        return x

class ImpUpBlock(nn.Module):
    """Decoder stage: upsample, concatenate the skip connection, then localise.

    Args:
        in_channels (int): Channels of the (lower-resolution) decoder input.
        skip_channel (int): Channels of the encoder skip feature map.
        out_channels (int): Channels produced by this stage.
        final (bool): If True, use a single ``ImpConvBlock`` instead of the
            full ``LocalisationBlock`` (used for the last, full-resolution
            stage). Default: False.
    """
    def __init__(self, in_channels, skip_channel, out_channels, final=False):

        super().__init__()

        self.up = UpscaleBlock(in_channels, out_channels)
        cat_channels = out_channels + skip_channel
        if final:
            self.loc = ImpConvBlock(cat_channels, out_channels)
        else:
            self.loc = LocalisationBlock(cat_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        x = self.loc(x)
        return x


class ImprovedUNet(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 6, base_channels = 16, softmax= False):
        """U-Net variant inspired by Isensee et al. (2017) with residual context modules and deep supervision.

        The encoder has five stages (the first at full resolution, the next four
        each halving the resolution) and the decoder has four stages. Segmentation
        maps from the last three decoder scales are combined by repeatedly
        upsampling and summing, so lower-resolution predictions contribute to the
        final full-resolution output.

        Args:
            in_channels (int): Number of input image channels.
            num_classes (int): Number of segmentation classes.
            base_channels (int): Channel width of the first stage
            softmax (bool): If True, apply softmax over the class dimension to the
                output.
        """


        super().__init__()
        self.softmax = softmax

        # Encoder:
        self.enc1 = ImpDownBlock(in_channels, base_channels, stride=1)
        self.enc2 = ImpDownBlock(base_channels, base_channels*2, stride=2)
        self.enc3 = ImpDownBlock(base_channels*2, base_channels*4, stride=2)
        self.enc4 = ImpDownBlock(base_channels*4, base_channels*8, stride=2)
        self.enc5 = ImpDownBlock(base_channels*8, base_channels*16, stride=2) # bottleneck

        # Decoder:
        self.dec1 = ImpUpBlock(base_channels*16, base_channels*8, base_channels*8)
        self.dec2 = ImpUpBlock(base_channels*8, base_channels*4, base_channels*4)
        self.dec3 = ImpUpBlock(base_channels*4, base_channels*2, base_channels*2)
        self.dec4 = ImpUpBlock(base_channels*2, base_channels, base_channels, final=True)

        # 1x1 convs mapping decoder features to class scores for deep supervision
        self.seg3 = nn.Conv2d(base_channels*4, num_classes, 1)
        self.seg2 = nn.Conv2d(base_channels*2, num_classes, 1)
        self.seg1 = nn.Conv2d(base_channels, num_classes, 1)
        self.upscale = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        # Encoder path
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        #Decoder path
        d1 = self.dec1(x5, x4)
        d2 = self.dec2(d1, x3)
        d3 = self.dec3(d2, x2)
        d4 = self.dec4(d3, x1)

        # Deep supervision:
        out = self.seg3(d2)
        out = self.upscale(out) + self.seg2(d3)
        out = self.upscale(out) + self.seg1(d4)

        if self.softmax:
            out = F.softmax(out, dim=1)

        return out

class DoubleConvBlock(nn.Module):
    """Two (3x3 conv -> BatchNorm -> ReLU) layers in sequence.

    Spatial size is preserved because of the padding of 1.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x

class DownBlock(nn.Module):
    """Encoder stage: 2x2 max pooling followed by a ``DoubleConvBlock``.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.d_block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.d_block(x)
        return x

class UpBlock(nn.Module):
    """Decoder stage: transposed conv, pad to match skip, concat, double conv.

    Args:
        in_channels (int): Channels of the (lower-resolution) decoder input.
            The transposed conv halves this to ``in_channels // 2``.
        skip_channels (int): Channels of the encoder skip feature map.
        out_channels (int): Channels produced by this stage.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConvBlock(in_channels//2 + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                      diff_y // 2, diff_y - diff_y // 2])

        x = self.conv(torch.cat([skip, x], dim=1))
        return x

class UNet2D(nn.Module):
    """Classic 2D U-Net with four down- and four up-sampling stages.

    Args:
        in_channels (int): Number of input image channels. Default: 1.
        num_classes (int): Number of segmentation classes. Default: 6.
        base_channels (int): Channel width of the first stage; deeper stages
            use x2, x4, x8 and x16 of this value. Default: 16.
    """
    def __init__(self, in_channels = 1, num_classes = 6, base_channels = 16):
        super().__init__()

        #Encoder
        self.double_conv = DoubleConvBlock(in_channels, base_channels)
        self.down1 = DownBlock(base_channels, base_channels*2)
        self.down2 = DownBlock(base_channels*2, base_channels*4)
        self.down3 = DownBlock(base_channels*4, base_channels*8)
        self.down4 = DownBlock(base_channels*8, base_channels*16)

        #Decoder
        self.up1 = UpBlock(base_channels*16, base_channels*8, base_channels*8)
        self.up2 = UpBlock(base_channels*8, base_channels*4, base_channels*4)
        self.up3 = UpBlock(base_channels*4, base_channels*2, base_channels*2)
        self.up4 = UpBlock(base_channels*2, base_channels, base_channels)

        self.outc = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.double_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits


def dice_loss(prediction, target, epsilon=1.0):
    """Soft multi-class Dice loss.

    Softmax is applied to ``prediction`` internally, so pass raw logits. The
    Dice score is computed per class over the batch and spatial dimensions,
    then averaged across classes.

    Args:
        prediction (Tensor): Logits of shape ``(N, C, H, W)``.
        target (Tensor): One-hot encoded ground truth of shape ``(N, C, H, W)``.
        epsilon (float): Smoothing term that avoids division by zero when a
            class is absent from both prediction and target. Default: 1.0.

    Returns:
        Tensor: Scalar loss, ``1 - mean Dice`` (0 is a perfect overlap).
    """
    prediction_prob = torch.softmax(prediction, dim=1)
    dimensions = (0, 2, 3)
    intersection = torch.sum(prediction_prob * target, dimensions)
    union = torch.sum(prediction_prob, dimensions) + torch.sum(target, dimensions)
    dice = (2.0 * intersection + epsilon) / (union + epsilon)
    dice_mean = dice.mean()

    return 1.0 - dice_mean

def dice_counts(prediction, target):
    """Per-class intersection and union counts using hard predictions.

    Useful for evaluation: accumulate these over a whole dataset and compute
    ``2 * inter / union`` at the end for a dataset-level Dice score per class.

    Args:
        prediction (Tensor): Logits or probabilities of shape ``(N, C, H, W)``.
        target (Tensor): One-hot encoded ground truth of shape ``(N, C, H, W)``.

    Returns:
        tuple[Tensor, Tensor]: ``(inter, union)``, each of shape ``(C,)``.
        ``union`` is the sum of predicted and true pixel counts (i.e. the
        Dice denominator), not the set union.
    """
    c = target.shape[1]
    index = prediction.argmax(dim=1)
    pred = (index.unsqueeze(1) == torch.arange(c, device=index.device).view(1, -1, 1, 1)).float()
    dims = (0, 2, 3)
    inter = (pred * target).sum(dims)
    union = pred.sum(dims) + target.sum(dims)
    return inter, union

def combined_loss(prediction, target):
    """Sum of cross-entropy and Dice loss.

    Cross-entropy gives stable per-pixel gradients, while Dice counteracts
    class imbalance by directly optimising region overlap.

    Args:
        prediction (Tensor): Logits of shape ``(N, C, H, W)``.
        target (Tensor): One-hot encoded ground truth of shape ``(N, C, H, W)``.

    Returns:
        Tensor: Scalar loss ``cross_entropy + dice_loss``.
    """
    ce = F.cross_entropy(prediction, target.argmax(dim=1))
    dice = dice_loss(prediction, target)
    return ce + dice
