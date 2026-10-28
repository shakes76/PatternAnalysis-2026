"""
Modular neural network components for 2D HipMRI Prostate Segmentation.

Implements:
1. BaselineUNet2D: Standard 2D U-Net architecture (Ronneberger et al., 2015).
2. ImprovedUNet2D: Normal-difficulty 2D Improved U-Net featuring pre-activation
   residual context modules, strided convolution downsampling, localization
   modules, and multi-scale deep supervision aggregation (Isensee et al., 2018).

Note: Strictly adheres to COMP3710 rules — implemented solely in pure PyTorch
without any NumPy dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Part 1: Baseline 2D U-Net Components (Ronneberger et al., 2015)
# ============================================================================

class DoubleConv(nn.Module):
    """
    Standard U-Net building block: (3x3 Conv2d -> BatchNorm2d -> ReLU) * 2.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DownBlock(nn.Module):
    """
    Downsampling block for Baseline U-Net: 2x2 MaxPool2d followed by DoubleConv.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """
    Upsampling block for Baseline U-Net: Transposed convolution upsampling,
    skip-connection concatenation, and DoubleConv refinement.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Ensure exact spatial alignment if input dimensions are not divisible by 16
        if x1.shape[2:] != x2.shape[2:]:
            x1 = F.interpolate(
                x1, size=x2.shape[2:], mode="bilinear", align_corners=False
            )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class BaselineUNet2D(nn.Module):
    """
    Standard 2D U-Net baseline model for benchmarking spatial boundary precision
    on the HipMRI dataset.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        base_filters: int = 32
    ):
        super().__init__()
        f = base_filters

        # Encoder pathway
        self.inc = DoubleConv(in_channels, f)
        self.down1 = DownBlock(f, f * 2)
        self.down2 = DownBlock(f * 2, f * 4)
        self.down3 = DownBlock(f * 4, f * 8)
        self.down4 = DownBlock(f * 8, f * 16)

        # Decoder pathway
        self.up1 = UpBlock(f * 16, f * 8)
        self.up2 = UpBlock(f * 8, f * 4)
        self.up3 = UpBlock(f * 4, f * 2)
        self.up4 = UpBlock(f * 2, f)

        # Final 1x1 segmentation head (returns raw logits)
        self.outc = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        d4 = self.up1(x5, x4)
        d3 = self.up2(d4, x3)
        d2 = self.up3(d3, x2)
        d1 = self.up4(d2, x1)

        logits = self.outc(d1)
        return logits


# ============================================================================
# Part 2: 2D Improved U-Net Components (Isensee et al., 2018)
# ============================================================================

class ConvNormLReLU(nn.Module):
    """
    Helper block: Conv2d -> InstanceNorm2d -> LeakyReLU(negative_slope=1e-2).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ContextModule(nn.Module):
    """
    Pre-activation style residual context module from Isensee et al. (2018).
    Consists of two 3x3 convolutions with Instance Normalization, LeakyReLU,
    and spatial Dropout2d (p=0.3) in between, combined with a residual skip.
    """
    def __init__(self, channels: int, dropout_p: float = 0.3):
        super().__init__()
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.act1 = nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.dropout = nn.Dropout2d(p=dropout_p)

        self.norm2 = nn.InstanceNorm2d(channels, affine=True)
        self.act2 = nn.LeakyReLU(negative_slope=1e-2, inplace=True)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(self.act1(self.norm1(x)))
        out = self.dropout(out)
        out = self.conv2(self.act2(self.norm2(out)))
        return out + residual


class UpsamplingModule(nn.Module):
    """
    Upsampling module from Isensee et al. (2018):
    Bilinear interpolation (scale_factor=2) followed by a 3x3 convolution
    that halves the feature channels.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvNormLReLU(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(
        self,
        x: torch.Tensor,
        target_size: tuple[int, int]
    ) -> torch.Tensor:
        x_up = F.interpolate(
            x, size=target_size, mode="bilinear", align_corners=False
        )
        return self.conv(x_up)


class LocalizationModule(nn.Module):
    """
    Localization module from Isensee et al. (2018):
    A 3x3 convolution followed by a 1x1 convolution that halves the channel
    dimension and aggregates concatenated skip-connection features.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv3x3 = ConvNormLReLU(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv1x1 = ConvNormLReLU(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv3x3(x)
        x = self.conv1x1(x)
        return x


class ImprovedUNet2D(nn.Module):
    """
    2D Improved U-Net [Normal Difficulty] based on Isensee et al. (2018).

    Key architectural enhancements over Baseline U-Net:
    1. InstanceNorm2d + LeakyReLU(0.01) across all layers to handle MRI
       scan intensity variations.
    2. Residual Context Modules with spatial Dropout2d (p=0.3) for regularization.
    3. Strided 3x3 convolutions (stride=2) instead of max-pooling to preserve
       spatial gradients during downsampling.
    4. Multi-level Deep Supervision: 1x1 segmentation heads from the last three
       decoder stages are upsampled and summed element-wise to enforce coherent
       multi-scale anatomical representations.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 6,
        base_filters: int = 16,
        dropout_p: float = 0.3
    ):
        super().__init__()
        f = base_filters  # Default: 16 -> 32 -> 64 -> 128 -> 256

        # ---------------- Encoder Pathway ----------------
        # Level 1
        self.in_conv = ConvNormLReLU(in_channels, f, kernel_size=3, stride=1, padding=1)
        self.context1 = ContextModule(f, dropout_p=dropout_p)

        # Level 2
        self.down1 = ConvNormLReLU(f, f * 2, kernel_size=3, stride=2, padding=1)
        self.context2 = ContextModule(f * 2, dropout_p=dropout_p)

        # Level 3
        self.down2 = ConvNormLReLU(f * 2, f * 4, kernel_size=3, stride=2, padding=1)
        self.context3 = ContextModule(f * 4, dropout_p=dropout_p)

        # Level 4
        self.down3 = ConvNormLReLU(f * 4, f * 8, kernel_size=3, stride=2, padding=1)
        self.context4 = ContextModule(f * 8, dropout_p=dropout_p)

        # Level 5 (Bottleneck)
        self.down4 = ConvNormLReLU(f * 8, f * 16, kernel_size=3, stride=2, padding=1)
        self.context5 = ContextModule(f * 16, dropout_p=dropout_p)
        self.up4 = UpsamplingModule(f * 16, f * 8)

        # ---------------- Decoder Pathway ----------------
        # Decoder Level 4
        self.loc4 = LocalizationModule(f * 16, f * 8)
        self.up3 = UpsamplingModule(f * 8, f * 4)

        # Decoder Level 3 (Deep Supervision Head 1)
        self.loc3 = LocalizationModule(f * 8, f * 4)
        self.seg3 = nn.Conv2d(f * 4, num_classes, kernel_size=1)
        self.up2 = UpsamplingModule(f * 4, f * 2)

        # Decoder Level 2 (Deep Supervision Head 2)
        self.loc2 = LocalizationModule(f * 4, f * 2)
        self.seg2 = nn.Conv2d(f * 2, num_classes, kernel_size=1)
        self.up1 = UpsamplingModule(f * 2, f)

        # Decoder Level 1 (Final Resolution & Deep Supervision Head 3)
        self.dec_conv1 = ConvNormLReLU(f * 2, f * 2, kernel_size=3, stride=1, padding=1)
        self.seg1 = nn.Conv2d(f * 2, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.context1(self.in_conv(x))        # [B, f, H, W]
        e2 = self.context2(self.down1(e1))         # [B, 2f, H/2, W/2]
        e3 = self.context3(self.down2(e2))         # [B, 4f, H/4, W/4]
        e4 = self.context4(self.down3(e3))         # [B, 8f, H/8, W/8]
        b = self.context5(self.down4(e4))          # [B, 16f, H/16, W/16]

        # Decoder Stage 4
        u4 = self.up4(b, target_size=e4.shape[2:])
        d4 = self.loc4(torch.cat([u4, e4], dim=1))  # [B, 8f, H/8, W/8]

        # Decoder Stage 3
        u3 = self.up3(d4, target_size=e3.shape[2:])
        d3 = self.loc3(torch.cat([u3, e3], dim=1))  # [B, 4f, H/4, W/4]
        ds3 = self.seg3(d3)                         # [B, C, H/4, W/4]

        # Decoder Stage 2
        u2 = self.up2(d3, target_size=e2.shape[2:])
        d2 = self.loc2(torch.cat([u2, e2], dim=1))  # [B, 2f, H/2, W/2]
        ds2 = self.seg2(d2)                         # [B, C, H/2, W/2]

        # Deep Supervision Aggregation (Stage 3 -> Stage 2)
        ds3_up = F.interpolate(
            ds3, size=ds2.shape[2:], mode="bilinear", align_corners=False
        )
        ds2_sum = ds3_up + ds2

        # Decoder Stage 1
        u1 = self.up1(d2, target_size=e1.shape[2:])
        d1 = self.dec_conv1(torch.cat([u1, e1], dim=1))  # [B, 2f, H, W]
        ds1 = self.seg1(d1)                              # [B, C, H, W]

        # Final Deep Supervision Aggregation (Stage 2 -> Stage 1)
        ds2_up = F.interpolate(
            ds2_sum, size=ds1.shape[2:], mode="bilinear", align_corners=False
        )
        logits = ds2_up + ds1
        return logits

# ============================================================================
# Part 3: Loss Functions & Metrics (Pure PyTorch, Zero NumPy)
# ============================================================================

class MultiClassDiceLoss(nn.Module):
    """
    Soft Sørensen-Dice loss for multi-class segmentation with class-specific weights.
    Helps prevent gradient collapse when the prostate volume is <2% of the slice.
    """
    def __init__(self, smooth: float = 1e-5, weights: torch.Tensor = None):
        super().__init__()
        self.smooth = smooth
        self.weights = weights

    def forward(self, logits: torch.Tensor, targets_onehot: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_onehot, dim=dims)
        cardinality = torch.sum(probs + targets_onehot, dim=dims)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice_score

        if self.weights is not None:
            w = self.weights.to(logits.device)
            return torch.sum(dice_loss * w) / torch.sum(w)
        return torch.mean(dice_loss)


class CombinedLoss(nn.Module):
    """
    Weighted combination of Cross-Entropy Loss and Multi-Class Dice Loss.
    Combines smooth cross-entropy gradients with Dice direct overlap optimization.
    """
    def __init__(
        self,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        class_weights: torch.Tensor = None
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.class_weights = class_weights
        self.dice_loss = MultiClassDiceLoss(weights=class_weights)

    def forward(self, logits: torch.Tensor, targets_onehot: torch.Tensor) -> torch.Tensor:
        target_labels = torch.argmax(targets_onehot, dim=1)
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        ce = F.cross_entropy(logits, target_labels, weight=weight)
        dice = self.dice_loss(logits, targets_onehot)
        return self.ce_weight * ce + self.dice_weight * dice


def compute_dice_iou(
    preds_onehot: torch.Tensor,
    targets_onehot: torch.Tensor,
    eps: float = 1e-5
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes per-class Sørensen-Dice (DSC) and Intersection-over-Union (IoU).
    """
    dims = (0, 2, 3)
    intersection = torch.sum(preds_onehot * targets_onehot, dim=dims).float()
    cardinality = torch.sum(preds_onehot + targets_onehot, dim=dims).float()
    union = cardinality - intersection

    dices = (2.0 * intersection + eps) / (cardinality + eps)
    ious = (intersection + eps) / (union + eps)
    return dices, ious

if __name__ == "__main__":
    # Pure PyTorch smoke test for tensor shapes and parameter counts
    dummy_input = torch.randn(2, 1, 256, 256)

    baseline_model = BaselineUNet2D(in_channels=1, num_classes=6, base_filters=32)
    improved_model = ImprovedUNet2D(in_channels=1, num_classes=6, base_filters=16)

    out_base = baseline_model(dummy_input)
    out_impr = improved_model(dummy_input)

    base_params = sum(p.numel() for p in baseline_model.parameters() if p.requires_grad)
    impr_params = sum(p.numel() for p in improved_model.parameters() if p.requires_grad)

    print(f"[Smoke Test] Input shape          : {tuple(dummy_input.shape)}")
    print(f"[BaselineUNet2D] Output shape     : {tuple(out_base.shape)} | Params: {base_params:,}")
    print(f"[ImprovedUNet2D] Output shape     : {tuple(out_impr.shape)} | Params: {impr_params:,}")