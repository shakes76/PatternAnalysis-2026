"""
PyTorch architectures for 2D prostate segmentation on HipMRI:
- ImprovedUNet: 2D adaptation based on Isensee et al. (2018).
- BaselineUNet: Standard 2015 2D U-Net baseline for empirical comparison.
"""

from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContextModule(nn.Module):
    """Residual context block with InstanceNorm and LeakyReLU."""
    def __init__(self, channels: int, dropout_p: float = 0.3):
        super(ContextModule, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_p)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.lrelu1(self.in1(self.conv1(x)))
        out = self.dropout(out)
        out = self.lrelu2(self.in2(self.conv2(out)))
        return out + residual


class LocalizationModule(nn.Module):
    """Combines concatenated decoder features and halves channel depth."""
    def __init__(self, in_channels: int, out_channels: int):
        super(LocalizationModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.in1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.in2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.lrelu1(self.in1(self.conv1(x)))
        out = self.lrelu2(self.in2(self.conv2(out)))
        return out


class UpsampleBlock(nn.Module):
    """Bilinear 2x upsampling followed by 3x3 conv to reduce checkerboard artifacts."""
    def __init__(self, in_channels: int, out_channels: int):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.in_norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.lrelu(self.in_norm(self.conv(out)))
        return out


class ImprovedUNet(nn.Module):
    """
    2D Improved U-Net based on Isensee et al. (2018).
    Includes residual context modules, instance norm, and deep supervision heads.
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        base_filters: int = 16,
        dropout_p: float = 0.3
    ):
        super(ImprovedUNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_filters = base_filters

        # Encoder stages
        self.in_conv = nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1, bias=False)
        self.in_norm = nn.InstanceNorm2d(base_filters, affine=True)
        self.in_lrelu = nn.LeakyReLU(0.01, inplace=True)
        self.context1 = ContextModule(base_filters, dropout_p=dropout_p)

        self.down1 = nn.Conv2d(base_filters, base_filters * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.down1_norm = nn.InstanceNorm2d(base_filters * 2, affine=True)
        self.down1_lrelu = nn.LeakyReLU(0.01, inplace=True)
        self.context2 = ContextModule(base_filters * 2, dropout_p=dropout_p)

        self.down2 = nn.Conv2d(base_filters * 2, base_filters * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.down2_norm = nn.InstanceNorm2d(base_filters * 4, affine=True)
        self.down2_lrelu = nn.LeakyReLU(0.01, inplace=True)
        self.context3 = ContextModule(base_filters * 4, dropout_p=dropout_p)

        self.down3 = nn.Conv2d(base_filters * 4, base_filters * 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.down3_norm = nn.InstanceNorm2d(base_filters * 8, affine=True)
        self.down3_lrelu = nn.LeakyReLU(0.01, inplace=True)
        self.context4 = ContextModule(base_filters * 8, dropout_p=dropout_p)

        self.down4 = nn.Conv2d(base_filters * 8, base_filters * 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.down4_norm = nn.InstanceNorm2d(base_filters * 16, affine=True)
        self.down4_lrelu = nn.LeakyReLU(0.01, inplace=True)
        self.context5 = ContextModule(base_filters * 16, dropout_p=dropout_p)

        # Decoder stages
        self.up4 = UpsampleBlock(base_filters * 16, base_filters * 8)
        self.loc4 = LocalizationModule(base_filters * 16, base_filters * 8)

        self.up3 = UpsampleBlock(base_filters * 8, base_filters * 4)
        self.loc3 = LocalizationModule(base_filters * 8, base_filters * 4)

        self.up2 = UpsampleBlock(base_filters * 4, base_filters * 2)
        self.loc2 = LocalizationModule(base_filters * 4, base_filters * 2)

        self.up1 = UpsampleBlock(base_filters * 2, base_filters)
        self.loc1 = LocalizationModule(base_filters * 2, base_filters)

        # Auxiliary deep supervision heads
        self.seg_head3 = nn.Conv2d(base_filters * 4, num_classes, kernel_size=1)
        self.seg_head2 = nn.Conv2d(base_filters * 2, num_classes, kernel_size=1)
        self.seg_head1 = nn.Conv2d(base_filters, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.InstanceNorm2d, nn.BatchNorm2d)) and m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        enable_deep_supervision: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        # Encoder
        x1 = self.in_lrelu(self.in_norm(self.in_conv(x)))
        c1 = self.context1(x1)

        x2 = self.down1_lrelu(self.down1_norm(self.down1(c1)))
        c2 = self.context2(x2)

        x3 = self.down2_lrelu(self.down2_norm(self.down2(c2)))
        c3 = self.context3(x3)

        x4 = self.down3_lrelu(self.down3_norm(self.down3(c3)))
        c4 = self.context4(x4)

        x5 = self.down4_lrelu(self.down4_norm(self.down4(c4)))
        c5 = self.context5(x5)

        # Decoder
        u4 = self.up4(c5)
        l4 = self.loc4(torch.cat([u4, c4], dim=1))

        u3 = self.up3(l4)
        l3 = self.loc3(torch.cat([u3, c3], dim=1))

        u2 = self.up2(l3)
        l2 = self.loc2(torch.cat([u2, c2], dim=1))

        u1 = self.up1(l2)
        l1 = self.loc1(torch.cat([u1, c1], dim=1))

        out1 = self.seg_head1(l1)

        if enable_deep_supervision and self.training:
            out2 = self.seg_head2(l2)
            out3 = self.seg_head3(l3)
            out2_up = F.interpolate(out2, size=out1.shape[2:], mode="bilinear", align_corners=False)
            out3_up = F.interpolate(out3, size=out1.shape[2:], mode="bilinear", align_corners=False)
            return out1, out2_up, out3_up

        return out1


class DoubleConv(nn.Module):
    """Two 3x3 convs with BatchNorm and ReLU for standard U-Net."""
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaselineUNet(nn.Module):
    """Standard 2D U-Net baseline (Ronneberger et al., 2015)."""
    def __init__(self, in_channels: int = 1, num_classes: int = 1, base_filters: int = 16):
        super(BaselineUNet, self).__init__()
        self.inc = DoubleConv(in_channels, base_filters)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_filters, base_filters * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_filters * 2, base_filters * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_filters * 4, base_filters * 8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_filters * 8, base_filters * 16))

        self.up1 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(base_filters * 16, base_filters * 8)

        self.up2 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(base_filters * 8, base_filters * 4)

        self.up3 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(base_filters * 4, base_filters * 2)

        self.up4 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(base_filters * 2, base_filters)

        self.outc = nn.Conv2d(base_filters, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        x = self.conv_up1(torch.cat([x, x4], dim=1))

        x = self.up2(x)
        x = self.conv_up2(torch.cat([x, x3], dim=1))

        x = self.up3(x)
        x = self.conv_up3(torch.cat([x, x2], dim=1))

        x = self.up4(x)
        x = self.conv_up4(torch.cat([x, x1], dim=1))

        return self.outc(x)


if __name__ == "__main__":
    x = torch.randn(2, 1, 256, 128)

    baseline = BaselineUNet(in_channels=1, num_classes=1, base_filters=16)
    out_b = baseline(x)
    print(f"BaselineUNet output: {out_b.shape}, params: {sum(p.numel() for p in baseline.parameters()):,}")

    improved = ImprovedUNet(in_channels=1, num_classes=1, base_filters=16)
    improved.eval()
    out_i = improved(x)
    print(f"ImprovedUNet output: {out_i.shape}, params: {sum(p.numel() for p in improved.parameters()):,}")

    improved.train()
    heads = improved(x, enable_deep_supervision=True)
    print(f"Deep supervision heads: {[h.shape for h in heads]}")
