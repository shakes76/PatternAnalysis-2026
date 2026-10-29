import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Convolution => BatchNorm => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
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


class UNet2D(nn.Module):
    def __init__(self, in_channels=1, out_classes=6):
        super().__init__()
        # Encoder
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        # Bottleneck
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)

        # Output mapper
        self.outc = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x):
        # Encoder passes
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder passes with concatenation (skip connections)
        u1 = self.up1(x5)
        u1 = torch.cat([x4, u1], dim=1)
        u1 = self.conv_up1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([x3, u2], dim=1)
        u2 = self.conv_up2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([x2, u3], dim=1)
        u3 = self.conv_up3(u3)

        u4 = self.up4(u3)
        u4 = torch.cat([x1, u4], dim=1)
        u4 = self.conv_up4(u4)

        logits = self.outc(u4)
        return logits

class DoubleConv3D(nn.Module):
    """(Convolution => BatchNorm => ReLU) * 2 for 3D volumetric data"""
    """ Identical to 2D except utilising 3D functions (Conv3D, BatchNorm3D)"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_classes=6, base_filters=16):
        super().__init__()
        """ Model architecture is identical to 2D """
        # Encoder
        self.inc = DoubleConv3D(in_channels, base_filters)
        self.down1 = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(base_filters, base_filters * 2))
        self.down2 = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(base_filters * 2, base_filters * 4))
        self.down3 = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(base_filters * 4, base_filters * 8))

        # Bottleneck
        self.down4 = nn.Sequential(nn.MaxPool3d(2), DoubleConv3D(base_filters * 8, base_filters * 16))

        # Decoder
        self.up1 = nn.ConvTranspose3d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv3D(base_filters * 16, base_filters * 8)

        self.up2 = nn.ConvTranspose3d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv3D(base_filters * 8, base_filters * 4)

        self.up3 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv3D(base_filters * 4, base_filters * 2)

        self.up4 = nn.ConvTranspose3d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv3D(base_filters * 2, base_filters)

        # Output mapper
        self.outc = nn.Conv3d(base_filters, out_classes, kernel_size=1)

    def forward(self, x):
        # Encoder passes
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder passes with concatenation (skip connections)
        u1 = self.up1(x5)
        u1 = torch.cat([x4, u1], dim=1)
        u1 = self.conv_up1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([x3, u2], dim=1)
        u2 = self.conv_up2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([x2, u3], dim=1)
        u3 = self.conv_up3(u3)

        u4 = self.up4(u3)
        u4 = torch.cat([x1, u4], dim=1)
        u4 = self.conv_up4(u4)

        logits = self.outc(u4)
        return logits