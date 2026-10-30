import torch
import torch.nn as nn

from modules import (
    FirstContext,
    DownContext,
    UpscaleBlock,
    LocalisationBlock,
)

class ImprovedUNet(nn.Module):
    """Improved 2D U-Net for MRI segmentation.
        Inspired by the segmentation architecture described
        by Isensee et al. (2018)
    """

    def __init__(self, in_channels=1, num_classes=6):
        super().__init__()

        # Encoder context pathway
        self.context1 = FirstContext(in_channels, 8)
        self.context2 = DownContext(8, 16)
        self.context3 = DownContext(16, 32)
        self.context4 = DownContext(32, 64)
        self.context5 = DownContext(64, 128)

        # Decoder upsampling pathway
        self.up4 = UpscaleBlock(128, 64)

        self.local4 = LocalisationBlock(128, 64)

        self.up3 = UpscaleBlock(64, 32)
        self.local3 = LocalisationBlock(64, 32)

        self.up2 = UpscaleBlock(32, 16)
        self.local2 = LocalisationBlock(32, 16)

        self.up1 = UpscaleBlock(16, 8)
        
        # Final decoder stage
        self.local1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(negative_slope=0.01)
        )

        # Final segmentation prediction
        self.out_conv = nn.Conv2d(16, num_classes, kernel_size=1)

        # Auxiliary segmentation heads for multi-scale prediction fusion
        self.aux3 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.aux2 = nn.Conv2d(32, num_classes, kernel_size=1)


    def forward(self, x):
        # Extract initial features
        x1, skip1 = self.context1(x)

        # Downsample through the encoder
        x2 = self.context2(x1)
        x3 = self.context3(x2)
        x4 = self.context4(x3)
        x5 = self.context5(x4)

        # Decoder with encoder skip connections

        # Level 4
        d4 = self.up4(x5)
        d4 = torch.cat((d4, x4), dim=1)
        d4, _ = self.local4(d4)

        # Level 3
        d3 = self.up3(d4)
        d3 = torch.cat((d3, x3), dim=1)
        d3, ds2_features = self.local3(d3)

        # Level 2
        d2 = self.up2(d3)
        d2 = torch.cat((d2, x2), dim=1)
        d2, ds3_features = self.local2(d2)

        # Level 1
        d1 = self.up1(d2)
        d1 = torch.cat((d1, skip1), dim=1)
        d1 = self.local1(d1)

        # Multi-scale prediction fusion

        # Intermediate prediction from level 3
        ds2 = self.aux3(ds2_features)
        ds2 = nn.functional.interpolate(
            ds2, scale_factor=2, mode="nearest"
        )

        # Intermediate prediction from level 2
        ds3 = self.aux2(ds3_features)

        # Combine intermediate predictions
        combined = ds2 + ds3

        # Upsample to full resolution
        combined = nn.functional.interpolate(
            combined, scale_factor=2, mode="nearest"
        )

        # Final prediction
        out = self.out_conv(d1)

        # Combine auxiliary predictions with the final segmentation logits
        out = out + combined

        return out


