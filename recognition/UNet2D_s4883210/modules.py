import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Baseline 2D U-Net architecture follows block structure described in Ronneberger et al (2015),
and the improved model was adapted from Isensee et al (2017). 
"""

# Baseline 2D U-Net

class StandardBlock(nn.Module):
    """Standard 3 x 3 conv -> ReLU -> 3 x 3conv -> ReLU block."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, features=(64, 128, 256, 512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # contracting path
        c = in_channels
        for f in features:
            self.downs.append(StandardBlock(c, f))
            c = f

        # bottleneck
        self.bottleneck = StandardBlock(features[-1], features[-1] * 2)

        # expansive path
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.ups.append(StandardBlock(f * 2, f))

        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)  # up-conv
            skip = skips[i // 2]
            
            # pad if odd spatial dims
            if x.shape != skip.shape:
                diff_h = skip.shape[2] - x.shape[2]
                diff_w = skip.shape[3] - x.shape[3]
                x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                              diff_h // 2, diff_h - diff_h // 2])
            
            x = torch.cat([skip, x], dim=1)
            x = self.ups[i + 1](x)

        return self.final(x)

# Improved U-Net (using Batch normalisation)
class ResidualBlock(nn.Module):
    """
    Residual block with batch normalisation.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
        )
        if in_ch != out_ch:
            # if channels dont match, use 1x1 conv to match dims
            self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.skip = nn.Identity()

        self.relu = nn.ReLU(inplace=True)
           

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))

class ImprovedUNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, features=(64, 128, 256, 512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # contracting paths
        c = in_channels
        for f in features:
            self.downs.append(ResidualBlock(c, f))
            c = f

        # bottleneck
        self.bottleneck = ResidualBlock(features[-1], features[-1] * 2)

        # expansive path
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.ups.append(ResidualBlock(f * 2, f))

        self.final = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # encoder 
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x) # bottleneck
        skips = skips[::-1] # skips for decoder

        # decoder
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i // 2]

            # add black padding if size doesnt match
            if x.shape != skip.shape:
                diff_h = skip.shape[2] - x.shape[2]
                diff_w = skip.shape[3] - x.shape[3]
                x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                                diff_h // 2, diff_h - diff_h // 2])

            x = torch.cat([skip, x], dim=1) # skip connection
            x = self.ups[i + 1](x)

        return self.final(x)





