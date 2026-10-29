import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    """
    Conv2d -> BatchNorm2d -> ReLU -> Dropout -> Conv2d -> BatchNorm2d -> ReLU
    """
    def __init__(self, in_channels, out_channels, drop_rate=0.0):
        super().__init__()
        self.block = nn.Sequential(
            # alter channel count
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=drop_rate),
            # refine feature extraction
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UpConv(nn.Module):
    """
    upConv -> +skip -> Conv2d -> BatchNorm2d -> ReLU -> Dropout -> Conv2d -> BatchNorm2d -> ReLU
    """
    def __init__(self, in_channels, out_channels, drop_rate=0.0):
        super().__init__()
        # Expand dimensions
        self.upConv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # Decrease channel count 
        self.block = Conv(out_channels*2, out_channels, drop_rate=drop_rate) #*2 since channels are doubled by cat

    def forward(self, x, skip):
        x = self.upConv(x)
        if x.shape != skip.shape:
            # resize x to match the HxW dimensions of skip
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1) # concatenate in dimension 1
        return self.block(x)

class UNet_2D(nn.Module):
    def __init__(self, in_channels, num_categories=5):
        """
        Basic UNet for 2D input
        in_channels(Int): number of channels in the input image
        out_channels(Int): number of channels in the bottleneck image
        """
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down1 = Conv(in_channels, 64)
        self.down2 = Conv(64, 128)
        self.down3 = Conv(128, 256)
        self.down4 = Conv(256, 512, drop_rate=0.2)

        self.bottleneck = Conv(512, 1024, drop_rate=0.2)

        self.up1 = UpConv(1024, 512, drop_rate=0.2)
        self.up2 = UpConv(512, 256)
        self.up3 = UpConv(256, 128)
        self.up4 = UpConv(128, 64)

        self.finalConv = nn.Conv2d(64, num_categories, kernel_size=1)

    def forward(self, x):
        # Downsampling: decrease dimensions, increase channels
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))

        b = self.bottleneck(self.pool(d4))

        # Upsampling: increase dimensions, decrease channels
        u1 = self.up1(b, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)
        
        return self.finalConv(u4) # Output has no. channels = no. categories

class DilatedConv(nn.Module):
    """
    Performs a dilated convolution -> InstanceNorm -> LeakyReLU
    """
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Conv_CAN(nn.Module):
    """
    Conv -> InstanceNorm -> leakyReLU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class CAN_2D(nn.Module):
    """
    2D Convolution Aggregate model with 9 dilation layers (1, 1, 2, 2, 3, 5, 8, 16)
    """
    def __init__(self, in_channels=1, out_channels=5):
        super().__init__()
        base_channel = 64

        self.initial = nn.Sequential(Conv_CAN(in_channels, base_channel),
                                     Conv_CAN(base_channel, base_channel))

        self.d1 = DilatedConv(base_channel, dilation=1)
        self.d2 = DilatedConv(base_channel, dilation=1)
        self.d3 = DilatedConv(base_channel, dilation=2)
        self.d4 = DilatedConv(base_channel, dilation=2)
        self.d5 = DilatedConv(base_channel, dilation=3)
        self.d6 = DilatedConv(base_channel, dilation=5)
        self.d7 = DilatedConv(base_channel, dilation=8)
        self.d8 = DilatedConv(base_channel, dilation=16)

        self.smooth = DilatedConv(base_channel, dilation=1)

        # 1x1 conv then decrease no.channels to equal no. segmentation categories
        self.final = nn.Sequential(Conv_CAN(base_channel, base_channel),
                                     nn.Conv2d(base_channel, out_channels, kernel_size=1))


    def forward(self, x):
        z = self.initial(x) # expand channels then
        # dilations
        z = self.d1(z)
        z = self.d2(z)
        z = self.d3(z)
        z = self.d4(z)
        z = self.d5(z)
        z = self.d6(z)
        z = self.d7(z)
        z = self.d8(z)
        
        z = self.smooth(z)

        # Decrease no.channels to equal no. segmentation categories
        return self.final(z)