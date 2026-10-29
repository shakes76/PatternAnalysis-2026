import torch
import torch.nn as nn
# U-Net Classifier
'''
DoubleConv for the U-net architecture is defined here.
It consists of two convolutional layers, each followed by a ReLU activation function.
'''
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)

'''
DownSample for the U-net architecture is defined here. It consists of a DoubleConv followed by a MaxPool operation.
The forward method returns both the output of the DoubleConv and the pooled output.
'''
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        pool = self.pool(down)
        return down, pool

'''
UpSample for the U-net architecture is defined here.
It consists of a ConvTranspose2d operation followed by a DoubleConv.
'''
class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], 1)
        return self.conv(x)

'''
U
'''
class UNetClassifier(nn.Module):
    def __init__(self, in_channels=1, out_channels=6):
        super().__init__()
        self.down1 = DownSample(in_channels, 32)
        self.down2 = DownSample(32, 64)
        self.down3 = DownSample(64, 128)

        self.bottleneck = DoubleConv(128, 256)

        self.up1 = UpSample(256, 128)
        self.up2 = UpSample(128, 64)
        self.up3 = UpSample(64, 32)

        self.output = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        down1, pool1 = self.down1(x)
        down2, pool2 = self.down2(pool1)
        down3, pool3 = self.down3(pool2)

        bottleneck = self.bottleneck(pool3)

        up1 = self.up1(bottleneck, down3)
        up2 = self.up2(up1, down2)
        up3 = self.up3(up2, down1)

        return self.output(up3)