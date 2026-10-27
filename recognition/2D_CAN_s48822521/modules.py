import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Apply two convolution, batch normalization, and LeakyReLU sequences.

    The convolutions preserve the input's spatial dimensions.

    Args:
        in_channels: Number of input feature channels.
        out_channels: Number of output feature channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), # padding=1 to preserve spatial size
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """Downsample a feature map with max pooling, then apply "DoubleConv".

    Args:
        in_channels: Number of input feature channels.
        out_channels: Number of output feature channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.max_pool = nn.MaxPool2d(2) # 2x2 max-pool
        self.conv_block = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        return self.conv_block(self.max_pool(x))

class UpBlock(nn.Module):
    """Upsample a feature map and combine it with an encoder skip connection.

    The combined features are processed by a "DoubleConv" block.

    Args:
        in_channels: Number of input feature channels.
        out_channels: Number of output feature channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2) # Stride = 2 to double spatial size
        self.conv_block = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x, skip_connection):
        x = self.up_conv(x)             # (B, out_channels, 2H, 2W)

        # From diagram, looks like the copied info is first, then we concatenate current info, x, onto its end. Same channels as x after up-convolution is applied, so we have double the out_channels.
        x = torch.cat([skip_connection, x], dim=1) # (B, out_channels*2, 2H, 2W)

        return self.conv_block(x)   # (B, out_channels, 2H, 2W)


class BasicUNet2D(nn.Module):
    """U-Net for 2D image segmentation.

    Uses an encoder, bottleneck, and decoder with skip connections to produce
    per-pixel class logits.

    Args:
        in_channels: Number of channels in each input image.
        base_channels: Number of feature channels in the first encoder stage.
        num_classes: Number of output segmentation classes.
    """

    def __init__(self, in_channels, base_channels, num_classes):
        super().__init__()

        # Encoder stuff
        self.down_layer1 = DoubleConv(in_channels, base_channels) # No max-pool at start
        self.down_layer2 = DownBlock(base_channels, base_channels*2) 
        self.down_layer3 = DownBlock(base_channels*2, base_channels*4)
        self.down_layer4 = DownBlock(base_channels*4, base_channels*8)

        self.down_layer_bottleneck = DownBlock(base_channels*8, base_channels*16) # At bottle-neck now

        # Decoder stuff
        self.up_layer4 = UpBlock(base_channels*16, base_channels*8)
        self.up_layer3 = UpBlock(base_channels*8, base_channels*4)
        self.up_layer2 = UpBlock(base_channels*4, base_channels*2)
        self.up_layer1 = UpBlock(base_channels*2, base_channels)

        # Final 1x1 convolution
        self.final_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        skip_connection1 = self.down_layer1(x)
        skip_connection2 = self.down_layer2(skip_connection1)
        skip_connection3 = self.down_layer3(skip_connection2)
        skip_connection4 = self.down_layer4(skip_connection3)

        bottleneck = self.down_layer_bottleneck(skip_connection4)

        # Decoder
        dec4 = self.up_layer4(bottleneck, skip_connection4)
        dec3 = self.up_layer3(dec4, skip_connection3)
        dec2 = self.up_layer2(dec3, skip_connection2)
        dec1 = self.up_layer1(dec2, skip_connection1)

        return self.final_conv(dec1)



############## 2D CAN ##############
class ContextModule(nn.Module):
    """Extract multi-scale context features using dilated convolutions (follows the paper).

    Produces per-pixel class logits while preserving the input's spatial
    dimensions. 

    Args:
        channels: Number of input and intermediate feature channels.
        num_classes: Number of output segmentation classes.
    """

    def __init__(self, channels, num_classes):
        super().__init__()

        # Context Module
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1) # Layer 1
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1) # Layer 2
        self.bn2 = nn.BatchNorm2d(channels)

        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2) # Layer 3
        self.bn3 = nn.BatchNorm2d(channels)

        self.conv4= nn.Conv2d(channels, channels, kernel_size=3, padding=4, dilation=4) # Layer 4
        self.bn4 = nn.BatchNorm2d(channels)

        self.conv5 = nn.Conv2d(channels, channels, kernel_size=3, padding=8, dilation=8) # Layer 5
        self.bn5 = nn.BatchNorm2d(channels)

        self.conv6 = nn.Conv2d(channels, channels, kernel_size=3, padding=16, dilation=16) # Layer 6
        self.bn6 = nn.BatchNorm2d(channels)

        self.conv7 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1) # Layer 7
        self.bn7 = nn.BatchNorm2d(channels)

        self.conv8 = nn.Conv2d(channels, num_classes, kernel_size=1) # Layer 8

        self.leaky_relu = nn.LeakyReLU(inplace=True)

        # Need the identity initialisation on all convolutions, as per the paper. 
        nn.init.dirac_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.dirac_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        nn.init.dirac_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)
        nn.init.dirac_(self.conv4.weight)
        nn.init.zeros_(self.conv4.bias)
        nn.init.dirac_(self.conv5.weight)
        nn.init.zeros_(self.conv5.bias)
        nn.init.dirac_(self.conv6.weight)
        nn.init.zeros_(self.conv6.bias)
        nn.init.dirac_(self.conv7.weight)
        nn.init.zeros_(self.conv7.bias)

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.leaky_relu(self.bn4(self.conv4(x)))
        x = self.leaky_relu(self.bn5(self.conv5(x)))
        x = self.leaky_relu(self.bn6(self.conv6(x)))
        x = self.leaky_relu(self.bn7(self.conv7(x)))
        return self.conv8(x) # Final layer doesn't have ReLU

class Basic2DCAN(nn.Module):
    """2D Context Aggregation Network for image segmentation.

    A convolutional front end extracts features, which are passed to a
    "ContextModule" to produce per-pixel class logits.

    Args:
        in_channels: Number of channels in each input image.
        base_channels: Width multiplier for the front-end convolutional layers.
        num_classes: Number of output segmentation classes.
    """
    def __init__(self, in_channels, base_channels, num_classes):
        super().__init__()

        self.front_end = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )

        self.context_module = ContextModule(base_channels*4, num_classes)

    def forward(self, x):
        feature_maps = self.front_end(x) # (B, C, 256, 128)
        return self.context_module(feature_maps) # (B, num_classes, 256, 128)