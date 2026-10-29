import torch
import torch.nn as nn
import torch.nn.functional as F
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


#DownSample for the U-net architecture is defined here. It consists of a DoubleConv followed by a MaxPool operation.
#The forward method returns both the output of the DoubleConv and the pooled output.

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        pool = self.pool(down)
        return down, pool


#UpSample for the U-net architecture is defined here.
#It consists of a ConvTranspose2d operation followed by a DoubleConv.

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], 1)
        return self.conv(x)

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

   

'''
# AttentionGates
class AttentionGate(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, intermediate_channels):
        super().__init__()

        # Transform encoder features
        self.W_x = nn.Sequential(
            nn.Conv2d(
                encoder_channels,
                intermediate_channels,
                kernel_size=1
            ),
            nn.BatchNorm2d(intermediate_channels)
        )

        # Transform decoder features
        self.W_g = nn.Sequential(
            nn.Conv2d(
                decoder_channels,
                intermediate_channels,
                kernel_size=1
            ),
            nn.BatchNorm2d(intermediate_channels)
        )

        # Produce attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(
                intermediate_channels,
                1,
                kernel_size=1
            ),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        # x = encoder features
        # g = decoder gating features

        x1 = self.W_x(x)
        g1 = self.W_g(g)

        attention = self.relu(x1 + g1)
        attention = self.psi(attention)

        return x * attention

# U-Net Classifier

#DoubleConv for the U-net architecture is defined here.
#It consists of two convolutional layers, each followed by a ReLU activation function.


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.copy = nn.Conv2d(
            in_channels, out_channels, kernel_size=1
            ) if in_channels != out_channels else nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Comment out to test base U-Net
        residuals = self.copy(x)
        output = self.conv_op(x)
        # Comment out to test base U-Net
        return self.relu(output + residuals)
        #return self.relu(output)



#DownSample for the U-net architecture is defined here. It consists of a DoubleConv followed by a MaxPool operation.
#The forward method returns both the output of the DoubleConv and the pooled output.

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        pool = self.pool(down)
        return down, pool


#UpSample for the U-net architecture is defined here.
#It consists of a ConvTranspose2d operation followed by a DoubleConv.

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
        # Attention
        self.attention = AttentionGate(in_channels // 2, in_channels // 2, max(1, in_channels // 4))
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Attention
        x2 = self.attention(x2, x1)
        x = torch.cat([x2, x1], 1)
        return self.conv(x)


class UNetClassifier(nn.Module):
    def __init__(self, in_channels=1, out_channels=6):
        
        super().__init__()
        self.aux1 = nn.Conv2d(128, out_channels, kernel_size=1)
        self.aux2 = nn.Conv2d(64, out_channels, kernel_size=1)
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

        output = self.output(up3)
        
        if self.training:
            aux1 = self.aux1(up1)
            aux2 = self.aux2(up2)
            aux1 = F.interpolate(aux1, size=output.shape[2:], mode="bilinear", align_corners=False)
            aux2 = F.interpolate(aux2, size=output.shape[2:], mode="bilinear", align_corners=False)
            return output, aux1, aux2
            
        
        return output

