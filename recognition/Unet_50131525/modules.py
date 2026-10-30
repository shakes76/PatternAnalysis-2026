import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.conv(x)

class ResidualBlock3D(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels,out_channels,kernel_size=3,padding=1),
            nn.BatchNorm3d(out_channels)
        )

        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels,out_channels,kernel_size=1)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        residual = self.shortcut(x)
        out = self.conv(x)
        return self.relu(out+residual)

class Down(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()

        self.down = nn.Sequential(
            nn.MaxPool3d(kernel_size = 2),
            ResidualBlock3D(in_channels,out_channels)
        )
    def forward(self,x):
        return self.down(x)

class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor = 2,mode = "trilinear", align_corners= False
        )

        self.conv = ResidualBlock3D(in_channels + skip_channels, out_channels)

    def forward(self,x,skip):
        x = self.up(x)

        x = torch.cat((x,skip), dim=1)

        x = self.conv(x)

        return x

class UNet3d(nn.Module):
    def __init__(self,in_channels = 1,num_classes = 6):
        super().__init__()

        #initial convolution
        self.inc = ResidualBlock3D(in_channels, 16)

        #encoder
        self.down1 = Down(16,32)
        self.down2 = Down(32,64)
        self.down3 = Down(64,128)

        #bottleneck
        self.down4 = Down(128,256)

        #decoder
        self.up1 = Up(256,128,128)
        self.up2 = Up(128,64,64)
        self.up3 = Up(64,32,32)
        self.up4 = Up(32,16,16)

        #output layer
        self.outc = nn.Conv3d(16,num_classes,kernel_size=1)
        self.aux2 = nn.Conv3d(64,num_classes,kernel_size=1)
        self.aux3 = nn.Conv3d(32,num_classes,kernel_size=1)

    def forward(self,x):
        #encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        #decoder
        u1 = self.up1(x5,x4)
        u2 = self.up2(u1,x3)
        u3 = self.up3(u2,x2)
        u4 = self.up4(u3,x1)

        #main segmentation output
        logits = self.outc(u4)

        #Auxiliary segmentation outputs
        aux2 = self.aux2(u2)
        aux3 = self.aux3(u3)

        if self.training:
            return logits,aux2,aux3

        return logits

#Temporarily testing block
if __name__ == "__main__":
    model = UNet3d(in_channels=1, num_classes=6)
    x = torch.randn(1, 1, 32, 32, 32)

    model.train()

    with torch.no_grad():
        outputs = model(x)

    for i, output in enumerate(outputs):
        print(f"Output {i}: {output.shape}")

    model.eval()

    with torch.no_grad():
        prediction = model(x)

    print("Evaluation output:", prediction.shape)