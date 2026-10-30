
import torch
import torch.nn as nn

# 连续进行两次 3D 卷积	提取三维图像特征
class DoubleConv3D(nn.Module):
    """Two consecutive 3D convolution blocks."""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    """Standard 3D U-Net for six-class MRI segmentation."""

    # 创建 Encoder、Bottleneck、Decoder	定义网络的层和参数
    def __init__(self, in_channels=1, num_classes=6, base_channels=8):
        super().__init__()

        c = base_channels

        # Encoder
        self.enc1 = DoubleConv3D(in_channels, c)
        self.enc2 = DoubleConv3D(c, c * 2)
        self.enc3 = DoubleConv3D(c * 2, c * 4)

        self.pool = nn.MaxPool3d(kernel_size=2)

        # Bottleneck
        self.bottleneck = DoubleConv3D(c * 4, c * 8)

        # Decoder
        self.up3 = nn.ConvTranspose3d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv3D(c * 8, c * 4)

        self.up2 = nn.ConvTranspose3d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv3D(c * 4, c * 2)

        self.up1 = nn.ConvTranspose3d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = DoubleConv3D(c * 2, c)

        # Final segmentation layer
        self.output = nn.Conv3d(c, num_classes, kernel_size=1)

    # 数据依次经过网络，并融合 Skip Connections	计算 MRI 的分割预测
    def forward(self, x):
        # Encoder: extract features
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b = self.bottleneck(self.pool(e3))

        # Decoder: upsample and combine skip connections
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.output(d1)

# 使用随机数据进行 Forward、Loss、Backward 测试	确认模型结构和梯度计算正常
if __name__ == "__main__":
    # Small synthetic 3D MRI for a smoke test
    model = UNet3D()

    x = torch.randn(1, 1, 16, 32, 32)
    target = torch.randint(0, 6, (1, 16, 32, 32))

    # Forward
    output = model(x)

    # Loss and backward
    # CrossEntropyLoss 只用于测试梯度能否反向传播，并没有开始真正训练模型。正式训练逻辑会写在 train.py
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(output, target)
    loss.backward()

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Target shape:", target.shape)
    print("Loss:", loss.item())
    print("Forward and backward: OK")
