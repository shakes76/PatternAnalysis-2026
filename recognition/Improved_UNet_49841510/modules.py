import torch
import torch.nn as nn


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)

class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.layers(x)

class Up(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=True
        )

        self.conv = DoubleConv(
            in_channels + skip_channels,
            out_channels
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=6):
        super().__init__()

        self.inc = DoubleConv(in_channels, 16)

        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 256)

        self.up1 = Up(256, 128, 128)
        self.up2 = Up(128, 64, 64)
        self.up3 = Up(64, 32, 32)
        self.up4 = Up(32, 16, 16)

        self.outc = OutConv(16, num_classes)

    def forward(self, x):
        #Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits


class AttentionGate(nn.Module):##Improved 2D-UNet Begins

    def __init__(self, skip_channels, gating_channels, inter_channels):
        super().__init__()

        self.W_x = nn.Conv2d(
            skip_channels,
            inter_channels,
            kernel_size=1
        )

        self.W_g = nn.Conv2d(
            gating_channels,
            inter_channels,
            kernel_size=1
        )

        self.relu = nn.ReLU(inplace=True)

        self.psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x, g):

        x_projected = self.W_x(x)
        g_projected = self.W_g(g)

        combined = self.relu(x_projected + g_projected)

        alpha = self.psi(combined)

        return x * alpha

class AttentionUp(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=True
        )
        self.attention = AttentionGate(
            skip_channels=skip_channels,
            gating_channels=in_channels,
            inter_channels=max(1, skip_channels // 2)
        )
        self.conv = DoubleConv(
            in_channels + skip_channels,
            out_channels
        )

    def forward(self, x1, x2):

        # 1. Upsample decoder feature
        x1 = self.up(x1)

        # 2. Apply attention to encoder skip feature
        x2 = self.attention(x2, x1)

        # 3. Concatenate features
        x = torch.cat([x2, x1], dim=1)

        # 4. Fuse features
        return self.conv(x)

class AttentionUNet(nn.Module):

    def __init__(self, in_channels=1, num_classes=6):
        super().__init__()
        self.inc = DoubleConv(in_channels, 16)

        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        self.down4 = Down(128, 256)
        self.up1 = AttentionUp(256, 128, 128)
        self.up2 = AttentionUp(128, 64, 64)
        self.up3 = AttentionUp(64, 32, 32)
        self.up4 = AttentionUp(32, 16, 16)

        self.outc = OutConv(16, num_classes)

    def forward(self, x):

        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )
        self.relu = nn.ReLU(inplace=True)

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1
            )

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        out = out + identity

        out = self.relu(out)

        return out

class ResidualDown(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.conv = ResidualBlock(
            in_channels,
            out_channels
        )

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)

class ResidualUp(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=True
        )

        self.conv = ResidualBlock(
            in_channels + skip_channels,
            out_channels
        )

    def forward(self, x1, x2):

        x1 = self.up(x1)

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)

class ResidualUNet(nn.Module):

    def __init__(self, in_channels=1, num_classes=6):
        super().__init__()
        self.inc = ResidualBlock(in_channels, 16)

        self.down1 = ResidualDown(16, 32)
        self.down2 = ResidualDown(32, 64)
        self.down3 = ResidualDown(64, 128)
        self.down4 = ResidualDown(128, 256)
        self.up1 = ResidualUp(256, 128, 128)
        self.up2 = ResidualUp(128, 64, 64)
        self.up3 = ResidualUp(64, 32, 32)
        self.up4 = ResidualUp(32, 16, 16)

        self.outc = OutConv(16, num_classes)

    def forward(self, x):

        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)

class ResidualAttentionUp(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=True
        )
        self.attention = AttentionGate(
            skip_channels=skip_channels,
            gating_channels=in_channels,
            inter_channels=max(1, skip_channels // 2)
        )
        self.conv = ResidualBlock(
            in_channels + skip_channels,
            out_channels
        )

    def forward(self, x1, x2):

        # Upsample decoder feature
        x1 = self.up(x1)

        # Apply attention to encoder skip feature
        x2 = self.attention(x2, x1)

        # Concatenate two features
        x = torch.cat([x2, x1], dim=1)

        # Residual feature extraction
        return self.conv(x)

class ResidualAttentionUNet(nn.Module):

    def __init__(self, in_channels=1, num_classes=6):
        super().__init__()

        self.inc = ResidualBlock(in_channels, 16)

        self.down1 = ResidualDown(16, 32)
        self.down2 = ResidualDown(32, 64)
        self.down3 = ResidualDown(64, 128)
        self.down4 = ResidualDown(128, 256)
        self.up1 = ResidualAttentionUp(256, 128, 128)
        self.up2 = ResidualAttentionUp(128, 64, 64)
        self.up3 = ResidualAttentionUp(64, 32, 32)
        self.up4 = ResidualAttentionUp(32, 16, 16)

        self.outc = OutConv(16, num_classes)

    def forward(self, x):

        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder with residual blocks and attention gates
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)

if __name__ == "__main__":
    block = DoubleConv(1, 16)

    x = torch.randn(4, 1, 256, 128)
    y = block(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)

    double_conv = DoubleConv(1, 16)
    down = Down(16, 32)

    y1 = double_conv(x)
    y2 = down(y1)

    print("Input:", x.shape)
    print("After DoubleConv:", y1.shape)
    print("After Down:", y2.shape)

    up = Up(32, 16, 16)

    y3 = up(y2, y1)

    print("After Up:", y3.shape)

    out_conv = OutConv(16, 4)

    y4 = out_conv(y3)

    print("After OutConv:", y4.shape)

    prediction = torch.argmax(y4, dim=1)

    print("Prediction shape:", prediction.shape)



    model = UNet(in_channels=1, num_classes=4)

    x = torch.randn(4, 1, 256, 128)

    with torch.no_grad():
        output = model(x)

    print("U-Net Input:", x.shape)
    print("U-Net Output:", output.shape)

    attention = AttentionGate(
        skip_channels=128,
        gating_channels=256,
        inter_channels=64
    )

    x = torch.randn(2, 128, 32, 16)
    g = torch.randn(2, 256, 32, 16)

    output = attention(x, g)

    print("Skip input:", x.shape)
    print("Gating input:", g.shape)
    print("Attention output:", output.shape)

    attention_up = AttentionUp(
        in_channels=256,
        skip_channels=128,
        out_channels=128
    )

    x1 = torch.randn(2, 256, 16, 9)
    x2 = torch.randn(2, 128, 32, 18)

    output = attention_up(x1, x2)

    print("Decoder input:", x1.shape)
    print("Encoder skip:", x2.shape)
    print("AttentionUp output:", output.shape)

    model = AttentionUNet(
        in_channels=1,
        num_classes=6
    )

    x = torch.randn(2, 1, 256, 144)

    with torch.no_grad():
        logits = model(x)

    print("Input:", x.shape)
    print("Attention U-Net output:", logits.shape)

    model.train()

    x = torch.randn(2, 1, 256, 144)
    logits = model(x)

    loss = logits.square().mean()
    loss.backward()

    print(
        "Attention gradient exists:",
        model.up1.attention.W_x.weight.grad is not None
    )

    block = ResidualBlock(
        in_channels=16,
        out_channels=32
    )

    x = torch.randn(2, 16, 128, 72)

    output = block(x)

    print("Input:", x.shape)
    print("Residual output:", output.shape)

    block = ResidualBlock(
        in_channels=64,
        out_channels=64
    )

    x = torch.randn(2, 64, 64, 36)

    output = block(x)

    print("Input:", x.shape)
    print("Residual output:", output.shape)

    down = ResidualDown(16, 32)

    x = torch.randn(2, 16, 256, 144)
    output = down(x)

    print("ResidualDown input:", x.shape)
    print("ResidualDown output:", output.shape)

    up = ResidualUp(
        in_channels=256,
        skip_channels=128,
        out_channels=128
    )

    x1 = torch.randn(2, 256, 16, 9)
    x2 = torch.randn(2, 128, 32, 18)

    output = up(x1, x2)

    print("ResidualUp decoder:", x1.shape)
    print("ResidualUp skip:", x2.shape)
    print("ResidualUp output:", output.shape)

    model = ResidualUNet(
        in_channels=1,
        num_classes=6
    )

    x = torch.randn(2, 1, 256, 144)

    with torch.no_grad():
        logits = model(x)

    print("Input:", x.shape)
    print("Residual U-Net output:", logits.shape)

    model.train()

    x = torch.randn(2, 1, 256, 144)

    logits = model(x)

    loss = logits.square().mean()

    model.zero_grad()
    loss.backward()

    print(
        "Residual gradient exists:",
        model.down1.conv.conv1.weight.grad is not None
    )

    print(
        "Shortcut gradient exists:",
        model.down1.conv.shortcut.weight.grad is not None
    )

    test_up = ResidualAttentionUp(
        in_channels=256,
        skip_channels=128,
        out_channels=128
    )

    x1 = torch.randn(2, 256, 16, 9)
    x2 = torch.randn(2, 128, 32, 18)

    output = test_up(x1, x2)

    print("Decoder input:", x1.shape)
    print("Encoder skip:", x2.shape)
    print("ResidualAttentionUp output:", output.shape)

    model = ResidualAttentionUNet(
        in_channels=1,
        num_classes=6
    )

    x = torch.randn(2, 1, 256, 144)

    with torch.no_grad():
        logits = model(x)

    print("Input:", x.shape)
    print("Residual Attention U-Net output:", logits.shape)

    model.train()
    model.zero_grad()

    x = torch.randn(2, 1, 256, 144)

    logits = model(x)

    loss = logits.square().mean()
    loss.backward()

    print(
        "Residual gradient exists:",
        model.down1.conv.conv1.weight.grad is not None
    )

    print(
        "Shortcut gradient exists:",
        model.down1.conv.shortcut.weight.grad is not None
    )

    print(
        "Attention gradient exists:",
        model.up1.attention.W_x.weight.grad is not None
    )