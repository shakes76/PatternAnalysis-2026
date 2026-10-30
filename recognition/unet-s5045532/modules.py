import torch
import torch.nn as nn

MODEL_SAVE_LOCATION = "C:\\Users\\lachl\\Documents\\UQ CS\\COMP3710\\PatternAnalysis-2026\\recognition\\unet-s5045532\\model.pth"
batch_size = 64
#Model, model Initialization and ResidualBlock all made with AI

class ResidualBlock(nn.Module):
    """Two convolution layers with a residual connection."""

    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.dropout = (
            nn.Dropout2d(dropout_rate)
            if dropout_rate > 0 else nn.Identity()
        )

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Project the residual when channel counts differ
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


class Improved2DUNet(nn.Module):
    """
    Improved residual 2D U-Net for image segmentation.

    Input:  (B, 1, 256, 128)
    Output: (B, num_classes, 256, 128) logits
    """

    def __init__(
        self,
        in_channels=1,
        num_classes=6,
        base_filters=32,
        dropout_rate=0.2
    ):
        super().__init__()

        f = base_filters

        # Encoder
        self.enc1 = ResidualBlock(in_channels, f)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ResidualBlock(f, f * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ResidualBlock(f * 2, f * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ResidualBlock(f * 4, f * 8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ResidualBlock(
            f * 8, f * 16, dropout_rate=dropout_rate
        )

        # Decoder
        self.up4 = nn.ConvTranspose2d(
            f * 16, f * 8, kernel_size=2, stride=2
        )
        self.dec4 = ResidualBlock(f * 16, f * 8)

        self.up3 = nn.ConvTranspose2d(
            f * 8, f * 4, kernel_size=2, stride=2
        )
        self.dec3 = ResidualBlock(f * 8, f * 4)

        self.up2 = nn.ConvTranspose2d(
            f * 4, f * 2, kernel_size=2, stride=2
        )
        self.dec2 = ResidualBlock(f * 4, f * 2)

        self.up1 = nn.ConvTranspose2d(
            f * 2, f, kernel_size=2, stride=2
        )
        self.dec1 = ResidualBlock(f * 2, f)

        # Output logits: no softmax needed during training
        self.out_conv = nn.Conv2d(
            f, num_classes, kernel_size=1
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)


# -----------------------------------
# Initialize the model
# -----------------------------------

NUM_CLASSES = 6  # Number of mask classes

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = Improved2DUNet(
    in_channels=1,
    num_classes=NUM_CLASSES,
    base_filters=32,
    dropout_rate=0.2
).to(device)

# Test the model
x = torch.randn(batch_size, 1, 256, 128, device=device)

with torch.no_grad():
    logits = model(x)

print("Input shape: ", x.shape)
print("Output shape:", logits.shape)
print("Device:", device)
print("Parameters:", sum(p.numel() for p in model.parameters()))

assert logits.shape == (batch_size, NUM_CLASSES, 256, 128)
