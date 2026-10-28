"""CNN baseline and ConvNeXt models for ADNI classification."""
import torch
from torch import nn


class CNNBaseline(nn.Module):
    """Simple CNN for comparison with ConvNeXt."""
    def __init__(self, num_classes=2):
        super().__init__()
        layers = []
        in_channels = 3

        for out_channels in (32, 64, 128):
            layers += [
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class LayerNorm2d(nn.Module):
    """Layer normalization for NCHW images."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    """Depthwise convolution, MLP and residual connection."""
    def __init__(self, channels):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, 7, padding=3, groups=channels
        )
        self.norm = nn.LayerNorm(channels)
        self.expand = nn.Linear(channels, 4 * channels)
        self.act = nn.GELU()
        self.project = nn.Linear(4 * channels, channels)
        self.gamma = nn.Parameter(1e-6 * torch.ones(channels))

    def forward(self, x):
        residual = x
        x = self.depthwise(x).permute(0, 2, 3, 1)
        x = self.project(self.act(self.expand(self.norm(x))))
        x = (self.gamma * x).permute(0, 3, 1, 2)
        return residual + x


class ConvNeXtClassifier(nn.Module):
    """Four-stage ConvNeXt-style binary classifier."""
    def __init__(
        self,
        num_classes=2,
        dims=(96, 192, 384, 768),
        depths=(3, 3, 9, 3)
    ):
        super().__init__()

        if len(dims) != 4 or len(depths) != 4:
            raise ValueError("dims and depths must have four elements")

        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], 4, stride=4),
            LayerNorm2d(dims[0])
        )

        stages = []
        for i in range(4):
            if i > 0:
                stages += [
                    LayerNorm2d(dims[i - 1]),
                    nn.Conv2d(dims[i - 1], dims[i], 2, stride=2)
                ]

            stages.extend(
                ConvNeXtBlock(dims[i]) for _ in range(depths[i])
            )

        self.features = nn.Sequential(*stages)
        self.norm = nn.LayerNorm(dims[-1])
        self.classifier = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        x = self.features(self.stem(x)).mean(dim=(-2, -1))
        return self.classifier(self.norm(x))


if __name__ == "__main__":
    torch.set_num_threads(2)
    sample = torch.randn(1, 3, 64, 64)

    models = [
        ("CNN baseline", CNNBaseline()),
        ("Small ConvNeXt", ConvNeXtClassifier(
            dims=(16, 32, 64, 128),
            depths=(1, 1, 1, 1)
        ))
    ]

    for name, model in models:
        model.eval()
        with torch.no_grad():
            output = model(sample)

        assert output.shape == (1, 2)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name}: output={tuple(output.shape)}, "
              f"parameters={params:,}")

    print("CPU model smoke test: PASS")
