"""Convolutional baselines for the ADNI AD vs CN comparison.

SimpleCNN   - the original small 3-layer baseline (23,714 parameters); kept
              unchanged so that earlier checkpoints and results stay loadable.
MatchedCNN  - a plain VGG-style CNN whose parameter count (1,834,122) is close
              to the compact ConvNeXt in modules.py (1,900,898), so that the
              comparison is not dominated by model size.

build_baseline(name) returns a fresh baseline for the names used by
train.py, predict.py and analyze.py ("cnn", "cnn_matched"). The ConvNeXt
itself is always imported from modules.py.
"""
import torch
import torch.nn as nn

BASELINE_NAMES = ("cnn", "cnn_matched")
MODEL_NAMES = ("convnext",) + BASELINE_NAMES


class SimpleCNN(nn.Module):
    """Three 3x3 conv + ReLU layers, max-pooling, global average pool, linear."""

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Linear(64, 2)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def conv_bn_relu(in_channels, out_channels):
    """Standard 3x3 convolution, batch normalisation and ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class MatchedCNN(nn.Module):
    """Plain CNN with about the same number of parameters as the ConvNeXt.

    Four stages of two conv-BN-ReLU layers (widths 40, 80, 160, 320), with
    2x2 max-pooling between stages, then global average pooling and a linear
    layer. It uses none of the ConvNeXt ideas (no depthwise 7x7 convolution,
    LayerNorm, inverted bottleneck, LayerScale or stochastic depth), so it
    tests whether those design choices help at a similar model size.
    """

    def __init__(self, num_classes=2, in_channels=3, widths=(40, 80, 160, 320)):
        super().__init__()
        layers = []
        channels = in_channels
        for index, width in enumerate(widths):
            layers += [conv_bn_relu(channels, width), conv_bn_relu(width, width)]
            if index < len(widths) - 1:
                layers.append(nn.MaxPool2d(2))
            channels = width
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(torch.flatten(x, 1))


def build_baseline(name):
    """Return a new, randomly initialised baseline for a name in BASELINE_NAMES."""
    if name == "cnn":
        return SimpleCNN()
    if name == "cnn_matched":
        return MatchedCNN()
    raise ValueError(f"Unknown baseline {name!r}; expected one of {BASELINE_NAMES}")


if __name__ == "__main__":
    from modules import ConvNeXt
    for model_name in MODEL_NAMES:
        net = (ConvNeXt() if model_name == "convnext" else build_baseline(model_name)).eval()
        count = sum(p.numel() for p in net.parameters() if p.requires_grad)
        with torch.no_grad():
            out = net(torch.randn(2, 3, 224, 224))
        assert out.shape == (2, 2)
        print(f"{model_name:12s} parameters: {count:,}  output: {tuple(out.shape)}")
