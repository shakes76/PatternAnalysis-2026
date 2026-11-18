"""
Neural network architectures for ADNI MRI classification.

Models:
    1. BaselineCNN
    2. ConvNeXt-Tiny

Classes:
    0 = Normal Control (NC)
    1 = Alzheimer's Disease (AD)

Both models output raw classification logits.
"""

import torch
import torch.nn as nn


NUM_CLASSES = 2


class ConvBlock(nn.Module):
    """
    Convolutional feature extraction block.

    Architecture:
        Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm
        -> ReLU -> MaxPool
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        return self.layers(x)


class BaselineCNN(nn.Module):
    """
    Conventional CNN baseline for MRI classification.

    Four convolutional blocks progressively extract features.
    Global average pooling reduces the spatial dimensions
    before classification.
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)


class ChannelLayerNorm(nn.LayerNorm):
    """Apply LayerNorm over the channel dimension of an NCHW tensor."""

    def __init__(self, channels):
        super().__init__(channels, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        return x.permute(0, 3, 1, 2)


class ConvNeXtMLP(nn.Module):
    """Expand channels fourfold, apply GELU, and reduce channels."""

    def __init__(self, channels):
        super().__init__()
        self.fc1 = nn.Linear(channels, 4 * channels)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(4 * channels, channels)

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class ConvNeXtBlock(nn.Module):
    """
    Native PyTorch implementation of a ConvNeXt residual block.

    Architecture reference:
        Liu et al., A ConvNet for the 2020s, CVPR 2022.
        https://arxiv.org/abs/2201.03545

    Spatial mixing uses a 7x7 depthwise convolution.
    Channel mixing uses a two-layer MLP with GELU activation.
    A learnable layer scale controls the residual contribution.
    """

    def __init__(self, channels):
        super().__init__()

        self.conv_dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
            bias=True,
        )

        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = ConvNeXtMLP(channels)

        self.gamma = nn.Parameter(
            torch.full((channels,), 1e-6)
        )

    def forward(self, x):
        residual = x

        # Spatial feature extraction.
        x = self.conv_dw(x)

        # Channel-last representation for normalization and MLP.
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.mlp(x)

        # Restore NCHW layout and apply learnable layer scaling.
        x = x.permute(0, 3, 1, 2)
        x = x * self.gamma.reshape(1, -1, 1, 1)

        return residual + x


class ConvNeXtStage(nn.Module):
    """Optional spatial downsampling followed by residual blocks."""

    def __init__(self, in_channels, out_channels, depth, downsample):
        super().__init__()

        if downsample:
            self.downsample = nn.Sequential(
                ChannelLayerNorm(in_channels),
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=2,
                    stride=2,
                    bias=True,
                ),
            )
        else:
            if in_channels != out_channels:
                raise ValueError("Channel mismatch without downsampling.")

            self.downsample = nn.Identity()

        self.blocks = nn.Sequential(
            *[ConvNeXtBlock(out_channels) for _ in range(depth)]
        )

    def forward(self, x):
        x = self.downsample(x)
        return self.blocks(x)


class ConvNeXtHead(nn.Module):
    """Global average pooling, normalization, and classification."""

    def __init__(self, channels, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = ChannelLayerNorm(channels)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.norm(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ConvNeXtTiny(nn.Module):
    """
    ConvNeXt-Tiny assembled entirely from native PyTorch layers.

    Architecture reference:
        Liu et al., A ConvNet for the 2020s, CVPR 2022.
        https://arxiv.org/abs/2201.03545

    Stage depths:   3, 3, 9, 3
    Stage channels: 96, 192, 384, 768

    Parameter names intentionally preserve compatibility with
    checkpoints produced by the original experimental implementation.
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.depths = (3, 3, 9, 3)
        self.dims = (96, 192, 384, 768)

        # Patchifying stem: 224x224 -> 56x56.
        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                self.dims[0],
                kernel_size=4,
                stride=4,
                bias=True,
            ),
            ChannelLayerNorm(self.dims[0]),
        )

        # Four hierarchical feature extraction stages.
        stages = []
        for i, (channels, depth) in enumerate(
            zip(self.dims, self.depths)
        ):
            in_channels = self.dims[max(0, i - 1)]

            stages.append(
                ConvNeXtStage(
                    in_channels=in_channels,
                    out_channels=channels,
                    depth=depth,
                    downsample=(i > 0),
                )
            )

        self.stages = nn.Sequential(*stages)
        self.head = ConvNeXtHead(self.dims[-1], num_classes)

        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def get_classifier(self):
        return self.head.fc

    def forward_features(self, x):
        x = self.stem(x)
        return self.stages(x)

    def forward(self, x):
        return self.head(self.forward_features(x))


class ConvNeXtClassifier(nn.Module):
    """
    ADNI classifier using the native PyTorch ConvNeXt implementation.

    Legacy checkpoint loading is supported through compatible
    parameter names. No external model constructor is used.
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        pretrained=False,
        freeze_backbone=False,
    ):
        super().__init__()

        if pretrained:
            raise ValueError(
                "ImageNet initialization requires an explicit "
                "weight-loading procedure in the revised implementation."
            )

        self.model = ConvNeXtTiny(num_classes=num_classes)

        if freeze_backbone:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

            for parameter in self.model.get_classifier().parameters():
                parameter.requires_grad = True

    def forward(self, x):
        return self.model(x)


def create_model(
    model_name,
    num_classes=NUM_CLASSES,
    pretrained=False,
    freeze_backbone=False,
):
    """
    Factory function for selecting a model architecture.

    Args:
        model_name:
            "baseline" or "convnext"

        pretrained:
            Whether to initialise ConvNeXt with
            ImageNet-pretrained weights.

        freeze_backbone:
            Whether to train only the ConvNeXt classifier.

    Returns:
        A PyTorch classification model.
    """

    model_name = model_name.lower()

    if model_name == "baseline":
        if pretrained:
            raise ValueError(
                "The baseline CNN does not support pretrained weights."
            )

        return BaselineCNN(num_classes=num_classes)

    if model_name == "convnext":
        return ConvNeXtClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

    raise ValueError(
        f"Unknown model: {model_name}. "
        "Choose 'baseline' or 'convnext'."
    )


def count_parameters(model, trainable_only=False):
    """Count model parameters."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


if __name__ == "__main__":

    print("=== MODEL ARCHITECTURE TEST ===")

    # Small synthetic MRI batch.
    inputs = torch.randn(2, 3, 224, 224)

    for model_name in ("baseline", "convnext"):

        print(f"\nTesting {model_name}...")

        model = create_model(
            model_name=model_name,
            pretrained=False,
        )

        model.eval()

        with torch.inference_mode():
            outputs = model(inputs)

        print("Input shape:", tuple(inputs.shape))
        print("Output shape:", tuple(outputs.shape))
        print(
            "Total parameters:",
            f"{count_parameters(model):,}",
        )

        assert outputs.shape == (2, 2)
        assert torch.isfinite(outputs).all()

        print("Forward pass successful.")

    print("\nBoth models passed the architecture test.")
