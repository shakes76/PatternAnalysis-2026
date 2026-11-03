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
import timm


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


class ConvNeXtClassifier(nn.Module):
    """
    ConvNeXt-Tiny classifier for ADNI MRI images.

    Uses the timm ConvNeXt implementation.

    When pretrained=True, ImageNet-pretrained weights
    initialise the backbone, while the classification
    head is adapted to two output classes.
    """

    def __init__(
        self,
        num_classes=NUM_CLASSES,
        pretrained=False,
        freeze_backbone=False,
    ):
        super().__init__()

        self.model = timm.create_model(
            "convnext_tiny",
            pretrained=pretrained,
            num_classes=num_classes,
        )

        if freeze_backbone:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

            # Keep the final classification head trainable.
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
