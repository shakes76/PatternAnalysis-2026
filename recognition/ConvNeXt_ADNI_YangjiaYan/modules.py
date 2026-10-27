"""Model definitions for ADNI MRI classification."""

import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class SimpleCNN(nn.Module):
    """Baseline CNN for binary MRI classification."""

    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),   # 224 -> 112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),   # 112 -> 56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),   # 56 -> 28

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ConvNeXtClassifier(nn.Module):
    """Adapt Torchvision ConvNeXt-Tiny for binary ADNI classification.

    Architecture: Liu et al. (2022), A ConvNet for the 2020s, CVPR.
    https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html
    Backbone implementation and pretrained weights: torchvision.models.
    https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py
    The backbone is reused; this wrapper replaces its output classifier.
    """

    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()

        if pretrained:
            weights = ConvNeXt_Tiny_Weights.DEFAULT
        else:
            weights = None

        self.backbone = convnext_tiny(weights=weights)

        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def build_model(model_name, num_classes=2, pretrained=False):
    """Factory function to create a model by name."""

    model_name = model_name.lower()

    if model_name == "cnn":
        return SimpleCNN(num_classes=num_classes)

    if model_name == "convnext":
        return ConvNeXtClassifier(
            num_classes=num_classes,
            pretrained=pretrained
        )

    raise ValueError(
        f"Unknown model_name: {model_name}. "
        f"Expected 'cnn' or 'convnext'."
    )


def count_parameters(model):
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    return total, trainable


if __name__ == "__main__":
    device = torch.device("cpu")
    dummy = torch.randn(2, 3, 224, 224).to(device)

    for name in ["cnn", "convnext"]:
        model = build_model(name, num_classes=2, pretrained=False).to(device)
        output = model(dummy)
        total, trainable = count_parameters(model)

        print(f"\nModel: {name}")
        print("Output shape:", tuple(output.shape))
        print("Total params:", total)
        print("Trainable params:", trainable)
