"""COMP3710 ADNI classification models. Labels: NC=0, AD=1."""

import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class BaselineCNN(nn.Module):
    """Simple four-block CNN. Input (B,3,224,224); output (B,2) logits."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256),
            nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ConvNeXtClassifier(nn.Module):
    def __init__(
        self,
        num_classes=2,
        pretrained=True,
        freeze_backbone=False,
        partial_finetune=False,
    ):
        super().__init__()

        if freeze_backbone and partial_finetune:
            raise ValueError(
                "Choose either freeze_backbone or partial_finetune."
            )

        weights = (
            ConvNeXt_Tiny_Weights.DEFAULT
            if pretrained else None
        )

        self.model = convnext_tiny(weights=weights)
        self.freeze_backbone = freeze_backbone
        self.partial_finetune = partial_finetune

        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(
            in_features, num_classes
        )

        if freeze_backbone or partial_finetune:
            # Initially freeze all parameters.
            for param in self.model.parameters():
                param.requires_grad = False

            # Always train the new classification layer.
            for param in self.model.classifier[-1].parameters():
                param.requires_grad = True

            if partial_finetune:
                # Also train the final ConvNeXt feature stage.
                for param in self.model.features[-1].parameters():
                    param.requires_grad = True

    def train(self, mode=True):
        super().train(mode)

        if self.freeze_backbone:
            # Experiment 3: entire backbone frozen
            self.model.features.eval()

        elif self.partial_finetune:
            # Experiment 4: early backbone stages frozen,
            # final feature stage trainable.
            for stage in self.model.features[:-1]:
                stage.eval()

        return self

    def forward(self, x):
        return self.model(x)



def build_model(
    model_name,
    num_classes=2,
    pretrained=True,
    freeze_backbone=False,
    partial_finetune=False,
):
    model_name = model_name.lower()

    if model_name == "baseline":
        return BaselineCNN(num_classes=num_classes)

    if model_name == "convnext":
        return ConvNeXtClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            partial_finetune=partial_finetune,
        )

    raise ValueError(f"Unknown model: {model_name}")


def test_models():
    x = torch.randn(2, 3, 224, 224)
    for name, frozen in (("baseline", False), ("convnext", False), ("convnext", True)):
        model = build_model(name, pretrained=False, freeze_backbone=frozen).eval()
        with torch.no_grad():
            output = model(x)
        assert output.shape == (2, 2)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if frozen:
            assert trainable == 1538, f"Expected 1538 trainable parameters, got {trainable}"
        print(f"{name} frozen={frozen}: output={tuple(output.shape)}, "
              f"params={total:,}, trainable={trainable:,}")
    print("All model tests passed!")


if __name__ == "__main__":
    test_models()

