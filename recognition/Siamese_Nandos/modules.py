"""Model definitions used by the COMP3710 ISIC 2020 experiments."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18Baseline(nn.Module):
    """Binary melanoma classifier matching the baseline evaluator checkpoint."""

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        # Keep torchvision's top-level parameter names so state_dicts can be
        # loaded directly by the evaluator's `resnet18(); model.fc = ...` code.
        for name in (
            "conv1", "bn1", "relu", "maxpool", "layer1", "layer2",
            "layer3", "layer4", "avgpool", "fc",
        ):
            setattr(self, name, getattr(backbone, name))
        self.fc = nn.Linear(self.fc.in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x).squeeze(1)


class SiameseNetWithClassifier(nn.Module):
    """Shared ResNet-18 encoder, metric-learning projector and binary head."""

    def __init__(self, dim: int = 128, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projector = nn.Linear(features, dim)
        self.classifier = nn.Linear(features, 1)

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


    def embed_features(self, features: torch.Tensor) -> torch.Tensor:
        # Unit-length vectors make pair distances less sensitive to feature scale.
        return F.normalize(self.projector(features), p=2, dim=1)

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        # These are logits, not probabilities; BCEWithLogitsLoss applies the
        # sigmoid operation in a numerically stable way during training.
        return self.classifier(features).squeeze(1)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed_features(self.encode_features(x))

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        return self.classify_features(self.encode_features(x))

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        # Compute each endpoint once, then use both heads on the resulting
        # features. The return order is (embedding A, embedding B, logit A, logit B).
        features_a = self.encode_features(a)
        features_b = self.encode_features(b)
        return (
            self.embed_features(features_a),
            self.embed_features(features_b),
            self.classify_features(features_a),
            self.classify_features(features_b),
        )


def contrastive_loss(
    z1: torch.Tensor, z2: torch.Tensor, same: torch.Tensor, margin: float = 1.0
) -> torch.Tensor:
    """Pull matching-class embeddings together and separate different-class pairs."""
    # `same` is 1 for a matching-class pair. Its squared distance is minimised.
    # A different-class pair is penalised only while it is closer than `margin`.
    # Cast to float32 so this calculation stays stable when the forward pass
    # uses mixed precision.
    distance = F.pairwise_distance(z1.float(), z2.float())
    same = same.float()
    return (same * distance.square() + (1.0 - same) * F.relu(margin - distance).square()).mean()
