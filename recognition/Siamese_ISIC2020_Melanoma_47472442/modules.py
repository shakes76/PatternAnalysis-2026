"""Define the shared encoder, classifier and embedding comparisons.

Encoder: He et al., Deep Residual Learning,
https://arxiv.org/abs/1512.03385.
Contrastive loss: Hadsell et al., CVPR 2006,
https://doi.org/10.1109/CVPR.2006.100.

Author: Alex Anderson (47472442), COMP3710 Pattern Recognition, 2026.
"""

from typing import cast

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


class MelanomaModel(nn.Module):
    """Extract class logits and embeddings with a shared ResNet-18.

    Parameters
    ----------
    pretrained : bool, optional
        Load ImageNet V1 encoder weights if True. Default is False.

    Attributes
    ----------
    encoder : torch.nn.Module
        ResNet-18 with its final classifier replaced by an identity layer.
    classifier : torch.nn.Linear
        Linear head mapping 512 features to benign and melanoma logits.
    """

    def __init__(self, pretrained: bool = False) -> None:
        """Initialize the encoder and classification head."""
        super().__init__()
        self.encoder = cast(nn.Module, resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None))
        self.encoder.fc = nn.Identity()
        self.classifier = nn.Linear(512, 2)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode an RGB batch for classification and image comparison.

        Parameters
        ----------
        images : torch.Tensor of shape (batch, 3, height, width)
            Normalised RGB images.

        Returns
        -------
        logits : torch.Tensor of shape (batch, 2)
            Unnormalised scores, ordered benign then melanoma.
        embeddings : torch.Tensor of shape (batch, 512)
            Unit-length encoder features for distance comparisons.
        """
        features = self.encoder(images)
        return self.classifier(features), F.normalize(features, dim=1)


def contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    """Compute contrastive loss over all distinct image pairs in a batch.

    Parameters
    ----------
    embeddings : torch.Tensor of shape (batch, features)
        Unit-length embeddings produced by the shared encoder.
    labels : torch.Tensor of shape (batch,)
        Class labels: 0 for benign and 1 for melanoma.
    margin : float
        Minimum desired distance between different-class embeddings.

    Returns
    -------
    torch.Tensor
        Scalar loss with gradients for the input embeddings.

    Notes
    -----
    Same-class pairs are pulled together using squared distance.
    Different-class pairs are penalised only below the margin. Each pair is
    counted once, and the two pair types are averaged separately and added.
    Benign pairs still dominate within the same-class term.
    """
    squared = (2 - 2 * embeddings @ embeddings.T).clamp_min(1e-12)  # unit vectors: |a-b|^2 = 2 - 2a.b
    same = labels[:, None] == labels[None, :]
    pairs = torch.ones_like(same).triu(diagonal=1)  # each unordered pair once, no self-pairs
    positive, negative = (same & pairs).float(), (~same & pairs).float()
    pull = squared
    push = F.relu(margin - squared.sqrt()).square()
    # A batch with only one class has no different-class pairs.
    return (pull * positive).sum() / positive.sum().clamp_min(1) + (push * negative).sum() / negative.sum().clamp_min(1)


def nearest_references(
    queries: torch.Tensor, references: torch.Tensor, reference_labels: torch.Tensor, neighbours: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find the nearest reference and a melanoma similarity score.

    Parameters
    ----------
    queries : torch.Tensor of shape (queries, features)
        Query embeddings on the same device as the references.
    references : torch.Tensor of shape (references, features)
        Embeddings from labelled training images.
    reference_labels : torch.Tensor of shape (references,)
        Reference labels: 0 for benign and 1 for melanoma.
    neighbours : int
        Number of nearest references to average per class. Each class must
        contain at least this many references.

    Returns
    -------
    nearest : torch.Tensor of shape (queries,)
        Index of each query's closest reference across both classes.
    scores : torch.Tensor of shape (queries,)
        Mean benign distance divided by the sum of mean benign and melanoma
        distances. Higher values indicate greater melanoma similarity;
        0.5 means equal mean distances. This score is not a probability.
    """
    distances = torch.cdist(queries, references)
    benign, melanoma = (
        distances[:, reference_labels == label].topk(neighbours, largest=False).values.mean(dim=1)
        for label in (0, 1)
    )
    return distances.argmin(dim=1), benign / (benign + melanoma)
