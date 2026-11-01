"""Models for melanoma classification."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_channels, out_channels):
    """Conv(3x3) -> BatchNorm -> ReLU -> MaxPool(2), halving height and width."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class BaselineCNN(nn.Module):
    """Plain CNN baseline that classifies a single image as benign or malignant.

    Four Conv-BatchNorm-ReLU-MaxPool blocks (32, 64, 128, 256 channels) are
    followed by global average pooling and a Linear layer. The output is one
    raw logit per image with shape [N, 1], intended for BCEWithLogitsLoss
    (no sigmoid is applied here).
    """

    def __init__(self, in_channels=3):
        super().__init__()
        self.block1 = _conv_block(in_channels, 32)  # 3x224x224 -> 32x112x112
        self.block2 = _conv_block(32, 64)           # -> 64x56x56
        self.block3 = _conv_block(64, 128)          # -> 128x28x28
        self.block4 = _conv_block(128, 256)         # -> 256x14x14
        self.fc = nn.Linear(256, 1)                 # pooled 256 features -> 1 logit

    def forward(self, x):
        """Return one raw logit per image, shape [N, 1]."""
        x = self.block4(self.block3(self.block2(self.block1(x))))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)  # global average pooling -> [N, 256]
        return self.fc(x)


class SiameseEncoder(nn.Module):
    """Shared-weight encoder that maps an image to a unit-length embedding.

    Siamese setup after Koch, Zemel and Salakhutdinov (2015), Siamese neural
    networks for one-shot image recognition, ICML Deep Learning Workshop.

    Uses the same four Conv-BatchNorm-ReLU-MaxPool blocks as BaselineCNN, then
    global average pooling and a Linear layer to `embed_dim`. The output is
    L2-normalised with F.normalize(dim=1), so every embedding has length 1 and
    lies on the unit sphere. For unit vectors a and b,
    ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b = 2 - 2 cos(theta), and cos(theta)
    is in [-1, 1], so ||a - b||^2 is in [0, 4] and the Euclidean distance is
    in [0, 2]: 0 for identical directions, 2 for opposite directions. This
    bounded range keeps distances comparable across images and makes a fixed
    contrastive-loss margin meaningful. Returns shape [N, embed_dim].
    """

    def __init__(self, in_channels=3, embed_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(in_channels, 32),  # 3x224x224 -> 32x112x112
            _conv_block(32, 64),           # -> 64x56x56
            _conv_block(64, 128),          # -> 128x28x28
            _conv_block(128, 256),         # -> 256x14x14
        )
        self.fc = nn.Linear(256, embed_dim)  # pooled 256 features -> embedding

    def forward(self, x):
        """Return L2-normalised embeddings, shape [N, embed_dim]."""
        x = F.adaptive_avg_pool2d(self.features(x), 1).flatten(1)  # [N, 256]
        return F.normalize(self.fc(x), dim=1)  # L2-normalise each embedding


class ContrastiveLoss(nn.Module):
    """Contrastive loss for pairs of embeddings.

    Follows Hadsell, Chopra and LeCun (2006), Dimensionality reduction by
    learning an invariant mapping, CVPR.

    Label convention: y = 0 for a SIMILAR pair (same class) and y = 1 for a
    DISSIMILAR pair (different class). With d the Euclidean distance between
    the two embeddings, the loss per pair is

        (1 - y) * 0.5 * d^2  +  y * 0.5 * clamp(margin - d, min=0)^2

    so similar pairs are pulled together and dissimilar pairs are pushed apart
    until they are at least `margin` away, after which they contribute nothing.
    The mean over the batch is returned.
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def loss_from_distance(self, d, y):
        """Mean contrastive loss given distances `d` and labels `y` (same shape)."""
        y = y.to(d.dtype)
        # Similar pairs (y = 0) cost 0.5 * d^2; dissimilar pairs (y = 1) cost
        # 0.5 * (margin - d)^2 while d < margin and nothing once d >= margin.
        similar = (1 - y) * 0.5 * d.pow(2)
        dissimilar = y * 0.5 * torch.clamp(self.margin - d, min=0).pow(2)
        return (similar + dissimilar).mean()

    def forward(self, emb1, emb2, y):
        """Loss for embedding batches `emb1`, `emb2` of shape [N, D] and labels `y` of shape [N]."""
        return self.loss_from_distance(F.pairwise_distance(emb1, emb2), y)
