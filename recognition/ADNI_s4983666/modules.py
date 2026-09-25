"""A small, randomly initialized PyTorch CNN for the AD/NC baseline."""

from torch import nn


class SmallCNN(nn.Module):
    """Map one grayscale slice to an uncalibrated AD logit.

    Group normalization has no running population statistics. Evaluation still
    explicitly switches to eval mode to disable dropout. Global spatial means
    support different image sizes without learning from validation images.
    """

    def __init__(self):
        super().__init__()
        layers = []
        in_channels = 1
        for out_channels in (16, 32, 64, 128):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(4, out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ])
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, images):
        """Return one logit per slice; BCEWithLogitsLoss applies sigmoid internally."""
        if images.ndim != 4 or images.shape[1] != 1 or min(images.shape[-2:]) < 16:
            raise ValueError("Expected [batch, 1, height, width] with height/width >= 16.")
        features = self.features(images).mean(dim=(2, 3))
        return self.classifier(features).squeeze(1)


def count_parameters(model):
    """Count trainable scalar parameters for resource reporting."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
