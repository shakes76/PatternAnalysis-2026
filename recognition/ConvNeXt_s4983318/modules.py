"""Neural-network components for COMP3710 ADNI AD-vs-NC classification.

Defines the baseline CNN and TorchVision ConvNeXt-Tiny model builder.
Input : float32 MRI tensor [batch, 3, 224, 224].
Output: raw class logits [batch, 2], with class 0=NC and 1=AD.

The model intentionally does NOT contain a Softmax layer. During training,
``torch.nn.CrossEntropyLoss`` expects unnormalised logits.

AI-use note: ChatGPT assisted with the initial implementation and comments.
The implementation was verified locally using test_models.py and test_convnext.py.
"""

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Two 3x3 convolutions followed by 2x spatial downsampling.

    Batch normalization stabilizes training; ReLU provides nonlinearity.
    The MaxPool2d halves the height and width while keeping the channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images)


class BaselineCNN(nn.Module):
    """Lightweight four-stage CNN for fair comparison with ConvNeXt-Tiny.

    Feature extraction gradually increases channel capacity:
      3 -> 16 -> 32 -> 64 -> 128 channels.

    AdaptiveAvgPool2d(4,4) keeps a rough spatial grid (important for MRI
    anatomy), while giving the classifier a fixed input size. A dropout
    layer reduces some overfitting risk. This model is trained from scratch.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        self.features = nn.Sequential(
            ConvBlock(3, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        return self.classifier(features)


def build_convnext_tiny(num_classes: int = 2, pretrained: bool = False) -> nn.Module:
    """Create ConvNeXt-Tiny with a two-class head for ADNI classification.

    Uses the standard TorchVision ConvNeXt-Tiny implementation rather than
    reimplementing the architecture from scratch.

    Architecture reference:
    Liu et al., "A ConvNet for the 2020s", CVPR 2022.
    https://openaccess.thecvf.com/content/CVPR2022/html/
    Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html

    TorchVision implementation:
    https://pytorch.org/vision/stable/models/convnext.html

    The caller controls whether ImageNet-1K pretrained weights are used.
    Replacing only the final linear layer retains the pretrained feature
    extraction backbone while adapting the classifier to NC versus AD.

    Input shape: [N, 3, 224, 224], ImageNet-normalized as in dataset.py.
    Output shape: [N, num_classes] raw logits (no Softmax).
    """
    if num_classes < 2:
        raise ValueError('num_classes must be at least 2')
    try:
        from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError('A compatible torchvision installation is required.') from exc

    weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
    model = convnext_tiny(weights=weights)
    input_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(input_features, num_classes)
    return model
