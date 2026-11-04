"""Course-facing model interface (PyTorch only, no NumPy).

The components are implemented in ``models/cnn.py`` (baseline) and
``models/convnext.py`` (ConvNeXt, Liu et al., CVPR 2022). ``create_model``
returns random weights; the optional ImageNet-1k ConvNeXt-Tiny weights of the
paper's authors are loaded by ``models/pretrained.py`` from a local file
(nothing is downloaded by the code).
"""

from models import (
    MODEL_CHOICES, MODEL_NAMES, PRETRAINED_MODELS, ConvNeXtBlock, ConvNeXtLite, ConvNeXtTiny,
    ConvNeXtTinyImageNet, LayerNorm2d, SmallCNN, StochasticDepth, count_parameters, create_model,
    model_minimum_size,
)

__all__ = [
    "MODEL_CHOICES", "MODEL_NAMES", "PRETRAINED_MODELS", "ConvNeXtBlock", "ConvNeXtLite", "ConvNeXtTiny",
    "ConvNeXtTinyImageNet",
    "LayerNorm2d", "SmallCNN", "StochasticDepth", "count_parameters", "create_model",
    "model_minimum_size",
]
