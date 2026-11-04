"""Map command-line model names to architectures.

``create_model`` always returns random weights; for ``convnext_tiny_in1k`` the
training script then copies in the ImageNet-1k backbone (models/pretrained.py).
"""

from torch import nn

from .cnn import SmallCNN
from .convnext import ConvNeXtLite, ConvNeXtTiny, ConvNeXtTinyImageNet


# CLI alias -> versioned architecture name stored in checkpoints.
MODEL_CHOICES = {
    "cnn": "small_cnn_v1",
    "convnext_lite": "convnext_lite_v1",
    "convnext_tiny": "convnext_tiny_v1",
    "convnext_tiny_in1k": "convnext_tiny_in1k_v1",
}
PRETRAINED_MODELS = ("convnext_tiny_in1k_v1",)
MODEL_NAMES = tuple(MODEL_CHOICES.values())


def model_minimum_size(name: str) -> int:
    """Smallest supported image side: four 2x poolings (CNN) or stride 32 (ConvNeXt)."""
    if name not in MODEL_NAMES:
        raise ValueError(f"Unsupported model name: {name}")
    return 16 if name == "small_cnn_v1" else 32


def create_model(name: str, *, input_channels: int = 1, drop_path: float | None = None) -> nn.Module:
    """Build the architecture with random weights (pretrained weights are loaded separately)."""
    if name == "small_cnn_v1":
        if drop_path is not None:
            raise ValueError("DropPath applies to ConvNeXt only.")
        return SmallCNN(input_channels)
    if name == "convnext_tiny_v1":
        return ConvNeXtTiny(input_channels, 0.1 if drop_path is None else drop_path)
    if name == "convnext_tiny_in1k_v1":
        return ConvNeXtTinyImageNet(input_channels, 0.1 if drop_path is None else drop_path)
    if name == "convnext_lite_v1":
        return ConvNeXtLite(input_channels, 0.1 if drop_path is None else drop_path)
    raise ValueError(f"Unsupported model name: {name}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters for the resource table."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
