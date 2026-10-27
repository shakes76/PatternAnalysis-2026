"""Neural network models for ADNI slice classification."""

import torch
from torch import nn


class SimpleCNN(nn.Module):
    """A compact CNN baseline for single-channel MRI slices."""

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # [B, 1, 240, 256] -> [B, 16, 120, 128]
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # -> [B, 32, 60, 64]
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # -> [B, 64, 30, 32]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, 2)

    def forward(self, images):
        features = self.features(images)
        pooled = self.pool(features)
        flattened = torch.flatten(pooled, start_dim=1)
        return self.classifier(flattened)

class ConvNeXtBlock(nn.Module):
    """A ConvNeXt block with layer scaling and no stochastic depth."""

    def __init__(self, channels, layer_scale_init=1e-6):
        super().__init__()

        # Apply a separate spatial filter to each channel.
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
        )

        # LayerNorm will operate on the last dimension.
        self.norm = nn.LayerNorm(channels, eps=1e-6)

        # Mix information across channels.
        self.expand = nn.Linear(channels, 4 * channels)
        self.activation = nn.GELU()
        self.project = nn.Linear(4 * channels, channels)

        # Start the residual adjustment with a small scale.
        self.layer_scale = nn.Parameter(
            layer_scale_init * torch.ones(channels)
        )

    def forward(self, x):
        residual = x

        x = self.depthwise(x)

        # [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1)

        x = self.norm(x)
        x = self.expand(x)
        x = self.activation(x)
        x = self.project(x)
        x = self.layer_scale * x

        # Return to [B, C, H, W].
        x = x.permute(0, 3, 1, 2)

        return residual + x

class ChannelLayerNorm(nn.Module):
    """Apply LayerNorm across channels of a [B, C, H, W] tensor."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, x):
        # Move channels to the last dimension for LayerNorm.
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class ConvNeXtTiny(nn.Module):
    """Single-channel ConvNeXt-Tiny with a two-class output.

    Architecture reference:
    https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

    Adaptations: grayscale input and binary classification.
    This initial version uses no stochastic depth or pretrained weights.
    """

    def __init__(self):
        super().__init__()

        depths = (3, 3, 9, 3)
        widths = (96, 192, 384, 768)

        # Input stem: reduce spatial dimensions by a factor of four.
        self.stem = nn.Sequential(
            nn.Conv2d(
                1, widths[0],
                kernel_size=4,
                stride=4,
            ),
            ChannelLayerNorm(widths[0]),
        )

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for stage_index in range(4):
            blocks = [
                ConvNeXtBlock(widths[stage_index])
                for _ in range(depths[stage_index])
            ]
            self.stages.append(nn.Sequential(*blocks))

            if stage_index < 3:
                self.downsamples.append(
                    nn.Sequential(
                        ChannelLayerNorm(widths[stage_index]),
                        nn.Conv2d(
                            widths[stage_index],
                            widths[stage_index + 1],
                            kernel_size=2,
                            stride=2,
                        ),
                    )
                )

        self.final_norm = nn.LayerNorm(widths[-1], eps=1e-6)
        self.classifier = nn.Linear(widths[-1], 2)

        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, images):
        x = self.stem(images)

        for stage_index, stage in enumerate(self.stages):
            x = stage(x)

            if stage_index < len(self.downsamples):
                x = self.downsamples[stage_index](x)

        # Average spatial dimensions: [B, C, H, W] -> [B, C].
        x = x.mean(dim=(-2, -1))
        x = self.final_norm(x)

        # Return logits in the order NC, AD.
        return self.classifier(x)

class ConvNeXtTinyRGB(ConvNeXtTiny):
    """Custom three-channel variant for matched scratch/ImageNet experiments.

    Keeps the custom stages; only the input convolution changes.
    No prebuilt model is instantiated here.
    """
    def __init__(self):
        super().__init__()
        self.stem[0] = nn.Conv2d(3, 96, kernel_size=4, stride=4)
        self._initialize_weights(self.stem[0])


if __name__ == "__main__":
    torch.manual_seed(42)

    model = SimpleCNN()
    model.eval()

    # Synthetic inputs only: this checks the model's output shape.
    example_images = torch.randn(2, 1, 240, 256)

    with torch.no_grad():
        logits = model(example_images)

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )

    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all().item()

    print("Input shape:", tuple(example_images.shape))
    print("Output shape:", tuple(logits.shape))
    print("Parameter count:", parameter_count)
    print("Forward check passed")