"""Independent reduced-width experiment; reuses the custom ConvNeXt blocks.

This is a ConvNeXt variant, not the standard ConvNeXt-Tiny configuration.
"""
import torch
from torch import nn
from modules import ConvNeXtBlock, ChannelLayerNorm


class ConvNeXtCompact(nn.Module):
    """Reduced-width ConvNeXt variant with a two-class output.

    Architecture reference:
    https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py

    Adaptations: grayscale input, binary classification, and half-width stages.
    Keeps all four stages and original (3, 3, 9, 3) depths.
    This initial version uses no stochastic depth or pretrained weights.
    """

    def __init__(self):
        super().__init__()

        depths = (3, 3, 9, 3)
        widths = (48, 96, 192, 384)

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

