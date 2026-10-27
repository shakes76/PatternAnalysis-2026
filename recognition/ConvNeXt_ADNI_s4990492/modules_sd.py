"""Stochastic depth for the project's self-implemented ConvNeXt.

Architecture reference: Liu et al., A ConvNet for the 2020s.
https://arxiv.org/abs/2201.03545
"""
import torch
from modules import ConvNeXtBlock, ConvNeXtTiny


class ConvNeXtBlockSD(ConvNeXtBlock):
    """Drop residual branches independently per training sample."""

    def __init__(self, channels, drop_probability):
        super().__init__(channels)
        if not 0 <= drop_probability < 1:
            raise ValueError("Drop probability must be in [0, 1).")
        self.drop_probability = drop_probability

    def forward(self, x):
        residual = x
        x = self.depthwise(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.expand(x)
        x = self.activation(x)
        x = self.project(x)
        x = self.layer_scale * x
        x = x.permute(0, 3, 1, 2)
        if self.training and self.drop_probability > 0:
            keep = 1.0 - self.drop_probability
            mask = x.new_empty((x.shape[0], 1, 1, 1)).bernoulli_(keep)
            x = x * mask / keep
        return residual + x


class ConvNeXtTinySD(ConvNeXtTiny):
    """Use a linearly increasing drop rate while preserving initialization."""

    def __init__(self, max_drop_probability=0.1):
        if not 0 <= max_drop_probability < 1:
            raise ValueError("Maximum drop probability must be in [0, 1).")
        super().__init__()
        total = sum(len(stage) for stage in self.stages)
        index = 0
        # Replacement-block construction must not change the CPU RNG stream.
        with torch.random.fork_rng(devices=[]):
            for stage in self.stages:
                for position, old_block in enumerate(stage):
                    probability = max_drop_probability * index / max(total - 1, 1)
                    new_block = ConvNeXtBlockSD(
                        old_block.depthwise.in_channels, probability
                    )
                    new_block.load_state_dict(old_block.state_dict(), strict=True)
                    stage[position] = new_block
                    index += 1
