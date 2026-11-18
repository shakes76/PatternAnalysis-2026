"""Exponential moving average (EMA) of model weights.

After every optimizer step: ema = decay * ema + (1 - decay) * weights.
Averaging recent weights smooths the noisy late-training updates that a small
dataset produces; ConvNeXt (Liu et al., 2022) also evaluates EMA weights.
Polyak and Juditsky, "Acceleration of Stochastic Approximation by Averaging",
SIAM J. Control Optim., 1992.

The decay warm-up min(decay, (1 + n) / (10 + n)) follows TensorFlow's
ExponentialMovingAverage(num_updates=...), so the average does not stay close
to the random initial weights during the first steps.
"""

import copy

import torch
from torch import nn


class ModelEMA:
    """Keep an averaged copy of a model for checkpoint selection and evaluation."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1).")
        self.decay = decay
        self.updates = 0
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Blend the current training weights into the average."""
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        current = model.state_dict()
        for name, average in self.module.state_dict().items():
            if average.dtype.is_floating_point:
                average.mul_(decay).add_(current[name].detach(), alpha=1 - decay)
            else:
                average.copy_(current[name])
