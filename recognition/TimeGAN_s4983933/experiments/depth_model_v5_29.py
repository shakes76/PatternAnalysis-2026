import math
import torch
from torch import nn


class ProbabilisticDepthNet(nn.Module):
    def __init__(self, levels=20, hidden=128, components=3):
        super().__init__()
        self.levels = levels
        self.components = components

        self.encoder = nn.Sequential(
            nn.Linear(levels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.mask_head = nn.Linear(hidden, levels)

        self.mixture_head = nn.Linear(
            hidden, levels * components * 3
        )

    def forward(self, state):
        h = self.encoder(state)

        mask_logits = self.mask_head(h)

        mixture = self.mixture_head(h).reshape(
            -1, self.levels, self.components, 3
        )

        weights = mixture[..., 0]
        means = mixture[..., 1]
        log_scales = mixture[..., 2].clamp(-2.5, 1.5)

        return mask_logits, weights, means, log_scales


def mixture_nll(target, weights, means, log_scales):
    y = target.unsqueeze(-1)
    normalized_error = (y - means) / torch.exp(log_scales)

    log_density = (
        -0.5 * normalized_error.square()
        - log_scales
        - 0.5 * math.log(2 * math.pi)
    )

    log_mixture = torch.log_softmax(weights, dim=-1)

    return -torch.logsumexp(
        log_mixture + log_density, dim=-1
    )
