"""
COMP3710 Project 7 - TimeGAN-based Limit Order Book Generation.

Model definitions:
1. TimeGAN and GRU components
2. Constrained price generation head
3. Probabilistic depth generation network

Neural network definitions use PyTorch only.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


# ============================================================
# 1. BASE TIMEGAN
# ============================================================

class GRUNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers=2,
        output_activation="sigmoid",
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.projection = nn.Linear(
            hidden_dim, output_dim
        )

        if output_activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif output_activation == "none":
            self.activation = nn.Identity()
        else:
            raise ValueError(
                f"Unknown activation: {output_activation}"
            )

    def forward(self, x):
        hidden_states, _ = self.gru(x)
        output = self.projection(hidden_states)
        return self.activation(output)


class TimeGAN(nn.Module):
    def __init__(
        self,
        feature_dim=40,
        hidden_dim=24,
        noise_dim=40,
        num_layers=2,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim

        self.embedder = GRUNetwork(
            feature_dim, hidden_dim, hidden_dim,
            num_layers=num_layers,
        )

        self.recovery = GRUNetwork(
            hidden_dim, hidden_dim, feature_dim,
            num_layers=num_layers,
        )

        self.generator = GRUNetwork(
            noise_dim, hidden_dim, hidden_dim,
            num_layers=num_layers,
        )

        self.supervisor = GRUNetwork(
            hidden_dim, hidden_dim, hidden_dim,
            num_layers=num_layers - 1,
        )

        self.discriminator = GRUNetwork(
            hidden_dim, hidden_dim, 1,
            num_layers=num_layers,
            output_activation="none",
        )

    def encode(self, x):
        return self.embedder(x)

    def reconstruct(self, x):
        return self.recovery(
            self.embedder(x)
        )

    def generate(self, z):
        e_hat = self.generator(z)
        h_hat = self.supervisor(e_hat)
        return self.recovery(h_hat)

    def discriminate(self, h):
        return self.discriminator(h)


class TimeGANV32(TimeGAN):
    """Identity activation for Generator."""

    def __init__(
        self,
        feature_dim=40,
        hidden_dim=24,
        noise_dim=40,
        num_layers=2,
    ):
        super().__init__(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            noise_dim=noise_dim,
            num_layers=num_layers,
        )

        self.generator.activation = nn.Identity()


class TimeGANV42(TimeGANV32):
    """Identity activations for Generator and Recovery."""

    def __init__(
        self,
        feature_dim=41,
        hidden_dim=24,
        noise_dim=41,
        num_layers=2,
    ):
        super().__init__(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            noise_dim=noise_dim,
            num_layers=num_layers,
        )

        self.recovery.activation = nn.Identity()


# ============================================================
# 2. CONSTRAINT-AWARE PRICE HEAD
# ============================================================

class ConstraintAwareLOBHeadV52(nn.Module):
    def __init__(
        self,
        hidden_dim,
        minimum,
        scale,
        min_spread=0.01,
        min_gap=0.01,
        eps=1e-8,
    ):
        super().__init__()

        minimum = torch.as_tensor(
            minimum, dtype=torch.float32
        )
        scale = torch.as_tensor(
            scale, dtype=torch.float32
        )

        if minimum.shape != (40,) or scale.shape != (40,):
            raise ValueError(
                "minimum and scale must each have 40 features"
            )

        if not bool(torch.isfinite(minimum).all()):
            raise ValueError("Nonfinite minimum")

        if not bool(torch.isfinite(scale).all()):
            raise ValueError("Nonfinite scale")

        if not bool((scale > 0).all()):
            raise ValueError("Scale must be positive")

        if min_spread <= 0 or min_gap < 0:
            raise ValueError("Invalid price constraints")

        self.register_buffer("minimum", minimum)
        self.register_buffer("scale", scale)

        self.min_spread = float(min_spread)
        self.min_gap = float(min_gap)
        self.eps = float(eps)

        self.projection = nn.Linear(
            hidden_dim, 39
        )

    def forward_raw(self, hidden):
        if hidden.ndim != 3:
            raise ValueError(
                "Expected (batch, sequence, hidden_dim)"
            )

        raw = self.projection(hidden)

        spread = (
            self.min_spread
            + F.softplus(raw[..., 0:1])
        )

        ask_gaps = (
            self.min_gap
            + F.softplus(raw[..., 1:10])
        )

        bid_gaps = (
            self.min_gap
            + F.softplus(raw[..., 10:19])
        )

        sizes_log1p = F.softplus(
            raw[..., 19:39]
        )

        ask1 = spread / 2.0
        bid1 = -spread / 2.0

        asks = torch.cat(
            [
                ask1,
                ask1 + torch.cumsum(
                    ask_gaps, dim=-1
                ),
            ],
            dim=-1,
        )

        bids = torch.cat(
            [
                bid1,
                bid1 - torch.cumsum(
                    bid_gaps, dim=-1
                ),
            ],
            dim=-1,
        )

        prices = torch.stack(
            [asks, bids], dim=-1
        ).flatten(-2)

        return torch.stack(
            [prices, sizes_log1p],
            dim=-1,
        ).flatten(-2)

    def forward(self, hidden):
        raw = self.forward_raw(hidden)
        return (
            raw - self.minimum
        ) / self.scale


class MLPConstraintAwareLOBHead(
    ConstraintAwareLOBHeadV52
):
    def __init__(
        self,
        hidden_dim,
        minimum,
        scale,
        min_spread=0.01,
        min_gap=0.01,
        intermediate=128,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            minimum=minimum,
            scale=scale,
            min_spread=min_spread,
            min_gap=min_gap,
        )

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, intermediate),
            nn.ReLU(),
            nn.Linear(intermediate, intermediate),
            nn.ReLU(),
            nn.Linear(intermediate, 39),
        )


def check_raw_constraints(x):
    if x.shape[-1] != 40:
        raise ValueError("Expected 40 features")

    prices = x[..., 0::2]
    sizes = x[..., 1::2]

    asks = prices[..., 0::2]
    bids = prices[..., 1::2]

    return {
        "positive_spread": bool(
            torch.all(asks[..., 0] > bids[..., 0]).item()
        ),
        "ask_ordered": bool(
            torch.all(
                torch.diff(asks, dim=-1) > 0
            ).item()
        ),
        "bid_ordered": bool(
            torch.all(
                torch.diff(bids, dim=-1) < 0
            ).item()
        ),
        "ask_above_mid": bool(
            torch.all(asks > 0).item()
        ),
        "bid_below_mid": bool(
            torch.all(bids < 0).item()
        ),
        "nonnegative_sizes": bool(
            torch.all(sizes >= 0).item()
        ),
        "finite": bool(
            torch.isfinite(x).all().item()
        ),
    }


# ============================================================
# 3. PROBABILISTIC DEPTH NETWORK
# ============================================================

class ProbabilisticDepthNet(nn.Module):
    def __init__(
        self,
        levels=20,
        hidden=128,
        components=3,
    ):
        super().__init__()

        self.levels = levels
        self.components = components

        self.encoder = nn.Sequential(
            nn.Linear(levels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.mask_head = nn.Linear(
            hidden, levels
        )

        self.mixture_head = nn.Linear(
            hidden,
            levels * components * 3
        )

    def forward(self, state):
        h = self.encoder(state)

        mask_logits = self.mask_head(h)

        mixture = self.mixture_head(h).reshape(
            -1,
            self.levels,
            self.components,
            3,
        )

        weights = mixture[..., 0]
        means = mixture[..., 1]
        log_scales = mixture[..., 2].clamp(
            -2.5, 1.5
        )

        return (
            mask_logits,
            weights,
            means,
            log_scales,
        )


def mixture_nll(
    target,
    weights,
    means,
    log_scales,
):
    y = target.unsqueeze(-1)

    normalized_error = (
        y - means
    ) / torch.exp(log_scales)

    log_density = (
        -0.5 * normalized_error.square()
        - log_scales
        - 0.5 * math.log(2 * math.pi)
    )

    log_mixture = torch.log_softmax(
        weights, dim=-1
    )

    return -torch.logsumexp(
        log_mixture + log_density,
        dim=-1,
    )


# V5.14: Multi-factor initial volume generator
import torch.nn.functional as F

class MultiFactorVolumeHead(nn.Module):
    def __init__(self, hidden_dim, mean_log, std_log):
        super().__init__()

        # Separate distribution parameters per level.
        init_mu = mean_log + torch.log(
            -torch.expm1(-mean_log.clamp_min(0.1))
        )
        self.mu = nn.Parameter(init_mu.clone())
        self.log_sigma = nn.Parameter(
            std_log.clamp_min(0.1).log()
        )

        # Independent AR coefficients for 20 levels.
        self.rho_logits = nn.Parameter(
            torch.full((20,), 2.2)
        )

        # Limited shared TimeGAN conditioning.
        self.condition = nn.Linear(
            hidden_dim, 20, bias=False
        )
        nn.init.zeros_(self.condition.weight)

    def forward(self, hidden, innovations):
        rho = torch.sigmoid(
            self.rho_logits
        ).clamp(0.05, 0.98)

        noise = []
        previous = innovations[:, 0]
        noise.append(previous)

        for t in range(1, innovations.shape[1]):
            previous = (
                rho * previous
                + torch.sqrt(
                    1.0 - rho.square()
                ) * innovations[:, t]
            )
            noise.append(previous)

        factors = torch.stack(noise, dim=1)

        shared = (
            0.15 * torch.tanh(
                self.condition(hidden)
            )
        )

        raw = (
            self.mu
            + torch.exp(self.log_sigma) * factors
            + shared
        )

        # Positive log1p volumes.
        return F.softplus(raw)
