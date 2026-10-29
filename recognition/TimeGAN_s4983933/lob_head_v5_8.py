import torch
from torch import nn

from lob_head_v5_2 import ConstraintAwareLOBHeadV52


class MLPConstraintAwareLOBHead(ConstraintAwareLOBHeadV52):
    """
    Nonlinear LOB head using the existing
    constraint-aware price construction.

    Input:  (batch, sequence, hidden_dim)
    Output: (batch, sequence, 40)
    """

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

        # Replace only the parameter projection.
        # Keep the original constrained forward
        # and normalization logic unchanged.
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, intermediate),
            nn.ReLU(),
            nn.Linear(intermediate, intermediate),
            nn.ReLU(),
            nn.Linear(intermediate, 39),
        )


if __name__ == "__main__":
    import numpy as np

    minimum = np.zeros(40)
    scale = np.ones(40)

    torch.manual_seed(42)

    model = MLPConstraintAwareLOBHead(
        hidden_dim=24,
        minimum=minimum,
        scale=scale,
        intermediate=128,
    )

    hidden = torch.randn(8, 24, 24)

    raw = model.forward_raw(hidden)
    normalized = model(hidden)

    assert raw.shape == (8, 24, 40)
    assert normalized.shape == (8, 24, 40)

    loss = normalized.square().mean()
    loss.backward()

    assert all(
        p.grad is not None
        for p in model.parameters()
        if p.requires_grad
    )

    print("=== V5.8 MLP Constraint-aware Head ===")
    print("Raw output shape:", tuple(raw.shape))
    print("Normalized output shape:",
          tuple(normalized.shape))
    print("Gradient flow: PASS")
    print("V5.8 architecture test: PASS")
