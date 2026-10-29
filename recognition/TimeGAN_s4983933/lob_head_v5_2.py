"""COMP3710 TimeGAN V5.2: constrained LOB head plus train-fitted scaling.

Output: 40 interleaved normalized features, with raw relative-price
and log1p-volume representations accessible through forward_raw().

Price columns: ask1, bid1, ..., ask10, bid10 (even feature indices).
Size columns: ask1_size, bid1_size, ..., ask10_size, bid10_size.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ConstraintAwareLOBHeadV52(nn.Module):
    def __init__(self, hidden_dim, minimum, scale, min_spread=0.01,
                 min_gap=0.01, eps=1e-8):
        super().__init__()
        minimum = torch.as_tensor(np.asarray(minimum), dtype=torch.float32)
        scale = torch.as_tensor(np.asarray(scale), dtype=torch.float32)
        if minimum.shape != (40,) or scale.shape != (40,):
            raise ValueError('minimum and scale must each have 40 features')
        if not bool(torch.isfinite(minimum).all()) or not bool(torch.isfinite(scale).all()):
            raise ValueError('nonfinite normalization parameters')
        if not bool((scale > 0).all()):
            raise ValueError('all feature ranges must be positive')
        if min_spread <= 0 or min_gap < 0:
            raise ValueError('invalid price constraints')
        self.register_buffer('minimum', minimum)
        self.register_buffer('scale', scale)
        self.min_spread = float(min_spread)
        self.min_gap = float(min_gap)
        self.eps = float(eps)
        # spread + 9 ask gaps + 9 bid gaps + 20 log1p-volume outputs = 39
        self.projection = nn.Linear(hidden_dim, 39)

    def forward_raw(self, hidden):
        if hidden.ndim != 3:
            raise ValueError('Expected (batch, sequence, hidden_dim)')
        raw = self.projection(hidden)
        spread = self.min_spread + F.softplus(raw[..., 0:1])
        ask_gaps = self.min_gap + F.softplus(raw[..., 1:10])
        bid_gaps = self.min_gap + F.softplus(raw[..., 10:19])
        sizes_log1p = F.softplus(raw[..., 19:39])
        ask1, bid1 = spread / 2.0, -spread / 2.0
        asks = torch.cat([ask1, ask1 + torch.cumsum(ask_gaps, dim=-1)], dim=-1)
        bids = torch.cat([bid1, bid1 - torch.cumsum(bid_gaps, dim=-1)], dim=-1)
        prices = torch.stack([asks, bids], dim=-1).flatten(-2)
        return torch.stack([prices, sizes_log1p], dim=-1).flatten(-2)

    def forward(self, hidden):
        raw = self.forward_raw(hidden)
        return (raw - self.minimum) / self.scale


def check_raw_constraints(x, atol=1e-6):
    if x.shape[-1] != 40:
        raise ValueError('Expected 40 features')
    prices, sizes = x[..., 0::2], x[..., 1::2]
    asks, bids = prices[..., 0::2], prices[..., 1::2]
    return {
        'positive_spread': bool(torch.all(asks[..., 0] > bids[..., 0]).item()),
        'ask_ordered': bool(torch.all(torch.diff(asks, dim=-1) > 0).item()),
        'bid_ordered': bool(torch.all(torch.diff(bids, dim=-1) < 0).item()),
        'ask_above_mid': bool(torch.all(asks > 0).item()),
        'bid_below_mid': bool(torch.all(bids < 0).item()),
        'nonnegative_sizes': bool(torch.all(sizes >= 0).item()),
        'finite': bool(torch.isfinite(x).all().item()),
    }


if __name__ == '__main__':
    torch.manual_seed(42)
    # Synthetic normalization parameters test the module without LOBSTER files.
    minimum = np.zeros(40, dtype=np.float32)
    scale = np.ones(40, dtype=np.float32) * 2
    net = ConstraintAwareLOBHeadV52(24, minimum, scale)
    hidden = torch.randn(8, 24, 24)
    raw = net.forward_raw(hidden)
    norm = net(hidden)
    assert raw.shape == norm.shape == (8, 24, 40)
    assert torch.allclose(norm * net.scale + net.minimum, raw, atol=1e-5)
    for key, passed in check_raw_constraints(raw).items():
        print(f'{key}: {"PASS" if passed else "FAIL"}')
        assert passed
    norm.square().mean().backward()
    assert net.projection.weight.grad is not None
    assert torch.isfinite(net.projection.weight.grad).all()
    print('Scale inversion: PASS')
    print('Gradient flow: PASS')
    print('V5.2 module smoke test: PASS')
