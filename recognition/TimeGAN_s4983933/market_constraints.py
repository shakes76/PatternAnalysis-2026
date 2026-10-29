
"""
COMP3710 TimeGAN - Market Microstructure Validation

Evaluate structural constraints of LOBSTER Level 10
order books in relative mid-price representation.
"""

import numpy as np


def check_market_constraints(data, tolerance=1e-8):
    """
    Evaluate market constraints on decoded order-book data.

    Expected shape: (sequences, steps, 40).

    Price columns contain relative mid-price offsets.
    Size columns contain decoded, nonnegative quantities.
    """
    x = np.asarray(data, dtype=np.float64)

    if x.ndim != 3 or x.shape[-1] != 40:
        raise ValueError(
            f"Expected (N, T, 40), got {x.shape}"
        )

    ask = x[..., 0::4]
    bid = x[..., 2::4]
    sizes = x[..., 1::2]

    spread = ask[..., 0] - bid[..., 0]

    valid_spread = spread > tolerance
    valid_ask_order = np.all(
        np.diff(ask, axis=-1) >= -tolerance,
        axis=-1,
    )
    valid_bid_order = np.all(
        np.diff(bid, axis=-1) <= tolerance,
        axis=-1,
    )
    valid_sizes = np.all(
        sizes >= -tolerance,
        axis=-1,
    )

    all_valid = (
        valid_spread
        & valid_ask_order
        & valid_bid_order
        & valid_sizes
    )

    return {
        "valid_spread_pct": float(valid_spread.mean() * 100),
        "valid_ask_order_pct": float(
            valid_ask_order.mean() * 100
        ),
        "valid_bid_order_pct": float(
            valid_bid_order.mean() * 100
        ),
        "nonnegative_sizes_pct": float(
            valid_sizes.mean() * 100
        ),
        "all_constraints_pct": float(
            all_valid.mean() * 100
        ),
        "mean_best_spread": float(spread.mean()),
        "std_best_spread": float(spread.std()),
    }


def print_report(name, results):
    print(f"\n=== {name} ===")
    for key, value in results.items():
        print(f"{key}: {value:.6f}")
