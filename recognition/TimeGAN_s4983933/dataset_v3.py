
"""
COMP3710 TimeGAN V4 Dataset

41-feature LOBSTER representation:
- 20 relative mid-price offsets
- 20 log-transformed volumes
- 1 mid-price log return

The final feature stores the return from the previous
raw order-book observation to the current observation.

Normalisation is fitted on training data only.
"""

import numpy as np

from dataset import (
    load_lobster,
    chronological_split,
    make_sequences,
    MinMaxNormalizer,
)
from dataset_v2 import RelativeOrderBookTransform


def prepare_dataset_v3(data_dir, seq_len=24, stride=24):
    if seq_len < 2:
        raise ValueError("seq_len must be at least 2")
    if stride < 1:
        raise ValueError("stride must be positive")

    orderbook, messages = load_lobster(data_dir)

    # Existing 40 features: relative prices and log volumes.
    orderbook_transform = RelativeOrderBookTransform()
    features40, mid_prices = orderbook_transform.encode(
        orderbook.to_numpy()
    )

    if np.any(mid_prices <= 0):
        raise ValueError("Mid-prices must be positive")

    # Causal log returns; no future information is used.
    log_mid = np.log(mid_prices)
    returns = np.zeros_like(log_mid)
    returns[1:] = np.diff(log_mid)

    # Each row now has 41 features.
    features41 = np.column_stack((features40, returns))

    # Split before fitting normalisation statistics.
    train, val, test = chronological_split(features41)

    normalizer = MinMaxNormalizer()
    normalizer.fit(train)

    train_norm = normalizer.transform(train)
    val_norm = normalizer.transform(val)
    test_norm = normalizer.transform(test)

    return {
        "train": make_sequences(train_norm, seq_len, stride),
        "val": make_sequences(val_norm, seq_len, stride),
        "test": make_sequences(test_norm, seq_len, stride),
        "normalizer": normalizer,
        "mid_prices": mid_prices,
        "timestamps": messages.iloc[:, 0].to_numpy(),
        "feature_dim": 41,
    }


if __name__ == "__main__":
    print("TimeGAN V4 dataset module ready.")
