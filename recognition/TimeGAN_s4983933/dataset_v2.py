
"""
COMP3710 TimeGAN V2 Dataset

Relative mid-price representation for LOBSTER
Level 10 order books.
"""

import numpy as np

from dataset import (
    load_lobster,
    chronological_split,
    make_sequences,
    MinMaxNormalizer,
)


class RelativeOrderBookTransform:
    def __init__(self):
        self.normalizer = MinMaxNormalizer()

    def encode(self, orderbook):
        x = np.asarray(orderbook, dtype=np.float64).copy()

        # Raw LOBSTER prices are stored as price * 10000.
        prices = x[:, 0::2] / 10000.0
        sizes = x[:, 1::2]

        # First ask and first bid are columns 0 and 2.
        mid = (prices[:, 0] + prices[:, 1]) / 2.0

        # Replace absolute prices with mid-price offsets.
        x[:, 0::2] = prices - mid[:, None]

        # Stabilize the scale of order quantities.
        x[:, 1::2] = np.log1p(sizes)

        return x, mid

    def fit(self, train_features):
        self.normalizer.fit(train_features)
        return self

    def transform(self, features):
        return self.normalizer.transform(features)

    def inverse_transform(self, normalized):
        """
        Recover relative price offsets and raw sizes.
        Absolute prices require a separate mid-price series.
        """
        x = self.normalizer.inverse_transform(normalized)
        x = np.asarray(x).copy()

        x[..., 1::2] = np.expm1(x[..., 1::2])
        return x


def prepare_dataset_v2(data_dir, seq_len=24, stride=24):
    orderbook, messages = load_lobster(data_dir)

    transform = RelativeOrderBookTransform()
    features, mid_prices = transform.encode(
        orderbook.to_numpy()
    )

    train, val, test = chronological_split(features)

    transform.fit(train)

    train_norm = transform.transform(train)
    val_norm = transform.transform(val)
    test_norm = transform.transform(test)

    return {
        "train": make_sequences(
            train_norm, seq_len, stride
        ),
        "val": make_sequences(
            val_norm, seq_len, stride
        ),
        "test": make_sequences(
            test_norm, seq_len, stride
        ),
        "transform": transform,
        "mid_prices": mid_prices,
        "timestamps": messages.iloc[:, 0].to_numpy(),
    }


if __name__ == "__main__":
    print("TimeGAN V2 dataset module ready.")
