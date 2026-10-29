"""TimeGAN V4.1 dataset: 40 Min-Max LOB features + asinh/Z-score returns.

Only training observations are used to fit the normalisation parameters.
"""

import numpy as np

from dataset import (
    load_lobster,
    chronological_split,
    make_sequences,
    MinMaxNormalizer,
)
from dataset_v2 import RelativeOrderBookTransform


class MixedLOBNormalizer:
    """Reversible normalisation of 41-feature order-book observations."""

    def __init__(self):
        self.book_normalizer = MinMaxNormalizer()
        self.return_scale = None
        self.return_mean = None
        self.return_std = None

    @staticmethod
    def _check_features(values):
        values = np.asarray(values)
        if values.ndim not in (2, 3) or values.shape[-1] != 41:
            raise ValueError("Expected (N, 41) or (N, T, 41) data")
        if not np.all(np.isfinite(values)):
            raise ValueError("Input contains non-finite values")
        return values

    def fit(self, train):
        train = self._check_features(train)
        if train.ndim != 2:
            raise ValueError("Fit requires unique chronological training rows (N, 41)")

        self.book_normalizer.fit(train[:, :40])
        returns = train[:, 40].astype(np.float64)
        nonzero = np.abs(returns[returns != 0.0])
        self.return_scale = float(np.median(nonzero)) if len(nonzero) else 1.0
        if not np.isfinite(self.return_scale) or self.return_scale <= 0:
            raise ValueError("Invalid return scale")

        transformed = np.arcsinh(returns / self.return_scale)
        self.return_mean = float(transformed.mean())
        self.return_std = float(transformed.std())
        if self.return_std < 1e-12:
            self.return_std = 1.0
        return self

    def _require_fit(self):
        if self.return_scale is None:
            raise RuntimeError("Call fit() before transform() or inverse_transform()")

    def transform(self, values):
        self._require_fit()
        values = self._check_features(values)
        shape = values.shape
        flat = values.reshape(-1, 41)

        book = np.asarray(self.book_normalizer.transform(flat[:, :40]))
        transformed_returns = np.arcsinh(
            flat[:, 40].astype(np.float64) / self.return_scale
        )
        transformed_returns = (
            transformed_returns - self.return_mean
        ) / self.return_std

        result = np.column_stack((book, transformed_returns))
        return result.reshape(shape).astype(np.float32)

    def inverse_transform(self, values):
        self._require_fit()
        values = self._check_features(values)
        shape = values.shape
        flat = values.reshape(-1, 41)

        book = np.asarray(self.book_normalizer.inverse_transform(flat[:, :40]))
        z = flat[:, 40].astype(np.float64)
        raw_returns = self.return_scale * np.sinh(
            z * self.return_std + self.return_mean
        )
        result = np.column_stack((book, raw_returns))
        return result.reshape(shape)


def prepare_dataset_v4(data_dir, seq_len=24, stride=24):
    """Prepare independent chronological train/val/test sequence sets."""
    if seq_len < 2:
        raise ValueError("seq_len must be at least 2")
    if stride < 1:
        raise ValueError("stride must be positive")

    orderbook, messages = load_lobster(data_dir)
    orderbook_transform = RelativeOrderBookTransform()
    features40, mid_prices = orderbook_transform.encode(orderbook.to_numpy())

    mid_prices = np.asarray(mid_prices, dtype=np.float64)
    if not np.all(np.isfinite(mid_prices)) or np.any(mid_prices <= 0):
        raise ValueError("Mid-prices must be finite and positive")

    log_mid = np.log(mid_prices)
    returns = np.zeros_like(log_mid)
    returns[1:] = np.diff(log_mid)
    features41 = np.column_stack((features40, returns))

    train, val, test = chronological_split(features41)
    normalizer = MixedLOBNormalizer().fit(train)

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
    print("TimeGAN V4.1 dataset module ready.")
