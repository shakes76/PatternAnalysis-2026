"""COMP3710 TimeGAN - standalone V4 dataset preprocessing."""


from pathlib import Path


import numpy as np


import pandas as pd


def load_lobster(data_dir):
    """Load matching order book and message CSV files."""
    data_dir = Path(data_dir)

    orderbook_files = list(data_dir.glob("*orderbook_10.csv"))
    message_files = list(data_dir.glob("*message_10.csv"))

    if len(orderbook_files) != 1 or len(message_files) != 1:
        raise FileNotFoundError(
            "Expected one Level-10 orderbook and message CSV pair."
        )

    orderbook = pd.read_csv(orderbook_files[0], header=None)
    messages = pd.read_csv(message_files[0], header=None)

    if orderbook.shape[1] != 40:
        raise ValueError("Level-10 orderbook must have 40 columns.")

    if messages.shape[1] != 6:
        raise ValueError("LOBSTER message file must have 6 columns.")

    if len(orderbook) != len(messages):
        raise ValueError("Orderbook and message row counts differ.")

    return orderbook, messages


def chronological_split(data, train_ratio=0.70, val_ratio=0.15):
    """Split observations in chronological order."""
    n = len(data)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return (
        data[:train_end],
        data[train_end:val_end],
        data[val_end:],
    )


class MinMaxNormalizer:
    """Fit normalization parameters on training data only."""

    def fit(self, train):
        self.minimum = train.min(axis=0)
        maximum = train.max(axis=0)
        self.scale = np.maximum(maximum - self.minimum, 1e-8)
        return self

    def transform(self, data):
        return (data - self.minimum) / self.scale

    def inverse_transform(self, data):
        return data * self.scale + self.minimum


def make_sequences(data, seq_len=24, stride=24):
    """Construct fixed-length temporal windows."""
    if seq_len <= 0 or stride <= 0:
        raise ValueError("seq_len and stride must be positive.")

    if len(data) < seq_len:
        return np.empty(
            (0, seq_len, data.shape[1]), dtype=np.float32
        )

    starts = range(0, len(data) - seq_len + 1, stride)

    return np.stack(
        [data[i:i + seq_len] for i in starts]
    ).astype(np.float32)


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
