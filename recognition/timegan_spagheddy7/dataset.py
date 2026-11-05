"""Audit LOBSTER Level-10 data and create chronological PyTorch datasets.

Format reference: the LOBSTER_SampleFiles_ReadMe.txt supplied with the data,
and https://lobsterdata.com/info/DataStructure.php . This module implements
preprocessing only; it does not train TimeGAN or evaluate generated samples.
"""

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


PROJECT_DIR = Path(__file__).resolve().parent
PRICE_COLUMNS = np.arange(0, 40, 2)
VOLUME_COLUMNS = np.arange(1, 40, 2)
PRICE_DIVISOR = 10000.0
FEATURE_NAMES = [
    f"{side}_{quantity}_{level}"
    for level in range(1, 11)
    for side, quantity in (
        ("ask", "log_return"), ("ask", "log1p_volume"),
        ("bid", "log_return"), ("bid", "log1p_volume"),
    )
]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw(data_dir):
    """Read exactly one matched day and reject unsupported/invalid snapshots."""
    candidates = sorted(Path(data_dir).glob("*_orderbook_10.csv"))
    if len(candidates) != 1:
        raise ValueError("Provide a directory containing exactly one Level-10 orderbook CSV.")
    book_path = candidates[0]
    message_path = book_path.with_name(book_path.name.replace("_orderbook_", "_message_"))
    messages = np.loadtxt(message_path, delimiter=",", dtype=np.float64, ndmin=2)
    book = np.loadtxt(book_path, delimiter=",", dtype=np.int64, ndmin=2)
    if book.shape[1] != 40 or messages.shape[1] != 6:
        raise ValueError("Expected 40 order-book columns and 6 message columns.")
    if len(book) != len(messages) or len(book) < 4:
        raise ValueError("Message/order-book row counts must match and contain sufficient rows.")
    if not np.isfinite(messages).all():
        raise ValueError("Messages contain non-finite values.")
    times = messages[:, 0]
    if np.any(np.diff(times) < 0):
        raise ValueError("Message timestamps must be nondecreasing.")

    prefix = book_path.name.removesuffix("_orderbook_10.csv")
    session_name, start_ms, end_ms = prefix.rsplit("_", 2)
    start, end = int(start_ms) / 1000, int(end_ms) / 1000
    if end <= start or times[0] < start or times[-1] > end:
        raise ValueError("Timestamps do not match the session interval in the filename.")
    asks, bids = book[:, 0::4], book[:, 2::4]
    volumes = book[:, VOLUME_COLUMNS]
    checks = {
        "missing_level_entries": int(np.count_nonzero(np.abs(book[:, PRICE_COLUMNS]) >= 9999999999)),
        "nonpositive_price_entries": int(np.count_nonzero(book[:, PRICE_COLUMNS] <= 0)),
        "negative_volume_entries": int(np.count_nonzero(volumes < 0)),
        "nonpositive_spread_rows": int(np.count_nonzero(asks[:, 0] <= bids[:, 0])),
        "unordered_ask_rows": int(np.count_nonzero(np.any(np.diff(asks, axis=1) <= 0, axis=1))),
        "unordered_bid_rows": int(np.count_nonzero(np.any(np.diff(bids, axis=1) >= 0, axis=1))),
        "trading_halt_rows": int(np.count_nonzero(messages[:, 1] == 7)),
    }
    if any(checks.values()):
        raise ValueError(f"Raw data needs explicit cleaning or halt handling before use: {checks}")
    event_types, counts = np.unique(messages[:, 1].astype(int), return_counts=True)
    metadata = {
        "session": session_name,
        "rows": len(book),
        "levels": 10,
        "book_columns": 40,
        "session_seconds": [start, end],
        "timestamp_range_seconds": [float(times[0]), float(times[-1])],
        "timestamps_nondecreasing": True,
        "event_type_counts": dict(zip(map(str, event_types), map(int, counts))),
        "raw_integrity": checks,
        "files": [
            {"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in (message_path, book_path)
        ],
    }
    return book, times, metadata


def make_features(book):
    """Create 20 price log returns and 20 current log1p volumes per event.

    Call separately inside each chronological split. The first state is an
    anchor, so the first return never uses a state from an earlier split.
    """
    features = np.empty((len(book) - 1, 40), dtype=np.float64)
    prices = book[:, PRICE_COLUMNS].astype(np.float64)
    features[:, PRICE_COLUMNS] = np.log(prices[1:] / prices[:-1])
    features[:, VOLUME_COLUMNS] = np.log1p(book[1:, VOLUME_COLUMNS])
    return features


@dataclass
class FeatureScaler:
    minimum: np.ndarray
    maximum: np.ndarray

    @classmethod
    def fit(cls, training_features):
        return cls(training_features.min(axis=0), training_features.max(axis=0))

    @property
    def span(self):
        difference = self.maximum - self.minimum
        return np.where(difference > 0, difference, 1.0)

    def transform(self, features):
        # Do not clip validation/test values: that would hide distribution shift.
        return ((features - self.minimum) / self.span).astype(np.float32)

    def inverse_transform(self, features):
        return np.asarray(features, dtype=np.float64) * self.span + self.minimum

    def to_dict(self):
        return {
            "method": "per-feature min-max, training period only, no clipping",
            "feature_names": FEATURE_NAMES,
            "minimum": self.minimum.tolist(),
            "maximum": self.maximum.tolist(),
            "constant_feature_count": int(np.count_nonzero(self.maximum == self.minimum)),
        }


def decode_features(features, scaler, initial_prices):
    """Invert (..., updates, features) sequences to USD prices and share volumes.

    Initial prices are conditioning information, not generated values. No
    clipping, reordering, or nonnegativity repair is applied, so subsequent
    model evaluation can detect invalid generated books honestly.
    """
    restored = scaler.inverse_transform(features)
    anchors = np.asarray(initial_prices, dtype=np.float64)
    if restored.ndim not in (2, 3) or restored.shape[-1] != 40:
        raise ValueError("Expected (updates, 40) or (batch, updates, 40) features.")
    if anchors.shape != (*restored.shape[:-2], 20):
        raise ValueError("Supply one 20-price anchor per sequence.")
    book = np.empty_like(restored)
    book[..., PRICE_COLUMNS] = anchors[..., None, :] * np.exp(np.cumsum(restored[..., PRICE_COLUMNS], axis=-2))
    book[..., VOLUME_COLUMNS] = np.expm1(restored[..., VOLUME_COLUMNS])
    return book


class LOBWindowDataset(Dataset):
    """Lazy windows within a single split; no duplicated window-sized arrays."""

    def __init__(self, book, times, scaled_features, sequence_length, stride, raw_offset):
        self.book = book
        self.times = times
        self.features = scaled_features
        self.sequence_length = sequence_length
        self.raw_offset = raw_offset
        self.starts = np.arange(0, len(scaled_features) - sequence_length + 1, stride)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = int(self.starts[index])
        stop = start + self.sequence_length
        return {
            "features": torch.from_numpy(self.features[start:stop]),
            "initial_prices": torch.from_numpy(
                self.book[start, PRICE_COLUMNS].astype(np.float64) / PRICE_DIVISOR
            ),
            "timestamps": torch.from_numpy(self.times[start + 1:stop + 1].copy()),
            "raw_start_index": self.raw_offset + start,
        }


def prepare_datasets(data_dir, sequence_length=32, stride=32, fractions=(0.70, 0.15, 0.15)):
    if sequence_length < 2 or stride < 1:
        raise ValueError("Sequence length must be at least 2 and stride must be positive.")
    if len(fractions) != 3 or min(fractions) <= 0 or not np.isclose(sum(fractions), 1):
        raise ValueError("Provide three positive split fractions summing to one.")
    book, times, report = load_raw(data_dir)
    session_start, session_end = report["session_seconds"]
    cuts = session_start + (session_end - session_start) * np.cumsum(fractions)[:2]
    # side='left' keeps all simultaneous events together on the later side.
    indices = [0, *np.searchsorted(times, cuts, side="left").tolist(), len(book)]
    bounds = dict(zip(("train", "validation", "test"), zip(indices[:-1], indices[1:])))
    if any(stop - start < sequence_length + 1 for start, stop in bounds.values()):
        raise ValueError("Each time split needs at least sequence_length + 1 raw states.")
    train_start, train_stop = bounds["train"]
    scaler = FeatureScaler.fit(make_features(book[train_start:train_stop]))
    datasets, split_report = {}, {}
    for name, (start, stop) in bounds.items():
        features = scaler.transform(make_features(book[start:stop]))
        if not np.isfinite(features).all():
            raise ValueError(f"Non-finite transformed features in {name}.")
        datasets[name] = LOBWindowDataset(
            book[start:stop], times[start:stop], features, sequence_length, stride, start
        )
        split_report[name] = {
            "raw_row_interval_half_open": [start, stop],
            "raw_rows": stop - start,
            "timestamp_range_seconds": [float(times[start]), float(times[stop - 1])],
            "feature_rows": len(features),
            "windows": len(datasets[name]),
        }
    report.update(
        split_method="70/15/15 of elapsed session time by default; split before returns and windows",
        split_fractions=list(fractions),
        cut_times_seconds=cuts.tolist(),
        sequence_length=sequence_length,
        stride=stride,
        features=40,
        sampling="event time; irregular wall-clock intervals are retained as metadata",
        initial_state="Each sequence requires its own 20-price anchor for reconstruction.",
        splits=split_report,
        scaler=scaler.to_dict(),
    )
    return datasets, scaler, report


def save_training_preview(training, destination):
    """Plot only a prefix of the training period; no held-out outcomes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sample = training.book[:2000]
    event_index = np.arange(len(sample))
    depth = np.concatenate((sample[:, 1::4][:, ::-1], sample[:, 3::4]), axis=1).T
    fig, (prices, volumes) = plt.subplots(2, 1, figsize=(11, 6.5), layout="constrained")
    prices.plot(event_index, sample[:, 0] / PRICE_DIVISOR, label="Best ask", color="#d55e00")
    prices.plot(event_index, sample[:, 2] / PRICE_DIVISOR, label="Best bid", color="#0072b2")
    prices.set(ylabel="Price (USD)", title="AMZN: first 2,000 training-period order-book states")
    prices.legend(loc="best")
    prices.grid(alpha=0.2)
    heatmap = volumes.imshow(np.log1p(depth), aspect="auto", cmap="viridis", interpolation="nearest")
    volumes.axhline(9.5, color="white", linewidth=1)
    volumes.set(xlabel="Event index (not evenly spaced clock time)", ylabel="Book level")
    volumes.set_yticks([0, 5, 9, 10, 14, 19], ["Ask 10", "Ask 5", "Ask 1", "Bid 1", "Bid 5", "Bid 10"])
    fig.colorbar(heatmap, ax=volumes, label="log(1 + shares)")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "reports")
    args = parser.parse_args()
    print("Auditing raw data and preparing chronological windows...", flush=True)
    datasets, scaler, report = prepare_datasets(args.data_dir, args.sequence_length, args.stride)

    # Numerical inverse check on a training sequence, not a model-quality metric.
    example = datasets["train"][0]
    restored = decode_features(example["features"].numpy(), scaler, example["initial_prices"].numpy())
    expected = datasets["train"].book[1:args.sequence_length + 1].astype(np.float64)
    expected[:, PRICE_COLUMNS] /= PRICE_DIVISOR
    np.testing.assert_allclose(restored[:, PRICE_COLUMNS], expected[:, PRICE_COLUMNS], rtol=0, atol=1e-5)
    np.testing.assert_allclose(restored[:, VOLUME_COLUMNS], expected[:, VOLUME_COLUMNS], rtol=1e-5, atol=1e-3)
    report["training_round_trip"] = {
        "max_price_error_usd": float(np.max(np.abs(restored[:, PRICE_COLUMNS] - expected[:, PRICE_COLUMNS]))),
        "max_volume_error_shares": float(np.max(np.abs(restored[:, VOLUME_COLUMNS] - expected[:, VOLUME_COLUMNS]))),
    }
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = next(iter(DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=False)))
    features = batch["features"].to(device)
    if not torch.isfinite(features).all().item():
        raise ValueError("The device minibatch contains non-finite values.")
    report["device_check"] = {
        "device": str(device),
        "name": torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "batch_shape": list(features.shape),
        "batch_memory_mib": features.numel() * features.element_size() / 2**20,
        "scope": "Minibatch transfer and finite-value check only; no TimeGAN training benchmark.",
    }
    report["status"] = "data and device checks passed; models and feasibility review not yet implemented"
    report["limitations"] = [
        "One stock on one trading day; later intraday performance is not cross-day generalisation.",
        "Event-time sequences do not model event-arrival times; anchors are supplied rather than generated.",
        "Standalone chronological splits discard the return at each split boundary.",
        "Training-only min-max scaling can place future values outside [0, 1]; these are not clipped.",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_training_preview(datasets["train"], PROJECT_DIR / "figures" / "training_data_preview.png")
    report_path = args.output_dir / "data_audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"splits": report["splits"], "round_trip": report["training_round_trip"], "device": report["device_check"]}, indent=2))
    print(f"Saved audit: {report_path}")


if __name__ == "__main__":
    main()
