"""Shared validation diagnostics; no fitting, clipping or structural repair."""

import numpy as np
import torch
from scipy.ndimage import uniform_filter

from dataset import PRICE_COLUMNS, PRICE_DIVISOR, VOLUME_COLUMNS, decode_features


def validation_windows(dataset, max_lags):
    """Use common starts, with all observed conditioning history inside this split."""
    starts = dataset.starts[dataset.starts >= max_lags]
    if len(starts) == 0:
        raise ValueError("No complete windows have enough within-split history.")
    anchors = dataset.book[starts][:, PRICE_COLUMNS].astype(np.float64) / PRICE_DIVISOR
    real = np.stack([dataset.book[s + 1:s + dataset.sequence_length + 1] for s in starts]).astype(np.float64)
    real[..., PRICE_COLUMNS] /= PRICE_DIVISOR
    return starts, anchors, real


def conditioning_history(dataset, starts, lags):
    indices = starts[:, None] + np.arange(-lags, 0)
    if np.any(indices < 0):
        raise ValueError("Conditioning history would cross a split boundary.")
    return torch.as_tensor(dataset.features[indices], dtype=torch.float64)


def decode_batch(features, scaler, anchors):
    values = features.detach().cpu().numpy() if isinstance(features, torch.Tensor) else np.asarray(features)
    if len(values) != len(anchors):
        raise ValueError("Each generated sequence requires its own price anchor.")
    with np.errstate(over="raise", invalid="raise"):
        books = decode_features(values, scaler, anchors)
    if not np.isfinite(books).all():
        raise FloatingPointError("Non-finite decoded books; no repair or sample filtering applied.")
    return books


def validity_rates(books):
    """Strict violations per snapshot; categories overlap and include all ten levels."""
    asks, bids, volumes = books[..., 0::4], books[..., 2::4], books[..., VOLUME_COLUMNS]
    flags = {
        "nonfinite_rate": ~np.isfinite(books).all(axis=-1),
        "nonpositive_price_rate": np.any(books[..., PRICE_COLUMNS] <= 0, axis=-1),
        "crossed_rate": asks[..., 0] <= bids[..., 0],
        "ask_order_rate": np.any(np.diff(asks, axis=-1) <= 0, axis=-1),
        "bid_order_rate": np.any(np.diff(bids, axis=-1) >= 0, axis=-1),
        "negative_volume_rate": np.any(volumes < 0, axis=-1),
    }
    flags["invalid_rate"] = np.logical_or.reduce(list(flags.values()))
    return {name: float(flag.mean()) for name, flag in flags.items()}


def market_series(books, anchors):
    spread = books[..., 0] - books[..., 2]
    mid = (books[..., 0] + books[..., 2]) / 2
    initial_mid = (anchors[:, 0] + anchors[:, 1]) / 2
    if not np.isfinite(books).all() or np.any(mid <= 0) or np.any(initial_mid <= 0):
        raise FloatingPointError("Distribution metrics require finite books and positive mid-prices.")
    returns = np.diff(np.log(np.column_stack((initial_mid, mid))), axis=1)
    return {"spread": spread, "mid_return": returns}


def metric_specification(training, bins=50, max_lag=10, smoothing=1e-8):
    """Freeze bin boundaries from training only, with separate infinite tail bins."""
    book = training.book.astype(np.float64)
    spreads = (book[:, 0] - book[:, 2]) / PRICE_DIVISOR
    mid = (book[:, 0] + book[:, 2]) / (2 * PRICE_DIVISOR)
    series = {"spread": spreads, "mid_return": np.diff(np.log(mid))}
    boundaries = {}
    for name, values in series.items():
        low, high = float(values.min()), float(values.max())
        if low == high:
            low, high = low - 1e-6, high + 1e-6
        boundaries[name] = np.linspace(low, high, bins + 1).tolist()
    return {"boundaries": boundaries, "regular_bins": bins, "tail_bins": 2, "smoothing": smoothing,
            "max_lag": max_lag, "kl_direction": "real || generated", "logarithm": "natural",
            "bin_source": "complete training segment only", "structural_repair": "none",
            "depth_ssim": depth_ssim_specification(training)}


def depth_ssim_specification(training):
    maximum = float(np.log1p(training.book[:, VOLUME_COLUMNS].max()))
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("Need finite, positive training depth to define the SSIM scale.")
    return {
        "transform": "sign(volume) * log1p(abs(volume)) / training_log_volume_max",
        "training_log_volume_max": maximum, "data_range": 1.0, "window_size": 7,
        "weights": "uniform", "sample_covariance": True, "k1": 0.01, "k2": 0.03,
        "images": "Separate ask and bid images, each 10 levels x sequence_length updates; levels 1 through 10.",
        "aggregation": "Mean over valid 7x7 windows, then equal mean of sides, sequences and generation seeds.",
        "clipping": "none; negative or above-training-range depths stay in the score and validity checks",
        "reference_range": "Nominal training reference range [0, 1]; fixed even for held-out or generated extremes.",
        "threshold": 0.6, "threshold_rule": "strictly greater than", "higher_is_better": True,
        "reference": "https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf",
        "variant": "Wang et al. SSIM equation; 7x7 uniform windows/sample covariance, not the paper's 11x11 Gaussian setting.",
    }


def depth_images(books, spec):
    """Numeric depth images (batch, side, level, time), never colour-rendered PNGs."""
    if books.ndim != 3 or books.shape[-1] != 40 or not np.isfinite(books).all():
        raise ValueError("SSIM requires finite (batch, updates, 40) order books.")
    scale = spec["training_log_volume_max"]
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("SSIM training scale must be positive and finite.")
    volumes = np.stack((books[..., 1::4], books[..., 3::4]), axis=1).swapaxes(-1, -2)
    return np.sign(volumes) * np.log1p(np.abs(volumes)) / scale


class DepthSSIM:
    """Cached local SSIM using Wang et al. (2004), equation 13.

    Formula and parameter meanings: https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf
    Uniform-window convention: https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity
    Filter only the last two axes; neither sequence nor bid/ask boundaries mix.
    """

    def __init__(self, real, spec):
        self.spec, self.x = spec, depth_images(real, spec)
        self.width = spec["window_size"]
        if self.width < 3 or self.width % 2 == 0 or min(self.x.shape[-2:]) < self.width:
            raise ValueError("SSIM needs an odd window fitting both depth-image dimensions.")
        self.covariance_factor = self.width ** 2 / (self.width ** 2 - 1) if spec["sample_covariance"] else 1
        self.mean_x = self.local_mean(self.x)
        self.var_x = self.covariance_factor * (self.local_mean(self.x ** 2) - self.mean_x ** 2)

    def local_mean(self, values):
        return uniform_filter(values, size=(1, 1, self.width, self.width), mode="reflect")

    def score(self, generated):
        y = depth_images(generated, self.spec)
        if y.shape != self.x.shape:
            raise ValueError("SSIM pairs must have matching sequence, side, level and time dimensions.")
        mean_y = self.local_mean(y)
        var_y = self.covariance_factor * (self.local_mean(y ** 2) - mean_y ** 2)
        covariance = self.covariance_factor * (self.local_mean(self.x * y) - self.mean_x * mean_y)
        c1, c2 = (self.spec["k1"] * self.spec["data_range"]) ** 2, (self.spec["k2"] * self.spec["data_range"]) ** 2
        numerator = (2 * self.mean_x * mean_y + c1) * (2 * covariance + c2)
        denominator = (self.mean_x ** 2 + mean_y ** 2 + c1) * (self.var_x + var_y + c2)
        pad = self.width // 2
        # Only windows wholly inside the image contribute; no padding in the reported score.
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            by_side = (numerator / denominator)[..., pad:-pad, pad:-pad].mean(axis=(-2, -1))
        return by_side.mean(axis=1), by_side


def histogram_probabilities(values, boundaries, smoothing):
    if not np.isfinite(values).all():
        raise FloatingPointError("Histogram inputs must be finite; do not silently drop invalid samples.")
    counts = np.histogram(np.asarray(values).ravel(), bins=np.r_[-np.inf, boundaries, np.inf])[0]
    probabilities = counts / counts.sum() + smoothing
    return probabilities / probabilities.sum()


def histogram_kl(real, generated, boundaries, smoothing=1e-8):
    p = histogram_probabilities(real, boundaries, smoothing)
    q = histogram_probabilities(generated, boundaries, smoothing)
    return float(np.sum(p * np.log(p / q)))


def absolute_return_acf(returns, max_lag=10):
    """Pooled, globally centred ACF with no pairs crossing 32-update windows.

    Divide each lag's cross-product sum by the full sum of squared deviations.
    This biased estimate is bounded; constant absolute returns give zero ACF.
    """
    if returns.ndim != 2 or max_lag >= returns.shape[1] or max_lag < 1:
        raise ValueError("Need (windows, updates) returns and 1 <= max_lag < updates.")
    values = np.abs(returns)
    centred = values - values.mean()
    denominator = np.sum(centred ** 2)
    if denominator == 0:
        return np.zeros(max_lag)
    return np.array([np.sum(centred[:, :-lag] * centred[:, lag:]) / denominator for lag in range(1, max_lag + 1)])


def evaluate_books(real, generated, anchors, spec):
    return ValidationEvaluator(real, anchors, spec).evaluate(generated)[0]


class ValidationEvaluator:
    """Cache real reference statistics once for a shared validation window set."""

    def __init__(self, real, anchors, spec):
        self.shape, self.anchors, self.spec = real.shape, anchors, spec
        actual = market_series(real, anchors)
        self.probabilities = self.histograms(actual)
        self.acf = absolute_return_acf(actual["mid_return"], spec["max_lag"])
        self.depth = DepthSSIM(real, spec["depth_ssim"])

    def histograms(self, series):
        return {name: histogram_probabilities(values, self.spec["boundaries"][name], self.spec["smoothing"])
                for name, values in series.items()}

    def evaluate(self, generated):
        if generated.shape != self.shape or generated.ndim != 3 or generated.shape[-1] != 40:
            raise ValueError("Real and generated books must have identical (windows, updates, 40) shapes.")
        simulated = market_series(generated, self.anchors)
        probabilities = self.histograms(simulated)
        acf = absolute_return_acf(simulated["mid_return"], self.spec["max_lag"])
        scores = validity_rates(generated)
        for name, p in self.probabilities.items():
            scores[f"kl_{name}"] = float(np.sum(p * np.log(p / probabilities[name])))
        scores["abs_return_acf_mae"] = float(np.abs(self.acf - acf).mean())
        scores["mean_kl"] = (scores["kl_spread"] + scores["kl_mid_return"]) / 2
        per_sequence, by_side = self.depth.score(generated)
        scores.update(depth_ssim=float(per_sequence.mean()), depth_ssim_ask=float(by_side[:, 0].mean()),
                      depth_ssim_bid=float(by_side[:, 1].mean()),
                      depth_ssim_above_target_rate=float((per_sequence > self.spec["depth_ssim"]["threshold"]).mean()))
        return scores, {"acf": acf, "depth_ssim_per_sequence": per_sequence, **probabilities}
