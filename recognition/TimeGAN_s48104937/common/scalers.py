from __future__ import annotations

import numpy as np

# --- Custom Scalder Classes ---
class LogMinMaxScaler:
    """Signed-log compression followed by per-feature min-max scaling to [0, 1].

    Works on arrays of any shape whose last axis is the features, e.g. 3D
    windows. Fit on the training windows only; overlapping windows repeat
    rows, but min and max are unaffected by repetition. inverse_transform
    undoes both steps, returning values in tick/share units (unrounded).
    """
    def __init__(self, lo: np.ndarray | None = None, hi: np.ndarray | None = None):
        self.lo = None if lo is None else np.asarray(lo, dtype=float)
        self.hi = None if hi is None else np.asarray(hi, dtype=float)

    @property
    def span(self) -> np.ndarray:
        """Width of each feature's log range; the small constant avoids dividing by 0."""
        if self.lo is None or self.hi is None:
            raise RuntimeError("Scaler is not fitted: call fit() or pass lo and hi")
        return self.hi - self.lo + 1e-8

    def fit(self, data: np.ndarray) -> LogMinMaxScaler:
        """Learn each feature's min and max in signed-log space."""
        logged = self._signed_log(data.reshape(-1, data.shape[-1]))
        self.lo = logged.min(axis=0)
        self.hi = logged.max(axis=0)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Signed log, then scale each feature to [0, 1] using the fitted range."""
        return (self._signed_log(data) - self.lo) / self.span

    def inverse_transform(self, scaled: np.ndarray) -> np.ndarray:
        """Undo transform: back to the original units (unrounded)."""
        logged = scaled * self.span + self.lo
        return np.sign(logged) * np.expm1(np.abs(logged))

    @staticmethod
    def _signed_log(x: np.ndarray) -> np.ndarray:
        """log1p that keeps the sign, so values of any sign compress symmetrically."""
        return np.sign(x) * np.log1p(np.abs(x))

class QuantileScaler:
    """Per-feature empirical CDF: maps each feature to [0, 1] by its training quantiles.

    Works on arrays of any shape whose last axis is the features. Ties (repeated
    lot sizes) map to the middle of their plateau. inverse_transform is clipped
    to the training range, returning values in original units (unrounded).
    """
    def __init__(self, n_quantiles: int = 1000, quantiles: np.ndarray | None = None):
        self.n_quantiles = n_quantiles
        self.quantiles = None if quantiles is None else np.asarray(quantiles, dtype=float)

    @property
    def refs(self) -> np.ndarray:
        """Evenly spaced CDF levels matching the stored quantiles."""
        if self.quantiles is None:
            raise RuntimeError("Scaler is not fitted: call fit() or pass quantiles")
        return np.linspace(0, 1, self.quantiles.shape[0])

    def fit(self, data: np.ndarray) -> QuantileScaler:
        """Learn each feature's quantiles."""
        flat = data.reshape(-1, data.shape[-1])
        self.quantiles = np.quantile(flat, np.linspace(0, 1, self.n_quantiles), axis=0)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Map each feature through its empirical CDF to [0, 1]."""
        q, r = self.quantiles, self.refs
        out = np.empty(data.shape, dtype=float)
        for j in range(data.shape[-1]):
            x = data[..., j]
            # Average forward and reverse interpolation so ties land mid-plateau
            out[..., j] = 0.5 * (np.interp(x, q[:, j], r) - np.interp(-x, -q[::-1, j], -r[::-1]))
        return out

    def inverse_transform(self, scaled: np.ndarray) -> np.ndarray:
        """Undo transform: back to the original units (unrounded)."""
        q, r = self.quantiles, self.refs
        out = np.empty(scaled.shape, dtype=float)
        for j in range(scaled.shape[-1]):
            out[..., j] = np.interp(scaled[..., j], r, q[:, j])
        return out

class DualScaler:
    """Scales the feature columns in `cols` with `inner` and every other column with `outer`."""
    def __init__(self, n_features: int, cols: list[int], inner, outer):
        self.cols = np.asarray(cols)
        self.rest = np.setdiff1d(np.arange(n_features), self.cols)
        self.inner = inner
        self.outer = outer

    def fit(self, data: np.ndarray) -> DualScaler:
        self.inner.fit(data[..., self.cols])
        self.outer.fit(data[..., self.rest])
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        out = np.empty(data.shape, dtype=float)
        out[..., self.cols] = self.inner.transform(data[..., self.cols])
        out[..., self.rest] = self.outer.transform(data[..., self.rest])
        return out

    def inverse_transform(self, scaled: np.ndarray) -> np.ndarray:
        out = np.empty(scaled.shape, dtype=float)
        out[..., self.cols] = self.inner.inverse_transform(scaled[..., self.cols])
        out[..., self.rest] = self.outer.inverse_transform(scaled[..., self.rest])
        return out
