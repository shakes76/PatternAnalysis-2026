from __future__ import annotations

import numpy as np


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
