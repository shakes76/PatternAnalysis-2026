import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.stats import entropy
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt


# Array Helpers
def tick_bins(*arrays: np.ndarray) -> np.ndarray:
    """Shared histogram edges with one bin per integer, for integer data like spreads."""
    values = np.concatenate([a.ravel() for a in arrays])
    return np.arange(values.min() - 0.5, values.max() + 1.5)

def central_bins(*arrays: np.ndarray, bins: int = 50) -> np.ndarray:
    """Shared histogram edges over the central 99% of the pooled values."""
    lo, hi = np.percentile(np.concatenate([a.ravel() for a in arrays]), [0.5, 99.5])
    return np.linspace(lo, hi, bins + 1)

# Metrics Helpers
def max_log_size(*books: dict) -> float:
    """Largest log(1 + size) across books: a shared colour scale for heatmaps."""
    return max(np.log1p(np.clip(b[k], 0, None)).max()
               for b in books for k in ("ask_sz", "bid_sz"))

def spread(book: dict) -> np.ndarray:
    """Best ask minus best bid, shape (window, step)."""
    return book["ask_px"][..., 0] - book["bid_px"][..., 0]

def mid_returns(book: dict) -> np.ndarray:
    """Log-returns of the mid-price within each window, shape (window, step - 1)."""
    mid = (book["ask_px"][..., 0] + book["bid_px"][..., 0]) / 2
    return np.diff(np.log(np.clip(mid, 1, None)), axis=1)

def vwap_returns(book: dict) -> np.ndarray:
    """Log-returns of the book's volume-weighted average price within each window.

    The VWAP here is the size-weighted average price of all resting liquidity
    across both sides and all levels at each step (the book holds no trades).
    Unlike the mid, it moves when deeper prices or sizes change, so it also
    tests whether the depth evolves realistically.
    """
    px = np.concatenate([book["ask_px"], book["bid_px"]], axis=-1)
    sz = np.clip(np.concatenate([book["ask_sz"], book["bid_sz"]], axis=-1), 0, None)
    vwap = (px * sz).sum(-1) / np.clip(sz.sum(-1), 1, None)
    return np.diff(np.log(np.clip(vwap, 1, None)), axis=1)

def depth_heatmap(book: dict, i: int) -> np.ndarray:
    """Window i as an image for SSIM and plotting, shape (2 x DEPTH, step).

    Rows run up the price ladder (bid levels 10..1, then ask 1..10); each pixel
    is log(1 + size), so a few huge levels don't wash out the rest.
    """
    sizes = np.concatenate([book["bid_sz"][i, :, ::-1], book["ask_sz"][i]], axis=-1)
    return np.log1p(np.clip(sizes, 0, None)).T

def violation_rates(book: dict) -> dict[str, float]:
    """Fraction of snapshots breaking each order book invariant."""
    crossed = spread(book) <= 0
    non_mono = ((np.diff(book["ask_px"], axis=-1) <= 0).any(-1)
                | (np.diff(book["bid_px"], axis=-1) >= 0).any(-1))
    bad_size = (book["ask_sz"] <= 0).any(-1) | (book["bid_sz"] <= 0).any(-1)
    return {
        "crossed_book": float(crossed.mean()),
        "non_monotonic": float(non_mono.mean()),
        "non_positive_size": float(bad_size.mean()),
    }

def unchanged_rates(book: dict) -> dict[str, float]:
    """Fraction of step-to-step transitions where a value did not change.

    Real 1-second books are mostly frozen, so these are high for real data;
    a generator that moves every step scores near 0.
    """
    best_px = np.stack([book["ask_px"][..., 0], book["bid_px"][..., 0]], -1)
    all_px = np.concatenate([book["ask_px"], book["bid_px"]], -1)
    all_sz = np.concatenate([book["ask_sz"], book["bid_sz"]], -1)
    return {
        "best_prices": float(np.mean(np.diff(best_px, axis=1) == 0)),
        "all_prices": float(np.mean(np.diff(all_px, axis=1) == 0)),
        "all_sizes": float(np.mean(np.diff(all_sz, axis=1) == 0)),
    }

def kl_divergence(real: np.ndarray, synth: np.ndarray, bins: int = 50) -> float:
    """KL(real || synthetic) between histograms on shared bins.

    `bins` equal bins span the central 99% of the pooled values, so a few
    extreme values cannot stretch the histogram; values outside are clipped
    into the end bins. A small constant avoids log(0).
    """
    real, synth = real.ravel(), synth.ravel()
    pooled = np.concatenate([real, synth])
    edges = np.histogram_bin_edges(pooled, bins, range=tuple(np.percentile(pooled, [0.5, 99.5])))
    p, _ = np.histogram(np.clip(real, edges[0], edges[-1]), edges)
    q, _ = np.histogram(np.clip(synth, edges[0], edges[-1]), edges)
    return float(entropy(p + 1e-6, q + 1e-6))

def mean_ssim(real: dict, synth: dict) -> float:
    """Average SSIM between real and synthetic depth heatmaps, paired by index."""
    real_maps = [depth_heatmap(real, i) for i in range(len(real["ask_sz"]))]
    synth_maps = [depth_heatmap(synth, i) for i in range(len(synth["ask_sz"]))]
    data_range = max(np.max(real_maps), np.max(synth_maps))
    return float(np.mean([
        structural_similarity(r, s, data_range=data_range)
        for r, s in zip(real_maps, synth_maps)
    ]))

def abs_return_autocorr(returns: np.ndarray, max_lag: int = 10) -> np.ndarray:
    """Autocorrelation of |returns| at lags 1..max_lag (volatility clustering)."""
    a = np.abs(returns)
    return np.array([
        np.corrcoef(a[:, :-k].ravel(), a[:, k:].ravel())[0, 1]
        for k in range(1, max_lag + 1)
    ])

def garch_fit(returns: np.ndarray, scale: float = 1e4) -> dict[str, float]:
    """Gaussian GARCH(1,1) fitted jointly to many short return series.

    `returns` has shape (window, step). All windows share (omega, alpha, beta)
    and their log-likelihoods are summed, so short windows can pin down the
    parameters without being stitched into one series with fake jumps. Each
    window's variance starts at the pooled sample variance. Returns are
    multiplied by `scale` (1e4: basis points) so the optimiser sees O(1)
    values; omega and long_run_var are reported in those units.
    """
    eps2 = (returns * scale - (returns * scale).mean()) ** 2
    var0 = eps2.mean()
    nan = float("nan")
    if var0 == 0:   # no movement at all, e.g. a frozen generator
        return dict(omega=nan, alpha=nan, beta=nan, persistence=nan, half_life=nan,
                    long_run_var=nan, converged=False)

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def logit(p):
        return np.log(p / (1 - p))

    def unpack(theta):
        """Unconstrained theta -> omega > 0, alpha, beta >= 0, alpha + beta < 1."""
        persistence = 0.9999 * sigmoid(theta[1])
        share = sigmoid(theta[2])                     # alpha's share of the persistence
        return np.exp(theta[0]), share * persistence, (1 - share) * persistence

    steps = np.arange(eps2.shape[1])

    def nll(theta):
        omega, alpha, beta = unpack(theta)
        # sigma2_t = omega + alpha * eps2_{t-1} + beta * sigma2_{t-1}, unrolled:
        # the eps2 part is a linear filter, the rest is closed form in t
        shocks = lfilter([0, alpha], [1, -beta], eps2, axis=1)
        sigma2 = shocks + omega * (1 - beta ** steps) / (1 - beta) + beta ** steps * var0
        return 0.5 * np.sum(np.log(sigma2) + eps2 / sigma2)

    # Start at alpha 0.1, beta 0.8, with omega matching the sample variance
    theta0 = np.array([np.log(0.1 * var0), logit(0.9 / 0.9999), logit(0.1 / 0.9)])
    res = minimize(nll, theta0, method="Nelder-Mead",
                   options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6})
    omega, alpha, beta = unpack(res.x)
    persistence = alpha + beta
    return dict(
        omega=float(omega),
        alpha=float(alpha),
        beta=float(beta),
        persistence=float(persistence),
        half_life=float(np.log(0.5) / np.log(persistence)) if persistence > 0 else nan,
        long_run_var=float(omega / (1 - persistence)),
        converged=bool(res.success),
    )
