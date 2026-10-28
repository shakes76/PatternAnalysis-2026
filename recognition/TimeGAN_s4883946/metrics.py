"""Evaluation of generated limit order book windows.

All functions take *raw* (denormalised) feature arrays of shape (N, T, 41), laid
out as described in dataset.py. Metrics:

    microstructure_audit  crossed books, invalid price ladders, negative depth
    kl_divergence         KL(real || synthetic) of spread and mid-return histograms
    abs_return_acf        autocorrelation of absolute returns (volatility clustering)
    depth_heatmaps/ssim   structural similarity of price-level x time depth surfaces

Binning rule for KL (fixed so results are comparable): 50 equal-width bins between
the 0.5th and 99.5th percentile of the *real* data, plus two open-ended outer bins,
with a small constant added to every bin before normalising so empty bins stay finite.
"""
import numpy as np
import torch
from skimage.metrics import structural_similarity

from dataset import LEVELS, TICK, decode_distances

ASK_DIST = slice(1, 1 + LEVELS)
BID_DIST = slice(1 + LEVELS, 1 + 2 * LEVELS)
ASK_VOL = slice(1 + 2 * LEVELS, 1 + 3 * LEVELS)
BID_VOL = slice(1 + 3 * LEVELS, 1 + 4 * LEVELS)


def distances(raw, ladder="levels"):
    """Ask and bid distances from the mid-price in dollars, each (N, T, 10)."""
    return (decode_distances(raw[..., ASK_DIST], ladder) * TICK,
            decode_distances(raw[..., BID_DIST], ladder) * TICK)


def spreads(raw, ladder="levels"):
    """Bid-ask spread in dollars, (N, T). Negative or zero means a crossed book."""
    ask_d, bid_d = distances(raw, ladder)
    return ask_d[..., 0] + bid_d[..., 0]


def snap_to_rules(raw, ladder="levels"):
    """Decode generated windows onto the market rules (an *enforced* variant).

    The model's output is projected so that: the spread is a whole number of ticks and at
    least one tick (best ask and bid are placed symmetrically about the mid-price, as in
    the real data); every gap between successive price levels is a whole number of ticks
    and at least one tick; volumes are whole numbers and at least 1. The return feature
    is untouched. Real windows already satisfy these rules and pass through unchanged.
    Reported separately because it enforces the rules instead of testing what the model
    learned.
    """
    out = np.array(raw, dtype=float, copy=True)
    # (slices are not hashable before Python 3.12, so keep them in a list, not a dict)
    dist = [(sl, decode_distances(raw[..., sl], ladder)) for sl in (ASK_DIST, BID_DIST)]
    spread = np.maximum(np.round(dist[0][1][..., 0] + dist[1][1][..., 0]), 1.0)
    first = (spread / 2)[..., None]
    for sl, d in dist:
        gaps = np.maximum(np.round(np.diff(d, axis=-1)), 1.0)
        if ladder == "increments":
            ticks = np.concatenate([first, gaps], axis=-1)
        else:
            ticks = first + np.concatenate([np.zeros_like(first), np.cumsum(gaps, axis=-1)], -1)
        out[..., sl] = np.log1p(ticks)
    for sl in (ASK_VOL, BID_VOL):
        out[..., sl] = np.log1p(np.maximum(np.round(np.expm1(raw[..., sl])), 1.0))
    return out


def microstructure_audit(raw, ladder="levels"):
    """Fraction of snapshots violating each market microstructure rule.

    crossed:  best ask <= best bid (spread <= 0)
    ladder:   ask prices not increasing, or bid prices not decreasing, with depth level
    neg_depth: any negative volume (log1p volume below 0)
    any:      at least one of the above
    windows_with_crossed: fraction of windows containing at least one crossed snapshot
    """
    ask_d, bid_d = distances(raw, ladder)
    crossed = (ask_d[..., 0] + bid_d[..., 0]) <= 0
    ladder = ((np.diff(ask_d, axis=-1) < 0).any(-1)
              | (np.diff(bid_d, axis=-1) < 0).any(-1))
    neg_depth = (raw[..., ASK_VOL] < 0).any(-1) | (raw[..., BID_VOL] < 0).any(-1)
    return {
        "crossed": float(crossed.mean()),
        "ladder": float(ladder.mean()),
        "neg_depth": float(neg_depth.mean()),
        "any": float((crossed | ladder | neg_depth).mean()),
        "windows_with_crossed": float(crossed.any(-1).mean()),
    }


def kl_divergence(real, synth, bins=50, eps=1e-6):
    """KL(real || synthetic) between 1-D samples, using the fixed binning rule above."""
    real = np.nan_to_num(np.asarray(real).ravel())
    synth = np.nan_to_num(np.asarray(synth).ravel())
    lo, hi = np.percentile(real, [0.5, 99.5])
    if hi <= lo:  # degenerate real distribution
        hi = lo + 1e-12
    edges = np.concatenate([[-np.inf], np.linspace(lo, hi, bins + 1), [np.inf]])
    p = np.histogram(real, edges)[0].astype(float) + eps
    q = np.histogram(synth, edges)[0].astype(float) + eps
    p, q = p / p.sum(), q / q.sum()
    return float((p * np.log(p / q)).sum())


def abs_return_acf(raw, lags=(1, 5, 10)):
    """Autocorrelation of absolute mid-returns inside windows (volatility clustering).

    Pooled over all windows using one global mean/variance. Positive values that
    decay with lag indicate clustering; values near 0 indicate white noise.
    """
    a = np.abs(np.nan_to_num(raw[..., 0]))
    a = a - a.mean()
    var = (a ** 2).mean() + 1e-30
    return {k: float((a[:, :-k] * a[:, k:]).mean() / var) for k in lags}


def depth_heatmaps(raw):
    """Depth surfaces (N, 20, T): bid levels 10..1 then ask levels 1..10, log1p volume."""
    bid = raw[..., BID_VOL][..., ::-1]
    ask = raw[..., ASK_VOL]
    return np.nan_to_num(np.concatenate([bid, ask], axis=-1).transpose(0, 2, 1))


def ssim_to_nearest(candidates, references, n=300, seed=0):
    """Mean SSIM between each candidate heatmap and its nearest reference heatmap.

    Generated windows are not paired with real ones, so each of up to n random
    candidates is compared with the reference closest in L2 distance. The data range
    is set from the references. Running the same function with held-out real windows
    as candidates gives the ceiling a perfect generator could reach.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(candidates), min(n, len(candidates)), replace=False)
    cand = candidates[idx]
    dists = torch.cdist(torch.from_numpy(cand.reshape(len(cand), -1)).float(),
                        torch.from_numpy(references.reshape(len(references), -1)).float())
    nearest = dists.argmin(dim=1).numpy()
    data_range = float(references.max() - references.min())
    scores = [structural_similarity(c, references[j], data_range=data_range, win_size=7)
              for c, j in zip(cand, nearest)]
    return float(np.mean(scores))


def ssim_mean_surface(synth_heatmaps, real_heatmaps):
    """SSIM between the average synthetic and average real depth surface.

    A looser check than ssim_to_nearest: it rewards matching the average depth
    profile across levels and time, but cannot detect mode collapse on its own.
    """
    real_mean, synth_mean = real_heatmaps.mean(0), synth_heatmaps.mean(0)
    data_range = float(real_mean.max() - real_mean.min()) + 1e-12
    return float(structural_similarity(synth_mean, real_mean, data_range=data_range,
                                       win_size=7))


def evaluate(real_raw, synth_raw, reference_raw=None, ladder="levels", enforce_rules=False):
    """Compare synthetic windows with real windows; returns a flat dict of metrics.

    reference_raw is the pool of real windows used for the SSIM nearest-neighbour
    search (defaults to real_raw).
    """
    synth_raw = np.nan_to_num(synth_raw)
    if enforce_rules:
        synth_raw = snap_to_rules(synth_raw, ladder)
    ref = real_raw if reference_raw is None else reference_raw
    audit = microstructure_audit(synth_raw, ladder)
    out = {f"audit_{k}": v for k, v in audit.items()}
    out["kl_spread"] = kl_divergence(spreads(real_raw, ladder), spreads(synth_raw, ladder))
    out["kl_return"] = kl_divergence(real_raw[..., 0], synth_raw[..., 0])
    for lag, v in abs_return_acf(synth_raw).items():
        out[f"acf_lag{lag}"] = v
    out["ssim"] = ssim_to_nearest(depth_heatmaps(synth_raw), depth_heatmaps(ref))
    out["ssim_mean_surface"] = ssim_mean_surface(depth_heatmaps(synth_raw),
                                                 depth_heatmaps(real_raw))
    return out
