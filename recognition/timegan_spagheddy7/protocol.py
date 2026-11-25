"""Shared information budget for AR and noise-driven TimeGAN comparisons."""

import hashlib
import json

import numpy as np

from dataset import PRICE_COLUMNS, PRICE_DIVISOR
from evaluation import conditioning_history


PRIOR_SEED_OFFSET = 10000


def comparison_protocol(sequence_length=32, max_lags=4, seeds=(42, 43, 44), context_scaler=None, volume_mode="none"):
    protocol = {
        "id": "anchor_only_v1", "sequence_length": sequence_length, "reference_history_length": max_lags,
        "generation_seeds": list(seeds),
        "generator_heldout_inputs": [], "decoder_heldout_inputs": ["20 prices at the window's anchor snapshot"],
        "ar_initialisation": "Uniform bootstrap of training-window histories; last p rows used by VAR(p).",
        "timegan_initialisation": "Generate feature windows from noise; do not encode validation histories or futures.",
        "prior_seed_offset": PRIOR_SEED_OFFSET, "pairing": "Generated sample i is paired with chronological real window i.",
        "window_policy": "Non-overlapping sequence_length-update windows with at least reference_history_length past within-split features.",
        "selection": "Lowest seed-mean invalid rate, then mean KL, ACF error, p and ridge; SSIM is reported, not a tie-breaker.",
        "prohibited": ["held-out feature histories as generator inputs", "held-out volumes as initial state",
                       "best-of-many matching to real futures", "repairing or dropping invalid generated books"],
        "scope": "Anchored feature generation, not forecasting the particular observed future or generating starting prices.",
    }
    if context_scaler is not None:
        protocol.update(id="anchor_conditioned_v1", context_scaler=context_scaler,
            generator_heldout_inputs=["19 standardised log relative price gaps from the anchor snapshot"],
            timegan_initialisation="Noise plus static anchor price-gap context; no held-out feature history or volumes.",
            ar_initialisation="Uniform training-history bootstrap plus the identical static context in a VARX(2).",
            scope="Conditional anchored generation; includes a same-architecture zero-context ablation.")
    if volume_mode != "none":
        if volume_mode not in ("matched", "zero") or context_scaler is None or context_scaler.get("representation") != "log_relative_price_gaps_log1p_volumes_v1":
            raise ValueError("Starting-volume conditioning requires the 39-dimensional training scaler.")
        protocol.update(id="anchor_price_volume_conditioned_v1",
            generator_heldout_inputs=["19 standardised log relative price gaps from the anchor snapshot",
                                     "20 standardised log1p volumes from that snapshot in the matched-volume arm only"],
            timegan_initialisation="Noise plus matched static anchor price context; compare matched anchor volumes against zeroed volume context at identical capacity.",
            ar_initialisation="No AR result is included under this expanded information budget.",
            prohibited=["held-out feature histories as generator inputs", "held-out future snapshots or volumes",
                        "fitting context statistics on validation or test snapshots",
                        "best-of-many matching to real futures", "repairing or dropping invalid generated books"],
            scope="Conditional anchored generation with a 39-dimensional context; the matched/zero volume pair shares this protocol and must not be pooled with historical price-only groups.")
    elif context_scaler is not None and context_scaler.get("representation") == "log_relative_price_gaps_log1p_volumes_v1":
        raise ValueError("The starting-volume scaler requires an explicit matched or zero volume mode.")
    return protocol


def anchor_gap_features(anchors):
    """Scale-invariant anchor shape: spread, nine ask gaps and nine bid gaps."""
    prices = np.asarray(anchors, dtype=np.float64)
    if prices.ndim != 2 or prices.shape[1] != 20 or not len(prices) or not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("Expected positive finite (windows, 20) anchor prices.")
    asks, bids = prices[:, ::2], prices[:, 1::2]
    gaps = np.concatenate((np.log(asks[:, :1] / bids[:, :1]), np.log(asks[:, 1:] / asks[:, :-1]), np.log(bids[:, :-1] / bids[:, 1:])), axis=1)
    if np.any(gaps <= 0):
        raise ValueError("Anchor prices must have a positive spread and strictly ordered ladders.")
    return np.log(gaps)


def fit_anchor_context(training_anchors, training_volumes=None):
    """Fit only on anchor snapshots of complete training windows.

    Optional volumes are the 20 raw share counts at those same starting rows,
    in ask/bid order by level. The caller must supply the training split only.
    Omitting volumes preserves the historical price-only scaler exactly.
    """
    values = anchor_gap_features(training_anchors)
    scale = values.std(axis=0)
    fitted = dict(representation="log_relative_price_gaps_v1", dimensions=19, training_windows=len(values),
                  mean=values.mean(axis=0).tolist(), scale=np.where(scale > 1e-12, scale, 1.).tolist())
    if training_volumes is not None:
        volumes = _anchor_volume_features(training_volumes, len(values))
        volume_scale = volumes.std(axis=0)
        fitted.update(representation="log_relative_price_gaps_log1p_volumes_v1", dimensions=39,
                      price_dimensions=19, volume_dimensions=20,
                      volume_order="ask_1,bid_1,...,ask_10,bid_10",
                      mean=fitted["mean"] + volumes.mean(axis=0).tolist(),
                      scale=fitted["scale"] + np.where(volume_scale > 1e-12, volume_scale, 1.).tolist())
    return fitted


def _anchor_volume_features(volumes, count):
    """Validate raw starting share counts before the fixed log1p transform."""
    values = np.asarray(volumes, dtype=np.float64)
    if values.shape != (count, 20) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Expected nonnegative finite (windows, 20) anchor volumes.")
    return np.log1p(values)


def fit_ladder_gap_scale(training):
    """Training-window log-gap change ranges, with 10% headroom and a 0.1 floor."""
    if training.raw_offset != 0:
        raise ValueError("Ladder statistics must be fitted on the training split.")
    prices = training.book[:, PRICE_COLUMNS] / PRICE_DIVISOR
    gaps = anchor_gap_features(prices)
    future = training.starts[:, None] + np.arange(1, training.sequence_length + 1)
    changes = gaps[future] - gaps[training.starts, None]
    return np.maximum(1.1 * np.abs(changes).max(axis=(0, 1)), .1)


def anchor_context(anchors, scaler, mode="matched", volumes=None, volume_mode="none"):
    """Use fixed training statistics; never clip future contexts or inspect futures."""
    if volume_mode != "none":
        if volume_mode not in ("matched", "zero") or mode != "matched" or scaler is None or scaler.get("representation") != "log_relative_price_gaps_log1p_volumes_v1":
            raise ValueError("Volume controls require matched price context and the 39-dimensional scaler.")
        mean, scale = np.asarray(scaler.get("mean"), dtype=np.float64), np.asarray(scaler.get("scale"), dtype=np.float64)
        if scaler.get("dimensions") != 39 or mean.shape != (39,) or scale.shape != (39,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Invalid starting-volume context scaler.")
        prices = (anchor_gap_features(anchors) - mean[:19]) / scale[:19]
        # The zero-volume arm need not read any held-out volume values.
        volume_values = np.zeros((len(prices), 20)) if volume_mode == "zero" else (_anchor_volume_features(volumes, len(prices)) - mean[19:]) / scale[19:]
        values = np.concatenate((prices, volume_values), axis=1)
        if not np.isfinite(values).all() or np.any(np.abs(values) > np.finfo(np.float32).max):
            raise ValueError("Invalid starting-volume context.")
        return values.astype(np.float32)
    if volumes is not None:
        raise ValueError("Supply an explicit matched or zero volume mode for anchor volumes.")
    if scaler is not None and scaler.get("representation") == "log_relative_price_gaps_log1p_volumes_v1":
        raise ValueError("The starting-volume scaler requires an explicit matched or zero volume mode.")
    if mode == "none":
        return None
    if mode not in ("matched", "zero") or scaler is None or scaler.get("representation") != "log_relative_price_gaps_v1":
        raise ValueError("Unknown anchor-conditioning configuration.")
    values = (anchor_gap_features(anchors) - np.asarray(scaler["mean"])) / np.asarray(scaler["scale"])
    if values.shape[1] != 19 or not np.isfinite(values).all():
        raise ValueError("Invalid anchor context.")
    return np.zeros_like(values, dtype=np.float32) if mode == "zero" else values.astype(np.float32)


def conditional_training_pairs(training, context_scaler, lags=2, max_lags=4):
    """Teacher-forced training pairs with the starting context fixed for 32 updates."""
    if training.raw_offset != 0 or not 1 <= lags <= max_lags:
        raise ValueError("Use the training split and lags within the shared history budget.")
    starts = training.starts[training.starts >= max_lags]
    if not len(starts):
        raise ValueError("No complete training windows with enough history.")
    targets = (starts[:, None] + np.arange(training.sequence_length)).ravel()
    previous = training.features[targets[:, None] - np.arange(1, lags + 1)].reshape(len(targets), -1)
    contexts = anchor_context(training.book[starts][:, PRICE_COLUMNS] / PRICE_DIVISOR, context_scaler)
    return previous, training.features[targets], np.repeat(contexts, training.sequence_length, axis=0)


def sample_training_prior(training, count, max_lags, seed):
    """No validation argument: held-out history cannot enter the AR initial state.

    All AR orders share the same sampled max_lags-row histories for a given
    seed; smaller orders take a suffix. Context and innovation seeds differ.
    """
    if training.raw_offset != 0:
        raise ValueError("The AR prior must come from the training split starting at raw row zero.")
    if count < 1 or max_lags < 1 or seed < 0 or seed >= 2**63 - PRIOR_SEED_OFFSET:
        raise ValueError("Invalid prior count, history length or seed.")
    pool = training.starts[training.starts >= max_lags]
    if len(pool) == 0:
        raise ValueError("No complete training windows have enough prior history.")
    rng = np.random.default_rng(seed + PRIOR_SEED_OFFSET)
    starts = pool[rng.integers(0, len(pool), size=count)]
    return conditioning_history(training, starts, max_lags), starts + training.raw_offset


def comparison_signature(protocol, metric_spec, raw_anchor_indices, input_files, scaler):
    """Record what must match in the later TimeGAN evaluation, apart from model code."""
    contract = {"protocol": protocol, "metrics": metric_spec, "anchors": list(map(int, raw_anchor_indices)),
                "input_files": input_files, "scaler": scaler}
    return hashlib.sha256(json.dumps(contract, sort_keys=True, allow_nan=False).encode()).hexdigest()
