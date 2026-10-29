"""COMP3710 TimeGAN V4.5: training-only, first-order Markov tick baseline.

This is a statistical baseline, NOT a trained TimeGAN generator or a
LOB-consistent simulator. Transition probabilities and nonzero tick sizes
are estimated from the chronological training partition only.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from dataset import load_lobster, chronological_split
from dataset_v4 import prepare_dataset_v4


def histogram_kl_common(real, generated, bins=50, eps=1e-8):
    """Shared fixed bin edges; epsilon-smoothed forward KL."""
    real = np.asarray(real, dtype=np.float64).ravel()
    generated = np.asarray(generated, dtype=np.float64).ravel()
    lo = min(float(real.min()), float(generated.min()))
    hi = max(float(real.max()), float(generated.max()))
    if hi <= lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    p = np.histogram(real, bins=edges)[0].astype(np.float64) + eps
    q = np.histogram(generated, bins=edges)[0].astype(np.float64) + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def per_sequence_lag1_abs_corr(x):
    """Pool within-window adjacent pairs; no artificial sequence-boundary pairs."""
    a = np.abs(np.asarray(x, dtype=np.float64))
    left, right = a[:, :-1].ravel(), a[:, 1:].ravel()
    if left.std() == 0 or right.std() == 0:
        return float('nan')
    return float(np.corrcoef(left, right)[0, 1])


def metrics(x, eps=1e-10):
    a = np.asarray(x, dtype=np.float64)
    moves = np.abs(a) > eps
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "near_zero_fraction": float((~moves).mean()),
        "positive_fraction": float((a > eps).mean()),
        "negative_fraction": float((a < -eps).mean()),
        "mean_absolute_nonzero": float(np.abs(a[moves]).mean()) if moves.any() else 0.0,
        "cross_sample_std": float(a.std(axis=0).mean()),
        "lag1_absolute_return_correlation": per_sequence_lag1_abs_corr(a),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--output-dir', default='outputs_v4_5_markov')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bins', type=int, default=50)
    parser.add_argument('--v43-file', default='outputs_v4_3_generated_norm.npy')
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    book, _ = load_lobster(args.data_dir)
    raw = book.to_numpy()
    mid2 = raw[:, 0].astype(np.int64) + raw[:, 2].astype(np.int64)
    train_ids, _, _ = chronological_split(np.arange(len(mid2)))
    mid_train = mid2[train_ids]
    changes = np.diff(mid_train)
    if not np.all(changes % 100 == 0):
        raise ValueError('Not every train mid2 price change is a multiple of 100')
    tick_changes = changes // 100
    state = (tick_changes != 0).astype(np.int64)
    previous, current = state[:-1], state[1:]
    trans_counts = np.zeros((2, 2), dtype=np.int64)
    for old in (0, 1):
        for new in (0, 1):
            trans_counts[old, new] = int(np.sum((previous == old) & (current == new)))
    probabilities = trans_counts / trans_counts.sum(axis=1, keepdims=True)
    tick_values, tick_counts = np.unique(tick_changes[state == 1], return_counts=True)
    tick_probs = tick_counts / tick_counts.sum()

    dataset = prepare_dataset_v4(args.data_dir)
    real_norm = dataset['test']
    real = dataset['normalizer'].inverse_transform(real_norm)[..., -1]
    n, t = real.shape
    sampled_ticks = np.zeros((n, t), dtype=np.int64)
    sampled_mid2 = np.zeros((n, t), dtype=np.float64)
    sampled_returns = np.zeros((n, t), dtype=np.float64)

    # Every synthetic window begins at a real TRAIN mid-price only.
    starting_mid2 = rng.choice(mid_train, size=n).astype(np.float64)
    last_mid2 = starting_mid2.copy()
    last_move_state = rng.choice([0, 1], size=n, p=[1-state.mean(), state.mean()]).astype(np.int64)
    for j in range(t):
        p_move = probabilities[last_move_state, 1]
        next_state = (rng.random(n) < p_move).astype(np.int64)
        increments = np.zeros(n, dtype=np.int64)
        is_move = next_state == 1
        if is_move.any():
            increments[is_move] = rng.choice(tick_values, size=is_move.sum(), p=tick_probs)
        next_mid2 = last_mid2 + increments * 100.0
        if np.any(next_mid2 <= 0):
            raise ValueError('Simulated non-positive midpoint; need a boundary rule')
        sampled_returns[:, j] = np.log(next_mid2 / last_mid2)
        sampled_mid2[:, j] = next_mid2
        sampled_ticks[:, j] = increments
        last_mid2 = next_mid2
        last_move_state = next_state

    reports = {'real_test': metrics(real), 'markov_baseline': metrics(sampled_returns)}
    kl = {'markov_baseline': histogram_kl_common(real, sampled_returns, args.bins)}
    path_v43 = Path(args.v43_file)
    if path_v43.is_file():
        g = np.load(path_v43)
        if g.shape != real_norm.shape:
            raise ValueError(f'V4.3 shape {g.shape} does not match test shape {real_norm.shape}')
        v43 = dataset['normalizer'].inverse_transform(g)[..., -1]
        reports['timegan_v4_3'] = metrics(v43)
        kl['timegan_v4_3'] = histogram_kl_common(real, v43, args.bins)

    print('=== V4.5 Training-Only Markov Tick Baseline ===')
    print('Transition counts (previous 0/1 -> next 0/1):')
    print(trans_counts)
    print('P(next move | previous no move):', probabilities[0, 1])
    print('P(next move | previous move):', probabilities[1, 1])
    print('Training nonzero tick categories:', len(tick_values))
    print('Windows / length:', n, t)
    for name, report in reports.items():
        print('\n' + name + ':')
        for k, v in report.items():
            print(' ', k + ':', v)
    print('\nComparable histogram KL (common bin edges per comparison):')
    for k, v in kl.items():
        print('  KL(real || ' + k + '):', v)
    print('NOTE: This is a statistical tick baseline, NOT a TimeGAN or complete LOB generator.')
    print('NOTE: Independent tick magnitudes given movement may miss volatility clustering.')

    np.save(outdir / 'markov_ticks.npy', sampled_ticks)
    np.save(outdir / 'markov_returns.npy', sampled_returns)
    np.save(outdir / 'markov_mid2.npy', sampled_mid2)
    result = {
        'seed': args.seed,
        'transition_counts': trans_counts.tolist(),
        'transition_probabilities': probabilities.tolist(),
        'reports': reports,
        'histogram_kl': kl,
        'histogram_bins': args.bins,
    }
    with (outdir / 'results.json').open('w', encoding='utf8') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print('\nSaved outputs to', outdir)


if __name__ == '__main__':
    main()
