"""COMP3710 Project 7: reproducible TimeGAN failure-case autopsy.

Produces five selected LOBSTER Level-10 sequence comparisons (24 steps x 20
volume channels), with price-spread plots and a machine-readable audit.
The real and generated sequences are *unpaired*: comparisons are illustrative,
not reconstruction or timestamp-aligned forecast errors.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def statistics(x):
    ask = x[..., 0::4]
    bid = x[..., 2::4]
    sizes = x[..., 1::2]
    spread = ask[..., 0] - bid[..., 0]
    invalid_ask = np.any(np.diff(ask, axis=-1) < -1e-8, axis=-1)
    invalid_bid = np.any(np.diff(bid, axis=-1) > 1e-8, axis=-1)
    return dict(
        spread=spread,
        spread_mean=spread.mean(axis=1),
        spread_std=spread.std(axis=1),
        invalid_ask_count=invalid_ask.sum(axis=1),
        invalid_bid_count=invalid_bid.sum(axis=1),
        depth_std=sizes.std(axis=1).mean(axis=1),
        depth_mean=sizes.mean(axis=(1, 2)),
    )


def choose_cases(real, fake):
    """Deterministic rules; no manual cherry-picking or paired-error claim."""
    r, f = statistics(real), statistics(fake)
    n = len(fake)
    if n < 5:
        raise ValueError('Need at least five generated sequences')
    available = set(range(n))

    def pick(score, allowed=None):
        candidates = available if allowed is None else available.intersection(allowed)
        if not candidates:
            candidates = available
        idx = max(sorted(candidates), key=lambda j: (float(score[j]), -j))
        available.remove(idx)
        return int(idx)

    cases = []
    candidates = np.where(f['invalid_ask_count'] > 0)[0].tolist()
    i = pick(f['invalid_ask_count'], candidates)
    cases.append(('ask_ladder_violation', i,
                  'Maximum number of time steps with inverted ask ladder'))

    i = pick(f['spread_mean'] - float(np.median(r['spread_mean'])))
    cases.append(('wide_spread', i,
                  'Largest generated mean spread among remaining sequences'))

    i = pick(-f['depth_std'])
    cases.append(('flat_depth', i,
                  'Smallest generated temporal depth variability'))

    i = pick(-f['spread_std'])
    cases.append(('flat_spread', i,
                  'Smallest generated spread variability'))

    # Pick a sample near the median generated depth to show typical collapse,
    # not an outlier chosen for a large unpaired discrepancy.
    median_depth = float(np.median(f['depth_mean']))
    i = pick(-np.abs(f['depth_mean'] - median_depth))
    cases.append(('typical_generated_depth', i,
                  'Generated sample closest to median generated mean depth'))
    return cases, r, f


def plot_case(index, label, reason, real, fake, r, f, out, vmin, vmax):
    """Plot time x 20 depth heatmaps and independent spread trajectories."""
    rd = real[index, :, 1::2].T
    fd = fake[index, :, 1::2].T
    rs = r['spread'][index]
    fs = f['spread'][index]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8),
                             gridspec_kw={'height_ratios': [2, 1]})
    for ax, data, name in [(axes[0, 0], rd, 'Real test sample'),
                           (axes[0, 1], fd, 'Synthetic sample')]:
        im = ax.imshow(data, origin='lower', aspect='auto', interpolation='nearest',
                       cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set(title=name + f' (sequence index {index})',
               xlabel='Time step', ylabel='Depth channel (20 volumes)')
    fig.colorbar(im, ax=list(axes[0, :]), shrink=0.75,
                 label='Decoded order volume (shared scale)')
    axes[1, 0].plot(rs, label='Real')
    axes[1, 1].plot(fs, label='Generated', color='tab:orange')
    for ax in axes[1]:
        ax.set(xlabel='Time step', ylabel='Best ask - best bid (USD)')
        ax.grid(alpha=0.25)
    fig.suptitle(f'{label}: {reason}\n'
                 'Real and generated are unpaired; same index does NOT imply time alignment',
                 fontsize=11)
    fig.subplots_adjust(top=0.83, bottom=0.10, left=0.07, right=0.92,
                        wspace=0.18, hspace=0.35)
    path = out / f'{label}.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def run(real, fake, output_dir):
    real = np.asarray(real, dtype=np.float64)
    fake = np.asarray(fake, dtype=np.float64)
    if real.shape != fake.shape or real.ndim != 3 or real.shape[2] != 40:
        raise ValueError(f'Expected matching (N, T, 40), got {real.shape} and {fake.shape}')
    if not (np.isfinite(real).all() and np.isfinite(fake).all()):
        raise ValueError('Non-finite values in input data')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases, r, f = choose_cases(real, fake)
    # Use one COMMON display scale for all figures; 99th percentile avoids outlier saturation.
    volume_max = max(float(np.percentile(real[..., 1::2], 99)),
                     float(np.percentile(fake[..., 1::2], 99)), 1.0)
    records = []
    for label, i, reason in cases:
        path = plot_case(i, label, reason, real, fake, r, f, out, 0, volume_max)
        entry = {
            'case': label, 'index': i, 'selection_rule': reason,
            'figure': path.name,
            'real_spread_mean': float(r['spread_mean'][i]),
            'generated_spread_mean': float(f['spread_mean'][i]),
            'real_spread_std': float(r['spread_std'][i]),
            'generated_spread_std': float(f['spread_std'][i]),
            'generated_ask_violation_steps': int(f['invalid_ask_count'][i]),
            'generated_bid_violation_steps': int(f['invalid_bid_count'][i]),
            'real_depth_temporal_std': float(r['depth_std'][i]),
            'generated_depth_temporal_std': float(f['depth_std'][i]),
        }
        records.append(entry)
        print(f'{label}: index={i} ask_violations={entry["generated_ask_violation_steps"]} '
              f'real_spread={entry["real_spread_mean"]:.5f} '
              f'generated_spread={entry["generated_spread_mean"]:.5f}')
    with (out / 'failure_cases.json').open('w', encoding='utf-8') as fp:
        json.dump({'pairing_note': 'UNPAIRED independent generations; comparisons are illustrative.',
                   'selection': 'Deterministic extrema and median-depth sample; same index for display only.',
                   'shared_volume_vmax': volume_max, 'cases': records}, fp, indent=2)
    with (out / 'failure_cases.csv').open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f'Figures and tables saved to: {out}')
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--generated', required=True, help='V2 normalized synthetic sequences (.npy)')
    p.add_argument('--output-dir', default='outputs_failure_cases_v3_3')
    args = p.parse_args()
    from dataset_v2 import prepare_dataset_v2
    data = prepare_dataset_v2(args.data_dir)
    transform = data['transform']
    real = transform.inverse_transform(data['test'])
    fake = transform.inverse_transform(np.load(args.generated))
    print('Real / generated:', real.shape, fake.shape)
    run(real, fake, args.output_dir)


if __name__ == '__main__':
    main()
