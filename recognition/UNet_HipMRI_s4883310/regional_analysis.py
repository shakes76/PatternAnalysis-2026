"""Evaluate prostate segmentation across approximate apex, mid-gland and base regions."""

import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import label as connected_components

from dataset import test_dataset
from evaluate import PROSTATE_CLASS, device, load_model


RESULTS_DIR = Path('results')

CHECKPOINTS = {
    'baseline': 'checkpoints/unet_baseline_best.pth',
    'improved': 'checkpoints/unet_improved_best.pth',
}

REGIONS = ('Apex', 'Mid-gland', 'Base')
SI_AXIS = 1  # Increasing columns move superiorly, toward the base.
FILENAME_RE = re.compile(r'case_(\d+)_week_(\d+)_slice_(\d+)\.nii\.gz$')


def save_csv(path, rows, fields):
    """Save result rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {path}')


def get_groups():
    """Group test slices by patient/week and sort by slice number."""
    groups = defaultdict(list)

    for index, path in enumerate(test_dataset.image_paths):
        match = FILENAME_RE.fullmatch(Path(path).name)

        if match is None:
            raise ValueError(f'Unexpected filename: {path}')

        patient, week, number = map(int, match.groups())
        groups[(patient, week)].append((number, index))

    for entries in groups.values():
        entries.sort()

    return groups


def load_truth(index):
    """Return a ground-truth mask as a 2D NumPy array."""
    _, mask = test_dataset[index]

    truth = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else np.asarray(mask)

    if truth.ndim != 2:
        raise ValueError(f'Expected 2D class-ID mask, got {truth.shape}')

    return truth


def get_region_bounds(groups):
    """Divide each scan's ground-truth prostate SI extent into three approximate regions."""
    bounds = {}

    for scan, entries in groups.items():
        occupied = []

        # Find the prostate's SI extent across all slices in the scan.
        for _, index in entries:
            coordinates = np.where(load_truth(index) == PROSTATE_CLASS)[SI_AXIS]

            if coordinates.size:
                occupied.extend((int(coordinates.min()), int(coordinates.max())))

        if not occupied:
            print(f'Skipping {scan}: no ground-truth prostate')
            continue

        positions = np.arange(min(occupied), max(occupied) + 1)

        if positions.size < 3:
            print(f'Skipping {scan}: SI prostate extent is too short')
            continue

        # Equal thirds approximate apex, mid-gland and base.
        thirds = np.array_split(positions, 3)

        bounds[scan] = {
            name: (int(part[0]), int(part[-1]))
            for name, part in zip(REGIONS, thirds)
        }

    return bounds


def counts(pred, truth, lower, upper):
    """Calculate prostate TP, FP and FN within one SI region."""
    band = np.zeros(truth.shape, dtype=bool)
    band[:, lower:upper + 1] = True

    actual = (truth == PROSTATE_CLASS) & band
    predicted = (pred == PROSTATE_CLASS) & band

    tp = int(np.count_nonzero(actual & predicted))
    fp = int(np.count_nonzero(predicted & ~actual))
    fn = int(np.count_nonzero(actual & ~predicted))

    return tp, fp, fn


def scores(tp, fp, fn):
    """Calculate Dice and IoU, returning NaN for an empty comparison."""
    denominator = 2 * tp + fp + fn

    if denominator == 0:
        return float('nan'), float('nan')

    return 2 * tp / denominator, tp / (tp + fp + fn)


def evaluate_regions(model_name, model, groups, bounds):
    """Evaluate regional accuracy, inter-slice consistency and disconnected components."""
    pooled = defaultdict(lambda: np.zeros(3, dtype=np.int64))
    per_scan = defaultdict(lambda: np.zeros(3, dtype=np.int64))

    slice_rows = []
    consistency_rows = []
    component_rows = []

    model.eval()

    with torch.inference_mode():
        for (patient, week), entries in groups.items():
            scan = (patient, week)

            if scan not in bounds:
                continue

            previous = None

            for number, index in entries:
                image, mask = test_dataset[index]
                truth = mask.cpu().numpy()

                if truth.ndim != 2:
                    raise ValueError('Test masks must contain 2D class IDs')

                # Generate a predicted class-ID mask.
                logits = model(image.unsqueeze(0).to(device))
                pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

                if pred.shape != truth.shape:
                    raise ValueError(
                        f'Prediction/GT mismatch: {pred.shape} vs {truth.shape}'
                    )

                filename = Path(test_dataset.image_paths[index]).name

                # Measure prostate area changes between consecutive coronal slices.
                gt_area = int(np.count_nonzero(truth == PROSTATE_CLASS))
                pred_area = int(np.count_nonzero(pred == PROSTATE_CLASS))

                if previous is not None and number == previous[0] + 1:
                    pred_delta = pred_area - previous[1]
                    gt_delta = gt_area - previous[2]

                    consistency_rows.append({
                        'model': model_name,
                        'patient': patient,
                        'week': week,
                        'slice': number,
                        'predicted_area_change': abs(pred_delta),
                        'ground_truth_area_change': abs(gt_delta),
                        'signed_change_discrepancy': abs(pred_delta - gt_delta),
                    })

                previous = (number, pred_area, gt_area)

                # Count separate prostate components in each 2D mask.
                _, n_pred = connected_components(pred == PROSTATE_CLASS)
                _, n_gt = connected_components(truth == PROSTATE_CLASS)

                component_rows.append({
                    'model': model_name,
                    'patient': patient,
                    'week': week,
                    'slice': number,
                    'filename': filename,
                    'predicted_components': n_pred,
                    'ground_truth_components': n_gt,
                    'extra_components': max(0, n_pred - n_gt),
                })

                # Calculate segmentation accuracy within each anatomical region.
                for region, (low, high) in bounds[scan].items():
                    tp, fp, fn = counts(pred, truth, low, high)

                    pooled[region] += (tp, fp, fn)
                    per_scan[(scan, region)] += (tp, fp, fn)

                    dice, iou = scores(tp, fp, fn)

                    if tp + fp + fn > 0:
                        slice_rows.append({
                            'model': model_name,
                            'patient': patient,
                            'week': week,
                            'slice': number,
                            'index': index,
                            'filename': filename,
                            'region': region,
                            'si_lower': low,
                            'si_upper': high,
                            'dice': dice,
                            'iou': iou,
                            'tp': tp,
                            'fp': fp,
                            'fn': fn,
                            'total_errors': fp + fn,
                        })

    regional_rows = []
    scan_rows = []

    # Calculate Dice and IoU separately for each patient/week scan.
    for (scan, region), (tp, fp, fn) in per_scan.items():
        dice, iou = scores(int(tp), int(fp), int(fn))

        scan_rows.append({
            'model': model_name,
            'patient': scan[0],
            'week': scan[1],
            'region': region,
            'dice': dice,
            'iou': iou,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
        })

    print(f'\n{model_name.upper()} REGIONAL RESULTS')

    # Compare pooled pixel-level scores and mean per-scan scores.
    for region in REGIONS:
        tp, fp, fn = map(int, pooled[region])
        pooled_dice, pooled_iou = scores(tp, fp, fn)

        applicable = [
            r for r in scan_rows
            if r['region'] == region and np.isfinite(r['dice'])
        ]

        mean_dice = (
            float(np.mean([r['dice'] for r in applicable]))
            if applicable else float('nan')
        )

        mean_iou = (
            float(np.mean([r['iou'] for r in applicable]))
            if applicable else float('nan')
        )

        regional_rows.append({
            'model': model_name,
            'region': region,
            'pooled_dice': pooled_dice,
            'pooled_iou': pooled_iou,
            'mean_scan_dice': mean_dice,
            'mean_scan_iou': mean_iou,
            'num_scans': len(applicable),
            'tp': tp,
            'fp': fp,
            'fn': fn,
        })

        print(
            f'{region}: pooled Dice={pooled_dice:.4f}, '
            f'mean scan Dice={mean_dice:.4f}, '
            f'pooled IoU={pooled_iou:.4f}'
        )

    return regional_rows, scan_rows, slice_rows, consistency_rows, component_rows


def main():
    """Evaluate both trained models and save regional analysis results."""
    print('Coronal slices: numbered slices progress anterior to posterior.')
    print('Within each slice, increasing axis-1 indices move toward the base.')
    print('Regions are approximate GT-derived SI thirds, not clinical landmarks.')
    print(f'Prostate class: {PROSTATE_CLASS}')

    groups = get_groups()
    bounds = get_region_bounds(groups)

    print(f'Analysing {len(bounds)} patient/week scans, {len(test_dataset)} test slices.')

    if not bounds:
        raise ValueError('No valid scans with prostate ground truth')

    all_regional = []
    all_scans = []
    all_slices = []
    all_consistency = []
    all_components = []

    # Evaluate the baseline and Improved U-Net using their best checkpoints.
    for model_name, checkpoint in CHECKPOINTS.items():
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f'Missing checkpoint: {checkpoint}')

        model = load_model(model_name, checkpoint)
        result = evaluate_regions(model_name, model, groups, bounds)

        for destination, rows in zip(
            (all_regional, all_scans, all_slices, all_consistency, all_components),
            result
        ):
            destination.extend(rows)

        del model

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Save regional, per-scan, per-slice and consistency results.
    save_csv(
        RESULTS_DIR / 'regional_comparison.csv',
        all_regional,
        ['model', 'region', 'pooled_dice', 'pooled_iou', 'mean_scan_dice',
         'mean_scan_iou', 'num_scans', 'tp', 'fp', 'fn']
    )

    save_csv(
        RESULTS_DIR / 'regional_per_scan.csv',
        all_scans,
        ['model', 'patient', 'week', 'region', 'dice', 'iou', 'tp', 'fp', 'fn']
    )

    save_csv(
        RESULTS_DIR / 'regional_per_slice.csv',
        all_slices,
        ['model', 'patient', 'week', 'slice', 'index', 'filename', 'region',
         'si_lower', 'si_upper', 'dice', 'iou', 'tp', 'fp', 'fn', 'total_errors']
    )

    save_csv(
        RESULTS_DIR / 'inter_slice_consistency.csv',
        all_consistency,
        ['model', 'patient', 'week', 'slice', 'predicted_area_change',
         'ground_truth_area_change', 'signed_change_discrepancy']
    )

    save_csv(
        RESULTS_DIR / 'disconnected_components.csv',
        all_components,
        ['model', 'patient', 'week', 'slice', 'filename', 'predicted_components',
         'ground_truth_components', 'extra_components']
    )

    # Rank apex/base failure candidates by lowest Dice, then highest error count.
    boundary = [
        r for r in all_slices
        if r['region'] in ('Apex', 'Base')
    ]

    boundary.sort(key=lambda r: (r['dice'], -r['total_errors']))

    save_csv(
        RESULTS_DIR / 'boundary_failure_candidates.csv',
        boundary,
        ['model', 'patient', 'week', 'slice', 'index', 'filename', 'region',
         'si_lower', 'si_upper', 'dice', 'iou', 'tp', 'fp', 'fn', 'total_errors']
    )

    print('\nInspect top-ranked cases visually before selecting 3–5 examples.')

    for row in boundary[:10]:
        print(
            f"{row['model']}: {row['filename']} {row['region']} "
            f"Dice={row['dice']:.3f}, FP={row['fp']}, FN={row['fn']}"
        )


if __name__ == '__main__':
    main()