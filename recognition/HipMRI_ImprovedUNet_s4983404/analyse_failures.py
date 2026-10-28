"""Validation-only slice error inspection; HD95 in array pixels, never millimetres."""
from pathlib import Path
import argparse
import json
import logging
import numpy as np
import torch
from scipy import ndimage
from dataset import ROOT, HipMRISlices
from modules import build_model, load_checkpoint
from predict import panel


def boundary_errors(predicted: np.ndarray, actual: np.ndarray) -> dict:
    intersection = int(np.logical_and(predicted, actual).sum())
    total = int(predicted.sum() + actual.sum())
    result = {'dice': 2*intersection/total if total else None,
              'tp_pixels': intersection,
              'fp_pixels': int((predicted & ~actual).sum()),
              'fn_pixels': int((~predicted & actual).sum()),
              'predicted_components_4connected': int(ndimage.label(predicted)[1]),
              'hd95_pixels': None, 'empty_case': None}
    if not predicted.any() or not actual.any():
        result['empty_case'] = ('both_empty' if not predicted.any() and not actual.any()
                                else 'one_empty_unbounded_distance')
        return result
    predicted_edge = predicted & ~ndimage.binary_erosion(predicted)
    actual_edge = actual & ~ndimage.binary_erosion(actual)
    distances = np.concatenate((ndimage.distance_transform_edt(~actual_edge)[predicted_edge],
                                ndimage.distance_transform_edt(~predicted_edge)[actual_edge]))
    result['hd95_pixels'] = float(np.percentile(distances, 95))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; pass --device cpu')
    torch.set_num_threads(2)
    checkpoint = load_checkpoint(args.checkpoint, args.device)
    config = checkpoint['config']
    model = build_model(config['model'], config['width']).to(args.device).eval()
    model.load_state_dict(checkpoint['state_dict'])
    dataset = HipMRISlices(args.root, 'validate')
    args.output.mkdir(parents=True, exist_ok=False)
    logging.info('Loaded checkpoint epoch=%d', checkpoint['epoch'])
    rows = []
    for image, target, name in dataset:
        with torch.inference_mode():
            prediction = model(image.unsqueeze(0).to(args.device)).argmax(1)[0].cpu().numpy()
        truth = target.argmax(0).numpy()
        rows.append({'name': name, **boundary_errors(prediction == 5, truth == 5)})
    total_tp = sum(row['tp_pixels'] for row in rows)
    denominator = sum(2*row['tp_pixels'] + row['fp_pixels'] + row['fn_pixels'] for row in rows)
    pooled_dice = 2*total_tp/denominator
    expected = checkpoint['validation']['dice'][5]
    if config['val_limit'] is None and abs(pooled_dice-expected) > 1e-4:
        raise AssertionError('Failure-analysis predictions disagree with saved validation metrics')
    logging.info('Validation consistency: observed=%.6f saved=%.6f', pooled_dice, expected)
    (args.output/'slice_metrics.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    # Worst positive-label slices are inspectable failures, not an unbiased sample.
    positive = [row for row in rows if row['fn_pixels'] > 0 and row['dice'] is not None]
    worst = sorted(positive, key=lambda row: (row['dice'], row['name']))[:5]
    lookup = {pair[0].name: index for index, pair in enumerate(dataset.pairs)}
    for row in worst:
        image, target, _ = dataset[lookup[row['name']]]
        with torch.inference_mode():
            prediction = model(image.unsqueeze(0).to(args.device)).argmax(1)[0].cpu().numpy()
        panel(image[0].numpy(), target.argmax(0).numpy(), prediction,
              args.output/f'{row["name"].removesuffix(".nii.gz")}.png')
    (args.output/'selected_failures.json').write_text(json.dumps(worst, indent=2), encoding='utf-8')
    logging.info('Inspected %d validation slices; saved %d worst-positive panels', len(rows), len(worst))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
