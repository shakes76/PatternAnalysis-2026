"""Read-only exhaustive pair/pixel audit; write manifests only to a NEW output dir."""
from pathlib import Path
import argparse
import csv
import json
import logging
import time
from collections import Counter
import nibabel as nib
import numpy as np
from dataset import ROOT, PATTERN, pair_paths, load_pair, check_patient_splits


def audit_split(root: Path, split: str, output: Path) -> dict:
    counts = Counter()
    shapes = Counter()
    rows = []
    for index, (image_path, label_path) in enumerate(pair_paths(root, split)):
        image, labels = load_pair(image_path, label_path)
        unique, frequencies = np.unique(labels.numpy(), return_counts=True)
        counts.update(dict(zip(map(int, unique), map(int, frequencies))))
        nifti = nib.load(image_path)
        shapes[str(tuple(image.shape))] += 1
        patient, week, slice_index = PATTERN.fullmatch(image_path.name).groups()
        rows.append({'image': image_path.name, 'label': label_path.name,
                     'patient': patient, 'week': week, 'slice': int(slice_index),
                     'label5_pixels': int((labels == 5).sum()),
                     'spacing_header': str(nifti.header.get_zooms()),
                     'units': str(nifti.header.get_xyzt_units())})
        if index % 500 == 0:
            logging.info('%s %d pairs checked', split, index)
    with (output/f'{split}.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return {'pairs': len(rows), 'shapes': dict(shapes), 'pixels_by_label': dict(counts),
            'label5_positive_slices': sum(row['label5_pixels'] > 0 for row in rows)}


def preview(root: Path, output: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    candidates = pair_paths(root, 'train')
    selected = []
    for image_path, label_path in candidates:
        image, labels = load_pair(image_path, label_path)
        if (labels == 5).sum() > 50:
            selected.append((image_path, image, labels))
        if len(selected) == 3:
            break
    figure, axes = plt.subplots(len(selected), 3, figsize=(9, 8))
    for row, (path, image, labels) in enumerate(selected):
        axes[row, 0].imshow(image[0].T, cmap='gray', origin='lower')
        axes[row, 1].imshow(labels.T, vmin=0, vmax=5, cmap='tab10', origin='lower')
        axes[row, 2].imshow(image[0].T, cmap='gray', origin='lower')
        axes[row, 2].contour(labels.T == 5, levels=[0.5], colors=['red'])
        for axis in axes[row]:
            axis.set_title(path.name.replace('.nii.gz', ''), fontsize=7)
            axis.axis('off')
    figure.tight_layout()
    figure.savefig(output/'train_label_preview.png', dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    summary = {'subjects': check_patient_splits(args.root)}
    for split in ('train', 'validate', 'test'):
        summary[split] = audit_split(args.root, split, args.output)
    preview(args.root, args.output)
    summary['elapsed_seconds'] = time.perf_counter() - started
    summary['label_names_status'] = '0..5 observed; anatomical names require provenance confirmation'
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    logging.info('%s', json.dumps(summary))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    main()
