"""Plot five selected test failures from saved predictions, without inference."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


CASES = [
    (42, 0, 40, 'Empty prediction on a positive slice'),
    (42, 0, 9, 'Prediction with no target overlap'),
    (42, 0, 16, 'Under-segmentation'),
    (41, 0, 39, 'Over-segmentation'),
    (40, 3, 33, 'False positive on a negative slice'),
]


def counts(target, prediction):
    gt, pred = target == 5, prediction == 5
    tp = int((gt & pred).sum())
    g, p = int(gt.sum()), int(pred.sum())
    return dict(gt_pixels=g, predicted_pixels=p, fp_pixels=p-tp,
                fn_pixels=g-tp, dice=2*tp/(g+p) if g else None)


def plot(image, target, predictions, key, description, output):
    fig, axes = plt.subplots(2, 4, figsize=(14, 8), constrained_layout=True)
    low, high = np.percentile(image, [1, 99])
    if high <= low:
        high = low + 1
    stats = {}
    for row, (label, prediction) in enumerate(predictions.items()):
        score = counts(target, prediction)
        stats[label] = score
        gt, pred = target == 5, prediction == 5
        overlay = np.zeros((*gt.shape, 4), dtype=np.float32)
        overlay[pred & ~gt] = [1, 0, 0, 0.85]
        overlay[gt & ~pred] = [0, 1, 1, 0.85]
        axes[row, 0].imshow(image, cmap='gray', vmin=low, vmax=high, interpolation='nearest')
        axes[row, 1].imshow(gt, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        axes[row, 2].imshow(pred, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        axes[row, 3].imshow(image, cmap='gray', vmin=low, vmax=high, interpolation='nearest')
        axes[row, 3].imshow(overlay, interpolation='nearest')
        dice = f"{score['dice']:.4f}" if score['dice'] is not None else 'N/A (GT absent)'
        titles = [f'{label}\nMRI', f"True prostate\n{score['gt_pixels']} pixels",
                  f'Predicted prostate\nDice: {dice}',
                  f"Errors\nFP: {score['fp_pixels']}; FN: {score['fn_pixels']}"]
        for axis, title in zip(axes[row], titles):
            axis.set_title(title, fontsize=11)
            axis.set_axis_off()
        axes[row, 3].legend(handles=[Patch(color='red', label='False positive'),
                                    Patch(color='cyan', label='False negative')],
                            loc='upper center', bbox_to_anchor=(0.5, -0.01), fontsize=9)
    fig.suptitle(f'Test case {key[0]:03d}, week {key[1]}, slice {key[2]}\n'
                 f'{description} | Stored array orientation', fontsize=13)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('runs/eval_test_713255/results'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Dataset loading uses CPU only. No model or checkpoint weights are loaded.
    import torch
    from dataset import HipMRIDataset
    torch.set_num_threads(2)
    summaries = {name: json.loads((args.results/name/'summary.json').read_text())
                 for name in ('standard', 'improved')}
    assert all(s['split'] == 'test' for s in summaries.values())
    data = HipMRIDataset(split='test')
    lookup = {(r.case_id, r.week, r.slice_id): i for i,r in enumerate(data.samples)}
    reference = {}
    for name in summaries:
        with (args.results/name/'slice_metrics_all_classes.csv').open() as handle:
            reference[name] = {
                tuple(int(r[k]) for k in ('case_id','week','slice_id')): r
                for r in csv.DictReader(handle) if int(r['class_id']) == 5
            }
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for case, week, section, description in CASES:
        key = (case, week, section)
        sample = data[lookup[key]]
        image, target = sample['image'][0].numpy(), sample['mask'].numpy()
        predictions = {}
        for name, label in [('standard','Standard U-Net'), ('improved','Improved U-Net')]:
            with np.load(args.results/name/f'case_{case:03d}_predictions.npz', allow_pickle=False) as archive:
                prediction = archive[f'week_{week}_slice_{section}'].copy()
            assert prediction.shape == target.shape
            assert np.isin(prediction, np.arange(6)).all()
            score = counts(target, prediction)
            expected = reference[name][key]
            for field in ('gt_pixels','predicted_pixels','fp_pixels','fn_pixels'):
                assert score[field] == int(expected[field]), (key, name, field)
            if score['dice'] is None:
                assert expected['dice'] == ''
            else:
                assert abs(score['dice'] - float(expected['dice'])) < 1e-12
            predictions[label] = prediction
        filename = f'test_case_{case:03d}_week_{week}_slice_{section}.png'
        stats = plot(image, target, predictions, key, description, args.output/filename)
        records.append(dict(case_id=case, week=week, slice_id=section,
                            selection_reason=description, figure=filename, metrics=stats))
        print('Saved:', filename, '| Saved-prediction/CSV agreement: PASS', flush=True)
    manifest = {'split':'test', 'source_results':str(args.results.resolve()),
                'checkpoint_sha256':{n:s['checkpoint_sha256'] for n,s in summaries.items()},
                'selection_protocol':'Five descriptive error patterns chosen after test evaluation; no model tuning',
                'examples': records}
    (args.output/'selection.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Complete. Five figures generated from stored predictions; no model inference.')


if __name__ == '__main__':
    main()
