"""Frozen-test boundary analysis and deterministic, post-hoc failure illustrations.

Boundary F1 uses metrics.boundary_f1: 8-neighbour contours and a two-pixel
Chebyshev tolerance. This is not physical surface Dice, HD95 or clinical accuracy.
Test analysis is descriptive only and does not select a new model/threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import HipMRIDataset
from metrics import boundary_f1, confusion_matrix, scores_from_confusion
from modules import build_model
from predict import _device, _validate_checkpoint


def mean_defined(values):
    valid = [v for v in values if v is not None]
    return {"mean": statistics.mean(valid) if valid else None,
            "defined_slices": len(valid), "undefined_slices": len(values) - len(valid)}


def select_failures(rows):
    """Worst positive prostate Dice per Case, then largest FN and FP; distinct slices."""
    chosen = []
    for case in sorted({r['case_alias'] for r in rows}):
        candidates = [r for r in rows if r['case_alias'] == case and r['prostate_truth_pixels'] > 0]
        if candidates:
            r = min(candidates, key=lambda x: (x['models']['improved_unet']['prostate_dice'], x['index']))
            chosen.append((r, 'Lowest Improved prostate Dice among positive slices in this Case'))
    for field in ('false_negative_pixels', 'false_positive_pixels'):
        candidates = [r for r in rows if r['index'] not in {x[0]['index'] for x in chosen}
                      and r['models']['improved_unet'][field] > 0]
        if candidates:
            chosen.append((min(candidates, key=lambda x: (-x['models']['improved_unet'][field], x['index'])),
                           'Largest remaining Improved ' + field))
    return chosen[:5]


def summarize(rows):
    """Report slice weighting explicitly, with separate absent/positive populations."""
    result = {}
    for model in rows[0]['models']:
        per_class = []
        for channel in range(1, 6):
            per_class.append({'class_index': channel,
                'all_slices': mean_defined([r['models'][model]['boundary_f1'][channel-1] for r in rows]),
                'truth_positive_slices': mean_defined([r['models'][model]['boundary_f1'][channel-1]
                                                       for r in rows if r['truth_pixels'][channel] > 0])})
        groups = {}
        for key, selected in (
            ('positive', [r for r in rows if r['prostate_truth_pixels'] > 0]),
            ('small_positive_le_100_pixels', [r for r in rows if 0 < r['prostate_truth_pixels'] <= 100]),
            ('larger_positive_gt_100_pixels', [r for r in rows if r['prostate_truth_pixels'] > 100]),
            ('annotation_endpoints', [r for r in rows if r['annotation_endpoint']]),
            ('positive_non_endpoints', [r for r in rows if r['prostate_truth_pixels'] > 0 and not r['annotation_endpoint']])):
            groups[key] = {'slices': len(selected),
                           'boundary_f1': mean_defined([r['models'][model]['boundary_f1'][4] for r in selected]),
                           'dice': mean_defined([r['models'][model]['prostate_dice'] for r in selected])}
        result[model] = {'per_class_boundary_f1': per_class, 'prostate_subgroups': groups,
            'prostate_absent_slices': sum(r['prostate_truth_pixels'] == 0 for r in rows),
            'prostate_absent_with_false_positive': sum(r['prostate_truth_pixels'] == 0 and r['models'][model]['false_positive_pixels'] > 0 for r in rows),
            'prostate_false_positive_pixels': sum(r['models'][model]['false_positive_pixels'] for r in rows),
            'prostate_false_negative_pixels': sum(r['models'][model]['false_negative_pixels'] for r in rows),
            'prostate_predictions_on_truth_bladder_pixels': sum(r['models'][model]['on_bladder'] for r in rows),
            'prostate_predictions_on_truth_rectum_pixels': sum(r['models'][model]['on_rectum'] for r in rows)}
    return result


def run(checkpoints, manifest, output, device='cuda', batch_size=4):
    output, manifest = Path(output), Path(manifest)
    if output.exists():
        raise FileExistsError('Choose a new analysis output directory')
    if isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError('batch_size must be positive')
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    digest = sha(manifest)
    selected_device = _device(device)
    models, configs, provenance = {}, {}, {}
    for checkpoint in checkpoints:
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        config, channel = _validate_checkpoint(payload, False)
        if config['manifest_sha256'] != digest or config['label_values'] != list(range(6)) or channel != 5:
            raise ValueError('Require frozen six-class course manifest and prostate channel 5')
        if config['image_size'] != 256 or config['resize_mode'] != 'pad':
            raise ValueError('Analysis protocol requires 256 pad grid')
        name = config['model']
        if name in models:
            raise ValueError('Duplicate model')
        model = build_model(name, num_classes=6, base_channels=config['base_channels'])
        model.load_state_dict(payload['model_state'])
        models[name], configs[name] = model.eval(), config
        provenance[name] = {'checkpoint_sha256': sha(checkpoint), 'epoch': payload['epoch']}
    if set(models) != {'unet', 'improved_unet'}:
        raise ValueError('Both model checkpoints required')
    data = HipMRIDataset(manifest, 'test', 256, list(range(6)), resize_mode='pad')
    aliases = {case: f'Case {i+1:02d}' for i, case in enumerate(sorted({r['patient_id'] for r in data.records}))}
    rows, predictions = [], {name: [] for name in models}
    # Save only compact CPU label arrays between models; no optimization occurs.
    for name, model in models.items():
        model.to(selected_device)
        with torch.inference_mode():
            offset = 0
            for batch in DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0):
                logits = model(batch['image'].to(selected_device))
                if not torch.isfinite(logits).all():
                    raise ValueError('Non-finite predictions')
                preds = logits.argmax(1).cpu()
                for j, (pred, truth) in enumerate(zip(preds, batch['mask'])):
                    index = offset + j
                    predictions[name].append(pred.numpy().astype(np.uint8))
                    if name == next(iter(models)):
                        record = data.records[index]
                        match = re.search(r'_week_(\d+)_slice_(\d+)', Path(record['image']).name)
                        if not match:
                            raise ValueError('Expected course week/slice naming')
                        counts = torch.bincount(truth.flatten(), minlength=6).tolist()
                        rows.append({'index': index, 'case_alias': aliases[record['patient_id']],
                                     'week': int(match[1]), 'slice_index': int(match[2]),
                                     'truth_pixels': counts, 'prostate_truth_pixels': counts[5], 'models': {}})
                    p, t = pred == 5, truth == 5
                    scores = scores_from_confusion(confusion_matrix(pred, truth, 6))
                    rows[index]['models'][name] = {
                        'boundary_f1': [boundary_f1(pred == k, truth == k, tolerance=2) for k in range(1, 6)],
                        'prostate_dice': scores['per_class'][5]['dice'],
                        'false_positive_pixels': int((p & ~t).sum()), 'false_negative_pixels': int((~p & t).sum()),
                        'on_bladder': int((p & (truth == 3)).sum()), 'on_rectum': int((p & (truth == 4)).sum())}
                offset += len(preds)
        model.to('cpu')
        print(f'{name}: analyzed {len(data)} test slices', flush=True)
    endpoints = set()
    for key in {(r['case_alias'], r['week']) for r in rows}:
        positive = [r for r in rows if (r['case_alias'], r['week']) == key and r['prostate_truth_pixels'] > 0]
        if positive:
            endpoints.update([min(positive, key=lambda r: r['slice_index'])['index'], max(positive, key=lambda r: r['slice_index'])['index']])
    for r in rows:
        r['annotation_endpoint'] = r['index'] in endpoints
    # Reconstruct pooled metrics as a guard against accidentally changing inference.
    pooled = {}
    for name in models:
        matrix = torch.zeros((6, 6), dtype=torch.int64)
        for i in range(len(data)):
            matrix += confusion_matrix(torch.from_numpy(predictions[name][i]), data[i]['mask'], 6)
        pooled[name] = scores_from_confusion(matrix)
    result = {'split': 'test', 'slices': len(rows), 'cases': len(aliases), 'manifest_sha256': digest,
              'checkpoints': provenance, 'tolerance_pixels': 2, 'distance': 'Chebyshev', 'grid': '256x256 pad',
              'aggregation': 'Arithmetic mean of defined per-slice boundary F1; both-empty excluded, one-empty zero. Truth-positive means use the same slices for both models.',
              'endpoint_definition': 'First and last prostate-positive filename slice index within each Case/week. Annotation-transition proxy only; not verified anatomical apex/base.',
              'small_target_definition': '1 to 100 GT prostate pixels; descriptive subgroup, not a tuned operating threshold.',
              'models': summarize(rows), 'pooled_overlap_check': pooled,
              'source_sha256': {n: sha(Path(__file__).parent/n) for n in ('analyze_boundaries.py','metrics.py','dataset.py','modules.py','predict.py')},
              'examples': [], 'clinical_performance_claim': False}
    output.mkdir(parents=True)
    for number, (row, reason) in enumerate(select_failures(rows), 1):
        example = f'F{number}'
        sample = data[row['index']]
        image, truth = sample['image'][0].numpy(), sample['mask'].numpy() == 5
        fig, axes = plt.subplots(2, 4, figsize=(12, 7))
        for axis_row, name in zip(axes, ('unet', 'improved_unet')):
            pred = predictions[name][row['index']] == 5
            overlay = np.zeros((*truth.shape, 4))
            overlay[pred & ~truth] = (1, .1, .1, .85)
            overlay[~pred & truth] = (.1, .35, 1, .85)
            axis_row[0].imshow(image, cmap='gray')
            axis_row[1].imshow(truth, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            axis_row[2].imshow(pred, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            axis_row[3].imshow(image, cmap='gray'); axis_row[3].imshow(overlay)
            for ax, title in zip(axis_row, (f'{name}: MRI', 'GT prostate (white)', 'Prediction (white)', 'FP red / FN blue')):
                ax.set_title(title, fontsize=9); ax.axis('off')
        fig.suptitle(f'{example} | {row["case_alias"]} | GT area {row["prostate_truth_pixels"]} px | selected failure, not typical performance', fontsize=10)
        fig.text(.5, .015, 'Data provided by HipMRI Study (CSIRO v2); course 2D slices. Research illustration, not clinical guidance.', ha='center', fontsize=8)
        fig.tight_layout(rect=(0,.04,1,.95)); fig.savefig(output/f'failure_{example}.png', dpi=140); plt.close(fig)
        result['examples'].append({'example': example, 'case_alias': row['case_alias'], 'selection_reason': reason,
                                   'prostate_truth_pixels': row['prostate_truth_pixels'], 'annotation_endpoint': row['annotation_endpoint'],
                                   'models': row['models'], 'figure': f'failure_{example}.png'})
    for path in checkpoints:
        assert sha(path) in {x['checkpoint_sha256'] for x in provenance.values()}
    assert sha(manifest) == digest
    (output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (output/'slice_details_private.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', type=Path, nargs=2, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--batch-size', type=int, default=4)
    torch.set_num_threads(4)
    run(**vars(parser.parse_args()))
