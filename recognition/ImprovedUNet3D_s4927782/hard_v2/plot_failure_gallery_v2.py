"""Descriptive v1/v2 prostate failure gallery on unchanged original GT.

Only completed, hash-bound retrospective comparison outputs are accepted. No
inference or candidate selection occurs. Display crops use GT plus all four
prostate predictions, and are never fed to a model or used for scoring.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import compare_common_grid as common

ORDER = [('v1', 'unet2d', 'v1 2D U-Net'), ('v1', 'improved3d', 'v1 residual 3D'),
         ('v2', 'unet2d', 'v2 2D U-Net'), ('v2', 'improved3d', 'v2 residual 3D')]
SELECTION = ('For each patient, choose the visit with the lowest v2 improved3d '
             'complete-volume prostate Dice; undefined values sort after defined '
             'values; ties by patient/visit. Sort patient minima the same way and '
             'show the first five distinct patients. All cases remain in the '
             'full common-grid report. This selects illustrations, never models.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def key(row):
    return row['patient'], row['visit']


def rank(row):
    score = row['per_class'][5]['dice']
    if score is not None:
        require(math.isfinite(float(score)) and 0 <= score <= 1, 'Invalid gallery ranking Dice.')
    return score is None, score if score is not None else 0., row['patient'], row['visit']


def select_keys(cases, count=5):
    """Patient-diverse worst-case display rule, fixed before viewing v2 results."""
    grouped, seen = defaultdict(list), set()
    for row in cases:
        require(key(row) not in seen, 'Duplicate ranking case.')
        seen.add(key(row)); grouped[row['patient']].append(row)
    minima = [min(rows, key=rank) for rows in grouped.values()]
    return [key(row) for row in sorted(minima, key=rank)[:count]]


def display_slices(target):
    """Unique GT-depth proxy slices; repeated roles are explicitly collapsed."""
    area = np.count_nonzero(target == 5, axis=(1, 2))
    present = np.flatnonzero(area)
    if present.size:
        roles = [('inferior label extent', int(present[0])),
                 ('largest GT area', int(area.argmax())),
                 ('superior label extent', int(present[-1]))]
    else:
        depth = target.shape[0]
        roles = [('GT absent: depth quartile', depth // 4),
                 ('GT absent: depth middle', depth // 2),
                 ('GT absent: depth three-quarter', min(depth - 1, 3 * depth // 4))]
    indices, labels = [], []
    for label, index in roles:
        if index in indices:
            labels[indices.index(index)].append(label)
        else:
            indices.append(index); labels.append([label])
    return [{'z': z, 'roles': role} for z, role in zip(indices, labels)]


def display_roi(target, predictions, padding=8):
    """One H/W rectangle from the full-depth union, shared by every panel."""
    union = target == 5
    for prediction in predictions:
        require(prediction.shape == target.shape, 'Display predictions use different grids.')
        union |= prediction == 5
    coords = np.argwhere(union)
    if not coords.size:
        return [0, target.shape[1], 0, target.shape[2]]
    y0, x0 = coords[:, 1:].min(0)
    y1, x1 = coords[:, 1:].max(0) + 1
    return [max(0, int(y0) - padding), min(target.shape[1], int(y1) + padding),
            max(0, int(x0) - padding), min(target.shape[2], int(x1) + padding)]


def prostate_counts(prediction, target):
    """Independent complete-volume binary counts, not a display-crop score."""
    predicted, actual = prediction == 5, target == 5
    tp = int(np.count_nonzero(predicted & actual))
    fp = int(np.count_nonzero(predicted & ~actual)); fn = int(np.count_nonzero(~predicted & actual))
    denominator = 2 * tp + fp + fn
    occupied = np.flatnonzero(np.any(actual, axis=(1, 2)))
    outside = np.ones(target.shape[0], dtype=bool)
    if occupied.size:
        outside[occupied[0]:occupied[-1] + 1] = False
    errors = np.count_nonzero(predicted != actual, axis=(1, 2))
    fp_by_true_class = {str(c): int(np.count_nonzero(predicted & (target == c))) for c in range(5)}
    fn_by_predicted_class = {str(c): int(np.count_nonzero(actual & (prediction == c))) for c in range(5)}
    require(sum(fp_by_true_class.values()) == fp and sum(fn_by_predicted_class.values()) == fn,
            'Prostate error-class counts must sum to complete-volume FP/FN.')
    return dict(tp=tp, fp=fp, fn=fn, target_voxels=tp + fn,
                predicted_voxels=tp + fp, intersection=tp, union=tp + fp + fn,
                dice=2 * tp / denominator if denominator else None,
                iou=tp / (tp + fp + fn) if tp + fp + fn else None,
                false_positive_by_true_class=fp_by_true_class,
                false_negative_by_predicted_class=fn_by_predicted_class,
                outside_gt_depth_false_positive_voxels=int(np.count_nonzero(predicted[outside])),
                maximum_error_slice_z=int(errors.argmax()))


def verify_case(prediction, target, spacing, expected):
    measured = prostate_counts(prediction, target)
    saved = expected['per_class'][5]
    for name in ('target_voxels', 'predicted_voxels', 'intersection', 'union', 'dice', 'iou'):
        actual, recorded = measured[name], saved[name]
        require((actual is None and recorded is None) or
                (actual is not None and recorded is not None and abs(actual - recorded) < 1e-12),
                'Complete-volume gallery metric differs from common-grid report: ' + name)
    surface = common.prostate_surface_metrics(prediction, target, spacing)
    require(surface['empty_status'] == expected['prostate_surface']['empty_status'], 'Gallery empty-mask policy differs.')
    a, b = surface['hd95_mm'], expected['prostate_surface']['hd95_mm']
    require((a is None and b is None) or (a is not None and b is not None and abs(a - b) < 1e-10),
            'Gallery HD95 differs from common-grid report.')
    return dict(measured, **surface)


def _contour(axis, mask, colour, width, style):
    if mask.any():
        height, width = mask.shape
        axis.contour(np.arange(-1, width + 1), np.arange(-1, height + 1),
                     np.pad(mask.astype(np.uint8), 1), levels=[.5],
                     colors=[colour], linewidths=width, linestyles=style)


def render_case(image, target, predictions, measured, spacing, selected_slices,
                roi, patient, visit, destination, synthetic=False):
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'hipmri-v2-gallery-mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    nonzero = image[image != 0]
    vmin, vmax = (np.percentile(nonzero, [1, 99]).tolist() if nonzero.size else [0., 1.])
    if vmax <= vmin:
        vmax = vmin + 1.
    y0, y1, x0, x1 = roi
    full = roi == [0, image.shape[1], 0, image.shape[2]]
    columns = len(selected_slices)
    fig, axes = plt.subplots(4, columns, figsize=(max(7.2, 4.1 * columns + 1.9), 12.), squeeze=False)
    fig.subplots_adjust(left=.20 if columns > 1 else .27, right=.985, top=.90,
                        bottom=.125, hspace=.20, wspace=.075)
    for row_index, (version, model, title) in enumerate(ORDER):
        prediction = predictions[version, model]
        item = measured[version, model]
        score = 'undefined' if item['dice'] is None else f"{item['dice']:.4f}"
        hd = 'undefined' if item['hd95_mm'] is None else f"{item['hd95_mm']:.2f} mm"
        row_label = (f"{title}\nWhole-volume Dice {score}\nTP {item['tp']}  FP {item['fp']}\n"
                     f"FN {item['fn']}  HD95 {hd}\nOut-of-extent FP {item['outside_gt_depth_false_positive_voxels']}")
        for col, selection in enumerate(selected_slices):
            z = selection['z']; axis = axes[row_index, col]
            im = image[z, y0:y1, x0:x1]
            gt = target[z, y0:y1, x0:x1] == 5
            pr = prediction[z, y0:y1, x0:x1] == 5
            axis.imshow(im, cmap='gray', vmin=vmin, vmax=vmax, origin='lower',
                        aspect=float(spacing[1] / spacing[2]), interpolation='nearest')
            overlay = np.zeros((*im.shape, 4), dtype=np.float32)
            overlay[pr & ~gt] = [.98, .15, .13, .35]
            overlay[~pr & gt] = [.18, .40, 1., .35]
            axis.imshow(overlay, origin='lower', aspect=float(spacing[1] / spacing[2]), interpolation='nearest')
            _contour(axis, gt, '#15ef6e', 1.7, '-')
            _contour(axis, pr, '#ffe134', 1.0, '--')
            axis.set_xlim(-.5, im.shape[1] - .5); axis.set_ylim(-.5, im.shape[0] - .5)
            axis.set_xticks([]); axis.set_yticks([])
            if row_index == 0:
                axis.set_title(' / '.join(selection['roles']) + f"\nz={z}", fontsize=10)
            if col == 0:
                axis.set_ylabel(row_label, rotation=0, ha='right', va='center', fontsize=9, labelpad=14)
    view = 'full field of view' if full else f'display ROI H[{y0}:{y1}] W[{x0}:{x1}]'
    prefix = 'SYNTHETIC FIXTURE — ' if synthetic else ''
    fig.suptitle(f'{prefix}{patient} / {visit} | Prostate on common original grid\n{view}', fontsize=14, y=.975)
    fig.legend(handles=[Line2D([0], [0], color='#15ef6e', lw=1.7, label='GT contour'),
                        Line2D([0], [0], color='#ffe134', lw=1.2, linestyle='--', label='Prediction contour'),
                        Patch(color='#fa2621', alpha=.5, label='False positive'),
                        Patch(color='#2e66ff', alpha=.5, label='False negative')],
               loc='lower center', bbox_to_anchor=(.58, .073), ncol=4, frameon=False, fontsize=9)
    crop_note = ('ROI = full-depth union of GT and all four prostate predictions +8 voxel padding; display only, never inference/metrics.'
                 if not full else 'Full field displayed for every model; all scores and FP/FN counts use the entire volume.')
    notes = (crop_note + '\nLabel-extent slices are proxies, not independently annotated anatomical apex/base. Repeated slice roles are merged.\n'
             'Only these planes are illustrated; out-of-extent FP and worst-error slice remain in the receipt. Canonical axial RAS; physical H/W aspect.')
    fig.text(.58, .023, notes, ha='center', va='center', fontsize=7.8)
    fig.savefig(destination, dpi=145, facecolor='white')
    plt.close(fig)
    return {'intensity_window': [float(vmin), float(vmax)], 'intensity_window_rule': 'Per-case nonzero MRI1st/99th percentiles, identical across models and views.',
            'display_roi_hw': roi, 'physical_in_plane_aspect': float(spacing[1] / spacing[2])}


def build_gallery(args):
    output = Path(args.output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError('Use a fresh gallery directory; earlier failures and figures are preserved.')
    start = time.perf_counter()
    project, artifacts = Path(args.v1_project), Path(args.artifacts_root)
    report_path = Path(args.common_test); report = common.read_json(report_path)
    require(report.get('split') == 'test' and report.get('retrospective_test_acknowledged') is True,
            'Gallery requires the complete acknowledged retrospective TEST comparison.')
    manifest_paths = {'v1': project / 'artifacts/cache64/manifest.json', 'v2': artifacts / 'cache_v2/manifest.json'}
    manifests, records = {}, {}
    source_files = {'common_test': {'sha256': common.file_sha256(report_path), 'path': str(report_path)}}
    for version, path in manifest_paths.items():
        digest = common.file_sha256(path)
        require(digest == report['input_files'][version + '_manifest']['sha256'], 'Gallery manifest differs from comparison.')
        manifests[version] = common.read_json(path); records[version] = common.index_records(manifests[version])
        source_files[version + '_manifest'] = {'sha256': digest, 'path': str(path)}
    require(manifests['v1']['patient_splits'] == manifests['v2']['patient_splits'], 'Gallery split changed.')
    keys = sorted(k for k, r in records['v1'].items() if r['split'] == 'test')
    require(keys == sorted(k for k, r in records['v2'].items() if r['split'] == 'test'), 'Gallery TEST cases differ.')
    identity_path = manifest_paths['v1'].parent / 'cache_identity.json'
    identity_digest = common.file_sha256(identity_path)
    require(identity_digest == common.FROZEN_V1_CACHE_IDENTITY_SHA256
            and identity_digest == report['input_files']['v1_cache_identity']['sha256'], 'Original image/GT identity book differs.')
    identity = common.read_json(identity_path)
    require(identity['manifest_sha256'] == common.file_sha256(manifest_paths['v1']), 'Image identity book manifest differs.')
    source_files['v1_cache_identity'] = {'sha256': identity_digest, 'path': str(identity_path)}
    common_cases, evaluations, eval_cases = {}, {}, {}
    for version, model, _ in ORDER:
        rows = report['models'][model][version]['cases']
        common_cases[version, model] = {key(r): r for r in rows}
        require(len(common_cases[version, model]) == len(rows) and set(common_cases[version, model]) == set(keys),
                'Common-grid gallery evidence must cover the complete split for all models.')
        eval_path = (project / f'artifacts/test_final/{model}/evaluation.json' if version == 'v1'
                     else artifacts / f'test_v2/{model}/evaluation.json')
        name = version + '_' + model
        require(common.file_sha256(eval_path) == report['input_files']['evaluations'][name]['sha256'], 'Evaluation differs from comparison.')
        ev, indexed = common.validate_evaluation(eval_path, manifest_paths[version], manifests[version], 'test', keys, model)
        require(ev['checkpoint_sha256'] == report['input_files']['evaluations'][name]['checkpoint_sha256'], 'Checkpoint changed.')
        if version == 'v1':
            require(ev['checkpoint_sha256'] == common.FROZEN_V1_CHECKPOINTS[model], 'Original checkpoint changed.')
        evaluations[version, model], eval_cases[version, model] = eval_path, indexed
        source_files[name + '_evaluation'] = {'sha256': common.file_sha256(eval_path), 'path': str(eval_path)}
    chosen = select_keys(list(common_cases['v2', 'improved3d'].values()))
    require(chosen and len(set(k[0] for k in chosen)) == len(chosen), 'Gallery must use distinct patients.')
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(status='running',purpose='Descriptive failure illustrations; no inference, tuning or model selection',
                   synthetic_fixture=bool(report.get('synthetic_fixture', False)),
                   selection_algorithm=SELECTION, selected_case_keys=[list(k) for k in chosen],
                   all_test_cases_retained_in_common_report=len(keys), score_grid='original immutable v1 downsampled grid',
                   native_resolution_accuracy_claimed=False, source_files=source_files,
                   context_class_index_names={'0': 'Background', '1': 'Body', '2': 'Bones', '3': 'Bladder', '4': 'Rectum', '5': 'Prostate'},
                   error_context_limit='FP/FN class-confusion counts describe label assignments; no organ-wall contact distance, anatomical apex/base annotation or causal mechanism is measured.',
                   script_sha256=common.file_sha256(__file__), mapping_source_sha256=common.file_sha256(common.__file__),
                   recorded_at_utc=datetime.now(timezone.utc).isoformat(), cases=[])
    def save():
        temporary = output / 'gallery_receipt.pending.json'
        temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
        temporary.replace(output / 'gallery_receipt.json')
    save()
    try:
        for patient, visit in chosen:
            selected = (patient, visit); old = records['v1'][selected]
            cache_path = common.resolve_relative(manifest_paths['v1'].parent, old['cache_path'])
            with np.load(cache_path, allow_pickle=False) as stored:
                image, target, spacing = (stored[x].copy() for x in ('image', 'mask', 'spacing'))
            require(image.shape == target.shape == tuple(manifests['v1']['target_shape_dhw'])
                    and np.isfinite(image).all(), 'Gallery MRI/GT grid is invalid.')
            frozen = identity['records'][old['cache_path']]['arrays']
            for label, values in [('image', image), ('mask', target), ('spacing', spacing)]:
                require(common.decoded_array_sha256(values) == frozen[label]['sha256'], 'Gallery MRI/GT bytes differ: ' + label)
            predictions, measured, prediction_sources = {}, {}, {}
            for version, model, _ in ORDER:
                record = records[version][selected]
                prediction, evidence = common.load_prediction(evaluations[version, model], eval_cases[version, model][selected],
                    manifests[version]['target_shape_dhw'], record, require_affine=version == 'v2')
                expected = common_cases[version, model][selected]
                require(evidence['sha256'] == expected['prediction_evidence']['sha256'], 'Gallery prediction file differs from scored file.')
                if version == 'v2':
                    prediction, _ = common.categorical_to_grid(prediction, common.affine(record), common.affine(old), target.shape)
                predictions[version, model] = prediction
                measured[version, model] = verify_case(prediction, target, spacing, expected)
                prediction_sources[version + '_' + model] = evidence
            slices = display_slices(target)
            roi = display_roi(target, list(predictions.values()), padding=8)
            filename = f'{patient}_{visit}'
            require(all(ch.isalnum() or ch in '_-' for ch in filename), 'Unsafe figure identity.')
            figures = {}
            for view, box in [('full_fov', [0, image.shape[1], 0, image.shape[2]]), ('display_roi', roi)]:
                destination = output / f'{filename}_{view}.png'
                detail = render_case(image, target, predictions, measured, spacing, slices, box,
                                     patient, visit, destination, synthetic=receipt['synthetic_fixture'])
                figures[view] = dict(file=destination.name, sha256=common.file_sha256(destination), **detail)
            receipt['cases'].append(dict(patient=patient, visit=visit, display_slices=slices,
                duplicated_slice_roles_collapsed=any(len(x['roles']) > 1 for x in slices),
                display_roi_hw=roi, roi_rule='3D GT+four-prediction prostate union projected to H/W, +8 voxel padding, clipped to full FOV; visualization only.',
                complete_volume_metrics={v + '_' + m: measured[v, m] for v, m, _ in ORDER},
                cache_file_sha256=common.file_sha256(cache_path), prediction_sources=prediction_sources, figures=figures))
            save()
        receipt.update(status='completed', elapsed_cpu_wall_seconds=time.perf_counter() - start,
                       raw_arrays_saved=False, model_or_threshold_changed=False)
        save()
    except BaseException as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        save(); raise
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v1-project', type=Path, required=True)
    parser.add_argument('--artifacts-root', type=Path, required=True)
    parser.add_argument('--common-test', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_gallery(args)
    print(json.dumps({'status': result['status'], 'selected_case_keys': result['selected_case_keys'],
                      'output': str(args.output_dir), 'figures': len(result['cases']) * 2}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
