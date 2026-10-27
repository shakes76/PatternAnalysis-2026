"""Three-pipeline descriptive gallery after the completed Hard-only follow-up.

Reads only already-scored predictions, the unchanged original image/GT cache and
their bound metadata. GT-guided crops are for display, never model inference or
scoring. The stopped new 2D arm is neither fabricated nor displayed as trained.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import textwrap
import time

import numpy as np

import compare_common_grid as common
from plot_failure_gallery_v2 import (
    display_roi, display_slices, key, prostate_counts, require, select_keys, verify_case,
)
import plot_failure_gallery_v2 as display_helpers

ORDER = (('v1_unet2d', 'unet2d', 'Original 2D U-Net'),
         ('v1_improved3d', 'improved3d', 'Original residual 3D'),
         ('hard_improved3d', 'improved3d', 'New Hard residual 3D'))
SELECTION = ('For each patient, choose the visit with the lowest new Hard improved3d '
             'complete-volume prostate Dice; undefined values sort after defined '
             'values; ties by patient/visit. Sort patient minima the same way and '
             'show the first five distinct patients. All30 cases remain in the '
             'complete common-grid report. This selects illustrations, never models.')


def _contour(axis, mask, colour, line_width, style):
    """Keep contour linewidth separate from the image width in pixels."""
    if mask.any():
        height, width = mask.shape
        axis.contour(np.arange(-1, width + 1), np.arange(-1, height + 1),
                     np.pad(mask.astype(np.uint8), 1), levels=[.5], colors=[colour],
                     linewidths=line_width, linestyles=style)


def render_case(image, target, predictions, measured, spacing, slices, roi,
                patient, visit, destination, synthetic=False):
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'hipmri-hard-gallery-mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    nonzero = image[image != 0]
    vmin, vmax = np.percentile(nonzero, [1, 99]).tolist() if nonzero.size else [0., 1.]
    if vmax <= vmin:
        vmax = vmin + 1.
    y0, y1, x0, x1 = roi
    full = roi == [0, image.shape[1], 0, image.shape[2]]
    columns = len(slices)
    thin = columns == 1
    fig, axes = plt.subplots(3, columns, figsize=(max(7.4, 4.1*columns+2.0), 10.6 if thin else 9.8), squeeze=False)
    fig.subplots_adjust(left=.21 if columns > 1 else .29, right=.985, top=.89,
                        bottom=.21 if thin else .145, hspace=.23, wspace=.075)
    for ri, (name, _, title) in enumerate(ORDER):
        prediction, item = predictions[name], measured[name]
        score = 'undefined' if item['dice'] is None else f"{item['dice']:.4f}"
        hd = 'undefined' if item['hd95_mm'] is None else f"{item['hd95_mm']:.2f} mm"
        label = (f"{title}\nWhole-volume Dice {score}\nTP {item['tp']}  FP {item['fp']}\n"
                 f"FN {item['fn']}  HD95 {hd}\nOut-of-extent FP {item['outside_gt_depth_false_positive_voxels']}")
        for ci, selection in enumerate(slices):
            z, axis = selection['z'], axes[ri, ci]
            im = image[z, y0:y1, x0:x1]
            gt = target[z, y0:y1, x0:x1] == 5
            pr = prediction[z, y0:y1, x0:x1] == 5
            aspect = float(spacing[1]/spacing[2])
            axis.imshow(im, cmap='gray', vmin=vmin, vmax=vmax, origin='lower',
                        aspect=aspect, interpolation='nearest')
            overlay = np.zeros((*im.shape, 4), dtype=np.float32)
            overlay[pr & ~gt] = [.98, .15, .13, .35]
            overlay[~pr & gt] = [.18, .40, 1., .35]
            axis.imshow(overlay, origin='lower', aspect=aspect, interpolation='nearest')
            _contour(axis, gt, '#15ef6e', 1.7, '-')
            _contour(axis, pr, '#ffe134', 1.0, '--')
            axis.set_xlim(-.5, im.shape[1]-.5)
            axis.set_ylim(-.5, im.shape[0]-.5)
            axis.set_xticks([]); axis.set_yticks([])
            if ri == 0:
                axis.set_title(' / '.join(selection['roles']) + f'\nz={z}', fontsize=10)
            if ci == 0:
                axis.set_ylabel(label, rotation=0, ha='right', va='center', fontsize=9, labelpad=14)
    view = 'full field of view' if full else f'display ROI H[{y0}:{y1}] W[{x0}:{x1}]'
    prefix = 'SYNTHETIC FIXTURE - ' if synthetic else ''
    fig.suptitle(f'{prefix}{patient} / {visit} | Previously inspected TEST\nProstate on common original grid | {view}',
                 fontsize=13.5, y=.98)
    fig.legend(handles=[Line2D([0], [0], color='#15ef6e', lw=1.7, label='GT contour'),
                        Line2D([0], [0], color='#ffe134', lw=1.0, linestyle='--', label='Prediction contour'),
                        Patch(color='#fa2621', alpha=.5, label='False positive'),
                        Patch(color='#2e66ff', alpha=.5, label='False negative')],
               loc='lower center', bbox_to_anchor=(.59, .125 if thin else .088), ncol=4, frameon=False, fontsize=9)
    note = ('ROI = full-depth union of GT and all three prostate predictions +8 voxel padding; display only, never inference/metrics.'
            if not full else 'Full field displayed for all three models. Scores and FP/FN use the entire volume, including planes not shown.')
    notes = [note, 'Label-extent slices are proxies, not independently annotated anatomical apex/base; duplicate slice roles are merged.',
             'No organ-wall contact distance or new independent test is claimed. Canonical axial RAS; physical H/W aspect.']
    notes = '\n'.join(textwrap.fill(line, width=76) for line in notes) if thin else '\n'.join(notes)
    fig.text(.59, .047 if thin else .035, notes,
             ha='center', va='center', fontsize=7.8)
    fig.savefig(destination, dpi=145, facecolor='white')
    plt.close(fig)
    return {'intensity_window': [float(vmin), float(vmax)],
            'intensity_window_rule': 'Per-case nonzero MRI1st/99th percentiles, identical across all models/views.',
            'display_roi_hw': roi, 'physical_in_plane_aspect': float(spacing[1]/spacing[2])}


def build_gallery(args):
    output = Path(args.output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError('Use a fresh gallery directory; earlier failures and figures are preserved.')
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    project, artifacts = Path(args.v1_project), Path(args.artifacts_root)
    report_path = Path(args.report)
    report = common.read_json(report_path)
    require(report.get('status') == 'completed' and report.get('split') == 'test',
            'Gallery requires the completed retrospective TEST comparison.')
    require('previously inspected test' in report.get('scientific_status', '').lower(),
            'Gallery requires explicit previously inspected TEST status.')
    require(set(report.get('models', {})) == {name for name, _, _ in ORDER},
            'Do not insert a failed or untrained new2D arm into the gallery.')
    source_files = {'comparison': {'path': str(report_path), 'sha256': common.file_sha256(report_path)}}
    scope_path = artifacts.resolve().parent/'hard_followup_protocol.json'
    require(common.file_sha256(scope_path) == report['scope_plan_sha256']
            == report['input_files']['scope_plan']['sha256'], 'Gallery scope plan differs from comparison.')
    source_files['scope_plan'] = {'path': str(scope_path), 'sha256': common.file_sha256(scope_path)}
    paths = {'v1': project/'artifacts/cache64/manifest.json', 'new': artifacts/'cache_v2/manifest.json'}
    manifests, records = {}, {}
    for version, path in paths.items():
        name = 'v1_manifest' if version == 'v1' else 'new_manifest'
        digest = common.file_sha256(path)
        require(digest == report['input_files'][name]['sha256'], 'Gallery manifest differs from comparison.')
        manifests[version] = common.read_json(path)
        require(manifests[version]['label_ids'] == list(range(6)), 'Gallery label semantics changed.')
        records[version] = common.index_records(manifests[version])
        source_files[name] = {'path': str(path), 'sha256': digest}
    require(manifests['v1']['patient_splits'] == manifests['new']['patient_splits'], 'Gallery patient split changed.')
    keys = sorted(k for k, r in records['v1'].items() if r['split'] == 'test')
    require(len(keys) == 30 and len({p for p, _ in keys}) == 6,
            'Gallery requires all thirty known TEST cases and six patients.')
    require(keys == sorted(k for k, r in records['new'].items() if r['split'] == 'test'), 'Gallery TEST cases differ.')
    identity_path = paths['v1'].parent/'cache_identity.json'
    identity_digest = common.file_sha256(identity_path)
    require(identity_digest == common.FROZEN_V1_CACHE_IDENTITY_SHA256
            and identity_digest == report['input_files']['v1_cache_identity']['sha256'], 'Original image/GT identity book changed.')
    identity = common.read_json(identity_path)
    require(identity['manifest_sha256'] == common.file_sha256(paths['v1']), 'Identity book and manifest differ.')
    source_files['v1_cache_identity'] = {'path': str(identity_path), 'sha256': identity_digest}
    selection_path = artifacts/'hard_selection.json'
    require(common.file_sha256(selection_path) == report['input_files']['selection']['sha256'], 'Hard selection freeze changed.')
    selection = common.read_json(selection_path)
    require(selection.get('status') == 'frozen' and selection.get('scope_plan_sha256') == report['scope_plan_sha256']
            and set(selection.get('models', {})) == {'improved3d'}, 'Gallery selection is not the one fixed Hard-only candidate.')
    source_files['selection'] = {'path': str(selection_path), 'sha256': common.file_sha256(selection_path)}
    common_cases, evaluations, eval_cases = {}, {}, {}
    for name, family, _ in ORDER:
        version = 'new' if name == 'hard_improved3d' else 'v1'
        rows = report['models'][name]['cases']
        common_cases[name] = {key(r): r for r in rows}
        require(len(common_cases[name]) == len(rows) and set(common_cases[name]) == set(keys),
                'All three common-grid evidence populations must be complete.')
        path = (artifacts/'hard_test/improved3d/evaluation.json' if version == 'new'
                else project/f'artifacts/test_final/{family}/evaluation.json')
        require(common.file_sha256(path) == report['input_files']['evaluations'][name]['sha256'],
                'Gallery evaluation differs from the scored comparison.')
        ev, indexed = common.validate_evaluation(path, paths[version], manifests[version], 'test', keys, family)
        require(ev['checkpoint_sha256'] == report['models'][name]['checkpoint_sha256'], 'Gallery checkpoint identity changed.')
        expected = selection['models']['improved3d']['checkpoint_sha256'] if version == 'new' else common.FROZEN_V1_CHECKPOINTS[family]
        require(ev['checkpoint_sha256'] == expected, 'Gallery selected weight differs from its fixed reference.')
        evaluations[name], eval_cases[name] = path, indexed
        source_files[name+'_evaluation'] = {'path': str(path), 'sha256': common.file_sha256(path)}
    chosen = select_keys(list(common_cases['hard_improved3d'].values()))
    require(len(chosen) == 5 and len({p for p, _ in chosen}) == 5, 'Gallery must show five distinct worst-case patients.')
    output.mkdir(parents=True, exist_ok=False)
    receipt = dict(status='running', purpose='Descriptive illustrations, no inference, tuning or candidate selection',
        scope_plan_sha256=report['scope_plan_sha256'], synthetic_fixture=bool(report.get('synthetic_fixture', False)),
        selection_algorithm=SELECTION, selected_case_keys=[list(k) for k in chosen], all_test_cases_retained_in_report=len(keys),
        model_order=[name for name, _, _ in ORDER], score_grid='original unchanged downsampled grid',
        native_resolution_accuracy_claimed=False, source_files=source_files,
        script_sha256=common.file_sha256(__file__), mapping_source_sha256=common.file_sha256(common.__file__),
        display_helpers_source_sha256=common.file_sha256(display_helpers.__file__),
        context_class_index_names={'0':'Background','1':'Body','2':'Bones','3':'Bladder','4':'Rectum','5':'Prostate'},
        error_context_limit='Label-class confusion, not measured organ-wall contact distance, anatomical apex/base annotation or causal mechanism.',
        recorded_at_utc=datetime.now(timezone.utc).isoformat(), cases=[])
    def save():
        temporary = output/'gallery_receipt.pending.json'
        temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
        temporary.replace(output/'gallery_receipt.json')
    save()
    try:
        for patient, visit in chosen:
            selected = patient, visit
            old = records['v1'][selected]
            cache = common.resolve_relative(paths['v1'].parent, old['cache_path'])
            with np.load(cache, allow_pickle=False) as stored:
                image, target, spacing = (stored[name].copy() for name in ('image','mask','spacing'))
            require(image.shape == target.shape == tuple(manifests['v1']['target_shape_dhw'])
                    and np.isfinite(image).all(), 'Gallery image/GT grid is invalid.')
            expected_arrays = identity['records'][old['cache_path']]['arrays']
            for name, array in [('image',image),('mask',target),('spacing',spacing)]:
                require(common.decoded_array_sha256(array) == expected_arrays[name]['sha256'], 'Gallery cache pixels changed: '+name)
            predictions, measured, evidence = {}, {}, {}
            for name, _, _ in ORDER:
                version = 'new' if name == 'hard_improved3d' else 'v1'
                record = records[version][selected]
                prediction, item = common.load_prediction(evaluations[name], eval_cases[name][selected],
                    manifests[version]['target_shape_dhw'], record, require_affine=version == 'new')
                require(item['sha256'] == common_cases[name][selected]['prediction_evidence']['sha256'],
                        'Gallery prediction bytes differ from the scored prediction.')
                if version == 'new':
                    prediction, _ = common.categorical_to_grid(prediction, common.affine(record), common.affine(old), target.shape)
                predictions[name] = prediction
                measured[name] = verify_case(prediction, target, spacing, common_cases[name][selected])
                evidence[name] = item
            slices = display_slices(target)
            roi = display_roi(target, list(predictions.values()), padding=8)
            filename = f'{patient}_{visit}'
            require(all(c.isalnum() or c in '_-' for c in filename), 'Unsafe gallery case identity.')
            figures = {}
            for view, box in [('full_fov',[0,image.shape[1],0,image.shape[2]]),('display_roi',roi)]:
                destination = output/f'{filename}_{view}.png'
                detail = render_case(image,target,predictions,measured,spacing,slices,box,patient,visit,destination,
                                     synthetic=receipt['synthetic_fixture'])
                figures[view] = {'file':destination.name,'sha256':common.file_sha256(destination),**detail}
            receipt['cases'].append(dict(patient=patient,visit=visit,display_slices=slices,display_roi_hw=roi,
                duplicated_slice_roles_collapsed=any(len(x['roles'])>1 for x in slices),
                roi_rule='Full-depth GT+three-prediction prostate union projected to H/W, +8voxel padding; visualization only.',
                complete_volume_metrics=measured,cache_file_sha256=common.file_sha256(cache),
                prediction_sources=evidence,figures=figures))
            save()
        receipt.update(status='completed',elapsed_wall_seconds=time.perf_counter()-start_wall,
            elapsed_cpu_seconds=time.process_time()-start_cpu,raw_arrays_saved=False,model_or_threshold_changed=False)
        save()
    except BaseException as error:
        receipt.update(status='failed',error=f'{type(error).__name__}: {error}')
        save()
        raise
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v1-project', type=Path, required=True)
    parser.add_argument('--artifacts-root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_gallery(args)
    print(json.dumps({'status':result['status'],'selected_case_keys':result['selected_case_keys'],
                      'output':str(args.output_dir),'figures':len(result['cases'])*2},indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
