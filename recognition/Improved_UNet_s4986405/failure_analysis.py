"""Render reproducible exploratory validation examples with numeric-label TP/FP/FN overlays."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import sys

from compare_results import inspect_run, metric_rows, require
from run_comparison import check_sources, write_json

RULES = (
    (38, 3, 64, 'dice_regression', 'Largest baseline-minus-Improved Dice; GT >=64 pixels'),
    (37, 4, 32, 'fp_increase', 'Largest Improved-minus-baseline FP count; GT >=32 pixels'),
    (38, 5, 16, 'lowest_improved_dice', 'Lowest Improved Dice; GT >=16 pixels'),
    (39, 5, 32, 'lowest_improved_dice', 'Lowest Improved Dice; GT >=32 pixels'),
    (36, 5, 32, 'dice_gain', 'Largest Improved-minus-baseline Dice; GT >=32 pixels; improvement example'),
)
CASE_PATTERN = re.compile(r'case_(\d+)_week_(\d+)_slice_(\d+)\.nii\.gz')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def select_examples(baseline, improved):
    """Post-hoc, fixed diagnostic rules; deterministic ties use the dataset index."""
    for report in (baseline, improved):
        require(report['split'] == 'validate' and report['smoke_only'] is False,
                'Only full validation records can select diagnostic examples')
        require(len(report['slices']) == report['processed_samples'] == report['split_samples'],
                'Incomplete slice records')
    require(len(baseline['slices']) == len(improved['slices']), 'Slice counts differ')
    left = {entry['index']: entry for entry in baseline['slices']}
    right = {entry['index']: entry for entry in improved['slices']}
    require(len(left) == len(baseline['slices']) and set(left) == set(right), 'Duplicate or mismatched indices')
    aligned = []
    for index in sorted(left):
        a, b = left[index], right[index]
        require(type(index) is int and index >= 0, 'Invalid slice index')
        for field in ('image', 'mask', 'patient_id', 'week', 'slice_id', 'shape', 'pixels'):
            require(a[field] == b[field], 'Slice identity/shape mismatch')
        match = CASE_PATTERN.fullmatch(a['image'])
        require(match is not None and tuple(map(int, match.groups())) ==
                (a['patient_id'], a['week'], a['slice_id']), 'Invalid filename identity')
        require(a['mask'] == 'seg_' + a['image'][5:], 'Invalid paired mask name')
        require(len(a['shape']) == 2 and all(type(x) is int and x > 0 for x in a['shape']) and
                a['pixels'] == a['shape'][0] * a['shape'][1], 'Invalid recorded native shape')
        a_metrics, b_metrics = metric_rows(a), metric_rows(b)
        require([x['target_pixels'] for x in a_metrics] == [x['target_pixels'] for x in b_metrics],
                'Ground-truth counts differ between reports')
        aligned.append((a, b, a_metrics, b_metrics))
    selected = []
    for patient, label, minimum, mode, description in RULES:
        candidates = []
        for a, b, a_metrics, b_metrics in aligned:
            lmetric, rmetric = a_metrics[label], b_metrics[label]
            if a['patient_id'] != patient or lmetric['target_pixels'] < minimum:
                continue
            require(lmetric['dice'] is not None and rmetric['dice'] is not None, 'Undefined eligible Dice')
            score = {'dice_regression': lmetric['dice'] - rmetric['dice'],
                     'fp_increase': rmetric['fp'] - lmetric['fp'],
                     'lowest_improved_dice': -rmetric['dice'],
                     'dice_gain': rmetric['dice'] - lmetric['dice']}[mode]
            candidates.append((score, -a['index'], a, b, lmetric, rmetric))
        require(bool(candidates), f'No eligible validation slice for patient {patient}, label {label}')
        _, _, a, b, lmetric, rmetric = max(candidates, key=lambda entry: entry[:2])
        selected.append({
            'example_id': f'{len(selected)+1:02d}_patient{patient:03d}_label{label}',
            'index': a['index'], 'image': a['image'], 'mask': a['mask'],
            'patient_id': patient, 'week': a['week'], 'slice_id': a['slice_id'],
            'shape': a['shape'], 'label_id': label, 'minimum_target_pixels': minimum,
            'selection_rule': mode, 'selection_description': description,
            'eligible_candidates': len(candidates), 'tie_break': 'Lowest dataset index',
            'reference_metrics': {'UNet2D': lmetric, 'ImprovedUNet2D': rmetric},
            'reference_confusion': {'UNet2D': a['confusion'], 'ImprovedUNet2D': b['confusion']},
        })
    require(len({entry['index'] for entry in selected}) == len(selected), 'Selected duplicate slice indices')
    return selected


def crop_bounds(shape, coordinates, margin=8):
    require(len(shape) == 2 and all(type(x) is int and x > 0 for x in shape), 'Invalid crop shape')
    require(type(margin) is int and margin >= 0, 'Invalid crop margin')
    if not coordinates:
        return [0, shape[0], 0, shape[1]]
    require(all(len(point) == 2 and all(type(v) is int for v in point) and
                0 <= point[0] < shape[0] and 0 <= point[1] < shape[1] for point in coordinates),
            'Crop coordinates outside native grid')
    ys, xs = zip(*coordinates)
    return [max(0, min(ys)-margin), min(shape[0], max(ys)+margin+1),
            max(0, min(xs)-margin), min(shape[1], max(xs)+margin+1)]


def render_figure(image, target, predictions, codes, record, destination):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch, Rectangle
    import numpy as np

    mri = image[0].numpy()
    low, high = np.percentile(mri, [1, 99])
    if high <= low:
        low, high = float(mri.min())-1, float(mri.max())+1
    colors = plt.get_cmap('tab10').colors[:6]
    segmentation_colors = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 6.5, 1), 6)
    overlap_colors = [(0, 0, 0, 0), (0.1, 0.8, 0.2, 0.65),
                      (0.95, 0.12, 0.1, 0.7), (0.1, 0.4, 1.0, 0.7)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 9), layout='constrained')
    fig.suptitle(f'Exploratory validation | {record["image"]} | numeric ID {record["label_id"]}\n'
                 f'{record["selection_description"]} | {record["execution_device"]} re-inference', fontsize=11)
    y0, y1, x0, x1 = record['crop_box_yx_exclusive']
    for axis in axes.flat:
        axis.imshow(mri, cmap='gray', vmin=low, vmax=high, origin='lower', interpolation='nearest')
        axis.set_xlim(x0-0.5, x1-0.5)
        axis.set_ylim(y0-0.5, y1-0.5)
        axis.set_axis_off()
    axes[0, 0].set_xlim(-0.5, mri.shape[1]-0.5)
    axes[0, 0].set_ylim(-0.5, mri.shape[0]-0.5)
    axes[0, 0].add_patch(Rectangle((x0-0.5, y0-0.5), x1-x0, y1-y0,
                                  edgecolor='yellow', facecolor='none', linewidth=1.2))
    axes[0, 0].set_title('Full native MRI; yellow box = detail views', fontsize=9)
    gt = target.eq(record['label_id']).numpy().astype(int)
    axes[1, 0].imshow(gt, cmap=ListedColormap([(0, 0, 0, 0), (1, 0.9, 0.05, 0.65)]),
                      vmin=-0.5, vmax=1.5, origin='lower', interpolation='nearest')
    axes[1, 0].set_title(f'Ground truth: numeric ID {record["label_id"]} (yellow)', fontsize=9)
    for column, architecture in enumerate(('UNet2D', 'ImprovedUNet2D'), start=1):
        plotted = axes[0, column].imshow(predictions[architecture].numpy(), cmap=segmentation_colors,
                                        norm=norm, origin='lower', interpolation='nearest')
        row = record['rendered_metrics'][architecture]
        axes[0, column].set_title(f'{architecture} | all predicted IDs\n'
                                  f'Selected ID Dice={row["dice"]:.4f}, IoU={row["iou"]:.4f}', fontsize=9)
        axes[1, column].imshow(codes[architecture].numpy(), cmap=ListedColormap(overlap_colors),
                              vmin=-0.5, vmax=3.5, origin='lower', interpolation='nearest')
        axes[1, column].set_title(f'TP={row["tp"]}, FP={row["fp"]}, FN={row["fn"]}', fontsize=9)
    # imshow resets limits, so restore the same native-coordinate crop on all detail views.
    for axis in (axes[0, 1], axes[0, 2], *axes[1]):
        axis.set_xlim(x0-0.5, x1-0.5)
        axis.set_ylim(y0-0.5, y1-0.5)
    fig.colorbar(plotted, ax=[axes[0, 1], axes[0, 2]], ticks=range(6), shrink=0.65,
                 label='Numeric label ID', fraction=0.035, pad=0.02)
    fig.legend([Patch(facecolor=color) for color in overlap_colors[1:]], ['TP', 'FP', 'FN'],
               loc='outside lower center', ncol=3, frameon=False)
    try:
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=160, facecolor='white',
                    metadata={'Description': record['image'] + ' | numeric label ' + str(record['label_id'])})
        with Path(destination).open('xb') as handle:
            handle.write(buffer.getvalue())
    finally:
        plt.close(fig)


def export_examples(models, dataset, selection, destination, device, metadata):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f'Analysis output already exists: {destination}')
    import torch
    from evaluate import summarize
    from metrics import confusion_matrix
    from predict import infer_mask, overlap_map
    destination.mkdir(parents=True, exist_ok=False)
    report = {**metadata, 'status': 'rendering', 'execution_device': str(device), 'examples': []}
    write_json(destination/'analysis_manifest.json', report)
    for selected in selection:
        image, target = dataset[selected['index']]
        require(list(target.shape) == selected['shape'], 'Selected native shape changed')
        predictions, codes, rendered, changed = {}, {}, {}, {}
        record = {**selected, 'execution_device': str(device)}
        support = target.eq(selected['label_id'])
        for architecture, model in models.items():
            prediction = infer_mask(model, image, device)
            matrix = confusion_matrix(prediction, target)
            scores = summarize(matrix)
            row = scores['classes'][selected['label_id']]
            require(row['target_pixels'] == selected['reference_metrics'][architecture]['target_pixels'],
                    'Selected ground-truth label count changed')
            require([sum(line) for line in scores['confusion']] ==
                    [sum(line) for line in selected['reference_confusion'][architecture]],
                    'Selected ground-truth counts changed')
            overlap = overlap_map(prediction, target, selected['label_id'])
            for number, key in ((1, 'tp'), (2, 'fp'), (3, 'fn')):
                require(int((overlap == number).sum()) == row[key], 'Overlay and metric counts differ')
            predictions[architecture], codes[architecture], rendered[architecture] = prediction, overlap, row
            changed[architecture] = scores['confusion'] != selected['reference_confusion'][architecture]
            support |= prediction.eq(selected['label_id'])
        record['rendered_metrics'] = rendered
        record['confusion_changed_from_cuda_reference'] = changed
        record['crop_box_yx_exclusive'] = crop_bounds(selected['shape'], support.nonzero().tolist())
        name = selected['example_id'] + '.png'
        render_figure(image, target, predictions, codes, record, destination/name)
        record['figure'] = name
        record['figure_sha256'] = sha256(destination/name)
        report['examples'].append(record)
        write_json(destination/'analysis_manifest.json', report)
        print('Rendered:', selected['example_id'], flush=True)
    report['status'] = 'rendered_pending_integrity_check'
    write_json(destination/'analysis_manifest.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-run', type=Path, required=True)
    parser.add_argument('--improved-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args(argv)
    destination = args.output_dir.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f'Analysis output already exists: {destination}')
    baseline = inspect_run(args.baseline_run, 'UNet2D')
    improved = inspect_run(args.improved_run, 'ImprovedUNet2D')
    for key in ('source_commit', 'source_sha256', 'filename_size_manifest_sha256'):
        require(baseline['provenance'][key] == improved['provenance'][key], 'Matched provenance required')
    require(baseline['protocol'] == improved['protocol'] and baseline['data'] == improved['data'],
            'Matched protocol and input manifests required')
    selection = select_examples(baseline['reports'][0], improved['reports'][0])
    runtime = Path(baseline['provenance']['source_directory']).resolve()
    require(runtime == Path(improved['provenance']['source_directory']).resolve(), 'Runtime snapshots differ')
    check_sources(runtime, baseline['provenance']['source_sha256'])
    sys.path.insert(0, str(runtime))
    import torch
    import nibabel as nib
    import numpy as np
    import dataset as dataset_module
    import predict as predict_module
    require(Path(dataset_module.__file__).resolve().parent == runtime and
            Path(predict_module.__file__).resolve().parent == runtime, 'Expected frozen runtime imports')
    torch.set_num_threads(1)
    data_root = Path(baseline['protocol']['data']['root'])
    images, masks = data_root/'keras_slices_validate', data_root/'keras_slices_seg_validate'
    normalization = baseline['config']['preprocessing']['normalize']
    require(normalization == improved['config']['preprocessing']['normalize'], 'Normalization differs')
    dataset = dataset_module.HipMRI2DDataset(images, masks, normalize=normalization)
    manifest = baseline['data']['manifest']['validation']
    require(len(dataset) == len(manifest), 'Validation dataset length changed')
    for (image, mask), expected in zip(dataset.pairs, manifest):
        require(image.name == expected['image'] and mask.name == expected['mask'], 'Validation filenames changed')
        require(image.stat().st_size == expected['image_bytes'] and mask.stat().st_size == expected['mask_bytes'],
                'Validation file byte size changed')
    for record in selection:
        image, mask = dataset.pairs[record['index']]
        require(image.name == record['image'] and mask.name == record['mask'], 'Selected file/index mismatch')
        require(np.allclose(nib.load(str(image)).affine, nib.load(str(mask)).affine, rtol=0, atol=1e-5),
                'Selected MRI/mask affines differ')
        record['input_sha256'] = {'image': sha256(image), 'mask': sha256(mask)}
    models, checkpoint_hashes, checkpoint_paths = {}, {}, {}
    for run in (baseline, improved):
        architecture = run['result']['architecture']
        checkpoint = Path(run['directory'])/'training/best.pt'
        checkpoint_hashes[architecture] = sha256(checkpoint)
        require(checkpoint_hashes[architecture] == run['validation']['checkpoint_sha256'], 'Best weights hash mismatch')
        model, saved = predict_module.load_predictor(checkpoint, args.device)
        require(saved['architecture'] == architecture and saved['epoch'] == run['validation']['best_epoch'],
                'Loaded checkpoint metadata mismatch')
        require(saved['config']['preprocessing']['normalize'] == normalization, 'Loaded normalization differs')
        models[architecture], checkpoint_paths[architecture] = model, checkpoint
    metadata = {'schema_version': 1, 'scope': 'Exploratory selected validation examples; not test results or a representative sample',
                'label_semantics': 'Numeric IDs 0-5; anatomical mapping unverified',
                'source_commit': baseline['provenance']['source_commit'], 'runtime_directory': str(runtime),
                'checkpoint_sha256': checkpoint_hashes,
                'checkpoint_epochs': {run['result']['architecture']: run['validation']['best_epoch'] for run in (baseline, improved)},
                'analysis_source_sha256': sha256(__file__),
                'normalization': 'per-slice all-pixel z-score' if normalization else 'none',
                'display': 'MRI 1st-99th percentile window for display only; native array orientation; no anatomical plane assignment. Yellow full-slice rectangle marks common detail-view crop.',
                'selection_scope': 'Four error examples and one improvement example chosen post hoc from validation pass 1 using declared thresholds/rules, ties by dataset index.',
                'numerical_scope': 'Figures/counts use fresh inference on the requested device. Small CPU/CUDA differences from the recorded reference are retained explicitly; this is not a latency benchmark.'}
    report = export_examples(models, dataset, selection, destination, args.device, metadata)
    try:
        for architecture, checkpoint in checkpoint_paths.items():
            require(sha256(checkpoint) == checkpoint_hashes[architecture], 'Weights changed during analysis')
        check_sources(runtime, baseline['provenance']['source_sha256'])
        for record in selection:
            image, mask = dataset.pairs[record['index']]
            require(sha256(image) == record['input_sha256']['image'] and
                    sha256(mask) == record['input_sha256']['mask'], 'Selected inputs changed during analysis')
    except Exception:
        report['status'] = 'failed_final_integrity_check'
        write_json(destination/'analysis_manifest.json', report)
        raise
    report['status'] = 'completed'
    write_json(destination/'analysis_manifest.json', report)
    print('Output directory:', destination)
    print('Exploratory validation overlays: PASS')
    return report


if __name__ == '__main__':
    main()
