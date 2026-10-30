"""Fixed, paired Validation-only brightness sensitivity experiment (not training).

Requires the existing diagnose_development.py, dataset.py, modules.py and
predict.py. Compatible with Python 3.8+/torch117. No test image inference.
"""
import argparse
import contextlib
import io
import json
import os
import platform
import random
import secrets
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from diagnose_development import (DiagnosticError, atomic_json, fail, inspect_samples,
    interval, metrics, sha, token, write_csv)

FACTORS = (1.0, 0.9, 1.1)
SEED = 42
THRESHOLD = 0.67


def sample_patients(rows):
    """Four quarters per patient; cycle shuffled scans to maximize scan coverage."""
    rng = random.Random(SEED)
    patients = defaultdict(lambda: defaultdict(list))
    for row in rows:
        patients[row['patient']][row['scan']].append(row)
    selected = []
    for patient in sorted(patients):
        scans = sorted(patients[patient])
        rng.shuffle(scans)
        quarters = list(range(4))
        rng.shuffle(quarters)
        used = set()
        for slot, quarter in enumerate(quarters):
            preferred = scans[slot % len(scans)]
            ordered = [preferred] + [s for s in scans if s != preferred]
            candidates = []
            for scan in ordered:
                candidates = sorted([r for r in patients[patient][scan]
                    if r['position'] // 5 == quarter and (scan, r['position']) not in used],
                    key=lambda r: r['position'])
                if candidates:
                    break
            if candidates:
                row = candidates[rng.randrange(len(candidates))]
                selected.append(row)
                used.add((row['scan'], row['position']))
    return selected


def inputs_for_image(path, transform):
    """Original branch preserves RGB loader semantics; other branches use 8-bit gray.

    MRI RGB channels must be identical, otherwise gray-only interventions would
    introduce an additional color-conversion difference. Never save modified MRI.
    """
    try:
        with Image.open(path) as source:
            if source.mode not in ('L', 'RGB'):
                fail('Selected image is not 8-bit grayscale/RGB.')
            original = source.convert('RGB')
            rgb = np.asarray(original)
            if not (np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) and
                    np.array_equal(rgb[:, :, 0], rgb[:, :, 2])):
                fail('Selected RGB image is not replicated grayscale; manual review required.')
            gray = np.asarray(original.convert('L'), dtype=np.uint8)
    except DiagnosticError:
        raise
    except Exception:
        fail('Selected validation image could not be decoded (path withheld).')
    mask = gray > 5
    if not mask.any():
        fail('Selected image has empty foreground under fixed >5 mask.')
    tensors, audit = [], []
    for factor in FACTORS:
        scaled = gray.astype(np.float64)
        scaled[mask] *= factor
        clipped = mask & (scaled > 255)
        adjusted = gray.copy()
        adjusted[mask] = np.clip(np.rint(scaled[mask]), 0, 255).astype(np.uint8)
        image = original if factor == 1.0 else Image.fromarray(adjusted, mode='L')
        tensors.append(transform(image))
        audit.append(dict(factor=factor, foreground_fraction=float(mask.mean()),
            clipped_fraction_all=float(clipped.mean()),
            clipped_fraction_foreground=float(clipped.sum() / mask.sum()),
            saturated_fraction_all=float((adjusted == 255).mean()),
            original_foreground_mean=float(gray[mask].mean()),
            resulting_foreground_mean=float(adjusted[mask].mean())))
    return tensors, audit


def prepare(args, key):
    import torch
    import torchvision
    import dataset
    from modules import ConvNeXt
    from predict import get_sample_info, fingerprint_split
    from torchvision import transforms
    if torchvision.get_image_backend() != 'PIL':
        fail('PIL image backend required to preserve historical loading semantics.')
    if not args.checkpoint.is_file() or not args.config.is_file():
        fail('Existing checkpoint/config not found.')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if config.get('model') != 'convnext' or config.get('smoke_test') or config.get('class_to_idx') != {'AD': 0, 'NC': 1}:
        fail('Historical config mismatch.')
    # Existing loader enumerates test identities ONLY for leakage exclusion.
    with contextlib.redirect_stdout(io.StringIO()):
        train, validation, test = dataset.load_datasets()
    fingerprints = {name: fingerprint_split(ds) for name, ds in [('train', train), ('validation', validation)]}
    for name, value in fingerprints.items():
        if value != config.get('split_fingerprint', {}).get(name):
            fail('Historical development fingerprint mismatch: ' + name)
    del train, test
    rows, stats = inspect_samples(get_sample_info(validation), dataset.load_patient_map(),
        dataset.get_image_id, decode=False, audit_prefix=args.output_dir / 'validation_index_audit', key=key)
    if stats['patients'] != 70 or len(rows) != config.get('val_images'):
        fail('Expected historical 70-patient validation split.')
    original = transforms.Compose([transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
        transforms.Normalize([.5] * 3, [.5] * 3)])
    if repr(original) != repr(dataset.transform):
        fail('Original preprocessing differs from dataset.transform.')
    selected = sample_patients(rows)
    selected_digest = []
    coverage = []
    for row in selected:
        # Check all selected inputs and perturbations, without any model forward.
        tensors, _ = inputs_for_image(row['path'], dataset.transform)
        if any(tuple(t.shape) != (3, 224, 224) or not torch.isfinite(t).all() for t in tensors):
            fail('Invalid preprocessed selected image.')
        row['patient'] = token(key, 'patient', row['patient'])
        row['scan'] = token(key, 'scan', row['scan'])
        selected_digest.append(dict(patient=row['patient'], scan=row['scan'],
            raw_slice_index=row['raw_slice_index'], position=row['position'],
            label=row['label'], content_sha256=sha(row['path'])))
    grouped = defaultdict(list)
    for row in selected:
        grouped[row['patient']].append(row)
    for entries in grouped.values():
        coverage.append(dict(class_name='AD' if entries[0]['label'] == 0 else 'NC',
            selected_slices=len(entries), selected_scans=len({r['scan'] for r in entries}),
            quarters=len({r['position'] // 5 for r in entries})))
    model = ConvNeXt(num_classes=2, in_channels=3)
    try:
        model.load_state_dict(torch.load(str(args.checkpoint), map_location='cpu', weights_only=True), strict=True)
    except Exception:
        fail('Checkpoint must be compatible raw ConvNeXt state_dict.')
    protocol = dict(version=1, seed=SEED, threshold=THRESHOLD, factors=list(FACTORS),
        foreground='original 8-bit grayscale >5; fixed across conditions',
        rounding='nearest integer, clip to [0,255], no JPEG re-encoding',
        checkpoint_sha256=sha(args.checkpoint), config_sha256=sha(args.config),
        split_fingerprints=fingerprints, preprocessing=repr(dataset.transform),
        selection=selected_digest, bootstrap=args.bootstrap, batch_size=args.batch_size,
        threads=args.threads, device=args.device,
        sources={f: sha(Path(__file__).parent / f) for f in ('test_brightness_sensitivity.py',
            'diagnose_development.py', 'dataset.py', 'modules.py', 'predict.py')},
        versions=dict(python=platform.python_version(), torch=torch.__version__,
            torchvision=torchvision.__version__, numpy=np.__version__))
    return model, selected, dataset.transform, protocol, dict(selected_slices=len(selected),
        patients=len(grouped), coverage=coverage, position_definition='within-scan rank 0-19, not anatomical coordinates')


def paired_summary(rows):
    result = {}
    for factor in FACTORS:
        rr = [dict(r, score=r['scores'][str(factor)]) for r in rows]
        deltas = [r['score'] - r['scores']['1.0'] for r in rr]
        result[str(factor)] = dict(**metrics(rr), mean_delta=float(np.mean(deltas)),
            median_delta=float(np.median(deltas)), mean_absolute_delta=float(np.mean(np.abs(deltas))),
            flip_fraction=float(np.mean([(r['score'] >= THRESHOLD) != (r['scores']['1.0'] >= THRESHOLD) for r in rr])),
            nc_to_ad_fraction=float(np.mean([r['scores']['1.0'] < THRESHOLD <= r['score'] for r in rr])),
            ad_to_nc_fraction=float(np.mean([r['score'] < THRESHOLD <= r['scores']['1.0'] for r in rr])))
        for field in ('accuracy', 'recall_ad', 'recall_nc', 'auroc_ad'):
            base = metrics([dict(r, score=r['scores']['1.0']) for r in rows])[field]
            value = result[str(factor)][field]
            result[str(factor)]['delta_' + field] = value - base if value is not None and base is not None else None
    return result


def report(rows, output, bootstrap):
    rng = np.random.default_rng(SEED)
    summary, csv_rows = {}, []
    for name, label in [('all', None), ('AD', 0), ('NC', 1)]:
        subset = [r for r in rows if label is None or r['label'] == label]
        summary[name] = paired_summary(subset)
        groups = defaultdict(list)
        for row in subset:
            groups[row['patient']].append(row)
        keys = sorted(groups)
        boot = defaultdict(list)
        for _ in range(bootstrap):
            sampled = [r for i in rng.integers(0, len(keys), len(keys)) for r in groups[keys[i]]]
            ss = paired_summary(sampled)
            for factor in FACTORS:
                for field in ('mean_delta', 'median_delta', 'mean_absolute_delta', 'flip_fraction',
                              'delta_accuracy', 'delta_recall_ad', 'delta_recall_nc', 'delta_auroc_ad'):
                    boot[(str(factor), field)].append(ss[str(factor)][field])
        for factor in FACTORS:
            m = summary[name][str(factor)]
            m['patient_cluster_ci'] = {field: interval(values, bootstrap) for (f, field), values in boot.items() if f == str(factor)}
            csv_rows.append(dict(class_name=name, factor=factor, **{k: v for k, v in m.items() if k != 'patient_cluster_ci'}))
    atomic_json(output / 'summary.json', summary)
    write_csv(output / 'condition_metrics.csv', csv_rows)
    saturation = []
    for name, label in [('all', None), ('AD', 0), ('NC', 1)]:
        selected = [r for r in rows if label is None or r['label'] == label]
        for i, factor in enumerate(FACTORS):
            audits = [r['audit'][i] for r in selected]
            saturation.append(dict(class_name=name, factor=factor, n=len(audits),
                **{field + '_mean': float(np.mean([a[field] for a in audits])) for field in
                   ('clipped_fraction_all', 'clipped_fraction_foreground', 'saturated_fraction_all')}))
    write_csv(output / 'saturation_summary.csv', saturation)
    detail = []
    for row in rows:
        for factor in FACTORS:
            score = row['scores'][str(factor)]
            detail.append(dict(anonymous_patient=row['patient'], anonymous_scan=row['scan'],
                raw_slice_index=row['raw_slice_index'], relative_position=row['position'],
                class_name='AD' if row['label'] == 0 else 'NC', factor=factor,
                probability_ad=score, delta_probability_ad=score-row['scores']['1.0'],
                flipped=(score >= THRESHOLD) != (row['scores']['1.0'] >= THRESHOLD),
                correct=(score >= THRESHOLD) == (row['label'] == 0),
                **{k: v for k, v in row['audit'][FACTORS.index(factor)].items() if k != 'factor'}))
    write_csv(output / 'local_paired_predictions.csv', detail)
    lines = ['# Fixed brightness sensitivity experiment', '',
        'Validation-only paired diagnostic; coefficients and threshold fixed before inference.', '',
        '| Class | Factor | N | Accuracy | AD recall | NC recall | AUROC | Mean AD probability change | Flip fraction |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in csv_rows:
        def fmt(v):
            return 'NA' if v is None else '{:.4f}'.format(v)
        lines.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(row['class_name'], row['factor'], row['n'],
            *[fmt(row[k]) for k in ('accuracy', 'recall_ad', 'recall_nc', 'auroc_ad', 'mean_delta', 'flip_fraction')]))
    lines += ['', '## Limits', '',
        '- This measures the fixed model response to +/-10% foreground scaling on a fixed validation subsample.',
        '- Paired bootstrap resamples entire patients, retaining the same images across conditions.',
        '- Class-only AUROC and recall for the absent class are undefined, not zero.',
        '- Image scaling can alter contrast/anatomical visibility; saturation is explicitly reported.',
        '- Zero sensitivity at these two factors does not exclude other intensity dependencies.',
        '- Validation already influenced historical model/threshold selection; this is exploratory.',
        '- Subsample metrics are not the full 2620-slice validation metrics and need not match 78.82%.',
        '- Relative positions are not verified anatomical coordinates.',
        '- No causal explanation of test accuracy, patient memorization, clinical robustness or preprocessing improvement follows.',
        '- Do not choose the best factor, update the official threshold, or claim the highest accuracy as an improved model.',
        '- local_paired_predictions.csv and the HMAC key are local-only; aggregate reports have no raw identities or paths.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'analyze'))
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoints/convnext/convnext_aug_adamw_run01/best.pth'))
    parser.add_argument('--config', type=Path, default=Path('checkpoints/convnext/convnext_aug_adamw_run01/config.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=3, help='Maximum variant batch size; at most three inputs per selected image')
    parser.add_argument('--bootstrap', type=int, default=200)
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1 or args.bootstrap < 0:
        parser.error('Invalid resource/statistic parameters.')
    output = args.output_dir
    if args.mode == 'check':
        if output.exists():
            parser.error('Output exists; select a new directory.')
        output.mkdir(parents=True)
        key = secrets.token_bytes(32)
        (output / '.local_hmac_key').write_bytes(key)
        if os.name != 'nt':
            os.chmod(str(output), 0o700)
            os.chmod(str(output / '.local_hmac_key'), 0o600)
    else:
        if not (output / 'manifest.json').is_file():
            parser.error('Run check successfully first.')
        if (output / 'complete.json').exists():
            print('Already complete; existing results preserved.')
            return
        key = (output / '.local_hmac_key').read_bytes()
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    model, rows, transform, protocol, sampling = prepare(args, key)
    if args.mode == 'check':
        atomic_json(output / 'manifest.json', dict(protocol=protocol, sampling=sampling,
            note='Check only: no model forward. Test identities enumerated only for leakage exclusion.'))
        print('Check passed: {} slices from {} patients; no model inference.'.format(sampling['selected_slices'], sampling['patients']))
        return
    previous = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    if previous['protocol'] != protocol:
        fail('Check/analyze protocol or selected content changed; use a new check directory.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        fail('CUDA requested but unavailable.')
    model = model.to(args.device).eval()
    result = []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            tensors, audit = inputs_for_image(row['path'], transform)
            scores = []
            for start in range(0, len(tensors), args.batch_size):
                logits = model(torch.stack(tensors[start:start + args.batch_size]).to(args.device))
                scores.extend(torch.softmax(logits, 1)[:, 0].cpu().tolist())
            if any(not np.isfinite(s) for s in scores):
                fail('Nonfinite model prediction.')
            result.append(dict(patient=row['patient'], scan=row['scan'], label=row['label'],
                raw_slice_index=row['raw_slice_index'], position=row['position'],
                scores={str(f): s for f, s in zip(FACTORS, scores)}, audit=audit))
            if (index + 1) % 20 == 0:
                print('Paired inference: {}/{} images, {:.1f}s elapsed'.format(index + 1, len(rows), time.monotonic() - started), flush=True)
    inference_done = time.monotonic()
    report(result, output, args.bootstrap)
    atomic_json(output / 'runtime.json', dict(preflight_and_inference_seconds=inference_done-started,
        reporting_seconds=time.monotonic()-inference_done, device=args.device, threads=args.threads,
        selected_images=len(rows), model_inputs=3 * len(rows)))
    atomic_json(output / 'complete.json', {'complete': True})
    print('Complete. Coefficients and threshold unchanged; outputs are diagnostics only.')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('Brightness diagnostic failed ({})'.format(type(error).__name__), file=sys.stderr)
        if isinstance(error, DiagnosticError):
            print(str(error), file=sys.stderr)
        sys.exit(1)
