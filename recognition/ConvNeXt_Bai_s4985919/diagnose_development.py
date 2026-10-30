"""Read-only ADNI development diagnostics. No training, test inference or tuning.

Python 3.8+/torch 2.0.1 compatible. Output directories must be new, or resumed
explicitly. Cached per-slice measurements are pseudonymous LOCAL research data.
"""
import argparse
import contextlib
import csv
import hashlib
import hmac
import io
import json
import os
import platform
import re
import secrets
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

THRESHOLD = 0.67
EXPECTED_SLICES = 20
STOP = False


class DiagnosticError(ValueError):
    """Only deliberately sanitized messages may be printed."""


def fail(message):
    raise DiagnosticError(message)


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
    os.replace(str(temp), str(path))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def token(key, kind, value):
    return hmac.new(key, (kind + ':' + str(value)).encode(), hashlib.sha256).hexdigest()[:24]


def parse_name(path):
    match = re.fullmatch(r'(.+)_([0-9]+)', Path(path).stem)
    if match is None or Path(path).suffix.lower() != '.jpeg':
        fail('Filename rule failed (source identifiers withheld).')
    return match.group(1), int(match.group(2))


def inspect_samples(samples, patient_map, old_id, decode=True, audit_prefix=None, key=None):
    scans, patients, result = defaultdict(list), {}, []
    problems = defaultdict(int)
    for path, label in samples:
        try:
            scan, position = parse_name(path)
        except DiagnosticError:
            problems['unparseable_filenames'] += 1
            continue
        if old_id(path) != scan:
            problems['scan_prefix_mismatches'] += 1
        patient = patient_map.get(scan)
        if patient is None or label not in (0, 1):
            problems['missing_mapping_or_invalid_label'] += 1
        if patient is not None:
            if patient in patients and patients[patient] != label:
                problems['mixed_patient_label_files'] += 1
            patients[patient] = label
        if decode:
            try:
                with Image.open(path) as image:
                    image.load()
                    if image.mode not in ('L', 'RGB') or min(image.size) < 1:
                        fail('Unsupported image encoding.')
            except Exception:
                problems['unreadable_or_unsupported_images'] += 1
        row = dict(path=str(path), label=label, patient=patient, scan=scan,
                   raw_slice_index=position)
        scans[scan].append(row)
        result.append(row)
    audits, index_sets, starts = [], defaultdict(int), defaultdict(int)
    for scan, entries in scans.items():
        positions = sorted(row['raw_slice_index'] for row in entries)
        unique = sorted(set(positions))
        duplicates = len(positions) - len(unique)
        continuous = unique == list(range(unique[0], unique[-1] + 1))
        missing_inside = unique[-1] - unique[0] + 1 - len(unique)
        if len(positions) != EXPECTED_SLICES:
            problems['scans_with_wrong_slice_count'] += 1
        if duplicates:
            problems['scans_with_duplicate_indices'] += 1
        if not continuous:
            problems['scans_with_internal_gaps'] += 1
        index_sets[tuple(unique)] += 1
        starts[unique[0]] += 1
        rank_map = {value: rank for rank, value in enumerate(unique)}
        for row in entries:
            row['position'] = rank_map[row['raw_slice_index']]
        audits.append(dict(anonymous_scan=token(key, 'scan', scan) if key else 'scan_{}'.format(len(audits) + 1),
            minimum=unique[0], maximum=unique[-1], count=len(positions),
            unique_count=len(unique), duplicate_count=duplicates,
            consecutive=continuous, missing_indices_inside_range=missing_inside,
            valid_20_consecutive=(len(positions) == EXPECTED_SLICES and not duplicates and continuous)))
        if len({(r['patient'], r['label']) for r in entries}) != 1:
            problems['scans_with_inconsistent_mapping_or_label'] += 1
    if not result or set(patients.values()) != {0, 1}:
        problems['split_missing_class_or_empty'] += 1
    stats = dict(input_files=len(samples), parsed_files=len(result), slices=len(result),
        scans=len(scans), patients=len(patients), relative_positions=list(range(EXPECTED_SLICES)),
        position_definition='Numeric rank within scan, 0-19; NOT anatomical coordinates.',
        distinct_index_sets=len(index_sets), start_index_counts={str(k): v for k, v in sorted(starts.items())},
        index_set_counts=[dict(indices=list(k), scans=v) for k, v in sorted(index_sets.items())],
        all_scans_valid_20_consecutive=not any(not a['valid_20_consecutive'] for a in audits),
        valid_scans_with_different_starts=bool(len(starts) > 1 and audits and all(a['valid_20_consecutive'] for a in audits)),
        problems=dict(problems),
        scans_with_non_numeric_loader_order=sum([r['raw_slice_index'] for r in e] != sorted(r['raw_slice_index'] for r in e) for e in scans.values()),
        class_counts={str(c): sum(r['label'] == c for r in result) for c in (0, 1)})
    if audit_prefix is not None:
        atomic_json(Path(str(audit_prefix) + '.json'), stats)
        write_csv(Path(str(audit_prefix) + '.csv'), audits)
    print('Slice index audit: ' + json.dumps({k: stats[k] for k in ('input_files', 'parsed_files', 'scans', 'distinct_index_sets', 'start_index_counts', 'valid_scans_with_different_starts', 'problems')}, sort_keys=True), flush=True)
    if problems:
        fail('Development integrity problems: ' + json.dumps(dict(problems), sort_keys=True))
    return result, stats


def load_development(args, key):
    import torch
    import torchvision
    import matplotlib
    import dataset
    from modules import ConvNeXt
    from predict import fingerprint_split, get_sample_info
    if not args.checkpoint.is_file() or not args.config.is_file():
        fail('Checkpoint/config file not found. No inference performed.')
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if config.get('model') != 'convnext' or config.get('class_to_idx') != {'AD': 0, 'NC': 1}:
        fail('Historical config model/class map mismatch.')
    if config.get('smoke_test'):
        fail('Smoke-test config is not the historical full-data run.')
    # Test directory is enumerated ONLY by existing code to exclude test patients.
    # Never index, decode, batch, or score its dataset.
    with contextlib.redirect_stdout(io.StringIO()):
        train, validation, unused_test = dataset.load_datasets()
    del unused_test
    patient_map = dataset.load_patient_map()
    sets, rows, stats, fingerprints = {}, {}, {}, {}
    integrity_failures = []
    for name, ds in [('train', train), ('validation', validation)]:
        fingerprints[name] = fingerprint_split(ds)
        if fingerprints[name] != config.get('split_fingerprint', {}).get(name):
            fail('Historical development split fingerprint mismatch: ' + name)
        try:
            records, stats[name] = inspect_samples(get_sample_info(ds), patient_map, dataset.get_image_id,
                audit_prefix=args.output_dir / ('slice_index_audit_' + name), key=key)
        except DiagnosticError as error:
            integrity_failures.append(name + ': ' + str(error))
            continue
        expected_count = config.get('train_images' if name == 'train' else 'val_images')
        if expected_count != len(records):
            fail('Development sample count disagrees with historical config: ' + name)
        sets[name] = ds
        rows[name] = records
    if integrity_failures:
        fail(' | '.join(integrity_failures))
    if {r['patient'] for r in rows['train']} & {r['patient'] for r in rows['validation']}:
        fail('Training/Validation patient overlap.')
    # Verify exact original transform, rather than silently allowing a modified one.
    from torchvision import transforms
    original = transforms.Compose([transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)])
    if repr(dataset.transform) != repr(original):
        fail('dataset.transform differs from historical input preprocessing.')
    model = ConvNeXt(num_classes=2, in_channels=3)
    try:
        state = torch.load(str(args.checkpoint), map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
    except Exception:
        fail('Checkpoint is not compatible raw ConvNeXt weights (details withheld).')
    del state
    identity = hashlib.sha256()
    content = hashlib.sha256()
    for name in ('train', 'validation'):
        for r in rows[name]:
            identity.update(hmac.new(key, (name + r['scan'] + r['patient'] + str(r['raw_slice_index'])).encode(), hashlib.sha256).digest())
            content.update(bytes.fromhex(sha(r['path'])))
    for name in rows:
        for r in rows[name]:
            r['patient'] = token(key, 'patient', r['patient'])
            r['scan'] = token(key, 'scan', r['scan'])
    protocol = dict(version=2, position_definition='within-scan numeric rank 0-19', threshold=THRESHOLD, bootstrap=args.bootstrap, seed=args.seed,
        checkpoint_sha256=sha(args.checkpoint), config_sha256=sha(args.config),
        split_fingerprints=fingerprints, mapping_hmac=identity.hexdigest(),
        development_content_sha256=content.hexdigest(), preprocessing=repr(dataset.transform),
        source_sha256={f: sha(Path(__file__).parent / f) for f in
                       ('diagnose_development.py', 'dataset.py', 'modules.py', 'predict.py', 'baseline.py')},
        versions=dict(python=platform.python_version(), torch=torch.__version__, torchvision=torchvision.__version__, numpy=np.__version__, matplotlib=matplotlib.__version__),
        batch_size=args.batch_size, device=args.device)
    return model, sets, rows, stats, protocol


def intensity(path):
    # Same default ImageFolder RGB conversion, followed by PIL grayscale.
    with Image.open(path) as image:
        rgb = image.convert('RGB')
        gray = np.asarray(rgb.convert('L'), dtype=np.float64)
        width, height = rgb.size
    result = dict(width=width, height=height, global_mean=float(gray.mean()),
                  saturation_fraction=float((gray == 255).mean()))
    for cutoff in (0, 5, 10):
        values = gray[gray > cutoff]
        result['foreground_fraction_' + str(cutoff)] = float(values.size / gray.size)
        for label, value in [('mean', values.mean() if values.size else None),
                             ('median', np.median(values) if values.size else None),
                             ('p10', np.percentile(values, 10) if values.size else None),
                             ('p90', np.percentile(values, 90) if values.size else None)]:
            result['foreground_' + label + '_' + str(cutoff)] = float(value) if value is not None else None
    return result


def ranks(x):
    x = np.asarray(x)
    order = np.argsort(x, kind='mergesort')
    result = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        result[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return result


def correlation(x, y, spearman=False):
    if len(x) < 3:
        return None
    x, y = np.asarray(x, float), np.asarray(y, float)
    if spearman:
        x, y = ranks(x), ranks(y)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def metrics(rows, weights=None):
    if not rows:
        return dict(n=0, accuracy=None, recall_ad=None, recall_nc=None, auroc_ad=None)
    y = np.array([r['label'] for r in rows])
    s = np.array([r['score'] for r in rows])
    w = np.ones(len(rows)) if weights is None else np.asarray(weights, float)
    positive = s >= THRESHOLD
    tp, fn, fp, tn = [float(w[mask].sum()) for mask in
        ((y == 0) & positive, (y == 0) & ~positive, (y == 1) & positive, (y == 1) & ~positive)]
    def div(a, b):
        return a / b if b else None
    # Weighted pairwise rank AUROC, including tied scores. O(n log n).
    order = np.argsort(s, kind='mergesort')
    numerator, negative_below, i = 0.0, 0.0, 0
    while i < len(rows):
        j = i + 1
        while j < len(rows) and s[order[j]] == s[order[i]]:
            j += 1
        group = order[i:j]
        pw = w[group][y[group] == 0].sum()
        nw = w[group][y[group] == 1].sum()
        numerator += pw * (negative_below + 0.5 * nw)
        negative_below += nw
        i = j
    return dict(n=len(rows), patients=len({r['patient'] for r in rows}),
        scans=len({r['scan'] for r in rows}), accuracy=div(tp + tn, w.sum()),
        recall_ad=div(tp, tp + fn), recall_nc=div(tn, tn + fp),
        auroc_ad=div(numerator, (tp + fn) * (tn + fp)),
        tp_ad=tp, fn_ad=fn, fp_ad=fp, tn_ad=tn,
        errors=fn + fp, error_rate=div(fn + fp, w.sum()))


def interval(values, total):
    valid = [v for v in values if v is not None and np.isfinite(v)]
    return dict(low=float(np.percentile(valid, 2.5)) if valid else None,
                high=float(np.percentile(valid, 97.5)) if valid else None,
                valid=len(valid), requested=total)


def metric_ci(rows, count, rng):
    groups = defaultdict(list)
    for r in rows:
        groups[r['patient']].append(r)
    keys = sorted(groups)
    values = {k: [] for k in ('accuracy', 'recall_ad', 'recall_nc', 'auroc_ad')}
    for _ in range(count):
        sample = [r for i in rng.integers(0, len(keys), len(keys)) for r in groups[keys[i]]] if keys else []
        m = metrics(sample)
        for k in values:
            values[k].append(m[k])
    return {k: interval(v, count) for k, v in values.items()}


def write_csv(path, rows):
    if not rows:
        Path(path).write_text('', encoding='utf-8')
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def associations(rows, count, rng):
    output = []
    fields = ['global_mean', 'foreground_fraction_5'] + ['foreground_mean_' + str(c) for c in (0, 5, 10)]
    for label in (None, 0, 1):
        selected = [r for r in rows if label is None or r['label'] == label]
        strata = [('all_positions', selected)] + [('position_' + str(p), [r for r in selected if r['position'] == p]) for p in sorted({r['position'] for r in selected})]
        for field in fields:
            for stratum, subset in strata:
                valid = [r for r in subset if r.get(field) is not None]
                for unit in ('slice', 'scan', 'patient', 'within_scan') if stratum == 'all_positions' else ('slice',):
                    points = []
                    if unit in ('scan', 'patient', 'within_scan'):
                        grouped = defaultdict(list)
                        for r in valid:
                            grouped[r['scan'] if unit != 'patient' else r['patient']].append(r)
                        for entries in grouped.values():
                            x = np.array([r[field] for r in entries])
                            y = np.array([r['score'] for r in entries])
                            if unit == 'within_scan':
                                points.extend((r['patient'], float(a - x.mean()), float(b - y.mean())) for r, a, b in zip(entries, x, y))
                            else:
                                points.append((entries[0]['patient'], float(x.mean()), float(y.mean())))
                    else:
                        points = [(r['patient'], r[field], r['score']) for r in valid]
                    clusters = defaultdict(list)
                    for point in points:
                        clusters[point[0]].append(point)
                    keys = sorted(clusters)
                    def corr(data, method):
                        return correlation([p[1] for p in data], [p[2] for p in data], method == 'spearman')
                    for method in ('spearman', 'pearson'):
                        boot = []
                        for _ in range(count):
                            sample = [p for i in rng.integers(0, len(keys), len(keys)) for p in clusters[keys[i]]] if keys else []
                            boot.append(corr(sample, method))
                        ci = interval(boot, count)
                        output.append(dict(class_name='all' if label is None else ('AD' if label == 0 else 'NC'),
                            stratum=stratum, field=field, unit=unit, method=method,
                            n=len(points), patients=len(keys), coefficient=corr(points, method),
                            ci_low=ci['low'], ci_high=ci['high'], valid_bootstrap=ci['valid']))
    return output


def summarize(records, output, bootstrap, seed):
    rng = np.random.default_rng(seed)
    summary, positions, concentration, repeat, assoc, influence, patient_rows, intensity_rows = {}, [], [], [], [], [], [], []
    for split in ('train', 'validation'):
        rows = [r for r in records if r['split'] == split]
        grouped = defaultdict(list)
        for r in rows:
            grouped[r['patient']].append(r)
        counts = {p: len({r['scan'] for r in entries}) for p, entries in grouped.items()}
        for patient, entries in grouped.items():
            patient_rows.append(dict(split=split, anonymous_patient=patient,
                class_name='AD' if entries[0]['label'] == 0 else 'NC',
                scans_per_patient=counts[patient], **metrics(entries)))
        weights = [1.0 / len(grouped[r['patient']]) for r in rows]
        summary[split] = dict(slice=metrics(rows), patient_equal_slice=metrics(rows, weights),
                              patient_cluster_ci=metric_ci(rows, bootstrap, rng))
        for label in (0, 1):
            for position in (None,) + tuple(sorted({r['position'] for r in rows})):
                subset = [r for r in rows if r['label'] == label and (position is None or r['position'] == position)]
                for field in ('global_mean', 'foreground_mean_0', 'foreground_mean_5', 'foreground_mean_10', 'foreground_fraction_5', 'saturation_fraction'):
                    values = [r[field] for r in subset if r[field] is not None]
                    intensity_rows.append(dict(split=split, class_name='AD' if label == 0 else 'NC',
                        position='all' if position is None else position, field=field, n=len(values),
                        mean=float(np.mean(values)) if values else None, std=float(np.std(values)) if values else None,
                        median=float(np.median(values)) if values else None))
        for p in sorted({r['position'] for r in rows}):
            subset = [r for r in rows if r['position'] == p]
            m = dict(split=split, position=p, **metrics(subset))
            for k, ci in metric_ci(subset, bootstrap, rng).items():
                m[k + '_ci_low'], m[k + '_ci_high'], m[k + '_ci_valid'] = ci['low'], ci['high'], ci['valid']
            positions.append(m)
        for label in (None, 0, 1):
            subset = [r for r in rows if label is None or r['label'] == label]
            class_name = 'all' if label is None else ('AD' if label == 0 else 'NC')
            pg = defaultdict(list)
            for r in subset:
                pg[r['patient']].append(r)
            ordered = sorted(pg.values(), key=lambda e: sum((r['score'] >= THRESHOLD) != (r['label'] == 0) for r in e), reverse=True)
            total_errors = sum((r['score'] >= THRESHOLD) != (r['label'] == 0) for r in subset)
            errors, slices = 0, 0
            for index, entries in enumerate(ordered, 1):
                errors += sum((r['score'] >= THRESHOLD) != (r['label'] == 0) for r in entries)
                slices += len(entries)
                concentration.append(dict(split=split, class_name=class_name, top_patients=index,
                    cumulative_error_fraction=errors / total_errors if total_errors else None,
                    cumulative_slice_fraction=slices / len(subset)))
            for bucket in ('1', '2', '3+'):
                rr = [r for r in subset if ('3+' if counts[r['patient']] >= 3 else str(counts[r['patient']])) == bucket]
                m = metrics(rr)
                per_patient = [metrics(entries)['error_rate'] for p, entries in pg.items() if ('3+' if counts[p] >= 3 else str(counts[p])) == bucket]
                repeat.append(dict(split=split, class_name=class_name, scans_per_patient=bucket,
                    patient_equal_error_rate=float(np.mean(per_patient)) if per_patient else None, **m))
        # Public influence distribution has no subject identifiers or linking ranks.
        remaining = [metrics([r for r in rows if r['patient'] != patient]) for patient in grouped]
        for k in ('accuracy', 'recall_ad', 'auroc_ad'):
            values = [m[k] - summary[split]['slice'][k] for m in remaining if m[k] is not None]
            influence.append(dict(split=split, metric=k, min_delta=min(values) if values else None,
                max_delta=max(values) if values else None,
                median_delta=float(np.median(values)) if values else None, valid_patients=len(values)))
        assoc.extend(dict(split=split, **r) for r in associations(rows, bootstrap, rng))
    summary['validation_minus_train'] = {k: summary['validation']['slice'][k] - summary['train']['slice'][k] for k in ('accuracy', 'recall_ad', 'recall_nc', 'auroc_ad')}
    atomic_json(output / 'summary.json', summary)
    write_csv(output / 'local_patient_metrics.csv', patient_rows)
    for filename, data in [('slice_position_metrics.csv', positions), ('error_concentration.csv', concentration),
                           ('repeat_scan_metrics.csv', repeat), ('intensity_associations.csv', assoc), ('patient_influence.csv', influence)]:
        write_csv(output / filename, data)
    write_csv(output / 'intensity_distribution.csv', intensity_rows)
    text = ['# Fixed-model development diagnostics', '', 'Threshold: 0.67. No test inference, training or threshold search.', '',
            '| Split | Slice accuracy | AD recall | NC recall | AUROC |', '|---|---:|---:|---:|---:|']
    for split in ('train', 'validation'):
        m = summary[split]['slice']
        text.append('| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(split, m['accuracy'], m['recall_ad'], m['recall_nc'], m['auroc_ad']))
    text.extend(['', '## Interpretation limits', '',
        '- Training scores measure fitting to training patients, not generalization.',
        '- Validation was already used for historical checkpoint/threshold selection.',
        '- Position is within-scan numeric rank 0-19, NOT anatomical coordinates. Original indices remain in the local cache.',
        '- Correlations are exploratory associations, not evidence of causal brightness reliance.',
        '- Within-scan centering does not remove slice-position/anatomy confounding.',
        '- Scan counts describe visits available in this dataset, not lifetime follow-up.',
        '- Bootstrap resamples whole patients, without class stratification; undefined replicates are reported.',
        '- Patient-equal slice accuracy is not patient probability-aggregation accuracy.',
        '- Foreground thresholds 0/5/10 are descriptive masks, not brain segmentation.',
        '- Raw slice cache and local_patient_metrics.csv are pseudonymous local research data; do not publish them.',
        '- Legacy weights have no cryptographic link to historical config.',
        '- No test-cohort conclusions or new operating threshold can be derived here.'])
    (output / 'report.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    plot_figures(output, positions, concentration, assoc)


def plot_figures(output, positions, concentration, assoc):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for split in ('train', 'validation'):
        pp = [r for r in positions if r['split'] == split]
        for ax, field in zip(axes, ('accuracy', 'recall_ad', 'auroc_ad')):
            ax.plot([r['position'] for r in pp], [r[field] for r in pp], marker='.', label=split)
            ax.set(xlabel='Within-scan relative position (0-19; not anatomy)', ylabel=field, ylim=(0, 1))
            ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'slice_positions.png', dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for split in ('train', 'validation'):
        cc = [r for r in concentration if r['split'] == split and r['class_name'] == 'all']
        ax.plot([r['cumulative_slice_fraction'] for r in cc], [r['cumulative_error_fraction'] for r in cc], label=split)
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set(xlabel='Cumulative slice fraction (patients sorted by errors)', ylabel='Cumulative error fraction')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'error_concentration.png', dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    aa = [r for r in assoc if r['field'] == 'foreground_mean_5' and r['method'] == 'spearman' and r['unit'] == 'patient' and r['stratum'] == 'all_positions']
    labels = [r['split'] + '/' + r['class_name'] for r in aa]
    ax.bar(labels, [r['coefficient'] if r['coefficient'] is not None else float('nan') for r in aa])
    ax.set(ylabel='Patient-mean intensity / AD score Spearman correlation', ylim=(-1, 1))
    ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    fig.savefig(output / 'intensity_correlations.png', dpi=140)
    plt.close(fig)


def stop_handler(signum, frame):
    global STOP
    STOP = True


def analyze(model, datasets, rows, args):
    import torch
    cache = args.output_dir / 'local_slice_cache.jsonl'
    records = []
    if cache.exists():
        # Truncate only an incomplete final line in this script's own cache.
        with cache.open('r+b') as handle:
            good = 0
            while True:
                line = handle.readline()
                if not line:
                    break
                if not line.endswith(b'\n'):
                    handle.truncate(good)
                    break
                records.append(json.loads(line))
                good = handle.tell()
    expected = [(name, r['patient'], r['scan'], r['position'], r['label']) for name in ('train', 'validation') for r in rows[name]]
    if len(records) > len(expected) or any((r['split'], r['patient'], r['scan'], r['position'], r['label']) != expected[i] for i, r in enumerate(records)):
        fail('Cache membership/order mismatch; refuse resume.')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        fail('CUDA requested but unavailable.')
    model = model.to(device).eval()
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop_handler)
    if hasattr(signal, 'SIGUSR1'):
        signal.signal(signal.SIGUSR1, stop_handler)
    start = time.monotonic()
    initial = len(records)
    offset = 0
    with cache.open('ab') as handle, torch.inference_mode():
        for split in ('train', 'validation'):
            ds = datasets[split]
            begin = max(0, len(records) - offset)
            for index in range(begin, len(ds), args.batch_size):
                if STOP or (args.max_seconds and time.monotonic() - start >= args.max_seconds):
                    print('Stopped safely; rerun analyze with --resume.', flush=True)
                    return False
                end = min(index + args.batch_size, len(ds))
                batch = [ds[i] for i in range(index, end)]
                if [int(b[1]) for b in batch] != [r['label'] for r in rows[split][index:end]]:
                    fail('Dataset labels/order mismatch.')
                logits = model(torch.stack([b[0] for b in batch]).to(device))
                scores = torch.softmax(logits, 1)[:, 0].cpu().tolist()
                differences = (logits[:, 0] - logits[:, 1]).cpu().tolist()
                for r, score, difference in zip(rows[split][index:end], scores, differences):
                    if not np.isfinite(score) or not np.isfinite(difference):
                        fail('Nonfinite model score.')
                    item = {k: v for k, v in r.items() if k != 'path'}
                    item.update(split=split, score=score, logit_difference=difference,
                                **intensity(r['path']))
                    handle.write((json.dumps(item, allow_nan=False) + '\n').encode())
                    records.append(item)
                handle.flush()
                os.fsync(handle.fileno())
                done = len(records) - initial
                seconds = time.monotonic() - start
                eta = (len(expected) - len(records)) * seconds / max(done, 1)
                print('Cached {}/{} slices; {:.1f} slices/s; inference ETA {:.1f} min'.format(len(records), len(expected), done / max(seconds, .001), eta / 60), flush=True)
            offset += len(ds)
    atomic_json(args.output_dir / 'runtime.json', dict(inference_seconds_this_invocation=time.monotonic() - start,
        new_slices=len(records) - initial, device=device,
        peak_cuda_tensor_mb=torch.cuda.max_memory_allocated() / 1024**2 if device == 'cuda' else None))
    print('Inference complete; building patient-cluster diagnostics.', flush=True)
    summarize(records, args.output_dir, args.bootstrap, args.seed)
    atomic_json(args.output_dir / 'complete.json', {'complete': True, 'slices': len(records)})
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('check', 'analyze'))
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoints/convnext/convnext_aug_adamw_run01/best.pth'))
    parser.add_argument('--config', type=Path, required=True, help='Historical augmented AdamW config.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--bootstrap', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-seconds', type=int, default=0, help='Graceful inference budget; 0 unlimited. Reporting is separately resumable.')
    args = parser.parse_args()
    if args.batch_size < 1 or args.bootstrap < 0 or args.max_seconds < 0:
        parser.error('Invalid batch-size/bootstrap/max-seconds.')
    if args.mode == 'analyze' and not args.resume:
        parser.error('Run check first, then analyze with --resume.')
    output = args.output_dir
    if output.exists() and not args.resume:
        parser.error('Output already exists; use a new directory or --resume.')
    if args.resume and not (output / 'manifest.json').is_file():
        parser.error('Resume requires a successful check manifest.')
    if not output.exists():
        output.mkdir(parents=True)
    key_path = output / '.local_hmac_key'
    if args.resume:
        key = key_path.read_bytes()
    else:
        key = secrets.token_bytes(32)
        with key_path.open('xb') as handle:
            handle.write(key)
        if os.name != 'nt':
            os.chmod(str(output), 0o700)
            os.chmod(str(key_path), 0o600)
    model, datasets, rows, stats, protocol = load_development(args, key)
    if args.resume:
        previous = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
        if previous['protocol'] != protocol:
            fail('Resume protocol/content/environment mismatch; use original settings.')
    else:
        atomic_json(output / 'manifest.json', dict(check_passed=True, protocol=protocol, data=stats,
            notes=['Test paths enumerated only for existing leakage exclusion; no test image decoded.',
                   'All development images decoded and content-hashed; weights loaded strictly without forward pass.',
                   'Numeric slice ordering differs from lexicographic loader ordering; association uses explicit numeric indices.',
                   'Legacy checkpoint provenance cannot be fully verified retroactively.']))
        atomic_json(output / 'check_summary.json', dict(check_passed=True, data=stats))
    print('Check passed: development files, mappings, split fingerprints and checkpoint structure verified.', flush=True)
    if args.mode == 'check':
        return
    if not args.resume:
        fail('Run check first, then analyze with --resume using its output directory.')
    if (output / 'complete.json').exists():
        print('Already complete; existing results preserved.')
        return
    analyze(model, datasets, rows, args)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Third-party errors can contain raw MRI paths. Never print their text.
        print('Diagnostic failed ({})'.format(type(error).__name__), file=sys.stderr)
        if isinstance(error, DiagnosticError):
            print(str(error), file=sys.stderr)
        sys.exit(1)
