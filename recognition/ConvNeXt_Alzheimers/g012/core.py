"""Grouped supervision math/data and matched randomness; Codex-assisted.

G1 uses logsumexp(log_softmax) without detaching the aggregate. G2 always
uses softmax(mean logits) at scan level, including validation/replay. There
are no model/head changes and no patient/class weighting.
"""
from __future__ import annotations
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from PIL import Image
from torch import nn
import dataset as data
import train_convnext_v2 as v2
from training_utils import binary_metrics

VARIANTS = ('G0', 'G1', 'G2')
NATIVE = {'G0': 'mean_slice_probabilities', 'G1': 'mean_slice_probabilities',
          'G2': 'softmax_mean_slice_logits'}
LOSSES = {'G0': 'mean_slice_hard_CE', 'G1': 'negative_log_mean_true_class_probability',
          'G2': 'hard_CE_mean_logits'}
require = data.require


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def tensor_digest(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        t = tensor.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(t.dtype).encode()); h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def config(variant, device='cuda'):
    require(variant in VARIANTS, 'Only G0/G1/G2 seed 42 are permitted.')
    base = v2.Config('scratch_augmented', batch_size=20, accumulation_steps=2, device=device)
    return {**asdict(base), 'variant': variant, 'loss': LOSSES[variant],
            'scan_aggregation': NATIVE[variant], 'patient_aggregation': 'mean_scan_AD_probabilities',
            'precision': 'float32', 'decision_threshold': .5}


def base_config(device='cuda'):
    return v2.Config('scratch_augmented', batch_size=20, accumulation_steps=2, device=device)


def scan_loss(logits, label, variant):
    require(variant in VARIANTS and logits.ndim == 2 and logits.shape[0] > 0 and logits.shape[1] == 2,
            'Expected positive-K by two finite logits and a declared objective.')
    require(label in (0, 1) and bool(torch.isfinite(logits).all()), 'Invalid label or logits.')
    if variant == 'G0':
        return -torch.log_softmax(logits, dim=1)[:, label].mean()
    if variant == 'G1':
        return -(torch.logsumexp(torch.log_softmax(logits, dim=1)[:, label], dim=0)
                 - math.log(logits.shape[0]))
    return -torch.log_softmax(logits.mean(dim=0), dim=0)[label]


def scan_probs(logits, variant):
    require(variant in VARIANTS and logits.ndim == 2 and logits.shape[0] > 0 and logits.shape[1] == 2
            and bool(torch.isfinite(logits).all()), 'Invalid native aggregation input.')
    if variant == 'G2':
        return torch.softmax(logits.mean(dim=0), dim=0)
    return torch.softmax(logits, dim=1).mean(dim=0)


@dataclass(frozen=True)
class Scan:
    scan_id: str
    subject_id: str
    label: int
    split: str
    rows: tuple


def group_records(records, split, expected_k=None):
    require(split in ('train', 'validation') and bool(records), 'Only train/validation bags are allowed.')
    grouped = defaultdict(list); paths = set()
    for row in records:
        require(row.split == split and row.label in (0, 1), 'Bag membership is outside the declared manifest.')
        require(row.image_path not in paths, 'Duplicate manifest image.')
        paths.add(row.image_path); grouped[row.scan_id].append(row)
    bags = []
    for scan_id, rows in sorted(grouped.items()):
        require(len({(r.subject_id, r.label, r.split) for r in rows}) == 1, 'Mixed scan patient/label/partition.')
        indices = [r.slice_index for r in rows]
        require(len(set(indices)) == len(indices), 'Duplicate scan slice index.')
        require(expected_k is None or len(rows) == expected_k, 'Unexpected scan slice count; no truncation allowed.')
        rows = tuple(sorted(rows, key=lambda r: (r.slice_index, r.image_path)))
        bags.append(Scan(scan_id, rows[0].subject_id, rows[0].label, split, rows))
    require(sum(len(b.rows) for b in bags) == len(records), 'Bag construction omitted manifest rows.')
    return tuple(bags)


def checked_groups(bundle, approved):
    require(bundle.manifest_hashes == approved['manifest_sha256'], 'Approved manifest identity differs.')
    result = {}
    for split in ('train', 'validation'):
        require(data.counts_for_records(bundle.records[split]) == approved['counts'][split],
                'Unexpected grouping/class counts.')
        result[split] = group_records(bundle.records[split], split, expected_k=20)
    require(len(result['train']) == 1061 and len(result['validation']) == 233,
            'Expected 1061 training and 233 validation scans.')
    return result


def seed_for(kind, epoch):
    return int(digest(['G012-seed42-v1', kind, epoch])[:15], 16)


def schedule(bags, epoch):
    """Separate order and per-slice angle generators, identical across variants.

    Angle draws follow canonical scan/slice order, independent of shuffle and
    model/validation RNG consumption. No shared angle within a scan.
    """
    require(epoch >= 1 and all(b.split == 'train' for b in bags), 'Training schedule requires training bags.')
    order_rng = torch.Generator().manual_seed(seed_for('scan_order', epoch))
    angle_rng = torch.Generator().manual_seed(seed_for('slice_rotation', epoch))
    order = torch.randperm(len(bags), generator=order_rng).tolist()
    angles = [(torch.rand(len(b.rows), generator=angle_rng) * 2 - 1).mul(5).tolist() for b in bags]
    private = {'epoch': epoch, 'order': order, 'angles': angles,
               'order_digest': digest([bags[i].scan_id for i in order]),
               'augmentation_digest': digest([[b.scan_id, [r.slice_index for r in b.rows], a]
                                             for b, a in zip(bags, angles)]),
               'order_rng_state': order_rng.get_state(), 'angle_rng_state': angle_rng.get_state(),
               'next_epoch': epoch + 1, 'next_scan_position': 0}
    return private


@contextmanager
def isolated_rng():
    state = v2.rng_state()
    try:
        yield
    finally:
        v2.restore_rng(state)


def load_scan(root, bag, angles=None):
    require(bag.split in ('train', 'validation'), 'No test pixels may be loaded.')
    require(angles is None or (bag.split == 'train' and len(angles) == len(bag.rows)), 'Invalid augmentation scope.')
    root = Path(root).resolve(); images = []
    for index, row in enumerate(bag.rows):
        require(row.split == bag.split, 'Scan row changed partition.')
        path = (root / row.image_path).resolve()
        require(root in path.parents, 'Image resolves outside the data root.')
        with Image.open(path) as image:
            image.load()
            images.append(data.image_to_tensor(image, 0. if angles is None else angles[index]))
    return torch.stack(images)


def train_epoch(model, bags, fetch, optimizer, device, variant, matched, resource=None):
    require(all(b.split == 'train' for b in bags), 'Optimizer may only receive training scans.')
    order = matched['order']; require(sorted(order) == list(range(len(bags))), 'Incomplete/duplicate scan order.')
    model.train(); optimizer.zero_grad(set_to_none=True)
    objective = 0.; updates = 0; gradients = []
    for position, index in enumerate(order):
        window = min(2, len(order) - (position // 2) * 2)
        x = fetch(bags[index], matched['angles'][index]).to(device)
        logits = model(x)
        loss = scan_loss(logits, bags[index].label, variant)
        require(bool(torch.isfinite(loss)), 'Nonfinite objective.')
        (loss / window).backward()
        objective += float(loss.detach())
        if resource: resource.observe()
        # Only detached scalars survive this microbatch. No graph across scans.
        del logits, loss, x
        if (position + 1) % 2 == 0 or position + 1 == len(order):
            norm = nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            gradients.append(float(norm)); optimizer.step(); optimizer.zero_grad(set_to_none=True); updates += 1
            if resource: resource.observe()
    require(updates == math.ceil(len(bags) / 2), 'Wrong update count, including final one-scan window.')
    v2.synchronize(device)
    return {'training_objective': objective / len(bags), 'optimizer_updates': updates,
            'training_scans': len(bags), 'max_preclip_gradient_norm': max(gradients)}


def scored(labels, probabilities):
    result = binary_metrics(labels, probabilities)
    result['brier'] = math.fsum((q-y)**2 for y, q in zip(labels, probabilities)) / len(labels)
    return result


def aggregate(bags, logits_by_scan, variant):
    require(len(bags) == len(logits_by_scan), 'Incomplete scan predictions.')
    scans = []; slices = []
    for bag, logits in zip(bags, logits_by_scan):
        require(len(logits) == len(bag.rows), 'Incomplete slice logits.')
        p = torch.softmax(logits, 1)[:, 1].tolist()
        for r, q, z in zip(bag.rows, p, logits.tolist()):
            slices.append({'image_path': r.image_path, 'scan_id': r.scan_id, 'subject_id': r.subject_id,
                           'label': r.label, 'p_ad': q, 'logits': z})
        scans.append({'scan_id': bag.scan_id, 'subject_id': bag.subject_id, 'label': bag.label,
                      'p_ad': float(scan_probs(logits, variant)[1]), 'slices': len(bag.rows)})
    by_subject = defaultdict(list)
    for r in scans: by_subject[r['subject_id']].append(r)
    patients = []
    for subject, rows in sorted(by_subject.items()):
        require(len({r['label'] for r in rows}) == 1, 'Mixed patient label.')
        patients.append({'subject_id': subject, 'label': rows[0]['label'], 'scans': len(rows),
                         'p_ad': math.fsum(r['p_ad'] for r in rows) / len(rows)})
    predictions = {'slices': slices, 'scans': scans, 'patients': patients}
    metrics = {name: scored([r['label'] for r in predictions[key]], [r['p_ad'] for r in predictions[key]])
               for name, key in [('slice', 'slices'), ('scan', 'scans'), ('patient', 'patients')]}
    return metrics, predictions


def predict(model, bags, fetch, device, variant, resource=None):
    require(all(b.split == 'validation' for b in bags), 'Full replay/selection uses validation only.')
    # Model initialization, validation and later checkpoint operations cannot
    # change the training stream. Every epoch also has isolated schedule RNGs.
    with isolated_rng(), torch.inference_mode():
        model.eval(); logits = []
        for bag in bags:
            logits.append(model(fetch(bag).to(device)).detach().cpu())
            if resource: resource.observe()
        return aggregate(bags, logits, variant)


def strict_state(model, state):
    expected = model.state_dict()
    require(isinstance(state, dict) and set(state) == set(expected), 'Checkpoint state keys differ.')
    for name, target in expected.items():
        value = state[name]
        require(isinstance(value, torch.Tensor) and value.shape == target.shape and value.dtype == target.dtype,
                'Checkpoint shape/dtype differs (including scalar shape).')
        require(bool(torch.isfinite(value).all()), 'Nonfinite checkpoint tensor.')
    model.load_state_dict(state, strict=True)


def replay_difference(a, b, tolerance=2e-5):
    result = {}
    for key, identity in [('slices', 'image_path'), ('scans', 'scan_id'), ('patients', 'subject_id')]:
        left = {r[identity]: r for r in a[key]}; right = {r[identity]: r for r in b[key]}
        require(len(left) == len(a[key]) and len(right) == len(b[key]) and left.keys() == right.keys(),
                'Replay coverage differs.')
        require(all(left[k]['label'] == right[k]['label'] for k in left), 'Replay labels differ.')
        delta = max(abs(left[k]['p_ad'] - right[k]['p_ad']) for k in left)
        decisions = all((left[k]['p_ad'] >= .5) == (right[k]['p_ad'] >= .5) for k in left)
        require(delta <= tolerance and decisions, 'Material probability/class replay failure; do not loosen tolerance.')
        result[key] = {'n': len(left), 'max_absolute_probability_difference': delta, 'classes_unchanged': decisions}
    return result
