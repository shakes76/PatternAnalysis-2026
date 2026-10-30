"""Shared checkpoint loading and explicit segmentation metric conventions."""
import hashlib
import math
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from support.baseline_model import make_model


LABEL_NAMES = ['Background', 'Body', 'Bones', 'Bladder', 'Rectum', 'Prostate']


def load_model(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    config = checkpoint.get('model_config')
    if not config or config.get('normalization') not in ('batch', 'instance'):
        raise ValueError('Checkpoint must specify model_config including normalization')
    model = make_model(base_channels=config['base_channels'],
                       normalization=config['normalization']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    return model, checkpoint


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def class_metrics(prediction, target):
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape:
        raise ValueError('Prediction and ground truth shapes differ')
    for array in (prediction, target):
        if not np.isfinite(array).all() or not np.equal(array, np.rint(array)).all():
            raise ValueError('Labels must be finite integers')
        if array.min() < 0 or array.max() >= 6:
            raise ValueError('Labels must be in 0..5')
    encoded = target.astype(np.int64).reshape(-1)*6 + prediction.astype(np.int64).reshape(-1)
    confusion = np.bincount(encoded, minlength=36).reshape(6,6)
    rows = []
    for label, name in enumerate(LABEL_NAMES):
        tp = int(confusion[label,label])
        gt, pred = int(confusion[label].sum()), int(confusion[:,label].sum())
        fp, fn = pred-tp, gt-tp
        rows.append(dict(class_id=label, class_name=name, gt_voxels=gt,
                         pred_voxels=pred, tp=tp, fp=fp, fn=fn,
                         dice=2*tp/(gt+pred) if gt+pred else None,
                         iou=tp/(tp+fp+fn) if tp+fp+fn else None,
                         precision=tp/pred if pred else None,
                         recall=tp/gt if gt else None,
                         volume_ratio=pred/gt if gt else None))
    return rows


def hd95(prediction, target, spacing):
    """P95 of pooled bidirectional six-connected surface distances in mm.

    Matches the project's earlier evaluation definition. Empty masks have
    undefined distances, reported as None with a status, never zero.
    """
    from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure
    prediction, target = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError('HD95 requires equal 3D arrays')
    if len(spacing) != 3 or any(not math.isfinite(s) or s <= 0 for s in spacing):
        raise ValueError('Invalid voxel spacing')
    if not prediction.any() and not target.any():
        return None, 'both_empty'
    if not prediction.any():
        return None, 'prediction_empty'
    if not target.any():
        return None, 'ground_truth_empty'
    structure = generate_binary_structure(3, 1)
    ps = prediction ^ binary_erosion(prediction, structure=structure, border_value=0)
    ts = target ^ binary_erosion(target, structure=structure, border_value=0)
    distances = np.concatenate((distance_transform_edt(~ts, sampling=spacing)[ps],
                                distance_transform_edt(~ps, sampling=spacing)[ts]))
    return float(np.percentile(distances,95)), 'ok'


def geometry(mri_path, target_shape):
    image = nib.load(str(mri_path))
    original = image.shape[:3]
    zooms = nib.affines.voxel_sizes(image.affine)
    spacing = [float(zooms[2]*original[2]/target_shape[0]),
               float(zooms[0]*original[0]/target_shape[1]),
               float(zooms[1]*original[1]/target_shape[2])]
    # Nearest-neighbour mask resize maps target W to original NIfTI axis 1.
    original_y = np.floor(np.arange(target_shape[2])*original[1]/target_shape[2]).astype(int)
    shared = np.flatnonzero((original_y >= 98) & (original_y < 158))
    return dict(original_shape=list(original), original_affine=image.affine.tolist(),
                original_axcodes=list(nib.aff2axcodes(image.affine)),
                spacing_dhw_mm=spacing, shared_target_w_indices=shared.tolist(),
                geometry_note='Shape-ratio spacing in resized space; approximate original-space mm')


def boundary_regions(prediction, target, z_direction):
    """First/last two GT-bearing depth planes are extent proxies, not clinical landmarks."""
    indices = np.flatnonzero((target == 5).any(axis=(1,2)))
    if not len(indices):
        return []
    lower = list(range(int(indices[0]), min(int(indices[0])+2,int(indices[-1])+1)))
    upper = list(range(max(int(indices[-1])-1,int(indices[0])),int(indices[-1])+1))
    names = ('inferior_extent_apex_proxy', 'superior_extent_base_proxy') if z_direction == 'S' else (
        ('superior_extent_base_proxy', 'inferior_extent_apex_proxy') if z_direction == 'I' else
        ('low_array_extent', 'high_array_extent'))
    result = []
    for name, planes in zip(names, (lower,upper)):
        row = class_metrics(prediction[planes], target[planes])[5]
        result.append(dict(row, region=name, depth_indices=','.join(map(str,planes))))
    return result


def summarize(rows):
    result = []
    for label, name in enumerate(LABEL_NAMES):
        group = [r for r in rows if r['class_id'] == label]
        dice = [r['dice'] for r in group if r['dice'] is not None]
        iou = [r['iou'] for r in group if r['iou'] is not None]
        totals = {k: sum(r[k] for r in group) for k in ('tp','fp','fn','gt_voxels','pred_voxels')}
        tp, fp, fn = totals['tp'], totals['fp'], totals['fn']
        patient_means = []
        for patient in sorted({r['patient_id'] for r in group}):
            values = [r['dice'] for r in group if r['patient_id']==patient and r['dice'] is not None]
            if values:
                patient_means.append(float(np.mean(values)))
        result.append(dict(class_id=label, class_name=name, **totals,
                           mean_case_dice=float(np.mean(dice)) if dice else None,
                           std_case_dice=float(np.std(dice)) if dice else None,
                           mean_case_iou=float(np.mean(iou)) if iou else None,
                           mean_patient_dice=float(np.mean(patient_means)) if patient_means else None,
                           valid_dice_cases=len(dice),
                           pooled_dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                           pooled_precision=tp/(tp+fp) if tp+fp else None,
                           pooled_recall=tp/(tp+fn) if tp+fn else None,
                           pooled_volume_ratio=totals['pred_voxels']/totals['gt_voxels']
                           if totals['gt_voxels'] else None))
    return result


def select_examples(prostate_rows, count=5):
    """Deterministic error categories, preferring different patients."""
    by_dice = sorted(prostate_rows, key=lambda r: (r['dice'] if r['dice'] is not None else -1, r['case']))
    strategies = [
        ('lowest_dice', by_dice),
        ('highest_hd95_or_missing_prediction', sorted(prostate_rows, key=lambda r:
            -(r['hd95_mm'] if r['hd95_mm'] is not None else float('inf')))),
        ('organ_confusion', sorted(prostate_rows, key=lambda r:
            -(r['prostate_fp_in_bladder']+r['prostate_fp_in_rectum']))),
        ('largest_missed_fraction', sorted(prostate_rows, key=lambda r:
            -(1-r['recall'] if r['recall'] is not None else -1))),
        ('median_dice', by_dice[len(by_dice)//2:]+by_dice[:len(by_dice)//2]),
    ]
    selected, patients = [], set()
    for reason, candidates in strategies[:count]:
        unused = [r for r in candidates if r['case'] not in {s['case'] for s in selected}]
        if not unused:
            break
        row = next((r for r in unused if r['patient_id'] not in patients), unused[0])
        selected.append(dict(case=row['case'], patient_id=row['patient_id'], reason=reason))
        patients.add(row['patient_id'])
    return selected
