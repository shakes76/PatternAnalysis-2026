"""Metric definitions reused from our frozen 3D evaluator for matched comparison."""

import math

import numpy as np

import nibabel as nib

LABEL_NAMES = ['Background', 'Body', 'Bones', 'Bladder', 'Rectum', 'Prostate']

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
