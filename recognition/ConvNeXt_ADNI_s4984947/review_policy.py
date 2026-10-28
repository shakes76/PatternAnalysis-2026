"""Validation-only selection of an illustrative confidence-based review rule.

Never uses Test for threshold selection. A frozen rule is subsequently reported
on previously saved, anonymized final-test predictions for descriptive audit.
The operating recommendation is NOT a clinical deployment policy.
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import ADNIDataset
from dataset_v2 import build_splits
from modules import ConvNeXtClassifier


def eval_prob_on_validation():
    rows = build_splits(seed=3710)['val']
    loader = DataLoader(ADNIDataset(rows), batch_size=16, shuffle=False,
                        num_workers=2, pin_memory=True)
    model = ConvNeXtClassifier(num_classes=2)
    for layer in model.modules():
        if isinstance(layer, nn.LayerNorm):
            layer.eps = 1e-6
    cp = torch.load('runs/convnext_v7_pretrained20_newsplit/best.pt',
                    map_location='cpu', weights_only=True)
    model.load_state_dict(cp['model_state_dict'], strict=True)
    assert cp['epoch'] == 13
    model.cuda().eval()
    y, p = [], []
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(batch['image'].cuda(non_blocking=True))
            p.extend(logits.float().softmax(1)[:, 1].cpu().tolist())
            y.extend(batch['label'].tolist())
    return np.asarray(y, int), np.asarray(p, float)


def summary(y, p, threshold):
    pred = (p >= 0.5).astype(int)
    confidence = np.maximum(p, 1 - p)
    accept = confidence >= threshold
    n = int(accept.sum())
    return {'threshold': float(threshold), 'total': len(y), 'accepted': n,
            'coverage': float(n / len(y)),
            'accepted_accuracy': float((pred[accept] == y[accept]).mean()) if n else None,
            'reviewed': int(len(y)-n),
            'review_fraction': float(1-n / len(y)),
            'errors_accepted': int((pred[accept] != y[accept]).sum()) if n else None,
            'errors_reviewed': int((pred[~accept] != y[~accept]).sum())}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('Use A100 Slurm job for validation inference')
    y, p = eval_prob_on_validation()
    # Candidate grid fixed in advance; max coverage for >=90% accuracy on
    # accepted validation slices. Minimum 20% coverage prevents degenerate 0-coverage rules.
    goal_acc, min_coverage = .90, .20
    candidates = [summary(y, p, float(t)) for t in np.linspace(.50, .99, 50)]
    feasible = [r for r in candidates if r['accepted'] and
                r['accepted_accuracy'] >= goal_acc and r['coverage'] >= min_coverage]
    choice = sorted(feasible, key=lambda r:(-r['coverage'], r['threshold']))[0] if feasible else None
    out = {'selection': 'validation-only, slice-level', 'target_accepted_accuracy': goal_acc,
           'minimum_coverage': min_coverage, 'validation': choice,
           'warning': 'Illustrative non-clinical rule; correlated slices and limited patients; '
                      'selection is optimistic until externally validated.'}
    if choice is not None:
        test_csv = Path('runs/phase2_final_analysis/test_predictions_anonymized.csv')
        if test_csv.is_file():
            with test_csv.open(newline='') as f:
                test_rows = list(csv.DictReader(f))
            ty = np.array([int(r['true']) for r in test_rows], int)
            tp = np.array([float(r['p_ad']) for r in test_rows], float)
            out['held_out_test_fixed_rule'] = summary(ty, tp, choice['threshold'])
        print('Selected threshold from validation only:', choice, flush=True)
    else:
        print('No candidate met accuracy and coverage targets on validation.', flush=True)
    folder = Path('runs/phase4_benchmark')
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'review_rule.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2), flush=True)


if __name__ == '__main__':
    main()
