"""Locked, fair CNN vs ConvNeXt final test benchmark + GPU inference profiling.

Runs both previously chosen checkpoints on the SAME held-out 6,080 slices.
Does NOT train, tune thresholds, or alter checkpoints. Writes only aggregated
statistics; neither MRI images nor patient identifiers are exported.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import ADNIDataset
from dataset_v2 import build_splits
from modules import CNNBaseline, ConvNeXtClassifier


def area_under_roc(labels, prob):
    yy = np.asarray(labels, dtype=int)
    score = np.asarray(prob, dtype=float)
    positives = int((yy == 1).sum())
    negatives = int((yy == 0).sum())
    if not positives or not negatives:
        return None
    order = np.argsort(-score, kind='stable')
    yy, ss = yy[order], score[order]
    tp = np.cumsum(yy == 1)
    fp = np.cumsum(yy == 0)
    idx = np.r_[np.flatnonzero(ss[:-1] != ss[1:]), len(ss) - 1]
    tpr = np.r_[0, tp[idx] / positives]
    fpr = np.r_[0, fp[idx] / negatives]
    return float(np.sum((tpr[1:] + tpr[:-1]) * 0.5 * (fpr[1:] - fpr[:-1])))


def class_metrics(yy, pred):
    yy, pred = np.asarray(yy), np.asarray(pred)
    tn = int(((yy == 0) & (pred == 0)).sum())
    fp = int(((yy == 0) & (pred == 1)).sum())
    fn = int(((yy == 1) & (pred == 0)).sum())
    tp = int(((yy == 1) & (pred == 1)).sum())
    def safe(a, b): return float(a / b) if b else 0.
    pnc, pad = safe(tn, tn + fn), safe(tp, tp + fp)
    rnc, rad = safe(tn, tn + fp), safe(tp, tp + fn)
    f1nc, f1ad = safe(2 * pnc * rnc, pnc + rnc), safe(2 * pad * rad, pad + rad)
    return {
        'accuracy': safe(tn + tp, len(yy)),
        'macro_precision': (pnc + pad) / 2,
        'macro_recall': (rnc + rad) / 2,
        'macro_f1': (f1nc + f1ad) / 2,
        'NC': {'precision': pnc, 'recall': rnc, 'f1': f1nc},
        'AD': {'precision': pad, 'recall': rad, 'f1': f1ad},
        'confusion_TN_FP_FN_TP': [tn, fp, fn, tp],
    }


def make_model(which, checkpoint_path, device):
    model = CNNBaseline() if which == 'CNN' else ConvNeXtClassifier()
    if which != 'CNN':
        for layer in model.modules():
            if isinstance(layer, nn.LayerNorm):
                layer.eps = 1e-6
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.to(device).eval()
    return model, checkpoint.get('epoch', None)


def evaluate(name, path, loader, device):
    model, checkpoint_epoch = make_model(name, path, device)
    params = sum(p.numel() for p in model.parameters())
    # Warm GPU kernels once (excluded from throughput / latency).
    warm = next(iter(loader))['image'].to(device)
    with torch.inference_mode():
        for _ in range(3):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _ = model(warm)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        yy, prob = [], []
        scan_probs, scan_patient, scan_label = defaultdict(list), {}, {}
        forward_ms, wall_start = 0.0, time.perf_counter()
        for batch in loader:
            images = batch['image'].to(device, non_blocking=True)
            event_start = torch.cuda.Event(enable_timing=True)
            event_end = torch.cuda.Event(enable_timing=True)
            event_start.record()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(images)
            event_end.record()
            torch.cuda.synchronize()
            forward_ms += event_start.elapsed_time(event_end)
            values = logits.float().softmax(1)[:, 1].cpu().tolist()
            labels = batch['label'].tolist()
            yy.extend(labels)
            prob.extend(values)
            for p, sid, pid, lab in zip(values, batch['scan_id'], batch['patient_id'], labels):
                scan_probs[sid].append(p)
                scan_patient[sid] = pid
                scan_label[sid] = lab
        torch.cuda.synchronize()
        wall_seconds = time.perf_counter() - wall_start
    yy, prob = np.array(yy, dtype=int), np.array(prob, dtype=float)
    pred = (prob >= 0.5).astype(int)
    result = class_metrics(yy, pred)
    confidence = np.maximum(prob, 1 - prob)
    correctness = (pred == yy).astype(float)
    ece = 0.0
    for k in range(10):
        lo, hi = k / 10, (k+1) / 10
        mask = (confidence >= lo) & ((confidence < hi) if k < 9 else (confidence <= hi))
        if mask.any():
            ece += float(mask.mean()) * abs(float(correctness[mask].mean()) - float(confidence[mask].mean()))
    result.update({
        'ece_10_bin': float(ece), 'brier_score': float(np.mean((prob - yy)**2)),
        'checkpoint_epoch': checkpoint_epoch, 'parameters': params,
        'slices': len(yy), 'slice_auroc': area_under_roc(yy, prob),
        'gpu_forward_ms_per_slice': forward_ms / len(yy),
        'end_to_end_seconds': wall_seconds,
        'end_to_end_slices_per_second': len(yy) / wall_seconds,
        'peak_gpu_reserved_gb_inference': torch.cuda.max_memory_reserved() / (1024 ** 3),
        'inference_batch_size': loader.batch_size,
    })
    patient_prob = defaultdict(list)
    patient_label = {}
    for sid, pp in scan_probs.items():
        pid = scan_patient[sid]
        patient_prob[pid].append(float(np.mean(pp)))
        if pid in patient_label:
            assert patient_label[pid] == scan_label[sid]
        patient_label[pid] = scan_label[sid]
    ids = sorted(patient_prob)
    pt = np.array([patient_label[p] for p in ids])
    pp = np.array([np.mean(patient_prob[p]) for p in ids])
    result['patient_accuracy'] = float(np.mean((pp >= 0.5) == pt))
    result['patient_auroc'] = area_under_roc(pt, pp)
    result['patients'] = len(ids)
    del model
    torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='runs/phase4_benchmark/benchmark.json')
    ap.add_argument('--batch-size', type=int, default=16)
    args = ap.parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Run with Slurm on A100 for BF16 inference profiling.')
    torch.manual_seed(3710)
    rows = build_splits(seed=3710)['test']
    loader = DataLoader(ADNIDataset(rows), batch_size=args.batch_size,
                        shuffle=False, num_workers=2, pin_memory=True)
    paths = {
        'CNN': 'runs/baseline_v3_15epochs/best.pt',
        'ConvNeXt V7': 'runs/convnext_v7_pretrained20_newsplit/best.pt',
    }
    results = {}
    for name, filename in paths.items():
        if not Path(filename).is_file():
            raise FileNotFoundError(filename)
        print('Evaluating', name, flush=True)
        results[name] = evaluate('CNN' if name == 'CNN' else 'V7', filename, loader,
                                 torch.device('cuda'))
        print(json.dumps(results[name], indent=2), flush=True)
    if results['ConvNeXt V7']['confusion_TN_FP_FN_TP'] != [2722, 418, 586, 2354]:
        raise RuntimeError('V7 differs from locked final test: do not report new benchmark.')
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'test_only_no_tuning': True, 'results': results}, indent=2))
    print('Saved', out, flush=True)


if __name__ == '__main__':
    main()
