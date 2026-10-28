"""Descriptive ONE-TIME evaluation visualizations for LOCKED V7 Epoch 13.
Does NOT train, tune threshold, choose checkpoint, or alter the Test split.
Contains no patient IDs in exported results and keeps MRI case images under ignored runs/.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import ADNIDataset
from dataset_v2 import build_splits
from v7_pretrained import load_custom_pretrained

OUT = Path('runs/phase2_final_analysis')
OUT.mkdir(parents=True, exist_ok=True)
WEIGHTS = 'pretrained/convnext_tiny-983f1562.pth'
CHECKPOINT = 'runs/convnext_v7_pretrained20_newsplit/best.pt'


def div(a, b):
    return float(a) / float(b) if b else 0.0


def roc_curve_auc(y, probs):
    order = np.argsort(-probs, kind='stable')
    yy, ss = y[order], probs[order]
    tp = np.cumsum(yy == 1)
    fp = np.cumsum(yy == 0)
    last_of_group = np.r_[np.flatnonzero(ss[:-1] != ss[1:]), len(ss)-1]
    tpr = np.r_[0., tp[last_of_group] / sum(y == 1)]
    fpr = np.r_[0., fp[last_of_group] / sum(y == 0)]
    auc = float(np.sum((tpr[1:] + tpr[:-1]) * 0.5 * (fpr[1:] - fpr[:-1])))
    return fpr, tpr, auc


def draw_case_grid(rows, chosen_indices, predictions, target, title, dest):
    if not chosen_indices:
        print('No errors available for', title)
        return
    fig, axs = plt.subplots(1, len(chosen_indices), figsize=(4.2 * len(chosen_indices), 4.1))
    axs = np.atleast_1d(axs)
    for ax, i in zip(axs, chosen_indices):
        path, label, _, _ = rows[i]
        with Image.open(path) as im:
            gray = ImageOps.pad(im.convert('L'), (224, 224),
                                method=Image.Resampling.BILINEAR, color=0)
            ax.imshow(gray, cmap='gray', vmin=0, vmax=255)
        rec = predictions[i]
        confidence = max(rec['p_ad'], 1 - rec['p_ad'])
        ax.set_title(f"Slice #{i} | {target}\nTrue: {'AD' if label else 'NC'}"
                     f"   Pred: {'AD' if rec['pred'] else 'NC'}\n"
                     f"P(AD)={rec['p_ad']:.3f}  Conf.={confidence:.3f}", fontsize=10)
        ax.axis('off')
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(dest, dpi=165, bbox_inches='tight')
    plt.close(fig)


def select_distinct_patient_errors(rows, records, error_type, number=3):
    candidates = [i for i, r in enumerate(records)
                  if r['pred'] != r['true'] and
                  ((error_type == 'FP' and r['true'] == 0) or
                   (error_type == 'FN' and r['true'] == 1))]
    candidates.sort(key=lambda i: max(records[i]['p_ad'], 1 - records[i]['p_ad']), reverse=True)
    chosen, seen = [], set()
    for idx in candidates:
        pid = rows[idx][2]
        if pid in seen: continue
        seen.add(pid); chosen.append(idx)
        if len(chosen) >= number: break
    return chosen


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('This analysis needs an allocated GPU: submit Slurm script.')
    torch.manual_seed(3710)
    splits = build_splits(seed=3710)
    rows = splits['test']
    assert len(rows) == 6080, f'Unexpected test set count: {len(rows)}'
    loader = DataLoader(ADNIDataset(rows), batch_size=16, shuffle=False,
                        num_workers=2, pin_memory=True)
    device = torch.device('cuda')
    model = load_custom_pretrained(WEIGHTS, num_classes=2).to(device)
    checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    assert checkpoint['epoch'] == 13, 'Expected locked V7 Epoch 13 best checkpoint.'
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()

    predictions = []
    scan_probs, scan_patient, scan_label = defaultdict(list), {}, {}
    offset = 0
    with torch.inference_mode():
        for batch in loader:
            imgs = batch['image'].to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(imgs)
            pp = logits.float().softmax(dim=1)[:, 1].cpu().numpy()
            yy = batch['label'].numpy()
            for j, (p, label, sid, pid) in enumerate(zip(pp, yy,
                   batch['scan_id'], batch['patient_id'])):
                i = offset + j
                pred = int(float(p) >= 0.5)
                predictions.append({'slice_index': i, 'true': int(label),
                                    'pred': pred, 'p_ad': float(p),
                                    'correct': int(pred == label)})
                scan_probs[sid].append(float(p))
                scan_patient[sid] = pid
                scan_label[sid] = int(label)
            offset += len(pp)
    assert offset == len(rows)
    y = np.array([r['true'] for r in predictions], dtype=int)
    prob = np.array([r['p_ad'] for r in predictions], dtype=float)
    pred = np.array([r['pred'] for r in predictions], dtype=int)
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tp = int(np.sum((y == 1) & (pred == 1)))
    # Prior published one-time test result. Do not silently present mismatched metrics.
    if (tn, fp, fn, tp) != (2722, 418, 586, 2354):
        raise RuntimeError(f'Test predictions differ from locked evaluation: {(tn,fp,fn,tp)}. '
                           'Check checkpoint, preprocessing, and partition before using plots.')
    fpr, tpr, auc = roc_curve_auc(y, prob)
    precision_ad = div(tp, tp + fp)
    precision_nc = div(tn, tn + fn)
    recall_ad = div(tp, tp + fn)
    recall_nc = div(tn, tn + fp)
    f1_ad = div(2 * precision_ad * recall_ad, precision_ad + recall_ad)
    f1_nc = div(2 * precision_nc * recall_nc, precision_nc + recall_nc)

    # Patient pooling: mean slices per scan, then mean scans per patient.
    pscan = defaultdict(list)
    patient_labels = {}
    for scan, vals in scan_probs.items():
        pid = scan_patient[scan]
        pscan[pid].append(float(np.mean(vals)))
        if pid in patient_labels: assert patient_labels[pid] == scan_label[scan]
        patient_labels[pid] = scan_label[scan]
    patient_ids = sorted(pscan)
    py = np.array([patient_labels[pid] for pid in patient_ids], dtype=int)
    pprob = np.array([np.mean(pscan[pid]) for pid in patient_ids], dtype=float)
    patient_acc = float(np.mean((pprob >= 0.5) == py))
    _, _, patient_auc = roc_curve_auc(py, pprob)

    # Confidence, reliability, and ECE are descriptive; no re-calibration is done.
    confidence = np.maximum(prob, 1 - prob)
    correct = (pred == y).astype(int)
    bins = np.linspace(0, 1, 11)
    bin_data, ece = [], 0.0
    for k in range(10):
        mask = (confidence >= bins[k]) & ((confidence < bins[k + 1]) if k < 9 else (confidence <= bins[k + 1]))
        if not np.any(mask): continue
        avg_conf = float(np.mean(confidence[mask]))
        avg_acc = float(np.mean(correct[mask]))
        n = int(np.sum(mask))
        ece += n / len(y) * abs(avg_acc - avg_conf)
        bin_data.append((avg_conf, avg_acc, n))

    metrics = {
        'model': 'hand-written ConvNeXt V7 with ImageNet initialization',
        'checkpoint_epoch': 13, 'decision_threshold': 0.5,
        'test_slices': len(y), 'test_patients': len(py),
        'slice_accuracy': float(np.mean(pred == y)),
        'slice_auroc': auc, 'slice_macro_precision': (precision_nc + precision_ad) / 2,
        'slice_macro_recall': (recall_nc + recall_ad) / 2,
        'slice_macro_f1': (f1_nc + f1_ad) / 2,
        'NC': {'precision': precision_nc, 'recall': recall_nc, 'f1': f1_nc},
        'AD': {'precision': precision_ad, 'recall': recall_ad, 'f1': f1_ad},
        'confusion_TN_FP_FN_TP': [tn, fp, fn, tp],
        'patient_accuracy': patient_acc, 'patient_auroc': patient_auc,
        'brier_score': float(np.mean((prob - y) ** 2)),
        'ece_10_bin': float(ece),
        'note': 'Descriptive evaluation only; no threshold/weight selection based on test.'
    }
    (OUT / 'final_test_detailed_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    with (OUT / 'test_predictions_anonymized.csv').open('w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(predictions[0]))
        wr.writeheader(); wr.writerows(predictions)

    # 05 confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5.1))
    cm = np.array([[tn, fp], [fn, tp]])
    ax.imshow(cm, cmap='Blues')
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f'{v:,}', ha='center', va='center', fontsize=18,
                color='white' if v > cm.max() * 0.60 else 'black')
    ax.set_xticks([0, 1], ['Pred NC', 'Pred AD'])
    ax.set_yticks([0, 1], ['True NC', 'True AD'])
    ax.set_title(f'V7 final held-out test | Accuracy {100*np.mean(pred==y):.2f}%')
    fig.tight_layout(); fig.savefig(OUT / '05_final_test_confusion_matrix.png', dpi=180); plt.close(fig)

    # 06 slice-level ROC curve
    plt.figure(figsize=(6.2, 5.1))
    plt.plot(fpr, tpr, linewidth=2, label=f'Slice AUROC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
    plt.xlim(0, 1); plt.ylim(0, 1.02)
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('V7 final test — ROC curve (AD positive)')
    plt.grid(alpha=.2); plt.legend(loc='lower right'); plt.tight_layout()
    plt.savefig(OUT / '06_final_test_roc.png', dpi=180); plt.close()

    # 07 correct vs incorrect confidence distribution
    plt.figure(figsize=(7.6, 4.5))
    plt.hist(confidence[correct==1], bins=np.linspace(.5, 1, 21), alpha=.65,
             density=True, label=f'Correct (n={sum(correct==1)})')
    plt.hist(confidence[correct==0], bins=np.linspace(.5, 1, 21), alpha=.65,
             density=True, label=f'Incorrect (n={sum(correct==0)})')
    plt.xlabel('Confidence in predicted class'); plt.ylabel('Density')
    plt.title('V7 test prediction confidence — correct vs incorrect')
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT / '07_test_confidence_histogram.png', dpi=180); plt.close()

    # 08 reliability diagram
    plt.figure(figsize=(6.2, 5.1))
    plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=1, label='Perfect calibration')
    plt.plot([x[0] for x in bin_data], [x[1] for x in bin_data], 'o-', linewidth=2,
             label=f'Observed (ECE={ece:.4f})')
    plt.xlim(.45, 1.01); plt.ylim(0, 1.01)
    plt.xlabel('Average predicted confidence'); plt.ylabel('Observed accuracy')
    plt.title('Test confidence reliability (descriptive, 10 bins)')
    plt.legend(); plt.grid(alpha=.2); plt.tight_layout()
    plt.savefig(OUT / '08_test_reliability.png', dpi=180); plt.close()

    fps = select_distinct_patient_errors(rows, predictions, 'FP', 3)
    fns = select_distinct_patient_errors(rows, predictions, 'FN', 3)
    draw_case_grid(rows, fps, predictions, 'FP', 'False positive cases (distinct patients)',
                   OUT / '09_false_positive_mri_cases.png')
    draw_case_grid(rows, fns, predictions, 'FN', 'False negative cases (distinct patients)',
                   OUT / '10_false_negative_mri_cases.png')
    case_summary = {'FP_anonymized_slice_indices': fps,
                    'FN_anonymized_slice_indices': fns,
                    'note': 'Examples selected by incorrect prediction confidence and patient diversity, not causes of model failure.'}
    (OUT / 'error_cases_index.json').write_text(json.dumps(case_summary, indent=2), encoding='utf-8')

    print('\n===== LOCKED V7 FINAL TEST: PHASE 2 =====', flush=True)
    print('Checkpoint epoch:', checkpoint['epoch'], flush=True)
    print('Slice accuracy:', f"{100*metrics['slice_accuracy']:.2f}%", flush=True)
    print('Slice Macro-F1:', f"{metrics['slice_macro_f1']:.4f}", flush=True)
    print('Slice AUROC:', f'{auc:.4f}', flush=True)
    print('Patient accuracy:', f'{patient_acc*100:.2f}%', flush=True)
    print('Patient AUROC:', f'{patient_auc:.4f}', flush=True)
    print('NC precision, recall, F1:', *(round(x, 4) for x in (precision_nc, recall_nc, f1_nc)), flush=True)
    print('AD precision, recall, F1:', *(round(x, 4) for x in (precision_ad, recall_ad, f1_ad)), flush=True)
    print('ECE (10 bins):', f'{ece:.4f}', flush=True)
    print('Confusion [TN FP FN TP]:', [tn, fp, fn, tp], flush=True)
    print('Distinct-patient FP/FN examples:', len(fps), len(fns), flush=True)
    print('Saved figures and raw metrics to:', OUT, flush=True)


if __name__ == '__main__':
    main()
