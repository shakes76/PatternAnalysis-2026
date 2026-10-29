"""
Evaluation helpers: classification metrics, calibration and selective prediction.

Used by train.py (end-of-training test report) and evaluate.py (the open-dilemma
analysis comparing ConvNeXt against the baseline).
"""

import numpy as np
import torch
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve,
)

from dataset import CLASSES, scan_id


def classification_report(probs, labels):
    """Metrics for binary AD-vs-NC probabilities ``probs`` [N, 2] and integer ``labels``.

    Returns a flat dict: accuracy, per-class precision/recall/F1, macro-F1, AUROC,
    sensitivity/specificity (AD = positive class), confusion matrix, ECE and NLL.
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    pred = probs.argmax(1)  # predicted class = the more probable one (threshold 0.5)
    # per-class precision / recall / F1, index 0 = NC, index 1 = AD
    p, r, f, _ = precision_recall_fscore_support(labels, pred, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(labels, pred, labels=[0, 1])  # rows = true class, cols = predicted
    tn, fp, fn, tp = cm.ravel()  # AD is the "positive" class
    out = {
        "accuracy": float((pred == labels).mean()),
        "macro_f1": float(f.mean()),
        # AUROC uses the continuous P(AD), so it measures ranking quality independent of threshold
        "auroc": float(roc_auc_score(labels, probs[:, 1])),
        "sensitivity": float(tp / max(tp + fn, 1)),  # fraction of AD cases caught (= AD recall)
        "specificity": float(tn / max(tn + fp, 1)),  # fraction of NC cases correctly cleared
        "ece": expected_calibration_error(probs, labels),
        # negative log-likelihood of the true class: penalises confident mistakes heavily
        "nll": float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1)).mean()),
        # Brier score: mean squared error between P(AD) and the 0/1 label (0 = perfect;
        # unlike accuracy it rewards well-judged confidence, unlike NLL it is bounded)
        "brier": float(((probs[:, 1] - labels) ** 2).mean()),
        "confusion": cm.tolist(),
    }
    for i, c in enumerate(CLASSES):
        out[f"precision_{c}"] = float(p[i])
        out[f"recall_{c}"] = float(r[i])
        out[f"f1_{c}"] = float(f[i])
    return out


def reliability_bins(probs, labels, n_bins=15):
    """Bin by top-class confidence; returns (bin_conf, bin_acc, bin_count, edges).

    For a well-calibrated model, the mean confidence of each bin equals the
    fraction of correct predictions in that bin (e.g. predictions made with 80%
    confidence are right 80% of the time).
    """
    conf = probs.max(1)
    correct = probs.argmax(1) == labels
    edges = np.linspace(0.5, 1.0, n_bins + 1)  # binary: confidence is always >= 0.5
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    bc, ba, bn = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        m = idx == b
        bn[b] = m.sum()
        if bn[b]:
            bc[b], ba[b] = conf[m].mean(), correct[m].mean()
    return bc, ba, bn, edges


def expected_calibration_error(probs, labels, n_bins=15):
    """ECE = sum_b (n_b / N) * |accuracy_b - confidence_b|: the average gap between how
    sure the model is and how often it is right, weighted by how many samples fall in each bin."""
    bc, ba, bn, _ = reliability_bins(np.asarray(probs), np.asarray(labels), n_bins)
    return float((bn / bn.sum() * np.abs(bc - ba)).sum())


def risk_coverage(probs, labels):
    """Selective prediction curve: sort by confidence, accept the top fraction.

    Returns (coverage, accuracy_on_accepted, threshold) arrays, one point per sample.
    """
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(float)
    order = np.argsort(-conf, kind="stable")  # most confident first
    # If we automate the k most confident cases, accuracy on them is cum[k-1] / k.
    cum = np.cumsum(correct[order])
    n = np.arange(1, len(conf) + 1)
    # coverage = fraction automated, accuracy on those, and the confidence cut-off reached
    return n / len(conf), cum / n, conf[order]


def threshold_for_accuracy(probs, labels, target):
    """Lowest confidence threshold whose accepted set reaches ``target`` accuracy.

    Picks the largest coverage that satisfies the target, i.e. automates as many
    cases as possible while keeping accepted-case accuracy >= target.
    """
    cov, acc, thr = risk_coverage(probs, labels)
    ok = np.where(acc >= target)[0]  # all operating points that meet the accuracy target
    if len(ok) == 0:
        return 1.0, 0.0
    i = ok.max()
    return float(thr[i]), float(cov[i])


def apply_threshold(probs, labels, thr):
    """Accuracy / coverage / missed-AD count when cases with confidence < thr are referred."""
    conf = probs.max(1)
    pred = probs.argmax(1)
    keep = conf >= thr  # True = automated, False = referred to a radiologist
    acc = float((pred[keep] == labels[keep]).mean()) if keep.any() else float("nan")
    # the dangerous error: AD predicted as NC *and* not referred, so no human sees it
    missed_ad = int(((pred == 0) & (labels == 1) & keep).sum())
    return {"threshold": float(thr), "coverage": float(keep.mean()), "accepted_accuracy": acc,
            "referred": int((~keep).sum()), "auto_false_negatives": missed_ad}


def fit_temperature(logits, labels, iters=200):
    """Temperature scaling (Guo et al., 2017): single scalar T minimising validation NLL.

    Calibrated probabilities are softmax(logits / T). T > 1 softens over-confident
    predictions, T < 1 sharpens under-confident ones. Dividing all logits by the same
    positive T never changes the argmax, so accuracy and AUROC are unchanged;
    only the confidence values move.
    """
    logits = torch.as_tensor(logits, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)  # optimise log T so T = exp(log_t) stays positive
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=iters)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def softmax_np(logits, T=1.0):
    z = np.asarray(logits, dtype=np.float64) / T
    z = z - z.max(1, keepdims=True)  # subtract the max for numerical stability (exp cannot overflow)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _group_mean(probs, labels, keys):
    """Average probabilities over all slices sharing a key -> (probs, labels, keys)."""
    groups = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    ids = sorted(groups)
    return (np.stack([probs[groups[k]].mean(0) for k in ids]),
            np.array([labels[groups[k][0]] for k in ids]), ids)


def patient_level(probs, labels, paths, patient_of):
    """Average every slice of every scan of each patient -> one decision per patient.

    This is the unit of acceptance criterion 1 (>= 0.80 patient-level accuracy).
    """
    return _group_mean(probs, labels, [patient_of[scan_id(p)] for p in paths])


def scan_level(probs, labels, paths):
    """Average the 20 slice probabilities of each MRI scan -> (scan_probs, scan_labels, scan_ids).

    A scan (one visit) is the unit a radiologist would actually read, so this is the
    clinically relevant decision level.
    """
    groups = {}
    for i, p in enumerate(paths):
        groups.setdefault(scan_id(p), []).append(i)
    ids = sorted(groups)
    # averaging probabilities (soft voting) keeps a confidence value per scan
    sp = np.stack([probs[groups[s]].mean(0) for s in ids])
    sl = np.array([labels[groups[s][0]] for s in ids])
    return sp, sl, ids


def roc_points(probs, labels):
    fpr, tpr, _ = roc_curve(labels, probs[:, 1])
    return fpr, tpr
