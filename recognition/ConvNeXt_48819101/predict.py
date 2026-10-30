import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
from collections import defaultdict
import time

from modules import ResNet
from modules import ConvNeXt
from dataset import get_dataloaders
device = "cuda" if torch.cuda.is_available() else "cpu"

def _metrics(probs, labels, cutoff):
    pred = (probs >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()

    ad_recall = tp / max(tp + fn, 1)
    nc_recall = tn / max(tn + fp, 1)
    ad_prec = tp / max(tp + fp, 1)      
    nc_prec = tn / max(tn + fn, 1)     
    ad_f1 = 2 * ad_prec * ad_recall / max(ad_prec + ad_recall, 1e-8)
    nc_f1 = 2 * nc_prec * nc_recall / max(nc_prec + nc_recall, 1e-8)

    return {"acc": (tp + tn) / len(labels),
            "ad_recall": ad_recall, "nc_recall": nc_recall,
            "ad_precision": ad_prec, "nc_precision": nc_prec,
            "ad_f1": ad_f1, "nc_f1": nc_f1,
            "macro_f1": (ad_f1 + nc_f1) / 2,
            "auc": roc_auc_score(labels, probs),
            "fn_rate": fn / max(tp + fn, 1)}

def plot_confusion(cm, title, path):
    names = ["NC", "AD"]
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)   # row-normalised = recall
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]:.1%})", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(names)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def patient_rule(pids, probs, labels, slice_cutoff=0.45, ad_frac=0.30):
    """
    Creates a rule for patient level splitting where if at least 
    ad_frac of the patient's images are flagged for AD they are 
    classified as having AD. If at least 10% of them have been flagged,
    they are sent to doctor's review.
    """
    groups = defaultdict(list)
    for p, pr, lb in zip(pids, probs, labels):
        groups[p].append((pr, lb))

    decisions, counts = {}, defaultdict(int)
    for pid, v in groups.items():
        slice_probs = np.array([a for a, _ in v])
        label = int(v[0][1])
        frac = float((slice_probs >= slice_cutoff).mean())
        if frac > ad_frac:
            d = "AD"
        elif frac > 0.1:
            d = "review"
        else:
            d = "NC"
        decisions[pid] = (d, frac, label)
        counts[(label, d)] += 1

    n_ad = sum(1 for _, _, l in decisions.values() if l == 1)
    n_nc = len(decisions) - n_ad
    return {
        "decisions": decisions,
        "ad_called_AD": counts[(1, "AD")] / max(n_ad, 1),
        "ad_called_review": counts[(1, "review")] / max(n_ad, 1),
        "ad_missed_NC": counts[(1, "NC")] / max(n_ad, 1),
        "nc_called_NC": counts[(0, "NC")] / max(n_nc, 1),
        "nc_called_review": counts[(0, "review")] / max(n_nc, 1),
        "nc_called_AD": counts[(0, "AD")] / max(n_nc, 1),
        "n_ad": n_ad, "n_nc": n_nc,
    }

@torch.no_grad()
def evaluate(model, data, criterion=None, cutoff=0.5, plot=False):
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()
    
    model.eval()
    probs = []
    labels = []
    pids = []
    loss_sum, n, offset = 0.0, 0, 0
    for batch in data:
        x, y = batch[0].to(device), batch[1].to(device)


        if len(batch) > 2:
            pid = [p.item() if torch.is_tensor(p) else p for p in batch[2]]
        else:
            pid = data.dataset.patient_ids[offset: offset + len(y)]
        offset += len(y)

        logits = model(x).squeeze(1)
        loss = criterion(logits, y.float())

        loss_sum += loss.item() * y.size(0)
        n += y.size(0)
        probs.append(torch.sigmoid(logits.float()).cpu())
        labels.append(y.cpu())
        pids.extend(pid)
    probs, labels = torch.cat(probs).numpy(), torch.cat(labels).numpy()

    metrics = _metrics(probs, labels, cutoff)

    groups = defaultdict(list)
    for p, pr, lb in zip(pids, probs, labels):
        groups[p].append((pr, lb))
    pat_probs = np.array([np.mean([a for a, _ in v]) for v in groups.values()])
    pat_labels = np.array([v[0][1] for v in groups.values()])
    pat_metrics = _metrics(pat_probs, pat_labels, cutoff)
    pat_metrics["n"] = len(pat_labels)
    cm = confusion_matrix(labels, (probs >= cutoff).astype(int), labels=[0, 1])
    pat_cm = confusion_matrix(pat_labels, (pat_probs >= cutoff).astype(int), labels=[0, 1])
    rule = patient_rule(pids, probs, labels, slice_cutoff=cutoff)
    print(f"---------------CUTOFF = {cutoff} ---------------------")
    if plot:
        plot_roc(probs, labels, np.arange(0.3, 0.6, 0.05), save_path="roc_resnet.png")
    return {"loss": loss_sum / n, "slice": metrics, "patient": pat_metrics, "rule": rule, "cm": cm, "patient_cm": pat_cm}


def plot_roc(probs, labels, cutoffs, save_path=None):
    """Plots AUC-ROC """
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")

    for c in cutoffs:
        pred = (probs >= c).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        tn = ((pred == 0) & (labels == 0)).sum()
        x, y = fp / max(fp + tn, 1), tp / max(tp + fn, 1)
        ax.scatter(x, y, s=40, zorder=3)
        ax.annotate(f"{c:.2f}", (x, y), textcoords="offset points",
                    xytext=(6, -10), fontsize=8)

    ax.set_xlabel("False positive rate (1 - nc_recall)")
    ax.set_ylabel("True positive rate (ad_recall)")
    ax.set_title("ROC curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

def main(ckpt="resnet50.pt"):
    plot = True
    _, _, test_loader = get_dataloaders()
    if ckpt == "resnet50.pt":
        model = ResNet(num_classes=1).to(device)
    else:
        model = ConvNeXt(num_classes=1).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)

    res = evaluate(model, test_loader)
    n_params = sum(t.numel() for t in state.values())
    print(f"{n_params / 1e6:.2f}M params")

    start_time = time.time()
    result = evaluate(model, test_loader, cutoff=0.45, plot=plot)
    print(f"Evaluate time: {time.time()-start_time:.4f}")
    r = evaluate(model, test_loader)["rule"]
    print(f"test loss: {result['loss']:.4f}")
    print("slice:  ", {k: round(float(v), 4) for k, v in result["slice"].items()})
    print("patient:", {k: round(float(v), 4) for k, v in result["patient"].items()})
    print({k: round(v, 3) for k, v in r.items() if k != "decisions"})
    plot_confusion(res["cm"], f"Test, slice level", f"cm_slice_resnet.png")
    plot_confusion(res["patient_cm"], f"Test, patient level", f"cm_patient_resnet.png")
    

if __name__ == "__main__":
    main(ckpt="resnet50.pt")