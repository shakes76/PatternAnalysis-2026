"""
Shared helpers for train.py and predict.py: seeding, volume prediction,
hard Dice/IoU metrics, per-patient summaries and training curves.
"""

import json
import math
import random

import numpy as np
import torch

import torch.nn.functional as F

from dataset import LABEL_NAMES, NUM_CLASSES, downsample, load_case, patient_of


def seed_everything(seed):
    """Fix all random seeds so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict_volume(model, image, mode, device, batch_size=32, amp=True, scale=2):
    """Predict a full-resolution class map (D, H, W) for one volume.

    mode "2d": run the model slice by slice along D and stack the results.
    mode "3d": downsample the volume by `scale`, run the model once, then
    upsample the class scores back to full size before taking the argmax.
    Either way, 2D and 3D models are scored on identical full-size volumes.
    """
    model.eval()
    use_amp = amp and device.type == "cuda"
    if mode == "2d":
        preds = []
        for start in range(0, image.shape[0], batch_size):
            x = image[start:start + batch_size, None].to(device)          # (b, 1, H, W)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
            preds.append(logits.argmax(dim=1).cpu())
        return torch.cat(preds)
    if mode == "3d":
        small, _ = downsample(image, None, scale)
        x = small[None, None].to(device)                                   # (1, 1, d, h, w)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
        logits = F.interpolate(logits.float(), size=tuple(image.shape), mode="trilinear", align_corners=False)
        return logits.argmax(dim=1)[0].cpu()
    raise ValueError(f"Unsupported mode '{mode}'")


def dice_iou_per_class(pred, target, num_classes=NUM_CLASSES):
    """Hard (argmax) Dice and IoU for each class.

    Returns two lists of floats; a class absent from both prediction and
    ground truth gives NaN and is left out of averages.
    """
    dice, iou = [], []
    for c in range(num_classes):
        p, t = pred == c, target == c
        inter = (p & t).sum().item()
        size = p.sum().item() + t.sum().item()
        if size == 0:
            dice.append(math.nan)
            iou.append(math.nan)
        else:
            dice.append(2 * inter / size)
            iou.append(inter / (size - inter))
    return dice, iou


def evaluate_cases(model, mode, cache_dir, case_ids, device, batch_size=32, amp=True, scale=2):
    """Per-case Dice/IoU for every case, computed at full resolution."""
    results = {}
    for cid in case_ids:
        image, label = load_case(cache_dir, cid)
        pred = predict_volume(model, image, mode, device, batch_size, amp, scale)
        dice, iou = dice_iou_per_class(pred, label)
        results[cid] = {"dice": dice, "iou": iou}
    return results


def summarise(results):
    """Average each patient's scans first, then report mean and std across
    patients, so patients with 8 scans don't outweigh those with 1."""
    by_patient = {}
    for cid, r in results.items():
        by_patient.setdefault(patient_of(cid), []).append(r)

    summary = {}
    for metric in ("dice", "iou"):
        per_patient = np.array([np.nanmean([r[metric] for r in rs], axis=0)
                                for rs in by_patient.values()])          # (patients, classes)
        summary[metric] = {
            name: {"mean": float(np.nanmean(per_patient[:, c])),
                   "std": float(np.nanstd(per_patient[:, c]))}
            for c, name in enumerate(LABEL_NAMES)
        }
    summary["prostate_dice"] = summary["dice"]["prostate"]["mean"]
    summary["mean_foreground_dice"] = float(np.mean(
        [summary["dice"][n]["mean"] for n in LABEL_NAMES[1:]]))
    summary["n_patients"] = len(by_patient)
    summary["n_cases"] = len(results)
    return summary


def print_summary(summary, title):
    print(f"\n{title}  ({summary['n_patients']} patients, {summary['n_cases']} scans)")
    print(f"{'class':<11}{'Dice':>16}{'IoU':>16}")
    for name in LABEL_NAMES:
        d, i = summary["dice"][name], summary["iou"][name]
        print(f"{name:<11}{d['mean']:>9.3f} ± {d['std']:.3f}{i['mean']:>9.3f} ± {i['std']:.3f}")


def plot_history(history, path):
    """Training loss and validation Dice per epoch, saved as a PNG for the README."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, [h["train_loss"] for h in history], marker="o")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("training loss (CE + Dice)")
    ax1.set_title("Training loss")
    ax2.plot(epochs, [h["val_prostate_dice"] for h in history], marker="o", label="prostate")
    ax2.plot(epochs, [h["val_mean_fg_dice"] for h in history], marker="o", label="mean of 5 organs")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("validation Dice")
    ax2.set_ylim(0, 1)
    ax2.set_title("Validation Dice (per-patient mean)")
    ax2.legend()
    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
