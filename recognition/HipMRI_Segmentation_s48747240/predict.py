"""
predict.py  (3D HipMRI)
 
Usage: python predict.py [baseline3d|improved3d]
 
Outputs (written to predictions_<model>/):
  * Per-class Dice and IoU on every test volume (printed table).
  * Axial-slice figures for the weakest class: 3 worst volumes, one partial
    overlap (Dice closest to 0.4) and the best volume.
  * zprofile_<model>.png: per-slice Dice along the z axis for each foreground
    class, to show where errors concentrate near the ends of a structure
    (base/apex slice-transition errors). If the other model's profile has
    already been saved, it is overlaid for comparison.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
 
from dataset import load_hipmri_3d, NUM_CLASSES
 
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "improved3d"
CKPT = {"baseline3d": "unet3d", "improved3d": "improved3d"}[MODEL_NAME]
OUTPUT_DIR = f"predictions_{MODEL_NAME}"
MIN_VOLUMES_PER_SLICE = 3  # hide z-profile points supported by fewer volumes
 
 
def dice(true_lab, pred_lab, c, smooth=1e-6):
    t, p = true_lab == c, pred_lab == c
    return (2.0 * np.sum(t & p) + smooth) / (t.sum() + p.sum() + smooth)
 
 
def iou(true_lab, pred_lab, c, smooth=1e-6):
    """Intersection over union for class c."""
    t, p = true_lab == c, pred_lab == c
    inter = np.sum(t & p)
    return (inter + smooth) / (t.sum() + p.sum() - inter + smooth)
 
 
def slice_dice_profile(true_lab, pred_lab, c):
    """
    Dice per axial slice (last axis) for class c. Slices where neither the
    ground truth nor the prediction contains the class are NaN, so a slice
    where the model predicts the class but the ground truth has none scores 0
    (this captures leakage beyond the ends of a structure).
    """
    t, p = true_lab == c, pred_lab == c
    inter = (t & p).sum(axis=(0, 1)).astype(np.float64)
    denom = t.sum(axis=(0, 1)) + p.sum(axis=(0, 1))
    out = np.full(denom.shape, np.nan)
    ok = denom > 0
    out[ok] = 2.0 * inter[ok] / denom[ok]
    return out
 
 
def show(image, true_lab, pred_lab, name, k, cls, d, out_dir):
    z = int(np.round(np.argwhere(true_lab == cls)[:, 2].mean()))  # axial slice through GT centroid
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.6))
    ax[0].imshow(image[:, :, z], cmap="gray"); ax[0].set_title("Input (axial)")
    ax[1].imshow(true_lab[:, :, z], cmap="tab10", vmin=0, vmax=NUM_CLASSES - 1); ax[1].set_title("Ground truth")
    ax[2].imshow(pred_lab[:, :, z], cmap="tab10", vmin=0, vmax=NUM_CLASSES - 1); ax[2].set_title("Prediction")
    ax[3].imshow((true_lab[:, :, z] != pred_lab[:, :, z]), cmap="Reds"); ax[3].set_title("Difference")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"{name}  slice z={z}  class {cls} Dice={d:.2f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"vol_{k:02d}_{name.split('_LFOV')[0]}.png"))
    plt.close(fig)
 
 
def z_profile_figure(trues, preds):
    """Mean per-slice Dice along z for classes 1..5; overlay the other model if saved."""
    depth = trues[0].shape[2]
    mean_prof = np.full((NUM_CLASSES, depth), np.nan)
    std_prof = np.full((NUM_CLASSES, depth), np.nan)
    n_prof = np.zeros((NUM_CLASSES, depth), dtype=int)
    for c in range(1, NUM_CLASSES):
        stack = np.stack([slice_dice_profile(t, p, c) for t, p in zip(trues, preds)])  # (vols, depth)
        n = np.sum(~np.isnan(stack), axis=0)
        n_prof[c] = n
        with np.errstate(all="ignore"):
            m, s = np.nanmean(stack, axis=0), np.nanstd(stack, axis=0)
        keep = n >= MIN_VOLUMES_PER_SLICE
        mean_prof[c, keep], std_prof[c, keep] = m[keep], s[keep]
    np.savez(os.path.join(OUTPUT_DIR, f"zprofile_{MODEL_NAME}.npz"),
             mean=mean_prof, std=std_prof, n=n_prof)
 
    other = "baseline3d" if MODEL_NAME == "improved3d" else "improved3d"
    other_path = os.path.join(f"predictions_{other}", f"zprofile_{other}.npz")
    other_prof = np.load(other_path) if os.path.exists(other_path) else None
 
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
    z = np.arange(depth)
    for c, a in zip(range(1, NUM_CLASSES), axes.flat):
        a.plot(z, mean_prof[c], label=MODEL_NAME)
        a.fill_between(z, mean_prof[c] - std_prof[c], mean_prof[c] + std_prof[c], alpha=0.2)
        if other_prof is not None:
            a.plot(z, other_prof["mean"][c], "--", label=other)
        a.set_title(f"Class {c}")
        a.set_xlabel("axial slice index (z)")
        a.set_ylabel("per-slice Dice")
        a.set_ylim(0, 1.05)
        a.legend(fontsize=8)
    axes.flat[-1].axis("off")
    fig.suptitle(f"Per-slice Dice along z (mean +/- std over test volumes, "
                 f"slices with >= {MIN_VOLUMES_PER_SLICE} volumes)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"zprofile_{MODEL_NAME}.png"))
    plt.close(fig)
 
 
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model = tf.keras.models.load_model(os.path.join("checkpoints", f"{CKPT}_best.keras"), compile=False)
    _, _, test_ds, names = load_hipmri_3d(1)
 
    imgs, trues, preds = [], [], []
    for x, y in test_ds:
        p = model(x, training=False).numpy()[0]
        imgs.append(x.numpy()[0, ..., 0])
        trues.append(np.argmax(y.numpy()[0], -1))
        preds.append(np.argmax(p, -1))
 
    D = np.array([[dice(t, p, c) for c in range(NUM_CLASSES)] for t, p in zip(trues, preds)])
    J = np.array([[iou(t, p, c) for c in range(NUM_CLASSES)] for t, p in zip(trues, preds)])
    present = np.array([[np.any(t == c) for c in range(NUM_CLASSES)] for t in trues])
 
    print("\n--- Per-class Dice and IoU, test volumes ---")
    for c in range(NUM_CLASSES):
        if present[:, c].any():
            m = present[:, c]
            print(f"  Class {c}: Dice {D[m, c].mean():.4f}  IoU {J[m, c].mean():.4f} "
                  f"(present in {m.sum()}/{len(D)} volumes; all-volume Dice {D[:, c].mean():.4f})")
    fg_d = [D[present[:, c], c].mean() for c in range(1, NUM_CLASSES) if present[:, c].any()]
    fg_j = [J[present[:, c], c].mean() for c in range(1, NUM_CLASSES) if present[:, c].any()]
    print(f"  Mean foreground Dice (classes 1-5): {np.mean(fg_d):.4f}")
    print(f"  Mean foreground IoU  (classes 1-5): {np.mean(fg_j):.4f}")
 
    z_profile_figure(trues, preds)
    print(f"\nSaved per-slice Dice profile to {OUTPUT_DIR}/zprofile_{MODEL_NAME}.png")
 
    cls = min((c for c in range(1, NUM_CLASSES) if present[:, c].any()),
              key=lambda c: D[present[:, c], c].mean())
    idx = np.where(present[:, cls])[0]
    order = idx[np.argsort(D[idx, cls])]
    chosen = list(order[:3])
    partial = idx[np.argmin(np.abs(D[idx, cls] - 0.4))]
    for extra in (partial, order[-1]):
        if extra not in chosen:
            chosen.append(extra)
    print(f"\nWeakest class: {cls}. Figures for volumes: {[names[i] for i in chosen]}")
    for k, i in enumerate(chosen):
        print(f"  {names[i]}: class {cls} Dice={D[i, cls]:.2f}  IoU={J[i, cls]:.2f}")
        show(imgs[i], trues[i], preds[i], names[i], k, cls, D[i, cls], OUTPUT_DIR)
 
 
if __name__ == "__main__":
    main()
