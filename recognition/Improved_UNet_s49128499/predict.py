"""
predict.py, used to evaluate the trained basic 2D U-Net, and Improved 2D U-Net

Produces the following:
per-class Dice, precision, recall, prostate HD95 in mm, comparison figures, and failure case autopsy panels

Usage:
python predict.py --model unet2d
python predict.py --model improvedunet2d
"""
import argparse
import os
import glob

import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import label as cc_label, binary_erosion, distance_transform_edt
import time

from modules import UNet2D, ImprovedUNet2D
from dataset import HipMRIDataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "keras_slices_data")

PROSTATE = 5
CLASS_NAMES = ["Background", "Body", "Bone", "Bladder", "Rectum", "Prostate"]

# per class metrics
def per_class_metrics(pred, gt, num_classes=6, eps=1e-6):
    """
    returns dict of arrays of dice, prec, and recall
    """
    dice = np.zeros(num_classes)
    prec = np.zeros(num_classes)
    rec = np.zeros(num_classes)
    iou  = np.zeros(num_classes)

    for c in range(num_classes):
        p = (pred == c)
        t = (gt == c)
        tp = (p & t).sum()
        fp = (p & ~t).sum()
        fn = (~p & t).sum()
        dice[c] = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        prec[c] = (tp + eps) / (tp + fp + eps)
        rec[c] = (tp + eps) / (tp + fn + eps)
        iou[c]  = (tp + eps) / (tp + fp + fn + eps)

    return {"dice": dice, "precision": prec, "recall": rec, "iou": iou}

# hd95 function
def hd95_2d(pred_mask, gt_mask, spacing=(1.0, 1.0)):
    """
    returns HD95 in mm between 2 binary masks, returns NaN if either is empty
    uses symmetric surface distances, 95th percentile
    """
    if pred_mask.sum() == 0 or gt_mask.sum() == 0:
        return float("nan")

    # extract surface via binary erosion
    pred_surf = pred_mask ^ binary_erosion(pred_mask)
    gt_surf = gt_mask ^ binary_erosion(gt_mask)

    # distance from each surface point to other surface points
    dt_to_gt = distance_transform_edt(~gt_surf, sampling=spacing)
    dt_to_pred = distance_transform_edt(~pred_surf, sampling=spacing)

    d_pred_to_gt = dt_to_gt[pred_surf]
    d_gt_to_pred = dt_to_pred[gt_surf]

    all_d = np.concatenate([d_pred_to_gt, d_gt_to_pred])

    return float(np.percentile(all_d, 95))

# fragment count function
def count_fragments(mask, min_size=50):
    """
    counts the number of connected components larger than the min size
    """
    labeled, n = cc_label(mask)
    if n == 0:
        return 0
    sizes = np.bincount(labeled.ravel())
    return int((sizes[1:] > min_size).sum())

# parameter counting function
def count_parameters(model):
    """
    counts how many learnable numbers are in the model
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# latency measuring function
@torch.no_grad()
def measure_latency(model, test_dataset, device, n_warmup=5, n_measure=50):
    """
    measures how long the model takes to segment a slice. Timing starts after 5 passes (n_warmup)
    and times 50 passes (n_measure). Median over mean as mean is sensitive to outliers. P95 for the worst latency
    95% of slices experience
    """
    model.eval()
    for i in range(n_warmup):
        img, _ = test_dataset[i % len(test_dataset)]
        _ = model(img.unsqueeze(0).to(device))
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for i in range(n_measure):
        img, _ = test_dataset[i % len(test_dataset)]
        img = img.unsqueeze(0).to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(img)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    times = np.array(times)
    return {"median": np.median(times), "p95": np.percentile(times, 95)}


# evaluation function
@torch.no_grad()
def evaluate(model, test_dataset, device):
    model.eval()

    TP = FP = FN = 0
    all_dice, all_prec, all_rec, all_iou = [], [], [], []
    hd95_values, fragment_counts = [], []

    for idx in range(len(test_dataset)):
        img, label = test_dataset[idx]
        logits = model(img.unsqueeze(0).to(device))
        pred = logits.argmax(1).squeeze(0).cpu().numpy()
        gt = label.argmax(0).numpy()

        m = per_class_metrics(pred,gt)
        all_dice.append(m["dice"])
        all_prec.append(m["precision"])
        all_rec.append(m["recall"])
        all_iou.append(m["iou"])

        p = (pred == PROSTATE)
        t = (gt == PROSTATE)
        TP += (p & t).sum()
        FP += (p & ~t).sum()
        FN += (~p & t).sum()

        if t.sum() > 0:
            hd95_values.append(hd95_2d(p, t))
            fragment_counts.append(count_fragments(p, min_size=50))

    mean_dice = np.array(all_dice).mean(axis=0)
    mean_prec = np.array(all_prec).mean(axis=0)
    mean_rec = np.array(all_rec).mean(axis=0)
    mean_iou = np.array(all_iou).mean(axis=0)

    micro_prec = TP / (TP + FP) if (TP + FP) > 0 else float("nan")
    micro_rec = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
    valid_hd = [h for h in hd95_values if not np.isnan(h)]
    mean_hd95 = float(np.mean(valid_hd)) if valid_hd else float("nan")

    print("\n=== Test results ===")
    print(f"{'Class':<12} {'Dice':>8} {'Prec':>8} {'Rec':>8} {'IoU':>8}")
    for c in range(6):
        print(f"{CLASS_NAMES[c]:<12} {mean_dice[c]:>8.4f} {mean_prec[c]:>8.4f} {mean_rec[c]:>8.4f} {mean_iou[c]:>8.4f}")

    print(f"\nProstate micro: precision={micro_prec:.4f}  recall={micro_rec:.4f}")
    print(f"Prostate HD95: {mean_hd95:.2f} mm")
    print(f"Mean fragments >50px: {np.mean(fragment_counts):.2f}" if fragment_counts else "Mean fragments: n/a")

    return {"dice": mean_dice, "prec": mean_prec, "rec": mean_rec, "iou": mean_iou,
            "micro_prec": micro_prec, "micro_rec": micro_rec, "hd95": mean_hd95}

# plots (4 panels)
@torch.no_grad()
def plot_panel(model, test_ds, idx, device, save_path):
    img, mask = test_ds[idx]
    logits = model(img.unsqueeze(0).to(device))
    pred = logits.argmax(1).squeeze(0).cpu().numpy()
    gt = mask.argmax(0).numpy()
    image = img.squeeze(0).numpy()

    gt_p = (gt == PROSTATE)
    pred_p = (pred == PROSTATE)
    tp = gt_p & pred_p
    fp = ~gt_p & pred_p
    fn = gt_p & ~pred_p

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image, cmap="gray"); axes[0].set_title("Input")
    axes[1].imshow(gt_p, cmap="gray"); axes[1].set_title("GT prostate")
    axes[2].imshow(pred_p, cmap="gray"); axes[2].set_title("Pred prostate")
    rgb = np.zeros((*gt_p.shape, 3), dtype=np.float32)
    rgb[..., 0] = fp.astype(np.float32)
    rgb[..., 1] = tp.astype(np.float32)
    rgb[..., 2] = fn.astype(np.float32)

    axes[3].imshow(image, cmap="gray")
    axes[3].imshow(rgb, alpha=0.8)
    axes[3].set_title(f"TP={tp.sum()} | FP={fp.sum()} | FN={fn.sum()}")
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

@torch.no_grad()
def find_failure_cases(model, test_ds, device, top_k=10):
    """
    Score every test slice by FP (leakage) and FN (under-contouring),
    return the worst cases for each failure mode.
    """
    model.eval()
    results = []

    for idx in range(len(test_ds)):
        img, mask = test_ds[idx]
        logits = model(img.unsqueeze(0).to(device))
        pred = logits.argmax(1).squeeze(0).cpu().numpy()
        gt = mask.argmax(0).numpy()

        p = (pred == PROSTATE)
        t = (gt == PROSTATE)

        tp = (p & t).sum()
        fp = (p & ~t).sum()   # leakage
        fn = (~p & t).sum()   # under-contouring

        # skip slices with no GT prostate (can't diagnose boundary failure)
        if t.sum() == 0:
            continue

        dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
        results.append({
            "idx": idx,
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "dice": float(dice),
            "gt_size": int(t.sum()),
        })

    # Rank by failure type
    by_fp = sorted(results, key=lambda r: r["fp"], reverse=True)   # worst leakage
    by_fn = sorted(results, key=lambda r: r["fn"], reverse=True)   # worst under-contouring
    by_dice = sorted(results, key=lambda r: r["dice"])             # worst overall

    print("\n=== Top 5 leakage cases (highest FP) ===")
    for r in by_fp[:5]:
        print(f"  idx={r['idx']}  FP={r['fp']}  FN={r['fn']}  "
              f"GT={r['gt_size']}  Dice={r['dice']:.3f}")

    print("\n=== Top 5 under-contouring cases (highest FN) ===")
    for r in by_fn[:5]:
        print(f"  idx={r['idx']}  FP={r['fp']}  FN={r['fn']}  "
              f"GT={r['gt_size']}  Dice={r['dice']:.3f}")

    print("\n=== Top 5 worst Dice cases ===")
    for r in by_dice[:5]:
        print(f"  idx={r['idx']}  FP={r['fp']}  FN={r['fn']}  "
              f"GT={r['gt_size']}  Dice={r['dice']:.3f}")

    return by_fp, by_fn, by_dice

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["unet2d", "improvedunet2d"], default="improvedunet2d")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--n-panels", type=int, default=5)
    parser.add_argument("--out-dir", default=os.path.join(SCRIPT_DIR, "images"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # load model
    if args.model == "unet2d":
        model = UNet2D(in_channels=1, num_classes=6).to(device)
        chk_path = os.path.join(SCRIPT_DIR, "models", "best_unet2d.pt")
    else:
        model = ImprovedUNet2D(in_channels=1, num_classes=6).to(device)
        chk_path = os.path.join(SCRIPT_DIR, "models", "best_improvedunet2d.pt")

    
    chk = torch.load(chk_path, map_location=device)
    model.load_state_dict(chk["model_state"])

    print(f"Loaded {chk_path}")
    print(f"Epoch: {chk.get('epoch', '?')}")
    print(f"Val Dice {chk.get('val_dice', float('nan')):.4f}")

    # load test dataset
    test_dataset = HipMRIDataset(
        os.path.join(args.data_dir, "keras_slices_test"),
        os.path.join(args.data_dir, "keras_slices_seg_test"),
        transform=None,
    )
    print(f"Test slices: {len(test_dataset)}")

    # latency
    lat = measure_latency(model, test_dataset, device)
    print(f"Latency (256x128): median={lat['median']:.2f} ms, p95={lat['p95']:.2f} ms")

    # count parameters
    total_param, trainable_param = count_parameters(model)
    print(f"Total Parameters: {total_param}")
    print(f"Trainable Parameters: {trainable_param}")
    # evaluate
    results = evaluate(model, test_dataset, device)

    # find all test slices that contain prostate
    # prostate_indices = []
    # for idx in range(len(test_dataset)):
    #     _, mask = test_dataset[idx]
    #     gt = mask.argmax(0).numpy()
    #     if (gt == PROSTATE).sum() > 0:
    #         prostate_indices.append(idx)

    # print(f"Slices containing prostate: {len(prostate_indices)} / {len(test_dataset)}")

    # sample evenly from slices that contain prostate
    # n = min(args.n_panels, len(prostate_indices))
    # step = max(1, len(prostate_indices) // n)
    # selected = [prostate_indices[i * step] for i in range(n)]
    # for i, idx in enumerate(selected):
    #     path = os.path.join(args.out_dir, f"{args.model}_panel_{i}.png")
    #     plot_panel(model, test_dataset, idx, device, path)
    # print(f"Saved {n} panels to {args.out_dir}/")

    by_fp, by_fn, _ = find_failure_cases(model, test_dataset, device)

    case_indices = [by_fp[0]["idx"], by_fp[1]["idx"], by_fn[0]["idx"]]
    for i, idx in enumerate(case_indices):
        path = os.path.join(args.out_dir, f"autopsy_case_{i+1}_idx{idx}.png")
        plot_panel(model, test_dataset, idx, device, path)

if __name__ == "__main__":
    main()