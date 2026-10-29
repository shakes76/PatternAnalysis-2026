"""
predict.py: one-time test-set evaluation of a trained model.

Author: Angus Slater (49615878), COMP3710 2026

Loads the best checkpoint (selected on validation prostate Dice in train.py) and
evaluates it once on the held-out test patients. The patient-level split is
rebuilt from --split_seed, so the test set is identical to the one held out
during training. The test set is never used for model selection.

For every test volume it records:
  - hard Dice and IoU (= Dice / (2 - Dice)) for all 6 classes
  - prostate HD95 in mm (using the post-downsampling voxel spacing)
  - prostate Dice by third (apex / mid / base)
  - inference time per volume (after one warm-up pass, timed with
    torch.cuda.synchronize) and peak inference VRAM
  - ground-truth and predicted prostate voxel counts

Outputs
  results/test_<run>.json            per-case rows and a mean / std summary
  figures/overlay_<run>_<case>.png   4-panel overlays (FP / FN error maps) for
                                     --cases, or the --n_overlays worst cases plus the best

Usage (on Rangpur, via predict.sh):
  sbatch predict.sh --model improved --tag v1 --cases K008_Week0 V027_Week7
"""

import argparse
import json
import os
import time

import nibabel as nib
import numpy as np
import torch

from dataset import ROOT, TARGET_SHAPE, HipMRI3D, list_pairs, split_by_patient
from train import MODELS, PROSTATE
from utils import CLASS_NAMES, binary_dice, dice_by_third, dice_per_class, hd95, plot_overlay


def parse_args():
    """Command-line settings: which trained run to evaluate."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=list(MODELS), required=True)
    p.add_argument("--tag", required=True, help="run tag used in training, e.g. v1")
    p.add_argument("--data_dir", default=ROOT, help="folder containing semantic_MRs/")
    p.add_argument("--split_seed", type=int, default=42, help="must match training")
    p.add_argument("--n_overlays", type=int, default=4, help="worst cases to plot (+ the best)")
    p.add_argument("--cases", nargs="*", default=None,
                help="specific cases to overlay, e.g. T014_Week7 K008_Week0 (default: worst + best)")
    return p.parse_args()


def downsampled_spacing(img_path):
    """Voxel size (mm) after load_volume's crop + resize, so HD95 is in real mm.

    Each axis is scaled by (cropped original size / target size), e.g. 256 -> 128 doubles it.
    """
    hdr = nib.load(img_path).header
    shape, zooms = hdr.get_data_shape(), hdr.get_zooms()
    cropped = (shape[0], shape[1], min(shape[2], 128))
    return tuple(float(z) * c / t for z, c, t in zip(zooms, cropped, TARGET_SHAPE))


def main():
    """Load the best checkpoint, evaluate every test volume once, save metrics and overlays."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = f"{args.model}_{args.tag}"
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    model = MODELS[args.model]().to(device)
    model.load_state_dict(torch.load(f"checkpoints/{run}.pt", map_location=device))
    model.eval()  # inference mode: dropout off

    # Same patient split as training; only the test patients are loaded.
    test_pairs = split_by_patient(list_pairs(args.data_dir), seed=args.split_seed)["test"]
    test_set = HipMRI3D(test_pairs)  # augment=False
    print(f"evaluating {run} on {len(test_set)} test volumes ({device})", flush=True)

    def infer(img):
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                             enabled=device.type == "cuda"):
            return model(img[None].to(device)).float()

    infer(test_set[0][0])  # warm-up, so the first timed volume isn't slowed by CUDA start-up
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    rows, preds = [], []
    for (img_path, _), (img, lab) in zip(test_pairs, test_set):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        logits = infer(img)
        if device.type == "cuda":
            torch.cuda.synchronize()  # GPU runs asynchronously: wait before stopping the clock
        ms = (time.time() - t0) * 1000

        dice = dice_per_class(logits, lab[None].to(device)).cpu().tolist()
        pred = logits.argmax(1)[0].cpu().numpy()
        gt = lab.numpy()
        p_pro, g_pro = pred == PROSTATE, gt == PROSTATE
        rows.append({
            "case": os.path.basename(img_path).replace("_LFOV.nii.gz", ""),
            "dice": dict(zip(CLASS_NAMES, dice)),
            "iou": {c: d / (2 - d) for c, d in zip(CLASS_NAMES, dice)}, #Dice / (2 - Dice) holds exactly for each binary mask pair
            "prostate_dice": binary_dice(p_pro, g_pro),
            "prostate_hd95_mm": hd95(p_pro, g_pro, downsampled_spacing(img_path)),
            "prostate_dice_thirds": dice_by_third(p_pro, g_pro),
            "infer_ms": ms,
            "prostate_gt_voxels": int(g_pro.sum()),
            "prostate_pred_voxels": int(p_pro.sum())  # > gt means over-segmentation
        })
        preds.append(pred)
        r = rows[-1]
        print(f"{r['case']:12s} prostate Dice {r['prostate_dice']:.3f} | HD95 "
              f"{r['prostate_hd95_mm']:.1f} mm | {ms:.0f} ms", flush=True)

    # Summary: mean and std over test volumes (nan-aware: undefined cases are skipped).
    def stat(values):
        v = np.array(values, dtype=float)
        return {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v)), "n": int(np.sum(~np.isnan(v)))}

    summary = {
        "run": run, "n_test_volumes": len(rows),
        "dice": {c: stat([r["dice"][c] for r in rows]) for c in CLASS_NAMES},
        "iou": {c: stat([r["iou"][c] for r in rows]) for c in CLASS_NAMES},
        "prostate_hd95_mm": stat([r["prostate_hd95_mm"] for r in rows]),
        "prostate_dice_thirds": {t: stat([r["prostate_dice_thirds"][t] for r in rows])
                                 for t in ("apex", "mid", "base")},
        "infer_ms": stat([r["infer_ms"] for r in rows]),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None,
    }
    with open(f"results/test_{run}.json", "w") as f:
        json.dump({"summary": summary, "per_volume": rows}, f, indent=1)

    # Overlays: the worst cases by prostate Dice (failure autopsy) plus the best for contrast.
    order = np.argsort([r["prostate_dice"] for r in rows])  # ascending: worst first
    if args.cases:
        picks = [i for i, r in enumerate(rows) if r["case"] in args.cases]
    else:
        picks = list(order[:args.n_overlays]) + [order[-1]]
    for i in picks:
        r = rows[i]
        title = (f"{run} | {r['case']} | prostate Dice {r['prostate_dice']:.3f}, "
                 f"HD95 {r['prostate_hd95_mm']:.1f} mm")
        plot_overlay(test_set[i][0][0].numpy(), test_set[i][1].numpy(), preds[i], title,
                     f"figures/overlay_{run}_{r['case']}.png")

    d, h, t = summary["dice"], summary["prostate_hd95_mm"], summary["prostate_dice_thirds"]
    print("\nTEST SET (mean ± std over volumes)")
    for c in CLASS_NAMES[1:]:
        print(f"  {c:9s} Dice {d[c]['mean']:.3f} ± {d[c]['std']:.3f} | "
            f"IoU {summary['iou'][c]['mean']:.3f} ± {summary['iou'][c]['std']:.3f}")
    print(f"  prostate HD95 {h['mean']:.1f} ± {h['std']:.1f} mm (n={h['n']})")
    print("  prostate Dice by third: " + ", ".join(f"{k} {v['mean']:.3f}" for k, v in t.items()))
    print(f"  inference {summary['infer_ms']['mean']:.0f} ms/volume | peak VRAM {summary['peak_vram_gb']} GB")


if __name__ == "__main__":
    main()