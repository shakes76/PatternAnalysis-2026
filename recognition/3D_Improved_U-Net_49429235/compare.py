import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    with open(path) as f:
        return json.load(f)


def fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--improved", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out-dir", default="comparison")
    ap.add_argument("--names", nargs=2, default=["Improved 3D U-Net", "3D U-Net (baseline)"])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    imp, base = load(args.improved), load(args.baseline)
    n_imp, n_base = args.names
    classes = imp["class_names"]
    prost = classes.index("prostate")
    lines = []

    # ---- 1. per-class accuracy and boundary precision ---------------------- #
    lines += [f"### Per-class test results (mean over test volumes)", "",
              f"| Class | Dice {n_imp} | Dice {n_base} | IoU {n_imp} | IoU {n_base} | "
              f"HD95 {n_imp} | HD95 {n_base} | ASD {n_imp} | ASD {n_base} |",
              "|---|---|---|---|---|---|---|---|---|"]
    for c, name in enumerate(classes):
        if c == 0:
            continue
        lines.append(
            f"| {name} | {fmt(imp['dice_per_class'][c])} | {fmt(base['dice_per_class'][c])} | "
            f"{fmt(imp['iou_per_class'][c])} | {fmt(base['iou_per_class'][c])} | "
            f"{fmt(imp['hd95_per_class'][c], 2)} | {fmt(base['hd95_per_class'][c], 2)} | "
            f"{fmt(imp['asd_per_class'][c], 2)} | {fmt(base['asd_per_class'][c], 2)} |")
    lines += ["", "HD95 and ASD are in voxels of the resized volume (lower is better). They are "
              "averaged only over volumes where both a ground-truth and a predicted structure exist.", ""]

    lines += ["### Structures missed entirely (ground truth present, nothing predicted)", "",
              f"| Class | {n_imp} | {n_base} |", "|---|---|---|"]
    for c, name in enumerate(classes):
        if c == 0:
            continue
        lines.append(f"| {name} | {imp['missed_per_class'][c]}/{imp['gt_present_per_class'][c]} | "
                     f"{base['missed_per_class'][c]}/{base['gt_present_per_class'][c]} |")
    lines.append("")

    # ---- 2. position along the prostate ------------------------------------ #
    th_names = ["third 1", "third 2 (middle)", "third 3"]
    lines += ["### Prostate Dice by position along the gland", "",
              f"| Position | {n_imp} | {n_base} | Difference |", "|---|---|---|---|"]
    for k, nm in enumerate(th_names):
        a, b = imp["prostate_dice_thirds"][k], base["prostate_dice_thirds"][k]
        lines.append(f"| {nm} | {fmt(a)} | {fmt(b)} | {fmt(a - b)} |")
    lines += ["", "Thirds are taken along the slice axis of the ground-truth prostate extent. "
              "Decide which end is the base and which is the apex from the images before "
              "labelling them in the report.", ""]

    # ---- 3. resources -------------------------------------------------------- #
    lines += ["### Computational cost", "", f"| Metric | {n_imp} | {n_base} |", "|---|---|---|",
              f"| Trainable parameters | {imp['parameters']:,} | {base['parameters']:,} |",
              f"| Training peak VRAM (GB) | {fmt(imp['train_peak_vram_gb'], 2)} | {fmt(base['train_peak_vram_gb'], 2)} |",
              f"| Inference peak VRAM (GB) | {fmt(imp['inference_peak_vram_gb'], 2)} | {fmt(base['inference_peak_vram_gb'], 2)} |",
              f"| Inference time (s/volume) | {fmt(imp['mean_inference_s'], 3)} | {fmt(base['mean_inference_s'], 3)} |",
              f"| Best epoch | {imp['best_epoch']} | {base['best_epoch']} |", "",
              "Add total training time from your Slurm logs.", ""]

    # ---- 4. paired comparison over test volumes ------------------------------ #
    ci = {r.get("case", i): r for i, r in enumerate(imp.get("per_case", []))}
    cb = {r.get("case", i): r for i, r in enumerate(base.get("per_case", []))}
    shared = [k for k in ci if k in cb]
    pairs = [(ci[k]["dice"][prost], cb[k]["dice"][prost], k) for k in shared]
    pairs = [(a, b, k) for a, b, k in pairs if a == a and b == b]   # drop NaN
    if pairs:
        diffs = [a - b for a, b, _ in pairs]
        wins = sum(d > 0 for d in diffs)
        lines += ["### Paired comparison of prostate Dice over test volumes", "",
                  f"- Volumes compared: {len(pairs)} (improved minus baseline)",
                  f"- Mean difference: {sum(diffs) / len(diffs):+.4f}",
                  f"- Improved model better on {wins}/{len(pairs)} volumes; "
                  f"worse on {sum(d < 0 for d in diffs)}"]
        try:
            from scipy.stats import wilcoxon
            if any(d != 0 for d in diffs) and len(diffs) >= 6:
                stat = wilcoxon(diffs)
                lines.append(f"- Wilcoxon signed-rank p = {stat.pvalue:.4f}")
        except Exception as e:  # scipy missing or test not applicable
            lines.append(f"- (Wilcoxon test skipped: {e})")
        lines += ["", "Volumes from the same patient are not independent, and this is a single "
                  "training run per model, so treat any p-value as indicative only.", ""]

        fig, ax = plt.subplots(figsize=(4.5, 5))
        for a, b, _ in pairs:
            ax.plot([0, 1], [b, a], marker="o", color="gray", alpha=0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([n_base, n_imp], rotation=15)
        ax.set_ylabel("Prostate Dice")
        ax.set_title("Per-volume prostate Dice (test)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "per_case_dice.png"), dpi=150)
        plt.close(fig)
    else:
        lines += ["### Paired comparison", "",
                  "No shared per-case results found - were both runs evaluated on the same split?", ""]

    # ---- thirds bar chart ----------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(6, 4))
    w = 0.38
    xs = range(3)
    ax.bar([x - w / 2 for x in xs], base["prostate_dice_thirds"], w, label=n_base)
    ax.bar([x + w / 2 for x in xs], imp["prostate_dice_thirds"], w, label=n_imp)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(th_names)
    ax.set_ylabel("Prostate Dice")
    ax.set_title("Dice by position along the prostate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "thirds.png"), dpi=150)
    plt.close(fig)

    text = "\n".join(lines)
    with open(os.path.join(args.out_dir, "tables.md"), "w") as f:
        f.write(text)
    print(text)
    print(f"\nSaved tables.md and charts to {args.out_dir}/")


if __name__ == "__main__":
    main()
