"""Print the README's result tables as GitHub markdown, straight from the
metric files, so no number is copied by hand (local).

    python tools/results_tables.py --metrics outputs/final-metrics-<date>/metrics.json \
        --ssim outputs/final-ssim-<date>/ssim.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ORDER = ("sd", "e0", "e2", "r1e2", "e3", "r1e3", "b3", "b1", "b2", "b2b")
NAMES = {
    "sd": "SD 1.5, no training (SD)",
    "e0": "First LoRA, no recipe (E0)",
    "e2": "**Full fine-tune**, cleaned data (E2)",
    "r1e2": "Full fine-tune, 2nd seed (R1-E2)",
    "e3": "**LoRA adapter**, same recipe (E3)",
    "r1e3": "LoRA adapter, 2nd seed (R1-E3)",
    "b3": "Diffusion from scratch, 92 M (B3)",
    "b1": "Conditional VAE (B1)",
    "b2": "Conditional GAN (B2)",
    "b2b": "One GAN per class (B2b)",
}


def interval(values, digits=3):
    low, high = values
    return f"{low:.{digits}f} to {high:.{digits}f}"


def fidelity_table(metrics, ssim):
    floor = metrics["floor"]["inception"]
    rows = [
        "| Model | Feature-nearest SSIM (95% CI) | > 0.6 | KID x 100 | FID | Precision | Recall | Density | Coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| Real slices (floor) | {:.3f} | {:.0f}% | {:.2f} | {:.1f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
            ssim["real_floor"]["mean"],
            100 * ssim["real_floor"]["fraction_above_0_6"],
            100 * floor["kid"]["mean"],
            floor["fid"],
            *(
                floor["prdc"][m]["mean"]
                for m in ("precision", "recall", "density", "coverage")
            ),
        ),
    ]
    for name in ORDER:
        if name not in metrics["models"]:
            continue
        entry = metrics["models"][name]["inception"]
        s = ssim["groups"][name]["all"]["nearest_ssim"]
        prdc = entry["prdc"]
        rows.append(
            "| {} | {:.3f} ({}) | {:.0f}% | {:.2f} ± {:.2f} | {:.1f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                NAMES[name],
                s["mean"],
                interval(s["mean_ci95"]),
                100 * s["fraction_above_0_6"],
                100 * entry["kid"]["mean"],
                100 * entry["kid"]["sd"],
                entry["fid"],
                *(
                    prdc[m]["mean"]
                    for m in ("precision", "recall", "density", "coverage")
                ),
            )
        )
    return "\n".join(rows)


def memorisation_table(metrics):
    reference = metrics["real_reference"]["heldout_to_train_pixel_rmse"]
    membership = metrics.get("membership", {})
    rows = [
        "| Model | Nearest-training RMSE, mean (min) | M2 pixel: difference (95% CI) | Closer to training (negative: farther) | M2 VAE: difference (95% CI) | Membership AUC | AUC minus SD (95% CI) | TPR at 1% FPR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| Real held-out slices | {reference['mean']:.4f} ({reference['min']:.4f}) | 0 (calibration) | 0% | 0 (calibration) | | | |",
    ]
    for name in ORDER:
        if name not in metrics["models"]:
            continue
        entry = metrics["models"][name]
        m1 = entry["m1_nearest_training_rmse"]
        pixel = entry["m2_training_preference"]["pixel"]
        vae = entry["m2_training_preference"]["vae"]
        mia = membership.get(name)
        if mia and name != "sd":
            mia_cells = "{:.3f} | {:+.3f} ({}) | {:.1f}%".format(
                mia["auc"],
                mia["auc_minus_control"],
                interval(mia["auc_minus_control_ci95"]),
                100 * mia["tpr_at_1pct_fpr"],
            )
        elif mia:
            mia_cells = "{:.3f} | control | {:.1f}%".format(
                mia["auc"], 100 * mia["tpr_at_1pct_fpr"]
            )
        else:
            mia_cells = "n/a | n/a | n/a"
        rows.append(
            "| {} | {:.4f} ({:.4f}) | {:+.4f} ({}) | {:+.1f}% | {:+.4f} ({}) | {} |".format(
                NAMES[name],
                m1["mean"],
                m1["min"],
                pixel["difference"],
                interval(pixel["difference_ci95"], 4),
                pixel["closer_to_training_percent"],
                vae["difference"],
                interval(vae["difference_ci95"], 4),
                mia_cells,
            )
        )
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--ssim", type=Path, required=True)
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # the tables use ±
    metrics = json.loads(args.metrics.read_text())
    ssim = json.loads(args.ssim.read_text())
    split = "test" if metrics["test_split_read"] else "validation (rehearsal)"
    print(f"<!-- generated by tools/results_tables.py from the {split} split -->\n")
    print(fidelity_table(metrics, ssim))
    print()
    print(memorisation_table(metrics))


if __name__ == "__main__":
    main()
