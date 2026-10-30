"""Collect the training logs of every run into the repository as evidence.

Copies config.json, history.json and best_meta.json of each run (metrics and
settings only: no images, no patient IDs, no weights) into run_logs/, writes
run_logs/summary.csv and draws figures/experiment_history.png (validation
accuracy per epoch for every run, strict and official split side by side).

Run on Rangpur from the repository folder, with the conda env active:
    python collect_runs.py ~/comp3710_convnext/checkpoints ~/COMP3710_split_mode_A/checkpoints
"""
import csv
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("run_logs")
FILES = ("config.json", "history.json", "best_meta.json")


def load(path):
    return json.loads(path.read_text()) if path.is_file() else None


def main(roots):
    runs = []
    for root in roots:
        for run_dir in sorted(Path(root).expanduser().glob("*/*/")):
            history = load(run_dir / "history.json")
            if not history:
                continue
            target = OUT / run_dir.parent.name / run_dir.name
            target.mkdir(parents=True, exist_ok=True)
            for name in FILES:
                if (run_dir / name).is_file():
                    shutil.copy2(run_dir / name, target / name)
            config = load(run_dir / "config.json") or {}
            best = load(run_dir / "best_meta.json") or {}
            accuracy = [row.get("accuracy") for row in history]
            top = max(range(len(accuracy)), key=lambda i: accuracy[i] or 0)
            runs.append({
                "run": f"{run_dir.parent.name}/{run_dir.name}",
                "split": config.get("split_mode", "strict"),
                "input": config.get("input_mode", "single"),
                "optimizer": config.get("optimizer_name", "adam"),
                "learning_rate": config.get("learning_rate"),
                "augmentation": config.get("training_augmentation", False),
                "intensity_augmentation": bool(config.get("intensity_augmentation")),
                # Early configs stored the key as "scheduler_name".
                "scheduler": (config.get("scheduler")
                              or config.get("scheduler_name") or "none"),
                "epochs": len(history),
                "best_val_accuracy": accuracy[top],
                "best_val_epoch": top + 1,
                "saved_epoch": best.get("best_epoch"),
                "curve": accuracy,
            })

    OUT.mkdir(exist_ok=True)
    with (OUT / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[k for k in runs[0] if k != "curve"])
        writer.writeheader()
        for run in runs:
            writer.writerow({k: v for k, v in run.items() if k != "curve"})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, split in zip(axes, ("strict", "official")):
        for run in (r for r in runs if r["split"] == split):
            epochs = range(1, len(run["curve"]) + 1)
            ax.plot(epochs, run["curve"], marker=".", label=run["run"].split("/")[1])
        ax.set(title=f"{split} split: validation slice accuracy (threshold 0.5)",
               xlabel="Epoch", ylabel="Validation accuracy")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/experiment_history.png", dpi=150)
    print(f"Collected {len(runs)} runs into {OUT}/ and wrote figures/experiment_history.png")


if __name__ == "__main__":
    main(sys.argv[1:])
