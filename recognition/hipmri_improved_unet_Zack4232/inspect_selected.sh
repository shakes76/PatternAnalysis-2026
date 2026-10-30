#!/bin/bash
#SBATCH --job-name=inspect-selected
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --time=00:05:00
#SBATCH --output=selected_%j.out
#SBATCH --error=selected_%j.err

set -euo pipefail
source "$HOME/miniconda3/bin/activate"
conda activate torch
cd "$SLURM_SUBMIT_DIR"

python -u - <<'PY'
import csv
from pathlib import Path

import torch

from dataset import HipMRIDataset
from metrics import positive_slice_stats, summarize_stats
from modules import UNet2D
from predict import plot_failure

torch.set_num_threads(2)
root = Path("runs/validation_inspection_710213")
with (root / "slice_metrics.csv").open(newline="") as handle:
    rows = list(csv.DictReader(handle))

# Select the lowest-Dice case 038 slice with more than 100 target pixels.
candidates = [
    row for row in rows
    if int(row["case_id"]) == 38
    and int(row["true_prostate_pixels"]) > 100
]
worst = min(
    candidates,
    key=lambda row: (
        float(row["prostate_dice"]),
        int(row["week"]),
        int(row["slice_id"]),
    ),
)
keys = [
    (37, 0, 37),
    (38, int(worst["week"]), int(worst["slice_id"])),
    (38, 0, 36),
]

checkpoint = torch.load(
    "runs/baseline_710034/results/best.pt",
    map_location="cpu",
    weights_only=True,
)
model = UNet2D(base_channels=checkpoint["base_channels"]).cuda()
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.eval()

dataset = HipMRIDataset(split="validate")
lookup = {
    (r.case_id, r.week, r.slice_id): i
    for i, r in enumerate(dataset.samples)
}
output = root / "selected_examples"
output.mkdir(exist_ok=True)

with torch.inference_mode():
    for key in keys:
        sample = dataset[lookup[key]]
        prediction = model(
            sample["image"].unsqueeze(0).cuda()
        ).argmax(dim=1).cpu()
        score = summarize_stats(positive_slice_stats(
            prediction, sample["mask"].unsqueeze(0), class_id=5
        ))
        name = f"case_{key[0]:03d}_week_{key[1]}_slice_{key[2]}.png"
        plot_failure(
            sample, prediction[0], score["mean_dice"], output / name
        )
        print(name, "Dice:", score["mean_dice"], flush=True)

print("Saved to:", output.resolve())
PY
