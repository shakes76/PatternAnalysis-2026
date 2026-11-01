"""
This script was produced with the help of AI.

Show slices of one test patient with their true and predicted labels, and print the
AD probability of every slice. Uses evaluate.py's saved test predictions; no GPU needed.

Usage (from the project folder):
    python scripts/show_patient.py 1005735
    python scripts/show_patient.py 1005735 --model cnn
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

DATA_ROOT = Path("/home/groups/comp3710/ADNI/AD_NC/test")
EVAL_DIR = Path("results/eval")
OUT_DIR = Path("images")
SLICE_POSITIONS = [0, 1, 2, 3, 4]  # slices to plot (0 = first of the patient's 20)
LABELS = {0: "NC", 1: "AD"}


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def slice_paths(pid):
    """The patient's slice files, sorted by slice number."""
    for cls in ("AD", "NC"):
        paths = list((DATA_ROOT / cls).glob(f"{pid}_*"))
        if paths:
            return sorted(paths, key=lambda p: int(p.stem.rsplit("_", 1)[1]))
    raise FileNotFoundError(f"Patient {pid} not found in {DATA_ROOT}")


ap = argparse.ArgumentParser()
ap.add_argument("pid", help="test patient ID")
ap.add_argument("--model", default="gfnet", choices=["gfnet", "cnn"])
args = ap.parse_args()

patients = {r["pid"]: r for r in read_csv(EVAL_DIR / args.model / "test_patient.csv")}
if args.pid not in patients:
    raise SystemExit(f"Patient {args.pid} is not in the {args.model} test predictions")
patient = patients[args.pid]
slices = sorted(
    (
        r
        for r in read_csv(EVAL_DIR / args.model / "test_slice.csv")
        if r["pid"] == args.pid
    ),
    key=lambda r: int(r["slice_pos"]),
)
paths = slice_paths(args.pid)

true_label = LABELS[int(patient["label"])]
mean_prob = float(patient["prob_ad"])
pred_label = "AD" if mean_prob >= 0.5 else "NC"

# Terminal table
print(
    f"\nPatient {args.pid} ({args.model.upper()}): true {true_label}, predicted {pred_label}"
)
print(f"{'Slice':>6}{'File':>18}{'P(AD)':>9}{'Prediction':>12}")
print("-" * 45)
for r, path in zip(slices, paths):
    p = float(r["prob_ad"])
    print(
        f"{int(r['slice_pos']) + 1:>6}{path.name:>18}{p:>9.3f}{'AD' if p >= 0.5 else 'NC':>12}"
    )
print("-" * 45)
print(f"{'Mean':>6}{'':>18}{mean_prob:>9.3f}{pred_label:>12}")
print(f"Slice std {float(patient['slice_prob_std']):.3f}")

# Figure
fig, axes = plt.subplots(
    1, len(SLICE_POSITIONS), figsize=(3 * len(SLICE_POSITIONS), 3.8)
)
for ax, pos in zip(axes, SLICE_POSITIONS):
    ax.imshow(Image.open(paths[pos]).convert("L"), cmap="gray")
    ax.set_title(f"Slice {pos + 1}\nP(AD) = {float(slices[pos]['prob_ad']):.3f}")
    ax.axis("off")
fig.suptitle(
    f"Patient {args.pid} ({args.model.upper()}) - True: {true_label}, "
    f"Predicted: {pred_label} (mean P(AD) = {mean_prob:.3f})"
)
fig.tight_layout(rect=(0, 0, 1, 0.92))  # leave room for the main title
OUT_DIR.mkdir(exist_ok=True)
out = OUT_DIR / f"patient_{args.pid}_{args.model}.png"
fig.savefig(out, dpi=150)
print(f"\nSaved {out}")
