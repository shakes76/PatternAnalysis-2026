"""Summarise repeated runs into README tables (mean +- std over seeds).

Reads results/<run>_test_per_volume.csv (written by train.py) and
results/<run>_analysis.csv (written by predict.py --analyse); runs whose files
do not exist yet are skipped. Writes results/summary.md.
"""
import csv
import os
import statistics as st

# Each group is the set of runs (training seeds) whose results are averaged; names match the files in results/.
GROUPS = {
    "2D U-Net (baseline)": ["unet2d", "unet2d_s1", "unet2d_s2"],
    "2D Improved U-Net": ["improved2d", "improved2d_s1", "improved2d_s2"],
    "3D Improved U-Net (matched settings)": ["improved3d_matched", "improved3d_matched_s1", "improved3d_matched_s2"],
    "3D Improved U-Net (lr 5e-4, 50 epochs)": ["improved3d"],
    "2D Improved U-Net (lr 5e-4)": ["improved2d_lr5e4"],
}
# The six test patients (the columns of the per-patient table).
PATIENTS = ["B006", "G021", "K018", "S035", "T009", "W029"]


def read(path):
    """Rows of a CSV file as a list of dicts, or None if the file does not exist."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def mean(xs):
    """Arithmetic mean of a non-empty list."""
    return sum(xs) / len(xs)


def fmt(xs):
    """mean +- sample std; the std is only a rough guide with so few runs."""
    if not xs:
        return "n/a"
    if len(xs) == 1:
        return f"{xs[0]:.3f} (1 run)"
    return f"{mean(xs):.3f} ± {st.stdev(xs):.3f} (n={len(xs)})"


def per_volume(run):
    """Per-volume test Dice rows written by train.py for this run (None if missing)."""
    return read(f"results/{run}_test_per_volume.csv")


def analysis_mean(run, key):
    """Mean over the test volumes of one column of results/<run>_analysis.csv (None if missing)."""
    rows = read(f"results/{run}_analysis.csv")
    return None if rows is None else mean([float(r[key]) for r in rows])


# Table 1: test prostate Dice per run, then mean and standard deviation over the runs of each group.
out = []

out += ["## Test prostate Dice (mean over the 28 test volumes, per run)", "",
        "| Model | Runs | Mean ± std over runs |", "|---|---|---|"]
# Per patient: average the scans inside a run, then average over the runs.
patient_means = {}
for group, runs in GROUPS.items():
    vals, by_patient = [], {}
    for run in runs:
        rows = per_volume(run)
        if rows is None:
            continue
        vals.append(mean([float(r["dice_class5"]) for r in rows]))
        for r in rows:
            by_patient.setdefault(r["patient"], {}).setdefault(run, []).append(float(r["dice_class5"]))
    patient_means[group] = {p: mean([mean(v) for v in d.values()]) for p, d in by_patient.items()}
    out.append(f"| {group} | {', '.join(f'{v:.3f}' for v in vals) or 'n/a'} | {fmt(vals)} |")

# Table 2: prostate Dice per test patient.
out += ["", "## Prostate Dice per test patient (mean over runs)", "",
        "| Model | " + " | ".join(PATIENTS) + " |", "|---|" + "---|" * len(PATIENTS)]
for group, pm in patient_means.items():
    out.append(f"| {group} | " + " | ".join(f"{pm[p]:.3f}" if p in pm else "n/a" for p in PATIENTS) + " |")
base, three_d = patient_means.get("2D U-Net (baseline)", {}), patient_means.get("3D Improved U-Net (matched settings)", {})
if base and three_d:
    wins = sum(three_d[p] > base[p] for p in PATIENTS if p in base and p in three_d)
    out += ["", f"3D (matched) beats the 2D baseline on {wins} of {len(PATIENTS)} test patients."]

# Table 3: base/middle/apex Dice, leakage (as %) and inference time, from the analysis files.
out += ["", "## Dice by prostate third, rectum/bladder leakage and inference time (from predict.py --analyse)", "",
        "| Model | Base | Middle | Apex | Leak rectum % | Leak bladder % | s / volume |", "|---|---|---|---|---|---|---|"]
for group, runs in GROUPS.items():
    cells = []
    for key in ("dice_base", "dice_middle", "dice_apex"):
        vals = [v for v in (analysis_mean(r, key) for r in runs) if v is not None]
        cells.append(fmt(vals))
    for key in ("leak_rectum", "leak_bladder"):
        vals = [v * 100 for v in (analysis_mean(r, key) for r in runs) if v is not None]
        cells.append(f"{mean(vals):.2f}" if vals else "n/a")
    vals = [v for v in (analysis_mean(r, "seconds") for r in runs) if v is not None]
    cells.append(f"{mean(vals):.2f}" if vals else "n/a")
    out.append(f"| {group} | " + " | ".join(cells) + " |")

# Table 4: Dice and IoU for every class.
out += ["", "## Per-class Dice / IoU on the test split (mean over the runs analysed)", "",
        "| Model | bladder | rectum | prostate | bone | body | background |", "|---|---|---|---|---|---|---|"]
for group, runs in GROUPS.items():
    cells = []
    for cname in ("bladder", "rectum", "prostate", "bone", "body", "background"):
        d = [v for v in (analysis_mean(r, f"dice_{cname}") for r in runs) if v is not None]
        i = [v for v in (analysis_mean(r, f"iou_{cname}") for r in runs) if v is not None]
        cells.append(f"{mean(d):.3f} / {mean(i):.3f}" if d else "n/a")
    out.append(f"| {group} | " + " | ".join(cells) + " |")

# Save the tables as markdown and print them.
text = "\n".join(out) + "\n"
with open("results/summary.md", "w") as f:
    f.write(text)
print(text)
