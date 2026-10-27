"""Aggregate results.json files from several seeds into a mean ± std markdown table."""
import glob
import json

import numpy as np

files = sorted(glob.glob("outputs/seed*/results.json"))
runs = [json.load(open(f))["results"] for f in files]
print(f"{len(runs)} runs: {files}\n")


def cell(values, digits=3):
    v = np.array(values, dtype=float)
    return f"{v.mean():.{digits}f} ± {v.std(ddof=1):.{digits}f}"


metrics = [("Test accuracy", "accuracy"), ("Test macro-F1", "macro_f1"),
           ("Class AUROC (OvR)", "auroc_ovr"), ("Confidence (correct)", "conf_correct"),
           ("Confidence (wrong)", "conf_wrong"), ("Confidence gap", "conf_gap"),
           ("Error-detection AUROC", "error_auroc"), ("ECE", "ece"),
           ("Inference (ms/pass)", "inference_ms"), ("Peak GPU memory (MB)", "peak_gpu_mb")]

print("| Metric | MLP | GCN |\n|---|---|---|")
for label, key in metrics:
    print(f"| {label} | " + " | ".join(cell([r[m][key] for r in runs]) for m in ("MLP", "GCN")) + " |")

for label, key in [("Reject rule: test accuracy", "test_accuracy"),
                   ("Reject rule: test coverage", "test_coverage")]:
    cells = []
    for m in ("MLP", "GCN"):
        vals = [r[m]["reject_rule"][key] for r in runs if r[m]["reject_rule"]]
        cells.append(cell(vals) if vals else "n/a")
    print(f"| {label} | {cells[0]} | {cells[1]} |")

print("\n| Class | MLP precision | MLP recall | GCN precision | GCN recall |")
print("|---|---|---|---|---|")
for c in range(4):
    row = [str(c)]
    for m in ("MLP", "GCN"):
        for key in ("per_class_precision", "per_class_recall"):
            row.append(cell([r[m][key][c] for r in runs]))
    print("| " + " | ".join(row) + " |")