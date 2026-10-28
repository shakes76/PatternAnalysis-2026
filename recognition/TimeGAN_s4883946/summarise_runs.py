"""Summarise experiment runs: mean +- std over seeds for each setting and model.

Benchmarks each setting on the test metrics and on resources (parameters, training
time, peak GPU memory).

Usage: python summarise_runs.py outputs/transform [outputs/step ...]
Reads <group>/<setting>/<model>_s<seed>/<model>_summary.json written by train.py and
prints a markdown table next to the real-data reference (real train windows scored
against real test windows) for the same setting.
"""
import glob
import json
import os
import sys

import numpy as np

COLS = ["audit_ladder", "audit_crossed", "kl_spread", "kl_return", "ssim",
        "acf_lag1", "acf_lag5"]
# resource profile recorded by train.py (peak VRAM is None on CPU runs)
RES = [("params", "parameters"), ("train_s", "train_seconds"), ("vram_mb", "peak_vram_mb")]


def main(groups):
    """Print one markdown row per (setting, model), with a real-data reference row after each."""
    heads = COLS + [r[0] for r in RES]
    print("| group | setting | model | seeds | " + " | ".join(heads) + " |")
    print("|" + "---|" * (4 + len(heads)))
    for group in groups:
        for setting_dir in sorted(glob.glob(f"{group}/*")):
            setting = os.path.basename(setting_dir)
            refs = []
            for model in ("timegan", "baseline"):
                runs = [json.load(open(f)) for f in
                        sorted(glob.glob(f"{setting_dir}/{model}_s*/{model}_summary.json"))]
                if not runs:
                    continue
                # every run stores the same real train-vs-test reference, so averaging is safe
                refs += [r["reference"] for r in runs if "reference" in r]
                cells = []
                for c in COLS:
                    v = np.array([r["test"][c] for r in runs])
                    cells.append(f"{v.mean():.3f} ± {v.std():.3f}" if len(v) > 1 else f"{v[0]:.3f}")
                for _, key in RES:
                    v = [r["profile"][key] for r in runs if r["profile"].get(key) is not None]
                    cells.append(f"{np.mean(v):.0f}" if v else "n/a")
                print(f"| {os.path.basename(group)} | {setting} | {model} | {len(runs)} | "
                      + " | ".join(cells) + " |")
            if refs:
                cells = [f"{np.mean([r[c] for r in refs]):.3f}" for c in COLS] + ["-"] * len(RES)
                print(f"| {os.path.basename(group)} | {setting} | real train vs test | - | "
                      + " | ".join(cells) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
