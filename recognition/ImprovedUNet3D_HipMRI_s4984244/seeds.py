"""
Summarise evaluations of the same models trained with different seeds.

    python seeds.py images/results.json images_s3711/results.json images_s3712/results.json

Each results.json is written by predict.py for one seed (same checkpoint
order, same split). This prints, for every model, the mean +/- standard
deviation over seeds of the headline metrics, then compares the first model
with each other one after averaging every scan's metrics over the seeds, so
that seed-to-seed noise does not drive the paired tests.
"""

import argparse
import json

import numpy as np

from predict import PAIRED_METRICS, compare

HEADLINE = {
    "bladder Dice": lambda s: s["dice"][3],
    "rectum Dice": lambda s: s["dice"][4],
    **PAIRED_METRICS,
}


def load(paths):
    """List over seeds of {model: scans}, checking that the runs match."""
    runs = []
    for path in paths:
        results = json.loads(open(path).read())
        runs.append({m: r["scans"] for m, r in results.items()
                     if isinstance(r, dict) and "scans" in r})
    models = list(runs[0])
    for run in runs[1:]:
        if list(run) != models:
            raise ValueError(f"Model order differs between runs: {list(run)} vs {models}")
    return runs, models


def seed_average(runs, model):
    """Per-scan metrics averaged over seeds, as scans that compare() accepts."""
    names = [s["name"] for s in runs[0][model]]
    averaged = []
    for name in names:
        per_seed = [next(s for s in run[model] if s["name"] == name) for run in runs]
        scan = {"name": name}
        scan["dice"] = np.nanmean([s["dice"] for s in per_seed], axis=0).tolist()
        scan["hd95"] = np.nanmean([s["hd95"] for s in per_seed], axis=0).tolist()
        scan["zone_dice"] = {z: float(np.nanmean([s["zone_dice"][z] for s in per_seed]))
                             for z in per_seed[0]["zone_dice"]}
        for key in ("spill_slices", "extra_prostate_ml", "missed_prostate_ml"):
            scan[key] = float(np.mean([s[key] for s in per_seed]))
        averaged.append(scan)
    return averaged


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", help="results.json of each seed")
    runs, models = load(parser.parse_args().results)

    print(f"Mean +/- SD over {len(runs)} seeds (each seed: mean over scans)")
    print("| Metric | " + " | ".join(models) + " |")
    print("|---" * (len(models) + 1) + "|")
    for metric, value in HEADLINE.items():
        cells = []
        for m in models:
            per_seed = [np.nanmean([value(s) for s in run[m]]) for run in runs]
            cells.append(f"{np.mean(per_seed):.3f} +/- {np.std(per_seed, ddof=1):.3f}")
        print(f"| {metric} | " + " | ".join(cells) + " |")

    averaged = {m: seed_average(runs, m) for m in models}
    for other in models[1:]:
        compare(models[0], averaged[models[0]], other, averaged[other])


if __name__ == "__main__":
    main()
