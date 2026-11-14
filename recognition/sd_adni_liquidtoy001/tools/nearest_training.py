"""Post-hoc: how close do generated samples come to training slices? (GPU job)

Tests the explanation of README Section 8, autopsy 4: B3 replicates training
slices because it overfits as training goes on. B3 kept two checkpoints,
epochs 49 and 98; if replication grows with training, the epoch-49 samples
should sit farther from their nearest training slices, and fewer of them
should be near-copies, than the epoch-98 samples from the same seeds.

For each cohort (seeds below --seed-limit only) it finds every sample's
nearest training slice in the 64 px grey pixels of the final evaluation and
reports:

- the nearest-training RMSE (mean, median, min);
- how many samples come closer to a training slice than real validation
  slices do (below the 1st percentile and below the minimum of the validation
  slices' own nearest-training RMSE, from a rehearsal-mode real run);
- among those near-copies, how many training slices are copied by more than
  one seed.

Only the training arrays and the validation calibration distances of a
rehearsal-mode final_real_features run are read; no test data. Row indices of
the nearest training slices are written to a private file.

    python tools/nearest_training.py --real outputs/final-real-rehearsal-<job> \
        --cohort e49=PATH --cohort e98=PATH --seed-limit 37328 --output DIR
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_common import cohort_records, generated_views, nearest  # noqa: E402


def summary(values: np.ndarray) -> dict:
    """Mean, median, min and max of an array."""
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


@torch.no_grad()
def main() -> None:
    """Nearest training slices of each cohort; writes report.json."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--cohort", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--seed-limit", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.perf_counter()
    real_report = json.loads((args.real / "report.json").read_text())
    if real_report["test_split_read"] or real_report["roles"]["calibration"] != "validation":
        raise ValueError("--real must be a rehearsal-mode run")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    train_gray = (
        torch.from_numpy(np.load(args.real / "private-real-arrays.npz")["train_gray64"])
        .to(device)
        .float()
        / 255
    )
    calibration = np.load(args.real / "private-real-features.npz")[
        "calibration_pixel_train"
    ]
    thresholds = {
        "validation_p1": float(np.percentile(calibration, 1)),
        "validation_min": float(calibration.min()),
    }

    report_cohorts, private = {}, {}
    for item in args.cohort:
        name, folder = item.split("=", 1)
        folder = Path(folder)
        records = [r for r in cohort_records(folder) if r["seed"] < args.seed_limit]
        gray = torch.stack(
            [generated_views(folder / r["file"])["gray64"] for r in records]
        ).to(device)
        distance, index = nearest(gray, train_gray)
        distance, index = distance.cpu().numpy(), index.cpu().numpy()
        near = {}
        for key, limit in thresholds.items():
            hit = distance < limit
            counts = collections.Counter(index[hit].tolist())
            near[key] = {
                "samples": int(hit.sum()),
                "share": float(hit.mean()),
                "training_slices_copied_by_several_seeds": sum(
                    1 for c in counts.values() if c > 1
                ),
                "most_seeds_on_one_slice": max(counts.values(), default=0),
            }
        labels = [r["label"] for r in records]
        report_cohorts[name] = {
            "folder": str(folder),
            "samples": len(records),
            "per_class": dict(collections.Counter(labels)),
            "nearest_training_rmse": summary(distance),
            "near_copies": near,
        }
        private[f"{name}_distance"] = distance
        private[f"{name}_train_row"] = index
        private[f"{name}_seed"] = np.array([r["seed"] for r in records])
        private[f"{name}_is_nc"] = np.array([x == "NC" for x in labels], dtype=np.int8)
        print(name, json.dumps(report_cohorts[name]), flush=True)

    np.savez(args.output / "private-nearest.npz", **private)
    script = Path(__file__).resolve()
    report = {
        "post_hoc": True,
        "real": str(args.real),
        "test_split_read": False,
        "seed_limit": args.seed_limit,
        "thresholds_from_validation": thresholds,
        "cohorts": report_cohorts,
        "code_sha256": {
            n: hashlib.sha256((script.parent / n).read_bytes()).hexdigest()
            for n in ("nearest_training.py", "final_common.py")
        },
        "seconds": time.perf_counter() - started,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
