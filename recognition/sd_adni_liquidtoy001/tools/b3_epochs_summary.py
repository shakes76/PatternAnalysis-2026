"""Summarise the B3 epoch comparison (local; post-hoc, see tools/b3_epochs.sbatch).

Membership: the loss test of tools/membership.py against validation slices
(rehearsal mode) for B3's epoch-49 and epoch-98 checkpoints, with the paired
bootstrap of tools/final_metrics.py (the same images and noise for both), and
SD as the control. Samples: the nearest-training report of
tools/nearest_training.py for the same 128 seeds per class.

    python tools/b3_epochs_summary.py --mia49 DIR --mia98 DIR --mia-sd DIR \
        --nearest DIR --json docs/b3-epochs-2026-11-15.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_metrics import auc, membership_block  # noqa: E402


def scores(folder: Path) -> tuple[np.lib.npyio.NpzFile, dict]:
    """Private scores and report of one rehearsal-mode membership run."""
    report = json.loads((folder / "report.json").read_text())
    if report["test_split_read"] or report["heldout_split"] != "validation":
        raise ValueError(f"{folder} is not a rehearsal-mode run")
    return np.load(folder / "private-scores.npz"), report


def main() -> None:
    """Write the summary JSON."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for name in ("--mia49", "--mia98", "--mia-sd", "--nearest"):
        parser.add_argument(name, type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    runs = {k: scores(getattr(args, k)) for k in ("mia49", "mia98", "mia_sd")}
    (s49, r49), (s98, r98), (ssd, _) = runs["mia49"], runs["mia98"], runs["mia_sd"]

    def block(a, b):
        return membership_block(
            a["member_primary"], a["nonmember_primary"],
            b["member_primary"], b["nonmember_primary"],
        )

    by_t = {
        name: [
            auc(s["member_by_t"][:, i], s["nonmember_by_t"][:, i])
            for i in range(len(s["timesteps"]))
        ]
        for name, s in (("epoch49", s49), ("epoch98", s98))
    }
    nearest = json.loads((args.nearest / "report.json").read_text())
    summary = {
        "post_hoc": True,
        "question": "Does B3's replication of training slices grow with training "
        "(README 8, autopsy 4)?",
        "membership_vs_validation": {
            "epoch49_vs_sd": block(s49, ssd),
            "epoch98_vs_sd": block(s98, ssd),
            "epoch49_minus_epoch98": block(s49, s98),
            "mean_primary_loss": {
                "epoch49": r49["mean_primary_loss"],
                "epoch98": r98["mean_primary_loss"],
            },
            "auc_by_timestep": {
                "timesteps": s49["timesteps"].tolist(),
                **by_t,
            },
        },
        "samples_nearest_training": {
            "seed_limit": nearest["seed_limit"],
            "thresholds_from_validation": nearest["thresholds_from_validation"],
            "cohorts": {
                name: {k: v for k, v in c.items() if k != "folder"}
                for name, c in nearest["cohorts"].items()
            },
        },
        "weights_sha256": {"epoch49": r49["weights_sha256"], "epoch98": r98["weights_sha256"]},
        "test_split_read": False,
    }
    args.json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["membership_vs_validation"]["epoch49_minus_epoch98"], indent=2))


if __name__ == "__main__":
    main()
