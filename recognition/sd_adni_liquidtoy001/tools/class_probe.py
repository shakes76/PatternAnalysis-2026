"""Does the requested class show in the samples? A post-hoc linear-probe check
(local; not in the pre-registered protocol, reported as post-hoc).

A logistic-regression probe separating AD from NC is fitted on the 1,024
class-balanced validation slices (Inception features, standardised, the
regularisation chosen by 5-fold cross-validation) and never sees test or
generated data while fitting. Its score is then read on:

- the real test slices, which says how far AD and NC are separable per slice
  in these features at all (the ceiling for any generator);
- each model's samples, split by the class they were asked for: an AUC near
  the real one means the conditioning shows; 0.5 means AD and NC samples look
  alike to the probe.

    python tools/class_probe.py --real outputs/final-test/real \
        --cohorts outputs/final-test/cohorts --output outputs/final-test/probe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegressionCV  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

SEED = 3710
BOOTSTRAP = 2000
SHOWN = ("e2", "e3", "b3")  # the densities drawn in the figure


def auc_with_interval(is_nc: np.ndarray, score: np.ndarray) -> dict:
    """AUC of the probe's NC score for telling NC from AD, with a bootstrap CI."""
    random = np.random.default_rng(SEED)
    values = []
    for _ in range(BOOTSTRAP):
        index = random.integers(0, len(score), len(score))
        if len(set(is_nc[index])) == 2:
            values.append(roc_auc_score(is_nc[index], score[index]))
    return {
        "auc": float(roc_auc_score(is_nc, score)),
        "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
        "n": int(len(score)),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    real = np.load(args.real / "private-real-features.npz")
    cohorts = np.load(args.cohorts / "private-cohort-features.npz")
    listing = json.loads((args.cohorts / "private-cohorts.json").read_text())

    probe = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(Cs=10, cv=5, max_iter=5000, random_state=SEED),
    )
    probe.fit(real["floor_inception"], real["floor_is_nc"])

    def score(features):
        return probe.predict_proba(features)[:, 1]  # probability of NC

    test_score = score(real["heldout_inception"])
    report = {
        "post_hoc": True,
        "probe": "standardised logistic regression on Inception-v3 pool features, "
        "fitted on 1,024 class-balanced validation slices, C by 5-fold CV",
        "chosen_C": float(probe[-1].C_[0]),
        "real_test": auc_with_interval(real["heldout_is_nc"], test_score),
        "models": {},
    }
    densities = {"real test": (real["heldout_is_nc"], test_score)}
    for name in listing:
        is_nc = cohorts[f"{name}_is_nc"]
        values = score(cohorts[f"{name}_inception"])
        report["models"][name] = {
            **auc_with_interval(is_nc, values),
            "mean_nc_probability": {
                "asked_ad": float(values[is_nc == 0].mean()),
                "asked_nc": float(values[is_nc == 1].mean()),
            },
        }
        if name in SHOWN:
            densities[name] = (is_nc, values)

    figure, axes = plt.subplots(1, len(densities), figsize=(4 * len(densities), 3.4))
    bins = np.linspace(0, 1, 26)
    for axis, (name, (is_nc, values)) in zip(axes, densities.items()):
        for c, label, colour in ((0, "AD", "#c8640a"), (1, "NC", "#1f6fb4")):
            axis.hist(
                values[is_nc == c],
                bins=bins,
                density=True,
                alpha=0.55,
                color=colour,
                label=label if name == "real test" else f"asked {label}",
            )
        auc = (report["real_test"] if name == "real test" else report["models"][name])[
            "auc"
        ]
        axis.set(
            title=f"{name}: AUC {auc:.2f}",
            xlabel="probe probability of NC",
            ylabel="density",
        )
        axis.legend(fontsize=8)
    figure.text(
        0.01,
        0.01,
        "Probe fitted on validation slices only. Post-hoc analysis, not pre-registered.",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(args.output / "class-probe.png", dpi=150)
    (args.output / "class-probe.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
