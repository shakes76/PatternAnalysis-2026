"""Distribution, memorisation and membership measures for the final evaluation.

Runs locally on the features the two GPU jobs write; see
docs/final-evaluation-protocol.md for why each measure is used.

    python tools/final_metrics.py --real outputs/final-real-<job> \
        --cohorts outputs/final-cohorts-<job> [--membership outputs/final-mia-<job> ...] \
        --output outputs/final-metrics-<date>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import linalg, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_common import LABELS  # noqa: E402

SEED = 3710
BOOTSTRAP = 10_000
PRDC_K = 5
PRDC_DRAWS = 10
KID_SUBSETS, KID_SUBSET_SIZE = 100, 500


# Distribution distances ----------------------------------------------------


def frechet_distance(mean_a, cov_a, mean_b, cov_b) -> float:
    """||m_a - m_b||^2 + Tr(C_a + C_b - 2 (C_a C_b)^(1/2)) (Heusel et al., 2017)."""
    root = linalg.sqrtm(cov_a @ cov_b)
    if np.iscomplexobj(root):
        if np.abs(root.imag).max() > 1e-3:
            raise ValueError("Covariance square root is not real")
        root = root.real
    return float(np.sum((mean_a - mean_b) ** 2) + np.trace(cov_a + cov_b - 2 * root))


def fid(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    return frechet_distance(
        a.mean(0), np.cov(a, rowvar=False), b.mean(0), np.cov(b, rowvar=False)
    )


def kid(a: np.ndarray, b: np.ndarray, seed: int = SEED) -> dict:
    """Unbiased squared MMD with the cubic polynomial kernel (Binkowski et al.,
    2018), averaged over random subsets of each set."""
    random = np.random.default_rng(seed)
    a, b = a.astype(np.float64), b.astype(np.float64)
    dimension, size = a.shape[1], min(KID_SUBSET_SIZE, len(a), len(b))
    values = []
    for _ in range(KID_SUBSETS):
        x = a[random.choice(len(a), size, replace=False)]
        y = b[random.choice(len(b), size, replace=False)]
        kxx = (x @ x.T / dimension + 1) ** 3
        kyy = (y @ y.T / dimension + 1) ** 3
        kxy = (x @ y.T / dimension + 1) ** 3
        off = size * (size - 1)
        values.append(
            (kxx.sum() - np.trace(kxx)) / off
            + (kyy.sum() - np.trace(kyy)) / off
            - 2 * kxy.mean()
        )
    return {"mean": float(np.mean(values)), "sd": float(np.std(values))}


def _pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.astype(np.float64), b.astype(np.float64)
    squared = (a**2).sum(1)[:, None] + (b**2).sum(1)[None] - 2 * a @ b.T
    return np.sqrt(np.maximum(squared, 0))


def prdc(real: np.ndarray, fake: np.ndarray, k: int = PRDC_K) -> dict:
    """Precision, recall, density and coverage (Naeem et al., ICML 2020)."""
    real_radius = np.sort(_pairwise(real, real), axis=1)[:, k]  # column 0 is self
    fake_radius = np.sort(_pairwise(fake, fake), axis=1)[:, k]
    cross = _pairwise(real, fake)  # [real, fake]
    inside_real = cross <= real_radius[:, None]
    return {
        "precision": float(inside_real.any(axis=0).mean()),
        "recall": float((cross <= fake_radius[None]).any(axis=1).mean()),
        "density": float(inside_real.sum() / (k * len(fake))),
        "coverage": float((cross.min(axis=1) <= real_radius).mean()),
    }


def balanced(is_nc: np.ndarray, per_class: int, seed: int) -> np.ndarray:
    random = np.random.default_rng(seed)
    return np.sort(
        np.concatenate(
            [
                random.choice(np.flatnonzero(is_nc == c), per_class, replace=False)
                for c in (0, 1)
            ]
        )
    )


def distribution_block(fake, fake_is_nc, real, real_is_nc) -> dict:
    """FID, KID and PRDC of one generated set against the held-out slices.

    A cohort with one class only (B2b, if one GAN has no result) is compared
    with that class's held-out slices only.
    """
    present = sorted(set(fake_is_nc.tolist()))
    if len(present) == 1:
        c = present[0]
        same = real[real_is_nc == c]
        draws = [
            prdc(
                same[
                    np.random.default_rng(SEED + d).choice(
                        len(same), min(len(fake), len(same)), replace=False
                    )
                ],
                fake,
            )
            for d in range(PRDC_DRAWS)
        ]
        value = fid(fake, same)
        return {
            "classes": [LABELS[c]],
            "fid": value,
            "kid": kid(fake, same),
            "prdc": {
                m: {
                    "mean": float(np.mean([x[m] for x in draws])),
                    "sd": float(np.std([x[m] for x in draws])),
                }
                for m in draws[0]
            },
            "per_class_fid": {LABELS[c]: value},
        }
    per_class = int(min(np.sum(real_is_nc == 0), np.sum(real_is_nc == 1)))
    pool = balanced(real_is_nc, per_class, SEED)
    draws = [
        prdc(real[balanced(real_is_nc, len(fake) // 2, SEED + d)], fake)
        for d in range(PRDC_DRAWS)
    ]
    block = {
        "fid": fid(fake, real[pool]),
        "kid": kid(fake, real[pool]),
        "prdc": {
            m: {
                "mean": float(np.mean([x[m] for x in draws])),
                "sd": float(np.std([x[m] for x in draws])),
            }
            for m in draws[0]
        },
        "per_class_fid": {
            label: fid(fake[fake_is_nc == c], real[real_is_nc == c])
            for c, label in enumerate(LABELS)
        },
    }
    return block


# Memorisation --------------------------------------------------------------


def bootstrap_mean_ci(values: np.ndarray, seed: int = SEED) -> list[float]:
    random = np.random.default_rng(seed)
    means = values[random.integers(0, len(values), (BOOTSTRAP, len(values)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def training_preference(train, heldout, calibration_train, calibration_heldout):
    """Mean log(nearest-training / nearest-held-out) of the samples, minus the
    same quantity for real held-out slices; negative means closer to training
    data than held-out data would be. Bootstrap CI over both sets."""
    generated = np.log(train / heldout)
    real = np.log(calibration_train / calibration_heldout)
    random = np.random.default_rng(SEED)
    g = generated[random.integers(0, len(generated), (BOOTSTRAP, len(generated)))]
    r = real[random.integers(0, len(real), (BOOTSTRAP, len(real)))]
    differences = g.mean(1) - r.mean(1)
    difference = float(generated.mean() - real.mean())
    return {
        "generated_log_ratio": float(generated.mean()),
        "calibration_log_ratio": float(real.mean()),
        "difference": difference,
        "difference_ci95": [
            float(np.percentile(differences, 2.5)),
            float(np.percentile(differences, 97.5)),
        ],
        # As a percentage: how much closer to training than held-out slices.
        "closer_to_training_percent": float(100 * (1 - np.exp(difference))),
    }


# Membership inference ------------------------------------------------------


def auc(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> float:
    """P(a member scores lower than a non-member); lower loss means member.
    Ties count one half (Mann-Whitney U from average ranks)."""
    ranks = stats.rankdata(np.concatenate([member_scores, nonmember_scores]))
    n_m, n_n = len(member_scores), len(nonmember_scores)
    nonmember_rank_sum = ranks[n_m:].sum()
    return float((nonmember_rank_sum - n_n * (n_n + 1) / 2) / (n_m * n_n))


def tpr_at_fpr(member_scores, nonmember_scores, fpr: float = 0.01) -> float:
    """Share of members flagged when the threshold flags `fpr` of non-members."""
    threshold = np.quantile(nonmember_scores, fpr)
    return float(np.mean(member_scores < threshold))


def membership_block(members, nonmembers, control_members, control_nonmembers):
    """AUC and TPR at 1% FPR, and the AUC difference from the SD control with a
    paired bootstrap (the same images are resampled for both models)."""
    random = np.random.default_rng(SEED)
    differences = []
    for _ in range(BOOTSTRAP):
        m = random.integers(0, len(members), len(members))
        n = random.integers(0, len(nonmembers), len(nonmembers))
        differences.append(
            auc(members[m], nonmembers[n])
            - auc(control_members[m], control_nonmembers[n])
        )
    value = auc(members, nonmembers)
    control = auc(control_members, control_nonmembers)
    return {
        "auc": value,
        "tpr_at_1pct_fpr": tpr_at_fpr(members, nonmembers),
        "control_auc": control,
        "auc_minus_control": value - control,
        "auc_minus_control_ci95": [
            float(np.percentile(differences, 2.5)),
            float(np.percentile(differences, 97.5)),
        ],
    }


# Main ------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--cohorts", type=Path, required=True)
    parser.add_argument(
        "--membership",
        type=Path,
        action="append",
        default=[],
        help="folder written by membership.py; one per model, SD's among them",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    real = np.load(args.real / "private-real-features.npz")
    real_report = json.loads((args.real / "report.json").read_text())
    cohorts = np.load(args.cohorts / "private-cohort-features.npz")
    listing = json.loads((args.cohorts / "private-cohorts.json").read_text())
    names = list(listing)

    report = {
        "protocol": "docs/final-evaluation-protocol.md",
        "mode": real_report["mode"],
        "test_split_read": real_report["test_split_read"],
        "real_reference": {
            "heldout_to_train_pixel_rmse": real_report["heldout_to_train_pixel_rmse"],
            "calibration_log_ratio": real_report["calibration_log_ratio"],
        },
        "floor": {},
        "models": {},
    }
    for space in ("inception", "vae"):
        report["floor"][space] = distribution_block(
            real[f"floor_{space}"],
            real["floor_is_nc"],
            real[f"heldout_{space}"],
            real["heldout_is_nc"],
        )
    for name in names:
        entry = {"samples": len(listing[name]["files"])}
        for space in ("inception", "vae"):
            entry[space] = distribution_block(
                cohorts[f"{name}_{space}"],
                cohorts[f"{name}_is_nc"],
                real[f"heldout_{space}"],
                real["heldout_is_nc"],
            )
        pixel_train = cohorts[f"{name}_pixel_train"]
        entry["m1_nearest_training_rmse"] = {
            "mean": float(pixel_train.mean()),
            "mean_ci95": bootstrap_mean_ci(pixel_train),
            "min": float(pixel_train.min()),
        }
        entry["m2_training_preference"] = {
            space: training_preference(
                cohorts[f"{name}_{space}_train"],
                cohorts[f"{name}_{space}_heldout"],
                real[f"calibration_{space}_train"],
                real[f"calibration_{space}_heldout"],
            )
            for space in ("pixel", "vae")
        }
        report["models"][name] = entry
        print(name, "done", flush=True)

    if args.membership:
        scores = {}
        for folder in args.membership:
            meta = json.loads((folder / "report.json").read_text())
            data = np.load(folder / "private-scores.npz")
            scores[meta["model"]] = (data, meta)
        if "sd" not in scores:
            raise SystemExit("The SD control's membership scores are required")
        control, _ = scores["sd"]
        report["membership"] = {}
        for model, (data, meta) in scores.items():
            block = membership_block(
                data["member_primary"],
                data["nonmember_primary"],
                control["member_primary"],
                control["nonmember_primary"],
            )
            block["auc_by_timestep"] = {
                str(int(t)): auc(
                    data["member_by_t"][:, j], data["nonmember_by_t"][:, j]
                )
                for j, t in enumerate(data["timesteps"])
            }
            report["membership"][model] = block
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", args.output / "metrics.json", "for", ", ".join(names))


if __name__ == "__main__":
    main()
