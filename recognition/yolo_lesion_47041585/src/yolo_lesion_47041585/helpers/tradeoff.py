"""
Trade-off between biopsy burden and sensitivity, from saved detections.

Builds the Markdown report sections (patient-level intervals, burden-limit tables, a chosen
operating point with its per-1,000 translation, and the clear-guard table) and the
`tradeoff.csv` plot input (the figure itself is drawn by `scripts/visualise/tradeoff.py`). Everything uses the lenient routing rule with the
benign threshold at its floor, as in `helpers/validation.py`. Nothing here reads test data.
"""

import csv
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from yolo_lesion_47041585.helpers.evaluation import bootstrap_by_patient, percentile_interval
from yolo_lesion_47041585.helpers.validation import (
    MAX_BENIGN_ESCALATION,
    THRESHOLDS,
    RunArrays,
    Sweep,
    auc,
    s30,
    sweep,
)

BURDEN_LIMITS = (0.10, 0.20, 0.30, 0.40, 0.50)
GUARDS = (None, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02)


def _interval(draws: Sequence[float], percent: bool = True) -> str:
    low, high = percentile_interval(draws)
    return f"[{low:.0%}, {high:.0%}]" if percent else f"[{low:.3f}, {high:.3f}]"


def _grid_index(value: float) -> int:
    """Index of the grid threshold nearest a value."""
    return int(np.argmin(np.abs(THRESHOLDS - value)))


def best_at_burden(result: Sweep, limit: float) -> int:
    """Index of the threshold with the highest sensitivity at a benign urgent rate within the limit
    (the lowest threshold of any tie, which clears the most)."""
    ok = result.benign_urgent <= limit + 1e-9
    if not ok.any():
        return len(THRESHOLDS) - 1
    candidates = np.flatnonzero(ok)
    return int(candidates[np.argmax(result.sensitivity[candidates])])


def _point_row(result: Sweep, i: int) -> str:
    return (
        f"| {THRESHOLDS[i]:.2f} | {result.sensitivity[i]:.0%} | {result.benign_urgent[i]:.0%} "
        f"| {result.benign_monitor[i]:.0%} | {result.malignant_monitor[i]:.0%} | {result.malignant_clear[i]:.0%} |"
    )


POINT_ROWS = (
    ("sensitivity", "malignant detected and alerted"), ("benign_urgent", "benign routed urgent (biopsy burden)"),
    ("benign_monitor", "benign routed monitor"), ("benign_clear", "benign routed clear"),
    ("malignant_monitor", "malignant routed monitor"), ("malignant_clear", "malignant routed clear (safety cost)"),
)  # fmt: skip

POINT_HEADER = (
    "| burden limit | tau_m | sensitivity | benign urgent (burden) | benign monitor | malignant monitor | malignant cleared |\n"
    "|---|---:|---:|---:|---:|---:|---:|"
)


def analysis_report(
    arrays: Mapping[str, RunArrays], conditions: Sequence[str], seeds_of: Mapping[str, Sequence[int]],
    is_malignant: NDArray[np.bool_], groups: list[list[int]], reference: str | None, baseline: str,
    candidate: str, primary_seed: int, tau_m: float | None, tau_c: float | None, repeats: int,
) -> tuple[str, dict[str, object]]:  # fmt: skip
    """Intervals, trade-off tables and (if chosen) the quantified operating point, as Markdown.

    Burden here is benign images routed urgent. The Sens@<=30% column of the intervals table
    keeps the plan's definition (urgent plus monitor); the two differ by the benign images
    the rule monitors, which is 1% or less at the points that matter.
    """
    base_name, cand_name = f"{baseline}-s{primary_seed}", f"{candidate}-s{primary_seed}"
    for name in (base_name, cand_name):
        if name not in arrays:
            raise ValueError(f"{name} was not scored; the trade-off needs both runs at --primary-seed")
    point_index = _grid_index(tau_m) if tau_m is not None else None
    prevalence = float(is_malignant.mean())
    everyone = np.arange(len(is_malignant))
    # the baseline is read at the same biopsy burden as the candidate's chosen point: its best
    # sensitivity among thresholds at or below that burden, fixed before any resampling
    base_index = None
    if point_index is not None:
        burden = float(sweep(arrays[cand_name], is_malignant, everyone, tau_c).benign_urgent[point_index])
        base_index = best_at_burden(sweep(arrays[base_name], is_malignant, everyone, tau_c), burden)

    def statistic(indices: list[int]) -> dict[str, float | None]:
        index = np.array(indices)
        if is_malignant[index].all() or not is_malignant[index].any():
            return {}
        out: dict[str, float | None] = {}
        for condition in conditions:
            aucs: list[float] = []
            s30s: list[float] = []
            for seed in seeds_of[condition]:
                run = arrays[f"{condition}-s{seed}"]
                aucs.append(auc(run, is_malignant, index))
                s30s.append(s30(sweep(run, is_malignant, index)))
            out[f"auc:{condition}"] = statistics.fmean(aucs)
            out[f"s30:{condition}"] = statistics.fmean(s30s)
        if point_index is not None and base_index is not None:
            for key, name, i in (("point", cand_name, point_index), ("base", base_name, base_index)):
                at = sweep(arrays[name], is_malignant, index, tau_c)
                for field, _ in POINT_ROWS:
                    out[f"{key}:{field}"] = float(getattr(at, field)[i])
        return out

    boot = bootstrap_by_patient(groups, statistic, repeats=repeats)
    lines = [
        f"## Patient-level 95% intervals ({boot.n_valid} of {boot.n_repeats} resamples valid)\n",
        f"| condition | AUC | AUC minus {reference} | Sens@≤30% | Sens@≤30% minus {reference} (points) |",
        "|---|---|---|---|---|",
    ]
    observed = {
        c: {
            "auc": statistics.fmean(auc(arrays[f"{c}-s{seed}"], is_malignant, everyone) for seed in seeds_of[c]),
            "s30": statistics.fmean(s30(sweep(arrays[f"{c}-s{seed}"], is_malignant, everyone)) for seed in seeds_of[c]),
        }
        for c in conditions
    }
    differences: dict[str, dict[str, dict[str, float]]] = {}
    for condition in conditions:
        a, s = boot.draws[f"auc:{condition}"], boot.draws[f"s30:{condition}"]
        cells = [f"{statistics.fmean(a):.3f} {_interval(a, False)}", "reference", f"{statistics.fmean(s):.0%} {_interval(s)}", "reference"]
        if reference is not None and reference in conditions and condition != reference:
            da = [x - y for x, y in zip(a, boot.draws[f"auc:{reference}"])]
            ds = [x - y for x, y in zip(s, boot.draws[f"s30:{reference}"])]
            lo, hi = percentile_interval(da)
            slo, shi = percentile_interval(ds)
            differences[condition] = {
                metric: {"difference": observed[condition][metric] - observed[reference][metric], "low": low, "high": high}
                for metric, (low, high) in (("auc", (lo, hi)), ("s30", (slo, shi)))
            }
            cells[1] = f"{statistics.fmean(da):+.3f} [{lo:+.3f}, {hi:+.3f}]"
            cells[3] = f"{100 * statistics.fmean(ds):+.1f} [{100 * slo:+.0f}, {100 * shi:+.0f}]"
        lines.append(f"| {condition} | " + " | ".join(cells) + " |")

    base, cand = sweep(arrays[base_name], is_malignant, everyone), sweep(arrays[cand_name], is_malignant, everyone)
    lines.append(f"\n## Trade-off: {baseline} and {candidate} at seed {primary_seed}\n")
    for name, result in ((baseline, base), (candidate, cand)):
        lines += [f"**{name}**\n", POINT_HEADER]
        for limit in BURDEN_LIMITS:
            i = best_at_burden(result, limit)
            lines.append(f"| ≤{limit:.0%} " + _point_row(result, i))
        lines.append("")

    detail: dict[str, object] = {}
    guard_i = point_index if point_index is not None else best_at_burden(cand, MAX_BENIGN_ESCALATION)
    if point_index is not None and base_index is not None:
        at = sweep(arrays[cand_name], is_malignant, everyone, tau_c)
        at_base = sweep(arrays[base_name], is_malignant, everyone, tau_c)
        n_mal = round(1000 * prevalence)
        n_ben = 1000 - n_mal
        guard_text = "no guard" if tau_c is None else f"clear guard {tau_c}"
        lines += [
            f"## Chosen operating point: {candidate} seed {primary_seed}, tau_m {THRESHOLDS[point_index]:.2f}, {guard_text}\n",
            f"| quantity | {candidate} | {baseline} (same burden) | {candidate} minus {baseline} |", "|---|---:|---:|---:|",
        ]
        for field, label in POINT_ROWS:
            c_draws, b_draws = boot.draws[f"point:{field}"], boot.draws[f"base:{field}"]
            diff = [x - y for x, y in zip(c_draws, b_draws)]
            d_lo, d_hi = percentile_interval(diff)
            c_seen, b_seen = float(getattr(at, field)[point_index]), float(getattr(at_base, field)[base_index])
            lines.append(
                f"| {label} | {c_seen:.1%} {_interval(c_draws)} | {b_seen:.1%} {_interval(b_draws)} "
                f"| {100 * (c_seen - b_seen):+.1f} points [{100 * d_lo:+.0f}, {100 * d_hi:+.0f}] |"
            )
        detected = float(at.sensitivity[point_index])
        urgent = float(at.benign_urgent[point_index])
        cleared_m = float(at.malignant_clear[point_index])
        detail = {"tau_m": float(THRESHOLDS[point_index]), "tau_c": tau_c, "sensitivity": detected,
                  "baseline_tau_m": float(THRESHOLDS[base_index]), "baseline_sensitivity": float(at_base.sensitivity[base_index]),
                  "benign_urgent": urgent, "malignant_clear": cleared_m, "per_1000": {"malignant": n_mal, "benign": n_ben}}  # fmt: skip

    lines += [
        f"## Clear guard at tau_m {THRESHOLDS[guard_i]:.2f} ({candidate}, seed {primary_seed})\n",
        "| tau_c | benign cleared | benign monitor | malignant monitor | malignant cleared |", "|---|---:|---:|---:|---:|",
    ]
    for guard in GUARDS:
        g = sweep(arrays[cand_name], is_malignant, everyone, guard)
        lines.append(
            f"| {'none' if guard is None else guard} | {g.benign_clear[guard_i]:.0%} | {g.benign_monitor[guard_i]:.0%} "
            f"| {g.malignant_monitor[guard_i]:.0%} | {g.malignant_clear[guard_i]:.0%} |"
        )
    return "\n".join(lines) + "\n", {
        "point": detail, "n_valid_resamples": boot.n_valid, "reference": reference, "differences_vs_reference": differences,
    }


def write_tradeoff_csv(
    arrays: Mapping[str, RunArrays], seeds_of: Mapping[str, Sequence[int]], is_malignant: NDArray[np.bool_],
    baseline: str, candidate: str, path: Path,
) -> None:  # fmt: skip
    """Every threshold's rates for the baseline and candidate, one block per seed.

    `scripts/visualise/tradeoff.py` plots this file.
    """
    everyone = np.arange(len(is_malignant))
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["condition", "seed", "tau_m", "sensitivity", "benign_urgent", "benign_monitor",
                         "benign_clear", "malignant_urgent", "malignant_monitor", "malignant_clear"])  # fmt: skip
        for condition in (baseline, candidate):
            for seed in seeds_of[condition]:
                r = sweep(arrays[f"{condition}-s{seed}"], is_malignant, everyone)
                for i, t in enumerate(THRESHOLDS):
                    writer.writerow([condition, seed, f"{t:.3f}", *(f"{float(x[i]):.4f}" for x in (
                        r.sensitivity, r.benign_urgent, r.benign_monitor, r.benign_clear,
                        r.malignant_urgent, r.malignant_monitor, r.malignant_clear))])  # fmt: skip
