"""Score every run on the validation split and write tables to read. No verdict.

Definitions, routing rules and justification are in artifacts/score-notes.md. assets/scores.md
holds only titles and tables.

    uv run python scripts/score.py --device 0                      # everything in runs/
    uv run python scripts/score.py --conditions A0 O35 --seeds 1 2  # just these
    uv run python scripts/score.py --tau-m 0.19 --tau-c 0.3          # quantify a chosen point
"""

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict

import numpy as np

from yolo_lesion_47041585.dataset import NormalizedBox
from yolo_lesion_47041585.helpers.policy import (
    build_evidence,
    search_policy,
    select_operating_point,
)
from yolo_lesion_47041585.helpers.tradeoff import analysis_report, write_tradeoff_csv
from yolo_lesion_47041585.helpers.validation import (
    RunArrays,
    arrays_for,
    cached_predictions,
    check_against_policy,
    patient_groups,
    read_targets,
    s30,
    sweep,
)
from yolo_lesion_47041585.predict import Detection

PREFERRED_ORDER = (
    "B0",
    "A0",
    "A0-wd",
    "O35",
    "O50",
    "A0-m",
    "A0-m-wd",
)  # display order only
RUN_NAME = re.compile(
    r"^(?P<condition>.+)-s(?P<seed>\d+)$"
)  # pilots end in "-pilot" and don't match
MEASURES = (
    "auc",
    "s30",
    "alert_at_90",
    "iou_benign",
    "iou_malignant",
    "strict_min_escalation",
    "map50_95",
    "train_minutes",
)


@dataclass(frozen=True)
class RunScores:
    """One run's scores.

    Attributes:
        condition (str): The condition, such as "A0-wd".
        seed (int): The training seed.
        auc (float): Image-level AUC of the highest malignant-class confidence.
        s30 (float): Sensitivity at a benign escalation of at most 30% (the lenient rule), 0.0 if
            no threshold gets within that rate.
        alert_at_90 (float | None): Lowest benign escalation at 90% sensitivity or more
            (the lenient rule), None if no threshold reaches it.
        iou_benign (float | None): Mean matched benign IoU at the Sens@<=30% point.
        iou_malignant (float | None): Mean matched malignant IoU there.
        strict_min_escalation (float): Lowest benign escalation reachable under the strict rule.
        map50_95 (float): Best validation mAP50:95 over training.
        train_minutes (float): Training time.
    """

    condition: str
    seed: int
    auc: float
    s30: float
    alert_at_90: float | None
    iou_benign: float | None
    iou_malignant: float | None
    strict_min_escalation: float
    map50_95: float
    train_minutes: float


class PredictionScores(TypedDict):
    """The prediction-based measures of one run, named as the `RunScores` fields they fill."""

    auc: float
    s30: float
    alert_at_90: float | None
    iou_benign: float | None
    iou_malignant: float | None
    strict_min_escalation: float


def prediction_scores(
    predictions: Sequence[Detection], ground_truth: Sequence[Detection]
) -> PredictionScores:
    """The prediction-based measures for one run (everything except training cost)."""
    evidence = build_evidence(predictions, ground_truth)
    malignant = [e.inputs.max_malignant_conf for e in evidence if e.true_class == 1]
    benign = [e.inputs.max_malignant_conf for e in evidence if e.true_class == 0]
    if not malignant or not benign:
        raise ValueError("Scoring needs at least one malignant and one benign image.")
    wins = sum((m > b) + 0.5 * (m == b) for m in malignant for b in benign)

    lenient = select_operating_point(
        search_policy(predictions, ground_truth, version="lenient")
    )
    strict = search_policy(predictions, ground_truth, version="strict")
    at_30 = lenient.workload_alternative
    at_90 = lenient.sensitivity_alternative
    return {
        "auc": wins / (len(malignant) * len(benign)),
        "s30": at_30.m if at_30 else 0.0,
        "alert_at_90": at_90.b if at_90 else None,
        "iou_benign": at_30.matched_iou_benign if at_30 else None,
        "iou_malignant": at_30.matched_iou_malignant if at_30 else None,
        "strict_min_escalation": min(p.b for p in strict),
    }


def training_cost(run_dir: Path) -> tuple[float, float]:
    """Best validation mAP50:95 and training minutes of a finished run."""
    with (run_dir / "results.csv").open(newline="") as file:
        best = max(float(r["metrics/mAP50-95(B)"]) for r in csv.DictReader(file))
    minutes = json.loads((run_dir / "result.json").read_text())["elapsed_seconds"] / 60
    return best, minutes


def discover(runs_dir: Path) -> dict[str, list[int]]:
    """Conditions and seeds of every finished run folder (pilots excluded)."""
    found: defaultdict[str, list[int]] = defaultdict(list)
    for folder in sorted(runs_dir.iterdir()):
        match = RUN_NAME.match(folder.name)
        if match and (folder / "weights" / "best.pt").is_file():
            found[match["condition"]].append(int(match["seed"]))
    return dict(found)


def ordered(conditions: Sequence[str]) -> list[str]:
    """Known conditions in the usual order, then any others alphabetically."""
    known = [c for c in PREFERRED_ORDER if c in conditions]
    return known + sorted(c for c in conditions if c not in PREFERRED_ORDER)


def summarise(values: Sequence[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def by_condition(
    runs: Sequence[RunScores],
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Mean, spread and range of each measure over a condition's seeds."""
    out: dict[str, dict[str, dict[str, float | int]]] = {}
    for condition in ordered(sorted({r.condition for r in runs})):
        group = [r for r in runs if r.condition == condition]
        stats: dict[str, dict[str, float | int]] = {}
        for measure in MEASURES:
            values = [v for r in group if (v := getattr(r, measure)) is not None]
            if values:
                stats[measure] = summarise(values)
        out[condition] = stats
    return out


def _pct(x: float) -> str:
    return f"{x:.0%}"


def _spread(stat: Mapping[str, float], percent: bool = False) -> str:
    """ "mean ± SD [min–max]", as percentages or to three decimals."""

    def fmt(v: float) -> str:
        return f"{v:.0%}" if percent else f"{v:.3f}"

    return f"{fmt(stat['mean'])} ± {fmt(stat['sd'])} [{fmt(stat['min'])}–{fmt(stat['max'])}]"


def render(
    runs: Sequence[RunScores],
    reference: str | None,
) -> str:
    """The report as Markdown: titles and tables only."""
    summary = by_condition(runs)
    lines = ["# Validation scores\n"]
    lines.append("## By condition: mean ± SD [min–max] over seeds\n")
    lines.append(
        "| condition | seeds | AUC | Sens@≤30% | Alert@≥90% | IoU benign | IoU malignant | strict min escal. | mAP50:95 | train min |"
    )
    lines.append("|---|---:|---|---|---|---:|---:|---:|---|---:|")
    for condition, s in summary.items():
        reach = s.get("alert_at_90")
        n_runs = sum(1 for r in runs if r.condition == condition)
        alert = (
            f"unreachable in all {n_runs}"
            if reach is None
            else f"{_spread(reach, percent=True)} ({int(reach['n'])} of {n_runs} reach it)"
        )
        lines.append(
            f"| {condition} | {int(s['auc']['n'])} | {_spread(s['auc'])} | {_spread(s['s30'], percent=True)} | {alert} "
            f"| {s['iou_benign']['mean']:.3f} | {s['iou_malignant']['mean']:.3f} "
            f"| {_pct(s['strict_min_escalation']['mean'])} | {_spread(s['map50_95'])} | {s['train_minutes']['mean']:.0f} |"
        )
    if reference in summary:
        base = summary[reference]
        lines.append(f"\n## Difference from {reference}\n")
        lines.append(
            f"| condition | AUC | Sens@≤30% (points) | mAP50:95 | train min ×{reference} |"
        )
        lines.append("|---|---|---|---|---|")
        base_auc = base["auc"]
        lines.append(
            f"| {reference} (range over its seeds) | {base_auc['min']:.3f}–{base_auc['max']:.3f} | "
            f"{_pct(base['s30']['min'])}–{_pct(base['s30']['max'])} | {base['map50_95']['min']:.3f}–{base['map50_95']['max']:.3f} | 1.0 |"
        )
        for condition, s in summary.items():
            if condition == reference:
                continue
            lines.append(
                f"| {condition} | {s['auc']['mean'] - base['auc']['mean']:+.3f} | {100 * (s['s30']['mean'] - base['s30']['mean']):+.1f} "
                f"| {s['map50_95']['mean'] - base['map50_95']['mean']:+.3f} | {s['train_minutes']['mean'] / base['train_minutes']['mean']:.1f} |"
            )
    lines.append("\n## Every run\n")
    lines.append(
        "| run | AUC | Sens@≤30% | Alert@≥90% | IoU benign | IoU malignant | strict min escal. | mAP50:95 | train min |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for condition in summary:
        for r in sorted(
            (r for r in runs if r.condition == condition), key=lambda r: r.seed
        ):
            alert = "unreachable" if r.alert_at_90 is None else _pct(r.alert_at_90)
            iou_b = "n/a" if r.iou_benign is None else f"{r.iou_benign:.3f}"
            iou_m = "n/a" if r.iou_malignant is None else f"{r.iou_malignant:.3f}"
            lines.append(
                f"| {r.condition}-s{r.seed} | {r.auc:.3f} | {_pct(r.s30)} | {alert} | {iou_b} | {iou_m} "
                f"| {_pct(r.strict_min_escalation)} | {r.map50_95:.3f} | {r.train_minutes:.1f} |"
            )
    return "\n".join(lines) + "\n"


def write_results_csv(runs: Sequence[RunScores], path: Path) -> None:
    """The by-condition summary as a table: mean, SD and range of each measure over seeds."""
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["condition", "measure", "n", "mean", "sd", "min", "max"])
        for condition, measures in by_condition(runs).items():
            for measure, v in measures.items():
                writer.writerow(
                    [
                        condition,
                        measure,
                        v["n"],
                        *(f"{v[k]:.4f}" for k in ("mean", "sd", "min", "max")),
                    ]
                )


def self_check() -> None:
    """Check the scoring on a small scenario with known answers."""
    box = NormalizedBox(0.5, 0.5, 0.4, 0.4)
    truths = [Detection(f"M{i}", box, 1, 1.0) for i in range(5)]
    truths += [Detection(f"B{i}", box, 0, 1.0) for i in range(5)]
    preds = [
        Detection(f"M{i}", box, 1, 0.8) for i in range(5)
    ]  # malignant images score 0.8
    for i in range(5):
        preds += [
            Detection(f"B{i}", box, 0, 0.9),
            Detection(f"B{i}", box, 1, 0.1),
        ]  # benign score 0.1
    scores = prediction_scores(preds, truths)
    assert (
        scores["auc"] == 1.0 and scores["s30"] == 1.0 and scores["alert_at_90"] == 0.0
    )
    assert (
        scores["strict_min_escalation"] == 1.0
    )  # two boxes per benign image, so strict never clears
    blind = [Detection(f"M{i}", box, 1, 0.5) for i in range(5)]
    for i in range(5):
        blind += [Detection(f"B{i}", box, 0, 0.9), Detection(f"B{i}", box, 1, 0.5)]
    assert prediction_scores(blind, truths)["auc"] == 0.5
    summary = summarise([0.7, 0.8, 0.9])
    assert (
        abs(summary["mean"] - 0.8) < 1e-9
        and summary["min"] == 0.7
        and summary["max"] == 0.9
    )
    print("self-check passed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--data", type=Path, default=Path("artifacts/isic2018/yolo"))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("artifacts/detection-cache/val")
    )
    parser.add_argument(
        "--conditions", nargs="+", help="default: every condition found in runs/"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="default: every seed found for each condition",
    )
    parser.add_argument(
        "--reference",
        default="A0",
        help="condition the difference tables compare against",
    )
    parser.add_argument("--baseline", default="B0", help="trade-off baseline condition")
    parser.add_argument(
        "--candidate", default="A0", help="trade-off candidate condition"
    )
    parser.add_argument(
        "--primary-seed",
        type=int,
        help="seed the trade-off is read from (default: the first seed run)",
    )
    parser.add_argument(
        "--tau-m",
        type=float,
        help="malignant alert threshold of a point to quantify (default: none)",
    )
    parser.add_argument(
        "--tau-c",
        type=float,
        help="optional clear guard for that point (default: none)",
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("artifacts/isic2018/split_manifest.csv")
    )
    parser.add_argument(
        "--assets",
        type=Path,
        default=Path("assets"),
        help="where scores.md goes (committed; figures come from scripts/visualise/)",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("artifacts"),
        help="where the plot-input CSVs go (tradeoff.csv, results-by-condition.csv)",
    )
    parser.add_argument("--repeats", type=int, default=2000, help="bootstrap resamples")
    parser.add_argument("--device", help="Ultralytics device, e.g. cpu, mps or 0")
    parser.add_argument("--output", type=Path, default=Path("artifacts/scores.json"))
    parser.add_argument(
        "--self-check", action="store_true", help="test the scoring and exit"
    )
    args = parser.parse_args(argv)

    if args.self_check:
        self_check()
        return 0

    found = discover(args.runs_dir)
    conditions = ordered(args.conditions if args.conditions else list(found))
    targets = read_targets(args.data / "labels" / "val")
    images = sorted((args.data / "images" / "val").glob("*.jpg"))
    malignant = sum(t.class_id == 1 for t in targets)

    runs: list[RunScores] = []
    arrays: dict[str, RunArrays] = {}
    is_malignant = np.array([t.class_id == 1 for t in targets])
    missing: list[str] = []
    for condition in conditions:
        for seed in args.seeds if args.seeds else sorted(found.get(condition, [])):
            name = f"{condition}-s{seed}"
            checkpoint = args.runs_dir / name / "weights" / "best.pt"
            if not checkpoint.is_file():
                missing.append(name)
                continue
            predictions = cached_predictions(
                checkpoint, images, args.cache_dir, args.device
            )
            best_map, minutes = training_cost(args.runs_dir / name)
            scores = prediction_scores(predictions, targets)
            arrays[name] = arrays_for(name, predictions, targets)
            worst = max(
                check_against_policy(
                    arrays[name], predictions, targets, is_malignant, guard
                )
                for guard in (None, 0.1)
            )
            if worst > 1e-9:
                raise AssertionError(
                    f"{name}: the fast sweep disagrees with the policy code by {worst:.1e}"
                )
            if (
                abs(
                    s30(sweep(arrays[name], is_malignant, np.arange(len(is_malignant))))
                    - (scores["s30"] or 0.0)
                )
                > 1e-9
            ):
                raise AssertionError(
                    f"{name}: Sens@<=30% differs between the sweep and the policy search"
                )
            runs.append(
                RunScores(
                    condition, seed, map50_95=best_map, train_minutes=minutes, **scores
                )
            )

            print(f"scored {name}", flush=True)
    if not runs:
        print("No finished runs found.")
        return 1

    if missing:
        print(f"Requested but not trained: {', '.join(missing)}")
    report = render(runs, args.reference)
    primary = (
        args.primary_seed
        if args.primary_seed is not None
        else min(r.seed for r in runs)
    )
    trade_off_extra: dict[str, object] = {}
    print("bootstrapping ...", flush=True)
    groups = patient_groups(args.manifest, [t.image_id for t in targets])
    seeds_of = {
        c: sorted(r.seed for r in runs if r.condition == c)
        for c in ordered(sorted({r.condition for r in runs}))
    }
    extra, trade_off_extra = analysis_report(
        arrays, list(seeds_of), seeds_of, is_malignant, groups, args.reference, args.baseline, args.candidate,
        primary, args.tau_m, args.tau_c, args.repeats,
    )  # fmt: skip
    report += "\n" + extra
    args.assets.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)
    write_tradeoff_csv(
        arrays,
        seeds_of,
        is_malignant,
        args.baseline,
        args.candidate,
        args.tables_dir / "tradeoff.csv",
    )
    write_results_csv(runs, args.tables_dir / "results-by-condition.csv")
    (args.assets / "scores.md").write_text(report)
    print("\n" + report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "validation": {"images": len(targets), "malignant": malignant},
                "reference_for_differences": args.reference,
                "runs": [asdict(r) for r in runs],
                "missing": missing,
                "conditions": by_condition(runs),
                "trade_off": trade_off_extra,
            },
            indent=2,
        )
    )
    args.output.with_suffix(".md").write_text(report)
    print(
        f"written to {args.output}, {args.output.with_suffix('.md')}, {args.assets}/scores.md and {args.tables_dir}/*.csv"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
