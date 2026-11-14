# Ultralytics and numpy calls return partly unknown types, so silence only those rules here;
# every other strict check still applies.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
"""Freeze the chosen operating points, then evaluate them once on the held-out test split.

Two commands:

  freeze  Reads the validation split only. Records the candidate's thresholds, finds the
          baseline's thresholds at the same biopsy burden, and writes assets/selection.json
          with "final": false. Add --final to mark the record as the one to test.

    uv run python scripts/final_test.py freeze --candidate A0 --baseline B0 \\
        --seed 47041585 --tau-m 0.07 [--tau-c 0.05] [--final]

  test    Applies the recorded thresholds, unchanged, to a split and writes
          assets/test-results.md and assets/test-results.json. On the test split it refuses
          unless the record is final, the checkpoints still match their recorded hashes, and
          no test result exists yet. --split val is a dry run on validation that has none
          of those limits and writes to artifacts/dry-run/ instead of assets/.

    uv run python scripts/final_test.py test --split val      # dry run
    uv run python scripts/final_test.py test                  # the one test evaluation

Routing uses the lenient rule with the benign threshold at its floor (0.001), so a record
holds `tau_m` and an optional clear guard `tau_c`. Nothing is tuned here: the test split
never influences a threshold. Patient-level 95% intervals resample whole patients.
"""

import argparse
import csv
import json
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from yolo_lesion_47041585.helpers.evaluation import (
    bootstrap_by_patient,
    percentile_interval,
)
from yolo_lesion_47041585.helpers.policy import (
    ImageRouting,
    PolicySpec,
    evaluate_policy,
)
from yolo_lesion_47041585.helpers.tradeoff import best_at_burden
from yolo_lesion_47041585.helpers.validation import (
    THRESHOLDS,
    arrays_for,
    cached_predictions,
    file_sha256,
    patient_groups,
    read_targets,
    sweep,
)
from yolo_lesion_47041585.predict import Detection, load_run

RULE = "lenient"
BENIGN_FLOOR = float(THRESHOLDS[0])  # tau_b: any benign box can clear an image
CLASS_NAMES = ("Benign", "Malignant")
ROUTES = ("clear", "monitor", "urgent")
TARGETS = {
    "sensitivity": 0.90,
    "benign_escalation": 0.30,
    "iou": 0.80,
}  # from the feasibility review
METRICS = (
    ("sensitivity", "malignant detected and alerted"),
    ("benign_urgent", "benign routed urgent (biopsy burden)"),
    ("benign_monitor", "benign routed monitor"),
    ("benign_clear", "benign routed clear"),
    ("benign_escalation", "benign escalation (urgent + monitor)"),
    ("malignant_monitor", "malignant routed monitor"),
    ("malignant_clear", "malignant routed clear (safety cost)"),
    ("coverage_benign", "benign targets matched (IoU >= 0.5)"),
    ("coverage_malignant", "malignant targets matched (IoU >= 0.5)"),
    ("iou_benign", "mean matched IoU, benign"),
    ("iou_malignant", "mean matched IoU, malignant"),
)  # fmt: skip


@dataclass(frozen=True)
class PerImage:
    """One split's per-image outcomes under a frozen policy, in image-ID order."""

    ids: list[str]
    true_class: NDArray[np.int_]
    route: NDArray[np.int_]  # index into ROUTES
    detected: NDArray[np.bool_]
    iou: NDArray[np.float64]  # matched IoU, nan where the target is not matched


def per_image(routings: Sequence[ImageRouting]) -> PerImage:
    return PerImage(
        ids=[r.image_id for r in routings],
        true_class=np.array([r.true_class for r in routings]),
        route=np.array([ROUTES.index(r.route) for r in routings]),
        detected=np.array([r.detected for r in routings]),
        iou=np.array(
            [np.nan if r.matched_iou is None else r.matched_iou for r in routings]
        ),
    )


def metrics(data: PerImage, index: NDArray[np.int_]) -> dict[str, float | None]:
    """Every reported rate on the images at `index`; None where a class has no matched box."""
    true, route, detected, iou = (
        data.true_class[index],
        data.route[index],
        data.detected[index],
        data.iou[index],
    )
    out: dict[str, float | None] = {}
    for class_id, name in ((0, "benign"), (1, "malignant")):
        here = true == class_id
        if not here.any():
            return {key: None for key, _ in METRICS}
        for r, label in enumerate(ROUTES):
            out[f"{name}_{label}"] = float((route[here] == r).mean())
        out[f"coverage_{name}"] = float(detected[here].mean())
        matched = here & detected
        out[f"iou_{name}"] = float(iou[matched].mean()) if matched.any() else None
    out["sensitivity"] = out["coverage_malignant"]
    out["benign_escalation"] = (out["benign_urgent"] or 0.0) + (
        out["benign_monitor"] or 0.0
    )

    return out


def spec_of(record: Mapping[str, Any]) -> PolicySpec:
    return PolicySpec(
        float(record["tau_m"]), float(record["tau_b"]), RULE, record.get("tau_c")
    )


def load_split(
    data: Path, split: str, record: Mapping[str, Any], cache_dir: Path, runs_dir: Path, device: str | None,
) -> tuple[list[Detection], PerImage]:  # fmt: skip
    """Run one recorded model on a split and route every image under its frozen thresholds."""
    targets = read_targets(data / "labels" / split)
    images = sorted((data / "images" / split).glob("*.jpg"))
    predictions = cached_predictions(
        runs_dir / record["run"] / "weights" / "best.pt", images, cache_dir, device
    )
    evaluation = evaluate_policy(predictions, targets, spec_of(record))

    return targets, per_image(evaluation.images)


def object_detection_ap(
    checkpoint: Path, dataset_yaml: Path, split: str, device: str | None
) -> dict[str, list[float]]:
    """Per-class AP50 and AP50:95 from the Ultralytics validator.

    The image size, confidence floor, NMS IoU and detection limit come from the run's saved
    `config.resolved.yaml` (through `load_run`), the same settings the predictions use.
    """
    model, settings = load_run(checkpoint)
    with tempfile.TemporaryDirectory() as scratch:
        result = model.val(
            data=str(dataset_yaml), split=split, imgsz=settings.imgsz, batch=16, conf=settings.conf,
            iou=settings.iou, max_det=settings.max_det, device=device, plots=False, verbose=False,
            project=scratch, name="val",
        )  # fmt: skip
    box = result.box
    ap = np.asarray(box.all_ap)  # (classes present, 10 IoU thresholds from 0.5 to 0.95)
    out: dict[str, list[float]] = {
        "ap50": [float("nan")] * 2,
        "ap50_95": [float("nan")] * 2,
    }
    for row, class_id in enumerate(np.asarray(box.ap_class_index)):
        out["ap50"][int(class_id)] = float(ap[row, 0])
        out["ap50_95"][int(class_id)] = float(ap[row].mean())

    return out


def pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}%}"


def interval(draws: Sequence[float], percent: bool = True) -> str:
    low, high = percentile_interval(draws)

    return f"[{low:.0%}, {high:.0%}]" if percent else f"[{low:.3f}, {high:.3f}]"


# ----------------------------------------------------------------------------------------------
# freeze


def freeze(args: argparse.Namespace) -> int:
    runs_dir: Path = args.runs_dir
    data: Path = args.data
    targets = read_targets(data / "labels" / "val")
    images = sorted((data / "images" / "val").glob("*.jpg"))
    is_malignant = np.array([t.class_id == 1 for t in targets])
    everyone = np.arange(len(targets))

    runs: dict[str, dict[str, Any]] = {}
    arrays = {}

    for role, condition in (("candidate", args.candidate), ("baseline", args.baseline)):
        run = f"{condition}-s{args.seed}"
        checkpoint = runs_dir / run / "weights" / "best.pt"
        predictions = cached_predictions(
            checkpoint, images, args.cache_dir, args.device
        )
        arrays[role] = arrays_for(run, predictions, targets)
        runs[role] = {
            "condition": condition,
            "run": run,
            "checkpoint": str(checkpoint),
            "sha256": file_sha256(checkpoint),
        }

    candidate_sweep = sweep(arrays["candidate"], is_malignant, everyone, args.tau_c)
    index = int(np.argmin(np.abs(THRESHOLDS - args.tau_m)))
    burden = float(candidate_sweep.benign_urgent[index])
    baseline_sweep = sweep(arrays["baseline"], is_malignant, everyone, args.tau_c)
    baseline_index = best_at_burden(baseline_sweep, burden)

    runs["candidate"] |= {
        "tau_m": args.tau_m,
        "tau_b": BENIGN_FLOOR,
        "tau_c": args.tau_c,
    }
    runs["baseline"] |= {
        "tau_m": float(THRESHOLDS[baseline_index]),
        "tau_b": BENIGN_FLOOR,
        "tau_c": args.tau_c,
    }

    validation: dict[str, dict[str, float | None]] = {}
    for role in ("candidate", "baseline"):
        _, outcomes = load_split(
            data, "val", runs[role], args.cache_dir, runs_dir, args.device
        )
        validation[role] = metrics(outcomes, everyone)

    summary = json.loads(args.summary.read_text()) if args.summary.is_file() else {}
    record = {
        "final": bool(args.final),
        "frozen_on": "validation",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rule": RULE,
        "baseline_threshold_rule": (
            "the baseline's tau_m is the one with the best validation sensitivity at a benign urgent rate "
            "at or below the candidate's, so both are compared at the same biopsy burden"
        ),
        "candidate": runs["candidate"],
        "baseline": runs["baseline"],
        "validation_at_frozen_thresholds": validation,
        "dataset": {
            k: summary.get(k)
            for k in ("split_seed", "raw_metadata_sha256", "split_manifest_sha256")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2))

    print(f"{'FINAL' if args.final else 'draft'} record written to {args.output}")
    for role in ("candidate", "baseline"):
        v, r = validation[role], runs[role]
        print(f"  {role:9s} {r['run']}: tau_m {r['tau_m']:.2f}"
              + (f", tau_c {r['tau_c']}" if r["tau_c"] is not None else "")
              + f" -> validation sensitivity {pct(v['sensitivity'])}, benign urgent {pct(v['benign_urgent'])}, "
              f"malignant cleared {pct(v['malignant_clear'])}")  # fmt: skip
    if not args.final:
        print(
            "Not final: `test` will refuse the test split until you rerun freeze with --final."
        )

    return 0


# ----------------------------------------------------------------------------------------------
# test


def run_test(args: argparse.Namespace) -> int:
    record = json.loads(args.record.read_text())
    on_test = args.split == "test"
    out_dir = args.assets if on_test else Path("artifacts/dry-run")
    results_json = out_dir / "test-results.json"

    if on_test:
        if not record["final"]:
            raise SystemExit(
                f"{args.record} is not final. Freeze with --final first; the test split is evaluated once."
            )
        if results_json.exists() and not args.again:
            raise SystemExit(
                f"{results_json} already exists: the test split has been evaluated. Use --again to overwrite it."
            )
        for role in ("candidate", "baseline"):
            r = record[role]
            checkpoint = args.runs_dir / r["run"] / "weights" / "best.pt"
            if file_sha256(checkpoint) != r["sha256"]:
                raise SystemExit(
                    f"{checkpoint} no longer matches the hash recorded in {args.record}."
                )

    cache_dir = args.cache_dir or Path("artifacts/detection-cache") / args.split

    outcomes: dict[str, PerImage] = {}
    targets: list[Detection] = []
    for role in ("candidate", "baseline"):
        targets, outcomes[role] = load_split(
            args.data, args.split, record[role], cache_dir, args.runs_dir, args.device
        )
    ids = outcomes["candidate"].ids
    if ids != outcomes["baseline"].ids or ids != [t.image_id for t in targets]:
        raise AssertionError("candidate and baseline do not cover the same images")

    everyone = np.arange(len(ids))
    groups = patient_groups(args.manifest, ids)
    n_malignant = int((outcomes["candidate"].true_class == 1).sum())
    prevalence = n_malignant / len(ids)

    observed = {role: metrics(outcomes[role], everyone) for role in outcomes}

    def statistic(indices: list[int]) -> dict[str, float | None]:
        index = np.array(indices)
        got = {role: metrics(outcomes[role], index) for role in outcomes}
        out: dict[str, float | None] = {}
        for key, _ in METRICS:
            a, b = got["candidate"][key], got["baseline"][key]
            out[f"candidate:{key}"], out[f"baseline:{key}"] = a, b
            out[f"diff:{key}"] = None if a is None or b is None else a - b

        return out

    boot = bootstrap_by_patient(groups, statistic, repeats=args.repeats)
    ap = {
        role: object_detection_ap(
            args.runs_dir / record[role]["run"] / "weights" / "best.pt",
            args.data / "dataset.yaml",
            args.split,
            args.device,
        )
        for role in outcomes
    }

    cand, base = record["candidate"], record["baseline"]
    lines = [
        f"# {'Test' if on_test else 'Validation (dry run)'} results\n",
        f"Split: {args.split}, {len(ids)} images ({n_malignant} malignant, {len(ids) - n_malignant} benign) from {len(groups)} patients. "
        f"Thresholds were frozen on validation ({record['created']}, final: {record['final']}) and applied unchanged. "
        f"Intervals are patient-level 95% ({boot.n_valid} of {boot.n_repeats} resamples valid).\n",
        "## Frozen settings\n",
        "| | candidate | baseline |", "|---|---|---|",
        f"| run | {cand['run']} | {base['run']} |",
        f"| checkpoint sha256 | {cand['sha256'][:12]} | {base['sha256'][:12]} |",
        f"| tau_m | {cand['tau_m']:.2f} | {base['tau_m']:.2f} |",
        f"| tau_b | {cand['tau_b']} | {base['tau_b']} |",
        f"| clear guard tau_c | {cand['tau_c']} | {base['tau_c']} |",
        "",
        "## Operating-point results\n",
        f"| quantity | {cand['condition']} | {base['condition']} | {cand['condition']} minus {base['condition']} |", "|---|---:|---:|---:|",
    ]  # fmt: skip
    for key, label in METRICS:
        iou = key.startswith("iou")
        c, b = observed["candidate"][key], observed["baseline"][key]

        def fmt(v: float | None, iou: bool = iou) -> str:
            return pct(v) if not iou else ("n/a" if v is None else f"{v:.3f}")

        c_draws, b_draws = boot.draws[f"candidate:{key}"], boot.draws[f"baseline:{key}"]
        cells = [
            f"{fmt(c)} {interval(c_draws, not iou)}" if c_draws else fmt(c),
            f"{fmt(b)} {interval(b_draws, not iou)}" if b_draws else fmt(b),
        ]
        d_draws = boot.draws[f"diff:{key}"]
        if c is None or b is None or not d_draws:
            cells.append("n/a")
        else:
            lo, hi = percentile_interval(d_draws)
            scale, unit = (1, "") if iou else (100, " points")
            cells.append(
                f"{scale * (c - b):+.{3 if iou else 1}f}{unit} [{scale * lo:+.{3 if iou else 0}f}, {scale * hi:+.{3 if iou else 0}f}]"
            )
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += ["\n## Diagnosis by action (image counts)\n"]
    for role, name in (
        ("candidate", cand["condition"]),
        ("baseline", base["condition"]),
    ):
        d = outcomes[role]
        lines += [
            f"**{name}**\n",
            "| true class | clear | monitor | urgent | total |",
            "|---|---:|---:|---:|---:|",
        ]
        for class_id, label in enumerate(CLASS_NAMES):
            row = [
                int(((d.true_class == class_id) & (d.route == r)).sum())
                for r in range(3)
            ]
            lines.append(f"| {label} | {row[0]} | {row[1]} | {row[2]} | {sum(row)} |")
        lines.append("")

    lines += [
        "## Detector average precision (Ultralytics validator, same split)\n",
        "| class | "
        + " | ".join(
            f"{n} AP50 | {n} AP50:95" for n in (cand["condition"], base["condition"])
        )
        + " |",
        "|---|---:|---:|---:|---:|",
    ]
    for class_id, label in enumerate(CLASS_NAMES):
        lines.append(
            f"| {label} | "
            + " | ".join(
                f"{ap[r]['ap50'][class_id]:.3f} | {ap[r]['ap50_95'][class_id]:.3f}"
                for r in ("candidate", "baseline")
            )
            + " |"
        )

    c = observed["candidate"]
    n_ben = len(ids) - n_malignant
    caught, urgent, cleared = (
        c[k] or 0.0 for k in ("sensitivity", "benign_urgent", "malignant_clear")
    )

    def verdict(ok: bool) -> str:
        return "met" if ok else "NOT met"

    iou_b, iou_m = c["iou_benign"], c["iou_malignant"]
    iou_line = (
        f"- Mean matched IoU at least {TARGETS['iou']}: benign {iou_b:.3f} ({verdict(iou_b >= TARGETS['iou'])}), "
        f"malignant {iou_m:.3f} ({verdict(iou_m >= TARGETS['iou'])}), over matched targets only; coverage is in the table above."
        if iou_b is not None and iou_m is not None
        else "- Mean matched IoU target: undefined, a class has no matched box."
    )
    lines += [
        f"\n## Targets for {cand['condition']} (feasibility review)\n",
        f"- Malignant sensitivity at least {TARGETS['sensitivity']:.0%}: {pct(c['sensitivity'])}, {verdict(caught >= TARGETS['sensitivity'])}.",
        f"- Benign escalation at most {TARGETS['benign_escalation']:.0%} at the same point: {pct(c['benign_escalation'])}, {verdict((c['benign_escalation'] or 0) <= TARGETS['benign_escalation'])}.",
        iou_line,
        "- Latency target: see artifacts/latency.json (measured separately on the validation split).",
        f"\nPer 1,000 images at this split's malignant prevalence ({prevalence:.1%}):",
        f"- {caught * 1000 * prevalence:.0f} of {1000 * prevalence:.0f} malignant lesions are caught.",
        f"- {urgent * 1000 * (1 - prevalence):.0f} of {1000 * (1 - prevalence):.0f} benign lesions are sent to urgent review"
        f" ({urgent * n_ben / max(caught * n_malignant, 1e-9):.1f} benign per malignant caught).",
        f"- {cleared * 1000 * prevalence:.0f} malignant lesions are cleared with no follow-up.",
    ]  # fmt: skip

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test-results.md").write_text("\n".join(lines) + "\n")
    results_json.write_text(json.dumps({
        "split": args.split, "images": len(ids), "n_valid_resamples": boot.n_valid,
        "record": str(args.record), "observed": observed, "average_precision": ap,
        "intervals": {k: list(percentile_interval(v)) for k, v in boot.draws.items() if v},
    }, indent=2))  # fmt: skip
    for (
        role
    ) in outcomes:  # raw per-image outcomes stay local (artifacts/ is git-ignored)
        path = (
            Path("artifacts/routes") / f"{args.split}-{record[role]['condition']}.csv"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        d = outcomes[role]
        with path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["image_id", "true_class", "route", "detected", "matched_iou"]
            )
            for i, image_id in enumerate(d.ids):
                writer.writerow(
                    [
                        image_id,
                        int(d.true_class[i]),
                        ROUTES[d.route[i]],
                        bool(d.detected[i]),
                        "" if np.isnan(d.iou[i]) else f"{d.iou[i]:.4f}",
                    ]
                )
    print("\n".join(lines))
    print(f"\nwritten to {out_dir}/test-results.md and test-results.json")

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--runs-dir", type=Path, default=Path("runs"))
        p.add_argument("--data", type=Path, default=Path("artifacts/isic2018/yolo"))
        p.add_argument("--device", help="Ultralytics device, e.g. cpu, mps or 0")

    f = sub.add_parser("freeze", help="record the thresholds, using validation only")
    common(f)
    f.add_argument("--candidate", required=True)
    f.add_argument("--baseline", required=True)
    f.add_argument("--seed", type=int, required=True, help="training seed of both runs")
    f.add_argument(
        "--tau-m",
        type=float,
        required=True,
        help="the candidate's malignant alert threshold",
    )
    f.add_argument("--tau-c", type=float, help="optional clear guard, below tau_m")
    f.add_argument(
        "--final",
        action="store_true",
        help="mark the record as the one to evaluate on test",
    )
    f.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("artifacts/detection-cache/val"),
        help="validation detections cache",
    )
    f.add_argument(
        "--summary", type=Path, default=Path("artifacts/isic2018/summary.json")
    )
    f.add_argument("--output", type=Path, default=Path("assets/selection.json"))

    t = sub.add_parser("test", help="apply the recorded thresholds to a split")
    common(t)
    t.add_argument("--split", choices=("val", "test"), default="test")
    t.add_argument("--record", type=Path, default=Path("assets/selection.json"))
    t.add_argument(
        "--manifest", type=Path, default=Path("artifacts/isic2018/split_manifest.csv")
    )
    t.add_argument(
        "--cache-dir",
        type=Path,
        help="detections cache (default: artifacts/detection-cache/<split>)",
    )
    t.add_argument("--assets", type=Path, default=Path("assets"))
    t.add_argument("--repeats", type=int, default=2000)
    t.add_argument(
        "--again", action="store_true", help="overwrite an existing test result"
    )

    args = parser.parse_args(argv)
    return freeze(args) if args.command == "freeze" else run_test(args)


if __name__ == "__main__":
    raise SystemExit(main())
