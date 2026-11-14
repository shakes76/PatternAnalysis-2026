"""Find and draw candidate failure cases for the recorded candidate and baseline.

Reads the thresholds from the selection record (assets/selection.json), so every route here
is the one `final_test.py` applies. For each category in the plan it picks the clearest case,
prints the numbers, and draws a side-by-side panel (baseline left, candidate right) into
artifacts/autopsy/ using scripts/visualise/panels.py. You choose which 3 to 5 to keep; the
picks are suggestions, not a selection. Detections come from the shared cache.

    uv run python scripts/autopsy.py                 # validation split, any record
    uv run python scripts/autopsy.py --split test    # needs a final record
"""

import argparse
import csv
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from yolo_lesion_47041585.helpers.evaluation import box_iou
from yolo_lesion_47041585.helpers.policy import PolicySpec, Reason, Route, route_case
from yolo_lesion_47041585.helpers.validation import cached_predictions, read_targets
from yolo_lesion_47041585.predict import Detection
from visualise.panels import PanelSide, draw_panel

NAMES = ("Benign", "Malignant")


@dataclass(frozen=True)
class Case:
    """One image as one model sees it, routed under that model's recorded thresholds."""

    image_id: str
    truth: Detection
    detections: tuple[Detection, ...]
    spec: PolicySpec

    @property
    def decision(self) -> tuple[Route, Reason]:
        return route_case(self.detections, self.spec)

    @property
    def route(self) -> Route:
        return self.decision[0]

    @property
    def benign(self) -> list[Detection]:
        return [d for d in self.detections if d.class_id == 0]

    def top_malignant(self) -> float:
        return max((d.confidence for d in self.detections if d.class_id == 1), default=0.0)

    def matched(self, class_id: int) -> float:
        """Highest confidence of a class-correct box overlapping the target at IoU 0.5 or more."""
        return max(
            (d.confidence for d in self.detections
             if d.class_id == class_id and box_iou(d.box, self.truth.box) >= 0.5),
            default=0.0,
        )  # fmt: skip

    def best_iou_of_top_box(self) -> float:
        top = max(self.detections, key=lambda d: d.confidence, default=None)
        return box_iou(top.box, self.truth.box) if top else 0.0


def summary(case: Case, label: str) -> str:
    top = sorted(case.detections, key=lambda d: -d.confidence)[:3]
    boxes = ", ".join(f"{NAMES[d.class_id].lower()} {d.confidence:.2f} (IoU {box_iou(d.box, case.truth.box):.2f})" for d in top) or "no detections"
    route, reason = case.decision
    return f"{label} [tau_m {case.spec.malignant_threshold:.2f}] -> {route} ({reason}): {boxes}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--record", type=Path, default=Path("assets/selection.json"))
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--data", type=Path, default=Path("artifacts/isic2018/yolo"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/isic2018/split_manifest.csv"))
    parser.add_argument("--cache-dir", type=Path, help="detections cache (default: artifacts/detection-cache/<split>)")
    parser.add_argument("--device", help="Ultralytics device, used only if detections are not cached")
    parser.add_argument("--out", type=Path, default=Path("artifacts/autopsy"))
    args = parser.parse_args(argv)

    record: dict[str, Any] = json.loads(args.record.read_text())
    if args.split == "test" and not record["final"]:
        raise SystemExit(f"{args.record} is not final; failure cases on the test split need the frozen record.")
    cache_dir = args.cache_dir or Path("artifacts/detection-cache") / args.split
    candidate, baseline = record["candidate"], record["baseline"]

    truths = {t.image_id: t for t in read_targets(args.data / "labels" / args.split)}
    images = sorted((args.data / "images" / args.split).glob("*.jpg"))
    with args.manifest.open(newline="") as file:
        patient = {r["isic_id"]: r["patient_id"] for r in csv.DictReader(file) if r["split"] == args.split}

    def cases(entry: dict[str, Any]) -> dict[str, Case]:
        predictions = cached_predictions(args.runs_dir / entry["run"] / "weights" / "best.pt", images, cache_dir, args.device)
        by_image: dict[str, list[Detection]] = {}
        for d in predictions:
            by_image.setdefault(d.image_id, []).append(d)
        spec = PolicySpec(float(entry["tau_m"]), float(entry["tau_b"]), record["rule"], entry.get("tau_c"))
        return {i: Case(i, t, tuple(by_image.get(i, [])), spec) for i, t in truths.items()}

    base, cand = cases(baseline), cases(candidate)
    malignant = [i for i, t in truths.items() if t.class_id == 1]
    benign = [i for i, t in truths.items() if t.class_id == 0]

    def pick(ids: list[str], key: Callable[[str], float]) -> str | None:
        return max(ids, key=key) if ids else None

    picks: dict[str, tuple[str | None, str]] = {
        "missed-malignancy": (
            pick([i for i in malignant if cand[i].route != "urgent" and cand[i].matched(0) > 0], lambda i: cand[i].matched(0)),
            "malignant lesion, well localised, but the candidate calls it benign and does not alert",
        ),
        "benign-false-alert": (
            pick([i for i in benign if cand[i].route == "urgent"], lambda i: cand[i].top_malignant()),
            "benign lesion that the candidate alerts as malignant with high confidence",
        ),
        "weak-or-no-detection": (
            pick([i for i in truths if not cand[i].benign and cand[i].route != "urgent"], lambda i: -len(cand[i].detections)),
            "the candidate finds no benign box and no alert, so nothing can clear the image",
        ),
        "poor-localisation": (
            pick([i for i in truths if cand[i].detections], lambda i: -cand[i].best_iou_of_top_box()),
            "the candidate's most confident box barely overlaps the annotated lesion",
        ),
        "baseline-misses-candidate-finds": (
            pick([i for i in malignant if base[i].route != "urgent" and cand[i].route == "urgent" and cand[i].matched(1) >= cand[i].spec.malignant_threshold], lambda i: cand[i].matched(1)),
            "malignant lesion the candidate alerts correctly and the baseline does not",
        ),
        "candidate-misses-baseline-finds": (
            pick([i for i in malignant if cand[i].route != "urgent" and base[i].route == "urgent" and base[i].matched(1) >= base[i].spec.malignant_threshold], lambda i: base[i].matched(1)),
            "malignant lesion the baseline alerts correctly and the candidate does not",
        ),
    }

    out = args.out / f"{candidate['condition']}-vs-{baseline['condition']}-{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    names = (baseline["condition"], candidate["condition"])
    print(f"{args.split} split, record {args.record}: {names[1]} tau_m {candidate['tau_m']:.2f}, {names[0]} tau_m {baseline['tau_m']:.2f}\n")
    for category, (image_id, why) in picks.items():
        if image_id is None:
            print(f"{category}: no such case in the {args.split} split\n")
            continue
        truth = truths[image_id]
        print(f"{category}: {image_id}  ({NAMES[truth.class_id].lower()}, patient {patient[image_id]}) - {why}")
        print("   ", summary(base[image_id], names[0]))
        print("   ", summary(cand[image_id], names[1]), "\n")
        left, right = base[image_id], cand[image_id]
        draw_panel(
            args.data / "images" / args.split / f"{image_id}.jpg",
            PanelSide(left.detections, left.truth, summary(left, names[0])),
            PanelSide(right.detections, right.truth, summary(right, names[1])),
            f"{category}: {image_id}, true class {NAMES[truth.class_id].lower()}", out / f"{category}-{image_id}.png",
        )  # fmt: skip
    print(f"panels written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
