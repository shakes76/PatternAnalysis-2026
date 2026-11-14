"""
Routing policy: detections in, clear / monitor / urgent review out.

Implements the primary rule from the operating-policy plan. The rule itself (`route_case`)
never sees ground truth, so it works on an unlabelled image. The evaluation, threshold
search and selection functions score it against ground truth on a labelled split.

All numbers are experimental workload proxies, not calibrated risk or clinical advice.
"""

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from yolo_lesion_47041585.dataset import CLASS_MAPPING
from yolo_lesion_47041585.helpers.evaluation import box_iou
from yolo_lesion_47041585.predict import Detection

BENIGN: int = CLASS_MAPPING["Benign"]
MALIGNANT: int = CLASS_MAPPING["Malignant"]

ROUTING_RULE_VERSION = "strict"  # the primary rule
ROUTING_RULES = ("strict", "lenient")  # lenient is a declared alternative, always reported beside strict
MATCH_IOU = 0.5  # a detection localises its target at or above this IoU
TOLERANCE = (
    1e-9  # so a rate that equals a target in exact arithmetic is not missed by rounding
)

Route = Literal["clear", "monitor", "urgent"]
Reason = Literal[
    "malignant_alert",
    "confident_benign",
    "no_detection",
    "insufficient_benign_confidence",
    "multiple_benign",
    "conflicting_classes",
    "weak_malignant_only",
    "malignant_present",
]
Status = Literal["feasible", "infeasible"]
Basis = Literal[
    "feasible", "workload_constrained", "sensitivity_constrained", "maximum_sensitivity"
]
MONITOR_REASONS: tuple[Reason, ...] = (
    "no_detection",
    "insufficient_benign_confidence",
    "multiple_benign",
    "conflicting_classes",
    "weak_malignant_only",
    "malignant_present",
)

# the fixed search grid: the confidence floor, 0.01 to 0.99 in steps of 0.01, and 1.0
THRESHOLD_GRID: tuple[float, ...] = (
    0.001,
    *(round(i / 100, 2) for i in range(1, 100)),
    1.0,
)


@dataclass(frozen=True)
class PolicySpec:
    """Thresholds and rule version for one operating point.

    Comparisons are inclusive: a confidence equal to a threshold meets it.

    Attributes:
        malignant_threshold (float): `tau_m`, the confidence at or above which a malignant
            detection raises an urgent review.
        benign_threshold (float): `tau_b`, the confidence at or above which a single
            benign detection clears the image.
        version (str): Routing-rule version.
        clear_guard (float | None): Rule "lenient" only. If set, an image is cleared only while
            its best malignant confidence is below this value, otherwise it is monitored.
            None leaves the rule unguarded.
    """

    malignant_threshold: float
    benign_threshold: float
    version: str = ROUTING_RULE_VERSION
    clear_guard: float | None = None


@dataclass(frozen=True)
class RoutingInputs:
    """What the routing rule needs to know about one image's detections.

    Attributes:
        n_detections (int): Detections kept after the confidence floor and NMS.
        n_benign (int): How many are benign.
        n_malignant (int): How many are malignant.
        max_malignant_conf (float): The highest malignant confidence, 0.0 if there is none.
        max_benign_conf (float): The highest benign confidence, 0.0 if there is none.
        sole_benign_conf (float | None): The confidence if the image has exactly one
            detection and it is benign, otherwise None.
    """

    n_detections: int
    n_benign: int
    n_malignant: int
    max_malignant_conf: float
    max_benign_conf: float
    sole_benign_conf: float | None


def routing_inputs(detections: Sequence[Detection]) -> RoutingInputs:
    """Summarise one image's detections for the routing rule."""
    benign = [d.confidence for d in detections if d.class_id == BENIGN]
    malignant = [d.confidence for d in detections if d.class_id == MALIGNANT]
    return RoutingInputs(
        n_detections=len(detections),
        n_benign=len(benign),
        n_malignant=len(malignant),
        max_malignant_conf=max(malignant, default=0.0),
        max_benign_conf=max(benign, default=0.0),
        sole_benign_conf=benign[0] if len(detections) == 1 and benign else None,
    )


def route_from_inputs(
    inputs: RoutingInputs,
    malignant_threshold: float,
    benign_threshold: float,
    version: str = ROUTING_RULE_VERSION,
    clear_guard: float | None = None,
) -> tuple[Route, Reason]:
    """A routing rule, applied to summarised detections.

    Version "strict", the primary rule:
    1. A malignant detection at or above the malignant threshold gives urgent review.
    2. Otherwise exactly one detection, benign and at or above the benign threshold,
       gives clear.
    3. Otherwise monitor, with the reason: no detection, a benign detection that is too
       weak, several benign detections, both classes present, or only weak malignant ones.

    Version "lenient", the declared alternative, which has no fixed floor and does not treat
    extra boxes as conflict, so each threshold alone decides what counts as confident:
    1. As in strict, a malignant detection at or above the malignant threshold gives urgent.
    2. Otherwise the best benign detection at or above the benign threshold gives clear.
       Weaker malignant boxes and other benign boxes are ignored.
    3. Otherwise monitor, with the reason: no detection, only weak malignant detections
       (no benign box), or a best benign detection that is too weak.
    With a `clear_guard`, step 2 also needs the best malignant confidence to be below the
    guard; a confident benign image that fails this is monitored as "malignant_present".

    Raises:
        ValueError: If the version is unknown.
    """
    if version not in ROUTING_RULES:
        raise ValueError(
            f"Unknown routing rule {version!r}, expected one of {ROUTING_RULES}"
        )

    if inputs.n_malignant and inputs.max_malignant_conf >= malignant_threshold:
        return "urgent", "malignant_alert"
    if version == "lenient":
        if inputs.n_benign and inputs.max_benign_conf >= benign_threshold:
            if clear_guard is not None and inputs.max_malignant_conf >= clear_guard:
                return "monitor", "malignant_present"
            return "clear", "confident_benign"
        if inputs.n_detections == 0:
            return "monitor", "no_detection"
        if not inputs.n_benign:
            return "monitor", "weak_malignant_only"
        return "monitor", "insufficient_benign_confidence"
    if (
        inputs.sole_benign_conf is not None
        and inputs.sole_benign_conf >= benign_threshold
    ):
        return "clear", "confident_benign"

    if inputs.n_detections == 0:
        return "monitor", "no_detection"
    if inputs.n_malignant and inputs.n_benign:
        return "monitor", "conflicting_classes"
    if inputs.n_malignant:
        return "monitor", "weak_malignant_only"
    if inputs.n_benign > 1:
        return "monitor", "multiple_benign"
    return "monitor", "insufficient_benign_confidence"


def route_case(
    detections: Sequence[Detection], spec: PolicySpec
) -> tuple[Route, Reason]:
    """Route one image from its detections alone.

    Args:
        detections (Sequence[Detection]): Every detection kept for the image, after the
            confidence floor and NMS. Absent, conflicting or weak detections give monitor.
        spec (PolicySpec): The thresholds.

    Returns:
        tuple[Route, Reason]: The route and why.
    """
    return route_from_inputs(
        routing_inputs(detections),
        spec.malignant_threshold,
        spec.benign_threshold,
        spec.version,
        spec.clear_guard,
    )


@dataclass(frozen=True)
class ImageEvidence:
    """One labelled image, reduced to what scoring a policy needs.

    Attributes:
        image_id (str): Image identifier.
        true_class (int): The annotated target's class.
        inputs (RoutingInputs): The image's detections as the routing rule sees them.
        candidates (tuple[tuple[float, float], ...]): (confidence, IoU) of every
            detection of the true class that overlaps the target at IoU 0.5 or more,
            highest confidence first. The first one at or above a threshold is the
            target's match, as in `match_detections`.
    """

    image_id: str
    true_class: int
    inputs: RoutingInputs
    candidates: tuple[tuple[float, float], ...]


def build_evidence(
    predictions: Sequence[Detection], ground_truth: Sequence[Detection]
) -> list[ImageEvidence]:
    """Reduce predictions and ground truth to per-image evidence, once.

    Every image in `ground_truth` is included, even one with no detections, so no
    denominator quietly drops images.

    Args:
        predictions (Sequence[Detection]): Detections for every image, at the fixed floor.
        ground_truth (Sequence[Detection]): The annotated target of each image.

    Returns:
        list[ImageEvidence]: One entry per ground-truth image, ordered by image ID.

    Raises:
        ValueError: If an image has more than one target or there are predictions for an
            image with none.
    """
    targets: dict[str, Detection] = {}
    for truth in ground_truth:
        if truth.image_id in targets:
            raise ValueError(f"{truth.image_id} has more than one target")
        targets[truth.image_id] = truth

    by_image: defaultdict[str, list[Detection]] = defaultdict(list)
    for prediction in predictions:
        by_image[prediction.image_id].append(prediction)
    if unknown := sorted(set(by_image) - set(targets)):
        raise ValueError(f"Predictions for images with no ground truth: {unknown[:3]}")

    evidence: list[ImageEvidence] = []
    for image_id, truth in sorted(targets.items()):
        detections = by_image.get(image_id, [])
        overlaps = (
            (d.confidence, box_iou(d.box, truth.box))
            for d in detections
            if d.class_id == truth.class_id
        )
        candidates = sorted((c, i) for c, i in overlaps if i >= MATCH_IOU)[::-1]
        evidence.append(
            ImageEvidence(
                image_id, truth.class_id, routing_inputs(detections), tuple(candidates)
            )
        )
    return evidence


@dataclass(frozen=True)
class ImageRouting:
    """One image's outcome under a policy, kept so detection and routing stay separate.

    Attributes:
        image_id (str): Image identifier.
        true_class (int): The annotated target's class.
        route (Route): The action assigned.
        reason (Reason): Why.
        detected (bool): Whether the target has a class-correct match at IoU 0.5 or more
            at or above its class's threshold (malignant threshold for a malignant
            target, benign threshold for a benign one). Independent of the route.
        matched_iou (float | None): The IoU of that match, if any.
        n_detections (int): Detections kept for the image.
    """

    image_id: str
    true_class: int
    route: Route
    reason: Reason
    detected: bool
    matched_iou: float | None
    n_detections: int


@dataclass(frozen=True)
class OperatingPoint:
    """The result of one pair of thresholds on a labelled split.

    Rates use all images of the class as the denominator, including images with no
    detections.

    Attributes:
        malignant_threshold (float): `tau_m`.
        benign_threshold (float): `tau_b`.
        n_malignant (int): Malignant images in the split.
        n_benign (int): Benign images in the split.
        malignant_detected (int): Malignant targets with a class-correct match at IoU 0.5
            or more and confidence at or above `tau_m`.
        m (float): `malignant_detected / n_malignant`, the malignant sensitivity.
        b (float): Benign images routed monitor or urgent over all benign images.
        benign_urgent (float): Benign images routed urgent, as a share of benign images.
        benign_monitor (float): Benign images routed monitor, as a share of benign images.
        benign_clear (float): Benign images routed clear, as a share of benign images.
        malignant_urgent (float): Malignant images routed urgent, as a share of them.
        malignant_monitor (float): Malignant images routed monitor.
        malignant_clear (float): Malignant images routed clear.
        clear_coverage (float): All images routed clear over all images.
        benign_monitor_reasons (dict[str, int]): Why benign images were monitored.
        matched_iou_benign (float | None): Mean IoU of benign matches, None if none.
        n_matched_benign (int): Benign targets matched.
        matched_iou_malignant (float | None): Mean IoU of malignant matches, None if none.
        n_matched_malignant (int): Malignant targets matched.
    """

    malignant_threshold: float
    benign_threshold: float
    n_malignant: int
    n_benign: int
    malignant_detected: int
    m: float
    b: float
    benign_urgent: float
    benign_monitor: float
    benign_clear: float
    malignant_urgent: float
    malignant_monitor: float
    malignant_clear: float
    clear_coverage: float
    benign_monitor_reasons: dict[str, int]
    matched_iou_benign: float | None
    n_matched_benign: int
    matched_iou_malignant: float | None
    n_matched_malignant: int


def _score(
    evidence: Sequence[ImageEvidence],
    malignant_threshold: float,
    benign_threshold: float,
    images: list[ImageRouting] | None = None,
    version: str = ROUTING_RULE_VERSION,
    clear_guard: float | None = None,
) -> OperatingPoint:
    """Score one threshold pair, optionally recording each image's outcome."""
    routes = {BENIGN: Counter[Route](), MALIGNANT: Counter[Route]()}
    reasons = Counter[Reason]()
    iou_sum = {BENIGN: 0.0, MALIGNANT: 0.0}
    matched = {BENIGN: 0, MALIGNANT: 0}
    totals = {BENIGN: 0, MALIGNANT: 0}

    for item in evidence:
        route, reason = route_from_inputs(
            item.inputs, malignant_threshold, benign_threshold, version, clear_guard
        )
        threshold = (
            malignant_threshold if item.true_class == MALIGNANT else benign_threshold
        )
        iou = next((i for c, i in item.candidates if c >= threshold), None)

        totals[item.true_class] += 1
        routes[item.true_class][route] += 1
        if item.true_class == BENIGN and route == "monitor":
            reasons[reason] += 1

        if iou is not None:
            matched[item.true_class] += 1
            iou_sum[item.true_class] += iou

        if images is not None:
            images.append(
                ImageRouting(
                    item.image_id, item.true_class, route, reason, iou is not None, iou,
                    item.inputs.n_detections,
                )
            )  # fmt: skip

    n_benign, n_malignant = totals[BENIGN], totals[MALIGNANT]

    if not n_benign or not n_malignant:
        raise ValueError(
            "Scoring a policy needs at least one benign and one malignant image."
        )

    benign, malignant = routes[BENIGN], routes[MALIGNANT]

    return OperatingPoint(
        malignant_threshold=malignant_threshold,
        benign_threshold=benign_threshold,
        n_malignant=n_malignant,
        n_benign=n_benign,
        malignant_detected=matched[MALIGNANT],
        m=matched[MALIGNANT] / n_malignant,
        b=(benign["monitor"] + benign["urgent"]) / n_benign,
        benign_urgent=benign["urgent"] / n_benign,
        benign_monitor=benign["monitor"] / n_benign,
        benign_clear=benign["clear"] / n_benign,
        malignant_urgent=malignant["urgent"] / n_malignant,
        malignant_monitor=malignant["monitor"] / n_malignant,
        malignant_clear=malignant["clear"] / n_malignant,
        clear_coverage=(benign["clear"] + malignant["clear"])
        / (n_benign + n_malignant),
        benign_monitor_reasons={reason: reasons[reason] for reason in MONITOR_REASONS},
        matched_iou_benign=iou_sum[BENIGN] / matched[BENIGN]
        if matched[BENIGN]
        else None,
        n_matched_benign=matched[BENIGN],
        matched_iou_malignant=(
            iou_sum[MALIGNANT] / matched[MALIGNANT] if matched[MALIGNANT] else None
        ),
        n_matched_malignant=matched[MALIGNANT],
    )


@dataclass(frozen=True)
class PolicyEvaluation:
    """One policy scored on a labelled split.

    Attributes:
        point (OperatingPoint): The aggregate result.
        images (list[ImageRouting]): Each image's outcome, ordered by image ID.
    """

    point: OperatingPoint
    images: list[ImageRouting]


def evaluate_policy(
    predictions: Sequence[Detection],
    ground_truth: Sequence[Detection],
    spec: PolicySpec,
) -> PolicyEvaluation:
    """Score one policy and keep every image's route, reason and match.

    Args:
        predictions (Sequence[Detection]): Detections for every image, at the fixed floor.
        ground_truth (Sequence[Detection]): The annotated target of each image.
        spec (PolicySpec): The thresholds.

    Returns:
        PolicyEvaluation: The aggregate rates and the per-image outcomes.
    """
    images: list[ImageRouting] = []
    point = _score(
        build_evidence(predictions, ground_truth),
        spec.malignant_threshold,
        spec.benign_threshold,
        images,
        spec.version,
        spec.clear_guard,
    )

    return PolicyEvaluation(point, images)


def search_policy(
    predictions: Sequence[Detection],
    ground_truth: Sequence[Detection],
    threshold_grid: Iterable[float] = THRESHOLD_GRID,
    version: str = ROUTING_RULE_VERSION,
) -> list[OperatingPoint]:
    """Score every pair of thresholds on a fixed grid under one routing rule.

    Detections are reduced to per-image evidence once, so the search does not rerun
    inference or matching for each pair.

    Args:
        predictions (Sequence[Detection]): Detections for every image, at the fixed floor.
        ground_truth (Sequence[Detection]): The annotated target of each image.
        threshold_grid (Iterable[float]): Values tried for both thresholds.
        version (str): The routing rule to score, "strict" or "lenient".

    Returns:
        list[OperatingPoint]: One row per (malignant, benign) pair, ordered by malignant
            threshold then benign threshold.
    """
    evidence = build_evidence(predictions, ground_truth)
    grid = tuple(threshold_grid)

    return [
        _score(evidence, tau_m, tau_b, version=version)
        for tau_m in grid
        for tau_b in grid
    ]


@dataclass(frozen=True)
class Targets:
    """The acceptance targets a selected operating point is judged against.

    Attributes:
        min_sensitivity (float): Required malignant sensitivity `m`.
        max_benign_escalation (float): Allowed benign escalation `b`.
        stretch_sensitivity (float): The stretch goal for `m`.
    """

    min_sensitivity: float = 0.90
    max_benign_escalation: float = 0.30
    stretch_sensitivity: float = 0.95


@dataclass(frozen=True)
class OperatingSelection:
    """The operating point chosen for one model, and the alternatives behind it.

    Attributes:
        status (Status): Whether any point meets both targets.
        basis (Basis): How `primary` was chosen: "feasible", "workload_constrained",
            "sensitivity_constrained" or "maximum_sensitivity".
        primary (OperatingPoint): The chosen point. If status is "infeasible" it fails
            acceptance.
        workload_alternative (OperatingPoint | None): The highest `m` among points with
            `b` within the limit, or None if no point is within it.
        sensitivity_alternative (OperatingPoint | None): The lowest `b` among points with
            `m` at the required level, or None if no point reaches it.
        meets_stretch (bool): Whether a feasible point also reaches the stretch sensitivity.
    """

    status: Status
    basis: Basis
    primary: OperatingPoint
    workload_alternative: OperatingPoint | None
    sensitivity_alternative: OperatingPoint | None
    meets_stretch: bool


def _by_sensitivity(point: OperatingPoint) -> tuple[float, float, float, float, float]:
    """Higher `m`, then lower `b`, lower benign urgent rate, then the lower thresholds."""
    return (
        -point.m,
        point.b,
        point.benign_urgent,
        point.malignant_threshold,
        point.benign_threshold,
    )


def _by_workload(point: OperatingPoint) -> tuple[float, float, float, float, float]:
    """Lower `b`, then higher `m`, lower benign urgent rate, then the lower thresholds."""
    return (
        point.b,
        -point.m,
        point.benign_urgent,
        point.malignant_threshold,
        point.benign_threshold,
    )


def select_operating_point(
    table: Sequence[OperatingPoint], targets: Targets = Targets()
) -> OperatingSelection:
    """Choose a model's operating point from its full threshold table.

    If any point meets both targets, the primary is the feasible one with the highest `m`,
    then the lowest `b`, then the lowest benign urgent rate, then the lowest thresholds. If
    none does, both constrained alternatives are still found and the primary follows the
    candidate-selection order: the workload-constrained alternative if it exists, else the
    sensitivity-constrained one, else the point with the highest `m` then lowest `b`.
    The targets are never relaxed: an infeasible primary is labelled as failing them.

    Args:
        table (Sequence[OperatingPoint]): The result of `search_policy`.
        targets (Targets): The acceptance targets.

    Returns:
        OperatingSelection: The primary point and both alternatives.

    Raises:
        ValueError: If the table is empty.
    """
    if not table:
        raise ValueError("Cannot select from an empty table.")

    def sensitive(p: OperatingPoint) -> bool:
        return p.m >= targets.min_sensitivity - TOLERANCE

    def within_workload(p: OperatingPoint) -> bool:
        return p.b <= targets.max_benign_escalation + TOLERANCE

    feasible = [p for p in table if sensitive(p) and within_workload(p)]
    workload = [p for p in table if within_workload(p)]
    sensitivity = [p for p in table if sensitive(p)]

    workload_alternative = min(workload, key=_by_sensitivity) if workload else None
    sensitivity_alternative = (
        min(sensitivity, key=_by_workload) if sensitivity else None
    )

    status: Status = "infeasible"
    basis: Basis
    if feasible:
        status, basis = "feasible", "feasible"
        primary = min(feasible, key=_by_sensitivity)

    elif workload_alternative:
        basis, primary = "workload_constrained", workload_alternative

    elif sensitivity_alternative:
        basis, primary = "sensitivity_constrained", sensitivity_alternative

    else:
        basis, primary = "maximum_sensitivity", min(table, key=_by_sensitivity)

    return OperatingSelection(
        status=status,
        basis=basis,
        primary=primary,
        workload_alternative=workload_alternative,
        sensitivity_alternative=sensitivity_alternative,
        meets_stretch=status == "feasible"
        and primary.m >= targets.stretch_sensitivity - TOLERANCE,
    )


@dataclass(frozen=True)
class RunSummary:
    """One condition's validation result, for choosing between conditions.

    Attributes:
        name (str): The condition, such as "O35".
        selection (OperatingSelection): Its selected operating point and alternatives.
        map50_95 (float): Its best validation mAP50:95, used to break ties.
        complexity (int): Lower is simpler or cheaper, used to break remaining ties.
    """

    name: str
    selection: OperatingSelection
    map50_95: float
    complexity: int


@dataclass(frozen=True)
class CandidateSelection:
    """Which condition was chosen and why.

    Attributes:
        winner (str): The chosen condition.
        basis (Basis): The comparison used: "feasible", "workload_constrained",
            "sensitivity_constrained" or "maximum_sensitivity".
        ranking (tuple[str, ...]): The conditions that took part in that comparison, best
            first. A condition that could not take part is not listed.
    """

    winner: str
    basis: Basis
    ranking: tuple[str, ...]


def select_candidate(runs: Sequence[RunSummary]) -> CandidateSelection:
    """Choose between conditions using each one's own validation-selected policy.

    1. Conditions with a point meeting both targets beat those without; rank them by `m`,
       then lower `b`, then validation mAP50:95, then lower complexity.
    2. If none is feasible, rank the workload-constrained alternatives the same way.
    3. If none exists, rank the sensitivity-constrained alternatives by lower `b`, then
       higher `m`, then mAP50:95, then complexity.
    4. Otherwise rank every condition's highest-sensitivity point by `m`, then `b`.

    Args:
        runs (Sequence[RunSummary]): One entry per condition, from the primary seed.

    Returns:
        CandidateSelection: The winner and the ranking behind it. Both constrained
            alternatives stay on each `RunSummary`, so a different workload preference can
            still be read off.

    Raises:
        ValueError: If there are no runs.
    """
    if not runs:
        raise ValueError("Cannot select from no runs.")

    def rank(
        group: list[tuple[RunSummary, OperatingPoint]], by_workload: bool = False
    ) -> tuple[str, ...]:
        key = _by_workload if by_workload else _by_sensitivity
        ordered = sorted(
            group,
            key=lambda item: (key(item[1])[:2], -item[0].map50_95, item[0].complexity),
        )
        return tuple(run.name for run, _ in ordered)

    feasible = [
        (r, r.selection.primary) for r in runs if r.selection.status == "feasible"
    ]
    workload = [
        (r, r.selection.workload_alternative)
        for r in runs
        if r.selection.workload_alternative is not None
    ]
    sensitivity = [
        (r, r.selection.sensitivity_alternative)
        for r in runs
        if r.selection.sensitivity_alternative is not None
    ]

    if feasible:
        ranking = rank(feasible)
        return CandidateSelection(ranking[0], "feasible", ranking)

    if workload:
        ranking = rank(workload)
        return CandidateSelection(ranking[0], "workload_constrained", ranking)

    if sensitivity:
        ranking = rank(sensitivity, by_workload=True)
        return CandidateSelection(ranking[0], "sensitivity_constrained", ranking)

    ranking = rank([(r, r.selection.primary) for r in runs])

    return CandidateSelection(ranking[0], "maximum_sensitivity", ranking)
