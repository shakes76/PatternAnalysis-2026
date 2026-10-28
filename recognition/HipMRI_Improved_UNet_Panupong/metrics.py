"""Segmentation metrics with the aggregation rules fixed before training.

Primary metric: 3D prostate Dice per series. The 60 predicted slices of one
case/week are treated as one volume, Dice = 2|P n G| / (|P| + |G|) is
computed for the prostate label, and the mean over all series is reported.
Every validation and test series contains prostate, so no series has an
empty ground truth. If one did, a series with neither prediction nor ground
truth is left out of the mean (and counted separately), while a prediction
without ground truth scores 0.

Also reported:
* case-weighted mean (average the series of each case first),
* per-class Dice and IoU for all six labels with the same per-series rule,
* global Dice over all pixels of the split,
* slice-level prostate Dice on slices whose ground truth contains prostate,
* the false-positive rate on slices whose ground truth has no prostate.

All values are in pixel space; the Nifti spacing has no physical units.
Counts are accumulated batch by batch, so whole volumes are never stored.
"""

from collections import defaultdict

import torch

NUM_CLASSES = 6
PROSTATE_LABEL = 5
CLASS_NAMES = ("Background", "Body", "Bones", "Bladder", "Rectum", "Prostate")


def _safe_ratio(numerator, denominator):
    """Return numerator / denominator, or None when the denominator is 0."""
    return numerator / denominator if denominator > 0 else None


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


class SegmentationMetrics:
    """Accumulate per-series and per-slice overlap counts for one split."""

    def __init__(self, num_classes=NUM_CLASSES):
        self.num_classes = num_classes
        # series key -> tensors of per-class intersection, prediction and target counts
        self.series = defaultdict(lambda: torch.zeros(3, num_classes, dtype=torch.long))
        # per-slice prostate counts: (series key, slice, intersection, predicted, target)
        self.prostate_slices = []
        self.seen = set()  # (case, week, slice) already added, to catch double counting

    @torch.no_grad()
    def update(self, pred_labels, target_labels, cases, weeks, slices):
        """Add one batch of predicted and true label maps of shape (B, H, W)."""
        for b in range(pred_labels.shape[0]):
            slice_key = (str(cases[b]), int(weeks[b]), int(slices[b]))
            if slice_key in self.seen:
                raise ValueError(f"Slice {slice_key} was added twice")
            self.seen.add(slice_key)
        pred = pred_labels.long()
        target = target_labels.long()
        for k in range(self.num_classes):
            p = pred == k
            t = target == k
            counts = torch.stack([(p & t).sum((1, 2)), p.sum((1, 2)), t.sum((1, 2))], dim=1).cpu()
            for b in range(pred.shape[0]):
                key = (str(cases[b]), int(weeks[b]))
                self.series[key][:, k] += counts[b]
                if k == PROSTATE_LABEL:
                    self.prostate_slices.append((key, int(slices[b]), *counts[b].tolist()))

    def summary(self):
        """Return a dictionary of all metrics (None where a value is undefined)."""
        per_series = {}
        for key, (inter, pred, target) in sorted(self.series.items()):
            dice, iou = [], []
            for k in range(self.num_classes):
                i, p, t = int(inter[k]), int(pred[k]), int(target[k])
                # Undefined (None) only when both prediction and target are
                # empty; a prediction with an empty target gives 0.
                dice.append(_safe_ratio(2 * i, p + t))
                iou.append(_safe_ratio(i, p + t - i))
            per_series[f"{key[0]}_week{key[1]}"] = {
                "case": key[0], "week": key[1], "dice": dice, "iou": iou,
                "slices": sum(1 for c, w, _ in self.seen if (c, w) == key)}

        def class_mean(metric, k):
            return _mean([s[metric][k] for s in per_series.values()])

        def case_weighted_mean(metric, k):
            by_case = defaultdict(list)
            for s in per_series.values():
                by_case[s["case"]].append(s[metric][k])
            return _mean([_mean(values) for values in by_case.values()])

        totals = sum(self.series.values(), torch.zeros(3, self.num_classes, dtype=torch.long))
        global_dice = [_safe_ratio(2 * int(totals[0, k]), int(totals[1, k] + totals[2, k]))
                       for k in range(self.num_classes)]

        positive = [s for s in self.prostate_slices if s[4] > 0]
        empty = [s for s in self.prostate_slices if s[4] == 0]
        slice_dice = [2 * i / (p + t) for _, _, i, p, t in positive]
        false_positive_slices = sum(1 for _, _, _, p, _ in empty if p > 0)

        empty_series = [name for name, s in per_series.items() if s["dice"][PROSTATE_LABEL] is None]
        return {
            "prostate_dice_series_mean": class_mean("dice", PROSTATE_LABEL),  # primary metric
            "prostate_dice_case_weighted": case_weighted_mean("dice", PROSTATE_LABEL),
            "prostate_dice_global": global_dice[PROSTATE_LABEL],
            "prostate_dice_slice_mean_positive": _mean(slice_dice),
            "per_class": {
                name: {
                    "dice_series_mean": class_mean("dice", k),
                    "iou_series_mean": class_mean("iou", k),
                    "dice_case_weighted": case_weighted_mean("dice", k),
                    "dice_global": global_dice[k],
                }
                for k, name in enumerate(CLASS_NAMES)
            },
            "foreground_dice_series_mean": _mean([class_mean("dice", k)
                                                  for k in range(1, self.num_classes)]),
            "num_series": len(per_series),
            "num_cases": len({s["case"] for s in per_series.values()}),
            "series_without_prostate": empty_series,
            "prostate_positive_slices": len(positive),
            "prostate_empty_slices": len(empty),
            "false_positive_empty_slices": false_positive_slices,
            "false_positive_rate_empty_slices": _safe_ratio(false_positive_slices, len(empty)),
            "per_series": per_series,
            # Kept so edge (base/apex) slices can be analysed without re-running the test.
            "prostate_slices": [
                {"series": f"{c}_week{w}", "slice": s, "pred_pixels": p, "gt_pixels": t,
                 "dice": 2 * i / (p + t) if t > 0 else None}
                for (c, w), s, i, p, t in sorted(self.prostate_slices)
            ],
        }
