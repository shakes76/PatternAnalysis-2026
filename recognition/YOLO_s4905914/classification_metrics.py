"""Post-hoc reporting for a single annotated lesion per image.

These metrics do not select checkpoints, thresholds or boxes using ground truth.
Detection matching is delegated to the unchanged, versioned evaluation.py.
Classification uses the highest-confidence retained box; no box is an abstention
and counts as incorrect in full-cohort accuracy and as an FN for the true class.
This is distinct from an any-malignant-box image alert and from detection recall.
"""
import argparse
from pathlib import Path

from evaluation import iou, safe_ratio, score_records
from runtime import NAMES, digest, read_json, write_json


def primary_prediction(predictions, threshold):
    """Choose without looking at the GT; use a deterministic tie rule."""
    retained = [p for p in predictions if p['confidence'] > threshold]
    return min(retained, key=lambda p: (-p['confidence'], p['class_id'], *p['box'])) if retained else None


def mean(values):
    return safe_ratio(sum(values), len(values))


def summarize_predictions(records, threshold, match_iou=0.5):
    detection = score_records(records, threshold, match_iou)
    # Rows are GT B/M. Columns are predicted B/M/no prediction; no GT abstention row.
    matrix = [[0, 0, 0], [0, 0, 0]]
    single_box = {c: {'primary_box_ignore_class': [], 'primary_box_correct_class': []} for c in NAMES}
    for record in records:
        c = record['class_id']
        pred = primary_prediction(record['predictions'], threshold)
        matrix[c][pred['class_id'] if pred else 2] += 1
        overlap = iou(record['box'], pred['box']) if pred else 0.0
        single_box[c]['primary_box_ignore_class'].append(overlap)
        single_box[c]['primary_box_correct_class'].append(overlap if pred and pred['class_id'] == c else 0.0)
    total = detection['images']
    correct = matrix[0][0] + matrix[1][1]
    covered = total - matrix[0][2] - matrix[1][2]
    classes = {}
    for c, name in NAMES.items():
        support = sum(matrix[c])
        tp = matrix[c][c]
        fp = sum(matrix[g][c] for g in NAMES if g != c)
        fn = support - tp
        classes[name] = {'support': support, 'tp': tp, 'fp': fp, 'fn': fn,
                         'abstentions': matrix[c][2],
                         'precision': safe_ratio(tp, tp + fp), 'recall': safe_ratio(tp, support),
                         'f1': safe_ratio(2 * tp, 2 * tp + fp + fn)}
    classification = {
        'rule': 'Highest-confidence box after the existing score threshold; no GT-based box choice; no IoU requirement',
        'tie_rule': 'Lower class ID, then lexicographically smaller box coordinates',
        'no_prediction_rule': 'Abstention; incorrect in full-cohort accuracy and FN for the true class',
        'confusion_matrix': matrix, 'row_labels': ['Benign', 'Malignant'],
        'column_labels': ['Benign', 'Malignant', 'No prediction'],
        'correct': correct, 'images': total, 'accuracy': safe_ratio(correct, total),
        'covered': covered, 'coverage': safe_ratio(covered, total),
        'conditional_accuracy_covered_only': safe_ratio(correct, covered),
        'classes': classes,
        'macro_f1': mean([v['f1'] for v in classes.values() if v['f1'] is not None]),
        'support_weighted_f1': safe_ratio(sum((v['f1'] or 0.0) * v['support'] for v in classes.values()), total),
        'balanced_accuracy': mean([v['recall'] for v in classes.values() if v['recall'] is not None]),
        'extra_boxes': 'Ignored only for the single image-class decision; still counted by detection metrics',
    }
    detection_f1 = {name: safe_ratio(2*v['tp'], 2*v['tp']+v['fp']+v['fn'])
                    for name, v in detection['classes'].items()}
    overlaps = {}
    for kind in ('all_gt_class_correct', 'all_gt_class_agnostic', 'tp_only',
                 'primary_box_ignore_class', 'primary_box_correct_class'):
        per_class = {}
        for c, name in NAMES.items():
            per_class[name] = ({'mean': mean(single_box[c][kind]), 'denominator': len(single_box[c][kind])}
                               if kind.startswith('primary_box') else detection['classes'][name]['iou'][kind])
        count = sum(v['denominator'] for v in per_class.values())
        total_iou = sum((v['mean'] or 0.0) * v['denominator'] for v in per_class.values())
        overlaps[kind] = {
            'overall': {'mean': safe_ratio(total_iou, count), 'denominator': count},
            'per_class': per_class,
            'macro_class_mean': mean([v['mean'] for v in per_class.values() if v['mean'] is not None]),
        }
    return {'reporting_version': 'course-closeout-v1', 'detection': detection,
            'detection_f1': detection_f1, 'classification': classification, 'localization': overlaps,
            'overall_iou_rule': 'Pool individual observations; do not substitute the unweighted mean of class means',
            'best_box_iou_caveat': 'All-GT best-box IoUs use GT overlap as an oracle diagnostic and do not penalize extra predictions',
            'tp_only_caveat': 'Excludes wrong-class, poorly localized and missed GTs; pooled denominator is TP count',
            'primary_box_iou_caveat': 'Uses the same GT-free highest-score choice as classification; no prediction is zero',
            'selection_changed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--match-iou', type=float, default=0.5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize_predictions(read_json(args.predictions), args.threshold, args.match_iou)
    result['predictions_sha256'] = digest(args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
