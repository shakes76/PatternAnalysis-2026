"""Distinguish classification, detection, abstentions and IoU denominators."""
import unittest
from classification_metrics import primary_prediction, summarize_predictions


def box(cls, score=.9, xyxy=None):
    return {'class_id': cls, 'confidence': score, 'box': xyxy or [0, 0, 1, 1]}


def case(name, cls, predictions):
    return {'image_id': name, 'class_id': cls, 'box': [0, 0, 1, 1], 'predictions': predictions}


class ClassificationMetricsTests(unittest.TestCase):
    def test_abstention_is_not_correct_benign_or_dropped_from_accuracy(self):
        r = summarize_predictions([case('a', 0, []), case('b', 1, [box(1)]), case('c', 1, [])], .4)
        c = r['classification']
        self.assertEqual(c['confusion_matrix'], [[0, 0, 1], [0, 1, 1]])
        self.assertEqual(c['accuracy'], 1/3)
        self.assertEqual(c['conditional_accuracy_covered_only'], 1)
        self.assertEqual(c['classes']['Malignant']['f1'], 2/3)
        self.assertEqual(c['macro_f1'], 1/3)

    def test_correct_class_wrong_location_counts_for_classification_only(self):
        r = summarize_predictions([case('a', 1, [box(1, xyxy=[2, 2, 3, 3])])], .4)
        self.assertEqual(r['classification']['accuracy'], 1)
        self.assertEqual(r['detection']['classes']['Malignant']['tp'], 0)
        self.assertEqual(r['detection_f1']['Malignant'], 0)
        self.assertEqual(r['localization']['tp_only']['overall'], {'mean': None, 'denominator': 0})

    def test_highest_score_is_not_gt_best_or_any_malignant_alert(self):
        preds = [box(1, .6), box(0, .9)]
        r = summarize_predictions([case('a', 1, preds)], .4)
        self.assertEqual(r['classification']['accuracy'], 0)
        self.assertEqual(r['detection']['classes']['Malignant']['tp'], 1)
        self.assertEqual(r['detection']['image_alerts']['malignant_tp'], 1)
        self.assertEqual(r['localization']['primary_box_correct_class']['overall']['mean'], 0)

    def test_duplicate_boxes_penalize_detection_not_image_accuracy(self):
        r = summarize_predictions([case('a', 0, [box(0, .9), box(0, .8), box(0, .7)])], .4)
        self.assertEqual(r['classification']['accuracy'], 1)
        self.assertEqual(r['detection_f1']['Benign'], .5)
        self.assertEqual(r['localization']['tp_only']['overall']['denominator'], 1)

    def test_overall_pools_images_not_class_means(self):
        r = summarize_predictions([case('a', 0, [box(0)]), case('b', 0, [box(0)]), case('c', 1, [])], .4)
        loc = r['localization']['all_gt_class_correct']
        self.assertEqual(loc['overall']['mean'], 2/3)
        self.assertEqual(loc['overall']['denominator'], 3)
        self.assertEqual(loc['macro_class_mean'], .5)

    def test_threshold_is_strict_and_ties_are_deterministic(self):
        self.assertIsNone(primary_prediction([box(1, .4)], .4))
        preds = [box(1, .5), box(0, .5)]
        self.assertEqual(primary_prediction(preds, .4), primary_prediction(list(reversed(preds)), .4))
        self.assertEqual(primary_prediction(preds, .4)['class_id'], 0)

    def test_existing_validation_rejects_duplicate_gt_and_invalid_predictions(self):
        with self.assertRaises(ValueError):
            summarize_predictions([case('a', 0, []), case('a', 1, [])], .4)
        with self.assertRaises(ValueError):
            summarize_predictions([case('a', 0, [box(1, float('nan'))])], .4)


if __name__ == '__main__':
    unittest.main()
