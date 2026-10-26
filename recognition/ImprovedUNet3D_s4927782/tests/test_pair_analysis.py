"""Synthetic checks for paired, patient-balanced boundary evidence."""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
    import scipy
except ImportError:
    np = None

analysis = None
if np is not None:
    path = Path(__file__).resolve().parents[1] / "tools" / "analyze_pair.py"
    spec = importlib.util.spec_from_file_location("pair_analysis", path)
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)


@unittest.skipIf(analysis is None, "NumPy and SciPy required for paired analysis")
class PairMathTests(unittest.TestCase):
    def test_exact_overlap_and_identical_surface_distance(self):
        target = np.zeros((7, 7, 7), dtype=bool)
        target[2:5, 2:5, 2:5] = True
        exact = analysis.binary_metrics(target, target, [3.0, 2.0, 1.0])
        self.assertEqual((exact["dice"], exact["iou"], exact["hd95_mm"]), (1.0, 1.0, 0.0))
        overlap = analysis.overlap_metrics(np.array([1, 1, 0], bool), np.array([0, 1, 1], bool))
        self.assertEqual(overlap["dice"], 0.5)
        self.assertAlmostEqual(overlap["iou"], 1 / 3)

    def test_surface_translation_uses_dhw_millimetre_spacing(self):
        target = np.zeros((5, 5, 5), bool)
        prediction = target.copy()
        target[1, 2, 2] = True
        prediction[2, 2, 2] = True
        result = analysis.binary_metrics(prediction, target, [3.0, 2.0, 1.0])
        self.assertEqual(result["hd95_mm"], 3.0)
        self.assertEqual(result["dice"], 0.0)

    def test_empty_masks_do_not_receive_zero_surface_error(self):
        empty = np.zeros((3, 3, 3), bool)
        positive = empty.copy(); positive[1, 1, 1] = True
        for prediction, target, status in ((empty, positive, "missed"), (positive, empty, "fp_only")):
            result = analysis.binary_metrics(prediction, target, [1, 1, 1])
            self.assertEqual(result["dice"], 0.0)
            self.assertIsNone(result["hd95_mm"])
            self.assertEqual(result["empty_status"], status)
        both = analysis.binary_metrics(empty, empty, [1, 1, 1])
        self.assertIsNone(both["dice"])
        self.assertIsNone(both["hd95_mm"])

    def test_extent_thirds_and_outer_false_positive_are_separate(self):
        target = np.zeros((10, 3, 3), bool)
        target[2:8, 1, 1] = True
        prediction = target.copy(); prediction[0, 0, 0] = True
        regions = analysis.region_metrics(prediction, target)
        self.assertEqual(regions["gt_extent_d"], [2, 7])
        self.assertEqual(regions["outside_gt_extent_fp_voxels"], 1)
        self.assertEqual(regions["regions"]["inferior"]["depth_indices"], [2, 3])
        self.assertEqual(regions["regions"]["mid"]["depth_indices"], [4, 5])
        self.assertEqual(regions["regions"]["superior"]["depth_indices"], [6, 7])
        self.assertTrue(all(region["dice"] == 1.0 for region in regions["regions"].values()))
        self.assertLess(analysis.overlap_metrics(prediction, target)["dice"], 1.0)

    def test_confusion_counts_include_other_gt_and_predicted_classes(self):
        result = analysis.confusion_counts(np.array([2, 5, 5]), np.array([5, 1, 5]), 5, 6)
        self.assertEqual(result["target_fn_by_predicted_class"]["2"], 1)
        self.assertEqual(result["target_fp_by_ground_truth_class"]["1"], 1)

    def test_bootstrap_weights_patients_equally_after_visit_averaging(self):
        cases = [{"patient": "many", "baseline": {"dice": 0.2}, "model": {"dice": 0.3}} for _ in range(10)]
        cases += [{"patient": "one", "baseline": {"dice": 0.8}, "model": {"dice": 0.4}}]
        result = analysis.paired_bootstrap(cases, samples=2000, seed=3710)
        self.assertAlmostEqual(result["delta_dice"], -0.15)
        self.assertEqual(result, analysis.paired_bootstrap(cases, samples=2000, seed=3710))
        self.assertTrue(np.allclose(result["ci95"], [-0.4, 0.1]))

    def test_bootstrap_preserves_model_pairing_and_zero_dice(self):
        cases = [{"patient": str(i), "baseline": {"dice": base}, "model": {"dice": base + 0.1}}
                 for i, base in enumerate((0.0, 0.3, 0.8))]
        result = analysis.paired_bootstrap(cases, samples=500)
        self.assertTrue(np.allclose(result["ci95"], [0.1, 0.1]))
        self.assertEqual(result["num_patients"], 3)
        self.assertEqual(result["excluded_undefined_case_pairs"], 0)

    def test_hd95_comparison_matches_support_and_weights_patients_not_visits(self):
        def case(patient, baseline, model, baseline_status="both_present", model_status="both_present"):
            return {"patient": patient,
                    "baseline": {"hd95_mm": baseline, "empty_status": baseline_status, "dice": 0.0},
                    "model": {"hd95_mm": model, "empty_status": model_status, "dice": 0.0}}

        cases = [case("many", 1.0, 3.0) for _ in range(10)]
        cases += [case("one", 9.0, 3.0),
                  case("missed", None, 2.0, baseline_status="missed"),
                  case("false_positive", None, None, "fp_only", "fp_only"),
                  case("absent", None, None, "both_empty", "both_empty")]
        result = analysis.paired_hd95_summary(cases)
        self.assertEqual(result["eligible_case_pairs"], 11)
        self.assertEqual(result["excluded_case_pairs"], 3)
        self.assertEqual(result["num_patients"], 2)
        self.assertEqual(result["baseline_hd95_mm"], 5.0)
        self.assertEqual(result["model_hd95_mm"], 3.0)
        self.assertEqual(result["delta_hd95_mm"], -2.0)
        self.assertEqual(result["excluded_status_counts_by_model"]["baseline"]["missed"], 1)
        self.assertEqual(result["excluded_status_counts_by_model"]["baseline"]["fp_only"], 1)
        self.assertEqual(result["excluded_status_counts_by_model"]["model"]["both_present"], 1)
        self.assertEqual(result["excluded_status_counts_by_model"]["model"]["both_empty"], 1)
        # All eleven eligible pairs intentionally have Dice zero; none may drop.
        self.assertEqual(sum(p["num_paired_visits"] for p in result["patients"]), 11)

    def test_hd95_comparison_without_common_present_masks_is_undefined(self):
        cases = [{"patient": "missing",
                  "baseline": {"hd95_mm": None, "empty_status": "missed"},
                  "model": {"hd95_mm": 2.0, "empty_status": "both_present"}}]
        result = analysis.paired_hd95_summary(cases)
        self.assertEqual(result["num_patients"], 0)
        self.assertEqual(result["eligible_case_pairs"], 0)
        self.assertEqual(result["excluded_case_pairs"], 1)
        for metric in ("baseline_hd95_mm", "model_hd95_mm", "delta_hd95_mm"):
            self.assertIsNone(result[metric])

    def test_failure_gallery_uses_distinct_patients_and_their_worst_visit(self):
        def case(patient, visit, dice):
            return {"patient": patient, "visit": visit, "model": {"dice": dice}}

        cases = [case("A", "Week2", 0.0), case("A", "Week0", 0.0),
                 case("A", "Week1", 0.9), case("B", "Week0", 0.1),
                 case("C", "Week0", 0.2), case("D", "Week0", 0.3),
                 case("E", "Week0", 0.4), case("F", "Week0", 0.5),
                 case("G", "Week0", None)]
        selected = analysis.select_failure_cases(cases)
        self.assertEqual([(item["patient"], item["visit"]) for item in selected],
                         [("A", "Week0"), ("B", "Week0"), ("C", "Week0"),
                          ("D", "Week0"), ("E", "Week0")])
        self.assertEqual(len({item["patient"] for item in selected}), 5)
        self.assertEqual(selected[0]["model"]["dice"], 0.0)
        self.assertEqual(selected, analysis.select_failure_cases(list(reversed(cases))))

    def test_gallery_keeps_gt_thirds_and_both_models_maximum_outer_error_slices(self):
        target = np.zeros((12, 4, 4), dtype=np.uint8)
        target[3:9, 1, 1] = 5
        baseline, model = target.copy(), target.copy()
        baseline[0] = 5
        model[11] = 5
        rows = analysis.select_gallery_slices(target, [baseline, model], 5)
        self.assertEqual([row["depth"] for row in rows], [4, 6, 8, 0, 11])
        self.assertIn("max 2D FP+FN (16 voxels)", rows[3]["labels"])
        self.assertIn("max 3D FP+FN (16 voxels)", rows[4]["labels"])

    def test_gallery_coalesces_repeated_depths_and_labels_absent_gt(self):
        target = np.zeros((12, 4, 4), dtype=np.uint8)
        target[3:9, 1, 1] = 5
        prediction = target.copy()
        prediction[0] = 5
        rows = analysis.select_gallery_slices(target, [prediction, prediction], 5)
        self.assertEqual([row["depth"] for row in rows], [4, 6, 8, 0])
        self.assertEqual(len(rows[-1]["labels"]), 2)
        empty = np.zeros((4, 4, 4), dtype=np.uint8)
        false_positive = empty.copy(); false_positive[3] = 5
        absent_rows = analysis.select_gallery_slices(empty, [empty, false_positive], 5)
        self.assertTrue(any("no GT third" in label for row in absent_rows for label in row["labels"]))
        self.assertTrue(any(row["depth"] == 3 and any("max 3D" in label for label in row["labels"])
                            for row in absent_rows))


@unittest.skipIf(analysis is None, "NumPy and SciPy required for paired analysis")
class EvidenceContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "manifest.json"
        self.baseline = self.root / "baseline.json"
        self.model = self.root / "model.json"
        self.content = {"num_classes": 6, "label_ids": list(range(6)), "target_shape_dhw": [4, 4, 4],
                        "partial": False, "development_only": False,
                        "patient_splits": {"train": ["B006"], "validation": ["C001"], "test": ["G001"]},
                        "records": [{"patient": patient, "visit": "Week0", "split": split, "cache_path": patient + ".npz"}
                                    for split, patient in (("train", "B006"), ("validation", "C001"), ("test", "G001"))]}
        self.manifest.write_text(json.dumps(self.content))
        for path, name in ((self.baseline, "unet2d"), (self.model, "improved3d")):
            path.write_text(json.dumps({
                "split": "test", "development_only": False, "final_test_explicitly_requested": True,
                "manifest_sha256": hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
                "model_config": {"model": name, "label_ids": list(range(6)), "num_classes": 6, "target_shape_dhw": [4, 4, 4]},
                "cases": [{"patient": "G001", "visit": "Week0", "prediction_file": "predictions/G001.npz"}],
            }))

    def test_contract_accepts_matching_complete_final_test(self):
        _, _, records, evaluations = analysis.load_evidence(self.manifest, self.baseline, self.model)
        self.assertEqual(set(records), {("G001", "Week0")})
        self.assertEqual(evaluations[0][1]["split"], "test")

    def test_contract_rejects_hash_split_missing_case_and_development(self):
        original = json.loads(self.model.read_text())
        for field, value in (("manifest_sha256", "bad"), ("split", "validation"),
                             ("cases", []), ("development_only", True)):
            changed = dict(original); changed[field] = value
            self.model.write_text(json.dumps(changed))
            with self.subTest(field=field), self.assertRaises(ValueError):
                analysis.load_evidence(self.manifest, self.baseline, self.model)


if __name__ == "__main__":
    unittest.main()
