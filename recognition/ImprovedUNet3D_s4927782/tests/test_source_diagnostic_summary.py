"""Synthetic report fixtures exercise scope, identity and publication boundaries."""

import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("source_summary", ROOT / "tools/summarize_source_diagnostics.py")
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)
SHA = "a" * 64


def score(predicted_class=5, p5=.8):
    probabilities = [0., 1 - p5, 0., 0., 0., p5]
    return {"gt_class5_voxels": 10, "nonfinite_gt_logit_rows": 0,
            "mean_softmax_probability_by_class": probabilities,
            "class5_rank_counts": {str(i): 10 if i == (1 if predicted_class == 5 else 2) else 0 for i in range(1, 7)},
            "class5_margin_over_max_competitor_quantiles": {q: 0. for q in tool.MARGIN_KEYS},
            "gt_class5_predicted_as_class_counts": [10 if i == predicted_class else 0 for i in range(6)]}


def cache_row(key, split):
    patient, visit = key.split("/")
    return {"patient": patient, "visit": visit, "split": split, "decoded_identity_verified": True,
            "contrasts": dict(zip(tool.CACHE_CONTRASTS, (1., .5, .5, .25, .5)))}


def fixtures():
    rows = [cache_row(k, v) for k, v in tool.CASES.items()]
    # Twenty-two other TRAIN patients and four other VAL patients retain repeated visits.
    for i in range(139):
        rows.append(cache_row(f"A{i % 22:03d}/Week{i // 22}", "train"))
    for i in range(36):
        rows.append(cache_row(f"V{i % 4:03d}/Week{i // 4}", "validation"))
    header_rows = [{"patient": r["patient"], "visit": r["visit"], "split": r["split"]} for r in rows]
    header_rows += [{"patient": "Z000", "visit": f"Week{i}", "split": "test"} for i in range(22)]
    for r in header_rows:
        for role in ("image", "mask"):
            r[role] = {"geometry_matches_frozen_record": True, "text_fields": {"descrip": {"free_text_exported": False}}}
    source_cases, mechanism_cases, intervention_cases = [], [], []
    for key, split in tool.CASES.items():
        source = {"case": key, "split": split, "source_file_sha256": {"image": SHA},
                  "decoded_cache_hashes_verified": {n: {"sha256": SHA} for n in ("image", "mask", "spacing")},
                  "policy_reconstruction": {"mask_equal_frozen": True, "after_support_image_max_abs_error": 0.},
                  "within_stage_class_contrasts": {s: dict(zip(tool.SOURCE_CONTRASTS, (2., 1., .5, .5, .25))) for s in tool.STAGES},
                  "intensity_stages": {s: {"5": {"n": 10, "zero_fraction": 0., "at_or_beyond_clip_fraction": None if s == tool.STAGES[0] else 0.}} for s in tool.STAGES},
                  "support_policy": {"native_target_zero_support_voxels": 0, "target_voxels": 10,
                                     "target_outside_resampled_nonzero_support": 0, "forced_zero_nonzero_voxels_all": 1, "forced_zero_nonzero_voxels_target": 0},
                  "target_volume": {"native_mm3_if_unit_confirmed": 1., "cache_mm3_if_unit_confirmed": 1., "cache_to_native_ratio": 1.}}
        source_cases.append(source)
        backward = {"status": "not_requested"}
        if key in tool.BACKWARD:
            backward = {"status": "completed", "parameters": [{"gradient_present": True, "nonfinite_elements": 0, "all_zero": False, "l2_norm": .1}],
                        "head_logits": [{"gradient_present": True, "nonfinite_elements": 0, "all_zero": False} for _ in range(3)]}
        heads = [{"head": name, "nearest_target_matches_torch": True, "formula_matches_at_1e_minus5": True,
                  "nearest_target_class_counts": [10] * 6, "gt_class5": score(),
                  "actual_ce": 1., "independent_ce": 1., "actual_dice_loss": .5, "independent_dice_loss": .5}
                 for name in ("main", "half", "quarter")]
        models = {"unet2d": {"parameters_and_buffers_unchanged": True, "all_head_logits_finite": True, "gt_class5": score()},
                  "improved3d": {"parameters_and_buffers_unchanged": True, "all_head_logits_finite": True,
                                 "all_group_norm_values_finite": True, "heads": heads, "backward": backward,
                                 "group_norm_layers": [{"input": {"variance_finite_population": 2.}, "output": {"variance_finite_population": 1.}}] * 14,
                                 "loss": {"actual_finite": True, "formula_matches_at_1e_minus5": True, "actual": 1.5, "independent_weighted_total": 1.5}}}
        mechanism_cases.append({"key": key, "split": split, "status": "checks_passed", "models": models})
        def variant(after):
            return {"all_main_logits_finite": True,
                    "class5": {"tp": 0 if after else 10, "fp": 0, "fn": 10 if after else 0,
                               "gt_voxels": 10, "predicted_voxels": 0 if after else 10, "dice": 0. if after else 1.},
                    "gt_class5_logits": score(1, 0.) if after else score()}
        intervention_cases.append({"case": key, "split": split, "status": "completed", "source_image_sha256": SHA,
                                   "cache_identity": {"array_sha256": {n: SHA for n in ("image", "mask", "spacing")}},
                                   "input_change": {"baseline_reconstruction_elementwise_equal": True, "changed_voxels_inside_retained_support": 0,
                                                    "different_voxels": 1, "total_voxels": 60, "gt_class5_affected_voxels": 0,
                                                    "original_gt_class5_voxels": 10, "gt_class5_outside_nearest_support": 0},
                                   "models": {n: {"parameters_and_buffers_unchanged": True, "baseline_frozen_cache_FP32": variant(False),
                                                  "without_support_zeroing_FP32": variant(True), "prediction_change": {"different_voxels": 10}} for n in tool.WEIGHTS}})
    model_identity = {n: {"checkpoint_sha256": v, "parameters_and_buffers_unchanged": True, "checkpoint_file_unchanged": True} for n, v in tool.WEIGHTS.items()}
    official_rows = [{"patient": key.split('/')[0], "visit": key.split('/')[1], "whole_decoded_NIfTI_bytes_equal": True,
                      "gzip_header_only_difference_confirmed": True, "whole_decoded_NIfTI_bytes": 100,
                      "whole_decoded_NIfTI_sha256": SHA, "deflate_payload_sha256": SHA}
                     for key in ("B006/Week0", *[f"M015/Week{i}" for i in range(8)])]
    source = {"status": "completed", "issues": [], "input_sha256": {"manifest": tool.FROZEN["manifest"], "cache_identity": tool.FROZEN["identity"], "training_sources": tool.FROZEN["sources"]}, "headers": header_rows, "cases": source_cases}
    mechanism = {"status": "checks_passed", "evidence": {k: {"sha256": v} for k, v in tool.FROZEN.items()}, "models": model_identity, "cases": mechanism_cases}
    cache = {"manifest_sha256": tool.FROZEN["manifest"], "uses_GT_regions": True, "deployment_transform_claimed": False, "cases": rows}
    archive = {"manifest_sha256": tool.FROZEN["manifest"], "files_checked": 422, "case_visits_checked": 211, "source_files_changed": False,
               "decoded_voxels_read": False, "official_catalog_sha256": SHA,
               "files": [{"patient": r["patient"], "visit": r["visit"], "role": role, "candidate_match": role == "image",
                          "global_join_status": "unique_numeric_candidate" if role == "image" else "no_match"}
                         for r in header_rows for role in ("image", "mask")]}
    official = {"files_checked": 9, "all_decoded_NIfTI_bytes_equal": True, "rows": official_rows}
    d2 = {"status": "completed", "protected_inputs_unchanged": True, "all_fourteen_cases_required": True,
          "d1_source_sha256": SHA, "models": model_identity, "cases": intervention_cases}
    reports = dict(zip(tool.ROLES, (source, mechanism, cache, archive, official, d2)))
    provenance = {r: {"report_sha256": SHA, "source_tool_sha256": SHA, "job_id": None} for r in tool.ROLES}
    return reports, provenance


class SourceSummaryTests(unittest.TestCase):
    def test_complete_scopes_all_cases_and_zero_results(self):
        reports, provenance = fixtures()
        summary = tool.build_summary(reports, provenance)
        self.assertEqual(len(summary["cases"]), 14)
        self.assertEqual(summary["source_identity"]["official_container_size_crc"]["image"]["size_crc_candidate_matches"], 211)
        self.assertEqual(summary["source_identity"]["official_mask_whole_nifti"]["n_files"], 9)
        self.assertEqual([g["n_patients"] for g in summary["cache_descriptors"]["groups"]], [26, 6, 1])
        self.assertEqual(len(summary["cache_descriptors"]["m015_cases"]), 8)
        self.assertEqual(summary["plot_data"]["support_intervention"][0]["unet2d"]["without_support_dice"], 0.)
        self.assertEqual(summary["d2_m015_direction_counts"]["improved3d"]["decreased"], 8)
        self.assertLess(len(json.dumps(summary, separators=(',', ':')).encode()), tool.MAX_BYTES)

    def test_unknown_private_metadata_and_paths_are_not_exported(self):
        reports, provenance = fixtures()
        for report in reports.values():
            report["private"] = {"path": "/home/PRIVATE_HOST/token", "url": "ssh://PRIVATE_HOST"}
        provenance["source"]["private"] = "/home/PRIVATE_HOST/token"
        summary = tool.build_summary(reports, provenance)
        encoded = json.dumps(summary)
        self.assertNotIn("PRIVATE_HOST", encoded)
        self.assertNotIn("/home/", encoded)
        self.assertNotIn("ssh://", encoded)
        self.assertNotIn('"headers"', encoded)
        self.assertNotIn('"prediction"', encoded)

    def test_d2_incomplete_or_wrong_binding_rejected(self):
        for field, value in (("status", "running"), ("d1_source_sha256", "b" * 64)):
            reports, provenance = fixtures()
            reports["support_intervention"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                tool.build_summary(reports, provenance)

    def test_other_test_descriptor_case_or_omitted_case_rejected(self):
        reports, provenance = fixtures()
        reports["cache_descriptors"]["cases"][6]["patient"] = "M020"
        with self.assertRaises(ValueError):
            tool.build_summary(reports, provenance)
        reports, provenance = fixtures()
        reports["mechanism"]["cases"].pop()
        with self.assertRaisesRegex(ValueError, "fourteen"):
            tool.build_summary(reports, provenance)

    def test_confusion_formula_and_cross_report_identity_rejected(self):
        reports, provenance = fixtures()
        reports["support_intervention"]["cases"][0]["models"]["unet2d"]["without_support_zeroing_FP32"]["class5"]["dice"] = .5
        with self.assertRaisesRegex(ValueError, "Dice formula"):
            tool.build_summary(reports, provenance)
        reports, provenance = fixtures()
        reports["support_intervention"]["cases"][0]["source_image_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "source identity"):
            tool.build_summary(reports, provenance)

    def test_undefined_descriptors_remain_undefined_not_zero(self):
        reports, provenance = fixtures()
        result = tool.distribution([None, 0., 2.])
        self.assertEqual(result, {"n_defined": 2, "min": 0., "median": 1., "max": 2.})
        self.assertEqual(tool.distribution([None])["median"], None)
        with self.assertRaises(ValueError):
            tool.number(float('nan'))


if __name__ == "__main__":
    unittest.main()
