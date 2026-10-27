"""Synthetic metadata/scope tests; no real arrays, checkpoint load or inference."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from tools import evaluate_recovery_m015 as tool


def fixture():
    r1 = {"references": {n: {"checkpoint": f"artifacts/{n}/best.pt", "checkpoint_sha256": h,
                             "selected_epoch": e} for n, (e, h) in tool.recovery.REFERENCES.items()},
          "models": {n: {"checkpoint": f"artifacts/recovery/{n}/best.pt"} for n in tool.recovery.ARMS}}
    r1["warm_start"] = {k: r1["references"]["gamma_polarity"][k] for k in ("checkpoint", "checkpoint_sha256")}
    models = {"gamma_polarity": copy.deepcopy(r1["references"]["gamma_polarity"])}
    models.update({n: {**r1["models"][n], "checkpoint_sha256": chr(97+i) * 64, "selected_epoch": i+2}
                   for i, n in enumerate(tool.recovery.ARMS)})
    plan = {"experiment_id": tool.EXPERIMENT, "test_informed_exploratory_acknowledged": True,
            "case_keys": list(tool.CASE_KEYS), "models": models, "manifest_sha256": tool.recovery.original.MANIFEST_SHA,
            "cache_identity_sha256": tool.recovery.original.IDENTITY_SHA,
            "training_sources_sha256": tool.recovery.original.SOURCES_SHA, "recovery_protocol_sha256": "c" * 64,
            "inference": {"split": "test", "device": "cpu", "amp": False, "saved_arrays": False},
            "m1_reference": {"evaluation": "artifacts/M1/gamma_polarity/evaluation.json", "evaluation_sha256": tool.M1_SHA}}
    records = [{"patient": f"VAL{i % 6}", "visit": f"Week{i // 6}", "split": "validation"} for i in range(38)]
    manifest = {"records": records}
    summary = {"experiment_id": tool.recovery.EXPERIMENT, "split": "validation", "status": "completed",
               "protocol_sha256": plan["recovery_protocol_sha256"], "manifest_sha256": plan["manifest_sha256"],
               "scope": {"n_models": 5, "n_conditions": 5, "n_scans": 38, "n_patients": 6, "n_case_evaluations": 950,
                         "test_arrays_read_by_summarizer": 0, "arrays_loaded": False}, "models": {}, "results": {},
               "input_file_sha256": {}, "recommendation": {"selected_model_id": "polarity_rehearsal",
                   "eligible_model_ids": list(tool.recovery.ARMS), "selection_basis": "validation_only", "same_platform_verified": True,
                   "tie_break_order": ["clean_prostate_dice_desc", "clean_foreground_dice_desc", "model_id_asc"]}}
    for name in tool.recovery.MODEL_IDS:
        spec = models[name] if name in models else r1["references"][name]
        summary["models"][name] = {"checkpoint_sha256": spec["checkpoint_sha256"], "checkpoint_epoch": spec["selected_epoch"], "state_unchanged": True}
        summary["results"][name] = {}
        for condition in tool.recovery.CONDITIONS:
            value, fg = .80, .87
            if name == "seed42_reference" and condition == "clean": value = .83
            if name in tool.recovery.ARMS and condition == "clean": value = .84
            if name == "polarity_rehearsal" and condition == "clean": fg = .875
            if condition == "polarity": value = .79 if name == "gamma_polarity" else .78
            tp = round(100 * value)
            rows = [{"patient": r["patient"], "visit": r["visit"], "tp": tp, "fp": 100-tp, "fn": 100-tp,
                     "dice": value, "iou": tp/(200-tp)} for r in records]
            summary["results"][name][condition] = {"aggregate": {"prostate_dice": value, "foreground_mean_dice": fg},
                "cases_prostate": rows, "patients": [{"patient": f"VAL{i}"} for i in range(6)]}
            summary["input_file_sha256"][f"artifacts/R1/{name}/{condition}/evaluation.json"] = "d" * 64
    return plan, r1, summary, manifest


class ScopeGuards(unittest.TestCase):
    def test_exact_plan_and_complete_frozen_val_choice_are_accepted(self):
        plan, r1, summary, manifest = fixture()
        tool.validate_protocol(plan, r1)
        choice = tool.validate_val_summary(summary, plan, r1, manifest)
        self.assertEqual(choice["selected_model_id"], "polarity_rehearsal")

    def test_scope_precision_checkpoint_and_ack_changes_are_rejected(self):
        for issue in ("ack", "case", "model", "precision", "parent", "epoch", "m1"):
            with self.subTest(issue=issue):
                plan, r1, _, _ = fixture()
                if issue == "ack": plan["test_informed_exploratory_acknowledged"] = False
                if issue == "case": plan["case_keys"][0] = "B037/Week0"
                if issue == "model": plan["models"]["seed42_reference"] = plan["models"]["gamma_polarity"]
                if issue == "precision": plan["inference"]["amp"] = True
                if issue == "parent": plan["models"]["gamma_polarity"]["checkpoint_sha256"] = "a" * 64
                if issue == "epoch": plan["models"]["clean_adapt"]["selected_epoch"] = 9
                if issue == "m1": plan["m1_reference"]["evaluation_sha256"] = "a" * 64
                with self.assertRaises(ValueError): tool.validate_protocol(plan, r1)

    def test_incomplete_summary_or_test_condition_or_changed_identity_fails(self):
        for issue in ("status", "scope", "condition", "case", "patient", "sha", "input"):
            with self.subTest(issue=issue):
                plan, r1, summary, manifest = fixture()
                if issue == "status": summary["status"] = "running"
                if issue == "scope": summary["scope"]["n_case_evaluations"] = 949
                if issue == "condition": summary["results"]["gamma_only"].pop("clean")
                if issue == "case": summary["results"]["gamma_only"]["clean"]["cases_prostate"][0]["patient"] = "M015"
                if issue == "patient": summary["results"]["gamma_only"]["clean"]["patients"].pop()
                if issue == "sha": summary["models"]["clean_adapt"]["checkpoint_sha256"] = "f" * 64
                if issue == "input": summary["input_file_sha256"].pop(next(iter(summary["input_file_sha256"])))
                with self.assertRaises(ValueError): tool.validate_val_summary(summary, plan, r1, manifest)

    def test_recommendation_must_follow_same_platform_val_gate_and_ties(self):
        for issue in ("M015", "choice", "eligible", "platform", "tie"):
            with self.subTest(issue=issue):
                plan, r1, summary, manifest = fixture()
                if issue == "M015": summary["recommendation"]["selection_basis"] = "M015"
                if issue == "choice": summary["recommendation"]["selected_model_id"] = "clean_adapt"
                if issue == "eligible": summary["recommendation"]["eligible_model_ids"].pop()
                if issue == "platform": summary["recommendation"]["same_platform_verified"] = False
                if issue == "tie": summary["results"]["polarity_rehearsal"]["clean"]["aggregate"]["foreground_mean_dice"] = .87
                with self.assertRaises(ValueError): tool.validate_val_summary(summary, plan, r1, manifest)

    def test_no_eligible_candidate_is_valid_but_cannot_be_replaced_by_M015_choice(self):
        plan, r1, summary, manifest = fixture()
        for name in tool.recovery.ARMS:
            summary["results"][name]["clean"]["aggregate"]["prostate_dice"] = .82
        summary["recommendation"].update(eligible_model_ids=[], selected_model_id=None)
        self.assertIsNone(tool.validate_val_summary(summary, plan, r1, manifest)["selected_model_id"])
        summary["recommendation"]["selected_model_id"] = "clean_adapt"
        with self.assertRaises(ValueError): tool.validate_val_summary(summary, plan, r1, manifest)

    def test_existing_output_rejected_before_protocol_or_dependency_read(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "must be new"):
                tool.main(["--protocol", "/missing", "--manifest", "/missing", "--output-dir", directory])


class HistoricalComparison(unittest.TestCase):
    def test_cross_host_differences_are_reported_instead_of_rejected(self):
        rows = [{"patient": "M015", "visit": f"Week{i}", "per_class": [
            {"class_index": c, "target_voxels": 100, "predicted_voxels": 100, "intersection": 70,
             "dice": .7, "iou": 70/130, "union": 130} for c in range(6)]} for i in range(8)]
        old = {"model_id": "gamma_polarity", "checkpoint_sha256": tool.recovery.REFERENCES["gamma_polarity"][1],
               "manifest_sha256": tool.recovery.original.MANIFEST_SHA, "device": "cpu", "amp": False,
               "cases": rows, "aggregate": {"per_class": [{"dice": .7} for _ in range(6)]}}
        new = copy.deepcopy(old)
        new["cases"][0]["per_class"][5].update(intersection=71, dice=.71, iou=71/129, union=129)
        new["aggregate"]["per_class"][5]["dice"] = .70125
        result = tool.compare_m1(new, old)
        self.assertAlmostEqual(result["M2_minus_M1_prostate_dice"], .00125)
        changed = result["cases"][0]["per_class"][5]
        self.assertEqual((changed["tp_delta"], changed["fp_delta"], changed["fn_delta"]), (1, -1, -1))
        self.assertAlmostEqual(changed["dice_delta"], .01)
        new["cases"][0]["per_class"][5]["target_voxels"] = 99
        with self.assertRaisesRegex(ValueError, "GT counts"):
            tool.compare_m1(new, old)


if __name__ == "__main__":
    unittest.main()
