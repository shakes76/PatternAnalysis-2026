"""Array-free synthetic metric, identity and full-study receipt fixtures."""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from experiments import summarize_appearance as summary


def case(patient="P1", visit="Week0", missed=False, shape=(1, 1, 6)):
    counts = [math.prod(shape) - 5] + [1] * 5
    rows = []
    for i in range(6):
        target, pred, tp = counts[i], counts[i], counts[i]
        if missed and i == 0:
            pred += 1
        if missed and i == 5:
            pred, tp = 0, 0
        union = pred + target - tp
        rows.append({"class_index": i, "intersection": tp, "predicted_voxels": pred,
                     "target_voxels": target, "union": union, "dice": 2 * tp / (pred + target), "iou": tp / union})
    return {"patient": patient, "visit": visit, "shape_dhw": list(shape), "per_class": rows,
            "inference_ms": 3.0, "peak_allocated_cuda_bytes": 100,
            "foreground_mean_dice": sum(r["dice"] for r in rows[1:]) / 5,
            "foreground_mean_iou": sum(r["iou"] for r in rows[1:]) / 5}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def study_fixture(root):
    """Build all 20 outputs and two 30-epoch histories, without actual arrays."""
    shape = [64, 128, 128]
    groups = {s: [f"{dict(train='A', validation='B', test='C')[s]}{i:03}" for i in range(n)]
              for s, n in (("train", 26), ("validation", 6), ("test", 6))}
    records = [{"patient": groups[split][i % len(groups[split])], "visit": f"Week{i // len(groups[split])}", "split": split}
               for split, n in (("train", 143), ("validation", 38), ("test", 30)) for i in range(n)]
    manifest = {"num_classes": 6, "label_ids": list(range(6)), "target_shape_dhw": shape,
                "patient_splits": groups, "records": records}
    write(root / "manifest.json", manifest)
    write(root / "identity.json", {})
    write(root / "training_sources.json", {"synthetic": True})
    sources = {p: summary.sha(root / p) for p in ("manifest.json", "identity.json", "training_sources.json")}
    models = {n: {"kind": "frozen_v1" if n.endswith("reference") else "appearance_arm",
                   "checkpoint": f"models/{n}/best.pt", "seed": 43 if n == "seed43_reference" else 42}
              for n in summary.MODELS}
    for n, spec in models.items():
        path = root / spec["checkpoint"]
        path.parent.mkdir(parents=True)
        path.write_bytes((n + "-not-a-real-model").encode())
        if n.endswith("reference"):
            spec.update(checkpoint_sha256=summary.sha(path), selected_epoch=29 if n == "seed42_reference" else 27)
    plan = {"experiment_id": summary.EXPERIMENT, "allowed_splits": ["train", "validation"],
            "training_seed": 42, "training_epochs_per_model": 30, "conditions": list(summary.CONDITIONS), "models": models,
            "manifest": "manifest.json", "manifest_sha256": summary.sha(root / "manifest.json"), "cache_identity": "identity.json",
            "training_sources_sha256": sources["training_sources.json"], "required_file_sha256": sources,
            "training_conditions": {"seed": 42, "model": "improved3d"}, "inference": {"amp": True, "split": "validation"},
            "engineering_gate": {"B_clean_prostate_Dice_minus_A_min": -0.02, "B_clean_foreground_Dice_minus_A_min": -0.02,
                                 "B_polarity_prostate_Dice_minus_A_min": 0.05}, "limits": ["Synthetic fixture."]}
    protocol = root / "protocol.json"
    write(protocol, plan)
    digest = summary.sha(protocol)
    cases = [case(r["patient"], r["visit"], shape=shape) for r in records if r["split"] == "validation"]
    keys = {(c["patient"], c["visit"]) for c in cases}
    aggregate, _ = summary.recompute(cases, keys, shape)
    evaluation = {"cases": cases, "aggregate": aggregate, "mean_inference_ms": 3.0, "peak_allocated_cuda_bytes": 100}
    receipts = {}
    for name, spec in models.items():
        modelroot = (root / spec["checkpoint"]).parent
        receipt = {"checkpoint_sha256": summary.sha(root / spec["checkpoint"]), "checkpoint_epoch": spec.get("selected_epoch", 1),
                   "seed": spec["seed"], "parameters": 100, "state_unchanged": True, "state_before_sha256": "b" * 64}
        if name in summary.MODELS[2:]:
            experiment = {"experiment_id": summary.EXPERIMENT, "arm_tag": name, "protocol_sha256": digest}
            verification = {"status": "verified", "manifest_sha256": plan["manifest_sha256"],
                            "counts": {"train": 143, "validation": 38}, "test_arrays_opened": 0,
                            "records": [{**r, "decoded_identity_verified": True} for r in records if r["split"] != "test"]}
            write(modelroot / "cache_verification.json", verification)
            cache_sha = summary.sha(modelroot / "cache_verification.json")
            run = {"experiment": experiment, "config": plan["training_conditions"], "status": "completed", "completed_epochs": 30,
                   "requested_total_epochs": 30, "development_only": False, "source_file_sha256": sources,
                   "cache_verification_sha256": cache_sha, "best_epoch": 1, "best_validation_foreground_dice": 1.0,
                   "initial_parameter_state_sha256": "a" * 64,
                   "runtime": {"gpu_name": "synthetic GPU", "python": "3", "torch": "2", "cuda": "13"}}
            history = [{"epoch": epoch, "training_examples": 143, "training_batches": 143, "development_only": False,
                        "optimizer_steps": 143, "amp_overflow_skipped_optimizer_steps": 0, "validation": evaluation,
                        "validation_foreground_dice": 1.0, "train_loss": 0.1, "train_seconds": 1.0,
                        "peak_allocated_cuda_bytes_training": 200} for epoch in range(1, 31)]
            write(modelroot / "run.json", run)
            write(modelroot / "history.json", history)
            provenance = {"experiment": experiment, "protocol_sha256": digest, "manifest_sha256": plan["manifest_sha256"],
                          "source_file_sha256": sources, "checkpoint_sha256": receipt["checkpoint_sha256"],
                          "run_sha256": summary.sha(modelroot / "run.json"), "history_sha256": summary.sha(modelroot / "history.json"),
                          "cache_verification_sha256": cache_sha, "cache_verification_path": str((modelroot / "cache_verification.json").relative_to(root))}
            write(modelroot / "provenance.json", provenance)
            receipt.update({field: summary.sha(modelroot / filename) for field, filename in
                            (("run_sha256", "run.json"), ("history_sha256", "history.json"), ("provenance_sha256", "provenance.json"))})
            receipt["cache_verification_sha256"] = cache_sha
        receipts[name] = receipt
    index = {"status": "completed", "experiment_id": summary.EXPERIMENT, "split": "validation", "protocol_sha256": digest,
             "manifest_sha256": plan["manifest_sha256"], "cache_identity_sha256": sources["identity.json"], "models": receipts,
             "conditions": list(summary.CONDITIONS), "n_scans_per_condition": 38, "n_unique_patients": 6, "saved_arrays": False,
             "runtime": {"device": "cuda", "amp": True}, "results": {}}
    outputroot = root / "eval"
    for name in summary.MODELS:
        index["results"][name] = {}
        for condition in summary.CONDITIONS:
            relative = f"{name}/{condition}/evaluation.json"
            value = {**evaluation, "model_id": name, "condition": condition, "split": "validation", "saved_arrays": False,
                     "protocol_sha256": digest, "manifest_sha256": plan["manifest_sha256"],
                     "checkpoint_sha256": receipts[name]["checkpoint_sha256"], "amp": True}
            write(outputroot / relative, value)
            index["results"][name][condition] = {"evaluation_file": relative, "evaluation_sha256": summary.sha(outputroot / relative),
                                                "aggregate": aggregate, "mean_inference_ms": 3.0, "peak_allocated_cuda_bytes": 100}
    indexpath = outputroot / "appearance_evaluation.json"
    write(indexpath, index)
    return protocol, indexpath


class SummaryTests(unittest.TestCase):
    def test_patient_weighting_keeps_zero_dice(self):
        cases = [case("P1", "Week0", True), case("P1", "Week1", True), case("P2", "Week0")]
        aggregate, compact = summary.recompute(cases, {(r["patient"], r["visit"]) for r in cases}, [1, 1, 6])
        self.assertEqual(aggregate["per_class"][5]["dice"], 0.5)
        self.assertEqual(aggregate["per_class"][5]["iou"], 0.5)
        self.assertEqual([r["dice"] for r in compact], [0, 0, 1])
        self.assertEqual((compact[0]["tp"], compact[0]["fp"], compact[0]["fn"]), (0, 0, 1))

    def test_count_corruption_and_duplicate_scope_are_rejected(self):
        value = case()
        value["per_class"][5]["dice"] = 0.99
        with self.assertRaisesRegex(ValueError, "number differs"):
            summary.recompute([value], {("P1", "Week0")}, [1, 1, 6])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summary.recompute([case(), case()], {("P1", "Week0")}, [1, 1, 6])

    def test_both_empty_class_remains_none(self):
        value = case()
        value["per_class"][0].update(intersection=2, predicted_voxels=2, target_voxels=2, union=2)
        value["per_class"][5].update(intersection=0, predicted_voxels=0, target_voxels=0, union=0, dice=None, iou=None)
        aggregate, compact = summary.recompute([value], {("P1", "Week0")}, [1, 1, 6])
        self.assertIsNone(aggregate["per_class"][5]["dice"])
        self.assertIsNone(compact[0]["iou"])
        self.assertEqual(aggregate["foreground_mean_dice"], 1.0)

    def test_guard_uses_all_three_criteria_and_requires_pairing(self):
        result = {n: {c: {"aggregate": {"prostate_dice": 0.8, "foreground_mean_dice": 0.85}}
                      for c in summary.CONDITIONS} for n in summary.MODELS[2:]}
        result["gamma_polarity"]["polarity"]["aggregate"]["prostate_dice"] = 0.9
        self.assertTrue(summary.engineering_gate(result, True)["development_indication_supported"])
        self.assertFalse(summary.engineering_gate(result, False)["development_indication_supported"])
        result["gamma_polarity"]["clean"]["aggregate"]["prostate_dice"] = 0.77
        self.assertFalse(summary.engineering_gate(result, True)["numeric_gate_passes"])

    def test_complete_study_and_tampered_file_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol, index = study_fixture(root)
            result = summary.summarize(protocol, index, root)
            self.assertEqual(result["scope"]["n_scans"], 38)
            self.assertEqual(len(result["results"]), 4)
            self.assertEqual(len(result["plot_data"]["patient_arm_deltas"]), 6)
            self.assertEqual(result["training"]["gamma_only"]["total_optimizer_steps"], 4290)
            self.assertTrue(all(result["paired_execution_checks"].values()))
            self.assertFalse(result["engineering_gate"]["numeric_gate_passes"])
            altered = root / "eval/gamma_polarity/polarity/evaluation.json"
            altered.write_text(altered.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "Evaluation file SHA"):
                summary.summarize(protocol, index, root)

    def test_test_split_is_rejected_before_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol, index = study_fixture(root)
            value = json.loads(index.read_text())
            value["split"] = "test"
            write(index, value)
            with self.assertRaisesRegex(ValueError, "split"):
                summary.summarize(protocol, index, root)

    def test_documented_runtime_confound_disables_indication_without_hiding_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol, indexpath = study_fixture(root)
            armroot = root / "models/gamma_polarity"
            run = json.loads((armroot / "run.json").read_text())
            run["runtime"]["gpu_name"] = "different synthetic GPU"
            write(armroot / "run.json", run)
            provenance = json.loads((armroot / "provenance.json").read_text())
            provenance["run_sha256"] = summary.sha(armroot / "run.json")
            write(armroot / "provenance.json", provenance)
            index = json.loads(indexpath.read_text())
            index["models"]["gamma_polarity"].update(run_sha256=provenance["run_sha256"],
                                                     provenance_sha256=summary.sha(armroot / "provenance.json"))
            write(indexpath, index)
            result = summary.summarize(protocol, indexpath, root)
            self.assertFalse(result["paired_execution_checks"]["training_runtime_exactly_equal"])
            self.assertTrue(result["confounds"])
            self.assertFalse(result["engineering_gate"]["development_indication_supported"])
            self.assertEqual(len(result["results"]["gamma_polarity"]), 5)


if __name__ == "__main__":
    unittest.main()
