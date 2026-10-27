"""Complete synthetic B/C/D metadata audit, negative outcomes and no reselection."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from tools import summarize_recovery_m015 as tool

spec = importlib.util.spec_from_file_location("m2_scope_fixture", PROJECT / "tests/test_evaluate_recovery_m015.py")
fixtures = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixtures)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def report(tp_values, digest, protocol_sha, manifest_sha, name):
    cases = []
    for visit, tp in enumerate(tp_values):
        n = 64 * 128 * 128
        counts = [(n-10, n-10, n-20+tp), *[(0, 0, 0)]*4, (10, 10, tp)]
        rows = []
        for i, (gt, pred, overlap) in enumerate(counts):
            union = gt+pred-overlap
            rows.append({"class_index": i, "target_voxels": gt, "predicted_voxels": pred, "intersection": overlap,
                         "union": union, "false_positive_voxels": pred-overlap, "false_negative_voxels": gt-overlap,
                         "dice": 2*overlap/(gt+pred) if gt+pred else None, "iou": overlap/union if union else None})
        cases.append({"patient": "M015", "visit": f"Week{visit}", "shape_dhw": [64,128,128], "per_class": rows,
                      "foreground_mean_dice": tp/10, "foreground_mean_iou": tp/(20-tp),
                      "inference_ms": 1., "peak_allocated_cuda_bytes": None})
    _, aggregate, _ = tool.counts.recompute(cases)
    return {"schema_version": 1, "experiment_id": tool.EXPERIMENT, "model_id": name, "condition": "clean", "split": "test_M015_only",
            "device": "cpu", "amp": False, "saved_arrays": False, "exploratory_after_test": True, "replaces_v1_test": False,
            "protocol_sha256": protocol_sha, "manifest_sha256": manifest_sha, "checkpoint_sha256": digest,
            "cases": cases, "aggregate": aggregate, "mean_inference_ms": 1., "peak_allocated_cuda_bytes": None}


def study(root):
    plan, r1, validation, manifest = fixtures.fixture()
    for i in range(8):
        manifest["records"].append({"patient": "M015", "visit": f"Week{i}", "split": "test",
            "cache_path": f"volumes/M015_Week{i}.npz", "class_voxel_counts": [64*128*128-10,0,0,0,0,10]})
    write(root/"manifest.json", manifest); write(root/"identity.json", {})
    plan["manifest_sha256"] = validation["manifest_sha256"] = tool.sha(root/"manifest.json")
    plan["cache_identity_sha256"] = tool.sha(root/"identity.json")
    (root/"guard.txt").write_text("frozen synthetic source")
    required = {"guard.txt": tool.sha(root/"guard.txt")}
    r1.update(manifest="manifest.json", cache_identity="identity.json", required_file_sha256=required,
              known_patient_retention_check={"mean_prostate_Dice_min": .6818369277754766, "minimum_visit_prostate_Dice_min": .6})
    write(root/"recovery_protocol.json", r1)
    plan.update(recovery_protocol_path="recovery_protocol.json", recovery_protocol_sha256=tool.sha(root/"recovery_protocol.json"),
                val_summary_path="validation.json", required_file_sha256=required)
    validation["protocol_sha256"] = plan["recovery_protocol_sha256"]
    # Freeze C as the VAL choice although D will have the higher M015 score.
    validation["results"]["clean_adapt"]["clean"]["aggregate"]["prostate_dice"] = .85
    validation["recommendation"]["selected_model_id"] = "clean_adapt"
    for relative in validation["input_file_sha256"]:
        write(root/relative, {"fixture": "metadata"})
        validation["input_file_sha256"][relative] = tool.sha(root/relative)
    write(root/"validation.json", validation); plan["val_summary_sha256"] = tool.sha(root/"validation.json")
    old = report([7]*8, plan["models"]["gamma_polarity"]["checkpoint_sha256"], "old-protocol", plan["manifest_sha256"], "gamma_polarity")
    old_path = root/plan["m1_reference"]["evaluation"]; write(old_path, old)
    plan["m1_reference"]["evaluation_sha256"] = tool.sha(old_path)
    write(root/"m2_protocol.json", plan); protocol_sha = tool.sha(root/"m2_protocol.json")
    index = {"schema_version": 1, "experiment_id": tool.EXPERIMENT, "status": "completed", "split": "test_M015_only", "condition": "clean",
             "protocol_sha256": protocol_sha, "manifest_sha256": plan["manifest_sha256"], "cache_identity_sha256": plan["cache_identity_sha256"],
             "recovery_protocol_sha256": plan["recovery_protocol_sha256"], "val_summary_sha256": plan["val_summary_sha256"],
             "frozen_val_recommendation": validation["recommendation"], "device": "cpu", "amp": False,
             "saved_arrays": False, "exploratory_after_test": True, "replaces_v1_test": False, "training_performed": False,
             "protected_files_unchanged": True, "n_cases": 8, "n_patients": 1, "case_keys": list(tool.KEYS),
             "source_file_sha256": required, "models": {}, "results": {},
             "runtime": {"device": "cpu", "gpu_name": None, "amp": False, "torch_num_threads": 4, "omp_num_threads": 4,
                         "python": "3.11.15", "torch": "2.13.0+cu130", "cuda": None, "mkldnn_enabled": True},
             "cache_verification": [{"case_key": f"M015/Week{i}", "cache_path": f"volumes/M015_Week{i}.npz",
                 "decoded_identity_verified": True, "cache_file_sha256": "e"*64} for i in range(8)]}
    for name, values in zip(tool.MODELS, ([7]*7+[6], [8]*7+[0], [9]*8)):
        spec = plan["models"][name]; cp_root = root/Path(spec["checkpoint"]).parent
        for filename in ("run.json", "history.json", "cache_verification.json"): write(cp_root/filename, {})
        provenance = {"checkpoint_sha256": spec["checkpoint_sha256"], "run_sha256": tool.sha(cp_root/"run.json"),
                      "history_sha256": tool.sha(cp_root/"history.json"), "cache_verification_path": str((cp_root/"cache_verification.json").relative_to(root)),
                      "cache_verification_sha256": tool.sha(cp_root/"cache_verification.json")}
        write(cp_root/"provenance.json", provenance)
        receipt = {"checkpoint_sha256": spec["checkpoint_sha256"], "checkpoint_epoch": spec["selected_epoch"],
                   "state_before_sha256": "f"*64, "state_after_sha256": "f"*64, "state_unchanged": True,
                   **{field: tool.sha(cp_root/f) for f,field in (("run.json","run_sha256"),("history.json","history_sha256"),
                       ("provenance.json","provenance_sha256"),("cache_verification.json","cache_verification_sha256"))}}
        index["models"][name] = receipt
        raw = report(values, spec["checkpoint_sha256"], protocol_sha, plan["manifest_sha256"], name)
        relative = f"{name}/evaluation.json"; write(root/"m2"/relative, raw)
        index["results"][name] = {"evaluation_file": relative, "evaluation_sha256": tool.sha(root/"m2"/relative), "aggregate": raw["aggregate"]}
        if name == "gamma_polarity": index["M1_B_reference_comparison"] = tool.evaluator.compare_m1(raw, old)
    write(root/"m2/recovery_m015.json", index)
    return root/"m2_protocol.json", root/"m2/recovery_m015.json", plan


class SummaryGuards(unittest.TestCase):
    def setup_study(self, root):
        # Fixture-specific identity constants, never production bypass flags.
        with patch.object(tool.evaluator.recovery.original, "MANIFEST_SHA", "fixture-placeholder"):
            # compare_m1 in fixture creation requires the computed manifest SHA.
            original_compare = tool.evaluator.compare_m1
            with patch.object(tool.evaluator, "compare_m1", lambda new,old: {"fixture_pending": True}):
                protocol,index,plan = study(root)
        with patch.object(tool.evaluator.recovery.original, "MANIFEST_SHA", plan["manifest_sha256"]):
            raw = tool.read(root/"m2/gamma_polarity/evaluation.json"); old = tool.read(root/plan["m1_reference"]["evaluation"])
            data=tool.read(index); data["M1_B_reference_comparison"]=original_compare(raw,old);write(index,data)
        return protocol,index,plan

    def audit(self, root, protocol, index, plan):
        with patch.object(tool.evaluator.recovery.original,"MANIFEST_SHA",plan["manifest_sha256"]), \
             patch.object(tool.evaluator.recovery.original,"IDENTITY_SHA",plan["cache_identity_sha256"]), \
             patch.object(tool.evaluator,"M1_SHA",plan["m1_reference"]["evaluation_sha256"]):
            return tool.summarize(protocol,index,root)

    def test_complete_24_rows_zero_empty_negative_retention_and_no_reselection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);protocol,index,plan=self.setup_study(root)
            idx=tool.read(index);idx["runtime"]["hostname"]="/synthetic/private/host";write(index,idx)
            data=self.audit(root,protocol,index,plan)
            self.assertEqual(sum(len(r["cases"])for r in data["results"].values()),24)
            self.assertEqual(data["frozen_val_recommendation"]["selected_model_id"],"clean_adapt")
            self.assertGreater(data["results"]["polarity_rehearsal"]["mean_prostate_dice"],data["results"]["clean_adapt"]["mean_prostate_dice"])
            self.assertFalse(data["prespecified_retention_check"]["models"]["clean_adapt"]["both_checks_pass"])
            self.assertTrue(data["prespecified_retention_check"]["models"]["polarity_rehearsal"]["both_checks_pass"])
            self.assertEqual(data["results"]["clean_adapt"]["minimum_visit_prostate_dice"],0.)
            self.assertIsNone(data["results"]["clean_adapt"]["cases"][0]["per_class"][1]["dice"])
            self.assertFalse(data["scope"]["checkpoint_files_read"])
            self.assertAlmostEqual(data["M1_B_reference_comparison"]["M2_minus_M1_prostate_dice"],-.0125)
            self.assertEqual(data["M1_B_reference_comparison"]["cases"][-1]["per_class"][5]["fp_delta"],1)
            self.assertNotIn(str(root),json.dumps(data))
            self.assertNotIn("/synthetic/private/host",json.dumps(data))

    def test_score_tampering_or_missing_visit_is_rejected(self):
        for issue in ("dice","visit","m1delta","cpu","state","sha","privacy"):
            with self.subTest(issue=issue),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);protocol,index,plan=self.setup_study(root);idx=tool.read(index)
                if issue in ("dice","visit"):
                    p=root/"m2/clean_adapt/evaluation.json";raw=tool.read(p)
                    if issue=="dice":raw["cases"][0]["per_class"][5]["dice"] = .99
                    else:raw["cases"].pop()
                    write(p,raw);idx["results"]["clean_adapt"]["evaluation_sha256"]=tool.sha(p)
                if issue=="m1delta":idx["M1_B_reference_comparison"]["M2_minus_M1_prostate_dice"] = .5
                if issue=="cpu":idx["runtime"]["torch_num_threads"]=8
                if issue=="state":idx["models"]["clean_adapt"]["state_after_sha256"]="a"*64
                if issue=="sha":idx["results"]["clean_adapt"]["evaluation_sha256"]="a"*64
                if issue=="privacy":idx["runtime"]["torch"]="/synthetic/private/runtime"
                write(index,idx)
                with self.assertRaises(ValueError):self.audit(root,protocol,index,plan)

    def test_source_or_frozen_val_recommendation_drift_is_rejected(self):
        for issue in ("source","recommendation"):
            with self.subTest(issue=issue),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);protocol,index,plan=self.setup_study(root)
                if issue=="source":(root/"guard.txt").write_text("changed")
                else:
                    idx=tool.read(index);idx["frozen_val_recommendation"]["selected_model_id"]="polarity_rehearsal";write(index,idx)
                with self.assertRaises(ValueError):self.audit(root,protocol,index,plan)

    def test_existing_output_refused_before_any_input_read(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/"exists.json";output.write_text("preserve")
            with self.assertRaisesRegex(ValueError,"already exists"):
                tool.main(["--protocol","/missing","--evaluation","/missing","--output",str(output)])


if __name__=="__main__":unittest.main()
