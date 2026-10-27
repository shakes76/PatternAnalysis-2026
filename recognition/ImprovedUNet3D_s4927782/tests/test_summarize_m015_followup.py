"""Synthetic counts and refusal checks; no MRI, checkpoints, or real results."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("m015_summary", Path(__file__).resolve().parents[1] / "tools/summarize_m015_followup.py")
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    (root / "frozen.py").write_text("# synthetic source identity\n")
    digest = "a" * 64
    protocol = {"experiment_id": tool.EXPERIMENT, "test_informed_exploratory_acknowledged": True,
                "case_keys": list(tool.KEYS), "models": {n: {"checkpoint_sha256": tool.hashlib.sha256(n.encode()).hexdigest()} for n in tool.MODELS},
                "required_file_sha256": {"frozen.py": tool.sha(root / "frozen.py")},
                "manifest_sha256": digest, "cache_identity_sha256": digest, "appearance_protocol_sha256": digest}
    write(root / "m015_followup_protocol.json", protocol)
    common = {"schema_version": 1, "experiment_id": tool.EXPERIMENT, "split": "test_M015_only", "device": "cpu", "amp": False,
              "exploratory_after_test": True, "replaces_v1_test": False, "saved_arrays": False, "condition": "clean",
              "protocol_sha256": tool.sha(root / "m015_followup_protocol.json"), "manifest_sha256": digest}
    index = {**common, "status": "completed", "protected_files_unchanged": True, "training_performed": False,
             "n_cases": 8, "n_patients": 1, "case_keys": list(tool.KEYS), "source_file_sha256": protocol["required_file_sha256"],
             "appearance_protocol_sha256": digest, "cache_identity_sha256": digest, "models": {}, "results": {},
             "runtime": {"device": "cpu", "amp": False, "gpu_name": None, "torch_num_threads": 4, "omp_num_threads": 4},
             "cache_verification": [{"case_key": k, "decoded_identity_verified": True, "cache_file_sha256": digest} for k in tool.KEYS],
             "not_for_publication": "/Users/private/credential"}
    volume = 64 * 128 * 128
    for model_index, name in enumerate(tool.MODELS):
        cases = []
        for week in range(8):
            prostate_tp = week + 1 + model_index
            if name == "gamma_only" and week == 2:
                prostate_tp = 1  # Preserve a real negative visit rather than selecting only improvements.
            counts = [volume - 13, 1, 1, 1, 0, 10]
            overlaps = [volume - 13 - (10 - prostate_tp), 1, 1, 1, 0, prostate_tp]
            rows = []
            for i, (gt, tp) in enumerate(zip(counts, overlaps)):
                rows.append({"class_index": i, "target_voxels": gt, "predicted_voxels": gt, "intersection": tp,
                             "union": 2 * gt - tp, "false_positive_voxels": gt - tp, "false_negative_voxels": gt - tp,
                             "dice": tp / gt if gt else None, "iou": tp / (2 * gt - tp) if gt else None})
            cases.append({"patient": "M015", "visit": f"Week{week}", "shape_dhw": tool.SHAPE, "per_class": rows,
                          "foreground_mean_dice": (3 + prostate_tp / 10) / 4,
                          "foreground_mean_iou": (3 + prostate_tp / (20 - prostate_tp)) / 4})
        per_class = [{"class_index": i, **{m: sum(c["per_class"][i][m] for c in cases) / 8 if i != 4 else None for m in ("dice", "iou")}} for i in range(6)]
        aggregate = {"num_cases": 8, "num_patients": 1, "per_class": per_class,
                     **{f"foreground_mean_{m}": sum(c[f"foreground_mean_{m}"] for c in cases) / 8 for m in ("dice", "iou")}}
        aggregate["patients"] = [{"patient": "M015", "num_cases": 8, "per_class": per_class,
                                  **{f"foreground_mean_{m}": aggregate[f"foreground_mean_{m}"] for m in ("dice", "iou")}}]
        evaluation = {**common, "model_id": name, "checkpoint_sha256": protocol["models"][name]["checkpoint_sha256"],
                      "cases": cases, "aggregate": aggregate, "private_path": "/private/unpublished"}
        relative = f"{name}/evaluation.json"
        write(root / "evaluations" / relative, evaluation)
        index["models"][name] = {"checkpoint_sha256": protocol["models"][name]["checkpoint_sha256"], "checkpoint_epoch": 29,
                                 "state_before_sha256": digest, "state_after_sha256": digest, "state_unchanged": True}
        index["results"][name] = {"evaluation_file": relative, "evaluation_sha256": tool.sha(root / "evaluations" / relative), "aggregate": aggregate}
    path = root / "evaluations/m015_followup.json"
    write(path, index)
    return path, index


class FollowupSummaryTests(unittest.TestCase):
    def test_complete_count_audit_keeps_negative_visit_absence_and_private_fields_out(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path, _ = fixture(root)
            report = tool.summarize(path, root)
            self.assertEqual(report["scope"]["retained_case_rows"], 24)
            self.assertEqual(report["scope"]["independent_patients"], 1)
            self.assertAlmostEqual(report["results"]["seed42_reference"]["aggregate"]["per_class"][5]["dice"], .45)
            self.assertAlmostEqual(report["results"]["gamma_only"]["aggregate"]["per_class"][5]["dice"], .5125)
            self.assertAlmostEqual(report["results"]["gamma_polarity"]["aggregate"]["per_class"][5]["dice"], .65)
            row = report["results"]["gamma_polarity"]["pooled_counts"][5]
            self.assertEqual((row["tp"], row["fp"], row["fn"]), (52, 28, 28))
            self.assertAlmostEqual(row["pooled_voxel_recall"], .65)
            self.assertIsNone(report["results"]["gamma_only"]["aggregate"]["per_class"][4]["dice"])
            self.assertAlmostEqual(report["new_minus_same_cpu_reference"]["gamma_only"]["per_visit"][2]["per_class"][5]["dice"], -.2)
            self.assertNotIn("/private", json.dumps(report)); self.assertNotIn("/Users", json.dumps(report))

    def test_file_arithmetic_scope_and_path_tampering_are_rejected(self):
        for mistake in ("sha", "arithmetic", "case", "path", "precision", "checkpoint"):
            with self.subTest(mistake=mistake), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); path, index = fixture(root)
                source = root / "evaluations/gamma_only/evaluation.json"
                raw = json.loads(source.read_text())
                if mistake == "sha":
                    raw["unexpected"] = 1; write(source, raw)
                elif mistake in ("arithmetic", "case"):
                    if mistake == "arithmetic": raw["cases"][0]["per_class"][5]["dice"] = .99
                    else: raw["cases"][0]["patient"] = "OTHER"
                    write(source, raw); index["results"]["gamma_only"]["evaluation_sha256"] = tool.sha(source)
                elif mistake == "path": index["results"]["gamma_only"]["evaluation_file"] = "../private.json"
                elif mistake == "precision": index["amp"] = True
                else: index["models"]["gamma_only"]["checkpoint_sha256"] = "f" * 64
                write(path, index)
                with self.assertRaises(ValueError): tool.summarize(path, root)

    def test_source_mismatch_and_existing_output_fail_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path, _ = fixture(root)
            (root / "frozen.py").write_text("changed")
            with self.assertRaisesRegex(ValueError, "SHA differs"): tool.summarize(path, root)
            output = root / "existing.json"; output.write_bytes(b"keep exactly")
            with self.assertRaisesRegex(ValueError, "refusing overwrite"):
                tool.main(["--evaluation", "/missing", "--output", str(output)])
            self.assertEqual(output.read_bytes(), b"keep exactly")


if __name__ == "__main__":
    unittest.main()
