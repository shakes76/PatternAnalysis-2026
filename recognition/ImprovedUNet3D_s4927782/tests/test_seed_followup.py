"""JSON-only synthetic follow-up contracts; never reads the real test split."""

import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest


path = Path(__file__).resolve().parents[1] / "tools" / "analyze_seed_followup.py"
spec = importlib.util.spec_from_file_location("seed_followup", path)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def validation(tp_values):
    cases, patients = [], []
    for index, tp in enumerate(tp_values):
        classes = []
        for c in range(6):
            count = (34 if tp is None else 24) if c == 0 else 0 if c == 4 or c == 5 and tp is None else 10
            intersection = count if c in (0, 2, 3) or c == 1 and tp is None else 0 if c == 4 else tp
            intersection = intersection or 0
            classes.append({"class_index": c, "dice": intersection / count if count else None,
                            "iou": intersection / (2 * count - intersection) if count else None,
                            "target_voxels": count, "predicted_voxels": count, "intersection": intersection,
                            "union": 2 * count - intersection})
        foreground = {f"foreground_mean_{m}": analysis.mean_defined(row[m] for row in classes[1:]) for m in ("dice", "iou")}
        row = {"patient": f"V{index}", "visit": "Week0", "per_class": classes, **foreground}
        cases.append(row)
        patients.append({"patient": row["patient"], "num_cases": 1, "per_class": classes, **foreground})
    for patient in patients:
        patient["per_class"] = [{k: row[k] for k in ("class_index", "dice", "iou")} for row in patient["per_class"]]
    aggregate = {"num_cases": 6, "num_patients": 6, "patients": patients,
                 "per_class": [{"class_index": c, **{m: analysis.mean_defined(row["per_class"][c][m] for row in patients)
                                                   for m in ("dice", "iou")}} for c in range(6)],
                 **{f"foreground_mean_{m}": analysis.mean_defined(row[f"foreground_mean_{m}"] for row in patients) for m in ("dice", "iou")}}
    return {"cases": cases, "aggregate": aggregate}


class FollowupFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest_path = self.root / "manifest.json"
        self.protocol_path = self.root / "followup_protocol.json"
        self.manifest = {"partial": False, "development_only": False, "num_classes": 6,
                         "label_ids": list(range(6)), "target_shape_dhw": [4, 4, 4],
                         "patient_splits": {"train": ["T0"], "validation": [f"V{i}" for i in range(6)], "test": ["X0"]},
                         "records": [{"patient": f"V{i}", "visit": "Week0", "split": "validation"} for i in range(6)]}
        write(self.manifest_path, self.manifest)
        manifest_sha = analysis.sha256(self.manifest_path)
        module_path = self.root / "modules.py"
        module_path.write_text("# frozen fixture source\n")
        source = {"manifest_sha256": manifest_sha, "training_code_commit": "fixture-commit",
                  "source_file_sha256": {"modules.py": analysis.sha256(module_path)}}
        write(self.root / "training_sources.json", source)
        source_sha = analysis.sha256(self.root / "training_sources.json")
        self.protocol = {"experiment_id": "paired_seed43_validation", "training_seed": 43,
                         "diagnostic_gate": {"approved_to_proceed": True},
                         "manifest_sha256": manifest_sha, "training_seeds": [42, 43],
                         "training_epochs_per_model": 30, "v1_test_already_observed": True,
                         "allowed_splits": ["train", "validation"], "training_sources_sha256": source_sha,
                         "training_code_commit": "fixture-commit", "training_conditions": {},
                         "required_file_sha256": {"modules.py": analysis.sha256(module_path), "manifest.json": manifest_sha,
                                                  "training_sources.json": source_sha}, "seed42_models": {}}
        for seed in (42, 43):
            for model in analysis.MODELS:
                selected = (26 if model == "unet2d" else 29) + (seed == 43)
                config = {"model": model, "seed": seed, "num_classes": 6, "label_ids": list(range(6)),
                          "target_shape_dhw": [4, 4, 4], "patient_splits": self.manifest["patient_splits"],
                          "manifest_sha256": manifest_sha, "lr": 0.001, "development_only": False, "max_batches": None}
                tps = [0, 4 + (seed == 43), 6, 7 + (model == "improved3d"), 9, None]
                metric = validation(tps)
                checkpoint_sha = ("a" if model == "unet2d" else "b") * 62 + str(seed)
                evaluation = {**copy.deepcopy(metric), "split": "validation", "development_only": False,
                              "final_test_explicitly_requested": False, "model_config": config, "manifest_sha256": manifest_sha,
                              "checkpoint_sha256": checkpoint_sha, "checkpoint_epoch": selected, "checkpoint_best_epoch": selected,
                              "mean_inference_ms": 2.0, "peak_allocated_cuda_bytes": 1024, "gpu_name": "fixture-gpu"}
                score = metric["aggregate"]["foreground_mean_dice"]
                run = {"status": "completed", "config": config, "completed_epochs": 30, "requested_total_epochs": 30,
                       "best_epoch": selected, "best_validation_foreground_dice": score, "parameters": 100, "gpu_name": "fixture-gpu"}
                history = [{"epoch": epoch, "development_only": False,
                            "validation": copy.deepcopy(metric if epoch == selected else validation([0, 0, 0, 0, 0, None])),
                            "train_seconds": 1.25, "optimizer_steps": 3, "amp_overflow_skipped_optimizer_steps": 1,
                            "training_batches": 4, "peak_allocated_cuda_bytes_training": 2048} for epoch in range(1, 31)]
                provenance = {"manifest_sha256": manifest_sha, "expected_sha256": source_sha,
                              "source_file_sha256": source["source_file_sha256"], "training_code_commit": source["training_code_commit"],
                              "checkpoint_sha256": checkpoint_sha}
                for name, data in (("run", run), ("history", history), ("source_provenance", provenance)):
                    write(self.root / f"seed{seed}" / model / f"{name}.json", data)
                write(self.root / f"val{seed}" / model / "evaluation.json", evaluation)
                if seed == 42:
                    self.protocol["seed42_models"][model] = {"selected_epoch": selected, "checkpoint_sha256": checkpoint_sha}
                    self.protocol["training_conditions"][model] = {k: v for k, v in config.items() if k != "seed"}
        write(self.protocol_path, self.protocol)

    def summary(self):
        return analysis.build_summary(self.root / "seed42", self.root / "seed43", self.root / "val42", self.root / "val43", self.manifest_path, self.protocol_path)

    def change_json(self, relative, transform):
        path = self.root / relative
        data = json.loads(path.read_text())
        transform(data)
        write(path, data)

    def prepare_cloud(self):
        training = self.root / "artifacts/followup_seed43_cloud"
        validation_root = self.root / "artifacts/validation_seed43_cloud"
        shutil.copytree(self.root / "seed43", training)
        shutil.copytree(self.root / "val43", validation_root)
        environment = {"python_version": "3.11.9", "torch_version": "2.5.1+cu124",
                       "cuda_version": "12.4", "numpy_version": "2.0.2"}
        for seed in (42, 43):
            for model in analysis.MODELS:
                for kind, path in (("run", (self.root / "seed42" if seed == 42 else training) / model / "run.json"),
                                   ("evaluation", (self.root / "val42" if seed == 42 else validation_root) / model / "evaluation.json")):
                    data = json.loads(path.read_text())
                    data.update({k: v for k, v in environment.items() if k != "numpy_version"})
                    data.update(gpu_name="NVIDIA A100-PCIE-40GB" if seed == 42 else "NVIDIA L40S", device="cuda")
                    if kind == "evaluation": data["amp"] = True
                    write(path, data)
        runner = self.root / "jobs/train_pair_seed43_cloud.sh"
        runner.parent.mkdir()
        runner.write_text("# frozen fixture runner\n")
        source = self.root / "tools/analyze_seed_followup.py"
        source.parent.mkdir()
        source.write_bytes(Path(analysis.__file__).read_bytes())
        required = dict(self.protocol["required_file_sha256"])
        required.update({"followup_protocol.json": analysis.sha256(self.protocol_path),
                         "jobs/train_pair_seed43_cloud.sh": analysis.sha256(runner),
                         "tools/analyze_seed_followup.py": analysis.sha256(source)})
        platform = {"provider": "Virtual GPU Lab", "region": "ap-northeast-1", "instance_type": "g6e.xlarge",
                    "gpu_name": "NVIDIA L40S", "cuda_device_count": 1, "allocation_id": 1860}
        roots = {"training": "artifacts/followup_seed43_cloud", "validation": "artifacts/validation_seed43_cloud"}
        amendment = {"amendment_id": "paired_seed43_cloud_validation", "authorized": True,
                     "base_protocol_sha256": analysis.sha256(self.protocol_path), "training_seed": 43,
                     "training_epochs_per_model": 30, "allowed_splits": ["train", "validation"],
                     "platform": platform, "environment": environment, "output_roots": roots,
                     "required_file_sha256": required}
        amendment_path = self.root / "cloud_amendment.json"
        write(amendment_path, amendment)
        execution = {"amendment_sha256": analysis.sha256(amendment_path),
                     "base_protocol_sha256": analysis.sha256(self.protocol_path), "runner_sha256": analysis.sha256(runner),
                     "manifest_sha256": self.protocol["manifest_sha256"], "training_sources_sha256": self.protocol["training_sources_sha256"],
                     "platform_operator_declaration": platform, "output_roots": roots,
                     "required_file_sha256_verified": required, "visible_cuda_device_count": 1,
                     "measured_gpu_name": "NVIDIA L40S", "measured_environment": environment,
                     "measured_extra_packages": {"matplotlib": "3.9.4"}}
        write(training / "cloud_execution.json", execution)
        return training, validation_root, amendment_path

    def cloud_summary(self, training, validation_root, amendment_path):
        return analysis.build_summary(self.root / "seed42", training, self.root / "val42", validation_root,
                                      self.manifest_path, self.protocol_path, cloud_amendment=amendment_path)

    def test_complete_fixture_preserves_zero_null_patient_identity_and_resources(self):
        result = self.summary()
        self.assertEqual(result["validation_patients_unique"], 6)
        self.assertEqual(result["training_seeds"], [42, 43])
        run = result["runs"]["42"]["unet2d"]
        self.assertEqual(run["patients"][0]["per_class"][5]["dice"], 0)
        self.assertIsNone(run["patients"][5]["per_class"][5]["dice"])
        self.assertAlmostEqual(run["prostate"]["dice"], (0 + 0.4 + 0.6 + 0.7 + 0.9) / 5)
        self.assertEqual(run["resources"]["train_loop_seconds"], 37.5)
        self.assertEqual(run["resources"]["successful_optimizer_steps"], 90)
        self.assertAlmostEqual(result["seed43_minus_seed42"]["unet2d"]["prostate"]["dice"], 0.02)
        self.assertIsNone(result["per_seed_patient_prostate_3d_minus_2d"]["42"][-1]["dice"])
        self.assertEqual(result["per_seed_prostate_3d_minus_2d"]["42"]["support"]["dice"]["eligible_n"], 5)
        self.assertEqual(result["analysis_source_sha256"], analysis.sha256(analysis.__file__))
        self.assertNotIn("p_value", result)

    def test_primary_difference_uses_common_patient_support_not_separate_means(self):
        relative = "val42/improved3d/evaluation.json"
        evidence = json.loads((self.root / relative).read_text())
        # V5 has no GT prostate in either model. Only 3D emits five FP voxels:
        # its Dice becomes defined zero while 2D remains undefined.
        case = evidence["cases"][-1]
        case["per_class"][0].update(target_voxels=34, predicted_voxels=29, intersection=29,
                                    union=34, dice=58 / 63, iou=29 / 34)
        case["per_class"][5].update(target_voxels=0, predicted_voxels=5, intersection=0,
                                    union=5, dice=0.0, iou=0.0)
        patient = evidence["aggregate"]["patients"][-1]
        patient["per_class"] = [{k: row[k] for k in ("class_index", "dice", "iou")} for row in case["per_class"]]
        for metric in ("dice", "iou"):
            case[f"foreground_mean_{metric}"] = patient[f"foreground_mean_{metric}"] = 0.75
            patients = evidence["aggregate"]["patients"]
            evidence["aggregate"][f"foreground_mean_{metric}"] = analysis.mean_defined(p[f"foreground_mean_{metric}"] for p in patients)
            for c in range(6):
                evidence["aggregate"]["per_class"][c][metric] = analysis.mean_defined(p["per_class"][c][metric] for p in patients)
        write(self.root / relative, evidence)
        self.change_json("seed42/improved3d/history.json", lambda h: h[28].update(validation={k: evidence[k] for k in ("cases", "aggregate")}))
        self.change_json("seed42/improved3d/run.json", lambda d: d.update(best_validation_foreground_dice=evidence["aggregate"]["foreground_mean_dice"]))
        result = self.summary()
        primary = result["per_seed_prostate_3d_minus_2d"]["42"]
        self.assertAlmostEqual(primary["dice"], 0.02)
        self.assertAlmostEqual(primary["iou"], ((0.8 / 1.2) - (0.7 / 1.3)) / 5)
        for metric in ("dice", "iou"):
            self.assertEqual(primary["support"][metric], {"eligible_n": 5,
                             "eligible_patient_ids": ["V0", "V1", "V2", "V3", "V4"], "excluded_ids": ["V5"]})
        secondary = result["secondary_prostate_difference_of_separate_aggregate_means"]["42"]
        self.assertAlmostEqual(secondary["dice"], 2.7 / 6 - 2.6 / 5)
        self.assertNotAlmostEqual(primary["dice"], secondary["dice"])

    def test_no_common_defined_patients_remains_null_without_dropping_zero(self):
        result = analysis.paired_patient_summary([{"patient": "A", "dice": None, "iou": None}])
        self.assertIsNone(result["dice"])
        self.assertEqual(result["support"]["dice"]["eligible_n"], 0)
        result = analysis.paired_patient_summary([{"patient": "A", "dice": 0.0, "iou": 0.0}])
        self.assertEqual(result["dice"], 0.0)
        self.assertEqual(result["support"]["dice"]["eligible_n"], 1)

    def test_cloud_identity_changes_interpretation_without_changing_paired_estimator(self):
        original = self.summary()
        self.assertNotIn("cloud", original)
        training, validation_root, amendment_path = self.prepare_cloud()
        result = self.cloud_summary(training, validation_root, amendment_path)
        self.assertEqual(result["per_seed_prostate_3d_minus_2d"], original["per_seed_prostate_3d_minus_2d"])
        self.assertEqual(result["cloud"]["amendment_sha256"], analysis.sha256(amendment_path))
        self.assertEqual(result["cloud"]["cloud_execution_sha256"], analysis.sha256(training / "cloud_execution.json"))
        self.assertIn("joint seed + cloud", result["study"])
        self.assertFalse(result["isolated_seed_effect_claimed"])
        self.assertFalse(any(line.startswith("Comparisons describe seed sensitivity") for line in result["limits"]))

    def test_cloud_roots_require_explicit_amendment_flag(self):
        training, validation_root, _ = self.prepare_cloud()
        with self.assertRaisesRegex(ValueError, "requires --cloud-amendment"):
            analysis.build_summary(self.root / "seed42", training, self.root / "val42", validation_root,
                                   self.manifest_path, self.protocol_path)

    def test_cloud_rejects_wrong_amendment_authorization_base_or_root(self):
        training, validation_root, amendment_path = self.prepare_cloud()
        original = amendment_path.read_text()
        for field, value in (("authorized", False), ("base_protocol_sha256", "wrong"),
                             ("output_roots", {"training": "artifacts/followup_seed43", "validation": "artifacts/validation_seed43"})):
            with self.subTest(field=field):
                data = json.loads(original)
                data[field] = value
                write(amendment_path, data)
                with self.assertRaises(ValueError): self.cloud_summary(training, validation_root, amendment_path)
        amendment_path.write_text(original)

    def test_cloud_rejects_execution_and_run_runtime_gpu_mismatches(self):
        training, validation_root, amendment_path = self.prepare_cloud()
        execution_path = training / "cloud_execution.json"
        original = execution_path.read_text()
        for field, value in (("amendment_sha256", "wrong"), ("measured_gpu_name", "NVIDIA A100"),
                             ("measured_environment", {"python_version": "wrong"})):
            with self.subTest(field=field):
                data = json.loads(original)
                data[field] = value
                write(execution_path, data)
                with self.assertRaises(ValueError): self.cloud_summary(training, validation_root, amendment_path)
        execution_path.write_text(original)
        for path in (training / "unet2d/run.json", validation_root / "improved3d/evaluation.json"):
            original = path.read_text()
            data = json.loads(original)
            data["gpu_name"] = "wrong GPU"
            write(path, data)
            with self.assertRaisesRegex(ValueError, "runtime/GPU"):
                self.cloud_summary(training, validation_root, amendment_path)
            path.write_text(original)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Matplotlib required for cloud figure rendering")
    def test_cloud_plot_renders_with_joint_change_context(self):
        training, validation_root, amendment_path = self.prepare_cloud()
        summary = self.cloud_summary(training, validation_root, amendment_path)
        output = self.root / "cloud_plot.png"
        analysis.plot_patients(summary, output)
        self.assertGreater(output.stat().st_size, 1000)

    def test_repeated_visits_do_not_count_as_extra_patients(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["records"].append({"patient": "V0", "visit": "Week1", "split": "validation"})
        evidence = validation([0, 4, 6, 7, 9, None])
        extra = validation([10, 4, 6, 7, 9, None])["cases"][0]
        extra["visit"] = "Week1"
        evidence["cases"].append(extra)
        aggregate = evidence["aggregate"]
        aggregate["num_cases"] = 7
        patient = aggregate["patients"][0]
        patient["num_cases"] = 2
        for c in (1, 5):
            patient["per_class"][c].update(dice=0.5, iou=0.5)
        patient.update(foreground_mean_dice=0.75, foreground_mean_iou=0.75)
        for metric in ("dice", "iou"):
            aggregate[f"foreground_mean_{metric}"] = analysis.mean_defined(p[f"foreground_mean_{metric}"] for p in aggregate["patients"])
            for c in range(6):
                aggregate["per_class"][c][metric] = analysis.mean_defined(p["per_class"][c][metric] for p in aggregate["patients"])
        checked = analysis.checked_metrics(evidence, manifest)["aggregate"]
        self.assertEqual(checked["num_patients"], 6)
        self.assertAlmostEqual(checked["per_class"][5]["dice"], 0.62)
        self.assertAlmostEqual(checked["foreground_mean_dice"], 5.05 / 6)

    def test_test_split_or_wrong_seed_rejected(self):
        for field, value in (("split", "test"), ("model_config", {"seed": 44})):
            with self.subTest(field=field):
                path = self.root / "val43/unet2d/evaluation.json"
                original = path.read_text()
                self.change_json("val43/unet2d/evaluation.json", lambda d: d.update({field: value}))
                with self.assertRaises(ValueError): self.summary()
                path.write_text(original)

    def test_config_change_beyond_seed_rejected(self):
        for relative in ("seed43/unet2d/run.json", "val43/unet2d/evaluation.json"):
            field = "config" if "run.json" in relative else "model_config"
            self.change_json(relative, lambda d: d[field].update(lr=0.002))
        with self.assertRaisesRegex(ValueError, "beyond seed"): self.summary()

    def test_checkpoint_sourcebook_and_manifest_identity_rejected(self):
        path = self.root / "seed43/unet2d/source_provenance.json"
        original = path.read_text()
        for field in ("checkpoint_sha256", "expected_sha256", "manifest_sha256"):
            with self.subTest(field=field):
                self.change_json(str(path.relative_to(self.root)), lambda d: d.update({field: "wrong"}))
                with self.assertRaises(ValueError): self.summary()
                path.write_text(original)

    def test_history_selection_and_epoch_count_rejected(self):
        path = self.root / "seed43/unet2d/history.json"
        original = path.read_text()
        for transform in (lambda h: h.pop(), lambda h: h[26].update(validation=validation([0, 0, 0, 0, 0, None]))):
            transform_data = json.loads(original)
            transform(transform_data)
            write(path, transform_data)
            with self.assertRaises(ValueError): self.summary()
            path.write_text(original)

    def test_standalone_valid_metrics_must_match_selected_history(self):
        self.change_json("val43/improved3d/evaluation.json", lambda d: d.update(validation([0, 6, 6, 8, 9, None])))
        with self.assertRaisesRegex(ValueError, "selected history"):
            self.summary()

    def test_explicit_test_label_inside_history_is_forbidden(self):
        self.change_json("seed43/unet2d/history.json", lambda h: h[0]["validation"].update(split="test"))
        with self.assertRaisesRegex(ValueError, "TEST"):
            self.summary()

    def test_main_refuses_existing_output_without_touching_it(self):
        args = []
        for name, value in (("seed42-root", "seed42"), ("seed43-root", "seed43"), ("seed42-val", "val42"),
                            ("seed43-val", "val43"), ("manifest", "manifest.json"), ("protocol", "followup_protocol.json")):
            args.extend(["--" + name, str(self.root / value)])
        with self.assertRaisesRegex(ValueError, "new output"):
            analysis.main(args + ["--output-dir", str(self.root)])
        self.assertFalse((self.root / "summary.json").exists())

    def test_required_file_tampering_and_refused_gate(self):
        self.change_json("followup_protocol.json", lambda d: d["diagnostic_gate"].update(approved_to_proceed=False))
        with self.assertRaises(ValueError): self.summary()
        write(self.protocol_path, self.protocol)
        (self.root / "modules.py").write_text("changed source")
        with self.assertRaisesRegex(ValueError, "Required frozen"): self.summary()

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Matplotlib required for figure rendering")
    def test_patient_plot_renders_without_turning_undefined_into_zero(self):
        output = self.root / "plot.png"
        analysis.plot_patients(self.summary(), output)
        self.assertGreater(output.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
