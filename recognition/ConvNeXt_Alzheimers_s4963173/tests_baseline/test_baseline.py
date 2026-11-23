"""Synthetic-only tests. Never reads the user's ADNI files or uses the GPU.

Run from the project after installing Step 7:
python -m unittest discover -s tests_baseline -p test_baseline.py -v
"""
from __future__ import annotations

import ast
import csv
import dataclasses
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import (ADNISliceDataset, COLUMNS, DataError, FINAL_STATUS, SliceRecord,
                     counts_for_records, load_finalized_splits, preprocessing_spec)
from modules import BaselineCNN, parameter_count
from training_utils import (classification_loss, aggregate_predictions, binary_metrics, checked_bundle, choose_device,
                            inference_benchmark, load_checkpoint, new_private_output,
                            save_checkpoint, seed_everything, select_tiny_training_indices,
                            selection_key, sha256_file, source_hashes)
from check_baseline import run_smoke
from train import TrainConfig, majority_reference, run_training

torch.set_num_threads(1)


def record(subject: str, scan: str, index: int, label: int, split: str = "validation") -> SliceRecord:
    name = "AD" if label else "NC"
    return SliceRecord(f"AD_NC/train/{name}/{scan}_{index}.jpeg", subject, scan, index,
                       "train", name, 2 * label, label, split)


class Fixture:
    """120 invented JPEGs: six subjects, one scan per subject, twenty slices each."""
    def __init__(self, base: Path) -> None:
        self.root, self.splits = base / "data", base / "splits"
        self.root.mkdir()
        self.splits.mkdir()
        self.summary = {
            "format_version": 1, "status": FINAL_STATUS,
            "mapping_structural_checks_passed": True,
            "patient_identifier_semantics_confirmed_by_user": True,
            "tutor_split_change_approved_as_reported_by_user": True,
            "zero_group_overlap_verified": True, "splits_finalized": True,
            "new_pairwise_candidate_subject_overlap": 0,
            "model_label_map": {"NC": 0, "AD": 1},
            "metadata_to_model_labels": {"0": 0, "2": 1},
            "evidence": {"images": 120, "scan_prefixes": 6, "candidate_subjects": 6},
            "manifest_sha256": {}, "proposed_or_final_counts": {}}
        self.rows: dict[str, list[SliceRecord]] = {}
        for split_index, split in enumerate(("train", "validation", "test")):
            rows = []
            for label in (0, 1):
                subject = f"SYNTHETIC_{split_index}_{label}"
                scan = str(100000 + split_index * 2 + label)
                for index in range(80, 100):
                    row = record(subject, scan, index, label, split)
                    rows.append(row)
                    path = self.root / row.image_path
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pixels = np.full((240, 256), 30 + 170 * label, dtype=np.uint8)
                    pixels[40:80, 40:100] = 90 + index
                    Image.fromarray(pixels).save(path)
            self.rows[split] = rows
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(dataclasses.asdict(r) for r in rows)
            raw = stream.getvalue().encode()
            (self.splits / f"{split}.csv").write_bytes(raw)
            self.summary["manifest_sha256"][f"{split}.csv"] = hashlib.sha256(raw).hexdigest()
            self.summary["proposed_or_final_counts"][split] = counts_for_records(tuple(rows))
        (self.splits / "split_summary.json").write_text(json.dumps(self.summary))
        self.pipeline = base / "pipeline.json"
        self.pipeline_value = {
            "data_pipeline_check_passed": True,
            "baseline_training_can_begin_after_visual_review": True,
            "manifest_sha256": self.summary["manifest_sha256"],
            "preprocessing": preprocessing_spec(0.0),
            "source_code_sha256": {"dataset.py": source_hashes()["dataset.py"]}}
        self.write_pipeline()

    def write_pipeline(self) -> None:
        self.pipeline.write_text(json.dumps(self.pipeline_value))

    def bundle(self):
        return load_finalized_splits(self.root, self.splits)


class ModelTests(unittest.TestCase):
    def test_expected_output_shape(self):
        model = BaselineCNN().eval()
        self.assertEqual(model(torch.zeros(3, 1, 224, 224)).shape, (3, 2))

    def test_single_image_batch_supported(self):
        self.assertEqual(BaselineCNN()(torch.zeros(1, 1, 224, 224)).shape, (1, 2))

    def test_wrong_shape_rejected(self):
        for shape in ((1, 3, 224, 224), (1, 1, 240, 256), (1, 224, 224)):
            with self.assertRaises(ValueError):
                BaselineCNN()(torch.zeros(shape))

    def test_invalid_dropout_rejected(self):
        for value in (-0.1, 1.0):
            with self.assertRaises(ValueError):
                BaselineCNN(value)

    def test_finite_gradients_and_update(self):
        model = BaselineCNN()
        optimizer = torch.optim.AdamW(model.parameters())
        before = next(model.parameters()).detach().clone()
        loss = torch.nn.functional.cross_entropy(model(torch.randn(2, 1, 224, 224)), torch.tensor([0, 1]))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(before, next(model.parameters())))

    def test_eval_mode_repeatable(self):
        model = BaselineCNN().eval()
        x = torch.randn(2, 1, 224, 224)
        self.assertTrue(torch.equal(model(x), model(x)))

    def test_seed_reproduces_initial_weights(self):
        seed_everything(42)
        a = BaselineCNN()
        seed_everything(42)
        b = BaselineCNN()
        self.assertTrue(all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict()))

    def test_parameter_count(self):
        model = BaselineCNN()
        self.assertEqual(parameter_count(model), sum(p.numel() for p in model.parameters()))
        self.assertLess(parameter_count(model), 1_000_000)

    def test_fixed_pooling_does_not_use_cuda_adaptive_backward(self):
        model = BaselineCNN()
        self.assertFalse(any(isinstance(m, torch.nn.AdaptiveAvgPool2d) for m in model.modules()))
        self.assertTrue(any(isinstance(m, torch.nn.AvgPool2d) for m in model.modules()))

    def test_cross_entropy_value_and_gradient_match(self):
        logits = torch.tensor([[2., -1.], [-.5, 1.], [3., 1.]], requires_grad=True)
        labels = torch.tensor([0, 1, 1])
        standard = torch.nn.functional.cross_entropy(logits, labels)
        custom = classification_loss(logits, labels)
        self.assertTrue(torch.allclose(standard, custom))
        a = torch.autograd.grad(standard, logits, retain_graph=True)[0]
        b = torch.autograd.grad(custom, logits)[0]
        self.assertTrue(torch.allclose(a, b))

    def test_cross_entropy_shape_rejected(self):
        with self.assertRaises(ValueError):
            classification_loss(torch.zeros(3, 4), torch.zeros(3, dtype=torch.long))

    def test_model_imports_only_torch_and_future(self):
        source = Path(sys.modules["modules"].__file__).read_text()
        tree = ast.parse(source)
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.append((node.module or "").split(".")[0])
        self.assertLessEqual(set(modules), {"torch", "__future__"})


class MetricTests(unittest.TestCase):
    def test_perfect_predictions(self):
        result = binary_metrics([0, 0, 1, 1], [.1, .2, .8, .9])
        self.assertEqual(result["accuracy"], 1)
        self.assertEqual(result["macro_f1"], 1)
        self.assertEqual(result["auroc"], 1)
        self.assertEqual(result["confusion_matrix_rows_true_columns_predicted_NC_AD"], [[2, 0], [0, 2]])

    def test_majority_predictions_not_good_macro_f1(self):
        result = binary_metrics([0, 0, 1], [.1, .1, .1])
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertEqual(result["per_class"]["AD"]["recall"], 0)
        self.assertAlmostEqual(result["macro_f1"], .4)
        self.assertEqual(result["auroc"], .5)

    def test_threshold_tie_is_AD(self):
        result = binary_metrics([0, 1], [.5, .5])
        self.assertEqual(result["confusion_matrix_rows_true_columns_predicted_NC_AD"], [[0, 1], [0, 1]])

    def test_invalid_probabilities_rejected(self):
        for probs in ([float("nan"), .4], [1.1, .2], [-.01, .3]):
            with self.assertRaises(DataError):
                binary_metrics([0, 1], probs)

    def test_invalid_labels_rejected(self):
        with self.assertRaises(DataError):
            binary_metrics([0, 2], [.1, .9])

    def test_metrics_length_mismatch(self):
        with self.assertRaises(DataError):
            binary_metrics([0, 1], [.1])

    def test_patient_mean_weights_scans_equally(self):
        # P1's first scan has three slices; its second has one. Mean scans=.5,
        # NOT the incorrect mean of all four slice probabilities (.25).
        rows = [record("P1", "1", i, 0) for i in range(3)]
        rows += [record("P1", "2", 0, 0), record("P2", "3", 0, 1)]
        result, private = aggregate_predictions(rows, [0, 0, 0, 1, .9])
        self.assertAlmostEqual(private["patients"][0]["p_ad"], .5)
        self.assertEqual(result["patient"]["n"], 2)
        self.assertEqual(result["scan"]["n"], 3)

    def test_test_partition_rejected_by_aggregation(self):
        with self.assertRaisesRegex(DataError, "test partition"):
            aggregate_predictions([record("P", "1", 0, 0, "test")], [.1])

    def test_conflicting_scan_subjects_rejected(self):
        with self.assertRaises(DataError):
            aggregate_predictions([record("P", "1", 0, 0), record("Q", "1", 1, 0)], [.1, .2])

    def test_conflicting_patient_labels_rejected(self):
        with self.assertRaises(DataError):
            aggregate_predictions([record("P", "1", 0, 0), record("P", "2", 0, 1)], [.1, .8])

    def test_selection_prioritises_macro_f1(self):
        a = {"patient": {"macro_f1": .8, "log_loss": .5}}
        b = {"patient": {"macro_f1": .7, "log_loss": .1}}
        self.assertGreater(selection_key(a), selection_key(b))

    def test_selection_tiebreaks_by_logloss(self):
        a = {"patient": {"macro_f1": .8, "log_loss": .3}}
        b = {"patient": {"macro_f1": .8, "log_loss": .5}}
        self.assertGreater(selection_key(a), selection_key(b))


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.f = Fixture(self.base)

    def test_passed_pipeline_accepted(self):
        bundle = checked_bundle(self.f.root, self.f.splits, self.f.pipeline)
        self.assertEqual(len(bundle.records["train"]), 40)

    def test_failed_pipeline_rejected(self):
        self.f.pipeline_value["data_pipeline_check_passed"] = False
        self.f.write_pipeline()
        with self.assertRaises(DataError):
            checked_bundle(self.f.root, self.f.splits, self.f.pipeline)

    def test_changed_dataset_code_rejected(self):
        self.f.pipeline_value["source_code_sha256"]["dataset.py"] = "bad"
        self.f.write_pipeline()
        with self.assertRaisesRegex(DataError, "dataset.py changed"):
            checked_bundle(self.f.root, self.f.splits, self.f.pipeline)

    def test_changed_preprocessing_rejected(self):
        self.f.pipeline_value["preprocessing"]["input_channels"] = 3
        self.f.write_pipeline()
        with self.assertRaises(DataError):
            checked_bundle(self.f.root, self.f.splits, self.f.pipeline)

    def test_changed_manifest_report_rejected(self):
        self.f.pipeline_value["manifest_sha256"] = {}
        self.f.write_pipeline()
        with self.assertRaises(DataError):
            checked_bundle(self.f.root, self.f.splits, self.f.pipeline)

    def test_output_reuse_rejected(self):
        out = self.base / "run"
        new_private_output(out, self.f.root, self.f.splits)
        with self.assertRaisesRegex(DataError, "already exists"):
            new_private_output(out, self.f.root, self.f.splits)

    def test_output_under_data_rejected(self):
        with self.assertRaises(DataError):
            new_private_output(self.f.root / "run", self.f.root, self.f.splits)

    def test_output_under_git_repo_rejected(self):
        repo = self.base / "repo"
        (repo / ".git").mkdir(parents=True)
        with self.assertRaises(DataError):
            new_private_output(repo / "run", self.f.root, self.f.splits)

    def test_checkpoint_roundtrip_and_hash_validation(self):
        model = BaselineCNN().eval()
        path = self.base / "checkpoint.pt"
        hashes = self.f.bundle().manifest_hashes
        save_checkpoint(path, model, epoch=3, manifest_hashes=hashes, config={})
        restored, checkpoint = load_checkpoint(path, torch.device("cpu"), hashes)
        x = torch.zeros(2, 1, 224, 224)
        self.assertTrue(torch.equal(model(x), restored(x)))
        self.assertEqual(checkpoint["epoch"], 3)
        with self.assertRaises(DataError):
            load_checkpoint(path, torch.device("cpu"), {"bad": "hash"})

    def test_tiny_subset_only_train_distinct_patients(self):
        ds = ADNISliceDataset(self.f.bundle(), "train")
        chosen = select_tiny_training_indices(ds)
        self.assertEqual(len(chosen), 2)
        self.assertEqual(len({ds.records[i].subject_id for i in chosen}), 2)
        self.assertEqual(chosen, select_tiny_training_indices(ds))
        with self.assertRaises(DataError):
            select_tiny_training_indices(ADNISliceDataset(self.f.bundle(), "validation"))

    def test_majority_selected_from_training_patients(self):
        result = majority_reference(self.f.rows["train"], self.f.rows["validation"])
        self.assertEqual(result["predicted_class"], "NC")
        self.assertEqual(result["validation_patient_metrics"]["accuracy"], .5)

    def test_cuda_not_silently_replaced_by_cpu(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(DataError, "unavailable"):
                choose_device("cuda")

    def test_config_rejects_invalid_values(self):
        for config in (TrainConfig(batch_size=0), TrainConfig(epochs=0), TrainConfig(patience=0),
                       TrainConfig(num_workers=-1), TrainConfig(learning_rate=0)):
            with self.assertRaises(DataError):
                config.validate()

    def test_model_only_timing_positive(self):
        result = inference_benchmark(BaselineCNN().eval(), torch.zeros(1, 1, 224, 224),
                                     torch.device("cpu"), repeats=2)
        self.assertGreater(result["median_ms_per_slice"], 0)
        self.assertEqual(result["batch_size"], 1)

    def test_no_test_image_access_during_cpu_smoke_and_training(self):
        # Deleting every final-test image makes any accidental test decoding fail.
        # The small test manifest remains readable for split-integrity checks.
        for row in self.f.rows["test"]:
            (self.f.root / row.image_path).unlink()
        smoke_out = self.base / "smoke"
        smoke = run_smoke(self.f.root, self.f.splits, self.f.pipeline, smoke_out,
                          device_name="cpu", batch_size=16, max_updates=100)
        self.assertTrue(smoke["smoke_check_passed"])
        out = self.base / "training"
        result = run_training(self.f.root, self.f.splits, self.f.pipeline,
                              smoke_out / "baseline_smoke_summary.json", out,
                              TrainConfig(device="cpu", epochs=2, batch_size=16))
        self.assertTrue(result["baseline_training_completed"])
        self.assertEqual(result["final_test_images_decoded"], 0)
        self.assertEqual(result["selected_validation_metrics"]["patient"]["n"], 2)
        self.assertTrue((out / "PRIVATE_best_baseline.pt").is_file())
        self.assertTrue((out / "PRIVATE_validation_predictions.json").is_file())
        self.assertTrue((out / "loss_curve.png").is_file())
        self.assertTrue((out / "README_baseline_results.md").is_file())
        with (out / "history.csv").open() as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 2)
        summary_text = (out / "baseline_summary.json").read_text()
        self.assertNotIn("SYNTHETIC_", summary_text)
        self.assertNotIn(str(self.f.root), summary_text)

    def test_failed_smoke_prevents_training_and_records_failure(self):
        smoke = self.base / "failed_smoke.json"
        smoke.write_text(json.dumps({"smoke_check_passed": False}))
        out = self.base / "failed_training"
        with self.assertRaises(DataError):
            run_training(self.f.root, self.f.splits, self.f.pipeline, smoke, out,
                          TrainConfig(device="cpu", epochs=1))
        result = json.loads((out / "baseline_summary.json").read_text())
        self.assertEqual(result["status"], "FAILED")
        self.assertFalse(result["baseline_training_completed"])


if __name__ == "__main__":
    unittest.main()
