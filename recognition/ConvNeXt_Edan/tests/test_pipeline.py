"""CPU-only unit and integration tests using synthetic data."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dataset import (
    ADNIDataset,
    create_patient_splits,
    extract_patient_id,
)
from predict import (
    aggregate_by_patient,
    calculate_metrics,
    collect_predictions,
    load_evaluation_data,
    resolve_patient_threshold,
    save_predictions,
)
from threshold_analysis import (
    analyse_model,
    calculate_threshold_metrics,
    load_predictions,
)


class TestDatasetSplitting(unittest.TestCase):

    def test_patient_id_extraction(self):
        record = {"raw": "/synthetic/ADNI_002_S_0413/image.nii"}
        self.assertEqual(extract_patient_id(record), "002_S_0413")

        with self.assertRaises(ValueError):
            extract_patient_id({"raw": "invalid_path"})

    def test_patient_disjoint_and_reproducible(self):
        records = []
        for patient in range(40):
            label = patient % 2
            for slice_number in range(2):
                records.append({
                    "path": f"synthetic_{patient}_{slice_number}.jpeg",
                    "patient_id": f"P{patient:03d}",
                    "scan_id": f"S{patient:03d}",
                    "label": label,
                })

        first = create_patient_splits(records, seed=42)
        second = create_patient_splits(records, seed=42)
        self.assertEqual(first, second)

        patient_sets = {
            split: {r["patient_id"] for r in rows}
            for split, rows in first.items()
        }

        self.assertEqual(
            {split: len(ids) for split, ids in patient_sets.items()},
            {"train": 28, "val": 6, "test": 6},
        )

        self.assertTrue(
            patient_sets["train"].isdisjoint(patient_sets["val"])
        )
        self.assertTrue(
            patient_sets["train"].isdisjoint(patient_sets["test"])
        )
        self.assertTrue(
            patient_sets["val"].isdisjoint(patient_sets["test"])
        )
        self.assertEqual(sum(map(len, first.values())), 80)

    def test_conflicting_patient_labels_rejected(self):
        records = [
            {"patient_id": "P001", "label": 0},
            {"patient_id": "P001", "label": 1},
        ]

        with self.assertRaises(ValueError):
            create_patient_splits(records)


class TestPatientEvaluation(unittest.TestCase):

    def test_patient_probability_aggregation(self):
        results = {
            "labels": np.array([0, 0, 1, 1]),
            "probabilities": np.array([0.1, 0.3, 0.7, 0.9]),
            "patient_ids": ["P001", "P001", "P002", "P002"],
        }

        aggregated = aggregate_by_patient(results)

        np.testing.assert_array_equal(
            aggregated["labels"], [0, 1]
        )
        np.testing.assert_allclose(
            aggregated["probabilities"], [0.2, 0.8]
        )

    def test_inconsistent_patient_labels_rejected(self):
        results = {
            "labels": np.array([0, 1]),
            "probabilities": np.array([0.2, 0.8]),
            "patient_ids": ["P001", "P001"],
        }

        with self.assertRaises(ValueError):
            aggregate_by_patient(results)

    def test_classification_metrics(self):
        results = {
            "labels": np.array([0, 0, 1, 1]),
            "probabilities": np.array([0.1, 0.8, 0.9, 0.3]),
        }

        metrics = calculate_metrics(results, threshold=0.5)

        self.assertEqual(
            [metrics[k] for k in ("tn", "fp", "fn", "tp")],
            [1, 1, 1, 1],
        )
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["sensitivity"], 0.5)
        self.assertAlmostEqual(metrics["specificity"], 0.5)


class TestThresholdSelection(unittest.TestCase):

    def test_validation_does_not_require_threshold_file(self):
        threshold, source = resolve_patient_threshold(
            "convnext", "val", "nonexistent.json"
        )
        self.assertEqual(threshold, 0.5)
        self.assertEqual(source, "default_preselection")

    def test_test_split_requires_validation_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "thresholds.json"
            config = {
                "convnext": {
                    "threshold": 0.64,
                    "selection_split": "validation",
                    "selection_metric": "patient_balanced_accuracy",
                }
            }
            path.write_text(json.dumps(config))

            threshold, source = resolve_patient_threshold(
                "convnext", "test", path
            )
            self.assertAlmostEqual(threshold, 0.64)
            self.assertEqual(source, "validation")

            config["convnext"]["selection_split"] = "test"
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                resolve_patient_threshold("convnext", "test", path)

    def test_threshold_metrics(self):
        labels = np.array([0, 0, 1, 1])
        probabilities = np.array([0.1, 0.4, 0.6, 0.9])

        metrics = calculate_threshold_metrics(
            labels, probabilities, 0.5
        )
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)

    def test_validation_threshold_tie_break(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "convnext" / "val"
            input_dir.mkdir(parents=True)

            csv_path = input_dir / "patient_predictions.csv"
            csv_path.write_text(
                "patient_id,true_label,probability_ad\n"
                "P001,0,0.10\n"
                "P002,0,0.35\n"
                "P003,1,0.65\n"
                "P004,1,0.90\n"
            )

            analysis = analyse_model(
                "convnext",
                root,
                np.array([0.4, 0.5, 0.6]),
            )

            self.assertAlmostEqual(
                analysis["selected"]["threshold"], 0.5
            )
            self.assertEqual(
                analysis["selected"]["balanced_accuracy"], 1.0
            )


class TestSyntheticIntegration(unittest.TestCase):

    def test_cpu_image_to_patient_prediction_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            records = []
            for index, (pixel, label) in enumerate(
                [(30, 0), (220, 1)]
            ):
                path = root / f"synthetic_{index}.jpeg"
                Image.fromarray(
                    np.full((32, 32), pixel, dtype=np.uint8)
                ).save(path)

                records.append({
                    "path": str(path),
                    "patient_id": f"P{index:03d}",
                    "scan_id": f"S{index:03d}",
                    "label": label,
                })

            dataset = ADNIDataset(
                records, image_size=32, training=False
            )
            image, label = dataset[0]
            self.assertEqual(tuple(image.shape), (3, 32, 32))
            self.assertEqual(label.dtype, torch.long)

            splits = {
                "train": records,
                "val": records,
                "test": records,
            }
            split_file = root / "synthetic_splits.json"
            split_file.write_text(json.dumps(splits))

            loader = load_evaluation_data(
                split_file, "val",
                image_size=32,
                batch_size=2,
                workers=0,
            )

            class TinySyntheticModel(torch.nn.Module):
                def forward(self, inputs):
                    score = inputs.mean(dim=(1, 2, 3))
                    return torch.stack((-score, score), dim=1)

            model = TinySyntheticModel().eval()
            results = collect_predictions(
                model, loader, torch.device("cpu")
            )

            self.assertEqual(len(results["labels"]), 2)
            self.assertTrue(
                np.isfinite(results["probabilities"]).all()
            )

            output_dir = root / "predictions"
            save_predictions(
                results, output_dir, patient_threshold=0.5
            )

            predictions = load_predictions(
                output_dir / "patient_predictions.csv"
            )
            self.assertEqual(len(predictions), 2)


if __name__ == "__main__":
    unittest.main()
