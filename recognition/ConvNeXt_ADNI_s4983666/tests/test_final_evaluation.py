"""Fold-model ensembles and acquisition-period breakdown in the final evaluation."""

import contextlib
import io
import unittest

import torch

from engine import prediction as predict
from engine import training as train
from evaluation.reporting import acquisition_breakdown
import test_adni_splits as split_fixture


class FinalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = split_fixture.ADNISplitTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        # Dated metadata paths so the acquisition-period breakdown can be computed.
        for number, (image_id, record) in enumerate(sorted(self.fixture.metadata.items())):
            patient = record["raw"].split("ADNI_")[2].split("_MR_")[0]
            year = 2014 if number % 3 == 0 else 2018
            record["raw"] = f"ADNI_T1_3T/ADNI_{patient}_MR_MPRAGE__br_raw_{year}0101120000_1_S1_I{image_id}.nii"
        self.fixture.save_metadata()
        self.fixture.prepare()
        torch.set_num_threads(1)
        self.runs = []
        for fold in (1, 2):
            output = self.fixture.base / f"fold{fold}"
            args = train.build_parser().parse_args([
                "--data-root", str(self.fixture.root), "--splits-dir", str(self.fixture.out),
                "--output", str(output), "--model", "cnn", "--fold", str(fold), "--epochs", "1",
                "--warmup-epochs", "0", "--batch-size", "8", "--workers", "0", "--threads", "1",
                "--image-height", "32", "--image-width", "32", "--device", "cpu", "--skip-inference-profile"])
            with contextlib.redirect_stdout(io.StringIO()):
                train.run(args)
            self.runs.append(output / "final.pt")

    def predict(self, role: str, name: str, checkpoints: list, *extra: str) -> dict:
        args = predict.build_parser().parse_args([
            "--checkpoint", *map(str, checkpoints), "--data-root", str(self.fixture.root),
            "--splits-dir", str(self.fixture.out), "--output", str(self.fixture.base / name), "--role", role,
            "--batch-size", "8", "--workers", "0", "--threads", "1", "--device", "cpu", "--skip-inference-profile",
            *extra])
        with contextlib.redirect_stdout(io.StringIO()):
            return predict.run(args)

    def test_ensemble_averages_logits_and_is_refused_on_development_patients(self):
        with self.assertRaisesRegex(ValueError, "development patients"):
            self.predict("val", "ensemble_val", self.runs)
        for index, run in enumerate(self.runs):
            self.predict("calibration", f"single{index}", [run])
        ensemble = self.predict("calibration", "ensemble_cal", self.runs)
        self.assertEqual(ensemble["ensemble_size"], 2)
        self.assertEqual([m["fold"] for m in ensemble["checkpoints"]], [1, 2])
        members = [split_fixture.read_csv(self.fixture.base / f"single{i}/slice_predictions.csv") for i in range(2)]
        averaged = split_fixture.read_csv(self.fixture.base / "ensemble_cal/slice_predictions.csv")
        for a, b, mean in zip(*members, averaged):
            self.assertAlmostEqual((float(a["logit"]) + float(b["logit"])) / 2, float(mean["logit"]), places=5)
        # The calibration binds to the ensemble: a single member cannot reuse it.
        calibration = self.fixture.base / "ensemble_cal/calibration.json"
        tested = self.predict("test", "ensemble_test", self.runs, "--calibration-file", str(calibration))
        self.assertEqual(tested["role"], "test")
        with self.assertRaisesRegex(ValueError, "different checkpoint"):
            self.predict("test", "member_test", self.runs[:1], "--calibration-file", str(calibration))

    def test_metrics_are_split_by_acquisition_period(self):
        result = self.predict("val", "val_single", self.runs[:1])
        periods = result["by_acquisition_period"]
        self.assertTrue(set(periods) <= {"before_2017", "from_2017"})
        total = sum(sum(entry["scan_counts"].values()) for entry in periods.values())
        self.assertEqual(total, result["metrics"]["scan"]["n_samples"])

    def test_breakdown_counts_referrals_and_missed_ad(self):
        scans = [{"patient_id": "a", "image_id": "1", "label": 1, "probability": 0.2},
                 {"patient_id": "b", "image_id": "2", "label": 0, "probability": 0.1},
                 {"patient_id": "c", "image_id": "3", "label": 1, "probability": 0.9}]
        patients = [{"patient_id": "a", "label": 1, "probability": 0.2, "decision": "NC"},
                    {"patient_id": "b", "label": 0, "probability": 0.1, "decision": "NC"},
                    {"patient_id": "c", "label": 1, "probability": 0.9, "decision": "REFER"}]
        result = acquisition_breakdown(scans, patients, {"1": 2018, "2": 2018, "3": 2013}, 2017)
        self.assertEqual(result["from_2017"]["referral"], {"patients": 2, "referred": 0, "automated_accuracy": 0.5,
                                                          "missed_ad_automated": 1})
        self.assertEqual(result["before_2017"]["referral"]["referred"], 1)
        self.assertIsNone(acquisition_breakdown(scans, patients, {}, 2017))


if __name__ == "__main__":
    unittest.main()
