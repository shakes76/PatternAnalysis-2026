"""Acquisition probe on synthetic data: target parsing, relabelling and a CPU run."""

import contextlib
import io
import json
import unittest

import torch

from adni_data.acquisition import acquisition_info
import probe_acquisition as probe
import test_adni_splits as split_fixture


class AcquisitionProbeTests(unittest.TestCase):
    def setUp(self):
        self.fixture = split_fixture.ADNISplitTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        # Give every scan a sequence name and acquisition date, as in the real metadata.
        for number, (image_id, record) in enumerate(sorted(self.fixture.metadata.items())):
            patient = record["raw"].split("ADNI_")[2].split("_MR_")[0]
            sequence, year = ("Sag_IR-SPGR", 2018) if number % 2 else ("MPRAGE", 2013 if number % 4 else 2019)
            record["raw"] = f"ADNI_T1_3T/ADNI_{patient}_MR_{sequence}__br_raw_{year}0101120000_1_S1_I{image_id}.nii"
        self.fixture.save_metadata()
        self.fixture.prepare()
        torch.set_num_threads(1)

    def test_targets_keep_only_one_diagnosis(self):
        info = acquisition_info(self.fixture.root)
        self.assertTrue(all(scan["year"] in (2013, 2018, 2019) for scan in info.values()))
        ad = {"image_id": "1", "label": "1"}
        nc = {"image_id": "1", "label": "0"}
        info = {"1": {"year": 2018, "sequence": "Sag_IR-SPGR"}}
        self.assertEqual(probe.probe_target("era_within_ad", ad, info), 1)
        self.assertIsNone(probe.probe_target("era_within_ad", nc, info))
        self.assertEqual(probe.probe_target("vendor_within_nc", nc, info), 1)
        self.assertIsNone(probe.probe_target("vendor_within_nc", ad, info))
        self.assertIsNone(probe.probe_target("era_within_ad", {"image_id": "2", "label": "1"}, info))

    def test_probe_runs_on_validation_patients_only(self):
        output = self.fixture.base / "probe"
        args = probe.build_parser().parse_args([
            "--data-root", str(self.fixture.root), "--splits-dir", str(self.fixture.out),
            "--output", str(output), "--probe", "vendor_within_nc", "--folds", "1", "--epochs", "1",
            "--warmup-epochs", "0", "--batch-size", "8", "--workers", "0", "--threads", "1",
            "--image-height", "32", "--image-width", "32", "--device", "cpu"])
        with contextlib.redirect_stdout(io.StringIO()):
            result = probe.run(args)
        val = split_fixture.read_csv(self.fixture.out / "fold_01/val.csv")
        nc_val_scans = {row["image_id"] for row in val if row["label"] == "0"}
        scored = {row["image_id"] for row in split_fixture.read_csv(output / "scan_predictions.csv")}
        self.assertEqual(scored, nc_val_scans)
        self.assertEqual(json.loads((output / "metrics.json").read_text())["probe"], "vendor_within_nc")
        self.assertIn("auroc", result["pooled_out_of_fold"]["scan"])


if __name__ == "__main__":
    unittest.main()
