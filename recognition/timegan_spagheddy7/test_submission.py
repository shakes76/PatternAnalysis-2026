"""Guard generated baseline contracts and the archive-free training entry points."""

import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from train import load_ar_baseline, write_json
from tune_ar import write_baseline_contract


class SubmissionChecks(unittest.TestCase):
    def setUp(self):
        self.signature = {"protocol": "anchor_only_v1", "data": {"sha256": "synthetic"}}
        self.report = {
            "comparison_signature": self.signature,
            "comparison_protocol": {"id": "anchor_only_v1"},
            "original_candidate": "original",
            "selected_candidate": "selected",
            "candidates": [{"candidate": name, "status": "ok", "score": score}
                           for name, score in (("original", 2.), ("selected", 1.), ("other", 3.))],
            "historical_notes": "A full legacy report may include additional fields.",
        }

    def test_compact_contract_roundtrip_retains_baselines_and_full_report(self):
        with tempfile.TemporaryDirectory() as folder:
            report_path = Path(folder) / "results.json"
            write_json(report_path, self.report)
            original_bytes = report_path.read_bytes()
            contract_path = write_baseline_contract(self.report, report_path)
            contract = load_ar_baseline(self.signature, contract_path)
            self.assertEqual(contract["candidates"], self.report["candidates"][:2])
            self.assertEqual(contract["comparison_signature"], self.signature)
            self.assertEqual(report_path.read_bytes(), original_bytes)
            self.assertNotIn("historical_notes", contract)

    def test_explicit_full_report_remains_supported(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "archived.json"
            write_json(path, self.report)
            self.assertEqual(load_ar_baseline(self.signature, path), self.report)

    def test_contract_mismatch_and_missing_signature_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "baseline.json"
            for value in ({"protocol": "different", "data": self.signature["data"]},
                          {"protocol": self.signature["protocol"], "data": {"sha256": "different"}}, None):
                changed = copy.deepcopy(self.report)
                changed["comparison_signature"] = value
                write_json(path, changed)
                with self.subTest(signature=value), self.assertRaisesRegex(ValueError, "contracts differ"):
                    load_ar_baseline(self.signature, path)

    def test_missing_selected_candidate_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "baseline.json"
            changed = copy.deepcopy(self.report)
            changed["candidates"] = changed["candidates"][:1]
            write_json(path, changed)
            with self.assertRaisesRegex(ValueError, "missing its original or selected"):
                load_ar_baseline(self.signature, path)

    def test_defaults_use_generated_contracts_for_both_conditioning_modes(self):
        with tempfile.TemporaryDirectory() as folder, patch("train.PROJECT_DIR", Path(folder)):
            project = Path(folder)
            for conditioned, filename in ((False, "baseline_contract.json"), (True, "anchor_conditioned_contract.json")):
                with self.subTest(conditioned=conditioned):
                    write_json(project / "outputs/ar" / filename, self.report)
                    self.assertEqual(load_ar_baseline(self.signature, anchor_conditioned=conditioned), self.report)
            self.assertFalse((project / "reports").exists())

    def test_missing_contract_explains_how_to_generate_or_override_it(self):
        with tempfile.TemporaryDirectory() as folder, patch("train.PROJECT_DIR", Path(folder)):
            with self.assertRaisesRegex(FileNotFoundError, "tune_ar.py.*--anchor-conditioned.*--baseline-report"):
                load_ar_baseline(self.signature, anchor_conditioned=True)

    def test_summary_uses_local_outputs_without_recreating_reports(self):
        from tune_timegan import collect_summary
        with tempfile.TemporaryDirectory() as folder, patch("tune_timegan.PROJECT_DIR", Path(folder)):
            summary = collect_summary()
            saved = Path(folder) / "outputs/timegan/tuning.json"
            self.assertEqual(json.loads(saved.read_text(encoding="utf-8")), summary)
            self.assertFalse((Path(folder) / "reports").exists())

    def test_training_cli_forwards_explicit_baseline(self):
        from train import main
        with patch.object(sys, "argv", ["train.py", "--model", "timegan", "--data-dir", "data", "--baseline-report", "saved.json"]), \
             patch("train.fit_timegan", return_value={"selected": {"step": 1}}) as fit, patch("builtins.print"):
            main()
        self.assertEqual(fit.call_args.kwargs["baseline_report"], Path("saved.json"))

    def test_candidate_forwards_explicit_baseline_without_running_summary_on_stop(self):
        from tune_timegan import run_candidate
        with tempfile.TemporaryDirectory() as folder, patch("tune_timegan.PROJECT_DIR", Path(folder)), \
             patch("tune_timegan.fit_timegan", return_value={"status": "incomplete"}) as fit, \
             patch("tune_timegan.collect_summary") as summary:
            run_candidate(None, "synthetic", baseline_report=Path("saved.json"))
        self.assertEqual(fit.call_args.kwargs["baseline_report"], Path("saved.json"))
        summary.assert_not_called()

    def test_conditioned_baseline_supports_relative_and_external_output_directories(self):
        import os
        from types import SimpleNamespace
        from unittest.mock import Mock
        import numpy as np
        import torch
        from tune_ar import run_conditioned_baseline

        windows = np.zeros((3, 32, 40))
        rows = np.zeros((3, 40))
        training = SimpleNamespace(book=np.ones((5, 40)), starts=np.array([4]))
        validation = SimpleNamespace(raw_offset=0)
        scaler = SimpleNamespace(to_dict=lambda: {})
        model = SimpleNamespace(state_dict=lambda: {}, residuals=torch.zeros(3, 40), spectral_radius=0.)
        fitter = Mock()
        fitter.from_pairs.return_value.fit.return_value = model
        # Exercise real report/checkpoint persistence, but never fit or score a model.
        replacements = dict(
            prepare_datasets=lambda *args: ({"train": training, "validation": validation}, scaler, {"files": {}}),
            fit_anchor_context=lambda *args: {}, comparison_protocol=lambda **kwargs: {"id": "synthetic"},
            validation_windows=lambda *args: (np.arange(3), np.ones((3, 20)), windows),
            metric_specification=lambda *args: {}, ValidationEvaluator=lambda *args: None,
            conditional_training_pairs=lambda *args: (rows, rows, rows),
            sample_training_prior=lambda *args: (torch.zeros(3, 4, 40), None),
            anchor_context=lambda *args: rows, VARFit=fitter,
            evaluate_candidate=lambda model, base, *args: (
                dict(base, invalid_rate_mean=0., kl_mid_return_mean=0.), [], None, None, windows),
            make_checkpoint=lambda *args: {"state_dict": {}},
            VectorAR=lambda **kwargs: SimpleNamespace(generate=lambda *args: torch.zeros(3, 32, 40)),
            decode_batch=lambda *args: windows, comparison_signature=lambda *args: self.signature,
            source_fingerprint=lambda: "synthetic")
        with tempfile.TemporaryDirectory() as folder:
            project = Path(folder) / "project"
            project.mkdir()
            previous_directory = Path.cwd()
            try:
                os.chdir(project)
                for name, output in (("relative", Path("outputs/ar/relative")),
                                     ("absolute_internal", project / "outputs/ar/absolute"),
                                     ("absolute_external", Path(folder) / "external")):
                    with self.subTest(output=output), patch("tune_ar.PROJECT_DIR", project), \
                         patch.multiple("tune_ar", **replacements), patch("builtins.print"):
                        report = run_conditioned_baseline(None, output_dir=output,
                                                         report_path=project / "reports_for_test" / name / "results.json", threads=1)
                    for candidate in report["candidates"]:
                        reference = Path(candidate["checkpoint"])
                        expected = output.resolve() / (candidate["anchor_conditioning"] + ".pt")
                        resolved = reference if reference.is_absolute() else project / reference
                        self.assertEqual(resolved, expected)
                        self.assertTrue(resolved.is_file())
                        self.assertEqual(reference.is_absolute(), name == "absolute_external")
            finally:
                os.chdir(previous_directory)

    def test_conditioning_cli_forwards_explicit_full_baseline_report(self):
        from tune_timegan import main
        with patch.object(sys, "argv", ["tune_timegan.py", "--conditioning-report", "--data-dir", "data",
                                      "--baseline-report", "full_conditioned.json"]), \
             patch("tune_timegan.analyse_conditioning", return_value={"conclusion": "synthetic"}) as analyse, \
             patch("tune_timegan.run_candidate") as fit, patch("builtins.print"):
            main()
        self.assertEqual(analyse.call_args.kwargs["baseline_path"], Path("full_conditioned.json"))
        fit.assert_not_called()

    def test_conditioning_diagnostics_reject_compact_contract_before_loading_data_or_checkpoints(self):
        from timegan_report import analyse_conditioning
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report_path = root / "anchor_conditioned.json"
            write_json(report_path, self.report)
            contract = write_baseline_contract(self.report, report_path, anchor_conditioned=True)
            with patch("timegan_report.prepare_datasets") as prepare, patch("timegan_report.load_timegan") as load:
                with self.assertRaisesRegex(ValueError, "Conditioning diagnostics require a full conditioned AR report"):
                    analyse_conditioning(None, root / "missing_matched", root / "missing_zero", root / "diagnostics",
                                         baseline_path=contract)
                prepare.assert_not_called()
                load.assert_not_called()
            self.assertFalse((root / "diagnostics").exists())

    def test_notebook_refresh_requires_existing_explicit_path_before_execution(self):
        from tune_timegan import main
        for extra in ([], ["--notebook", "missing-archival-notebook.ipynb"]):
            with self.subTest(extra=extra), patch.object(sys, "argv", ["tune_timegan.py", "--refresh-notebook", *extra]), \
                 patch("tune_timegan.refresh_notebook") as refresh, patch("tune_timegan.run_candidate") as fit, \
                 patch("sys.stderr", io.StringIO()) as errors, self.assertRaises(SystemExit) as failure:
                main()
            self.assertEqual(failure.exception.code, 2)
            self.assertIn("restored archival notebook", errors.getvalue())
            refresh.assert_not_called()
            fit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
