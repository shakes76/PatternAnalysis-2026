"""Exercise the public scripts and provenance after the package refactor."""

import hashlib
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]


class EntrypointTests(unittest.TestCase):
    def run_python(self, *arguments):
        """Use fresh processes so existing test imports cannot hide import errors."""
        result = subprocess.run(
            [sys.executable, "-B", *arguments], cwd=ROOT,
            capture_output=True, text=True, timeout=60, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_training_help_exposes_models_and_augmentation_settings(self):
        help_text = self.run_python("train.py", "--help")
        self.assertIn("--model", help_text)
        for name in ("cnn", "convnext_lite", "convnext_tiny"):
            self.assertRegex(help_text, rf"(?<![\w]){name}(?![\w])")
        for option in ("--augmentation", "--context-slices", "--ema-decay", "--epochs"):
            self.assertIn(option, help_text)
        for name in ("none", "light", "strong"):
            self.assertRegex(help_text, rf"(?<![\w]){name}(?![\w])")

    def test_prediction_help_keeps_preprocessing_checkpoint_controlled(self):
        help_text = self.run_python("predict.py", "--help")
        for option in ("--checkpoint", "--data-root", "--splits-dir", "--output", "--role", "--calibration-file"):
            self.assertIn(option, help_text)
        for option in ("--model", "--augmentation", "--context-slices"):
            self.assertNotIn(option, help_text)

    def test_audit_entrypoint_does_not_require_training_dependencies(self):
        """Simulate an audit-only environment with unavailable training packages."""
        script = textwrap.dedent("""\
            import importlib.abc
            import runpy
            import sys

            blocked = {"torch", "torchvision", "numpy", "matplotlib", "sklearn"}

            class BlockTrainingImports(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split(".")[0] in blocked:
                        raise RuntimeError("Audit imported a training dependency: " + fullname)
                    return None

            sys.meta_path.insert(0, BlockTrainingImports())
            import adni_data
            from adni_data import splits

            command = sys.argv[1]
            sys.argv = ["adni_splits.py", command, "--help"]
            try:
                runpy.run_path("adni_splits.py", run_name="__main__")
            except SystemExit as result:
                if result.code not in (None, 0):
                    raise
            assert not any(name.split(".")[0] in blocked for name in sys.modules)
            """)
        for command in ("audit", "prepare", "verify"):
            with self.subTest(command=command):
                help_text = self.run_python("-c", script, command)
                self.assertIn("--data-root", help_text)
                self.assertIn("--output", help_text)

    def test_code_fingerprints_cover_the_executed_packages(self):
        """Moving implementation out of entry scripts must not remove provenance."""
        from utils.artifacts import code_fingerprints

        fingerprints = code_fingerprints()
        expected = {"train.py", "predict.py", "adni_splits.py"}
        for package in ("models", "adni_data", "engine", "evaluation", "utils"):
            directory = ROOT / package
            self.assertTrue(directory.is_dir(), package)
            sources = list(directory.rglob("*.py"))
            self.assertTrue(sources, package)
            expected.update(path.relative_to(ROOT).as_posix() for path in sources)
        self.assertTrue(expected.issubset(fingerprints), sorted(expected - fingerprints.keys()))
        for relative, digest in fingerprints.items():
            with self.subTest(source=relative):
                path = Path(relative)
                self.assertFalse(path.is_absolute())
                resolved = (ROOT / path).resolve()
                self.assertTrue(resolved.is_relative_to(ROOT))
                self.assertEqual(digest, hashlib.sha256(resolved.read_bytes()).hexdigest())

    def test_dataset_module_is_the_course_facing_loader(self):
        """`import dataset` must resolve to dataset.py and expose the adni_data package API."""
        import adni_data
        import dataset

        self.assertEqual(Path(dataset.__file__).resolve(), (ROOT / "dataset.py").resolve())
        for symbol in dataset.__all__:
            self.assertIs(getattr(dataset, symbol), getattr(adni_data, symbol))


if __name__ == "__main__":
    unittest.main()
