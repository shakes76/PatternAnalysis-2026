"""Ensure a login-session inventory stays bounded and leaves source data alone."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inspect_rangpur import inventory


class InventoryTests(unittest.TestCase):
    def test_total_and_per_directory_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for number in range(15):
                (root / f"sample_{number}.txt").write_text("unchanged", encoding="utf-8")
            self.assertEqual(inventory(root, max_entries=3)["entries_sampled"], 3)
            self.assertEqual(inventory(root, per_directory=2)["entries_sampled"], 2)
            self.assertTrue(all(path.read_text(encoding="utf-8") == "unchanged" for path in root.iterdir()))

    def test_depth_limit_never_opens_nested_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "first" / "second").mkdir(parents=True)
            (root / "first" / "second" / "not_examined.txt").write_text("private", encoding="utf-8")
            report = inventory(root, max_depth=2)
            self.assertEqual(report["directories_opened"], 2)
            self.assertNotIn("first/second/not_examined.txt", [entry["path"] for entry in report["entries"]])
            self.assertFalse(report["training_ready"])
            self.assertIn("unknown", report["label_meanings"])

    def test_symlink_directory_is_not_followed(self):
        # Mock only the directory entry so this check also works on Windows
        # accounts that cannot create real symlinks.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            class LinkedEntry:
                path = str(root / "linked_directory")
                def is_symlink(self):
                    return True
                def is_dir(self, **kwargs):
                    raise AssertionError("A symlink must never be followed")
            class Entries:
                def __enter__(self):
                    return iter([LinkedEntry()])
                def __exit__(self, *args):
                    pass
            with patch("inspect_rangpur.os.scandir", return_value=Entries()) as scan:
                report = inventory(root)
            self.assertEqual(scan.call_count, 1)
            self.assertEqual(report["entries"][0]["kind"], "symlink_not_followed")

    def test_cli_refuses_to_write_into_source_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = Path(__file__).resolve().parents[1] / "inspect_rangpur.py"
            result = subprocess.run(
                [sys.executable, str(script), "--data-root", str(root), "--output", str(root / "report.json")],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("outside the read-only source", result.stderr)
            self.assertFalse((root / "report.json").exists())


if __name__ == "__main__":
    unittest.main()
