"""New entry points must not silently relabel the previously frozen experiment."""

from pathlib import Path
import tempfile
import unittest

from run_r6_final import preflight


class FrozenProvenanceTests(unittest.TestCase):
    def test_new_source_requires_historical_checkout_for_exact_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "split_seed42_60_20_20" / "split_samples.csv"
            manifest.parent.mkdir()
            manifest.write_text("synthetic placeholder\n")
            with self.assertRaisesRegex(RuntimeError, "0967d26"):
                preflight(Path(tmp))


if __name__ == "__main__":
    unittest.main()
