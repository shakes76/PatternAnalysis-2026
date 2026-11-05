"""Regression checks for leakage prevention and invertible preprocessing."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataset import PRICE_COLUMNS, VOLUME_COLUMNS, decode_features, prepare_datasets


class DatasetChecks(unittest.TestCase):
    def setUp(self):
        temp_root = Path(__file__).resolve().parent / "outputs"
        temp_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temp_root)
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        rows = 200
        times = 34200 + np.arange(rows) * (57600 - 34200) / rows
        times[139] = times[140]  # Equal-time events straddle a naive 70% row split.
        messages = np.column_stack((times, np.ones(rows), np.arange(rows), np.ones(rows), np.full(rows, 1000000), np.ones(rows)))
        book = np.zeros((rows, 40), dtype=np.int64)
        for level in range(10):
            mid = 1000000 + np.arange(rows) * 10
            book[:, 4 * level] = mid + 100 + level * 100
            book[:, 4 * level + 2] = mid - 100 - level * 100
            book[:, 4 * level + 1] = 100 + level + np.arange(rows) % 7
            book[:, 4 * level + 3] = 200 + level + np.arange(rows) % 5
        book[139:, VOLUME_COLUMNS] *= 1000  # Unseen future range must not fit the scaler.
        self.book_path = self.directory / "TEST_2026-01-01_34200000_57600000_orderbook_10.csv"
        self.message_path = self.directory / "TEST_2026-01-01_34200000_57600000_message_10.csv"
        np.savetxt(self.book_path, book, delimiter=",", fmt="%d")
        np.savetxt(self.message_path, messages, delimiter=",", fmt="%.9f")
        self.book = book

    def prepare(self):
        return prepare_datasets(self.directory, sequence_length=8, stride=8)

    def test_split_before_windowing_and_equal_timestamps_stay_together(self):
        datasets, _, report = self.prepare()
        self.assertEqual(report["splits"]["train"]["raw_row_interval_half_open"], [0, 139])
        previous_last_time = -np.inf
        for name, data in datasets.items():
            self.assertGreater(data.times[0], previous_last_time)
            previous_last_time = data.times[-1]
            start, stop = report["splits"][name]["raw_row_interval_half_open"]
            for example in data:
                self.assertGreaterEqual(example["raw_start_index"], start)
                self.assertLess(example["raw_start_index"] + 8, stop)
                self.assertEqual(tuple(example["features"].shape), (8, 40))

    def test_scaling_does_not_learn_or_clip_future_volume_extremes(self):
        datasets, scaler, _ = self.prepare()
        self.assertLess(scaler.maximum[VOLUME_COLUMNS].max(), np.log1p(1000))
        self.assertGreater(datasets["validation"].features[:, VOLUME_COLUMNS].max(), 1)
        self.assertGreater(datasets["test"].features[:, VOLUME_COLUMNS].max(), 1)

    def test_inverse_transform_reconstructs_each_split(self):
        datasets, scaler, _ = self.prepare()
        for data in datasets.values():
            example = data[0]
            restored = decode_features(example["features"].numpy(), scaler, example["initial_prices"].numpy())
            expected = data.book[1:9].astype(float)
            expected[:, PRICE_COLUMNS] /= 10000
            np.testing.assert_allclose(restored, expected, rtol=2e-6, atol=1e-5)

    def test_mismatched_row_counts_are_rejected(self):
        np.savetxt(self.book_path, self.book[:-1], delimiter=",", fmt="%d")
        with self.assertRaisesRegex(ValueError, "row counts"):
            self.prepare()

    def test_missing_levels_are_not_treated_as_real_prices(self):
        self.book[0, 0] = 9999999999
        np.savetxt(self.book_path, self.book, delimiter=",", fmt="%d")
        with self.assertRaisesRegex(ValueError, "missing_level_entries"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
