"""Verify byte/GiB accounting and invalid-value rejection without needing a GPU."""
import unittest

from gpu_profile import build_memory_report


class MemoryReportTests(unittest.TestCase):
    def test_known_byte_to_gib_conversion(self):
        report = build_memory_report(1073741824, 2147483648, 42949672960, 12.5)
        self.assertEqual(report['peak_allocated_gib'], 1.0)
        self.assertEqual(report['peak_reserved_gib'], 2.0)
        self.assertEqual(report['device_total_gib'], 40.0)
        self.assertEqual(report['run_wall_seconds_including_io_and_plot'], 12.5)

    def test_invalid_memory_counters_rejected(self):
        for counters in ((-1, 2, 40), (1, -1, 40), (0, 2, 40), (1, 2, 0), (True, 2, 40), (1.0, 2, 40)):
            with self.subTest(counters=counters):
                with self.assertRaises(ValueError):
                    build_memory_report(*counters, elapsed_seconds=1)

    def test_invalid_elapsed_time_rejected(self):
        for value in (-1, float('nan'), float('inf'), True, '1'):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    build_memory_report(1, 2, 40, value)


if __name__ == '__main__':
    unittest.main()
