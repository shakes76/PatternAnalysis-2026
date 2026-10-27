"""CPU and mocked-CUDA checks; no GPU tensors, MRI reads, or training."""

from pathlib import Path
import sys
import unittest
from unittest.mock import call, patch

import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from resource_monitor import StageMeasurement


class ResourceMonitorChecks(unittest.TestCase):
    def test_cpu_timing_has_no_cuda_activity_or_fabricated_memory(self):
        resident = {"model": True, "optimizer": False}
        with patch("resource_monitor.time.perf_counter", side_effect=[10.0, 12.5]), \
                patch("resource_monitor.torch.cuda") as cuda:
            with StageMeasurement("cpu", "validation", resident) as measure:
                pass
        self.assertEqual(measure.result["status"], "passed")
        self.assertEqual(measure.result["stage"], "validation")
        self.assertEqual(measure.result["device"], "cpu")
        self.assertEqual(measure.result["resident_state"], resident)
        self.assertEqual(measure.result["elapsed_seconds"], 2.5)
        self.assertTrue(all(measure.result[key] is None for key in measure.cuda_fields))
        self.assertEqual(cuda.mock_calls, [])

    def test_cpu_body_exception_propagates_with_failed_result(self):
        failure = ValueError("constructed stage failure")
        measure = StageMeasurement(torch.device("cpu"), "training", {"optimizer": True})
        with patch("resource_monitor.time.perf_counter", side_effect=[1.0, 3.0]), \
                patch("resource_monitor.torch.cuda") as cuda:
            with self.assertRaises(ValueError) as caught:
                with measure:
                    raise failure
        self.assertIs(caught.exception, failure)
        self.assertEqual(measure.result["status"], "failed")
        self.assertEqual(measure.result["error"], {"type": "ValueError", "message": str(failure)})
        self.assertEqual(measure.result["elapsed_seconds"], 2.0)
        self.assertTrue(all(measure.result[key] is None for key in measure.cuda_fields))
        self.assertEqual(cuda.mock_calls, [])

    def test_cuda_synchronization_reset_queries_and_device_are_ordered(self):
        device = torch.device("cuda:1")
        with patch("resource_monitor.time.perf_counter", side_effect=[5.0, 6.25]), \
                patch("resource_monitor.torch.cuda") as cuda:
            cuda.memory_allocated.side_effect = [11, 17]
            cuda.memory_reserved.side_effect = [22, 29]
            cuda.max_memory_allocated.return_value = 33
            cuda.max_memory_reserved.return_value = 44
            with StageMeasurement(device, "checkpoint_restoration", {"restored_model": True}) as measure:
                self.assertEqual(cuda.mock_calls, [
                    call.synchronize(device), call.reset_peak_memory_stats(device),
                    call.memory_allocated(device), call.memory_reserved(device),
                ])
            self.assertEqual(cuda.mock_calls, [
                call.synchronize(device), call.reset_peak_memory_stats(device),
                call.memory_allocated(device), call.memory_reserved(device),
                call.synchronize(device), call.max_memory_allocated(device),
                call.max_memory_reserved(device), call.memory_allocated(device),
                call.memory_reserved(device),
            ])
        self.assertEqual(measure.result["device"], "cuda:1")
        self.assertEqual(measure.result["status"], "passed")
        self.assertEqual(measure.result["elapsed_seconds"], 1.25)
        self.assertEqual([measure.result[key] for key in measure.cuda_fields], [11, 22, 33, 44, 17, 29])

    def test_cuda_body_exception_is_not_swallowed(self):
        failure = RuntimeError("constructed CUDA-body failure")
        with patch("resource_monitor.time.perf_counter", side_effect=[1.0, 2.0]), \
                patch("resource_monitor.torch.cuda") as cuda:
            cuda.memory_allocated.return_value = 11
            cuda.memory_reserved.return_value = 22
            cuda.max_memory_allocated.return_value = 33
            cuda.max_memory_reserved.return_value = 44
            with self.assertRaises(RuntimeError) as caught:
                with StageMeasurement("cuda:0", "training", {}) as measure:
                    raise failure
        self.assertIs(caught.exception, failure)
        self.assertEqual(measure.result["status"], "failed")
        self.assertEqual(measure.result["error"]["message"], str(failure))
        self.assertEqual(measure.result["peak_cuda_allocated_bytes"], 33)

    def test_collection_failure_propagates_without_masking_a_body_exception(self):
        for body_fails in (False, True):
            with self.subTest(body_fails=body_fails):
                body_failure = ValueError("original body error")
                collection_failure = RuntimeError("statistics synchronization failed")
                with patch("resource_monitor.time.perf_counter", side_effect=[2.0, 3.0]), \
                        patch("resource_monitor.torch.cuda") as cuda:
                    cuda.synchronize.side_effect = [None, collection_failure]
                    cuda.memory_allocated.return_value = 11
                    cuda.memory_reserved.return_value = 22
                    expected_failure = body_failure if body_fails else collection_failure
                    with self.assertRaises(type(expected_failure)) as caught:
                        with StageMeasurement("cuda:0", "training", {}) as measure:
                            if body_fails:
                                raise body_failure
                self.assertIs(caught.exception, expected_failure)
                self.assertEqual(measure.result["status"], "failed")
                self.assertEqual(measure.result["error"]["message"], str(expected_failure))
                self.assertEqual(measure.result["measurement_error"]["message"], str(collection_failure))
                self.assertIsNone(measure.result["peak_cuda_allocated_bytes"])


if __name__ == "__main__":
    unittest.main()
