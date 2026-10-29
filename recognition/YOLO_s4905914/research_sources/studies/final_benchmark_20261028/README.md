# Final geometry n: resource timing only

This follow-up fills the final-model latency and throughput cells in the course report. It does not train a model, calculate accuracy metrics, inspect reassessment images, or select a checkpoint or threshold.

The frozen model is geometry YOLOv8n, seed 3710, confidence **> 0.40**, checkpoint SHA-256 `a1bd556186a1dc98d4b1d731fd59897e3084a70f6484e6b700586e6a09ef794c`. `inputs.sha256` records the checkpoint, selection lock, configuration, validation manifest, benchmark source and its dependencies. The job checks these hashes before and after timing.

## Protocol

- Reuse the original n/s `benchmark_inference.py` unchanged (SHA-256 `6d6d72fa4fe54677890229bc6ae8e97a73e3e9ba00bec046c1e3062ca028c8a3`).
- NVIDIA A100, FP32, batch one, 640-pixel inputs; pinned Python and package environment from the original study.
- Same 165 validation image paths and order; ten warm-up calls, then three complete passes (495 timed calls).
- Class-aware NMS at 0.70, score floor 0.001, max_det 300, final strict score filter > 0.40. No threshold search.
- Wall time uses CUDA synchronization and covers file read/decode, preprocessing, transfers, model, NMS, threshold filtering and copying predictions to CPU. Model-stage time is the library's inference timer. Neither includes startup, warm-up, drawing, saving or UI.
- Throughput is timed image count divided by total wall time. Repeated image reads may use warm caches; these measurements are not cold-start or mobile performance.

## Run and evidence

From the school research workspace, after restoring the recorded inputs:

```bash
sbatch studies/final_benchmark_20261028/benchmark_a100.sh
```

Job 706962 completed successfully on A100-5 on 28 October 2026 (exit 0; 2 min 29 s including environment setup). The script creates a fresh `outputs/final_benchmark_20261028_JOBID` directory with the raw timing observations, summary, input checks, hardware, dependency freeze and scheduler log. It uses node-local scratch for dependency installation rather than the home quota.

The original n/s timings are historical comparators, measured in an earlier session at their own locked thresholds. The final n measurement is a separate session and is descriptive resource evidence; differences in load, clock state and file caching are not controlled enough to attribute speed differences to training or augmentation. This benchmark does not alter the earlier CPU reassessment's device record or classification/detection results.

## Completed measurements

Mean model-stage time: **6.51 ms**. Mean file-to-prediction time: **66.06 ms**. Serial throughput: **15.14 images/s**. The 495 observations were independently replayed to verify means, medians, 95th percentiles and throughput. Before/after input hashes matched.

The report reads `recognition/YOLO_s4905914/final_candidate_benchmark_20261028.json`; adjacent files preserve the raw observations and verification record. Hardware and dependency evidence are retained in the local job output directory.
