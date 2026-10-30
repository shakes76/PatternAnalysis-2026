# Frozen 3D U-Net evaluation

The root `predict.py` now supports direct checkpoint inference with labelled
MRI input, in addition to its original cached-render mode. See
[the direct example and resource protocol](resource_profile.md).

Slurm job **715670** completed on 2026-10-30 with an empty error log and five saved prediction figures. Held-out test mean scan prostate Dice was **0.773216**, mean HD95 **6.4995 mm** (approximate resized-grid spacing). Original metrics, training records and figures are archived in `support/saved_results/test_results/`. All eight synthetic checks on Rangpur and a synthetic figure layout inspection passed before submission; visual interpretation of the actual patient figures remains a separate step.

Evaluate `results/refine_v2/best_model.pt`, refinement epoch 11, after 15 initial scratch epochs. Validation prostate Dice was 0.75597. This does not establish test performance.

`support/submit_jobs/evaluate.sbatch` uses the original resource template (comp3710 partition/account, GPU, 1.5-hour time limit, no explicit CPU/memory request). It runs one held-out evaluation, then renders the saved predictions without a second inference pass. The frozen source directory on Rangpur is `evaluation_refine_v2_final/`; its model/data helpers are copied from the actual training snapshot `baseline_refine_v2/`.

## Compatibility and safeguards

- Model factory reads base channel count and BatchNorm/InstanceNorm choice from checkpoint metadata; parameter loading is strict.
- Evaluation rechecks the frozen training/validation manifest and derives the disjoint test group from the same seed-42 patient split.
- Uses the same preprocessing, [64,128,128] shape and BF16 inference as training validation. No learned parameter updates or prediction postprocessing.
- Records checkpoint SHA256, model/configuration, software/hardware and the evaluated split.
- A new output directory is required; existing evaluation results are not silently overwritten.

## Outputs under `results/final_test_refine_v2/`

- `summary.json`: per-class mean case Dice/IoU, mean patient Dice, pooled counts/scores, prostate HD95 and confusion counts, model parameters, GPU allocation and forward latency.
- `per_case_metrics.csv`: full-volume six-class scores, prostate HD95, predicted/GT volume ratio and erroneous prostate voxels within GT bladder/rectum. Confusion counts appear only on the prostate row, avoiding duplicate summation.
- `shared_2d_region_metrics.csv`: Dice/IoU restricted to the course 2D data's original axis-1 band [98,157], represented on the current downsampled 3D grid. This partial-band result must not replace full-volume test results.
- `boundary_extent_metrics.csv`: prostate scores on the first/last two GT-bearing depth planes. These approximate inferior/superior extent regions, not independently annotated clinical base/apex landmarks.
- `prediction_inputs/`: private cached image/GT/prediction arrays for five automatically selected examples, plus a manifest recording the selection reasons.
- `figures/`: five PNG comparisons showing MRI, GT, prediction and prostate TP/FP/FN, with inferior/middle/superior axial planes and a coronal shared-band plane.
- `evaluation_complete.txt` and `figures/plots_complete.txt`: completion markers for the two stages.

The deterministic example selection covers low Dice, high/undefined HD95, organ confusion, missed voxels and a middle Dice case, preferring different patients; it is not a random representative sample. All case metrics remain available rather than only selected images.

## Metric conventions

Mean case Dice matches training validation. Also report pooled Dice and patient averages separately; they are different aggregations. Both-empty class masks have undefined Dice/IoU and are excluded from valid-case averages, with valid counts reported. HD95 is undefined if either mask is empty and has an explicit status, never a misleading zero.

HD95 follows the earlier project convention: the 95th percentile of concatenated bidirectional six-connected surface distances. Spacing is scaled by original/target shape ratios after the (X,Y,Z)->(Z,X,Y) transpose. Millimetre values are approximate in the resized space and should not be presented as original-resolution clinical measurements. The course 2D files' identity affines must not be used to infer real physical distances.

Class names use the project's existing numeric mapping; metric computation itself uses numeric labels 0..5. The semantic naming requires the same dataset provenance as the training report.

Latency reports the mean/median over per-case median BF16 forward timings (three warmups, three measured repeats). It excludes data loading, transfer, argmax, metrics and plotting. Peak GPU allocation includes model/input/forward allocations, not training memory. Cross-hardware CPU/GPU timings must not be attributed solely to 2D/3D architecture.

## Use after submission

Read the Slurm output/error log and `summary.json` when the job finishes. Do not keep tuning against this held-out test set. Do not commit raw patient caches, datasets or checkpoint weights. Only appropriate de-identified visual content should be considered for the final report under course/data-use requirements.

## Repository layout

Current code is stored once at the project root. The original remote source
snapshot names above are historical run provenance; the cluster snapshots were
not changed. Current local scripts use the canonical code paths. Exact original
sources can be recovered through `support/old_code_and_notes/source_snapshots.json`.
