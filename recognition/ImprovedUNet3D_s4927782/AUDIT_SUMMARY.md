# Initial feasibility evidence: 26 October 2026

The first inventory below is retained as development evidence. Later entries
record the completed integrity checks and experiments; final results are in README.

## Confirmed

- Authenticated access to Rangpur and read access to the course HipMRI root.
- `semantic_MRs` contains **211 NIfTI filenames**; `semantic_labels_only` also
  contains **211**. Equal counts alone do not establish correct pairing.
- The subsequent strict audit matched **211 image/mask pairs from 38 anonymous
  patient codes**. Every pair passed shape, affine, orientation and spacing
  checks. This establishes matching grids, not anatomical registration quality.
- Seed 3710 gives **26/6/6 patients** in train/validation/test and **143/38/30
  volumes**, respectively. Patient lists are disjoint and cover every pair.
- Three inspected masks contained finite integer IDs 0-5. Their anatomical
  meanings were then independently confirmed from the CSIRO v2 original
  `BriefDataDescription.txt`: 0 Background, 1 Body, 2 Bones, 3 Bladder, 4 Rectum,
  5 Prostate. `label_schema.json` records the source URL and verified digest.
- The existing `torch` environment has Python **3.11.15**, PyTorch **2.13.0**,
  NumPy **2.4.6** and Matplotlib **3.11.2**, as read from package metadata.
- Nibabel and SciPy were absent from that environment. The isolated project
  environment now provides Nibabel 5.3.2 and SciPy 1.16.3 without modifying it.
- Slurm exists under `/opt/slurm/bin/`. The course `comp3710` partition reports
  **10 A100 GPU nodes**. No GPU was allocated or model trained by this inventory.
- Scheduler metadata advertises 1 MiB of host memory per node and an unlimited
  memory default. An explicit 4 GiB CPU-job request was rejected. Setup uses
  the cluster's configured memory default and one CPU, matching the existing
  course workflow; this metadata is not a measurement of physical RAM.
- The audit utility passed **seven local automated checks**.
- The pairing/header/split audit passed **14 synthetic-data checks** on a CPU
  node. Initial 2D/3D model checks also passed five tests and a synthetic
  forward/backward/checkpoint roundtrip on CPU. A later A100 check also passed.
- The integrated pipeline then passed **56 checks**, including the new cache,
  metrics and weighted loss. A training-only cache of two scans supported one
  real-data update of each model and a successful 2D checkpoint resume. These
  deliberately tiny runs have no validation/test accuracy claim.
- A100 synthetic GPU smoke job 700900 completed successfully in 22 seconds.
  For the base-8, three-synthetic-class 3D prototype at 64x128x128, peak allocated
  memory was 737,915,904 bytes (0.687 GiB) and warmed model-only forward time
  22.20 ms. These are not final six-class/base-16 model resource results.
- Job 700962 reloaded all **211** complete cache volumes, checked image/mask/
  spacing validity and verified every mask's class counts against the manifest.
  The official label metadata is attached before formal training. Five distinct
  TRAIN patients were plotted and visually reviewed; sampled contours align.
- The latest integrated suite passed **75 checks** in job 700969, including
  actual CPU GradScaler overflow/recovery, official-label attachment and paired
  HD95 comparisons on common nonempty support. No final test score is implied.
- The expanded suite passed **85 checks** in job 701148. Failure galleries now
  choose distinct patients and show each model's largest full-depth FP/FN slice,
  including errors outside the ground-truth extent. Remote provenance checks
  verified eleven source/environment/job files against commit 69400a7 and the
  frozen manifest digest, then wrote sidecars beside both experiment runs.

## Completed experiment evidence: 27 October 2026

- Job 700974 completed a two-epoch full-data pilot; 700981 resumed each model
  to 30 total epochs on one A100 (22m15s for the continuation). Best validation
  epochs are 26 for 2D and 29 for 3D. Independent predictor job 701513 exactly
  reproduced both selected validation metrics.
- `test_protocol.json` was committed in c1edf16 before final test job 701544.
  The fixed comparison evaluated 30 scans from six held-out patients once per
  model. Patient-balanced prostate Dice is 0.67898 (2D), 0.67719 (3D); the
  original feasibility target of 0.70 was not met. Foreground Dice is 0.84091
  and 0.83753. No setting was retuned using these results.
- Job 701555 completed paired patient bootstrap, matched-support HD95 and five
  preselected patient-distinct failures. Job 701560 generated larger ROI panels
  without changing selection or predictions. The report preserves M015's severe
  failure rather than excluding it from the means.
- Job 701569 passed the final **95 checks**, exported decoded array digests for
  all 211 caches and rebuilt one actual TRAIN NIfTI pair (B006 Week0) with
  identical decoded image/mask/spacing arrays, without rewriting the manifest.
- Job 701590 completed retrospective CPU-only location/size descriptors against
  the training distribution. These diagnostics support hypotheses rather than
  establish a causal acquisition or clinical diagnosis.
- Both selected checkpoint SHA-256 values were verified locally against the
  predictor output and source-provenance sidecars. Data, arrays and weights
  remain outside Git. The prepared local weight bundle includes metadata and
  scores, without raw MRI, mask or prediction arrays; public release is pending
  explicit authorization.

## Remaining external requirements

- The official final deadline in Learn.UQ.
- Physical tutor check-off, final report organization and LMS submission.
- The teaching team's later PR review and response window.

The raw inventory remains under ignored `artifacts/`; scan files and trained
weights are never committed. All development-stage model checks must use
training/validation data.

## Data acknowledgement

Data were provided by HipMRI Study. The dataset usage note requires attribution,
an appropriate study citation, research/non-commercial use, and no attempt to
identify or contact participants. README includes the exact study attribution and publication, the original
CSIRO v2 label key and the data DOI.
