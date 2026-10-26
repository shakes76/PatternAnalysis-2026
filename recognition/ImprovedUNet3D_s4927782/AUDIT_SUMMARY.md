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
  remain outside Git. The publicly available weight bundle includes metadata and
  scores, without raw MRI, mask or prediction arrays. Its GitHub asset size and
  SHA-256 match the independently verified local archive; README links the release.

## Retrospective diagnostics and fixed follow-up: 27 October 2026

- CPU diagnostic 701739 passed 114 checks, audited 136 old prediction files,
  and independently reconstructed 12 native-to-cache cases exactly. All 12
  three-plane panels were reviewed. GPU diagnostic 701740 reproduced all 20
  comparable old AMP predictions voxel-for-voxel; independent GT metrics agree.
  All 48 diagnostic forwards were finite. Maximum AMP/FP32 prostate Dice change
  was 0.0014603. These scoped checks found no actionable discrepancy and do not
  identify M015's root cause or prove the full implementation is bug-free.
- Commit 27527d3 fixed `followup_protocol.json` before the second-seed run.
  Job 701767 completed both original models at seed43 for 30 epochs each on
  one A100; CPU analysis 701791 also completed with exit 0:0. Only TRAIN/VAL
  were evaluated. No seed43 TEST was run and v1 was not replaced or retuned.
- Selected epochs are 25/27 for 2D/3D at seed43. Validation prostate Dice is
  0.7809149/0.8351302; the paired-patient 3D-minus-2D mean is +0.0542154,
  versus +0.0411862 at seed42. All six validation patients favour 3D in each
  run. They remain the same six people, also used for checkpoint selection;
  these results are not a new independent test or evidence that M015 is fixed.
- Remote analysis passed 13 fixtures; the subsequent cloud-aware local version
  had passed 18 fixtures. Independent local recomputation matches all actual
  metrics and evidence identities. Both analysis-source hashes are retained;
  they are not falsely presented as the same program bytes. Actual bytes of
  all four selected checkpoints were rehashed and match their provenance and
  evaluations. Public evidence and the patient plot are linked in README.
- All four runs report A100-PCIE-40GB, Python 3.11.15, PyTorch 2.13.0+cu130 and
  CUDA 13.0. Unsupported deterministic CE/max-pool backward operations still
  warn, so retraining is not promised bitwise identical. Prepared cloud tools
  were unused after the original queued A100 job started; no cloud training
  or hardware/software comparison is reported. The public v1 release is unchanged.

## Source provenance and frozen-model investigation: 27 October 2026

- D1 CPU **702061** completed in 4m30s and GPU **702062** in 14s, both exit
  0:0. The 11 source/NiBabel, four archive-identity and 11 Torch fixtures all
  passed on the actual cluster. Commit **a085386** recorded the diagnostic
  methods before execution; the scope was fixed after the earlier failures.
- All 211 course MRI gzip files match the official ZIP catalogue's byte
  lengths/CRC32. All eight M015 masks and one TRAIN mask have whole decompressed
  NIfTI bytes and SHA-256 identical to the official files. Their gzip-header
  differences are not voxel corruption. This exact-content check covers nine
  masks, not the remaining 202, and does not establish anatomical truth.
- All 422 source headers match frozen geometry. Fourteen explicit cases were
  inspected from source through five processing stages. M015's negative global
  bladder-minus-Body contrast already exists in the original scaled images.
  Independently, cached descriptors cover 143 TRAIN, 38 VAL and only the eight
  already-examined M015 TEST visits: their Body-IQR-standardized contrast is
  [-0.6664,-0.4615], below the TRAIN minimum -0.3318. This GT-based association
  is not proof of a particular sequence, acquisition error or unique cause.
- Forty-two head/target checks agree with independent nearest targets and
  CE/Dice/weighted losses. Three selected backward checks have finite, nonzero
  gradients in all 58 parameter tensors. M015 Week2's mean GT-prostate class-5
  probability is 0.0274/0.0399/0.0400 at the three 3D heads, versus 0.4362 in
  2D; the main 3D head strongly prefers Body/Bladder. These fixed-weight checks
  do not reconstruct historical training gradients or certify all code.
- D1 directly observed only 16/10,521 M015 prostate voxels changed by support
  zeroing, but substantial whole-image changes left an indirect explanation.
  Commit **b233b15** froze a single-factor D2 intervention before GPU **702134**.
  That job completed in 1m06s, exit 0:0; all six actual fixtures and all 14
  exact baseline reconstructions passed. Removing only final support zeroing
  lowers 3D prostate Dice in seven/eight M015 visits, including Week2
  0.0014514 to zero. It does not rescue the 3D failure. All original weights,
  protected files and parameter/buffer states remain unchanged. No input
  policy or replacement TEST result is selected from this intervention.
- The CAN3D paper documents propagated prostate labels and N4 in its experiments,
  but no per-case sequence/registration records establish M015-specific errors
  or whether released images were already corrected. Appearance-related
  generalization failure is a supported interpretation; its unique causal
  mechanism remains unresolved. The report preserves counterexamples and the
  original unmet target rather than concealing negative evidence.

## Remaining external requirements

- The official final deadline in Learn.UQ.
- Physical tutor check-off, final report organization and LMS submission.
- The teaching team's later PR review and response window.

The raw inventory remains under ignored `artifacts/`; scan files and trained
weights are never committed. Training comparisons and model selection use
training/validation data. The explicitly retrospective selected TEST replays
above diagnose frozen results; they do not authorize tuning or replace the
original complete test benchmark.

## Data acknowledgement

Data were provided by HipMRI Study. The dataset usage note requires attribution,
an appropriate study citation, research/non-commercial use, and no attempt to
identify or contact participants. README includes the exact study attribution and publication, the original
CSIRO v2 label key and the data DOI.
