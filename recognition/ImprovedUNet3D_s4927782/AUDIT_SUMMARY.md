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
  and 0.83753. No v1 setting was retuned to replace this result; later post-test-informed exploratory runs are recorded separately.
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

## Completed appearance ablation A1: 27 October 2026

- Pre-result commit **5c49de3** froze `appearance_protocol.json`; **4e23d40**
  added independent audit/plot tools before results. Job **702249** completed
  in **40m06s**, exit 0:0, at **05:48:20 AEST**. Both 8-test suites and two
  isolated five-batch smokes passed before the two formal 30-epoch arms and
  all four-model/five-condition/38-VAL-volume evaluations. No cloud was needed.
- Both arms retain the exact original 3D training configuration at seed42;
  A adds keyed gamma, B adds the same gamma plus independently keyed polarity.
  Actual initial parameter SHA, runtime dictionaries and complete configurations
  match. The runtime remains A100-PCIE-40GB / Python 3.11.15 / Torch 2.13.0+cu130 /
  CUDA 13.0. CUDA warn-only nondeterminism remains a limitation.
- Selected clean-foreground epochs are **A30/B29**. Clean patient-balanced
  prostate Dice is **0.8335597/0.8071380**; under synthetic polarity it is
  **0/0.7881705**. Original seed42/43 references have clean **0.8332346/0.8351302**
  and polarity zero. Their clean predictions' six-class counts and metrics
  exactly match the earlier validation evidence.
- B-minus-A differences are clean prostate **-0.0264218**, clean foreground
  **-0.0111258**, polarity prostate **+0.7881705**. The fixed allowed clean
  drops were 0.02 each, with polarity gain at least 0.05. Thus the clean-prostate
  guard and overall indication **fail**, despite the synthetic-stress gain.
  Five of six clean patient means worsen; no threshold or recipe is changed.
- A/B training loops take **1,023.29/1,021.84 s**, each with **4,287 successful
  updates, 3 AMP overflow skips**, and peak **2.227 GiB** allocated CUDA memory.
  Job wall time includes validation, smokes, saving, checks and evaluation.
- Downloaded private archive: 34 whitelisted protocol/metadata/log/best-weight
  files, **48,373,760 bytes**, SHA256
  `65826f430f6ae811bd42d14632b18cec0fb01d601c4adbc9cb51384daa722465`.
  Safe extraction refused changed existing files; no raw data or predictions
  were included. New weights remain ignored and outside the v1 public release.
- The summary's seven fixtures passed, then actual source/receipt/epoch/case
  checks passed. Two independent audits recomputed all **240** class-level
  patient-balanced Dice/IoU means from all 20 evaluations, with agreement;
  GT voxel counts remain identical across conditions. Public anonymous evidence
  and a complete four-by-five/paired-patient plot are linked in README.
- These are the same six checkpoint-selecting validation patients. The result
  shows a synthetic invariance/clean-performance trade-off; it neither repairs
  the original TEST nor establishes M015's unique cause. V1 remains unchanged.

## Completed M015-only exploratory follow-up M1: 27 October 2026

- Separate protocol commit **a11a597** preceded CPU job **704030**, which
  completed in **45 s**, exit 0:0; all **seven actual fixtures** passed.
  The A1 protocol and its no-TEST scope remain unchanged. The separate
  M1 evaluates only the eight already-examined M015 clean caches with the
  existing v1/A/B best checkpoints, all on **CPU FP32, AMP off, four threads**.
  It performs no training, selection, augmentation, TTA, postprocessing,
  other-patient TEST evaluation or array export.
- Mean prostate Dice over eight visits is **v1 0.1134373 / A 0.2781180 /
  B 0.7018369**. B improves all eight visits against both comparators; A
  improves four and worsens four against v1. Week2 changes from **0.0014514**
  to **0.4967949 / 0.7447917** with A/B. All eight visit results remain in README.
- For unchanged GT **10,521** voxels, v1 TP/FP/FN is **657/122/9,864**;
  B is **6,557/1,557/3,964**. This is substantial omission mitigation with
  more FP and residual FN, not complete repair. B's visit Dice ranges
  **0.6187384–0.7634011**; pooled counts are not substituted for visit-mean Dice.
- All selected cache identities, three checkpoints/states, receipts and **28
  source/input guards** remain unchanged. Independent arithmetic covers all
  **144 six-class case rows**. CPU-reference prostate counts/Dice match the
  existing GPU FP32 diagnostics per visit; compared with original CUDA AMP,
  only Week4 prostate Dice differs (-0.0014603), shifting the mean -0.0001825.
  This does not assert full-array equality across devices or change v1 scores.
- `evidence/m015_followup_review.json` records anonymous counts and checks.
  M1 is exploratory evidence from one previously inspected patient, not an
  independent new benchmark or confirmation of a unique acquisition/data cause.
  B still fails A1's clean-VAL guard and is not adopted; the original unmet
  six-patient prostate target and v1 public release remain unchanged.

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

## Bounded recovery and methodological closure: 27 October 2026

- Commit **d046435** fixed R1 before training: two B29 warm starts, fresh optimizer/scaler, lr1e-4, eight full epochs, polarity probability0/.10 only, with original clean-FG selector. One GPU Lab allocation1884 ran both arms and all5models×5conditions×38VAL on L40S/Python3.10.21/Torch2.14.0+cu130. Both selected epoch7. Actual18fixtures and two smokes passed; each arm had1,144 successful updates, no AMP skips. Training-loop seconds205.37/204.28 exclude validation/saving.
- All32protectedfiles, parent/weight/source/cache receipts and paired initial states/sharedgamma/runtime were verified. Independent count audit reaggregated5,700case-class rows and300patient-balanced Dice/IoU values. Same-platform v1/B/C/D clean prostate Dice .83326/.80715/.82078/.82019; C/D restore part of B's loss, but each fails the clean-prostate gate. FG and polarity gates pass. Frozen VAL recommendation is none; no threshold changed. Historical A100 numbers remain separate.
- **85b9a9e** fixed the VAL summary and M2 protocol before CPU704150, which completed38s0:0 with all11fixtures passing. B/C/D all8knownM015 visits use the same CPU4 FP32/Python3.11.15/Torch2.13.0+cu130. Mean Dice .70184/.63956/.63603, minimum .61874/.58284/.56823. Both new arms fail both prespecified retention checks, while VAL recommendation stays none. B's48six-class case rows match M1 counts/metrics exactly; this is not an array-equality claim. Independent144-row count and43source/88metadata hash audit passed.
- Public evidence/recovery_review.json and evidence/recovery_m015_review.json retain all positive/negative outcomes, and figures/recovery_validation.png shows all six patients. Later weights remain private/ignored; originalv1/Release/core11 remain unchanged. Required cloud artifacts were downloaded and all45file SHA checked before user-approved instance shutdown.
- Human raised the validity of repeated result-driven tuning. The report now keeps originalv1 as the primary benchmark and places retrospective diagnostics, seed follow-up, A1/M1/R1/M2 in an exploratory appendix. Per-round prespecification does not remove cross-round adaptive test-informed bias or repeated-VAL overfitting. No joint success, unique cause, model adoption or independent improvement is claimed. No further tuning or experiments will be added; remaining work is report QA and submission organization.

## Final rubric and submission-material audit: 27 October 2026

- Re-read the supplied v2.5 specification pp5-6,12-17 against actual code, report and GitHub metadata. Its total is40 (implementation20, commit5, README10, PR5), with compulsory ungraded check-off as well as commit/PR hurdles. Hard eligibility is supported by actual3D residual units and deep supervision; it is not an automatic20/20.
- The Hard table does specify test Dice>=.7 without naming the class/aggregation. Both foreground.8375 and prostate.6772 remain visible, with the explicitly planned prostate.70 criterion unmet. The specification provides no automatic fixed deduction formula. The Easy2D row's prostate.75 goal is not imposed as another threshold on this Hard project's comparator.
- Pure documentation repairs: justify the unchanged70/15/15 patient allocation, explain on-the-fly six-channel one-hot labels forDice/class-indexCE, map the five required files and separated finalTEST driver, and explicitly locate existing superior/inferior failure evidence. No new arrays, inference, training, tuning or source/protocol changes. Added a limit on unmeasured organ-wall contact distances. PR description now directly cites architecture/data and own prior experience.
- Static review identified a latest/best two-file crash window in originaltrain.py. This is a robustness limitation, not evidence that it occurred. README now limits resume claims and requires independent best/history/VAL consistency after abnormal interruption. Actualcompletedv1 already passed selected-history/reloadedVAL matching; original11sourcebytes and scores stay unchanged. No new test or training is claimed.
- Existingcheckoffdate is labelled planned, not completed; the one-page snapshot remains historical and is exported consistently. No tutor completion, exactcourse-profile midpoint/deadline, Turnitinreceipt or laterreview response is fabricated. CurrentPRisOPEN/DRAFT, with no teachercomments asofthisaudit; Draft is not explicitly forbidden by the supplied specification.
- Mandatory AI disclosure remains absent at human's explicit request to organize it together last. It is recorded as a pre-submission requirement, not treated as complete. Student verification/understanding also cannot be inferred from the31commits available before this documentation pass.


## Integrated higher-resolution Hard follow-up: 28 October 2026

The original report is preserved byte-for-byte as `README_V1_HISTORY.md`; the original model/data/training sources and sealed v1 scores remain unchanged. The current README presents the completed higher-resolution Hard pipeline alongside those original references. It does not replace or hide the failed original prostate goal.

The separately recorded Hard-only study completed 30 epochs in GPU job705405 (47m03s) and CPU comparison job705447 (2m55s). The attempted new2D/new3D pair had stopped after new2D failed its fixed TRAIN fitting gate; that failure and the subsequent explicit scope reduction remain documented. NewHard selected epoch30 using scheduled VAL prostate Dice; its checkpoint SHA256 is `852e3e1ee0a8f41be8a50902b7f6f2e28ffaa8227144d6ddf7806e06670e88bf`.

On the same original64x128x128 ground-truth grid, six patients and30 TEST scans, patient-balanced prostate Dice is .678982/.677191/.781784 for original2D/original3D/newHard, and foreground Dice .840911/.837528/.862869. The new numerical prostate goal is met under the declared aggregation, while10/30scans remain below.70. M015 improves .113620 to.665621, W029 remains.635941, and W029/Week6 worsens .544709 to.351085. These are retained with all per-class results, matched-surface HD95, five distinct-patient examples and costs.

This is a retrospective follow-up on already-inspected TEST patients with reused VAL patients, not a new blind test. Resolution, TRAIN sampling/exposure, class weights recomputed on the new TRAIN grid, seed, validation selector and sliding-window inference differ; no individual cause or isolated dimensional benefit is claimed. NewHard training-loop time is2.11x and complete inference20.64x the original3D, despite the same parameter count.

The earlier internally imposed VAL gain adoption gate was not a teacher requirement. Its use as a current assignment submission restriction was withdrawn after results at the user's request; the original frozen protocol and actual failed gate remain historical evidence. It is not recast as a prospective pass or independent generalization result. The selected weights, metrics and scientific code are unchanged by report integration.

This integration performs no new training, tuning, model selection or real-MRI inference. Public source/evidence is limited to the required code, configuration, anonymous counts/provenance and authorized visual examples. MRI, GT/prediction arrays and all model weights stay outside Git. Author verification, recorded tutor check-off, Turnitin receipt and later teacher review responses are not claimed complete. Final integration checks and artifact identities are recorded in the private project state and package manifest.
