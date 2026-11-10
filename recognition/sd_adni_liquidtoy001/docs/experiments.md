# Experiment Ledger

No model experiment has completed yet.

Pre-experiment audit: [Rangpur/ADNI, 2026-10-02](rangpur-audit-2026-10-02.md), CPU job 620988, completed with exit 0:0. This is a dataset/resource inspection, not a model experiment. The report also records unsuccessful preliminary shell probes.

Data preparation: [status](data-preparation-status.md). CPU job 623310 completed with exit 0:0 in 40 seconds. Full decode, source coverage, patient separation, deterministic allocation, sampled loader checks and negative controls passed. Aggregate split counts are recorded in the linked report.

For each real run, record:

- Run ID, purpose/hypothesis, and date/time with timezone.
- Git commit and any uncommitted changes; exact command and configuration.
- Dataset/split manifest version and patient separation checks.
- Model identifier/revision, trainable components, seed, and preprocessing.
- Environment versions, node/GPU, job ID, allocated host resources.
- Training steps, batch/accumulation settings, precision, and evaluation protocol.
- Measured loss/metrics, GPU/host memory, timing, and checkpoint size.
- Exit status, artifact paths, recovery verification, failures, and interpretation.

Keep failed and inconclusive runs. Distinguish measurements from estimates and proposed experiments. Store bulk logs and weights outside Git; reference them with stable run identifiers and checksums where appropriate.

## VAE diagnostic queued — 2 October 2026

- Protocol: [pre-experiment design](vae-protocol.md).
- CPU environment/download job **623826**: last checked PENDING (Resources).
- GPU reconstruction job **623828**: submitted with `afterok:623826` and
  `--kill-on-invalid-dep=yes`; ten-minute limit, one GPU, two CPUs.
- If setup fails, reconstruction must not run. Neither job trains a model.
- Local syntax compilation and Ruff lint/format checks passed. This does not
  establish package compatibility, download success or GPU acceptance.
- Source: `requirements-vae.txt`, `tools/download_vae.py`, `tools/check_vae.py`,
  and their two Slurm launchers. Uses the existing accepted private manifest.
- Remote project: `~/comp3710-sd-audit/project`.
- Setup logs: `logs/vae_setup_623826.{out,err}`; full dependency snapshot:
  `logs/vae_environment.txt`; pinned model hashes: `models/sd15-vae/provenance.json`.
- Reconstruction logs: `logs/vae_check_623828.{out,err}`; completion record:
  `outputs/vae-623828/results.json`; triptychs in the same ignored directory.
- Retrieve both exit statuses and inspect numerical results and images before
  reporting success. No reconstruction metric is claimed while queued.

## Dependency failure and retry — 3 October 2026

Job 623826 FAILED with exit 1:0 after 10 seconds. Pip rejected the requested
huggingface-hub 2.1.1 because diffusers 0.40.0 requires >=1.23.0,<2.0.
The initial pins were incompatible; no VAE results were produced. Dependent GPU
job 623828 was CANCELLED without running, as intended by the dependency guard.

Verified the version constraint from PyPI package metadata and changed only
huggingface-hub to 1.33.0. CPU setup retry **625832** and dependent GPU diagnostic
**625833** were submitted. Last observed: setup pending resources, diagnostic
waiting on successful setup. Real environment resolution and model execution
remain unverified until those jobs complete. New outputs use their new job IDs;
failed-run records are retained.

### Setup retry accepted

Job **625832 COMPLETED, exit 0:0, 1 minute 19 seconds**. Package installation,
`pip check`, pinned VAE download and CPU model loading all passed. The downloaded
weights occupy 334,643,276 bytes. See `vae-model-provenance.json` for revision and
file hashes and `vae-environment-2026-10-03.txt` for resolved package versions.
GPU reconstruction job **625833** remains pending; no reconstruction figures or
quality metrics are available yet. This supersedes the retry's pending-setup
snapshot above. No SD training has been run.

## VAE reconstruction accepted — 4 October 2026

After about 29 hours pending in comp3710, existing job 625833 was moved with
`scontrol update JobId=625833 Partition=a100-test`. Eligibility and the test
partition's 20-minute QoS limit were checked first. No duplicate was submitted.
It COMPLETED with exit 0:0 in 33 seconds on a100-b, NVIDIA A100-PCIE-40GB.

All 16 records (8 preselected slices at two resolutions) and corresponding
triptychs were retrieved into ignored `outputs/vae-625833/`. Finite checks passed;
the held-out project test split was not evaluated. Original source directory
names may say test because the project uses a new patient-grouped manifest.

| Resolution | Mean per-image PSNR, 8 images (dB) | Peak torch allocated GiB |
| --- | ---: | ---: |
| 256 | 27.0383 | 0.5218 |
| 512 | 35.3320 | 1.0866 |

PSNR is measured against each resolution's own preprocessed input after decoder
clipping. These numbers are not a common-native-resolution comparison or SD
training metrics. Upsampling to 512 does not create new acquired detail. GPU
allocation is VAE-only, not total device use or future training memory. Timing
includes a cold first inference and is not a warmed throughput benchmark.

Visual inspection of validation-1-AD at both sizes found preserved gross shape,
more visible fine-detail smoothing at 256, and smaller displayed error at 512.
This is one inspected pair, not a medical or population-wide quality judgment.
The raw decoder clipping fractions span roughly 47–73%; before finalizing the
protocol, measure unclipped error and distinguish background undershoot from
foreground overshoot. These fractions alone do not establish severe distortion.
Next: refine that diagnostic and prepare bounded training/save/resume validation;
no optimizer update has yet been run.

## Short training and restart acceptance — 4 October 2026

Protocol was recorded before execution in `smoke-training-protocol.md`.
CPU job 629842 completed in 2m28s, exit 0:0: dry-run resolution, installation,
pip check, pinned component download, CPU load and LoRA attachment passed.
VAE diagnostic 629851 completed in 12s, exit 0:0. Intensity-mask (>0.02) foreground
clipping averaged 0.97%/0.21% at 256/512; background clipping averaged 66.9%/76.3%
but correction magnitude was approximately 0.004. Raw/clipped MSE is retained
in `vae-clipping-2026-10-04.json`. This mask is not anatomical segmentation.

GPU job **629852 completed in 55 seconds, exit 0:0**, in a100-test. Real pretrained
SD1.5 LoRA updates ran as continuous8, interrupted4 and resumed4 in three Python
processes. Final adapter maximum difference was **0.0**. Optimizer, scheduler,
CPU/CUDA RNG and input/noise/timestep/loss traces matched within documented
floating tolerances. Corrupt-contract and changed-adapter negative controls were
rejected. No decreasing-loss or image-quality claim is inferred.

Trainable parameters: 797,184. Peak torch allocated bytes: 6,109,799,424 (~5.69 GiB).
Continuous checkpoint: 9,884,717 bytes (~9.43 MiB), includes optimizer and RNG.
Measured eight-update/save region: 4.376s; full three-process job: 55s. Do not use
this tiny run as a production throughput estimate. Frozen source weights are
loaded from the pinned fp16 files into float32; bfloat16 autocast is used for UNet
compute. Diffusers emitted dtype-casting advisory warnings; no non-finite output
or acceptance failure occurred. The constructor dtype cleanup can be considered
with future trainer changes, without altering this accepted historical run.

All raw output JSONs are retained in ignored local `outputs/`; model manifests,
environment snapshot and aggregate acceptance are in docs. No medical images or
patient-level manifests are uploaded to the website or public repository.
The HTML is updated with these measured states. Formal baseline/adaptation,
generation evaluation, and signal-driven checkpoint recovery remain outstanding.

## Paired generation pilot submitted — 4 October 2026

Job629919: a100-test, one GPU,20-minute wall-time cap. Protocol is in
pilot-protocol.md. Selected32 training patients (16AD/16NC),200 updates, fixed
8-image DDIM grids at steps0/100/200. Checkpoint100 resumes in a new process.
Submission is not success; results are pending. Held-out test data stays sealed.

### Pilot629919 result

COMPLETED, exit0:0,2m15s on A100. Retrieved both checkpoints, reports and all24
images into ignored local outputs/pilot-629919. Verified steps100/200, resumed
flag, identical contracts, preserved first100 traces,32 unique training indices,
finite loss/gradients, eight records per stage, PNG hashes, and identical paired
initial-noise hashes across all three stages. Training peak allocation5.69GiB;
final checkpoint9,912,045bytes. Aggregate: pilot-acceptance-2026-10-04.json.

Visual inspection of all three uncurated grids: step0 is strongly colored and
non-MRI-like; steps100/200 become predominantly grayscale with brain-like textures.
However, multiple disconnected structures, implausible geometry and collage-like
composition persist. AD/NC images sharing a seed are visually very similar.
This does not demonstrate anatomical validity or learned disease conditioning.
These are observed failure modes, not verified causal diagnoses. Sampling uses
256 pixels and guidance1, so these observations do not characterize every possible
base-model inference setting. Do not attribute changes solely to generalization
or infer privacy; nearest-neighbor auditing has not been done.

Next: investigate the sampling/settings contribution and training-slice composition,
then implement the simple baseline and broader evaluation before extending training.
No automatic long training was submitted.

## Full-split long experiment queued — 4 October 2026

User requested prioritizing the longer training queue to preserve analysis/rework
 time. Recorded full-training-protocol.md before submission. Frozen remote source:
~/comp3710-sd-audit/runs/full20k-20261004, with source-sha256.txt; shared model/data
and environment symlinks. No queued source will be edited in place.

Gate629930 runs actual continuous8 versus4+4 state comparison in a100-test.
Long job629931 uses afterok:629930, comp3710 account/partition,1GPU,2CPUs,4h cap.
Initial state: gate RUNNING; long PENDING(Dependency). Fresh base weights, all
training-split slices eligible,20000 updates, complete checkpoints and fixed
sampling every2000 updates. Approximately0.94 slice epochs, not proof of convergence.
Queue position and completion time are unknown. No validation/test evaluation.

Gate629930 COMPLETED in46s, exit0:0. Full state comparison passed with final
adapter max difference0.0; corruption negative controls rejected. Acceptance is
saved in full-gate-2026-10-04.json. Long job629931 dependency is satisfied and its
latest queue reason is Priority, not a failed test or missing GPU request.

## Long run629931 completed — verified5 October2026

Slurm COMPLETED, exit0:0, elapsed1h21m39s,20000 steps. Retrieved all ten
stage directories (approximately140MiB) including full checkpoints to ignored
outputs/full-629931. Verified report step counts, checkpoint byte sizes, finite
loss/gradient traces, matching contracts and all resumed trace prefixes, and all
88 generated PNG hashes with matching initial noise across stages. This is artifact
and trace verification, not a new continuous20000-step equivalence test.

Inspected the final eight-image uncurated grid: brain-like textures persist, but
fragmented/collage structures and background block artifacts remain. Matched-seed
AD/NC images look very similar. Generation quality is not accepted. Causes are
not established. Next priority is inference parity against the official pipeline
and controlled resolution/guidance comparisons, followed by training-data visual
inspection and nearest-neighbor audit. Do not simply infer that more steps will
resolve the artifacts. No new long job submitted during this verification.

## Inference audit submitted — 5 October 2026

Job632530 in a100-test is RUNNING on a100-a. Protocol recorded before execution
in inference-audit-protocol.md. No optimizer updates and no held-out data.
The official pipeline is compared against four historical quantized images and
then used for16 fixed factorial samples. A mismatch is diagnostic evidence, not
a reason to silently loosen tolerances. In particular, the historical loop casts
UNet output tofloat32 before the scheduler; the official mixed-precision path may
retainbf16 there. If outputs differ, distinguish arithmetic precision from sampler
logic before attributing generation artifacts to an implementation error.

### Inference audit632530 result

COMPLETED, exit0:0,1m57s. Retrieved all16 images and four uncurated grids.
Verified PNG hashes and identical initial noise across guidance settings within
each resolution. Four historical pairs do not match exactly: mean absolute pixel
differences0.93–2.29 on0–255 scale, maximum46–105. A follow-up632535 aligns only
the UNet-output float32 scheduler boundary and repeats the four historical pairs;
this distinguishes the suspected precision effect without changing trained weights.

Visual inspection of all four grids: guidance7.5 removes much of the collage and
background artifacts in these fixed examples at256 and512. Guidance1 at512 is
heavily textured/fragmented; merely increasing resolution does not fix it.
The256/guidance7.5 seeds look highly similar. These tiny selected-in-advance grids
cannot establish distribution coverage, anatomical validity, disease specificity
or privacy. Do not claim guidance7.5 is optimal or512 is universally better.

Follow-up632535 COMPLETED, exit0:0,26s. After matching the float32 scheduler-input
boundary, all four historical outputs match the official pipeline pixel-for-pixel
(max difference0), with identical historical initial-noise hashes. PNG hashes
verified locally. This strongly supports sampler consistency on the tested cases;
it does not prove all prompts/settings or intermediate latent equality. The prior
small mismatch is explained by the controlled precision-boundary change. No weight
update was needed. The large collage failures remain reproducible with guidance1,
while guidance7.5 changes sample appearance substantially on this small fixed set.
Next: broader predeclared seeds, nearest training/validation neighbors, diversity,
and a separately trained simple baseline. No further long training submitted.

## Cohort audit and baseline queued — 5 October 2026

Protocol: cohort-baseline-protocol.md, recorded before execution. GPU632557
produces64 base and64 adapted SD outputs at256/guidance7.5, with predeclared
seeds37200–37231. CPU632558 depends on it and performs private full/matched
train/validation nearest-neighbor and diversity diagnostics. Test remains sealed.

Baseline gate632563 was cancelled while pending to replace it with an immutable
source checkout after lint cleanup. Replacement gate632569 and dependent5-epoch
baseline632570 use ~/comp3710-sd-audit/runs/baseline128-20261005 with recorded
source hashes. Basic conditional conv VAE,128 native pixels,64 latent dimensions,
no pretrained weights. Epoch checkpoints support bounded loss on interruption;
no emergency signal-save claimed. Gate tests actual updates and exact save/reload
inference, not full interrupted-training trajectory equivalence. Submission and
static checks are not successful model or metric results.

## Cohort and baseline results verified — 6 October 2026

632557 COMPLETED6m08s;632558 COMPLETED53s;632569 COMPLETED8s;632570 COMPLETED3m02s;
all exit0:0. Retrieved128 SD images,64 baseline images and all five baseline
checkpoints. Verified192 image hashes, SD initial-noise pairing, five epochs each
covering21280 training samples, finite objectives and final checkpoint SHA256.
Supplementary baseline CPU audit635709 completed51s, exit0:0, using the same
previously recorded metric/reference protocol. Private neighbors remain ignored.

| Model | Matched train NN mean RMSE | Matched validation NN mean RMSE | Generated pairwise mean RMSE |
| --- | ---: | ---: | ---: |
| Base SD | 0.493368 | 0.495440 | 0.442523 |
| Adapted SD | 0.130497 | 0.131289 | 0.182098 |
| Small VAE | 0.077038 | 0.077355 | 0.117431 |

References:102 patients per split (33AD/69NC), one deterministic slice each.
All distances are grayscale64 pixel RMSE. Exact-copy reference controls passed.
The full-reference adapted-training minimum distance0.07021 is nonzero, but this
cannot exclude approximate memorization. Similar matched train/validation means
are a limited observation, not a privacy guarantee or statistical equivalence.

Visual review of all three uncurated grids: base SD produces colorful/illustrative
or collage-like images far from the target appearance; adapted SD produces much
sharper, similar brain-like structures with residual colored/background artifacts;
small VAE generates blurred structures. The blurred baseline wins low-resolution
RMSE. This is evidence that this metric alone is insufficient to rank fine-detail
quality, not evidence that SD is categorically better. Pairwise variability of
invalid base outputs is not useful anatomical diversity. Neither disease-label
control nor anatomical validity is established by these grids.

Baseline1268961 parameters;5 epochs=106400 training-image exposures, compared with
SD20000 updates at batch1. Baseline native128 versus SD256 further limits direct
fairness. Training VAE objective declined224.81->89.90, but this is not held-out
validation quality. Next: real-vs-real distance calibration, foreground/feature
metrics and private nearest-neighbor visual panels, then3–5 representative failure
analyses. Keep the test split untouched until final evaluation protocol is fixed.

## 2026-10-06 — Real-reference calibration and case inspection

CPU job635718 completed43s, exit0:0. Matched queries contain102 patients per
split (33AD/69NC), one deterministic slice each. Training queries exclude all
images from the same patient. Grayscale64 RMSE: real validation to full training
mean0.073101; real training to other-patient full training mean0.076297; real
validation to matched training mean0.091846. Real matched-set pairwise means:
train0.161217, validation0.152429. These are context, not privacy thresholds.

Five private panels inspected: gray background, colored artifact, and simple-VAE
blur are observed failures; minimum-distance and repeated-shape examples are
unconfirmed risk probes. Causes remain hypotheses requiring controlled checks.
See calibration-2026-10-06.json and failure-autopsies-2026-10-06.md. No test-set
evaluation performed. Next: foreground/color/edge and feature-space diagnostics.

## 2026-10-07 — Appearance reference calibration

CPU job636215 COMPLETED in7s, exit0:0. All192 generated images and102 validation
images passed recorded SHA256 checks; downloaded report script hash matches local
source. No test images read. Local refactoring reproduced all nine earlier
aggregate means within1e-12. Ruff and shell syntax checks passed.

| Group | n | Mean color range | Mean outer-border intensity | Mean edge difference |
| --- | ---: | ---: | ---: | ---: |
| Original SD | 64 | 0.344538 | 0.439551 | 0.060737 |
| Adapted SD | 64 | 0.020839 | 0.059221 | 0.022961 |
| Simple VAE | 64 | 0 | 0.000077 | 0.007563 |
| Real validation | 102 | 0 | 0.000000100 | 0.017551 |

AD and NC separately show the same edge ordering. Validation edge means are
0.017292/0.017674; VAE0.007623/0.007502; adapted0.023217/0.022704.
The results support the smoothing observation, not anatomical-quality rankings.
Native128 vs256, slice position, model architecture and losses remain confounded.
Single-channel real/VAE outputs are grayscale by construction, not a learned win.
Border is a spatial proxy, affected by padding; no segmentation was performed.

F1 border0.411542 versus real-reference maximum0.000004085 corroborates the
background mismatch. F2 color0.021035 confirms non-grayscale output, but does not
localize the cyan patch or establish its cause. F4 edge0.005648 is below the real
reference range0.010177-0.025855, consistent with smoothing; not a diagnostic
threshold. F3/F5 remain unconfirmed risk probes. No retraining decision yet.

Next controlled step: compare the existing fixed-seed CFG/resolution probes with
these same descriptive measures, then choose any new sampling ablation. Learned
features, foreground segmentation, interpolation and final test evaluation remain
unperformed. See appearance-protocol.md and appearance-result-2026-10-07.json.

## 2026-10-07 — Retrospective sampling appearance comparison

Measured all16 images from job632530. SHA256 verification and complete factorial
coverage passed. Each within-resolution CFG pair shares its initial-noise hash.
This is the original unaligned scheduler-precision run; do not merge it with the
later aligned cohort as though precision were controlled across runs.

| Resolution | CFG | Mean color range | Mean border | Mean edge |
| --- | ---: | ---: | ---: | ---: |
| 256 | 1 | 0.010626 | 0.178816 | 0.030590 |
| 256 | 7.5 | 0.016531 | 0.004558 | 0.022557 |
| 512 | 1 | 0.007888 | 0.398652 | 0.041371 |
| 512 | 7.5 | 0.009361 | 0.051831 | 0.021844 |

All eight CFG pairs decrease border and edge measures. Seven increase color
range. This refines the visual statement that higher guidance reduces artifacts:
it improves the border proxy here but does not consistently improve grayscale
fidelity. Edge decrease is not itself anatomical improvement. There are only two
independent seeds; AD/NC share noise, so these are not eight independent replicates.
Across resolutions noise tensors differ in shape; equal seed is not equal latent.
512 does not eliminate border mismatch in these samples. No optimal CFG, causal
training defect, or need for retraining is established.

Next bounded experiment should hold aligned precision, checkpoint, DDIM30 and
256 resolution fixed, use all32 existing cohort seeds and both labels, and vary
CFG only (e.g.3,5,7.5). Reuse7.5 outputs only after confirming identical pipeline
and noise contracts. Retain every image and report paired changes, including
known failure seeds37220/37217. This is a proposed experiment, not submitted.
No test images used; no new GPU job launched in this analysis.

## 2026-10-07 — CFG sweep submitted

Job636273: a100-test, oneGPU,20-minute limit. Fixed checkpoint,256/DDIM30/eta0,
aligned precision,32 seeds and2 labels, CFG3/5/7.5 (192 outputs). See
cfg-sweep-protocol.md. Regenerate7.5 for subsequent historical parity checking.
Local Ruff/format/syntax checks and remote shell syntax passed. A login-node
--help import stalled in I/O; stopped that exact invocation before submission.
Model import/execution remains to be verified by the scheduled job. No results
or best setting claimed at submission; test data stays sealed.

## 2026-10-07 — CFG sweep acceptance and visual review

Job636273 completed9m06s, exit0:0. Verified all192 PNG hashes and64 matched-noise
triplets. All64 regenerated CFG7.5 PNG hashes equal historical cohort632557.
Scheduler contracts match after sorting only the metadata list of defaulted keys.
Full three64-image grids and six individual F1/F2 outputs were visually reviewed.

| CFG | Mean color | Mean border | Mean edge |
| --- | ---: | ---: | ---: |
| 3 | 0.021165 | 0.204864 | 0.022431 |
| 5 | 0.017498 | 0.109868 | 0.023333 |
| 7.5 | 0.020839 | 0.059221 | 0.022961 |

Paired result:3->5 lowers border in64/64 samples,5->7.5 in63/64. In contrast,
3->5 lowers color in only23/64 samples (11/32 seed blocks); its lower mean is not
majority improvement.5->7.5 increases color in45/64 samples, with23/32 seed blocks
increasing. AD/NC share noise, so block counts avoid pretending64 independent seeds.
No inferential significance test or anatomical-quality ranking is claimed.

F1 AD37220: border0.57695/0.52643/0.41154 at CFG3/5/7.5. Gray background remains
in all three. CFG3 also has visibly textured background; higher guidance suppresses
some texture while leaving the brightness mismatch. F2 NC37217: color range
0.00769/0.00854/0.02104; cyan patch is conspicuous at7.5 and not conspicuous at3/5.
The observed dependence is on guidance within this fixed setup; it does not identify
the learned mechanism or establish that retraining would fix it.

The grids retain similar gross orientations/shapes and paired AD/NC resemblance.
No label-conditioned disease fidelity or mode-collapse conclusion follows from
visual resemblance. Existing F3/F5 remain unconfirmed risk probes.

Decision: retain7.5 as the existing comparison reference, with5 as a documented
tradeoff, not a newly proven optimum. Stop expanding CFG search for now. Prioritize
remaining required evidence: feature-space coverage/label sensitivity, interpolation,
and student check-off/implementation scope. Do not replace formal held-out validation
with selection on these32 diagnostic seeds. Test remains sealed. No new job submitted.

Reproduce with tools/audit_cfg_sweep.py; outputs stay private, aggregate evidence
is in docs/cfg-sweep-result-2026-10-07.json. Ruff passed on the real-data audit tool.

## 2026-10-07 — Interpolation submitted

Job639671 submitted: two noise-seed pairs, five linear mixing coefficients,
two fixed prompts,20 total images; CFG7.5/256/DDIM30 with aligned precision.
See interpolation-protocol.md for lower interior variance and endpoint checks.
Ruff, Python syntax and remote shell syntax passed. No result acceptance yet.
Private HTML updated with completed appearance/CFG evidence; interpolation remains
unverified there. No training or test-set evaluation initiated.

## 2026-10-07 — Initial-noise interpolation accepted

Job639671 COMPLETED1m05s,exit0:0. All20 PNG hashes verified; all eight endpoint
noise hashes and PNG hashes exactly match the accepted historical CFG7.5 cohort.
Complete four paths inspected at alpha0,.25,.5,.75,1. Reproduce integrity checks
with tools/audit_interpolation.py (Ruff passed).

Contours, local texture and intensity visibly vary along the four sampled paths.
Five positions cannot establish continuous perceptual smoothness or anatomical
validity. NC37216->37217 ends with the known cyan artifact; interpolation does not
repair it. Gross orientation remains similar across the displayed paths. This is
not evidence of disease-state interpolation, semantic coverage, or non-memorization.

Endpoint noise norms are63.74-64.51; midpoint norms45.26/45.69. Linear mixing
therefore changes noise magnitude as well as direction. Interior appearance cannot
be attributed to direction alone. AD/NC use the same two noise paths; there are
two independent endpoint pairs, not four independent noise experiments.

The narrow initial-noise interpolation diagnostic is complete; feature-space
coverage, label fidelity, and final test evaluation remain unperformed. Keep these
requirements separate. Next priority is a documented feature-space protocol with
real validation references, rather than further cosmetic CFG search. No new GPU
job submitted in this acceptance step. Student review remains pending.

## 2026-10-07 — Frozen VAE feature diagnostic

Job639765 COMPLETED19s,exit0:0.396 images ->256 features. Source-code hash matches
report; downloaded feature and private-row SHA256 match remote files. Reference
selection and every input image hash verified during extraction. No test images.

| Query/reference | Mean VAE-feature RMSE |
| --- | ---: |
| Validation ->102 matched training references | 0.899889 |
| Validation ->64 label-count-matched training references | 0.953590 |
| Validation ->64 original SD samples | 6.260788 |
| Validation ->64 adapted SD samples | 1.265907 |
| Validation ->64 simple VAE samples | 2.193512 |

This is a finite-sample reverse nearest-distance diagnostic, not a coverage percent.
Adapted SD is closer than the simple VAE by this directional measure but remains
farther than the real-reference context. Forward generated->validation means:
base7.327543, adapted1.926602, baseline1.871396. Forward and reverse directions
answer different questions; do not claim SD wins every measure. Class-stratified
results are preserved in feature-result-2026-10-07.json.

Joint t-SNE (396 points,perplexity30,seed3710,PCA init,1000 iterations) was rendered
and inspected. Baseline forms a separate region; base SD is also separated; adapted
samples overlap parts of the real projection and also extend away from it. These
are visualization observations, not numeric overlap/coverage evidence. AD/NC
colors for generated samples are prompt labels, not verified disease labels.

SD VAE is shared with the generator, trained for compression rather than clinical
assessment; pooled latents retain alignment/background/domain bias. The comparison
is supplementary and cannot establish anatomical quality, privacy, or AD control.
Plot and private feature arrays remain in outputs/features-639765. No patient
identifiers are included in the aggregate report or plot labels.

PDF2.4 checked: initial-noise interpolation alone does not complete the requested
synthetic/training-image interpolation. SSIM with an explicit pairing protocol and
encoded real/synthetic interpolation still remain; final test is sealed. Do not
label the entire evaluation complete. Student interpretation pending.

## 2026-10-07 — Explicit SSIM and encoded real/synthetic paths

Job639777 completed13s,exit0:0; exported204 preprocessed private real references
and two five-frame VAE-code interpolation paths, with original endpoints shown
separately. Extractor code hash verified. SSIM computed locally using pinned
scikit-image0.25.2; actual-reference identity control returned1. Ruff passed.

| Model | Fixed-reference mean SSIM | Feature-nearest mean SSIM | Nearest fraction>0.6 |
| --- | ---: | ---: | ---: |
| Original SD | 0.088068 | 0.087000 | 0/64 |
| Adapted SD | 0.350366 | 0.376754 | 23/64 |
| Simple VAE | 0.672548 | 0.726595 | 64/64 |

Real validation->same-label training feature-neighbor mean0.709904 (102 queries).
Under BOTH stated pairings, adapted SD's mean falls short of the course's0.6
reference target and below the simple VAE. Do not present a pass or silently alter
pairing to obtain a higher score. Native resolution, smoothing, black background,
alignment and arbitrary correspondence limit interpretation; low SD scores and
its visibly sharper outputs must both be reported. Reference selection used VAE
features before SSIM, not SSIM maximization. No ground-truth counterpart exists
for unconditional outputs. Scores are not clinical judgments or privacy guarantees.

Both encoded paths inspected: NC37215 and AD37220 to their same-label feature
nearest TRAIN reference. Intermediate frames show blurred/blended textures, with
structure changing between endpoints; no claim of anatomical continuity. VAE
endpoint generated reconstruction MSE0.000604/0.000546, training reconstruction
MSE0.004103/0.003971. Endpoints therefore reconstruct inputs rather than exactly
reproducing their pixels. This is posterior-mean interpolation in the frozen VAE,
not reverse-diffusion inversion. It does not disprove memorization.

Original/reference images and per-image SSIM stay private. Both interpolation
scope and remaining deficiencies should be discussed at check-off. Next: reconcile
course requirement coverage and prepare the one-page feasibility discussion;
prioritize reviewer-approved changes over open-ended parameter searches. Test
set remains sealed; no new training submitted on the strength of these scores.

## 2026-10-12 — E1 full-UNet fine-tuning: implementation and setup

Protocol fixed before any job ran: [e1-protocol.md](e1-protocol.md), commit de6f30e.
Code: `modules.py`, `train.py`, `predict.py` (commits 824bca7, 9b09bab, d0c2475);
`dataset.py` now verifies SHA-256 per slice when asked (693df5a). 17 CPU unit tests
pass with tiny random models, including equality of the guided DDIM sampler with
diffusers' StableDiffusionPipeline latents. These test mechanics, not image quality.

Remote run directory `~/comp3710-sd-audit/runs/e1-20261012`, symlinking the shared
`.venv-train`, `data` and `models`; `source-sha256.txt` records commit de6f30e and
the hash of each copied file, all equal to the committed blobs.

CPU job 651837 added matplotlib 3.10.9 to `.venv-train` for `train.py`'s loss
curves: COMPLETED, 2 min 07 s, exit 0:0, `pip check` clean, environment frozen to
`logs/e1_environment_651837.txt`. Diffusers 0.40.0 and PEFT 0.21.2 unchanged.

### S1 probe, job 651838

`a100-test`, node a100-a, NVIDIA A100-PCIE-40GB: COMPLETED in 3 min 05 s, exit 0:0.
200 full-UNet steps at batch 16 on the first 3,200 training slices (859,520,964
trainable parameters), 256 fixed validation slices. Steady throughput 94 images/s
after the first window (28 images/s while cuDNN benchmarked); peak PyTorch
allocation 23.24 GiB. Encoding 3,456 slices with SHA-256 checks took 35 s.
Validation loss after 200 steps: 0.0701 live, 0.0704 EMA; training-window loss
fell from 0.0847 (steps 1-50) to 0.0681 (steps 151-200), still in warm-up (lr 4e-6).

The 1.72 GB float16 EMA file was written to job scratch and reloaded by
`predict.py` for two seeds per class. Its printed statistics (channel spread
0.005, border mean 0.0004-0.0006) come from four images after 200 steps and are a
pipeline check only, not an E1 result. Visual check of `samples/epoch-01.png`:
black background and grayscale brain structure, while the pre-training grid
(`epoch-00.png`, base SD at CFG 7.5) shows colourful non-MRI images. The eight
probe samples share a similar pose, so the S3 audit must examine diversity.

At 94 images/s the protocol's 10 epochs need about 38 min of training; the
pre-registered epoch count is kept rather than extended. Main run: job 651851.

### S2 scheduling change

Job 651851 (4 h request, 200 min budget, as in the protocol) waited on priority
with an estimated start of 22:08. With S1's measured throughput the run needs
about 50 min including setup, validation and sampling, so it was cancelled
before starting and resubmitted unchanged except for scheduling: job 651860,
`--time=01:30:00`, `--time-budget-minutes 75`. The training recipe, epochs and
acceptance criteria are as recorded in e1-protocol.md.

### S2 main run, job 651860

`comp3710`, node a100-6, NVIDIA A100-PCIE-40GB: started 2026-10-12 03:36,
COMPLETED in 45 min 29 s, exit 0:0, stop reason `completed`. 10 epochs x 1,330
steps = 13,300 steps at batch 16 over all 21,280 training slices, every slice
hash-checked during encoding (133 s). 93.1 images/s over 2,286 s of training;
peak PyTorch allocation 23.82 GiB. EMA weights saved at epochs 5 and 10
(1.72 GB each, `runs/e1-20261012/outputs/e1-651860/weights`); home then had
5.0 GB free.

| Epoch | Train loss | Val loss, live | Val loss, EMA |
| ---: | ---: | ---: | ---: |
| 1 | 0.06282 | 0.05820 | 0.05755 |
| 2 | 0.05834 | 0.05675 | 0.05619 |
| 3 | 0.05579 | 0.05633 | 0.05552 |
| 5 | 0.05500 | 0.05555 | 0.05487 |
| 7 | 0.05349 | 0.05537 | 0.05458 |
| 9 | 0.05206 | 0.05534 | 0.05452 |
| 10 | 0.05264 | 0.05530 | 0.05457 |

Validation is 1,024 fixed slices with fixed noise and timesteps, plain noise
and true prompts. EMA validation loss falls to a minimum at epoch 9 and rises
once at epoch 10 (+0.00006), so the two-consecutive-rise stop condition was not
met and epoch 10 remains the primary weights. From epoch 7 the training loss
keeps falling while validation loss is flat: a widening gap that makes the
memorisation audit in S3 more important, not less.

Visual review of the per-epoch fixed-seed grids (8 images each, CFG 7.5):
from epoch 1 every sample has a black background and grayscale brain anatomy
in the same orientation as the real slices. Remaining defects at epoch 10:
frayed, speckled brain outlines; faint blue tints along some ventricle and
cortex edges; and one dark, low-contrast, ghosted sample (NC seed 37101) that
looked normal at epoch 5. Eight images are not the evaluation; S3 is.

## 2026-10-13 — E1 S3 evaluation against the pre-registered criteria

Jobs, all COMPLETED with exit 0:0: cohorts 656354 (epoch 10, primary) and 656355
(epoch 5, exploratory, added at the student's request after training), 2 min 54 s
and 2 min 43 s on a100-b; frozen-VAE features 656360 (23 s); pixel nearest
neighbours and appearance 656361 (1 min 09 s); closest-pair panels 656362;
real-slice foreground census 656380 (34 s). All 128 generated PNGs matched their
recorded SHA-256 after download. SSIM and t-SNE ran locally (scikit-image 0.25.2);
the historical rows reproduced the 7 October values exactly. Aggregates:
`e1-ssim-result-2026-10-13.json`, `e1-appearance-2026-10-13.json`,
`e1-neighbours-2026-10-13.json`, `e1-census-2026-10-13.json`.

| Criterion (validation data) | E0 | E1 epoch 10 | E1 epoch 5 (exploratory) | Target | Epoch 10 |
| --- | ---: | ---: | ---: | --- | --- |
| Feature-nearest SSIM, mean | 0.377 | 0.688 | 0.674 | >= 0.6 | met |
| Fixed-reference SSIM, mean | 0.350 | 0.641 | 0.625 | reported | — |
| Nearest SSIM > 0.6 | 23/64 | 60/64 | 59/64 | — | — |
| Border intensity, mean | 0.0592 | 0.0016 | 0.0047 | <= 0.01 | met |
| Colour range, mean | 0.0208 | 0.0061 | 0.0062 | < 0.0208 | met |
| Generated-to-training nearest RMSE, mean | 0.1122 | 0.0655 | 0.0700 | >= 0.0731, no near-copy | **not met** |

Other context: simple VAE nearest SSIM 0.727; real validation-to-training 0.710.
Edge measure 0.0135 (epoch 10) against real 0.0176 and VAE 0.0076: E1 is slightly
smoother than real slices, much less than the VAE. Nearest SSIM by class at epoch
10: AD 0.709, NC 0.666.

**Memorisation finding.** The three closest epoch-10 pairs (RMSE 0.0016-0.0030)
are near-black generated images matched to near-black training slices; the
private panels show no brain in either. Seven of 32 epoch-10 NC images (2 of 32
at epoch 5, 0 of 32 AD at either epoch, 0 in E0) have under 5% foreground, and
all seven nearest training slices belong to one scan of one subject. The census
finds near-empty slices in 0.26% of real NC training slices (about 29 of
11,120), none in AD training and none in validation, spread over every slice
position, so they are a defective scan rather than edge slices. Full
fine-tuning amplifies that rare training item about 85-fold by epoch 10, more
than at epoch 5. This is the fidelity-versus-memorisation trade-off of section
2.4 observed directly: longer training improves SSIM and background while
reproducing a single training scan more often.

Post-hoc and exploratory only, not a substitute for the criterion: excluding the
near-empty images, nearest-training RMSE averages 0.0712 at both epochs (minimum
0.0445 at epoch 10), against real validation-to-training 0.0731 (minimum 0.0484)
and validation-full 0.0771. Close to, and slightly below, the real reference;
pixel RMSE alone neither shows nor rules out anatomical copying. Unequal
reference-set sizes also bias the full-set distances.

Joint t-SNE of 524 frozen-VAE features (102 train + 102 validation patients,
64 per generated cohort): both E1 checkpoints fall inside the real cloud, while
original SD, E0 and the VAE baseline form separate clusters; a small NC cluster
of E1 points matches the near-empty images. Descriptive only; the shared VAE is
not an independent evaluator.

Remaining visible defects at epoch 10: faint blue tints on some ventricle edges;
frayed outlines; seven empty or ghosted NC images. Test split still sealed.

## 2026-10-13 — E2: E1 without two defective training scans

Protocol approved and committed before any E2 job: [e2-protocol.md](e2-protocol.md),
commit 283a9e2. Rule check before the protocol: CPU job 656452 (45 s, exit 0:0)
applied `dataset.defective_scans` to train and validation at 256 px. Training: 30
near-empty slices in 2 NC scans of 2 subjects, 40 slices flagged (19 and 11
near-empty slices per scan); validation: none
([aggregate](e2-flag-2026-10-13.json)). Private panels of all flagged slices,
beside a typical NC scan, show one scan blank apart from fragments and one dark,
grainy ghost outline: failed acquisition or preprocessing.

Code: `train.py --exclude-defective-scans --expect-excluded 2 40` (19 CPU tests
pass). Commit 3aaf187 shipped two wrappers whose line continuation was a literal
backslash-n that would have reached `train.py` as an argument; it was caught
before any submission and fixed in a64e1b9, after a dry run with stub commands
parsed both wrappers' arguments with the real parser. Remote run directory
`~/comp3710-sd-audit/runs/e2-20261013`, source hashes equal to commit a64e1b9.

Storage, as approved: both E1 weight files were copied to the local
`outputs/e1-651860/weights` with matching SHA-256, then the E1 epoch-5 file was
deleted from Rangpur after a checksum match (6.56 GiB free afterwards). Its
cohort 656355 had already been generated and evaluated.

Jobs: probe 656466; main run 656467 (afterok on the probe); cohorts 656472
(epoch 10) and 656473 (epoch 5); features 656474 over historical, E1 and E2
cohorts; CPU audits and closest-pair panels 656475.

### E2 S1 probe, job 656466

`a100-test`, a100-a: COMPLETED in 4 min 25 s, exit 0:0. The filter on the full
training split removed exactly 2 scans and 40 slices, as the protocol expects.
200 steps at 93 images/s; peak allocation 23.80 GiB; EMA weights saved to job
scratch and reloaded by `predict.py`. Main run 656467 then waited on
`comp3710` resources (at the front of the queue on 14 October 02:34, estimated
start 03:44).

### E2 S2 and S3, jobs 656467 and 656472-656475

Main run 656467: `comp3710`, a100-7, started 2026-10-14 03:40, COMPLETED in
45 min 26 s, exit 0:0. Excluded 2 scans / 40 slices as expected; 21,240 training
slices, 13,270 steps, 93.2 images/s, peak 23.82 GiB. EMA validation loss tracks
E1 within 0.0001 at every epoch (epoch 10: 0.05455 vs 0.05457); minimum at epoch
9 again with a single rise at epoch 10, so the stop condition was not met.

Evaluation, all COMPLETED with exit 0:0: cohorts 656472 (epoch 10) and 656473
(epoch 5), features 656474 over historical, E1 and E2 cohorts, CPU audits and
panels 656475. All 128 E2 PNGs matched their SHA-256 after download. SSIM and
t-SNE ran locally; every historical and E1 row reproduced its earlier value.
Aggregates: `e2-ssim-result-2026-10-14.json`, `e2-appearance-2026-10-14.json`,
`e2-neighbours-2026-10-14.json`, `e2-features-2026-10-14.json`.

| Criterion (validation data) | E1 epoch 10 | E2 epoch 10 | E2 epoch 5 (secondary) | E2 target | E2 epoch 10 |
| --- | ---: | ---: | ---: | --- | --- |
| NC samples under 5% foreground (H1) | 7/32 | **0/32** | 0/32 | <= 1/32 | met |
| AD samples under 5% foreground | 0/32 | 0/32 | 0/32 | 0/32 | met |
| Feature-nearest SSIM, mean | 0.688 | **0.705** | 0.683 | >= 0.6 | met |
| Nearest SSIM > 0.6 | 60/64 | 64/64 | 62/64 | — | — |
| Fixed-reference SSIM, mean | 0.641 | 0.654 | 0.634 | reported | — |
| Border intensity, mean | 0.0016 | 0.00007 | 0.0007 | <= 0.01 | met |
| Colour range, mean | 0.0061 | 0.0059 | 0.0062 | < 0.0208 | met |
| Generated-to-training nearest RMSE, mean (H2) | 0.0655 | 0.0673 | 0.0726 | >= 0.0731, no near-copy | **not met** |

Nearest SSIM by class at epoch 10: AD 0.714, NC 0.697 (E1: 0.709, 0.666); the
lowest single value rose from 0.402 to 0.645. Edge measure 0.0134 against real
0.0176. Real validation-to-training SSIM reference: 0.710.

**H1 holds.** Removing the two defective NC scans removed every near-empty and
ghosted sample at both epochs, so E1's blank outputs were learned from those
scans. **H3 holds.** No fidelity criterion regressed; SSIM improved, mostly for NC.

**H2, as predicted in the protocol, is not met by the pre-registered measure:**
mean generated-to-training RMSE is 0.0673 against 0.0731. Visual review of the
five closest pairs at each epoch (RMSE 0.040-0.052) found no near-copy: outline
and pose are similar, but gyral folding, cerebellar folia and ventricle shape
differ; at epoch 10, two generated images share the same nearest training slice.
The near-identical blank pairs of E1 (RMSE 0.0016) are gone; the closest pair is
now 0.0425 against a real validation-to-training minimum of 0.0484.

Post-hoc context, not a substitute for the criterion: with equal-size
references of one slice for each of 102 patients, epoch-10 samples are no closer
to training patients (0.0920) than to validation patients (0.0897); real
validation slices give 0.0918 to the same training references. The full-set
measure is lower partly because the training reference set is 4.7 times larger
than the validation set (21,240 vs 4,540 slices), and because many generated
slices are dimmer and lower in contrast than real ones, which shortens pixel
distances to dim real slices. A final protocol should use a size-matched
comparison; pixel distance cannot prove the absence of memorisation either way.

Remaining visible defects at epoch 10: faint blue tints on some ventricle edges,
a few dim or low-contrast slices, frayed outlines. Test split still sealed.

## 2026-10-14 — E3: E2's recipe with E0's rank-4 adapter

Protocol approved and committed before any E3 job: [e3-protocol.md](e3-protocol.md),
commit e8d6a5a. Wrappers (commit 24ca5ab) written directly rather than derived
with sed, dry-run with stub commands, and their train.py arguments checked with
the real parser (`--trainable lora --lora-rank 4 --lr 1e-4`, exclusion of 2 scans /
40 slices). Remote run directory `~/comp3710-sd-audit/runs/e3-20261015`, source
hashes equal to commit 24ca5ab.

Jobs submitted as one dependency chain: probe 660841; main run 660842; cohorts
660843 (epoch 10) and 660844 (epoch 5); features 660845 over the historical, E1,
E2 and E3 cohorts (9 generated groups); CPU audits and closest-pair panels 660846.

## 2026-10-15 — B2 and B3 submitted

Protocols committed before any job: [b2-gan-protocol.md](b2-gan-protocol.md) and
[b3-ldm-scratch-protocol.md](b3-ldm-scratch-protocol.md) (6fe8a54). While porting
the demonstration-2 GAN, the B2 protocol was found to misdescribe that work's
diversity ratio; a dated erratum aligning it with the measure its thresholds
were calibrated on was committed before implementation (babe480).

Code: `tools/baseline_gan.py` (72ceecd), `--unet-init scratch` (cb1daaa), wrappers
(0c90a65); 27 CPU tests pass, and a CPU smoke run of the GAN on synthetic slices
excluded the planted defective scan, trained, snapshotted and generated a cohort.
Wrappers were written with the file tool; shell heredocs had double-escaped
backslashes, which is also how E2's earlier wrapper bug arose. Every wrapper's
commands were parsed with the real argument parsers before submission. Run
directories `runs/b2-20261015` and `runs/b3-20261015` match commit 0c90a65.

Jobs: B2 probe 663301, B2 training 663302 (afterok on the probe); B3 probe 663303.
B3 training is submitted after its probe, with the epoch count set by the
protocol's rule from the measured throughput.

E3 (660841-660846) completed with exit 0:0 overnight; evaluation pending. Its probe
reported a peak allocation of 36.8 GiB, above full fine-tuning's 23.8 GiB, to be
examined with the results.

## 2026-10-17 — B2 trained; B3 epoch count from its probe

**B2 probe 663301** (a100-test, 4 min 06 s, exit 0:0): excluded 2 scans / 40
slices; 500 steps; both diversity checks healthy (0.92, 0.83); snapshot and
cohort generation worked.

**B2 training 663302** (comp3710, a100-7, 1 h 45 min 23 s, exit 0:0). Deviation
from the protocol: it stopped at **29,000 of the planned 30,000 steps**, by the
105-minute safety budget in the wrapper. The budget was set from
demonstration 2's speed; label conditioning and an R1 penalty on every step ran
at 296 images/s instead. The collapse rule never fired (lowest ratio after step
5,000 was 0.65), so the result is the EMA generator at step 29,000, as the
protocol prescribes without collapse. Generator 9.17 M and discriminator
12.51 M parameters; peak allocation 3.79 GiB.

**B3 probe 663303** (a100-test, 5 min 17 s, exit 0:0): 300 steps from scratch,
steady 281.9 images/s at batch 64, peak allocation 34.27 GiB; save and reload
worked. Step-count rule, applied before submitting the main run: an epoch is
331 steps (75.2 s of training at the measured rate) plus about 16 s of
validation and sampling (14.1 s measured with 256 validation slices, plus the
cost of 1,024), so about 91.3 s. 121 epochs would need about 184 minutes; the
largest whole number that fits 150 minutes is **98 epochs (32,438 steps)**, with
EMA weights at epochs 49 and 98.

## 2026-10-17 — E3 results: E2's recipe with E0's rank-4 adapter

Jobs, all COMPLETED with exit 0:0 on 2026-10-15: probe 660841 (excluded 2 scans /
40 slices); main run 660842 (a100, 37 min 52 s wall, 13,270 steps, 105 images/s,
797,184 trainable parameters, adapter files 1.6 MB each); cohorts 660843 (epoch
10) and 660844 (epoch 5); features 660845; audits and panels 660846. All 128 PNGs
matched their SHA-256. SSIM and t-SNE ran locally; all earlier rows reproduced.
Aggregates: `e3-ssim-result-2026-10-17.json`, `e3-appearance-2026-10-17.json`,
`e3-neighbours-2026-10-17.json`, `e3-features-2026-10-17.json`.

| Measure (validation) | E0 | **E3 epoch 10** | E3 epoch 5 | E2 epoch 10 |
| --- | ---: | ---: | ---: | ---: |
| Trainable parameters | 797,184 | 797,184 | — | 859.5 M |
| Feature-nearest SSIM, mean | 0.377 | **0.696** | 0.688 | 0.705 |
| Nearest SSIM > 0.6 | 23/64 | 63/64 | 60/64 | 64/64 |
| Fixed-reference SSIM, mean | 0.350 | 0.652 | 0.642 | 0.654 |
| Border intensity, mean | 0.0592 | 0.00038 | 0.0012 | 0.00007 |
| Colour range, mean | 0.0208 | 0.0061 | 0.0061 | 0.0059 |
| Edge measure (real 0.0176) | 0.0230 | 0.0177 | 0.0199 | 0.0134 |
| Near-empty samples AD / NC | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Nearest-training RMSE, mean (min) | 0.1122 | **0.0846** (0.0621) | 0.0870 (0.0593) | 0.0673 (0.0425) |
| Size-matched nearest RMSE, train / validation | — | 0.1035 / 0.1002 | 0.1037 / 0.1032 | 0.0920 / 0.0897 |
| EMA validation loss, epoch 10 | — | 0.0583 | — | 0.0546 |
| Training peak allocation | 5.69 GiB | 36.82 GiB | — | 23.82 GiB |

**H1 (recipe) holds.** With capacity fixed at E0's adapter, E2's recipe alone
removes E0's grey background (border 0.059 to 0.0004) and colour artefacts
(0.021 to 0.006), and raises SSIM from 0.377 to 0.696.

**H2 (capacity) is not supported.** The protocol predicted E3 at least 0.05 SSIM
below E2; the gap is 0.009 (0.696 vs 0.705). Of the E0-to-E2 gain of 0.328, the
recipe accounts for 0.319 and the extra 859 M trainable parameters for 0.009.
Full fine-tuning does fit the denoising objective better (validation loss 0.0546
vs 0.0583) and gives smoother images (edge 0.013 vs 0.018, real 0.018); E3's edge
measure matches real slices, though some E3 samples look over-contrasted on
visual review.

**H3 (memorisation) supported by the pixel measures.** E3 meets the
memorisation criterion E1 and E2 failed: mean nearest-training RMSE 0.0846 against
the 0.0731 reference, closest pair 0.0621 against the real-to-real minimum
0.0484. Size-matched references show no training preference (0.1035 vs 0.1002).
The five closest epoch-10 pairs differ in gyral folding, cerebellum and
ventricle shape on visual review. The membership test in the final protocol
will check this with a model-based measure.

**Cost observation.** Despite training 0.09% of the parameters, E3's peak
allocation (36.8 GiB) exceeded full fine-tuning's (23.8 GiB). Optimizer state
and the EMA copy shrink to almost nothing, so the extra memory is in
activations; the likely cause, not yet verified, is that PEFT's adapter layers
keep float32 copies of their inputs under bfloat16 autocast. The practical
savings are a 1.6 MB weight file instead of 1.7 GB, and 13% faster steps.

**Implication for the recommendation.** At this data scale, almost all of the
fidelity comes from the training recipe and the data cleaning, not from
fine-tuning every weight; the small adapter reaches nearly the same SSIM with a
weaker memorisation signal. This is reported as found; it argues for the adapter
as the deployment candidate, pending the final test protocol.

## 2026-10-17 — B2 results: conditional mode collapse

Evaluation jobs, COMPLETED with exit 0:0: cohort 668935 from the step-29,000 EMA
generator, CPU audits and panels 668936, features 668943 (historical, E1-E3 and
B2 cohorts). SSIM ran locally; earlier rows reproduced. Aggregates:
`b2-ssim-result-2026-10-17.json`, `b2-appearance-2026-10-17.json`,
`b2-neighbours-2026-10-17.json`.

| Measure (validation) | B2 GAN | B1 VAE | E2 | Real reference |
| --- | ---: | ---: | ---: | ---: |
| Feature-nearest SSIM, mean | 0.581 | 0.727 | 0.705 | 0.710 |
| Nearest SSIM > 0.6 | 32/64 | 64/64 | 64/64 | — |
| Fixed-reference SSIM, mean | 0.556 | 0.673 | 0.654 | — |
| Edge measure | 0.0231 | 0.0076 | 0.0134 | 0.0176 |
| Border intensity | 0.0492 | 0.00008 | 0.00007 | 0 |
| Within-class pairwise RMSE, AD / NC | 0.072 / 0.063 | — | — | about 0.15-0.16 overall |
| Nearest-training RMSE, mean | 0.1254 | 0.0565 | 0.0673 | 0.0731 |

**Visual review.** All 32 AD samples are nearly the same image, and so are all 32
NC samples; neither template is a plausible brain (smooth blob outlines,
grid-like texture, vertical artefacts at the image edges, hence the border
value). The progress grids show the same thing from the first check at step
1,000: within each class the eight fixed-noise samples are identical, and the
per-class template changes shape over training without becoming anatomical.
The generator ignored its noise input and depended only on the label: the
conditional mode collapse described by Mao et al. (Mode Seeking GANs, CVPR 2019).

**Hypotheses.** H2 (narrower coverage than E2) holds in the strongest form: the
collapse the protocol named as the expected failure. H1 (sharper than the VAE)
holds numerically (edge 0.0231 vs 0.0076; real 0.0176), but the sharp structure
is not anatomy. H3: SSIM 0.581 is reported without a target.

**Why the pre-registered collapse rule did not fire.** Its diversity ratio pooled
AD and NC samples (alternating labels). The two class templates differ strongly,
so the pooled ratio stayed between 0.65 and 1.92 while each class had a single
output. The rule was carried over from demonstration 2's unconditional GAN and
was not adapted to conditioning; the per-class measure needed is noted for any
future run. The minibatch standard-deviation feature also mixes classes within a
batch, which plausibly weakened its pressure against collapse; not verified.

**Lesson for the metrics.** A completely collapsed generator still scores mean
SSIM 0.58, with half its samples above the course's 0.6 target. SSIM measures
resemblance to some real slice, not coverage of the real distribution; coverage
needs a separate measure, which the final protocol adds.

## 2026-10-17 — B2b submitted

Protocol committed before any job: [b2b-gan-per-class-protocol.md](b2b-gan-per-class-protocol.md)
(6ee115f). Code: `--train-class` and per-class generation (73c0453; 28 CPU tests
pass; a CPU smoke run trained an AD-only GAN after the whole-split exclusion and
generated a cohort from two per-class checkpoints). Wrappers d33bf84, dry-run and
parsed. Run directory `runs/b2b-20261017` matches commit d33bf84.

Jobs: probe 669123; AD GAN 669124 and NC GAN 669125 (afterok on the probe; the
course QoS runs them one at a time, after B3's 668934); cohort 669126 (afterok on
both); CPU audits and panels 669127.

## 2026-10-17 — R1 submitted; B3 evaluation queued; B2b probe

**R1.** Protocol committed before any job (7745101). Code: `--val-seed` separates the
validation draws from the training seed (bcc96f1; a test with lr 0 confirms the
training seed no longer moves validation loss; 29 tests pass). Wrappers 048d3f1,
dry-run and parsed; the empty-array expansion was checked against the cluster's
bash 4.4. Run directory `runs/r1-20261017` matches commit 048d3f1. Jobs: R1-E2
669198, R1-E3 669199; cohorts 669200 and 669201; features 669202 (historical, E2,
E3, R1 cohorts); audits 669203.

**B3.** Evaluation queued behind training 668934, as the protocol specifies:
cohorts from epoch 98 at guidance 7.5 (669204, primary), epoch 49 at 7.5 (669205)
and epoch 98 at 3.0 (669206); features 669207; audits 669208. Observed while
running: at epoch 93, training loss 0.0419 against validation loss 0.0835, a far
larger gap than E2's; and an epoch takes 65.9 s, not the 91.3 s estimated from
the probe, so in hindsight about 121 epochs would have fitted the 150-minute
budget. The rule was applied as written from the probe's measurements.

**B2b probe 669123** (a100-test, 3 min 48 s, exit 0:0): excluded 2 scans / 40
slices, then trained on the 10,160 AD slices only; per-class generation path
worked. Diversity ratio 0.90 at step 250 and 0.29 at step 500. Demonstration 2
saw an early dip in every run, which is why the collapse rule starts at step
5,000; the main runs will show whether it recovers.

## 2026-10-18 — B3 results: matches E2's fidelity, with a clear memorisation signal

Jobs, all COMPLETED with exit 0:0: training 668934 (comp3710, A100, 1 h 52 min
wall; 98 epochs, 32,438 steps at batch 64, 401 images/s, 86 min of training,
peak allocation 34.29 GiB; 92,235,396 parameters from random initialisation);
cohorts 669204 (epoch 98, guidance 7.5, primary), 669205 (epoch 49) and 669206
(epoch 98, guidance 3.0); features 669207; audits and panels 669208. All 192
PNGs matched their SHA-256. SSIM and t-SNE ran locally; the earlier rows
reproduced. Aggregates: `b3-ssim-result-2026-10-18.json`,
`b3-appearance-2026-10-18.json`, `b3-neighbours-2026-10-18.json`,
`b3-features-2026-10-18.json`.

| Measure (validation) | **B3 epoch 98** | B3 epoch 49 | B3 epoch 98, guidance 3.0 | E2 epoch 10 | E3 epoch 10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| UNet parameters / initialisation | 92.2 M / random | — | — | 859.5 M / SD1.5 | adapter / SD1.5 |
| Feature-nearest SSIM, mean | **0.710** | 0.711 | 0.716 | 0.705 | 0.696 |
| Nearest SSIM > 0.6 | 64/64 | 64/64 | 64/64 | 64/64 | 63/64 |
| Fixed-reference SSIM, mean | 0.658 | 0.650 | 0.660 | 0.654 | 0.652 |
| Border intensity, mean | 0.0005 | 0.0003 | 0.00001 | 0.00007 | 0.00038 |
| Colour range, mean | 0.0056 | 0.0061 | 0.0058 | 0.0059 | 0.0061 |
| Edge measure (real 0.0176) | 0.0128 | 0.0133 | 0.0116 | 0.0134 | 0.0177 |
| Near-empty samples AD / NC | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Pairwise RMSE between samples | 0.137 | 0.141 | 0.122 | 0.135 | 0.155 |
| Nearest-training RMSE, mean (min) | **0.0516** (0.0212) | 0.0631 (0.0385) | 0.0487 (0.0181) | 0.0673 (0.0425) | 0.0846 (0.0621) |
| Nearest-validation RMSE, mean | 0.0726 | 0.0757 | 0.0661 | 0.0738 | 0.0886 |
| Nearest training / nearest validation | **0.71** | 0.83 | 0.74 | 0.91 | 0.95 |
| Size-matched nearest RMSE, train / validation | 0.0919 / 0.0873 | 0.0928 / 0.0914 | 0.0882 / 0.0810 | 0.0920 / 0.0897 | 0.1035 / 0.1002 |
| EMA validation loss at the reported epoch | 0.0838 | 0.0694 | — | 0.0546 | 0.0583 |

The real references are unchanged: feature-nearest SSIM 0.710 between real
slices, and 0.0731 mean nearest-training RMSE for real validation slices
(minimum 0.0484).

**Loss curves.** B3's EMA validation loss reached its minimum, 0.0632, at
epoch 23, then rose in every one of the remaining 75 epochs, to 0.0838 at epoch
98, while training loss fell from 0.060 to 0.041. E2's validation loss fell
until epoch 9 and was flat at epoch 10 (0.0545, then 0.0546).
The protocol fixed the saved epochs at 49 and 98, so no checkpoint near epoch 23
exists.

**H1 (pretraining helps fidelity) is not supported.** B3 matches E2 on every
fidelity measure at a ninth of the size and with no pretraining: SSIM 0.710
against 0.705, the same edge, border and colour values, no blank samples, and
both cohorts inside the real cloud in the t-SNE plot. On visual review of the
uncurated 64-image grid, beside E2's, there is no systematic anatomical
difference at grid scale.

**H2 (memorisation) is supported by the full-training nearest-neighbour
measure.** B3's mean nearest-training RMSE is 0.0516, well below the 0.0731
real reference and E2's 0.0673; its closest pair (0.0212) is less than half the
closest real-to-real distance (0.0484). Every model is measured against the same
two reference sets, so the ratio of nearest-training to nearest-validation
distance is comparable between models: B3's samples are 29% closer to their
nearest training slice than to their nearest validation slice, against 9% for
E2 and 5% for E3. The signal grows with training (17% at epoch 49, 29% at epoch
98), in step with the rising validation loss. The five closest epoch-98 pairs,
all NC, reproduce their training slice's outline, ventricle shape and
cerebellum position, with smeared or noisy gyri: replication of a slice's layout,
not a pixel copy (the kind Somepalli et al. describe).

The size-matched comparison (102 references per side) and the patient-matched
VAE feature distances show no preference for training patients. A likely
reason, not verified: a 102-slice subset rarely contains the particular
training slice a sample replicates, so these tests have little power against
slice-level replication. The membership test in the final protocol is
model-based and does not depend on reference subsets.

**Guidance 3.0** gives the highest SSIM of any model (0.716) and an even
stronger memorisation signal (0.0487); it is reported, not chosen.

**Implication.** SSIM rated the from-scratch model as highly as the pretrained
one; only the memorisation measures tell them apart. With B2, where SSIM missed
a complete collapse, this is the second case where SSIM alone would mislead.
At this data scale, SD's pretraining is not needed for fidelity; what it adds is
generalisation: E2 and E3 reach the same fidelity without B3's replication
signal. Not shown: one seed, one architecture, no early-stopped checkpoint.

## 2026-10-18 — B2b results: both per-class GANs failed; a calibration bug

**AD GAN 669124** (a100-7, 36 min 23 s, exit 1:0). No diversity ratio reached
0.8 from step 5,000 on (0.62, 0.19, 0.76, 0.75, 0.44, 0.44 at steps 5,000 to
10,000). The rule stopped it at step 10,000 after two ratios below 0.6, with no
snapshot, so there is no AD result: the script exited with "Collapsed before any
healthy snapshot; no result to keep", as designed.

**NC GAN 669125** (a100-7, 25 min 40 s, exit 0:0). Ratio 0.81 at step 5,000
(snapshot), then 0.34 and 0.40; stopped at step 7,000. The result is the EMA
generator at step 5,000. Peak allocation 3.16 GiB.

The cohort (669126) and audit (669127) jobs depended on both GANs and were
cancelled. With no AD generator, the protocol's 64-image cohort cannot be made.

**Visual review.** In both classes, at every check from step 1,000, the 16
fixed-noise samples are variations of one template that is not a brain: grey
fields, smooth blob outlines, no gyri, ventricles or cerebellum. The template
changes from check to check. This is B2's failure, now within a single class.

**A bug in the per-class code, found during this review (Claude's, commit
73c0453).** Before every sample grid, diversity check and saved checkpoint, the
generator's BatchNorm statistics are recalibrated by running it on random noise.
That step still drew random AD/NC labels. A per-class GAN is trained on one
label only, so half the calibration batches used a label embedding that was
never trained, and the statistics were distorted. Training itself was not
affected (BatchNorm uses batch statistics there). CPU diagnostic on the NC
step-5,000 weights, same 64 noise vectors:

| NC samples, step 5,000 | Mean pairwise RMSE (64 px) |
| --- | ---: |
| Saved statistics (mixed labels) | 0.136 |
| Recalibrated with NC only | 0.166 |
| Real slices, for reference | about 0.15 |

The bug made every logged ratio too low (by about a fifth for NC; AD cannot be
checked, as no AD weights were kept) and so may have changed the stop decisions.
With ratios a fifth higher, the AD run might have kept a snapshot around steps
7,000 to 8,000; this is an estimate, not a measurement. It does not explain the
failure: correctly calibrated NC samples are just as non-anatomical, only more
varied in position and brightness. B2 was not affected (its batches mixed both
labels, matching its calibration). Fixed in 67d4057, with a test that
fails on the old behaviour.

**Lesson for the metrics.** The correctly calibrated NC generator matches real
slices' pixel diversity (0.166 against about 0.15) while drawing one
non-anatomical template. Pixel diversity, like SSIM in B2, can pass a failed
generator; coverage needs a feature-based measure, which the final protocol adds.

**Rerun.** The protocol allows one rerun, for infrastructure failure only. The
bug corrupted the measurements the stop rule acted on, so a rerun with the fix is
arguably allowed; it would cost about 1.5 A100-hours per class in the
one-job course queue. The decision is the student's and is recorded below when made.

## 2026-10-18 — R1: E2 replicate trained; E3 replicate resubmitted

**R1-E2 669198** (comp3710, 44 min 44 s, exit 0:0): seed 4710, validation seed
3710, excluded 2 scans / 40 slices, 13,270 steps, 92.8 images/s, peak 23.82
GiB; EMA validation loss at epoch 10 0.05458 (E2 with seed 3710: 0.05455).
Cohort 669200 completed.

**R1-E3 669199** failed before training: the wrapper required 2 GiB free for
either model, and home had 1.1 GiB after R1-E2's 1.7 GB checkpoint. E3 writes a
1.6 MB adapter. The check is now sized per model (f90dd4f: 2 GiB for E2, 200 MiB
for E3); the recipe and arguments are unchanged, as the dry run confirmed. This
is the protocol's one infrastructure rerun, resubmitted as 672769, with cohort
672770, features 672771 (historical E2 and E3 cohorts, both R1 cohorts) and
audits 672772. The original dependants 669201-669203 could never start and were
cancelled.

**Rerun decision (2026-10-18).** The student approved the rerun. It is the
protocol's one rerun: S2 (both classes) and S3, with the calibration fix from
67d4057 and nothing else changed. The probe is not repeated; a CPU smoke run of
the fixed single-class path (20 steps, two collapse checks, checkpoint written)
passed. S3 now includes a frozen-VAE feature job, which the first chain lacked
and the SSIM audit needs.

## 2026-10-18 — Final evaluation: protocol, Stage 0 code, rehearsal submitted

**Protocol.** [final-evaluation-protocol.md](final-evaluation-protocol.md),
approved by the student and committed (85abb7b) before any code read the test
split. The student also agreed to the blinded sort and approved deleting the
E1 epoch-10 weights from Rangpur: the file's SHA-256 matched the local backup
immediately before deletion, and home went from 1.1 GB to 2.7 GB free.

**Stage 0 code** (dd872d7): batched DDIM in `modules.guided_ddim`,
`ddim_invert` and `slerp`; `predict.py --batch-size`; `export_e0_adapter.py`
(E0's adapter copied unchanged, in float32); `sample_vae.py`; per-class grids
limited to the first 32 seeds; GPU jobs `final_real_features.py`,
`final_cohort_features.py`, `membership.py`, `profile_sampling.py`,
`interpolate_final.py`; local `final_metrics.py`, `final_ssim.py`,
`final_tsne.py`, `blind_sort.py`; wrappers `final_*.sbatch` with one weights
table (`final_models.sh`). Tests: 51 in the training environment and 18 in the
analysis environment, each skipping what needs the other's libraries; the
inversion test fails on two deliberate mutations. All wrappers were dry-run
with stub commands, every tool imported, and the three local tools ran end to
end on fake features in the jobs' file layout. Inception-v3 weights
(torchvision ImageNet, SHA-256 0cc3c7bd75…) were downloaded on the login node
with torchvision's hash check. Run directory `runs/final-20261019`: 24 files
match dd872d7.

**Known deviation, recorded before the rehearsal ran.** The protocol asks
batched generation to reproduce the existing 64 images. That applies to E2, E3,
R1-E2, B3 (the same `predict.py`, batch 1 before), B1 and B2 (unchanged
generators). SD and E0 are now sampled with `predict.py`, like every other
diffusion model, so all are compared under one sampler; their 7 October cohorts
came from the diffusers pipeline in float32, so their images cannot be
bit-identical. For these two, the rehearsal compares validation SSIM with the
historical values instead.

**Rehearsal** (validation stands in for test; no test slice read): generation at
32 seeds per class for E2, E3, B3, SD, E0, B1 and B2 (jobs 672901-672907); real
features 672908; cohort features on the existing 64-image cohorts 672909;
membership for SD, E2, E3, B3 and R1-E2 (672910-672914); profiling 672915;
interpolation for E2 and E3 (672916, 672917).

## 2026-10-18 — R1 results: seed replicates of E2 and E3

R1-E3 ran as 672769 after the free-space fix (comp3710, 42 min 26 s, exit 0:0;
13,270 steps, 100 images/s, peak 36.82 GiB, EMA validation loss 0.05835 against
E3's 0.0583). Cohort 672770, features 672771 and audits 672772 completed; all
128 R1 PNGs matched their SHA-256. SSIM ran locally; earlier rows reproduced.
Aggregates: `r1-ssim-result-2026-10-18.json`, `r1-appearance-2026-10-18.json`,
`r1-neighbours-2026-10-18.json`, `r1-features-2026-10-18.json`.

| Validation, 64 samples | E2 (seed 3710) | R1-E2 (4710) | E3 (3710) | R1-E3 (4710) |
| --- | ---: | ---: | ---: | ---: |
| Feature-nearest SSIM | 0.7054 | 0.7005 | 0.6962 | 0.6936 |
| Bootstrap 95% CI (64 images) | 0.699-0.712 | 0.692-0.708 | 0.687-0.705 | 0.686-0.701 |
| Nearest SSIM > 0.6 | 64/64 | 63/64 | 63/64 | 64/64 |
| Nearest-training RMSE, mean (min) | 0.0673 (0.0425) | 0.0730 (0.0433) | 0.0846 (0.0621) | 0.0789 (0.0541) |
| Near-empty samples AD / NC | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| EMA validation loss, epoch 10 | 0.05455 | 0.05458 | 0.0583 | 0.05835 |

**Decision rules, as fixed in the protocol.**

- **Fidelity gap: a difference, small.** Seed means: E2 0.7030, E3 0.6949; gap
  0.0081. The larger within-model range is E2's, 0.0049. The gap exceeds it, so
  it is reported as a difference of 0.008 SSIM, not as seed variation. The
  bootstrap intervals of single cohorts overlap, so the difference is small
  relative to the evaluation noise of 64 samples.
- **Memorisation: robust by the rule, with no margin.** Both E3 seeds meet the
  0.0731 criterion (0.0846, 0.0789) and both E2 seeds miss it (0.0673, 0.0730).
  R1-E2 misses by 0.0001, so the rule's verdict rests on a negligible margin for
  that seed; the test protocol's calibrated measures replace this criterion.
- **Failure modes:** no near-empty samples in either replicate; E2's data
  cleaning holds across seeds.

Validation loss is reproducible across seeds to 0.0001, while sample-level
measures move by up to 0.006 in nearest-training RMSE: most of the spread in the
memorisation numbers comes from which 64 samples are drawn, not from training.

## 2026-10-19 — Rehearsal results; B2b rerun; code frozen for the test run

**Rehearsal** (validation in place of test; no test slice read). All 17 jobs
(672901-672917) completed with exit 0:0.

- **SSIM reproduction (required): passed.** The new test-SSIM path, run with
  validation references, reproduces every published value to six decimals:
  SD 0.087000, E0 0.376754, B1 0.726595, E2 0.705447, E3 0.696168, B3 0.709494,
  B2 0.580715 (fixed-reference values equal too). Real floor (training patients
  against validation patients) 0.707.
- **Batched generation (required): failed, so it is not used.** Batch 16 starts
  from identical noise (all noise hashes equal) but does not reproduce the
  existing images: median pixel correlation 0.9986, yet 3 of 64 E2 images and
  3 of 64 E3 images fall to 0.72-0.88, because guidance 7.5 over 30 DDIM steps
  amplifies rounding that depends on the batch shape. B1 and B2 reproduce 64/64.
  Diffusion cohorts are therefore generated at batch size 1, the path of every
  earlier cohort (7f162ef). Check jobs 673018 and 673019 regenerated E2 and E3
  at batch 1 with the new code: **64/64 PNGs byte-identical** for both.
- **Distribution, memorisation and membership measures** ran end to end. On these
  64-image cohorts (FID is strongly biased at this size; the test uses 1,024):
  E2 coverage 0.62, E3 0.28, B3 0.49, real floor 0.84; training preference M2
  E2 -0.012 (CI -0.037 to 0.014), E3 +0.031, B3 -0.291 (CI -0.354 to -0.231);
  membership AUC E2 0.684, E3 0.479, SD control 0.465.
- **Profiling** (672915, A100-PCIE-40GB, 17 min): seconds per image at batch 1
  and 16, deterministic settings: E2 1.97 and 0.49, E3 2.67 and 0.54, B3 1.90 and
  0.32; with fast settings: E2 1.32 and 0.23, E3 1.96 and 0.27, B3 1.29 and 0.11;
  B1 and B2 about 1 ms. Peak allocation 3.8 GiB (E2, batch 1) to 8.7 GiB (batch
  16). Profiling reads no evaluation data, so these are the final cost figures.
- **Interpolation** (672916, 672917): no I1 path jumps towards a training slice
  for E2 or E3; I2 inversion reconstructs its training slice to RMSE 0.04-0.06,
  and the distance to it rises smoothly along the path.

**B2b rerun** (the protocol's one rerun, with the calibration fix): AD 672812
and NC 672813 both stopped with no healthy snapshot (AD ratios after step 5,000:
0.18, 0.35; NC: 0.39, 0.35). The first run's NC snapshot at step 5,000 does not
recur. B2b has no result in either class, so it is reported as failed and, as
the protocol specifies, does not enter the test evaluation. Its cohort and
audit jobs (672814-672816) were cancelled.

**Code frozen.** The commit recording this entry is the frozen commit; the run
directory is refreshed to it and verified before any test job is submitted.
B2b is excluded; R1-E3's weights are from 672769.

## 2026-10-19 — Test-split results and the recommended model

**Run.** Frozen commit 2104414 (run directory verified against it). All 30
jobs and the cohort job completed with exit 0:0: real features 673089 (9 min
44 s; 4,700 test slices read once, hash-checked); cohorts at batch size 1, SD
673090, E0 673091 and B3 673092 whole on `comp3710`, E2, E3, R1-E2 and R1-E3
in four seed chunks each (673093-673108, 8-12 min each) joined by
`merge_cohorts.py`, B1 673109, B2 673110; membership 673111-673116;
interpolation 673117, 673118; cohort features 673504 (5 min 42 s). Each
cohort holds 1,024 samples, 512 per class; all 9,216 PNGs matched their
SHA-256 after download. B2b is excluded (no result in either class). Metrics,
SSIM, t-SNE and figures ran locally. Aggregates (no identifiers):
`test-metrics-2026-10-19.json`, `test-ssim-2026-10-19.json`,
`test-real-reference-2026-10-19.json`, `test-cohorts-2026-10-19.json`,
`test-membership-2026-10-19.json`, `test-interpolation-2026-10-19.json`,
`test-autopsy-e2-2026-10-19.json`; tables are printed by
`tools/results_tables.py`.

| Test split, 1,024 samples | SSIM (95% CI) | KID x 100 | Coverage | Precision | Recall | M2 pixel (95% CI) | Membership AUC minus SD (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Real floor (validation) | 0.707 | 0.06 | 0.813 | 0.754 | 0.755 | calibration | |
| SD zero-shot | 0.088 | 29.25 | 0.000 | 0.000 | 0.476 | +0.074 | control, AUC 0.499 |
| E0 | 0.328 | 10.92 | 0.028 | 0.079 | 0.001 | +0.027 | not measured |
| **E2** | **0.704** (0.703-0.706) | **1.27** | **0.274** | 0.267 | 0.849 | -0.0064 (-0.0131 to +0.0004) | **+0.237** (0.219-0.255) |
| R1-E2 | 0.702 | 1.19 | 0.288 | 0.302 | 0.816 | -0.0043 (-0.0111 to +0.0027) | +0.234 (0.216-0.252) |
| **E3** | 0.696 (0.694-0.698) | 4.25 | 0.066 | 0.065 | 0.675 | +0.0257 | +0.030 (0.014-0.047) |
| R1-E3 | 0.694 | 4.19 | 0.035 | 0.044 | 0.751 | +0.0106 | +0.032 (0.016-0.049) |
| B3 | 0.703 | 1.99 | 0.242 | 0.279 | 0.803 | -0.333 (-0.352 to -0.315) | +0.501 (AUC 1.000) |
| B1 VAE | 0.718 | 23.84 | 0.000 | 0.000 | 0.000 | -0.031 | not measured |
| B2 GAN | 0.582 | 38.98 | 0.000 | 0.000 | 0.000 | +0.075 | not measured |

Real references on test: mean nearest-training RMSE of test slices 0.0745
(closest 0.0424); calibration log ratio (validation slices) -0.086 in pixel
space. M2 in VAE space (secondary): E2 -0.0145 (-0.0233 to -0.0058), R1-E2
-0.0122, E3 +0.0269, B3 -0.272. True-positive rate at 1% false positives:
E2 13.5%, E3 1.8%, B3 100%, SD 3.0%. Near-empty samples: none in 1,024 for
E2, R1-E2, E3, R1-E3 or B3.

**M4 closest pairs** (private panels, visual review). E2 and E3: the five
closest pairs share outline and pose but differ in gyral folding, ventricle
shape and contrast: no copies. B3: three AD seeds (37554, 37563, 37442)
reproduce the same training slice and two NC seeds another one, at RMSE
0.017-0.021, half the closest real-to-real distance: near-copies.

**Interpolation.** No I1 path of E2 or E3 has an interior frame closer to a
training slice than the closest real pair; one E2 NC path (37202 to 37203)
changes brightness and size abruptly between frames 5 and 6. I2 inversion
reconstructs its training slice to RMSE 0.04-0.06, and the distance to it
rises along the path to about 0.06, a typical sample distance.

**Recommended model, by the rule fixed in the protocol.**

1. Exclusion: both candidates pass SSIM > 0.6 (E2 0.704, E3 0.696; 99% of
   samples above 0.6) and neither shows a copy.
2. Memorisation signal: E2 has one (M3 +0.237, interval far from 0; its
   primary M2 interval touches 0). E3 also has one: its M3 interval
   (0.014-0.047) excludes 0, replicated by R1-E3. Both have a signal, so this
   step does not decide.
3. Coverage: E2 0.274 against E3 0.066; the difference (0.208) far exceeds the
   seed spread (E2 0.014, E3 0.031).

**The recommended model is E2.** The rule treats a memorisation signal as
present or absent; it does not weigh size. E2's model-level leak is about eight
times E3's in AUC excess (0.237 against 0.030) and seven times in true
positives at 1% false positives (13.5% against 1.8%), while its samples are
at most 0.6% (pixel) to 1.4% (VAE features) closer to training slices than
held-out slices would be. This is reported with the recommendation (Section
10 of the README): E2's samples are low-risk to share, its weights are not.

**Autopsy** (rule: the recommended model's 5 lowest-SSIM samples plus every
near-empty one; none was near-empty). All five are NC: NC-37263 (0.536) and
NC-37595 (0.586) are over-bright with anatomy cut off at the frame edge;
NC-37407 (0.563) has a ragged, disordered posterior fossa; NC-37489 (0.577)
has a vertical stripe running from the brainstem to the image edge and a bright
outer rim; NC-37341 (0.587) has its lower half replaced by dark, mottled
texture. Bright rims also occur in some training slices (imperfect skull
stripping), so that feature is likely learned rather than invented; the other
causes are hypotheses, not verified.

**Not yet done:** the blinded sort. Its first build used the one-per-patient
SSIM references, which are middle slices only and would give the sort a
slice-position cue; it was discarded unseen and the tool now draws random test
slices (39dff86), to be extracted on Rangpur when the connection returns.

## 2026-10-21 — Post-hoc: does the requested class show in the samples?

**Not pre-registered.** Prompted by previous years' reviews, which expected an
embedding with clustered classes; the test t-SNE (VAE features) shows real and
generated slices overlapping but AD and NC mixed. This analysis uses only
features already extracted in the frozen test run; no data were read again.

A logistic-regression probe (standardised Inception features, C = 0.006 by
5-fold cross-validation) was fitted on the 1,024 class-balanced validation
slices and read on the test slices and on each model's samples, split by the
class requested (`tools/class_probe.py`; `test-class-probe-2026-10-21.json`):

| | Probe AUC, NC against AD (95% CI) | Mean P(NC), asked AD / asked NC |
| --- | ---: | ---: |
| Real test slices | 0.758 (0.744-0.771) | |
| E2 / R1-E2 | 0.948 / 0.950 | 0.17 / 0.81 |
| E3 / R1-E3 | 0.919 / 0.918 | 0.04 / 0.48 |
| B3 | 0.757 | 0.29 / 0.59 |
| E0 | 0.770 | 0.19 / 0.45 |
| B2 GAN | 0.963 | 0.11 / 0.83 |
| B1 VAE | 0.525 | 0.36 / 0.38 |
| SD zero-shot | 0.491 | 0.21 / 0.20 |

**Reading.** The conditioning works and shows: E2's and E3's samples are
separated by the requested class. They are *more* separable than real slices
(0.95 against 0.76): the samples exaggerate the class difference, which real
slices show only weakly per slice. A likely cause, not tested, is guidance 7.5,
which extrapolates along the class direction. B3 separates as much as real
slices; the VAE barely uses its class input; SD, without training, not at all.
The GAN's high value comes from one template per class, so separability alone
is not quality. Consequence for the verdict: a classifier trained on E2's
samples would see cleaner class differences than real data contain, one more
reason downstream use is not established.

## 2026-10-24 — Deviation: real reference slices shown before the blinded sort

The protocol's blinded sort assumed a non-expert with no preparation. The
student, who has no medical training, asked to see real slices first. Before
sorting, the student is shown 16 real slices (8 AD, 8 NC) drawn at random from
the **training** split (`blind_sort.py extract --split train --per-class 8`,
seed 3710, on the Rangpur login node; `blind_sort.py reference` builds the
page). The sort's real slices come from the test split, so none of these can
appear in it, and no synthetic example or description of synthetic artefacts
is shown. The page stays private (real slices). The student had already read
the README, with its uncurated synthetic grids and autopsy figures, so the sort
measures a non-expert who has seen both real and synthetic examples, not a
naive viewer.

## 2026-10-24 — Blinded sort result

Scored with `blind_sort.py score` against the private key; aggregate in
[test-blind-sort-2026-10-24.json](test-blind-sort-2026-10-24.json).

| | Called real | Called synthetic |
| --- | ---: | ---: |
| 32 real test slices | 22 | 10 |
| 32 E2 samples | 12 | 20 |

Accuracy 42/64 = 66% (Clopper-Pearson 95% CI 53-77%), above chance (two-sided
binomial p = 0.017). By class, 10/16 real AD, 12/16 real NC, 10/16 synthetic AD
and 10/16 synthetic NC were sorted correctly: no class is much easier.

**Reading.** The protocol's reading was that accuracy near 50% would mean a
non-expert cannot tell the two apart. The interval excludes 50%, so this
non-expert, prepared with real reference slices and having read the README, can
tell them apart better than chance, though with many errors: 12 of 32 samples
passed as real. The README's plausibility verdict changes from "plausible to a
non-expert" to "partly". One rater and 64 items make this a weak test either
way; it is not clinical validation.
