# AI Assistance Record

## 2026-10-02 — Project foundation

- Tool: OpenAI Codex.
- Assistance: assessment interpretation, SD/Rangpur planning, repository setup, and drafting project documentation, ignore patterns, and line-ending rules.
- Sources: supplied assessment PDF, Rangpur/GPU Lab records, and current Git repository state.
- Verification by the assistant: read the local assessment and repository documentation; inspect the staged changes and exercise ignore rules before the initial project commit.
- Student review: pending. No claim is made that the student has already reviewed or tested these artifacts.
- Experimental evidence: none. No dataset access, training, or evaluation has been performed for this project.

Add entries as work proceeds. Describe substantive AI contributions accurately and record the actual audit or empirical verification performed. Summarize this record in the README disclosure before submission.

## 2026-10-02 — Private development and submission workflow

- Codex checked the local branch and the public fork branch, confirmed the course topic commit is an ancestor, and consulted GitHub documentation on fork visibility.
- Documented local commits, private local backups, final publication, and a single PR with subsequent review updates. Updated the tentative deadline note using the student's supplied course-profile/tutor excerpts; the final submission timestamp is still unverified.
- Configured a repository-local guard against bare pushes and created a verified Git bundle backup. No project commits were pushed and no PR was opened.

## 2026-10-02 — Read-only Rangpur and dataset audit

- Codex implemented `tools/audit_adni.py`, executed it on actual course data in CPU job 620988, and recorded aggregate output and limitations.
- Checked supplied scan IDs against metadata patient IDs, counted overlap between existing splits, inspected image headers, and inventoried dependencies and disk space. No fabricated test data or training metrics were used.
- Initial shell quoting/CRLF failures are recorded in the audit report. The final audit completed with exit 0:0. Existing GPU training was not interrupted, and no packages were installed or datasets modified.
- Student review of implementation and findings is pending. Patient identifiers and source images are not included in the committed audit output.

## 2026-10-02 — Data preparation before model storage is available

- Codex drafted the pre-experiment data protocol, implemented patient-stratified manifests and image loading, and added a CPU-only real-data validation command.
- Seven meaningful local unit tests passed using explicitly synthetic fixtures; they check data/geometry contracts and rejection of corrupt manifests, not training success. Ruff checks passed. An unnecessary torchvision import was removed after local inspection showed it was absent; tensor conversion uses PyTorch directly.
- Full-data validation is queued as CPU job 623310 and is not yet accepted. No dependency installation, model download, source-data modification, or GPU allocation occurred. Student review remains pending.

## 2026-10-02 — Accepted data and queued VAE diagnostic

Codex verified completed CPU job 623310 and retrieved its real aggregate results.
After the student confirmed storage release, a read-only check found approximately
12 GiB free and no running user jobs. Codex documented the VAE protocol, pinned
the SD 1.5 VAE revision, implemented isolated package setup and a bounded paired
reconstruction diagnostic, and submitted dependent CPU/GPU jobs 623826/623828.
Local syntax and Ruff checks passed; remote setup and reconstruction remain
pending. Student review is still pending. No patient-level artifacts were pushed.

## 2026-10-03 — Repair incompatible dependency pins

Codex inspected the actual setup failure, acknowledged the incompatible initial
Hub pin, checked PyPI metadata, changed Hub to 1.33.0 and resubmitted the guarded
CPU/GPU job chain (625832/625833). This is a dependency correction, not a successful
model experiment. Student review remains pending.

## 2026-10-04 — Real LoRA restart acceptance and website update

Codex authored the pre-run protocol, VAE clipping diagnostics and minimal LoRA
training/restart comparator; verified package constraints against PyPI metadata;
and ran CPU/GPU jobs 629842, 629851 and 629852. All completed with exit 0:0.
The restart comparator used real checkpoints and included intentional corruption
controls. Raw medical artifacts remain ignored/private; aggregate evidence was
used for the learning website. Student review is pending. This is engineering
acceptance, not a successful formal generation experiment.

## 2026-10-04 — Paired 200-step pilot

Codex read assessment Section2.4, authored the pre-run pilot protocol and adapted
an isolated copy of the accepted smoke trainer for 32 training patients and200
updates, with deterministic shuffled order and fixed DDIM sampling at0/100/200.
The original smoke code is unchanged. Local Ruff and syntax checks passed; remote
Python/shell syntax checks passed. Submitted job629919. Execution and visual
acceptance are pending; student code/protocol review remains pending.

Pilot629919 completed with exit0:0 in2m15s. Codex retrieved private artifacts,
verified actual traces/paired noise/file hashes and inspected all three grids.
Observed appearance changes and persistent structural failures are recorded in the
ledger without claiming medical validity. Student verification remains pending.

## 2026-10-04 — Queue full-split experiment

At the student's explicit request, Codex inspected live Slurm/account/QoS/storage,
recorded the long-run protocol, adapted an isolated trainer, ran local Ruff/syntax
and remote Python/shell syntax checks, and submitted gate629930 plus dependent
training629931. Preserved accepted pilot/smoke code and frozen queued source.
Full-split shuffled order and stage checkpoints are included in the resume contract.
Student review pending; queue submission is not training or quality acceptance.

## 2026-10-05 — Long-run artifact verification

Codex checked Slurm completion for629931, retrieved all private outputs, verified
report/trace/image consistency and inspected the final uncurated sample grid.
Recorded persistent failures without claiming anatomical or clinical validity.
Student review remains pending.

## 2026-10-05 — Official inference diagnostic

Codex consulted official Diffusers API/source, implemented a fixed-checkpoint
pipeline comparison and resolution/guidance experiment, ran Ruff and Python/shell
syntax checks, and submitted632530. All generated probes will be retained.
Comparison and quality findings pending; student verification pending.

Both inference jobs completed:632530 (1m57s) and precision-aligned632535 (26s).
Codex verified all output hashes and the four exact historical pixel matches after
aligning the scheduler-input dtype. Inspected all four factorial grids. Improved
appearance with guidance7.5 is preliminary, not evidence of privacy or coverage.

## 2026-10-05 — Cohort, nearest-neighbor audit and baseline

Codex authored the pre-run protocol, expanded fixed-seed generation, implemented
pixel-distance audits with an exact-copy positive control, and wrote a small
from-scratch conditional convolutional VAE using the standard analytic Gaussian
KL objective. Local Ruff/syntax checks and shell validation preceded submission.
Jobs632557/632558 cover SD generation/audit;632569/632570 gate and train the
baseline. All conclusions await real outputs and student review. Patient-level
nearest-neighbor records stay in ignored private outputs.

## 2026-10-06 — Cohort and baseline acceptance

Codex verified all four jobs completed, downloaded private artifacts, checked192
PNG hashes and final VAE checkpoint hash, reviewed all three full grids, and ran
baseline audit635709 under the existing metric protocol. Recorded the blurred
baseline's better pixel RMSE and limitations of image-distance/privacy claims.
No patient-level neighbors were added to Git. Student review remains pending.

## 2026-10-06 — Real-reference calibration and failure panels

Codex implemented patient-excluding real-reference calibration and assembled five
private generated/train/validation panels. Job635718 completed43s, exit0:0.
Reviewed all five panels and separated three observed visual failure types from
two unconfirmed risk probes. Updated the private learning website with aggregate
evidence and explicit limitations. Patient images/identifiers remain excluded
from Git and Site. Student interpretation and final report review remain pending.

## 2026-10-07 — Appearance diagnostics

Codex formalized exploratory color/border/edge measurements, documented the
protocol, implemented and submitted CPU job636215, verified source hashes and
successful outputs, and interpreted aggregate and case-level results. Real
reference calibration completed7s, exit0:0. No test image was opened. These are
descriptive proxies, not anatomical or privacy metrics. Student review pending.

## 2026-10-07 — Existing factorial probe analysis

Codex wrote and ran audit_sampling.py on16 hash-verified existing images, checked
factorial completeness and paired initial noise, and documented both improvements
and contrary color evidence. Ruff passed. No new training/generation was run;
student review remains pending.

## 2026-10-07 — Controlled CFG sweep preparation

Codex added a validated multi-guidance option to the existing generator, retained
its7.5 default, and documented the192-image protocol before running it. Local
Ruff, formatting, syntax and diff checks passed. Regeneration of7.5 is intended
for historical parity checking, not assumed equivalent before outputs arrive.
Student interpretation remains pending.

## 2026-10-07 — CFG sweep acceptance

Codex checked192 outputs, historical parity and noise pairing, reviewed all three
full grids plus six individual case images, implemented reproducible paired
summaries, and corrected a mean-only interpretation using per-sample/seed-block
counts. Student interpretation and clinical/anatomical validation remain pending.

## 2026-10-07 — Website and interpolation preparation

Codex updated the private HTML with measured appearance and paired CFG evidence,
added an optional initial-noise interpolation mode and a pre-run protocol, and
submitted639671. Interior noise variance is explicitly disclosed. Endpoint parity
and visual interpretation await outputs; student review remains pending.

## 2026-10-07 — Interpolation acceptance

Codex retrieved639671, verified20 generated images and eight exact historical
endpoints, inspected all four five-frame paths, and documented the observed
noise-norm shrinkage and retained artifact. No medical interpretation or proof of
smoothness/coverage is claimed. Student review remains pending.

## 2026-10-07 — Frozen feature evaluation

Codex rechecked PDF2.4, documented remaining SSIM/real-synthetic interpolation
requirements, implemented frozen VAE extraction and bidirectional distances,
submitted639765, verified transferred hashes, and rendered/inspected joint t-SNE.
Feature-space conclusions are explicitly encoder-dependent; student review pending.

## 2026-10-07 — SSIM and encoded paths

Codex checked official scikit-image documentation, installed pinned0.25.2 into
the existing E-drive environment (cache on E), wrote pairing/interpolation
protocols, submitted639777 and evaluated all192 generated images locally. Reviewed
both complete encoded paths and disclosed SD's below-target SSIM and blurred
intermediate frames. Student understanding and final interpretation remain pending.

## 2026-10-07 — Feasibility discussion draft

Codex rechecked task-sheet section3, reconciled the requirement coverage ledger,
and drafted/generated a one-page English PDF for student review. Existing
experiments are labelled retrospective evidence; tutor consultation is not recorded
as completed. The student must verify, understand and revise this draft before use.

## 2026-10-12 — Review and E1 implementation (Claude Code)

- Tool: Anthropic Claude Code (Claude Opus 5.5), from this date alongside the earlier Codex work.
- Review: compared assessment v2.2 with v2.1 page by page (text identical; two links changed), re-ran the
  ADNI subject-overlap count on Rangpur from `meta_data_with_label.json` (565 / 331 / 216 of 680 subjects,
  matching the 2 October audit), and explained why the Ed statement of "no overlap" holds only for scan IDs.
- Proposed E1 (full-UNet fine-tuning) with hypotheses tied to observed E0 failures; the student approved it
  on 12 October before any job ran. The protocol and acceptance criteria were committed first.
- Implemented `modules.py`, `train.py`, `predict.py`, hash verification in `dataset.py`, `--extra` cohort
  options for three audit tools, and the E1 Slurm wrappers.
- Verification: 17 CPU unit tests with tiny random models, including a check that the guided DDIM sampler
  matches diffusers' `StableDiffusionPipeline` latents; Ruff lint and format; rerunning the modified SSIM audit
  with default arguments reproduced all seven 7 October means exactly; SHA-256 of every file copied to
  Rangpur matched the committed blob.
- Student review of the new code and of the E1 results is pending.

## 2026-10-13 — E1 evaluation (Claude Code)

- Ran E1's pre-registered evaluation: cohort generation, frozen-VAE features, SSIM, appearance
  and pixel nearest-neighbour audits, and added `closest_panels.py` and `foreground_census.py`.
- Verification: the extended SSIM audit reproduced every historical row exactly; generated PNGs
  were hash-checked after download; the closest pairs were inspected visually, which showed the
  near-zero distances were blank images rather than copied anatomy, and traced them to one scan.
- Reported the unmet memorisation criterion as unmet, with post-hoc analysis labelled as such.
- Mistake caught during the work: the generalised t-SNE script silently dropped a seventh cohort
  from the figure (`zip` against six marker shapes). Found by inspecting the plot legend; fixed
  in 593def9 with a hard error, and the 7 October output re-checked as bit-identical.

## 2026-10-13/14 — E2 (Claude Code)

- Proposed E2 from the E1 diagnosis; the student approved the protocol before any job ran.
- Implemented `dataset.foreground_share` / `defective_scans`, `flag_scans.py`, the
  `--exclude-defective-scans` option with an expected-count stop, and the E2 wrappers.
- Verification: unit tests for the rule and the exclusion (19 pass); a CPU job and private panels
  confirmed the two flagged scans were defective before E2 was proposed; the probe confirmed the
  expected 2 scans / 40 slices; all remote files matched the committed blobs.
- Mistakes caught before any submission: commit 3aaf187 contained wrappers with a literal
  backslash-n (from a sed edit) that would have reached `train.py` as an argument; fixed in
  a64e1b9 after a dry run with stub commands. Commit 46a11ba was made while a new unit test was
  failing (a float tolerance in the test, not in the rule), because the command did not gate the
  commit on the test result; fixed in 3ea01d4, and later commits were gated on passing tests.
- With the student's approval, backed up both E1 weight files locally (SHA-256 verified) and then
  deleted the E1 epoch-5 file on Rangpur.

## 2026-10-14 — E3 and check-off material (Claude Code)

- Proposed and wrote the E3 capacity-ablation protocol, approved by the student before any job;
  wrote the E3 wrappers directly and checked their arguments with `train.py`'s own parser.
- Updated the one-page check-off draft to 14 October and prepared check-off notes in the private
  study site. Video recommendations were checked against their publishers' pages before listing.
- Student review of the code, results and the final interpretation remains pending.

## 2026-10-15 to 18 — Baselines B2, B2b and B3; seed replicates R1 (Claude Code)

- Proposed the B2 GAN (ported from the student's demonstration-2 GAN), the B3 from-scratch latent
  diffusion model, the B2b per-class GAN rerun and the R1 seed replicates, each as a protocol the
  student approved before any job ran. Implemented `tools/baseline_gan.py`, the scratch-UNet option,
  `--train-class`, `--val-seed`, and every Slurm wrapper; ran the evaluations and wrote the ledger.
- Verification: CPU tests with tiny models (30 pass), dry runs of every wrapper with stub commands,
  SHA-256 checks of remote files against committed blobs and of every downloaded PNG, and visual
  review of uncurated grids and closest pairs before interpreting any number.
- Mistakes, all mine:
  - The B2 protocol first misdescribed demonstration 2's diversity measure; corrected by a dated
    erratum (babe480) before any run.
  - B2's collapse rule pooled both classes, so it could not see a conditional collapse. Found in the
    evaluation and reported; B2b checks each class separately.
  - B2b's BatchNorm recalibration kept drawing random labels in single-class mode (73c0453), which
    lowered every logged diversity ratio. A CPU check showed it does not explain the failure. Found by
    visual review; fixed with a test that fails on the old code.
  - R1's wrapper required 2 GiB free for both models, although E3 writes a 1.6 MB adapter; the R1-E3
    job (669199) refused to start. The check was sized per model (f90dd4f) and the same job resubmitted.

## 2026-10-18 to 19 — Final test-split evaluation and README (Claude Code)

- Drafted the final evaluation protocol, which the student approved before any code read the test
  split; implemented the evaluation tools (batched and one-image sampling, DDIM inversion,
  Inception features, FID/KID/PRDC, the calibrated training-preference ratio, loss-based membership
  inference, profiling, interpolation, the blinded sort, figure and table generators); ran the
  validation rehearsal, froze the code (2104414) and ran the test evaluation once; applied the
  protocol's selection rule; drafted the README.
- Verification: unit tests in two environments (53 and 20), including closed-form checks of FID and
  exact DDIM inversion and mutation checks; dry runs of every wrapper; hashes of all code copied to
  Rangpur and of all 9,216 test images; the rehearsal had to reproduce every published validation
  SSIM to six decimals before the test split was read, and did; README tables are printed from the
  metric files by `tools/results_tables.py`, not typed.
- Mistakes, all mine:
  - Batched sampling was planned without first checking that it reproduces earlier images; the
    rehearsal showed it does not (a few images diverge). Replaced by one-image sampling before the
    freeze, which reproduces 64/64 images byte for byte.
  - The R1 results were first labelled 19 October, a day early; corrected (ad4432f).
  - The blinded sort was first built from the one-slice-per-patient SSIM references, all middle
    slices, which would have given a slice-position cue; noticed before the student saw it,
    discarded, and rebuilt to draw random test slices (39dff86).
  - A background command defined its variables in the wrong subshell, so the first local SSIM run
    did not start; rerun unchanged. No result was affected.

## 2026-10-21 to 24 — Class probe, report figures, README revision, blinded sort (Claude Code)

- Ran the post-hoc class probe; drew the report figures (`tools/figure_report.py`) from committed
  records only; shortened the README from about 7,700 to 5,300 words at the student's request,
  with the latent diffusion figure of Rombach et al. (rendered from the paper's arXiv source,
  credited in the caption); added `predict.py --classes`; made the training-only tests skip
  cleanly in the analysis environment; checked the full unpushed history for patient identifiers,
  data files, local paths and secrets (none found) and made a git bundle backup.
- Blinded sort: at the student's request, built a private page of 16 real training-split slices to
  view first (recorded as a deviation in the ledger), then scored the student's answers with
  `tools/blind_sort.py score`. The student did the sort alone; I did not see the page or the
  answers before scoring. The result (42/64) weakened the README's plausibility verdict, which
  was revised to match.
- Mistakes, all mine: a shell heredoc broke backslashes and non-ASCII text more than once (fixed
  with the editor, re-tested); a script wrote three test files with Windows line endings (restored
  before commit).

## Student verification (24 October 2026)

"I reviewed and examined all AI-generated work, assessed each proposal, personally carried out
the verification and blinded classification, and cross-checked the instructions in the README
file against the recorded accounts."

## 2026-11-10 — Response to the PR review (Claude Code)

- The initial review of PR #44 asked for the AI disclosure to say, in the student's own words,
  which parts the student checked personally, and noted that a single blinded-sort rater who had
  seen many samples probably understates how often samples pass as real.
- The student wrote their account in Chinese; I put it into English in README Section 13 ("What I
  checked myself", "Problems I found"), adding only items already in this record (the reference
  slices before the blinded sort), and the student checked the wording. No second rater was available; README Sections 7.3 and 12 now state the direction of
  the bias as a limitation.
