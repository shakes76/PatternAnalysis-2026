# Feasibility Review — 2D Improved U-Net for Prostate Segmentation (HipMRI)

**Kelsen Ze Yang Gho (s4942544)** · COMP3710 Report · Segmentation project · **Normal difficulty** · Check-off: 20 Oct 2026

## 1. User need, scope and acceptance criteria

**User:** a radiotherapy planner who manually contours the prostate on pelvic MR before external-beam treatment. **Prototype:** a 2D slice-wise annotation assistant that proposes six-class contours (background, body, bone, bladder, rectum, prostate) for the planner to review and correct — a drafting aid, never an autonomous contour. **Open dilemma:** a good *global* Dice can hide failures on the prostate's **edge slices** — where the gland first appears and disappears in the scan and its cross-section is tiny (in this coronal data, its front and back edges; the analogue of the base and apex on axial scans). There, under-contouring risks recurrence and over-contouring irradiates the rectum and bladder.

| # | Acceptance criterion (held-out, patient-level test split) |
|---|---|
| AC1 | Improved U-Net prostate Dice **≥ 0.75** (specification target). |
| AC2 | Prostate Dice reported **separately for edge slices** (outer 20% of each scan's prostate extent) vs central slices, for both models. The Improved U-Net is judged better only if its base/apex Dice exceeds the baseline's by more than the seed-to-seed spread. |
| AC3 | **False-positive control:** on test slices with no prostate, ≤ 5% contain a spurious prostate prediction (> 20 px). |
| AC4 | **Resources:** ≤ 1 h training per model on one A100, ≤ 4 GB peak VRAM, ≤ 50 ms inference per slice (interactive use). |
| AC5 | **Reproducibility:** final numbers are mean ± s.d. over 3 random seeds with pinned dependency versions. |

## 2. Model choice and course concepts

* **Data:** HipMRI 2D `keras_slices_data`, 6 labels. Provided splits kept — audit proved them patient-level.
* **Baseline:** standard 2D U-Net (Ronneberger et al., 2015) — encoder–decoder CNN with skip connections, batch norm.
* **Chosen model:** 2D **Improved U-Net** (Isensee et al., 2017): residual *context modules* (the residual learning from ResNet), strided-conv downsampling, **instance norm** (stable at small batch sizes), and **deep supervision** that sums segmentation maps from three decoder scales.
* **Loss:** cross-entropy + soft Dice. Dice weights every class equally, countering the prostate's < 0.5% pixel share.
* **Why this tests the dilemma:** deep supervision injects coarse multi-scale context — the hypothesis is that this helps the small, low-contrast edge slices. Both models see identical data, augmentation, optimiser and schedule.

## 3. Preliminary feasibility evidence

**Data audit (all 12,660 slices):** 31 / 4 / 3 patients in train / validate / test with **zero patient overlap**; every slice paired with a mask; 99.5% are 256×128 (one scan 256×144 → centre-cropped, never resized, so labels are not interpolated). Prostate = **0.43% of pixels**, present in only 36% of slices; present in **93% of mid-scan slices but only 2–7% near scan ends** — the edge-slice regime the dilemma targets.

**GPU smoke test (5 epochs each, full data, A100):**

| Model | Params | Peak VRAM | Time / epoch (steady) | Inference / slice | Test prostate Dice |
|---|---|---|---|---|---|
| Baseline U-Net | 7.76 M | 0.89 GB | 34 s | 2.6 ms | 0.865 |
| Improved U-Net | 3.19 M | 0.67 GB | 38 s | 6.0 ms | 0.862 |

Both exceed AC1 after 5 epochs and are well inside AC4. Notably, the models are **indistinguishable on global Dice** — direct evidence that the comparison must be stratified (AC2, AC3), not a single headline number. Fewer parameters also did **not** mean faster inference (2.3× slower), which matters for the cost–benefit verdict.

## 4. Risks, budget and fallback

* **Small test cohort (3 patients, 9 scans)** → high variance. *Mitigation:* 3 seeds, per-patient breakdown, honest confidence statements.
* **No real difference between models.** This is a valid answer to the dilemma, not a failure; the stratified analysis will show *where* (if anywhere) the extra complexity helps.
* **Overfitting:** train/val loss gap already visible (0.14 vs 0.37 at epoch 5). *Mitigation:* augmentation in place; checkpoint chosen on validation prostate Dice.
* **Cluster:** `comp3710` partition rejected some `#SBATCH` options. *Mitigation:* minimal, tested Slurm scripts; short smoke runs before long jobs.

**Compute budget:** 2 models × 3 seeds × 50 epochs × ~35 s ≈ **3 A100-hours**. **Next experiment:** full 50-epoch runs of both models, then a `predict.py` evaluation stratified by slice position, plus a failure autopsy on 3–5 edge-slice cases. **Fallback:** if the Improved U-Net under-delivers, the Normal tier is already secured (AC1 met) and the comparison stands as a negative result. **Stretch:** extend to the 3D Improved U-Net (Hard) to test whether true volumetric context fixes edge-slice errors.
