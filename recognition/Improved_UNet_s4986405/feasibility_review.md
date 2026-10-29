## Feasibility Review

**Project:** HipMRI 2D Improved U-Net | **Difficulty:** Normal | **Student:** s4986405
**Updated:** 29 October 2026 | **Tutor consultation:** completed, as reported by the student
**Status:** Compulsory tutor check-off completed, as reported by the student on 29 October. Tutor feedback on anatomical identification and explicit baseline comparison is addressed in the revised report.

### 1. User need, scope and engineering dilemma

The course project targets prostate segmentation from the supplied HipMRI 2D pelvic MRI slices. The prototype predicts a label for each pixel to provide a contour draft for human review. The engineering question is whether a more complex model improves segmentation enough to justify additional memory and inference time. The revised image-based analysis supports prostate = ID 5 from its position below the bladder-like region and relationship to the rectum-like region across patients/slices. This is an inferred mapping, not an official dictionary; numeric IDs are preserved.

### 2. Model choice, data and course concepts

A basic U-Net [1] is compared with a 2D Improved U-Net inspired by Isensee et al. [2], using residual context blocks and combined multi-scale outputs. This is a 2D adaptation of a 3D brain-MRI design. Implemented preprocessing includes per-slice z-score normalization, six-channel one-hot training targets, paired training-only flips and native-grid batches. Both models share cross-entropy plus soft-Dice loss, 20 epochs and seed 3710. The supplied patient-separated train/validation/test splits contain 11,460/660/540 slices from 31/4/3 patients. Checkpoints are selected by minimum validation loss and then fixed for testing. Course concepts include convolutions, encoder-decoder skip connections, residual learning, normalization and overlap metrics.

### 3. Feasibility evidence and five acceptance criteria

Both models completed full training and validation on A100 GPUs. Fixed weights then completed three full local test passes per model on an RTX 4070. The mean of six pooled-label test Dice scores was **0.9179 baseline / 0.9243 Improved**, with declines on IDs 3 and 4. ID-5 test Dice increases **0.8037 to 0.8430 (+3.94 percentage points)**, with fewer FP and FN pixels. Improved is not uniformly better.

1. **Data integrity:** preserve patient separation, paired masks and native-grid alignment. Split checks, coverage audits and overlays passed; every test pass processed 540/540 slices.
2. **Target accuracy:** proposed prostate pooled test Dice >=0.75. Under the illustrated image-based ID-5 interpretation, pooled test Dice is 0.8430, exceeding 0.75. The mapping is inferred from anatomy, not selected by Dice; all six IDs remain reported.
3. **Comparison and error review:** compare identical held-out patients using Dice/IoU, pooled counts and patient means. Completed; four real validation failures and one improvement have MRI/reference/prediction and FP/FN overlays.
4. **Operational limits:** proposed median model-only forward <=100 ms per 256x128 slice at batch 1. Three-pass median summaries were **1.605 / 4.675 ms** after warmup and CUDA synchronization, excluding I/O/transfers. Proposed whole-device peak VRAM <=12 GiB remains unverified; A100 allocated-training peaks of 0.120/0.302 GiB measure tensors, not whole-device memory.
5. **Reproducibility:** record dependencies, seed, configuration, source and checkpoint hashes; keep weights fixed after validation. Completed records and consistency checks are available. Repeated inference does not constitute independent training replication.

### 4. Risks, budget, next steps and fallback

Risks include one training seed, only three test patients, unequal patient contribution, missing 3D context, unconfirmed anatomy and label-specific regressions. Each training run was capped at four GPU-hours and finished in about 49/63 minutes; local test evaluation/export took about 60 seconds, excluding setup/download. Next: review the illustrated label inference and explicit comparison with the tutor as needed, and address PR feedback. Check-off is complete as reported by the student. Retain the baseline as the simpler reference and human contour review as essential; additional complexity needs a target-specific benefit. Any future tuning requires a separate evaluation design rather than reuse of test scores for model selection.

### References

[1] Ronneberger, Fischer and Brox (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. https://arxiv.org/abs/1505.04597
[2] Isensee et al. (2018). Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge. https://arxiv.org/abs/1802.10508

AI assistance: ChatGPT/Codex supported implementation, execution, checks and drafting; the student remains responsible for reviewing and explaining the work. Evidence and full disclosure: project README, VALIDATION_RESULTS.md, TEST_RESULTS.md and ERROR_ANALYSIS.md.
