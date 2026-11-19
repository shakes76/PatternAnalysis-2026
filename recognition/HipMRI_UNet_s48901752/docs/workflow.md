# COMP3710 — HipMRI Segmentation Architecture & Workflow

> **Purpose:** The purpose of this is to ensure I am on track and ensuring I am progressing as quickly while still keeping quality
>
> **Research question:** Does incorporating 3D spatial context improve multi-class MRI segmentation when compared to a standard 2D U-Net segmentation baseline and does and improved 3D U-Net provide enough improvement for the increased computational cost?
>
> **Hypothesis**: IF A 3D U-Net uses spatial information from neighbouring MRI slices THEN It may produce more accurate segmentation than a standard 2D U-Net BECAUSE Organs are three-dimensional structures whose cross-sectional shapes and boundaries change across depth. Learning these relationships may help identify anatomical boundaries that are ambiguous in individual 2D slices.
> 
> **Status:** Planning document. Tick tasks only when completed and verified; link each milestone to its real commit and evidence. The plan can evolve as experiments are run.

## 1. Overall workflow

1. Verify and prepare the HipMRI dataset.
2. Implement and validate a standard **2D U-Net** as a baseline.
3. Train and evaluate the 2D model.
4. Implement and validate a standard **3D U-Net**.
5. Train and evaluate 3D model using matching patient splits and metrics.
6. Decide on and implement a justified **Improved 3D U-Net** (candidate: residual blocks and deep supervision).
7. Evaluate architectures using Dice, per-class performance, computational cost, and segmentation failure cases.
8. Write the final README report, disclose AI assistance, and submit the PR and report PDF.

## 2. Architecture decisions to verify

### Dataset and task
- HipMRI volumetric images and segmentation masks (NIfTI).
- Expected classes: 0–5, six segmentation labels; verify label-name correspondence with dataset documentation. 
- Known scan shapes include 256 × 256 × 128 and 256 × 256 × 144; voxel spacing varies across scans.
- Use patient-level train/validation/test splits so different weeks for a patient remain in one split.
- Define consistent preprocessing, orientation, and full-volume evaluation before comparing architectures.

### 2D baseline
- Input: `[B, 1, H, W]` (one grayscale MRI slice).
- Output: `[B, 6, H, W]` (logits for six classes).
- Encoder, bottleneck, decoder, and skip connections.
- Initial loss candidate: 0.5 Dice + 0.5 Cross-Entropy. Reasoning: balance region overlap with per-pixel class supervision; verify formulas and class handling.
- This will be baseline and we are planning to improve it. Dice score and cross entropy will be used to see how baseline works. We won't improve this or add stuff like loss functions right now. Just do a base 2D Unet and feed the data to then  record dice score and cross entropy 

### Standard 3D
- Input: `[B, 1, D, H, W]` (3D volume/patch).
- Output: `[B, 6, D, H, W]`.
- Replace 2D convolutions, pooling, and upsampling with 3D operations; retain skip connections.
- Preferred first trial: patch-based 3D training, subject to measured GPU memory and data geometry.
- Hypothesis: neighbouring slices provide spatial context and improve structure consistency. This is a test, not a guaranteed result.
- This will become the new baseline for next stage which is improved 3D which uses residual blocks first then add in deep supervision or either or depending how well its performing and then measure once again dice and cross entropy loss and record it 

### Improved 3D
- Candidate: residual blocks and/or deep supervision following the chosen reference method.
- Final modification and supporting literature to be recorded **before** implementation.
- Compare standard versus improved 3D while controlling data splits and evaluation methodology.
- Dice 0.75 is an **aspirational diagnostic reference**, not a progression gate; the primary research result is the measured baseline comparison.
- This will be final test to see if our improvements worked. If time persists then we can add in either L1/L2 Regularisation as loss functions or random forests etc to keep improving it 
## 3. Implementation and verification checklist

### A — Data contract
- [ ] Verify original image/mask pairing, alignment, orientation, and voxel spacing.
- [ ] Verify six-class label meaning and valid integer values.
- [ ] Verify patient-level separation across train, validation, and test.
- [ ] Decide preprocessing and augmentation without leakage.
- [ ] Record dataset manifest generation and reproducibility commands.
- [ ] Commit audited data-pipeline milestone with test evidence.

### B — 2D U-Net baseline
- [ ] Verify 2D model architecture and tensor dimensions.
- [ ] Unit-test loss, metrics, forward/backward pass, and six-class outputs.
- [ ] Confirm correct slice-to-mask alignment.
- [ ] Complete tiny-data overfit/smoke test before full training.
- [ ] Train baseline on GPU; select checkpoints using validation only.
- [ ] Save config, seed, logs, learning curves, model settings, and validation results.
- [ ] Review, explain, and commit baseline implementation/results.

### C — Standard 3D U-Net
- [ ] Decide patch size/batch size using measured memory requirements.
- [ ] Implement 3D convolutions, encoder/decoder, and skip connections.
- [ ] Unit-test 3D input/output shapes, masks, loss, and gradients.
- [ ] Complete small patch / overfit smoke test.
- [ ] Train with the frozen patient split.
- [ ] Save validation Dice by class, runtime, GPU memory, and model parameters.
- [ ] Review, explain, and commit standard 3D milestone.

### D — Improved 3D U-Net
- [ ] Write down architectural change and rationale before coding.
- [ ] Implement modification in a separate, testable component.
- [ ] Test tensor compatibility, gradient propagation, and deep-supervision output handling if used.
- [ ] Train with comparable protocol and record metrics and resource use.
- [ ] Diagnose poor scores using train/validation curves and per-class outcomes before changing regularisation or loss.
- [ ] Review, explain, and commit improved 3D milestone.

### E — Scientific comparison
- [ ] Freeze the model-selection procedure before final test evaluation.
- [ ] Run held-out test evaluation on all selected models under a consistent metric definition.
- [ ] Calculate per-class Dice and mean foreground Dice; report IoU and pixel accuracy with class imbalance caveats.
- [ ] Record peak GPU VRAM, parameter count, and inference latency.
- [ ] Prepare visual ground-truth-versus-prediction overlays / 3D visualisations.
- [ ] Analyse 3–5 representative failure cases and likely causes.
- [ ] Explain whether extra 3D context and architectural changes justified additional cost.

### F — Git, report, and submission
- [ ] Commit meaningful milestones progressively, with truthful messages and verification evidence.
- [ ] Place final assessed code under the required `recognition/` folder and ensure required files are present.
- [ ] Open one PR against upstream `topic-recognition`; keep it updated as work progresses.
- [ ] Include the original feasibility review and a transparent record of design changes.
- [ ] Write reproducibility instructions, plots, benchmarking, limitations, and conclusions in README.
- [ ] Add mandatory **Artificial Intelligence Usage Disclosure**, including tasks assisted and how outputs were audited.
- [ ] Export required README PDF and verify final submission steps.
- [ ] Mark PR ready for review by the deadline and retain code-review follow-up responsibilities.

## 4. Design decision log (complete for each meaningful change)

- **Date / milestone:**
- **Problem or observation:**
- **Alternatives considered:**
- **Decision and why:**
- **Expected outcome / trade-off:**
- **Validation evidence:**
- **Actual result:**
- **Commit / PR reference:**

## 5. AI-assisted development policy

AI will be used for grunt work. But I am responsible for architectural design, code review any conflicts, understanding model outputs, direction the model should be heading in. I will be in charge of prompting to do the manual code but with my guidance and I will be micromanaging it if it starts going in a different direction. I will be in charge of final report and no AI will be used for that. So only thing AI will be used for is grunt physical coding with my oversight as this takes time. I will not blindly merge or commit 
