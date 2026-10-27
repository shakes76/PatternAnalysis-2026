# Feasibility Review: 3D Improved U-Net for HipMRI Prostate MRI Segmentation

**Marcus Wee Lam Seow, s48747240** | COMP3710 Pattern Recognition | Hard difficulty (3D Improved U-Net, downsampled HipMRI 3D data)

## 1. User need, scope and acceptance criteria

**User.** A radiotherapy treatment planning specialist who currently contours the prostate and surrounding organs-at-risk by hand on each MRI volume.

**Purpose.** A prototype that proposes draft 3D segmentation masks (six labels: background plus five anatomical structures) for the specialist to review and edit. It assists manual contouring and does not replace it.

**Scope.** Downsampled volumes (128x128x64) from the HipMRI `semantic_MRs` and `semantic_labels_only` folders on Rangpur.

**Acceptance criteria.**
1. Dice >= 0.7 on every foreground class on a held-out test set.
2. Strict patient-level separation: every scan of a patient lies in exactly one of train, validation and test, enforced by an assertion.
3. Per-class Dice reported per test volume, with 3 to 5 failure cases inspected visually.
4. Resource limits: peak training memory within one A100 (40 GB), inference under 1 second per volume, a full training run within a few GPU-hours.
5. The Improved U-Net compared against a plain 3D U-Net on the same split.

## 2. Model choice and course concepts

- **Model.** 3D Improved U-Net (Isensee et al., 2018): residual context modules, instance normalisation with leaky ReLU, strided-convolution downsampling, localization modules and deep supervision.
- **Baseline.** Plain 3D U-Net (Cicek et al., 2016) with skip connections, batch norm and max pooling.
- **Loss.** Per-class soft Dice plus cross-entropy. Small structures such as class 5 occupy about 0.1% of voxels, so a global Dice would be dominated by background.
- **Data handling.** Downsampling by 2 in each axis to fit GPU memory (images block-averaged, labels nearest-neighbour). Images held as float16 and labels as uint8, with one-hot encoding per batch to limit RAM use.

## 3. Preliminary feasibility evidence

- **Data audit.** 211 MR volumes pair one-to-one with 211 label volumes from 38 patients. Voxel spacing is about 1.56 mm in all axes. 210 volumes are 256x256x128 and one is 256x256x144 (centre-cropped to 128). Labels take values 0 to 5. In the first label volume, class 5 covers about 0.12% of voxels.
- **Model build.** Both models build with output shape (128, 128, 64, 6): 5,650,886 parameters (baseline) and 8,267,618 (Improved U-Net).
- **Smoke test.** A 16-volume, 2-epoch run on one A100 trained at about 1 s per step (batch size 2) after an 87 s first-epoch compile. Peak GPU memory was 3.32 GB. Loss fell from 2.02 to 1.02 and mean foreground Dice rose from 0.11 to 0.21.
- **Time estimate.** About 70 steps per epoch at roughly 0.1 to 0.15 s per step puts a full run at well under a GPU-hour, even for 100 epochs.

## 4. Risks, budget and fallback

| Risk | Mitigation |
|---|---|
| Few patients (38), so a small, noisy test set | Patient-level split; report per-class Dice and state the limitation |
| Severe class imbalance for small structures | Per-class Dice loss plus cross-entropy |
| Downsampling removes detail of small structures | Nearest-neighbour label sampling; state that Dice is at reduced resolution |
| Memory pressure and node crashes seen on the cluster | float16 images, uint8 labels, per-batch one-hot encoding; exclude a crashing node |
| Long queue waits | Smoke test first, short time limits, dependent prediction jobs |

**Compute budget.** One A100 on partition `comp3710`: a 20-minute smoke job, then one training job and one prediction job per model.

**Next planned experiment.** Full runs (up to 100 epochs, early stopping) of both models on the patient-level split.

**Fallback.** If the Improved U-Net did not train reliably, report the plain 3D U-Net as the delivered model and describe the Improved U-Net as incomplete.
