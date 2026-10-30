# ConvNeXt V2: bounded development phase after V1 test exposure

This plan must be committed before starting these experiments. It is a new,
explicitly exploratory development phase. V1 and its completed test remain intact.
The previously exposed test is NOT a fresh held-out evaluation, regardless of
future performance. No V2 test-inference interface is supplied. Training/validation
manifests and patient assignments are unchanged; test rows are checked only for
manifest/group integrity by the original loader. Do not inspect or optimise against
the V1 patient-level test failures while developing these experiments.

## Purpose and hypotheses

V1 ConvNeXt fit training slices almost perfectly but generalized poorly. A scoped
source review found the expected Tiny depths, widths, 7x7 depthwise convolution,
LayerNorm, LayerScale and stochastic depth; no single decisive coding error was
established. This is not a proof that the training configuration is optimal.

Three fixed recipes, not an automatic search:

1. **scratch_augmented**: original custom ConvNeXt-Tiny, random initialization,
   original optimizer settings, plus uniformly sampled training-only rotations
   from -5 to +5 degrees. Validation geometry/normalisation is unchanged. Inspect
   the private preview: rotation may clip peripheral pixels; preservation of all
   anatomy is NOT guaranteed by the existing rotation implementation. No flips,
   random crops, elastic warps or validation augmentation are introduced.
2. **scratch_patient**: same recipe plus per-slice weights
   `N / (P * number_of_scans_for_patient * number_of_slices_for_scan)`.
   Each patient contributes the same total loss and each of their scans contributes
   equally. All training slices are visited once per epoch; no sampling duplication.
   This is not AD/NC class balancing. It changes the effective class weighting from
   approximately 49% AD by slices to approximately 33% AD by patients. AD recall may
   decrease; report it, not just accuracy. No claim that weighting must improve results.
3. **transfer_patient**, OPTIONAL, only with real course permission: retain custom
   PyTorch blocks; strictly map the torchvision `ConvNeXt_Tiny_Weights.IMAGENET1K_V1`
   backbone into them; replace the 1000-class head with a fresh two-class head.
   Keep the three-channel pretrained stem; recover [0,1] from existing [-1,1]
   greyscale tensors, repeat channels and apply ImageNet mean/std. Do not crop the
   MRI to reproduce the default torchvision preset. Train only the head for two
   epochs, then fine-tune all layers; backbone maximum LR 1e-5, head maximum 3e-4.
   This is a multi-setting transfer-learning intervention, NOT evidence that
   pretraining alone caused any change. No downloaded model code implements the
   network; torchvision supplies the approved state dictionary only. The package
   does not download weights during installation, testing, preview or smoke checks.
   A user's approval assertion is recorded, not independently authenticated.

## Common settings

Seed 42, float32, no mixed precision/TF32, batch 8, accumulate 4 (effective 32),
AdamW weight decay .05 only on matrix/kernel weights, clipping norm 5, three-epoch
warmup and cosine decay toward 1e-6. At most 30 epochs, minimum 12 before patience
stopping, patience 8. No new arbitrary wall-time cutoff; keep the laptop powered.
Workers=0 to simplify repeatable epoch recovery. Limit initial work to one run per
recipe; no opportunistic seed or hyperparameter sweeps. If time is limited, execute
scratch_augmented first and review its evidence before starting the next recipe.
Approval for transfer can be obtained while scratch experiments run.

Checkpoint selection changes explicitly from V1 patient macro-F1 to patient
accuracy at threshold .5, then macro-F1, then lower patient log loss; earliest
exact tie retained. The primary target motivates this declared change. Accuracy
alone is not clinical utility. Report precision/recall for both classes, macro-F1,
AUROC, confusion matrix, and slice/scan/patient results. No threshold search,
calibration fitting, ensemble, test-time augmentation or test evaluation is done.
Do not overwrite or reinterpret the V1 checkpoint selection. Epoch-specific
shuffle seeds are used for resumability, and the former three-hour soft limit is
removed. Therefore V1-versus-A1 is not a perfectly isolated augmentation ablation:
the optimizer hyperparameters match, but ordering/selection/stopping also differ.
A1-versus-A2 uses the same V2 schedule, seeds and selection to isolate the additional
loss weighting more directly.

Patient score = mean of scan means of slice P(AD), unchanged from V1. Training
weighted per-slice cross-entropy is still a surrogate, not direct optimization of
that final patient probability. Patient-equal weighting uses training records only.

## Resumption and provenance

New files only; original code, V1 source hashes, evidence, plan, checkpoints and
pointers are untouched. A V2 run directory holds full current and best model
states, AdamW states, Python/NumPy/torch CPU/CUDA RNG states, completed epoch,
schedule state, selection counters and complete history. Save after each completed
epoch by fsync and atomic replacement; retain one prior checkpoint. Recovery can
repeat the unfinished epoch (or one earlier epoch if the latest checkpoint was
corrupt). No exact bitwise guarantee across different hardware/software; resume
checks recorded environment, code hashes and manifest hashes. Process file locks
are released automatically by the OS on exit. Never run two processes for a recipe.

Do not import a V1 checkpoint as if it were a resumable V2 training state. V1
lacks the necessary optimizer/RNG state. The optional transfer weights are
ImageNet weights, never a V1 test-selected checkpoint. Model weights and patient-
linked predictions/checkpoints stay private, outside Git and OneDrive.

## Reporting and further decisions

These are **selected validation results**, repeatedly reused for development.
Passing 0.80 here is not passing the task's test-accuracy requirement. Keep every
attempt and its configuration. A promising recipe should be checked for
seed sensitivity or further patient-grouped development validation before claiming
robustness. Additional experiments require a bounded documented plan, not repeated
use of exposed test scores. Refit calibration/referral rules on development data
if a new model is selected. A tutor must clarify the revised evaluation protocol
and pretrained-weight acceptability; no waiver is assumed. Do not randomly
reshuffle the old data to manufacture an "untouched" test.

Accuracy target reference: supplied COMP3710 brief §2.1 (ConvNeXt, >=0.80).
Architecture: Liu et al., *A ConvNet for the 2020s*, CVPR 2022,
https://arxiv.org/abs/2201.03545 (course reference [7]).
Transfer provider: installed torchvision's named ImageNet1K V1 weights; source:
https://github.com/pytorch/vision/blob/v0.22.1/torchvision/models/convnext.py .
This is a pinned implementation reference, not a claim about the newest library.

ChatGPT assisted with this plan, code and tests. The student must review the code,
run local checks, understand its decisions and accurately disclose assistance.
