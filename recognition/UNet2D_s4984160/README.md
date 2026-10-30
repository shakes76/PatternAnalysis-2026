# Standard 2D U-Net: baseline comparison

This directory contains the 2D data loader, standard U-Net, training and
frozen-checkpoint evaluation/prediction scripts. Training job 716367 completed
at epoch 13; its selected epoch-5 mean validation prostate Dice is **0.834130**.
**Held-out native 2D test Dice is 0.825359**, meeting the 0.75 mean target.
On the shared grid, 2D Dice is 0.823198 versus 3D 0.773216. These scopes are
reported separately in the [completed result record](support/saved_results/test_results/README.md).

## Folder guide

```text
UNet2D_s4984160/
├── dataset.py
├── modules.py
├── train.py
├── evaluate.py
├── predict.py
├── README.md
└── support/
    ├── evaluation_metrics.py
    ├── profile_inference.py
    ├── patient_split.json
    ├── submit_jobs/
    ├── check_code/
    ├── how_to_run/
    ├── saved_results/
    │   ├── training_results/
    │   └── test_results/
    └── old_code_and_notes/
```

## Training

The training loop retains the author's train/validate/save structure, Adam
and initial learning rate 0.001. The default objective is equal-weight CE
and soft foreground Dice (classes 1..5), masking label 255 from both terms.
Use `--dice-weight 0` for CE only. No class frequencies or test scores are
used to set loss weights. Validation accumulates a six-class confusion matrix
for each patient/week scan over all 60 supplied slices, then averages each
class's scan Dice. Both-empty scan/class pairs are undefined and excluded;
false positives on an otherwise absent class score zero. Best model selection,
LR reduction and early stopping use **mean scan prostate Dice (class 5)**.
Pooled class Dice, valid scan counts, precision/recall and volume ratio are
recorded separately. These native 2D partial-band scores are not equivalent
to full-volume 3D scores.

Formal runs traverse the entire training and validation sets. The test loader
is constructed by the shared helper but never iterated. Defaults: 30 maximum
epochs, batch size 16, base width 16, minimum 10 epochs, early-stop patience 8;
LR halves after 4 scheduler patience epochs without improvement, down to 1e-6.
Seed 42 initializes Python/PyTorch RNGs; exact cross-hardware reproducibility
is not guaranteed. CPU thread default 4 is a runtime option, not a Slurm CPU
resource request. No job is submitted automatically.

```powershell
# From the repository root; choose the actual local dataset path.
.\.venv\Scripts\python.exe recognition/UNet2D_s4984160/train.py --data-dir D:/HipMRI --output-dir recognition/UNet2D_s4984160/results/smoke --device cpu --epochs 1 --smoke-steps 2
# After checking real-data throughput, omit --smoke-steps for full training.
.\.venv\Scripts\python.exe recognition/UNet2D_s4984160/train.py --data-dir D:/HipMRI --output-dir recognition/UNet2D_s4984160/results/baseline --device cpu
```

Recorded cluster training curve (13 epochs; best checkpoint at epoch 5):

![Training loss and validation prostate Dice](support/saved_results/training_results/training_plot.png)

Original CSV/configuration and curve interpretation are in the
[training record](support/saved_results/training_results/README.md).

The output directory must be new. Outputs include `config.json`, the exact
`patient_split.json`, a source snapshot, per-epoch `history.csv`,
`best_model.pt`, `last_model.pt` and `training_summary.json`. Checkpoints
include model configuration, model/optimizer/scheduler states, selected metric
and validation results; this script does not implement exact resume. Smoke
outputs are explicitly marked and must never be reported as performance.
`training_plot.png` is generated if Matplotlib is installed; missing plotting
dependencies do not prevent training. Loss curves are valid-pixel-weighted
averages of batch objectives; soft Dice loss itself is batch aggregated.

Do not infer success from completed epochs: inspect the recorded prostate
validation Dice and then evaluate the selected checkpoint once on held-out
patients. The recorded run reached validation Dice 0.834130; the subsequent
independent test reached native mean Dice 0.825359.

## Cluster handoff and 3D comparison

The frozen training snapshot was submitted as Rangpur job **716367** using
`support/submit_jobs/train_baseline.sbatch` (remote name `train_baseline_v1.sbatch`); see
[CLUSTER_RUN.md](support/how_to_run/CLUSTER_RUN.md) for the submitted configuration and query commands.
The run completed successfully in 17 minutes 31 seconds on a100-8. Epoch 5
was best; eight subsequent epochs did not improve the validation prostate Dice.
The smoke run and formal training have separate output directories.

The final report and comparison tables are maintained in the
[3D project README](../UNet3D_s4984160/README.md). Use the fixed
[comparison protocol](../UNet3D_s4984160/support/how_to_run/comparison_protocol.md) for the completed
shared-grid metrics. Native 2D 60-slice scores and aligned 30-plane scores
have separate tables. Improved 2D U-Net is only an optional empty comparison
row; it is not implemented or scheduled by this handoff. Do not launch another
training job merely to populate that optional row.

## Frozen test evaluation and paired figures

[`EVALUATION_USAGE.md`](support/how_to_run/EVALUATION_USAGE.md) documents `evaluate.py`,
`support/evaluation_metrics.py`, `predict.py` and `support/submit_jobs/evaluate_final.sbatch`. The selected
epoch-5 checkpoint is hash-locked; an exact read-only copy is preserved before
submission. Native 60-slice 2D metrics and shared 30-plane metrics are separate.
Shared GT must match the 3D grid exactly. The frozen 3D checkpoint is inferred
again only to obtain matched-region distances/confusion counts, with its
overlap counts checked against the archived reference. No training is repeated.
The five paired figures reuse the cases and positions reviewed for 3D.

`predict.py evaluate <evaluate.py arguments>` supports checkpoint loading and
inference followed by plotting. `predict.py render` uses already cached arrays
without another inference pass. Raw caches and weights remain private under
the ignored `results/` directory.

## Model

`UNet2D(in_channels=1, num_classes=6, base_features=16)` uses four max-pooling levels, double convolutions with BatchNorm/ReLU, transposed-convolution upsampling and skip concatenations. It returns raw logits at the input spatial size; do not apply softmax before CrossEntropyLoss. The default width was reduced from the author's draft width of 64 to 16, with no architectural enhancements. The recorded GPU run used width 16, now verified in the selected checkpoint; `base_features=64` restores the original draft width. Training history and measured segmentation scores are archived in `support/saved_results/`. Record the width in saved checkpoint configuration so evaluation reconstructs the same model.

### Completed resource measurement

Job **717015** measured the frozen 2D model separately from 3D on the same
A100-PCIE-40GB. All 60 slices per case (batch 16, FP32) take **18.63 ms**
forward-only or **21.75 ms** including transfers, argmax and output copy;
these are medians across 27 per-case medians, each with three repeats.
Peak allocated/reserved memory is **0.2710 / 0.3418 GiB**, including resident
weights. Disk loading, preprocessing, model loading and training are excluded.
See the [raw records and matched 3D measurements](../UNet3D_s4984160/support/saved_results/resource_results/README.md)
for protocol, limitations and the successful direct 3D prediction example.

```python
from modules import UNet2D

model = UNet2D()
logits = model(image)  # [B,6,256,144] for the loader output
loss = criterion(logits, mask)  # criterion must ignore label 255
```

The example uses `image`, `mask` and `criterion` from the loader example below. Call `model.train()` for training and `model.eval()` with `torch.inference_mode()` for validation/prediction because BatchNorm behaves differently in each mode. Masks use labels 0..5; padding label 255 must also be excluded from Dice loss and all metrics.

## Data layout

Pass either `/home/groups/comp3710/HipMRI_Study_open` or its `keras_slices_data` directory. MRI and mask files reside in separate sibling folders:

```text
keras_slices_data/
  keras_slices_train/          keras_slices_seg_train/
  keras_slices_validate/       keras_slices_seg_validate/
  keras_slices_test/           keras_slices_seg_test/
```

Files are paired by patient, week and numeric slice index. Missing, orphan or duplicate files raise an error instead of being silently skipped. A prepared flat local directory with paired `case_...` and `seg_...` files is also supported.

## Patient separation

`support/patient_split.json` preserves the existing 3D experiment's split: sort the original letter-prefixed 3D patient IDs, shuffle using Python `random.Random(42)`, then select 26/6/6 patients. The verified unique numeric suffix maps those IDs to 2D filenames. Do not sort numeric IDs and take the first 26, or use the provided directory split as our experimental split.

The loader indexes all supplied source folders, then applies this manifest without moving data. With the complete dataset, expected slice-pair counts are 8,700 training, 2,340 validation and 1,620 test. Passing a different manifest is explicit; overlapping, missing or unassigned patients are rejected.

## Preprocessing and usage

Each original MRI slice is Z-score normalized using its original pixels before padding. The mask must contain integer labels 0..5. Shapes `(256,128)` and `(256,144)`, including a trailing singleton dimension, are accepted. Smaller widths are padded on the right to `(256,144)`; MRI padding is zero and label padding is **255**. The optional transform jointly processes image and mask before padding, and must preserve shape and integer labels. Validation/test transforms are disabled by the loader helper.

```python
from dataset import IGNORE_INDEX, get_dataloaders
import torch

train_loader, val_loader, test_loader = get_dataloaders(
    '/home/groups/comp3710/HipMRI_Study_open',
    batch_size=16,
    num_workers=0,
)
criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
image, mask, metadata = next(iter(train_loader))
# image: float32 [B,1,256,144]; mask: int64 [B,256,144]
```

Every loss and metric must exclude `mask == IGNORE_INDEX`. Remove the recorded right padding before restoring predictions to the original image grid. Raw mask values are checked before integer conversion to avoid silently truncating fractional labels. Per-slice normalization here differs from the 3D pipeline's per-volume nonzero-voxel normalization; report this preprocessing difference when interpreting a future 2D/3D comparison.

## Spatial mapping

The directory audit matched all 211 scan keys to 60 numbered slices each. Pixel checks across 69 different slice pairs from four scans found:

```text
slice_2d[i] = original_volume[::-1, 98+i, :], i=0..59
```

The original 3D headers use L/P/S axes; the matched planes are coronal. The slice metadata records this source axis/index/flip. This rule was content-checked on those samples, not every slice. Sampled 2D headers have identity affines; they do not independently provide the original physical geometry. Use original 3D affines for distances and alignment. Two-dimensional data cover only the central 60 planes; future comparisons must use a shared spatial region/grid rather than compare partial coverage with full-volume scores.

## Checks and provenance

Run `python -m unittest discover -s recognition/UNet2D_s4984160/support/check_code -v` from the repository root with NumPy, NiBabel and PyTorch installed. Tests create synthetic NIfTI fixtures; they do not use patient images. Synthetic tests check loading behavior, not full real-data correctness or model performance.

The loader, model and training loop were adapted from the author's submitted drafts, with AI assistance reviewing directory pairing, explicit split handling, normalization/padding, input validation, model width, scan-level metric aggregation, checkpointing and checks. General CE/Adam/early-stopping ideas were also discussed in the author's linked reference conversation; the linked friend's implementation was not used. The author's original attachments were preserved. No friend's implementation, model weights, experimental scores or report text are included. Final project documentation will need the full AI-use disclosure, exact environment versions and measured experimental results.

## Completed test result

Evaluation 716656 saved all five paired figures and both completion markers.
The shared GT matched for all 27 scans. 2D improves scan-average Dice in 21/27
cases, but has more rectal invasion (400 versus 21 shared voxels) and does not
win every patient. Keep the selected checkpoint frozen. See the
[result and case analysis](support/saved_results/test_results/README.md) and the
[main comparative report](../UNet3D_s4984160/README.md).
