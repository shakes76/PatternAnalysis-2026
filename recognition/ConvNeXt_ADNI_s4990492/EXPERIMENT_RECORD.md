# ADNI Classification with ConvNeXt

## Project status

Updated 27 October 2026. Training, validation analysis, fixed-checkpoint held-out evaluation and local resource profiling are complete. The final custom ConvNeXt with ImageNet initialization achieved 81.7686% slice-level test accuracy; the SimpleCNN baseline achieved 71.1572% on the same patient-disjoint test split. These results use the disclosed replacement split, not the original folder split. Check-off was completed on 20 October 2026, as confirmed by the student; the historical review and tutor feedback are recorded below. Report and submission packaging remain in progress.

## Objective and model principles

Classify Alzheimer's disease (AD) versus normal control (NC, called CN in the assignment) MRI slices, and investigate whether prediction confidence can identify errors for human review. Compare classification performance, confidence reliability and computational cost.

The baseline uses three convolution/ReLU/pooling blocks, global average pooling and a two-class linear head (23,426 parameters). ConvNeXt-Tiny uses depthwise spatial convolution, LayerNorm, channel expansion/projection, GELU and residual connections. Stage depths are [3, 3, 9, 3] and widths [96, 192, 384, 768]. The implementation adapts the input to one grayscale channel and the output to two classes (27,818,594 parameters). These initial models were trained from scratch without stochastic depth. Later controlled experiments added stochastic depth, a smaller model, RGB224 preprocessing and ImageNet initialization. The final custom three-channel ConvNeXt contains 27,821,666 parameters and is fine-tuned from mapped official ImageNet weights; its architecture remains defined in `modules.py`. Architecture source: [Liu et al., A ConvNet for the 2020s, CVPR 2022](https://arxiv.org/abs/2201.03545) and [author implementation](https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py).

## Data audit and preprocessing

Course data: `/home/groups/comp3710/ADNI/AD_NC`, with `meta_data_with_label.json` in the parent directory. The audit matched all 30,520 images to metadata and identified 680 subjects: 221 AD and 459 NC, with no subjects assigned to both classes in this collection. Original JSON labels AD=2 and NC=0 are mapped to model targets AD=1 and NC=0.

The supplied training/test directories share 216 subjects despite having no common scan IDs. A fixed class-stratified split groups all scans and slices from a subject together, using seed 42:

| Split | Subjects | Scans | Images |
|---|---:|---:|---:|
| Training | 475 | 1,068 | 21,360 |
| Validation | 101 | 229 | 4,580 |
| Test | 104 | 229 | 4,580 |

Subject overlap is zero. Source images are not moved. The loader converts images to grayscale float32 tensors in [0,1], retaining dimensions; sampled inputs were [1,240,256], and both pilot runs completed with batching at this size. The original runs used no augmentation; subsequent experiments apply training-only affine augmentation as described below. Different numbers of scans per subject remain an evaluation/weighting consideration. The test split was held out during model and threshold selection. Final evaluation has now completed with fixed checkpoints and predefined thresholds; its results are not used for further tuning.

### Leakage audit completed on 22 October 2026

Student-run checks found zero pairwise overlap in patient IDs, scan IDs and paths across the three splits. The combined manifests exactly match the source manifest, with no duplicate paths, conflicting patient classes or scans assigned to multiple patients. All 30,520 images decoded successfully. No duplicate file contents or identical decoded grayscale pixels were found. These checks do not exclude every possible near-duplicate.

Review of the pasted remote code confirmed patient-grouped splitting, per-image preprocessing, validation without parameter updates and checkpoint selection by validation loss. Historical pilot configurations match the current train/validation counts, but do not contain source or manifest hashes. Test-image integrity checks did not involve model-performance evaluation.

## Environment and usage

Reported working environment: Python 3.11.15, PyTorch 2.13.0+cu130, torchvision 0.28.0+cu130, Pillow 12.3.0, NVIDIA A100-PCIE-40GB, compute-node driver 590.48.01. A complete environment export remains to be added. Fixed seeds support repeatability but do not guarantee bitwise reproducibility across hardware/software.

On Rangpur, activate the existing environment:

```bash
source "$HOME/miniconda3/bin/activate"
conda activate torch
cd "$HOME/comp3710_adni_project"
```

For a fresh working directory with these source files, generate local manifests:

```bash
python audit_data.py
python make_splits.py
```

`make_splits.py` refuses to overwrite an existing `splits/` directory. Reuse the established split rather than repeatedly changing it. Paths in generated manifests point to Rangpur data and are not portable to a local machine without those data.

Submit training through Slurm:

```bash
sbatch train_baseline.sh
sbatch train_convnext.sh
squeue --me
```

The scripts run `train.py --epochs 3 --batch-size 8` and `train.py --model convnext --epochs 3 --batch-size 8`, respectively. They currently change to `$HOME/comp3710_adni_project`; adjust that line if running elsewhere. Explicit memory requests were omitted because this partition reports only 1 MB per node to Slurm; this is a site-specific configuration, not physical memory capacity.

Read logs using the actual job ID:

```bash
cat train_baseline_JOBID.out
cat train_baseline_JOBID.err
cat train_convnext_JOBID.out
cat train_convnext_JOBID.err
```

Each run creates `runs/<model>_<timestamp>/config.json`, `training_log.csv` and `best_<model>.pth`. Checkpoints are selected by lowest validation loss. Datasets, manifests, weights and raw runtime outputs are excluded by the project `.gitignore`; generated outputs are not part of the public submission.

## Preliminary results

Both pilots used batch size 8, seed 42 and the same train/validation manifests and preprocessing. The historical baseline config records learning rate 1e-3 but omits optimizer name and weight decay; the current baseline code uses Adam with no weight decay. ConvNeXt: AdamW, learning rate 1e-4, weight decay 0.05. Optimizer differences mean this is not an isolated architecture comparison.

| Model | Epoch | Train accuracy | Validation accuracy | Validation loss |
|---|---:|---:|---:|---:|
| SimpleCNN | 1 | 53.0% | 60.4% | 0.6633 |
| SimpleCNN | 2 | 59.0% | 62.8% | 0.6394 |
| SimpleCNN | 3 | 61.8% | 66.6% | 0.6136 |
| ConvNeXt-Tiny | 1 | 52.3% | 58.4% | 0.6733 |
| ConvNeXt-Tiny | 2 | 63.5% | 53.2% | 0.7440 |
| ConvNeXt-Tiny | 3 | 72.5% | 54.7% | 0.8257 |

Baseline epoch 3 AD/NC recall: 75.2%/57.9%. ConvNeXt best-validation-loss epoch 1 AD/NC recall: 57.6%/59.2%. ConvNeXt's increasing validation loss despite improving training accuracy suggests overfitting or optimization instability; the cause has not been established.

| Model | Train + validation time/epoch | Peak PyTorch-allocated GPU memory |
|---|---:|---:|
| SimpleCNN | 29.3-35.0 s | 124.2 MiB |
| ConvNeXt-Tiny | 132.2-136.8 s | 1,405.7 MiB |

Times exclude startup and checkpoint writing. Allocated tensor memory is not total GPU process usage. Inference latency was not measured in these pilots; completed local profiling is reported below. Baseline checkpoint reload and single-image inference passed; Both selected twenty-epoch checkpoints subsequently passed strict model loading and full validation inference, as documented below.

Baseline run: `baseline_20261018_191208_445054`. ConvNeXt pilot run: `convnext_20261019_190724_687263` (job `675518`). The twenty-epoch learning curves are described below.

## Twenty-epoch experiment

Both models were trained from fresh initialization using the retained patient-independent split, batch size 8 and seed 42. The current training configuration uses Adam (learning rate 1e-3, weight decay 0) for SimpleCNN and AdamW (learning rate 1e-4, weight decay 0.05) for ConvNeXt. Checkpoints are selected by minimum validation cross-entropy, not maximum validation accuracy. The results below are from student-supplied cluster logs; selected-epoch validation loss and accuracy were also confirmed by reading the saved training CSV files.

| Model | Selected epoch | Validation loss | Validation accuracy | NC recall | AD recall |
|---|---:|---:|---:|---:|---:|
| SimpleCNN | 9 | 0.525835 | 73.78% | 76.2% | 71.4% |
| ConvNeXt-Tiny | 1 | 0.673260 | 58.54% | 59.3% | 57.8% |

SimpleCNN achieved its lowest validation loss at epoch 9 (0.5258350444315042). Its highest validation accuracy was 75.6% at epoch 14, but the predefined loss-based selection rule was retained. At epoch 20, training accuracy was 80.6% and validation accuracy was 73.6%.

ConvNeXt achieved its lowest validation loss at epoch 1 (0.6732601917987308). By epoch 20, training accuracy reached 99.2%, while validation accuracy was 56.5% and validation loss was 2.7095. The widening training/validation gap is evidence of overfitting under this configuration. This motivated the affine-augmentation experiment reported below; simply extending the same training run is not supported by these results. Optimizer settings differ between models, so the comparison does not isolate architecture as the cause of the performance difference.

| Model | Train + validation time/epoch | Peak PyTorch-allocated GPU memory |
|---|---:|---:|
| SimpleCNN | 27.4-32.9 s | 124.2 MiB |
| ConvNeXt-Tiny | 131.5-138.1 s | 1,405.7 MiB |

Times exclude startup and checkpoint writing. These are slice-level validation results, not final test results; the test split remains reserved for evaluation after model selection.

Run directories:

- `baseline_20261022_211305_056381`
- `convnext_20261022_212324_269607`

![Twenty-epoch training and validation learning curves](figures/training_curves_20ep.png)

The plotting command completed successfully on Rangpur. Copy the generated PNG into the local `figures/` directory and inspect it before committing this report and figure together.

Reproduce the figure in the activated training environment (requires Matplotlib; no GPU required):

```bash
python plot_training_curves.py \
  --baseline runs/baseline_20261022_211305_056381 \
  --convnext runs/convnext_20261022_212324_269607
```

The plot reads saved metrics only; it does not load images or evaluate a model. Dashed lines mark the minimum-validation-loss epoch for each model.

## Feasibility review

### Historical one-page review presented at check-off

**COMP3710 | s4990492 | 20 October 2026.** The following content is transcribed from the student's one-page English feasibility review, with its split table reformatted. Statements about proposed work, approval requests, budgets and unmeasured test performance describe the plan at that time, not the final project status.

#### 1. User need, scope and acceptance criteria
Build an ADNI MRI classification prototype for research reviewers: compare ConvNeXt with a simple CNN and investigate
whether confidence identifies errors well enough to refer uncertain cases for human review.
Acceptance targets: (1) Zero patient overlap across splits. (2) Test accuracy at least 0.80 (course target), with per-class
precision/recall, macro-F1 and AUROC. (3) Compare calibration and error-versus-coverage curves; select referral
thresholds on validation data only. (4) Fit a real-data training batch on one A100 40GB and measure peak allocated
memory, parameters and inference latency. (5) Explain 3-5 failure cases and the benefit/cost of the advanced model.
Targets are prospective; test performance is not yet measured.

#### 2. Model choice and course concepts
Main model: ConvNeXt-Tiny (Hard proposal), with stage depths [3, 3, 9, 3] and widths [96, 192, 384, 768], a one-channel
input stem and two-class head. Implement and explain the architecture in PyTorch, citing [1]; initially train from scratch.
Core concepts are convolution, residual learning, normalization and backpropagation. The initial plan uses no pretrained
weights.
Baseline: a three-block CNN with global average pooling (23,426 parameters). Initially use the same native 240 x 256
grayscale tensors in [0,1], split manifests and evaluation protocol for both models. Select checkpoints by validation loss.
Propose slice-level metrics plus patient-level metrics from equally weighted scan averages; confirm the required unit for the
0.80 target. Keep the test split untouched during model selection.

#### 3. Preliminary feasibility evidence
Data: 30,520 images from 680 subjects (221 AD, 459 NC); all images matched metadata and subject IDs, with no subjects
in both classes. The provided train/test folders share 216 subjects despite having no shared scan IDs. Provisional
class-stratified patient splits (seed 42) eliminate this overlap; approval is requested.

| Provisional split | Subjects | Scans | Images |
|---|---:|---:|---:|
| Training | 475 | 1,068 | 21,360 |
| Validation | 101 | 229 | 4,580 |
| Test | 104 | 229 | 4,580 |

Baseline evidence: on A100 40GB, three epochs reduced validation loss from 0.6633 to 0.6136 and raised slice-level
validation accuracy from 60.4% to 66.6%. Final AD/NC recall: 75.2%/57.9%. Train-plus-validation time: 29-35 s/epoch;
peak PyTorch-allocated memory: 124.2 MiB (not total GPU usage). Checkpoint reload and inference passed. ConvNeXt
pilot: 27,818,594 parameters; three epochs on the same split reached best validation accuracy 58.4% and loss 0.6733 at
epoch 1. Training accuracy rose 52.3% to 72.5%, but validation loss rose to 0.8257, suggesting overfitting or optimization
instability. Runtime: 132-137 s/epoch; peak allocated memory: 1,405.7 MiB. The baseline currently performs better;
optimizer settings differ, so this is a preliminary comparison, not an isolated architecture effect.

#### 4. Risks, budget and fallback
Risks: repeated scans can overweight some patients; limited independent subjects can cause overfitting and unreliable
confidence. Report patient-level results, monitor validation loss, and evaluate calibration on validation data. The
replacement patient-level split requires tutor confirmation.
Budget and next experiment: one A100; the three-epoch pilot used about 6.7 minutes for training and validation
(excluding startup/checkpointing). Reserve up to 1 GPU-hour for a controlled follow-up after consultation: examine
validation errors, then test learning-rate or regularization changes one at a time, retaining the best validation checkpoint.
Keep the test set untouched. Reduce batch size if memory becomes limiting; any architecture/topic fallback will be
discussed with the tutor.
[1] Liu et al., A ConvNet for the 2020s, CVPR 2022. arxiv.org/abs/2201.03545. Evidence: student-run Rangpur audit and training logs;
baseline run 20261018_191208_445054.

### Tutor feedback and subsequent decisions

The student confirmed completion of check-off on 20 October 2026 and recalled one recommendation: construct the validation set from the training data. No additional feedback or explicit approval of the replacement split is recorded.

The implemented protocol pools the original directories and then partitions by patient, because 216 patients occur in both original folders. This differs from constructing validation solely from the original training folder. The fixed split and its zero-overlap audit are disclosed above. Check-off completion is not evidence that the tutor explicitly approved this replacement protocol; results must be interpreted under the stated split.

Subsequent teaching clarification established slice-level accuracy as the required measure. Patient-level aggregation was therefore deferred rather than substituted for the primary metric. The project expanded beyond the initial one-hour follow-up proposal: augmentation, learning-rate, stochastic-depth, cosine-schedule, reduced-width and matched initialization experiments were completed. The final ImageNet-initialized model uses the student's architecture and is supported by a matched scratch comparison and source attribution. Final evaluation and laptop profiling replace the prospective measurements in the historical review. These later decisions are documented separately so the original check-off proposal remains identifiable.


## Training-only affine augmentation results

This experiment adds random rotations within +/-5 degrees and translations within 3% of image width/height to training images only, with bilinear interpolation and zero fill. No flips, random crops or scaling are added.

The patient split, model architecture, optimizer settings, batch size 8, seed 42 and 20-epoch budget remain unchanged. Validation preprocessing remains deterministic. Training starts from fresh initialization; checkpoint selection uses minimum validation loss. Test performance is not used to select augmentation settings. Training metrics in this experiment are measured on augmented inputs, unlike the earlier unaugmented training metrics.

New entry points: `dataset_aug.py`, `train_convnext_aug.py` and `train_convnext_aug.sh`. Outputs use separate `runs/convnext_aug_<timestamp>/` directories. Previous scripts and results are retained.

The run `convnext_aug_20261023_161053_071278` completed 20 epochs. Minimum validation loss selected epoch 12: loss 0.6243266117852588, validation accuracy 68.1%, NC recall 68.8% and AD recall 67.4%. These values come from student-supplied cluster logs. The selected checkpoint improves on the unaugmented ConvNeXt run but remains below SimpleCNN. The highest validation accuracy was 69.1% at epoch 18; the loss-based selection rule was retained.

At epoch 20, augmented training accuracy was 88.6%, validation accuracy 65.8% and validation loss 0.8484. Validation loss generally increased after epoch 12 while training loss decreased, indicating continued overfitting. This single-seed result does not establish improvement across seeds. Epoch times were 158.4-163.4 seconds, excluding startup and checkpoint writing; peak PyTorch-allocated GPU memory was 1,405.7 MiB.

```bash
sbatch train_convnext_aug.sh
# Replace JOBID with the returned job number.
squeue -j JOBID
cat train_convnext_aug_JOBID.out
cat train_convnext_aug_JOBID.err
```

## Lower-learning-rate augmentation results

This controlled experiment reduced the ConvNeXt AdamW learning rate from 1e-4 to 3e-5. Affine augmentation, model architecture, weight decay 0.05, batch size 8, seed 42, patient split and the 20-epoch budget remain unchanged. Training starts from fresh initialization. Validation remains unaugmented and checkpoint selection uses minimum validation loss. Test performance is not used for selection.

Entry points: `train_convnext_aug_lr3e5.py` and `train_convnext_aug_lr3e5.sh`, sharing `dataset_aug.py`. The run completed 20 epochs. Outputs use `runs/convnext_aug_lr3e5_<timestamp>/` to preserve previous experiments.

```bash
sbatch train_convnext_aug_lr3e5.sh
# Replace JOBID with the returned job number.
squeue -j JOBID
cat train_convnext_aug_lr3e5_JOBID.out
cat train_convnext_aug_lr3e5_JOBID.err
```

The run `convnext_aug_lr3e5_20261024_043935_000396` selected epoch 6 by minimum validation loss: 0.5767051260723167, with validation accuracy 67.0%, NC recall 62.1% and AD recall 71.8%. Accuracy and recalls are rounded values from student-supplied logs. At epoch 20, training accuracy was 83.7%, validation accuracy 64.8% and validation loss 0.7844. Epoch times were 159.3-163.9 seconds; peak PyTorch-allocated GPU memory was 1,405.7 MiB. Times exclude startup and checkpoint writing.

Compared with augmentation at learning rate 1e-4, the selected validation loss improved while selected-checkpoint accuracy decreased slightly. The predefined loss-based selection rule was retained. Lower cross-entropy alone does not establish good confidence calibration. SimpleCNN still achieved lower validation loss and higher validation accuracy.

### Validation-loss-selected comparison

| Configuration | Selected epoch | Validation loss | Validation accuracy | NC recall | AD recall |
|---|---:|---:|---:|---:|---:|
| SimpleCNN | 9 | 0.525835 | 73.78% | 76.2% | 71.4% |
| ConvNeXt, no augmentation | 1 | 0.673260 | 58.54% | 59.3% | 57.8% |
| ConvNeXt, affine augmentation, LR 1e-4 | 12 | 0.624327 | 68.1% | 68.8% | 67.4% |
| ConvNeXt, affine augmentation, LR 3e-5 | 6 | 0.576705 | 67.0% | 62.1% | 71.8% |

These are validation results used during development, not final test estimates. The checkpoints evaluated below are the SimpleCNN epoch-9 and augmented LR-3e-5 ConvNeXt epoch-6 checkpoints. No test performance was used for selection.

### Checkpoint file verification (24 October 2026)

The student successfully read both selected checkpoint files on CPU using `torch.load(..., map_location="cpu", weights_only=True)`:

| Run | Saved model | Saved epoch | Saved validation loss | Learning rate | State-dictionary entries |
|---|---|---:|---:|---:|---:|
| `baseline_20261022_211305_056381` | SimpleCNN | 9 | 0.5258350444315042 | 0.001 | 8 |
| `convnext_aug_lr3e5_20261024_043935_000396` | ConvNeXtTiny | 6 | 0.5767051260723167 | 0.00003 | 182 |

Both record `validation_loss` as the selection metric. This initial check verified file deserialization and agreement with the training logs. Subsequent strict model loading, forward inference and aggregate validation-loss reproduction succeeded, as reported below.

## Validation evaluation

`predict.py` strictly loads a saved checkpoint and evaluates only the fixed validation manifest without augmentation or parameter updates. It reports cross-entropy, accuracy, balanced accuracy, per-class precision/recall/F1, macro-F1, AD-positive AUROC and a confusion matrix (rows: true; columns: predicted; order: NC, AD). Predictions use logit argmax. Undefined precision is reported as zero; AUROC requires both classes and groups tied scores.

`evaluate_validation.sh` evaluates the selected SimpleCNN and ConvNeXt checkpoints. Aggregate results are saved in each run's `validation_metrics.json`, including the validation-manifest hash and difference from the recorded checkpoint loss. Existing results are not overwritten. No patient identifiers or image paths are included in this JSON. Probabilities are not yet calibrated. Both evaluations completed successfully; this version does not implement final test evaluation or patient-level aggregation.

```bash
sbatch evaluate_validation.sh
# Replace JOBID with the returned job number.
squeue -j JOBID
cat validation_JOBID.out
cat validation_JOBID.err
```

The `validation_*.out` and `validation_*.err` files are local job logs; do not select them for Git commits.

### Reproduced validation results (24 October 2026)

Student-supplied evaluation outputs confirm successful strict checkpoint loading and inference on the same 4,580 validation slices from 101 patients. Both models exactly reproduced the loss recorded in their checkpoints (`loss_difference = 0.0`). Both reported the audited validation manifest SHA-256: `e1217d294d7d0c93749e7661b1dff3d929488a65ad1ab8aecbad19498e16945a`. Evaluation used CUDA, batch size 8 and PyTorch 2.13.0+cu130.

| Metric | SimpleCNN, epoch 9 | ConvNeXt + augmentation, LR 3e-5, epoch 6 |
|---|---:|---:|
| Cross-entropy | 0.525835 | 0.576705 |
| Accuracy | 73.78% | 67.01% |
| Balanced accuracy | 73.79% | 66.99% |
| Macro-F1 | 0.7377 | 0.6692 |
| AD-positive AUROC | 0.8157 | 0.7520 |
| NC precision | 72.53% | 68.62% |
| NC recall | 76.18% | 62.15% |
| NC F1 | 0.7431 | 0.6522 |
| AD precision | 75.15% | 65.69% |
| AD recall | 71.39% | 71.83% |
| AD F1 | 0.7322 | 0.6862 |

Class supports are 2,280 NC and 2,300 AD slices. Confusion matrices use true classes as rows and predicted classes as columns:

| Model | True NC, predicted NC | True NC, predicted AD | True AD, predicted NC | True AD, predicted AD |
|---|---:|---:|---:|---:|
| SimpleCNN | 1,737 | 543 | 658 | 1,642 |
| ConvNeXt | 1,417 | 863 | 648 | 1,652 |

At the default argmax rule, ConvNeXt correctly classified 10 more AD slices but misclassified 320 more NC slices as AD than SimpleCNN. SimpleCNN achieved higher overall accuracy, macro-F1 and AUROC. AUROC 0.8157 is not 81.57% classification accuracy and does not establish the assignment's test-accuracy target.

These are slice-level validation results used during development, not independent final test estimates or patient-level metrics. Confidence probabilities remain uncalibrated. Next steps are slice-level validation prediction export and confidence/referral analysis before final test evaluation. Patient aggregation is optional and deferred.

## Deferred optional patient aggregation

This optional workflow is deferred following teaching-team clarification; the slice-only workflow below is the current priority. The previously proposed `export_validation.py` workflow exports validation logits and probabilities for confidence analysis. Patient predictions average slice probabilities within each scan, then average scan probabilities within each patient. This aggregation rule is fixed before inspecting patient-level results. Predictions use argmax, with exact ties assigned to NC.

Private `slice_predictions.csv` and `patient_predictions.csv` files remain under each run's `validation_export/` directory and must not be committed because they contain patient and scan identifiers. `summary.json` contains aggregate metrics only.

The export also recomputes slice-level metrics to check consistency with the earlier evaluation. Probabilities remain uncalibrated. Implementation execution and patient-level results have not yet been confirmed. This workflow does not retrain models or evaluate the test split.

Once the export scripts are available in the remote project directory, submit and inspect the job using:

```bash
sbatch export_validation.sh
# Replace JOBID with the returned job number.
cat validation_export_JOBID.out
cat validation_export_JOBID.err
```

## Teaching-team clarifications and current evaluation plan

According to teaching-team Q&A supplied by the student, the 0.80 test-accuracy target applies to individual slices. Patient aggregation is optional and cannot replace the required slice-level metric. Patient-independent splitting remains necessary to prevent leakage.

The model architecture must be defined by the student; directly calling a prebuilt ConvNeXt does not meet that requirement. The self-implemented ConvNeXt remains the primary architecture. Public implementations may be referenced and adapted with proper attribution, but experiments and analysis must be the student's own work. Training from scratch is the default; pretrained weights require explicit justification, acknowledgement and supporting comparison.

The teaching team states that a shortfall against the 0.80 target will not itself lose marks when supported by reasonable trial experiments, best effort and sound analysis. This does not remove other assessment requirements. Dataset preprocessing may provisionally be treated as consistent between the original partitions, but the exact 20-slice selection procedure has not been confirmed.

### Slice-level prediction export

`export_slice_predictions.py` exports validation logits, probabilities, predictions and correctness for the self-implemented models. `row_index` is the zero-based data-row index in the validation manifest and can support later failure-case lookup. Outputs contain no patient IDs or image paths, but remain private runtime artifacts under each run's `validation_slice_export/` directory. Existing export directories are not overwritten.

The script recomputes slice-level metrics for consistency checking. It does not train a model, aggregate patients, calibrate probabilities or load test images. Both validation slice exports completed and reproduced the previously reported metrics and manifest hash, according to student-supplied outputs.

```bash
sbatch export_slice_predictions.sh
# Replace JOBID with the returned job number.
cat validation_slices_JOBID.out
cat validation_slices_JOBID.err
```

Do not commit generated predictions or `validation_slices_*.out/.err` logs. Use the exported validation predictions for confidence and referral analysis; final test evaluation remains held out.

## Local Windows execution

Local hardware reported on 25 October 2026: NVIDIA RTX 4050 Laptop GPU, 6,141 MiB total VRAM, driver 596.21. The local environment uses Python 3.14.7, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128 and Matplotlib 3.11.2, as confirmed by the student. The student subsequently completed a batch-size-8 local GPU smoke run: two training and two validation batches in 2.3 seconds, with peak allocated memory 1,407.2 MiB. No checkpoint was saved. These partial-batch results verify execution only; subsequent full-run results are reported below.

`prepare_local_data.py` verifies the original split hashes and every local image path, then writes separate `local_data/splits_local/` manifests without changing patient membership. Original manifests remain unchanged. The repository already ignores `/local_data/`. This path audit does not verify downloaded image contents.

`train_convnext_local.py` preserves the self-implemented ConvNeXt, affine augmentation, AdamW LR 3e-5 and weight decay 0.05. It reads local manifests and uses zero DataLoader workers for Windows. `--smoke` processes only two training and two validation batches, saves no checkpoint, and labels its outputs diagnostic-only. Results go into ignored project `runs/` directories. Full training defaults should be passed explicitly; no stochastic-depth or mixed-precision changes are introduced during the environment migration.

From the project folder in PowerShell, use the repository virtual-environment Python:

```powershell
$py = "D:\Github\PatternAnalysis-2026\.venv\Scripts\python.exe"
& $py prepare_local_data.py
& $py -u train_convnext_local.py --model convnext --batch-size 8 --smoke 2>&1 | Tee-Object -FilePath local_smoke.log
cat local_smoke.log
```

Do not commit `local_smoke.log`. If the local check passes, use measured memory and speed to decide full-run settings. Keeping batch size 8 preserves the earlier optimization setup; lowering it changes the experiment and must be recorded. Local package versions and hardware differ from Rangpur, so exact numerical reproduction is not guaranteed.

## Local stochastic-depth comparison

`modules_sd.py` extends the self-implemented ConvNeXt with per-sample stochastic depth on residual branches. Drop probabilities increase linearly across 18 blocks from 0 to 0.1. Surviving branches are scaled by inverse keep probability during training; evaluation uses all branches. Model initialization is copied from the original implementation, and replacement construction preserves the CPU RNG state. No pretrained or prebuilt model is used.

`train_convnext_local_sd.py` is a separate local entry point. The completed comparison ran both the original local trainer and this variant from fresh initialization for 20 epochs, with the same patient split, affine augmentation, batch size 8, seed 42, AdamW LR 3e-5 and weight decay 0.05. Different hardware/software means Rangpur alone is not a matched local control. Stochastic depth adds training randomness; identical seeds do not imply identical training trajectories.

First check the new variant in PowerShell:

```powershell
$py = "D:\Github\PatternAnalysis-2026\.venv\Scripts\python.exe"
& $py -u train_convnext_local_sd.py --model convnext --batch-size 8 --smoke 2>&1 | Tee-Object -FilePath local_sd_smoke.log
cat local_sd_smoke.log
```

Both 20-epoch runs completed. Minimum validation loss selected epoch 6 in each run:

| Local configuration | Selected epoch | Validation loss | Validation accuracy | NC recall | AD recall |
|---|---:|---:|---:|---:|---:|
| Control: affine augmentation, LR 3e-5 | 6 | 0.561936 | 70.1% | 60.3% | 79.8% |
| Same settings with maximum stochastic depth 0.1 | 6 | 0.563209 | 69.8% | 59.5% | 80.0% |

Exact selected losses from student-supplied logs are 0.5619357655483692 and 0.5632090744444507. The run identifiers are `local_convnext_aug_lr3e5_20261025_123556_489058` and `local_convnext_aug_lr3e5_sd_20261025_162020_159868`. This single-seed comparison does not show a clear benefit from stochastic depth; retain the control as the current local candidate. At epoch 20, train/validation accuracy was 83.7%/63.1% for the control and 83.2%/63.9% for stochastic depth. Validation deterioration persists in both.

Measured epoch times were 516.0-1164.2 seconds for the control and 514.9-614.9 seconds for stochastic depth, excluding checkpoint writing and startup. These timings do not establish a speed advantage because laptop load and operating conditions were not controlled. Peak allocated memory was 1407.2 MiB and 1405.1 MiB, respectively.

Preserve old runs and do not commit local log files. The SD checkpoint records model name `ConvNeXtTinySD`; the current `predict.py` model registry does not yet support it and must be extended before evaluating an SD checkpoint.

## Training-input inspection

`inspect_training_inputs.py` uses the actual training loader and affine transform on CPU. With seed 42, it selects four distinct training patients per class and one slice per patient, displaying each original alongside two augmented versions using a fixed [0,1] grayscale display range. It writes a PNG, image statistics and sampling metadata into ignored `local_data/diagnostics/` timestamped directories. No validation or test images are read, and no dataset files are modified. These small samples cannot establish dataset-wide anatomical orientation or preprocessing consistency.

```powershell
$py = "D:\Github\PatternAnalysis-2026\.venv\Scripts\python.exe"
& $py inspect_training_inputs.py
```

The generated MRI preview and row-index statistics are private inspection artifacts; do not commit them to the public repository.

## Local validation confidence and error analysis

`analyse_validation_confidence.py` reads the saved local control validation predictions on CPU, verifies the manifest hash, row order, labels, probability consistency and reproduced accuracy, and writes separate artifacts under the ignored run directory `validation_confidence_analysis/`. It does not train, access test data or select a new operating threshold. Existing output directories are preserved (a repeated run raises an error).

The completed analysis reproduced 70.0873% accuracy on all 4,580 validation slices at epoch 6, with NC recall 60.2632% and AD recall 79.8261%. There were 906 NC-to-AD errors and 464 AD-to-NC errors. Ten equal-width confidence bins over [0,1] gave ECE 0.0240802; the binary AD-probability Brier score was 0.193046. These are descriptive validation estimates, not evidence of clinical calibration; slices within patients are correlated.

| Minimum confidence retained | Coverage | Retained slices | Retained accuracy | Referred slices |
|---|---:|---:|---:|---:|
| 0.50 | 100.00% | 4580 | 70.09% | 0 |
| 0.70 | 49.30% | 2258 | 80.47% | 2322 |
| 0.80 | 27.07% | 1240 | 88.47% | 3340 |
| 0.90 | 10.68% | 489 | 97.55% | 4091 |

Referral preserves the original predictions and retains slices with confidence >= the stated threshold. Retained accuracy does not satisfy the assignment's full-test accuracy target. In particular, retaining confidence >= 0.70 requires referring over half the validation slices. Twelve incorrect predictions remain at confidence >= 0.90, including ten AD-to-NC errors.

An exploratory AD threshold grid from 0.10 to 0.90 in steps of 0.01 classified every validation slice (AD if probability > threshold, otherwise NC). The highest observed grid accuracy was 70.0873% at 0.50, so this grid provides no accuracy improvement. Continue model-training experiments; do not use test results to tune thresholds.

The script also redraws the same six saved error examples as `high_confidence_errors_v2.png` with separated titles. The original plot and predictions are preserved. The plotting layout in `inspect_validation_errors.py` is corrected for future exports. All generated MRI previews and per-slice outputs remain private under `runs/`.

To run on a fresh export from the project directory (already completed for the current control):

```powershell
$py = "D:\Github\PatternAnalysis-2026\.venv\Scripts\python.exe"
& $py analyse_validation_confidence.py 2>&1 | Tee-Object -FilePath validation_confidence.log
cat validation_confidence.log
```

Do not commit the log or runtime outputs. Codex added and ran this CPU analysis and visually checked both generated figures; the student performed the underlying GPU training and prediction export.

## Local cosine learning-rate comparison

`train_convnext_local_cosine.py` is a separate from-scratch experiment using the original self-implemented ConvNeXtTiny. Against the local fixed-LR control, only the learning-rate schedule changes: for 20 epochs, epoch 1 uses 3e-5 and epoch 20 uses 1e-6, following an epoch-wise cosine curve without warmup. The purpose is to test whether smaller late updates improve validation performance after the earlier epoch-6 optimum; improvement is not assumed.

Patient membership, seed 42, batch size 8, affine augmentation, AdamW weight decay 0.05, full precision and minimum validation-loss checkpoint selection remain unchanged. No stochastic depth, pretrained weights or test data are used. The configuration records schedule endpoints and duration; the CSV records the actual learning rate each epoch. Each invocation creates a new run directory, and smoke runs save no checkpoint. This comparison requires a fresh 20-epoch run, not continuation from the selected epoch-6 checkpoint.

Syntax and CPU checks verified the schedule endpoints, monotonic decrease and bounds. The student subsequently completed the 20-epoch run `local_convnext_aug_cosine_20261025_201434_601293`. Minimum validation loss selected epoch 6: loss 0.5626075655547292, accuracy 69.8%, NC recall 58.7% and AD recall 80.7% (percentages rounded in the supplied log). At epoch 20, training accuracy was 79.1% and validation accuracy 65.6%. This single-seed experiment did not improve on the fixed-LR local control (loss 0.5619357655483692, accuracy 70.0873%). Do not infer that all learning-rate schedules are ineffective.

Run the smoke check first in PowerShell:

```powershell
$py = "D:\Github\PatternAnalysis-2026\.venv\Scripts\python.exe"
& $py -u train_convnext_local_cosine.py --model convnext --epochs 20 --batch-size 8 --smoke 2>&1 | Tee-Object -FilePath local_cosine_smoke.log
cat local_cosine_smoke.log
```

If the smoke check passes, run the full comparison separately:

```powershell
& $py -u train_convnext_local_cosine.py --model convnext --epochs 20 --batch-size 8 2>&1 | Tee-Object -FilePath local_cosine_20ep.log
cat local_cosine_20ep.log
```

Do not commit the logs or generated run artifacts. Compare the selected checkpoint with the local fixed-LR control (validation loss 0.561936, accuracy 70.0873%) using the same selection rule. A negative result remains part of the experimental evidence.

## Custom architecture parity check

`check_convnext_parity.py` checks the custom implementation against the installed torchvision 0.26.0 ConvNeXt implementation on CPU using synthetic inputs only. The reference uses `weights=None`, a single-channel stem, two output classes and stochastic depth disabled to match the custom control. It is a diagnostic oracle only; training continues to use `modules.ConvNeXtTiny`. No pretrained weights or medical data are loaded, and no optimizer update is performed.

Reference source: [torchvision ConvNeXt implementation](https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py). The result records the installed source hash and versions rather than assuming the linked main branch is immutable.

Source inspection found matching stage depths/widths, depthwise convolution, channel expansion, GELU, normalization epsilon, residual computation and downsampling. Both sources initialize convolution/linear weights with `trunc_normal_(std=0.02)` and zero biases. Equal seeds need not produce identical parameters because module construction consumes random numbers in different orders; the numerical check explicitly copies all custom parameters into the diagnostic reference.

The executed check mapped all 182 parameter tensors, totaling 27,818,594 parameters. At synthetic input shape `[1,1,240,256]`, the stem, all four stages, all downsampling outputs and final logits had maximum absolute difference 0.0. The final feature map was `[1,768,7,8]` in both implementations. This includes the same floor behavior for the odd spatial dimension at the last downsampling.

A second check increased residual Layer Scale to 0.1 in memory, to avoid testing only nearly-zero residual contributions at initialization. In training mode, with stochastic depth disabled and synthetic inputs `[2,1,64,64]`, logits, cross-entropy loss, input gradients and all parameter gradients had maximum absolute difference 0.0. This temporary change was not saved to any model checkpoint or training source. Assertions used rtol=1e-4 and atol=2e-6.

These results support computational parity for the tested cases; they do not validate the complete training pipeline, stochastic-depth variant or generalisation. A prebuilt implementation alone is not an evidenced remedy for the current performance gap.

The check has already been run; to reproduce it from the project folder:

```powershell
$py = "D:\Github\PatternAnalysis-2026\.venv\Scripts\python.exe"
& $py check_convnext_parity.py 2>&1 | Tee-Object -FilePath architecture_parity.log
cat architecture_parity.log
```

JSON results are saved in a new ignored `runs/architecture_parity_*` directory. Do not commit runtime logs. Codex authored and ran this synthetic check; the coursework training architecture remains student-defined.

## Reduced-width ConvNeXt experiment

The teaching-team answer supplied by the student explicitly permits a reduced-size ConvNeXt at Hard difficulty when its core design is retained and explained. `modules_compact.py` defines an independent ConvNeXt variant using the existing custom blocks and channel normalization. It retains four stages and depths [3,3,9,3], and halves the widths to [48,96,192,384]. Depthwise 7x7 convolutions, fourfold channel expansion, GELU, Layer Scale, residual connections and downsampling remain in place. This is a reduced-width variant, not standard ConvNeXt-Tiny. It has 7,052,594 parameters, compared with 27,818,594 in the original custom model.

The hypothesis is that lower capacity may improve validation performance given the observed training/validation gap. This is an experiment, not an established explanation or promised improvement. `train_convnext_local_compact.py` trains from fresh random initialization and preserves the fixed-LR control's patient split, original image dimensions, affine augmentation, seed 42, batch size 8, AdamW LR 3e-5, weight decay 0.05 and minimum-validation-loss checkpoint selection. No cosine schedule, stochastic depth or pretrained weights are added. Same seeds do not imply identical random draws or optimization trajectories across different architectures.

Codex ran CPU synthetic checks at input shape [2,1,240,256]: syntax, stage depths/widths, forward/backward execution, finite gradients for every parameter, an optimizer update and evaluation confusion counts passed. These checks do not measure medical-data performance. The student subsequently completed GPU smoke and the 20-epoch run `local_convnext_compact_aug_lr3e5_20261026_004510_717473`. Minimum validation loss selected epoch 6: 0.5665020534500285, accuracy 68.7%, NC recall 68.1%, AD recall 69.3% (rounded log percentages). Epoch 18 reached 69.8% but was not selected under the fixed loss rule. Peak allocated memory was 620.4 MiB; the selected performance did not improve on the full-width local control. Epoch 20 training/validation accuracy was 80.8%/67.9%.

From the project folder in PowerShell:

```powershell
$py = "D:\Github\PatternAnalysis-2026\.venv\Scripts\python.exe"
& $py -u train_convnext_local_compact.py --model convnext --epochs 20 --batch-size 8 --smoke 2>&1 | Tee-Object -FilePath local_compact_smoke.log
cat local_compact_smoke.log
```

After SMOKE PASS, run the full experiment separately:

```powershell
& $py -u train_convnext_local_compact.py --model convnext --epochs 20 --batch-size 8 2>&1 | Tee-Object -FilePath local_compact_20ep.log
cat local_compact_20ep.log
```

Each invocation creates a new run directory; smoke checks save no checkpoint. Full checkpoints retain the filename `best_convnext.pth` but record model `ConvNeXtCompact` in their configuration. Existing inference tools must be extended explicitly before loading these checkpoints; do not load them as ConvNeXtTiny. Compare the selected epoch against the local fixed-LR control (validation loss 0.561936, accuracy 70.0873%). Do not commit logs, model weights or runtime predictions. No test data are read by this trainer.

## Matched initialization comparison: completed 10-epoch runs

Teaching-team correspondence supplied by the student allows official ImageNet weights to be loaded into a self-defined ConvNeXt when the source is cited and results are compared with training from scratch. Direct use of a prebuilt model as the submitted architecture remains disallowed. `modules.py` now defines `ConvNeXtTinyRGB`, extending the existing custom implementation with a three-channel stem; the original single-channel model remains unchanged.

`dataset_rgb224.py` applies grayscale conversion, aspect-preserving bilinear resize with black padding to 224x224, training-only affine augmentation, RGB replication, and normalization with means [0.485,0.456,0.406] and standard deviations [0.229,0.224,0.225]. This is used identically in both arms, with deterministic validation preprocessing. RGB replication introduces no new MRI information. There is no center crop.

`train_convnext_local_initialization.py` requires `--initialization scratch` or `--initialization imagenet`. Both arms use the same fixed patient manifests, seed 42, 10 epochs, batch size 8, AdamW LR 3e-5, weight decay 0.05, full precision, no stochastic depth and minimum validation-loss checkpoint selection. All layers are trainable. The ImageNet arm changes only the backbone initialization; the binary classifier is initialized identically. The previous single-channel experiments are historical context, not a matched initialization control.

Official weight source: [torchvision ConvNeXt-Tiny IMAGENET1K_V1](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html); [weight file](https://download.pytorch.org/models/convnext_tiny-983f1562.pth). `imagenet_weight_mapping.py` uses the weight state dictionary without constructing a prebuilt training model. It checks full key coverage and shapes, maps 180 tensors, discards the two ImageNet head tensors and preserves the custom binary head. Download/hash verification uses the official filename hash prefix; the complete SHA256 is recorded in the run configuration. Cached files are checked as well. Default cache: `D:/codex/cache/torch/checkpoints`, outside the repository. No classmate checkpoint or training code is used.

The student ran `check_imagenet_mapping.py` successfully: all 180 backbone tensors mapped, the binary head was preserved, synthetic reference outputs matched and gradients were finite. Both GPU smoke checks subsequently passed. Official weights loaded with SHA256 `983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d`. These checks do not establish generalisation or probability calibration. Smoke runs process partial batches only and save no checkpoint. Never interpret their accuracy as performance. Model name `ConvNeXtTinyRGB` and preprocessing are recorded; existing single-channel inference scripts must be extended before evaluating these checkpoints.

From the project directory in PowerShell, run these sequentially, checking each exit/output before proceeding:

```powershell
$py = "D:\Github\PatternAnalysis-2026\.venv\Scripts\python.exe"
& $py -u train_convnext_local_initialization.py --model convnext --initialization scratch --epochs 10 --batch-size 8 --smoke 2>&1 | Tee-Object -FilePath rgb224_scratch_smoke.log
cat rgb224_scratch_smoke.log
& $py -u train_convnext_local_initialization.py --model convnext --initialization imagenet --epochs 10 --batch-size 8 --smoke 2>&1 | Tee-Object -FilePath rgb224_imagenet_smoke.log
cat rgb224_imagenet_smoke.log
```

The first ImageNet invocation downloads approximately 109 MiB of official weights if absent. After both smoke checks pass, run each full arm separately (no simultaneous training):

```powershell
& $py -u train_convnext_local_initialization.py --model convnext --initialization scratch --epochs 10 --batch-size 8 2>&1 | Tee-Object -FilePath rgb224_scratch_10ep.log
cat rgb224_scratch_10ep.log
& $py -u train_convnext_local_initialization.py --model convnext --initialization imagenet --epochs 10 --batch-size 8 2>&1 | Tee-Object -FilePath rgb224_imagenet_10ep.log
cat rgb224_imagenet_10ep.log
```

Each invocation creates a separate run. Do not commit logs, weights or private runtime outputs. This training script reads no test data. Report results as a single-seed comparison under this specific recipe; improvement is not guaranteed.

### Results and interpretation

The training budget was reduced to 10 epochs per arm at the student's request before the reported runs. Both runs completed. Results below are from student-supplied training logs; percentages are rounded to one decimal place, while the minimum losses are reported at full supplied precision. Minimum validation loss selected epoch 1 in both arms.

| Initialization | Selected epoch | Validation loss | Validation accuracy | NC recall | AD recall |
|---|---:|---:|---:|---:|---:|
| Random | 1 | 0.5658538799667853 | 70.8% | 59.8% | 81.7% |
| ImageNet | 1 | 0.6851202758443928 | 73.9% | 91.6% | 56.3% |

Run identifiers:
- Random: `local_rgb224_scratch_20261026_164828_261146`
- ImageNet: `local_rgb224_imagenet_20261026_182506_250413`

At the selected checkpoints, ImageNet initialization increased validation accuracy by approximately 3.1 percentage points, but increased validation loss and reduced AD recall by approximately 25.4 percentage points. This is a mixed outcome, not an improvement on every metric. Training accuracy reached 87.9% in the first pretrained epoch versus 60.8% from scratch, supporting faster fitting under this recipe. By epoch 10, train/validation accuracy was 99.5%/76.7% for ImageNet and 81.1%/66.2% for random initialization.

The ImageNet run's highest logged validation accuracy was 79.8% at epoch 4 (validation loss 0.8884). This is an exploratory trajectory observation, not the selected-checkpoint result. The trainer saves only improvements in validation loss, so it did not save the epoch-4 weights. Checkpoint selection was not changed retrospectively. At this experiment stage no test results had been used; the later fixed final evaluation is reported below.

Peak allocated memory was approximately 1204.5 MiB in both arms. Epoch times varied substantially; the observations do not establish a controlled hardware speed comparison. Accuracy and cross-entropy can move in different directions because cross-entropy also depends on confidence. Per-slice probability inspection is needed before attributing the loss increase to particular high-confidence errors. This single-seed result is specific to the fixed split and recipe.

The completed lower-learning-rate pretrained experiment is reported separately below.

### Split-protocol explanation

The provided original training and test folders shared 216 patients. We pooled the available images and generated fixed diagnosis-stratified patient-level splits to avoid that overlap. Patients, scans and paths are disjoint across the new splits, and exact duplicate content checks passed. The new test set has not been used for model selection or tuning. Because the new validation set can include files from the original test folder, this protocol differs from the original folder protocol and results are not directly comparable. A tutor advised deriving validation from training data; the subsequent response requested reporting dataset problems and how they were handled, without explicitly approving or rejecting the pooled re-split. We disclose this distinction and retain the fixed protocol rather than changing partitions in response to performance.

## Five-epoch lower-learning-rate fine-tuning

`train_convnext_local_imagenet_lr3e6.py` starts fresh from the same official ImageNet weights and random binary head. It does not resume the previous trained checkpoint. The learning rate is reduced from 3e-5 to 3e-6; RGB224 preprocessing, custom architecture, fixed split, seed 42, batch size 8, AdamW weight decay 0.05, training augmentation and full-layer fine-tuning remain unchanged. The budget is five epochs to limit runtime. Compare against the first five epochs of the previous ImageNet run for an equal-budget comparison; comparison with its full ten-epoch trajectory has a different training budget. Both the first-five-epoch and ten-epoch minimum-loss selections in that previous run were epoch 1.

The hypothesis is that smaller updates may improve validation loss and AD recall given the rapid training fit and validation deterioration in the earlier run. Improvement is not guaranteed. The selection rule remains minimum validation loss; a higher-accuracy epoch is not substituted retrospectively. Only validation guides this experiment; no test data are read. Syntax and the intended source differences were checked. The student subsequently completed the five-epoch GPU run; results follow.

Run smoke first, then full training separately after SMOKE PASS:

```powershell
$py = "D:\Github\PatternAnalysis-2026\.venv\Scripts\python.exe"
& $py -u train_convnext_local_imagenet_lr3e6.py --model convnext --epochs 5 --batch-size 8 --smoke 2>&1 | Tee-Object -FilePath rgb224_imagenet_lr3e6_smoke.log
cat rgb224_imagenet_lr3e6_smoke.log
```

```powershell
& $py -u train_convnext_local_imagenet_lr3e6.py --model convnext --epochs 5 --batch-size 8 2>&1 | Tee-Object -FilePath rgb224_imagenet_lr3e6_5ep.log
cat rgb224_imagenet_lr3e6_5ep.log
```

Weights are reused from the verified D:/codex cache. Separate run names include `lr3e6`; old checkpoints are preserved. Smoke outputs are diagnostic only. Do not commit logs or runtime outputs.

### Completed lower-LR results

Student-supplied logs select epoch 3 by minimum validation loss: 0.6428447342546877, validation accuracy 76.6%, NC recall 84.6% and AD recall 68.7%. Percentages in the training comparison are rounded log values; full checkpoint validation subsequently reproduced loss and accuracy exactly, as reported below. Run identifier: `local_rgb224_lr3e6_imagenet_20261026_201343_178908`, subsequently supplied by the student and matched to the local configuration.

| ImageNet fine-tuning LR | Comparison budget | Loss-selected epoch | Validation loss | Validation accuracy | NC recall | AD recall |
|---|---|---:|---:|---:|---:|---:|
| 3e-5 | First 5 epochs of prior run | 1 | 0.6851202758443928 | 73.9% | 91.6% | 56.3% |
| 3e-6 | 5 epochs | 3 | 0.6428447342546877 | 76.6% | 84.6% | 68.7% |

Under this single-seed, equal-five-epoch comparison, the lower LR improved selected validation accuracy by about 2.7 percentage points and AD recall by about 12.4 points, while reducing NC recall by about 7.0 points. Validation loss decreased by approximately 0.04228. The prior run's first-five-epoch selection is also its full-ten-epoch selection. Its peak logged accuracy of 79.8% at epoch 4 remains an unselected trajectory observation, not an available loss-selected checkpoint.

The lower-LR run ended with train/validation accuracy of 97.1%/75.0% and validation loss 0.7955; continued fitting did not improve on epoch 3. Logged epoch times sum to 2568.7 seconds (about 42.8 minutes, excluding startup and checkpoint-writing overhead); peak allocated memory was 1204.5 MiB. This is the highest reported accuracy among our loss-selected ConvNeXt checkpoints so far, but it does not establish superiority across all metrics or seeds. The scratch RGB224 control still has lower selected validation loss (0.565854). At this experiment stage test evaluation had not been performed; see the later fixed final evaluation below. The selected epoch-3 checkpoint has subsequently passed full validation reproduction.


## RGB224 checkpoint validation: reproduced

`evaluate_rgb224_validation.py` defaults to the lower-LR run `local_rgb224_lr3e6_imagenet_20261026_201343_178908`. It checks the checkpoint model, preprocessing configuration, train/validation manifest hashes and patient/scan/path separation. It verifies that the saved epoch matches the minimum validation-loss entry in the training log, then performs full validation inference using RGB224 preprocessing without augmentation and the recorded batch size.

Outputs include accuracy, balanced accuracy, macro-F1, AD-positive AUROC, per-class metrics, confusion matrix, loss reproduction, and manifest/checkpoint hashes. Private row-index predictions are exported without patient IDs or image paths to `validation_rgb224_evaluation/` inside the run directory. Existing outputs are never overwritten. Loss mismatch beyond absolute tolerance 1e-6 or accuracy mismatch against the log raises an error after saving diagnostics. No test data are loaded and no weights are downloaded.

```powershell
$py = "D:\Github\PatternAnalysis-2026\.venv\Scripts\python.exe"
& $py -u evaluate_rgb224_validation.py 2>&1 | Tee-Object -FilePath rgb224_lr3e6_validation.log
cat rgb224_lr3e6_validation.log
```

Syntax was checked, and the student completed full GPU evaluation: validation loss difference was 0.0 and accuracy matched the training log. Do not commit runtime predictions or logs. The script reuses the existing classification metric functions in `predict.py` without running its CLI.

The verified checkpoint is epoch 3, SHA256 `6b89086579452034b671dff60844f8416829bce60a6fb9a6c23609a877425cd5`. Evaluation used all 4,580 validation slices from 101 patients, with manifest SHA256 `0a9dff42f564228558cdb4a7cd5706bf15a48a9bebb2fdfcad1f863ff59f80b3`. Loss was 0.6428447342546877, exactly matching the saved training value. These are validation results; the separate final test evaluation is reported below.

| Metric | Verified validation result |
|---|---:|
| Accuracy | 0.7661572052401747 |
| Balanced accuracy | 0.766506483600305 |
| Macro-F1 | 0.7647756586411514 |
| AUROC (AD positive) | 0.8613848207475209 |
| NC precision / recall / F1 | 0.728027 / 0.846491 / 0.782803 |
| AD precision / recall / F1 | 0.818559 / 0.686522 / 0.746749 |

Confusion matrix (rows=true, columns=predicted; order NC, AD): `[[1930,350],[721,1579]]`. There were 1,071 incorrect slice predictions, including 721 AD-to-NC errors and 350 NC-to-AD errors. AD recall therefore remains an important limitation despite higher overall validation accuracy. Probabilities are uncalibrated. Any decision-threshold exploration must use validation data only and report the sensitivity/specificity trade-off; AUROC does not guarantee an operating threshold achieving 80% accuracy. Test data were held out during this validation analysis; the subsequent fixed final evaluation is reported below.


## Verified RGB224 validation threshold exploration

Codex ran `analyse_rgb224_confidence.py` on the saved epoch-3 validation predictions and checked the generated figure. The script verifies the validation manifest hash, row ordering, labels, probability/decision consistency and baseline accuracy. It reads no test data and performs no model inference. Outputs remain private under the run's `validation_rgb224_confidence/` directory, with existing directories preserved.

An exploratory fixed grid of AD thresholds 0.10 through 0.90 in increments of 0.01 classifies every slice as AD when p(AD) > threshold. Ranking uses validation accuracy, then proximity to 0.5, then the lower threshold. The grid maximum was at 0.21:

| AD threshold | Slices evaluated | Accuracy | NC recall | AD recall | NC-to-AD errors | AD-to-NC errors |
|---|---:|---:|---:|---:|---:|---:|
| 0.50 (default) | 4580 | 76.6157% | 84.6491% | 68.6522% | 350 | 721 |
| 0.21 (exploratory grid maximum) | 4580 | 78.1659% | 77.1930% | 79.1304% | 520 | 480 |

The 0.21 threshold corrects a net 71 additional predictions, with fewer AD false negatives but more NC false positives. This is a validation-tuned observation, not an unbiased performance estimate or a locked operating rule. It does not reach 80% on this grid and does not guarantee test improvement. Default-threshold and tuned-threshold results must remain clearly distinguished; any final operating rule must be fixed before test evaluation.

With the original 0.50 decisions, top-label ECE over 10 equal-width bins in [0,1] was 0.1402141 and binary Brier score was 0.1833502. Confidence >=0.80 retained 3725/4580 slices (81.33% coverage), with 81.50% retained accuracy and 855 slices referred. Confidence >=0.90 retained 3250 slices (70.96% coverage), with 84.43% retained accuracy but still 506 errors (346 AD-to-NC and 160 NC-to-AD). Referral accuracy is conditional on selection and cannot replace full-test slice accuracy. These referral calculations do not apply the exploratory 0.21 classification threshold. Probabilities remain uncalibrated, and slice dependence within patients limits naive uncertainty estimates.

The analysis has already completed; no rerun is needed. To reproduce on a fresh export directory:

```powershell
$py = "D:\Github\PatternAnalysis-2026\.venv\Scripts\python.exe"
& $py analyse_rgb224_confidence.py 2>&1 | Tee-Object -FilePath rgb224_confidence.log
cat rgb224_confidence.log
```

## Candidate failure cases and inference profiling

`inspect_rgb224_candidate.py --mode failures` has been run by Codex on the saved validation export. It checks manifest identity and prediction consistency, then selects five distinct patients globally: the two highest-confidence AD-to-NC errors, two highest-confidence NC-to-AD errors, and the remaining incorrect prediction closest to p(AD)=0.5. The private figure displays grayscale images after the same deterministic 224x224 resize/padding as the model, before normalization. It uses the default classification threshold 0.50, not the exploratory threshold 0.21. The selection is purposive, not representative of overall error frequency.

| Validation row | True / predicted | p(AD) | Interpretation |
|---|---|---:|---|
| 1524 | AD / NC | 0.002124 | High-confidence false negative; would remain wrong at 0.21 |
| 3481 | AD / NC | 0.002849 | High-confidence false negative; would remain wrong at 0.21 |
| 2777 | NC / AD | 0.999369 | High-confidence false positive; would remain wrong at 0.21 |
| 4054 | NC / AD | 0.999212 | High-confidence false positive; would remain wrong at 0.21 |
| 1407 | AD / NC | 0.499264 | Borderline false negative; would switch to AD at 0.21 |

These examples show that moving a decision threshold can correct borderline mistakes while leaving extreme-confidence errors unchanged. They do not establish that image anatomy, labels or preprocessing caused the errors. The selected images are readable, but a visual sample cannot establish dataset-wide preprocessing consistency or a clinical diagnosis. Figure and selection metadata are saved under the run's ignored `validation_rgb224_failures/` directory. No raw MRI figure or patient identifiers are added to the public README by this script.

The student completed GPU profiling. To reproduce in a fresh output location, run separately while no training is active (the script preserves existing outputs):

```powershell
$py = "D:\Github\PatternAnalysis-2026\.venv\Scripts\python.exe"
& $py -u inspect_rgb224_candidate.py --mode profile 2>&1 | Tee-Object -FilePath rgb224_profile.log
cat rgb224_profile.log
```

The script verifies the evaluated checkpoint hash and preprocessing, then profiles float32 inference at batch sizes 1 and 8, each with 10 warmup forwards and 50 synchronized timed forwards on GPU-resident inputs. It separately times a bounded pipeline over the first 256 validation images, including file loading, preprocessing, CPU-to-GPU transfer, forward pass, softmax and probability transfer to CPU. Model startup, metric computation and file export are excluded; OS file caching is uncontrolled. Batch-8 time divided by eight is amortized per-slice throughput cost, not single-request latency. Memory is peak PyTorch allocated memory, not total VRAM usage. Results are saved under `validation_rgb224_profile/` without overwriting prior outputs. No test data are read. Syntax, CPU figure generation and the student-run GPU profile completed.

### Measured inference resources

The verified epoch-3 checkpoint (SHA256 `6b89086579452034b671dff60844f8416829bce60a6fb9a6c23609a877425cd5`) was profiled on an NVIDIA GeForce RTX 4050 Laptop GPU using PyTorch 2.11.0+cu128 and float32. It contains 27,821,666 parameters. Each forward-only setting used 10 warmup batches and 50 timed batches with CUDA synchronization.

| Measurement | Batch size | Median batch time | Per-slice time | Peak allocated GPU memory |
|---|---:|---:|---:|---:|
| Forward only, GPU-resident inputs | 1 | 6.898 ms | 6.898 ms | 129.90 MiB |
| Forward only, GPU-resident inputs | 8 | 24.377 ms | 3.047 ms (amortized) | 212.61 MiB |

A separate bounded pipeline processed the first 256 validation images at batch size 8 in 1.2871001 seconds, averaging 5.027735 ms per slice (approximately 199 slices/second in this measurement). This includes image reading, RGB224 preprocessing, transfers, model forward, softmax and CPU probability copy, but excludes model loading, metric computation and file export. OS caching was uncontrolled. This is one local measurement, not a cold-start or deployment latency guarantee. Batch-8 amortized time is not single-image request latency; peak allocated memory is not total device memory.

The measurements support execution feasibility on the available laptop GPU, while validation error rates, AD recall and uncalibrated high-confidence errors remain the principal model limitations. The local baseline profile and descriptive resource comparison are reported below; separate-session timing is not a controlled speed guarantee.

## Fixed final evaluation protocol and completed main-model test

`final_evaluation_protocol.json` records the selected RGB224 ImageNet checkpoint (epoch 3, LR 3e-6), its verified SHA256 and the three fixed local manifest hashes. Primary test predictions use the default 0.50 decision threshold; 0.21 is a predeclared supplementary threshold derived from validation exploration. Both results must be reported regardless of their ordering. Test outcomes must not be used to revise the model, threshold or split.

`evaluate_rgb224_test.py` verifies checkpoint identity, prior validation reproduction, manifest hashes, population counts and pairwise patient/scan/path separation. It evaluates all 4580 held-out test slices from 104 patients with deterministic RGB224 preprocessing, then saves both metric sets, predictions and the protocol snapshot under the ignored run directory `final_test_evaluation/`. Existing evaluation directories are not overwritten. Syntax passed; the student subsequently completed main-model test inference under the fixed protocol, with results below.

```powershell
$py = "D:\Github\PatternAnalysis-2026\.venv\Scripts\python.exe"
& $py -u evaluate_rgb224_test.py 2>&1 | Tee-Object -FilePath rgb224_final_test.log
cat rgb224_final_test.log
```

The baseline remains the validation-loss-selected SimpleCNN epoch 9 from `baseline_20261022_211305_056381`. Its checkpoint is not present in the local runs directory and must be retrieved from the student's remote project before baseline test evaluation and matched local profiling. Baseline inference must retain its original single-channel preprocessing; its earlier validation result is not a substitute for a test result. The main-model evaluation does not complete the final baseline comparison.

### Main-model held-out test results

The student evaluated the predefined epoch-3 checkpoint on all 4,580 held-out test slices from 104 patients. Protocol SHA256: `cc1467bf2427a95ba60fc4a784aee3b1b717e7e8274cf933be6b8295838dad92`; test manifest SHA256: `837bde3a34eaac673ffc4a4331214051c159aed44a2e4e4a03cfbb8941189f11`. Test cross-entropy loss was 0.45708363340947955. No confidence-based sample filtering was used.

| Metric | Primary threshold 0.50 | Supplementary threshold 0.21 |
|---|---:|---:|
| Accuracy | 81.7686% | 79.9782% |
| Balanced accuracy | 81.4166% | 80.5384% |
| Macro-F1 | 0.815163 | 0.799655 |
| AUROC (AD positive) | 0.903406 | 0.903406 |
| NC precision | 0.824663 | 0.868506 |
| NC recall | 0.849206 | 0.749603 |
| NC F1 | 0.836755 | 0.804686 |
| AD precision | 0.808564 | 0.737630 |
| AD recall | 0.779126 | 0.861165 |
| AD F1 | 0.793572 | 0.794625 |

Primary confusion matrix (rows=true, columns=predicted; NC, AD): `[[2140,380],[455,1605]]`. The primary result correctly classified 3,745/4,580 slices and exceeds the numerical 80% slice-accuracy target under our disclosed patient-independent re-split protocol. It does not imply that all coursework requirements are complete or that the original folder split was used.

Supplementary confusion matrix: `[[1889,631],[286,1774]]`. The validation-derived threshold reduces AD false negatives from 455 to 286 but increases NC false positives from 380 to 631. Its accuracy is below 80% before rounding and must not be represented as meeting the target. Both thresholds were specified before test evaluation; the primary threshold was not selected retrospectively. AUROC is identical because the underlying probability ranking is unchanged. No further model, split or threshold tuning will use these test outcomes.

Test accuracy is higher than validation accuracy (76.6157% at the default threshold). Different held-out patient groups can produce different observed performance; this single split does not establish the cause or external generalisation. Slices from a patient are correlated, so independent-slice uncertainty estimates would be inappropriate. Probabilities remain uncalibrated. Baseline test evaluation is complete below; the local baseline resource comparison is reported below.

## Final baseline evaluation: completed

`evaluate_baseline_final.py` expects the preselected remote checkpoint at `runs/baseline_20261022_211305_056381/best_baseline.pth`. It checks model identity, epoch 9, recorded validation loss and split hashes/separation, and uses the original single-channel, original-size [0,1] preprocessing. RGB224 preprocessing must not be substituted.

Before test inference, the script reproduces full validation results. It requires the original confusion matrix `[[1737,543],[658,1642]]` and loss within 1e-4 of 0.5258350444315042 for approximate cross-environment reproduction. The original 1e-5 strict check is retained separately in the output; the diagnostic basis for revising the acceptance tolerance is documented below. A mismatch stops execution before test images are loaded. The tolerance allows small numerical differences between the original remote and local software/hardware; a failure needs investigation, not test-driven adjustment. No historical baseline checkpoint hash was recorded, so the new hash identifies the retrieved artifact without proving archived historical identity.

After this gate passes, it evaluates the same fixed test split at threshold 0.50 and saves metrics and private row-index predictions under `final_baseline_evaluation/`. Existing output is preserved. Syntax passed; checkpoint retrieval and GPU evaluation have now completed, following the approximate validation reproduction check described below. Download the original `config.json` and `training_log.csv` alongside the checkpoint for provenance; the evaluator uses metadata embedded in the checkpoint.

```powershell
$py = "D:\Github\PatternAnalysis-2026\.venv\Scripts\python.exe"
& $py -u evaluate_baseline_final.py 2>&1 | Tee-Object -FilePath baseline_final_evaluation.log
cat baseline_final_evaluation.log
```

Do not commit the checkpoint, logs or private predictions. Baseline test results are reported below; local baseline resource profiling is completed below.

### Baseline cross-environment validation diagnostic

The retrieved epoch-9 baseline SHA256 is `593339ff8dd1dc992c55db6412611531bd347d3b30c04f76bf395d4f99a94e93`. The first local CUDA validation reproduced accuracy 0.7377729257641922 and the original confusion matrix exactly. However, loss was 0.5258488687039984, differing from the remote reference by 1.3824272494189493e-5, slightly exceeding the existing 1e-5 gate. Local AUROC was 0.8157362700228833. Test inference stopped before execution. The cause is not yet established; identical aggregate classifications do not prove identical probabilities or per-slice predictions.

The evaluator now supports `--validation-only --device cpu` (or cuda), reports device and TF32 settings, and returns validation diagnostics without test inference. The initial diagnostic retained the original tolerance. The completed CPU comparison and subsequent validation-only acceptance-rule revision are documented below.

```powershell
& $py -u evaluate_baseline_final.py --validation-only --device cpu 2>&1 | Tee-Object -FilePath baseline_validation_cpu.log
cat baseline_validation_cpu.log
```

### Completed baseline CPU comparison and acceptance-rule revision

CPU validation loss was 0.5258488635764372 versus local CUDA 0.5258488687039984, a difference of approximately 5.13e-9. Both reproduced the historical validation confusion matrix and accuracy exactly; both differed from remote loss by approximately 1.38e-5 (about 0.00263% of the remote loss). CPU AUROC was 0.8157364607170099. This evidence does not support local GPU arithmetic as the main source of the discrepancy; cross-environment preprocessing, library or historical artifact differences have not been conclusively isolated. Matching aggregate confusion matrices does not establish identical per-slice outputs.

Before any baseline test inference, the acceptance rule was revised to absolute validation-loss tolerance 1e-4, still requiring the exact historical confusion matrix. This is an explicitly approximate cross-environment check, not exact reproduction. The report retains the original strict 1e-5 check (failed on the observed local results), actual loss difference and rationale. No model weights, prediction threshold, preprocessing or split were changed. This procedural tolerance revision uses validation evidence only, not baseline test results. The already-completed ConvNeXt test result was not used to choose the numerical tolerance. This revision preceded baseline test inference; that evaluation has now completed and is reported below.

## Final held-out model comparison

Both preselected models were evaluated on the same 4,580 test slices from 104 patients (2,520 NC and 2,060 AD), using the predefined primary threshold 0.50. No test-based model or threshold retuning was performed. The retrieved SimpleCNN checkpoint passed the explicitly approximate cross-environment validation gate; the original stricter check failure and loss difference remain documented above.

| Test metric | SimpleCNN baseline (epoch 9) | Custom ConvNeXt, ImageNet fine-tuned (epoch 3) |
|---|---:|---:|
| Accuracy | 71.1572% | **81.7686%** |
| Balanced accuracy | 70.8434% | **81.4166%** |
| Macro-F1 | 0.708516 | **0.815163** |
| AUROC (AD positive) | 0.771076 | **0.903406** |
| NC precision | 0.737050 | 0.824663 |
| NC recall | 0.739683 | 0.849206 |
| NC F1 | 0.738364 | 0.836755 |
| AD precision | 0.680156 | 0.808564 |
| AD recall | 0.677184 | 0.779126 |
| AD F1 | 0.678667 | 0.793572 |
| Cross-entropy loss | 0.569680 | 0.457084 |
| Incorrect slices | 1321 | 835 |

Baseline confusion matrix (rows=true, columns=predicted; NC, AD): `[[1864,656],[665,1395]]`. The main model correctly classified 486 more slices, increasing accuracy by 10.6114 percentage points. AD false negatives decreased from 665 to 455, and NC false positives from 656 to 380. These are observed differences on one patient-disjoint test split, not a statistical-significance claim.

This is a comparison of complete prediction pipelines, not an isolated architecture ablation: the main model uses ImageNet initialization and RGB224 normalization, while the baseline was trained from scratch on original-size grayscale inputs, with different training settings. The matched initialization experiments earlier in this report address initialization separately on validation data. The final accuracy difference cannot be attributed solely to the ConvNeXt architecture or solely to pretraining. No claim of clinical readiness follows from these results.

The baseline test artifacts are saved privately under `runs/baseline_20261022_211305_056381/final_baseline_evaluation/`. Its retrieved checkpoint SHA256 is `593339ff8dd1dc992c55db6412611531bd347d3b30c04f76bf395d4f99a94e93`. Local baseline timing and memory profiling are completed in the resource comparison below.

### Final test visualizations

![Final test confusion matrices at the predefined primary threshold](figures/final_test_confusion.png)

![Final test ROC curves for the fixed checkpoints](figures/final_test_roc.png)

`plot_final_results.py` renders these aggregate figures from saved predictions and metrics, checking matched test-row identities, confusion-matrix totals and reproduced AUROC. It does not run inference or select models.

## Baseline resource profiling: completed

`profile_baseline.py` verifies the retrieved epoch-9 checkpoint hash and validation manifest against the completed baseline evaluation. It uses the same timing procedure as `inspect_rgb224_candidate.py --mode profile`: float32 on the local GPU, batch sizes 1 and 8, 10 warmup batches, 50 synchronized forward measurements, and a separate 256-image validation pipeline measurement at batch size 8. The baseline retains its original-size single-channel [0,1] preprocessing. No test images are used and no parameters are updated.

The resulting comparison measures the two actual prediction pipelines, including their different input dimensions and preprocessing; it is not an architecture-only speed experiment. Laptop power/load and OS cache remain uncontrolled across sessions. Syntax was checked; the student completed the GPU measurements reported below. Outputs are preserved under `runs/baseline_20261022_211305_056381/validation_baseline_profile/`.

```powershell
$py = "D:\Github\PatternAnalysis-2026\.venv\Scripts\python.exe"
& $py -u profile_baseline.py 2>&1 | Tee-Object -FilePath baseline_profile.log
cat baseline_profile.log
```

Run with no concurrent training or GPU workload. Do not commit runtime logs or private run directories.

## Accuracy and resource trade-off

Both models were profiled on the NVIDIA GeForce RTX 4050 Laptop GPU with PyTorch 2.11.0+cu128, float32, 10 warmup batches and 50 timed forward batches. Each keeps its own training-time preprocessing. The measurements were taken in separate sessions, so power/load and caching differences limit causal speed comparisons.

| Measurement | SimpleCNN | Custom ConvNeXt RGB |
|---|---:|---:|
| Parameters | 23,426 | 27,821,666 |
| Test accuracy, primary threshold | 71.1572% | 81.7686% |
| Batch-1 median forward | 0.2744 ms | 6.8982 ms |
| Batch-8 median forward per batch | 2.1767 ms | 24.3770 ms |
| Batch-8 amortized forward per slice | 0.2721 ms | 3.0471 ms |
| Batch-1 peak allocated GPU memory | 16.95 MiB | 129.90 MiB |
| Batch-8 peak allocated GPU memory | 71.09 MiB | 212.61 MiB |
| 256-image pipeline elapsed time | 0.2063843 s | 1.2871001 s |
| Pipeline average per slice | 0.8062 ms | 5.0277 ms |

The baseline is substantially smaller and faster in these measurements. ConvNeXt improves test accuracy by 10.6114 percentage points at higher computational cost; its measured batch-8 forward cost is approximately 11.2 times the baseline's. Both fit within the available laptop GPU memory. These are observations of complete pipelines, not an architecture-only benchmark or a single-request latency estimate derived from batching. Peak allocated memory is not total GPU memory, and pipeline timings exclude model startup and output writing.

For this coursework's predictive-performance objective, the fixed ConvNeXt candidate is the stronger tested pipeline and exceeds the numerical 80% slice-accuracy target under the disclosed re-split protocol. The SimpleCNN remains a useful low-resource reference. Neither pipeline is established as suitable for autonomous clinical diagnosis: the primary ConvNeXt still misses 455 AD test slices, probabilities are uncalibrated, and no external validation is available. Any proposed human-review workflow must report coverage and residual errors rather than substitute selective accuracy for full-set accuracy.

![Measured resource comparison on the local laptop GPU](figures/final_resource_comparison.png)

## Submission entry points and final prediction example

The model definitions remain in `modules.py`. `dataset.py` now contains both the original grayscale loader and `RGB224Dataset`; `dataset_rgb224.py` preserves the historical import path. The RGB224 implementation was moved without changing its preprocessing or augmentation operations.

`train.py` exposes explicit final-workflow subcommands while retaining the original command-line interface:

| Command | Workflow |
|---|---|
| `train.py final-train --model convnext --epochs 5 --batch-size 8` | Reproduce the final ImageNet-initialized, AdamW 3e-6 training recipe in a new run |
| `train.py rgb224-comparison --model convnext --initialization scratch --epochs 10 --batch-size 8` | Matched scratch recipe; use `imagenet` for the paired initialization experiment |
| `train.py final-validate --run RUN_DIRECTORY` | Full RGB224 validation through the existing verified evaluator |
| `train.py final-test` | Frozen-protocol test evaluator; refuses to overwrite the completed result directory |
| `predict.py example --device cpu` | One validation-slice prediction and input/output visualization from the frozen final checkpoint |

These are reproduction interfaces, not instructions to resume model selection after test evaluation. Existing evaluation outputs are preserved. Historical scripts remain available and final workflows delegate to their documented implementations. Training and held-out testing are separate explicit commands so training never automatically exposes test performance.

Run from `recognition/ConvNeXt_ADNI_s4990492` using the repository virtual environment. The final example defaults to validation row 0 and writes `prediction.json` and `prediction.png` under the private `runs/final_prediction_example/` directory. It verifies the checkpoint and validation-manifest hashes, applies the training-time deterministic RGB224 preprocessing, and uses the predefined threshold 0.50. Its uncalibrated probability and single prediction are not an accuracy estimate. Use `--row` to select a different validation example and `--output` to choose a new output directory; existing output is never overwritten. Keep MRI example images private unless the submission channel permits the supplied dataset images.


## Remaining work

- Consolidate the experimental narrative, environment details, confidence analysis and five validation failure examples into the submission report.
- Complete the final report layout and review the documented canonical entry points and private prediction example.
- Keep the completed test results, model checkpoints and operating rules fixed.
- Export and visually inspect the final README PDF; check references, AI disclosure and submission contents.
- Submit through Turnitin and prepare the required upstream PR using GitHub Desktop; retain meaningful source/documentation commits without data or weights.

## Artificial Intelligence Usage Disclosure

ChatGPT/Codex assisted with project planning, explanation of concepts, drafting and revising audit/split/data-loader/model/training/check scripts, Slurm troubleshooting, interpretation of supplied outputs, learning-curve plotting, and feasibility-review/README drafting. The ConvNeXt architecture was referenced from the paper and author implementation linked above.

The student ran the data and split audits, CPU model checks, GPU training smoke tests, a fixed-batch learning check, both three-epoch train/validation pilots, both original twenty-epoch runs, the twenty-epoch affine-augmentation run, the lower-learning-rate augmentation run, baseline checkpoint inference and CPU deserialization of both selected checkpoint files and full validation inference with exact loss reproduction on Rangpur. These historical checks support the corresponding experiment records. The student subsequently completed fixed-checkpoint test evaluations and local profiling for both final pipelines, as reported above. Clinical use and calibrated probabilities are not established by these experiments. The student also completed the local stochastic-depth, cosine-schedule and reduced-width comparisons, synthetic ImageNet mapping verification, both RGB224 GPU smoke checks, and the matched 10-epoch scratch/ImageNet runs. Codex assisted with their implementation and documentation. Further code understanding, review and empirical validation remain part of the student's responsibility.



