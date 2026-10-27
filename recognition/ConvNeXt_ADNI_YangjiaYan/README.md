# ADNI MRI Classification with ConvNeXt-Tiny

Yangjia Yan | Student ID: 49840232 | COMP3710

This project compares a small CNN trained from scratch with an
ImageNet-pretrained ConvNeXt-Tiny fine-tuned for binary classification
of Alzheimer's disease (AD) and normal controls (NC).

The project adapts ConvNeXt to MRI classification using Torchvision's
implementation and ImageNet weights [1, 2].

## Feasibility review

Chosen project: ADNI classification with ConvNeXt, Hard difficulty.

### Scope and acceptance criteria

The project compares a CNN trained from scratch with a pretrained
ConvNeXt-Tiny for AD versus NC classification. The intended user is an
MRI reviewer assessing predictions and deciding which patients need
further review. The prototype tests whether confidence helps with this
selection.

The five acceptance criteria were:

1. No patient overlap between training, validation and test splits.
2. ConvNeXt test slice accuracy of at least 80%.
3. Evaluation of both models on the same split using Macro-F1, AUROC,
   AD sensitivity and NC specificity.
4. Analysis of three to five errors and a confidence-based review rule.
5. Measurement of GPU memory and inference latency, with training
   feasible on one A100 GPU with 40 GB VRAM.

### Model choice and course concepts

ConvNeXt-Tiny [1, 2] combines convolutional feature learning and residual
connections. ImageNet pretraining allows transfer learning on MRI slices;
cross-entropy and AdamW provide supervised optimisation. The small CNN
tests whether this added model complexity improves the same task.

### Evidence and resource plan

The initial audit found 30,520 slices from 1,526 scans and 680 patients,
with consistent patient labels. A patient-stratified split produced
476 training, 102 validation and 102 test patients without overlap.

Before test evaluation, both models had completed 15 epochs on Rangpur.
The CNN reached validation slice Macro-F1 of 0.8339 in 7.1 minutes;
ConvNeXt reached 0.8600 in 21.1 minutes. These runs showed that the
project was feasible with one A100 GPU and four CPU cores per job.
Each training job requested a two-hour limit; the two final training
runs used about 28.2 minutes in total, excluding queue time.

The main risks were patient leakage, unstable CNN validation results,
ConvNeXt overfitting and confident errors. The plan used a fixed patient
manifest, validation-based checkpoint selection and held-out testing.
The smaller CNN was the fallback if ConvNeXt proved too costly.
The next planned steps were fixed-checkpoint test evaluation, GPU
profiling and analysis of confidence and misclassifications.

### Tutor feedback

I completed the feasibility review on 23 October 2026. The tutor asked
for a model diagram, prerequisites and training settings, which are
included in this report.

## Environment and prerequisites

Training and evaluation were run on the UQ Rangpur cluster using Slurm,
the `comp3710` partition, one NVIDIA A100-PCIE-40GB GPU and four CPU cores
per job.

| Software | Recorded version |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.14.0+cu130 |
| PyTorch CUDA build | 13.0 |
| cuDNN | 9.24.0 |
| Torchvision | 0.29.0 |
| NumPy | 2.4.6 |
| Scikit-learn | 1.9.1 |
| Pillow | 12.3.0 |
| Matplotlib | 3.11.1 |

The full environment record is in `results/software_environment.txt`.

The commands below assume this project's Rangpur directory and an
existing Conda environment named `torch`:

```bash
cd ~/PatternAnalysis-2026/recognition/ConvNeXt_ADNI_YangjiaYan
source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch
```

Prerequisites:

- Access to the course ADNI dataset and its metadata.
- A CUDA-compatible GPU environment for full training and evaluation.
- Slurm access for the supplied training scripts.
- Access to the Torchvision pretrained weights, or a cached copy, for
  the first pretrained ConvNeXt training run.

Dataset files, trained checkpoints and individual prediction CSVs are
stored outside this repository. Paths below describe the author's
Rangpur setup and must be adjusted on another system.

## Dataset and patient-independent split

Dataset root: `/home/groups/comp3710/ADNI`

The loader reads `meta_data_with_label.json` and JPEG slices under
`AD_NC/{train,test}/{AD,NC}`. It combines the provided folders and creates
a new patient-stratified split using seed 42. The original folder
assignment is retained as `original_split` in the manifest.

Labels are NC = 0 and AD = 1.

| Split | Patients | Slices |
|---|---:|---:|
| Train | 476 | 21,640 |
| Validation | 102 | 4,280 |
| Test | 102 | 4,600 |
| Total | 680 | 30,520 |

The data contain 1,526 scans. All slices and scans from a patient remain
in the same split. The split audit found no patient overlap or
conflicting patient labels.

Generate and audit the manifest:

```bash
python dataset.py \
  --root /home/groups/comp3710/ADNI \
  --manifest ~/data/comp3710_a3_adni_split.csv \
  --seed 42
```

Reuse the same manifest for all model comparisons.

The 70/15/15 split keeps separate groups for training, model selection
and final testing. Grouping by patient prevents repeat scans or slices
from the same person appearing in different splits. Stratification
preserves patient class proportions. This is a new internal split of
the combined course data.

## Preprocessing

Images are converted to RGB and resized to 224 × 224 pixels.
Training additionally uses random rotation within ±5 degrees.
All splits use tensor conversion and ImageNet normalisation:

- Mean: [0.485, 0.456, 0.406]
- Standard deviation: [0.229, 0.224, 0.225]

Normalisation follows the Torchvision pretrained input convention [2].

Validation and test transforms contain no random augmentation.

## How the models work

### CNN baseline

The baseline learns spatial filters using four convolutional layers
with 32, 64, 128 and 256 output channels. Each convolution uses a
3 x 3 kernel, followed by batch normalisation and ReLU. Shared filter
weights allow the model to detect patterns at different image positions.

The first three blocks use max pooling to reduce spatial resolution
from 224 to 112, 56 and 28 pixels. The final block uses adaptive average
pooling to produce one value per channel. A two-layer classifier maps
the resulting 256 features to NC and AD logits. Dropout with probability
0.3 provides regularisation during training.

The baseline starts from random weights.

### ConvNeXt-Tiny

ConvNeXt is a convolutional architecture developed by modernising
ResNet-style design using choices also found in hierarchical vision
Transformers [1]. It remains a convolutional network.

A 4 x 4 convolution with stride 4 forms the input stem. Four stages
contain 3, 3, 9 and 3 residual blocks, with channel widths of 96, 192,
384 and 768. Downsampling reduces spatial resolution between stages.

Within each block, a 7 x 7 depthwise convolution processes spatial
information separately in each channel. Layer normalisation is followed
by linear layers that expand the channel dimension fourfold and project
it back, with GELU between them. These layers mix information across
channels. Layer scaling, stochastic depth and residual addition complete
the block [2].

Residual connections preserve a direct path for features and gradients
through the network. Global average pooling and the final classifier
produce two class logits. In this project, the original output layer
is replaced by a linear layer with two outputs.

### Transfer learning and MRI input

All ConvNeXt parameters are fine-tuned from ImageNet weights. Grayscale
slices are copied into three identical channels to match its input.
Both models use the same resizing and ImageNet normalisation.

### Learning objective and decision rule

For logits z_NC and z_AD, softmax gives:

$$
P(AD \mid x) =
\frac{\exp(z_{AD})}{\exp(z_{NC})+\exp(z_{AD})}.
$$

Cross-entropy penalises low probability assigned to the true class:

$$
L = -\frac{1}{N}\sum_{i=1}^{N}\log p_{i,y_i}.
$$

AdamW updates the trainable parameters using gradients of this loss,
with weight decay for regularisation. The final classification rule
predicts AD when P(AD) is at least 0.5. Scan and patient probabilities
are subsequently calculated using the fixed aggregation rule described
in the evaluation section.

## Models and training

![Model architectures](results/figures/model_architectures.png)

| Setting | SimpleCNN | ConvNeXt-Tiny |
|---|---|---|
| Initialisation | Random | ImageNet pretrained |
| Trainable parameters | 422,530 | 27,821,666 |
| Optimiser | AdamW | AdamW |
| Learning rate | 0.0001 | 0.0001 |
| Weight decay | 0.01 | 0.01 |
| Loss | Cross-entropy | Cross-entropy |
| Batch size | 64 | 32 |
| Epochs | 15 | 15 |
| DataLoader workers | 4 | 4 |
| Seed | 42 | 42 |
| Selected epoch | 6 | 6 |
| Training time (seconds) | 426.59 | 1,263.56 |

All parameters were trainable. ConvNeXt uses
`ConvNeXt_Tiny_Weights.DEFAULT`, with its final linear layer replaced
by a two-class output layer.

Both runs completed 15 epochs. Checkpoints were selected using the
highest validation slice-level Macro-F1, rather than the last epoch.
No early stopping was used.

Submit the final training configurations:

```bash
sbatch train_cnn_lr1e4.slurm
sbatch train_convnext.slurm
```

The earlier CNN run at learning rate 0.001 is retained as a development
experiment. The final CNN baseline uses learning rate 0.0001.

## Evaluation

Run the following commands inside an allocated GPU job, after activating
the environment and entering the project directory.

```bash
python -u evaluate.py \
  --checkpoint ~/data/comp3710_a3_runs_lr1e4/cnn/full/best_model.pth \
  --split val \
  --output-dir ~/data/comp3710_a3_evaluation/cnn/val

python -u evaluate.py \
  --checkpoint ~/data/comp3710_a3_runs_pretrained/convnext/full/best_model.pth \
  --split val \
  --output-dir ~/data/comp3710_a3_evaluation/convnext/val
```

Validation evaluation reproduced the saved checkpoint metrics for both
models. After model selection, the checkpoints, threshold and aggregation
rule were fixed before test evaluation:

```bash
python -u evaluate.py \
  --checkpoint ~/data/comp3710_a3_runs_lr1e4/cnn/full/best_model.pth \
  --split test \
  --output-dir ~/data/comp3710_a3_evaluation/cnn/test

python -u evaluate.py \
  --checkpoint ~/data/comp3710_a3_runs_pretrained/convnext/full/best_model.pth \
  --split test \
  --output-dir ~/data/comp3710_a3_evaluation/convnext/test
```

Predictions use P(AD) ≥ 0.5 for AD. Scan probabilities are the mean of
slice probabilities within each scan. Patient probabilities are the
mean of scan probabilities, giving equal weight to each scan within
a patient.

The evaluator saves metrics and predictions at slice, scan and patient
levels. Confusion matrices use true labels as rows and predicted labels
as columns, ordered [NC, AD].

### Test evaluation through train.py

`train.py` also provides a test-only entry that calls `evaluate.py`.
It loads the specified checkpoint and evaluates the fixed test split
without running training. The model type comes from the checkpoint.

Run inside an allocated GPU Slurm job:

```bash
python -u train.py \
  --test-checkpoint ~/data/comp3710_a3_runs_lr1e4/cnn/full/best_model.pth \
  --test-output-dir ~/data/comp3710_a3_entry_check/cnn \
  --batch-size 32 --workers 4

python -u train.py \
  --test-checkpoint ~/data/comp3710_a3_runs_pretrained/convnext/full/best_model.pth \
  --test-output-dir ~/data/comp3710_a3_entry_check/convnext \
  --batch-size 32 --workers 4
```

The test entry reproduced the results obtained with `evaluate.py`.

## Test results

| Level | Model | Accuracy | Macro-F1 | AUROC |
|---|---|---:|---:|---:|
| Slice | CNN | 0.7865 | 0.7865 | 0.8763 |
| Slice | ConvNeXt-Tiny | 0.8254 | 0.8254 | 0.9138 |
| Scan | CNN | 0.8000 | 0.8000 | 0.9080 |
| Scan | ConvNeXt-Tiny | 0.8652 | 0.8652 | 0.9435 |
| Patient | CNN | 0.7843 | 0.7536 | 0.8797 |
| Patient | ConvNeXt-Tiny | 0.8725 | 0.8507 | 0.9275 |

Full validation and test metrics are in `results/metrics_summary.csv`.

### Test precision and recall by class

Precision measures how often a predicted class is correct. Recall measures
how many members of that true class were identified. AD recall is AD
sensitivity; NC recall is NC specificity. All values below use the fixed
classification threshold of 0.5.

| Level | Model | Class | Support | Precision | Recall | F1 |
|---|---|---|---:|---:|---:|---:|
| slice | CNN | NC | 2180 | 0.7461 | 0.8330 | 0.7872 |
| slice | CNN | AD | 2420 | 0.8319 | 0.7446 | 0.7859 |
| scan | CNN | NC | 109 | 0.7603 | 0.8440 | 0.8000 |
| scan | CNN | AD | 121 | 0.8440 | 0.7603 | 0.8000 |
| patient | CNN | NC | 69 | 0.8406 | 0.8406 | 0.8406 |
| patient | CNN | AD | 33 | 0.6667 | 0.6667 | 0.6667 |
| slice | ConvNeXt-Tiny | NC | 2180 | 0.7877 | 0.8647 | 0.8244 |
| slice | ConvNeXt-Tiny | AD | 2420 | 0.8663 | 0.7901 | 0.8265 |
| scan | ConvNeXt-Tiny | NC | 109 | 0.8250 | 0.9083 | 0.8646 |
| scan | ConvNeXt-Tiny | AD | 121 | 0.9091 | 0.8264 | 0.8658 |
| patient | ConvNeXt-Tiny | NC | 69 | 0.8889 | 0.9275 | 0.9078 |
| patient | ConvNeXt-Tiny | AD | 33 | 0.8333 | 0.7576 | 0.7937 |

Validation and test values are saved in
`results/confidence/per_class_metrics.csv`.

## Training behaviour and test interpretation

![Training and validation curves](results/figures/training_curves.png)

CNN training accuracy rose steadily, while validation loss and Macro-F1
fluctuated. The run at learning rate 0.0001 achieved a best validation
slice Macro-F1 of 0.8339, compared with 0.8097 at 0.001.

ConvNeXt reached approximately 99.6% training accuracy by epoch 15,
while validation accuracy remained between approximately 75% and 86%.
This generalisation gap suggests overfitting. Both final checkpoints
came from epoch 6, selected by validation slice Macro-F1.

![Test ROC curves](results/figures/test_roc.png)

ConvNeXt had higher test AUROC at all three levels. Its accuracy was
82.54% per slice, 86.52% per scan and 87.25% per patient. These levels
have different class distributions and sample counts.

![Test confusion matrices](results/figures/test_confusion_matrices.png)

At patient level, the test set contained 69 NC patients and 33 AD
patients. Predicting NC for every patient would achieve 67.65% accuracy,
which illustrates why accuracy alone is insufficient.

The CNN missed 11 of 33 AD patients and incorrectly classified 11 of
69 NC patients as AD. ConvNeXt reduced these counts to eight missed AD
patients and five incorrectly flagged NC patients. Its AD sensitivity
of 75.76% remained lower than its NC specificity of 92.75%.

ConvNeXt improved both classes, but still missed nearly one quarter
of the AD patients.

## GPU inference resource comparison

Both models were profiled on an NVIDIA A100-PCIE-40GB using a single
validation slice with shape (1, 3, 224, 224), float32 precision,
20 warm-up iterations and 100 timed iterations. TF32 and cuDNN
benchmarking were disabled.

| Measurement | SimpleCNN | ConvNeXt-Tiny |
|---|---:|---:|
| Parameters | 422,530 | 27,821,666 |
| Mean latency (ms) | 0.659 | 4.266 |
| Median latency (ms) | 0.666 | 4.259 |
| 95th-percentile latency (ms) | 0.680 | 4.336 |
| Peak allocated GPU memory (MiB) | 22.57 | 128.90 |
| Peak reserved GPU memory (MiB) | 44.00 | 150.00 |

Timing covers GPU forward inference and softmax, excluding image loading,
preprocessing and transfer. Peak allocated memory includes the model,
input and temporary tensors; reserved memory includes the allocator
cache. Both measurements were taken during inference.

ConvNeXt gained 8.82 percentage points in patient accuracy, with 6.47
times the mean inference latency and 5.71 times the allocated memory.

Recorded profiles are in `results/resource_profiles/`.

## Prediction confidence and referral analysis

Patient confidence was calculated as max(P(AD), 1 − P(AD)) from the
aggregated patient probability.

| Test measurement | CNN | ConvNeXt-Tiny |
|---|---:|---:|
| Brier score, lower is better | 0.1347 | 0.1036 |
| Confidence ECE, 10 bins | 0.1194 | 0.0917 |
| Mean confidence of correct predictions | 0.7991 | 0.9072 |
| Mean confidence of incorrect predictions | 0.6384 | 0.7903 |
| Predictions with confidence at least 0.90 | 22 | 64 |
| Errors among those predictions | 0 | 2 |

ConvNeXt had lower Brier score and ECE, although two of its predictions
with confidence of at least 0.90 were wrong. ECE uses ten bins across
102 patients.

### Validation-selected referral rule

Low-confidence patients were flagged for human review. Candidate
confidence thresholds ranged from 0.50 to 1.00 in steps of 0.01.
The rule retained as many validation patients as possible while requiring
at least 20 retained patients and an empirical retained error rate of
at most 5%. The classification threshold remained fixed at P(AD) = 0.5.

| Test outcome | CNN | ConvNeXt-Tiny |
|---|---:|---:|
| Validation-selected confidence threshold | 0.77 | 0.82 |
| Retained patients | 53/102 | 80/102 |
| Coverage | 51.96% | 78.43% |
| Patients referred for review | 49 | 22 |
| Accuracy among retained patients | 96.23% | 92.50% |
| Errors among retained patients | 2 | 6 |
| Retained false negatives | 1 | 4 |
| Retained false positives | 1 | 2 |
| Errors flagged for review | 20/22 | 7/13 |

ConvNeXt retained more patients, but its retained test error rate was
7.5%, above the 5% validation target. The CNN retained fewer patients
and had a 3.77% error rate. Their retained accuracies cover different
subsets and must be read alongside coverage.

This analysis was developed after inspecting the initial test results.
Thresholds were selected on validation patients and applied unchanged
to test predictions. Review flags were counted; reviewer decisions
were not measured.

![Patient confidence and referral analysis](results/confidence/confidence_analysis.png)

Detailed results are in `results/confidence/`.

## Prediction examples and reproduction

### Run checkpoint inference and resource profiling

Activate the documented environment and run these commands from this
project directory inside an allocated GPU Slurm job:

```bash
python -u predict.py \
  --checkpoint ~/data/comp3710_a3_runs_lr1e4/cnn/full/best_model.pth \
  --manifest ~/data/comp3710_a3_adni_split.csv \
  --split val --examples 5 \
  --output-dir ~/data/comp3710_a3_prediction_examples/cnn

python -u predict.py \
  --checkpoint ~/data/comp3710_a3_runs_pretrained/convnext/full/best_model.pth \
  --manifest ~/data/comp3710_a3_adni_split.csv \
  --split val --examples 5 \
  --output-dir ~/data/comp3710_a3_prediction_examples/convnext
```

The script writes `resource_profile.json`, `prediction_examples.json`
and `prediction_examples.png`. Use `--split test` for test examples;
the profiling input remains a validation slice.

The input tensor has shape (1, 3, 224, 224); the output contains NC
and AD logits. The first three recorded validation examples were:

| Example | True label | CNN P(AD) | CNN prediction | ConvNeXt P(AD) | ConvNeXt prediction |
|---|---|---:|---|---:|---|
| 1 | AD | 0.6990 | AD | 0.9628 | AD |
| 2 | AD | 0.6133 | AD | 0.9986 | AD |
| 3 | AD | 0.8438 | AD | 0.9974 | AD |

Examples follow manifest order and come from different patients.
All five recorded examples were AD slices.

### Reproduce the confidence analysis

First generate the validation and test CSVs using `evaluate.py`.
Then run this CPU-only analysis from the project directory:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python analyse_confidence.py
```

The script reads slice, scan and patient prediction CSVs from the
CNN and ConvNeXt validation/test directories under
`~/data/comp3710_a3_evaluation/`. It writes metrics, referral rules and
plots to `results/confidence/`. Change `BASE` and `OUT` in the script
for other paths; it has no command-line arguments.

Training configurations and epoch metrics are saved in `history.json`.
Validation evaluation reproduced the selected checkpoints' saved scores
within numerical tolerance. Retraining with a different GPU or software
version may produce different weights and scores. To reproduce the
workflow without the author's checkpoints, generate the manifest, run
the two training scripts, then evaluate their saved checkpoints.

## Figures

The current `train.py` updates `training_curves.png` in the run
directory after every epoch, showing training and validation loss,
accuracy and Macro-F1. This plotting feature was added after the
reported experiments and checked using their saved histories.
The report figures were generated separately with `plot_results.py`.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python plot_results.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python plot_architecture.py
```

`plot_results.py` reads the saved training histories and prediction CSVs.
It writes figures to `~/data/comp3710_a3_figures/`.
`plot_architecture.py` writes to `results/figures/`.

## Failure-case analysis

![Five selected test errors](results/figures/test_failure_cases.png)

Five patients were selected from ConvNeXt's test errors: the three
most confident false negatives and two most confident false positives.
The probabilities below are patient-level P(AD), rounded to three
decimal places. Predictions use a threshold of 0.5.

| Case | True label | ConvNeXt P(AD) | ConvNeXt prediction | CNN P(AD) | CNN prediction |
|---|---|---:|---|---:|---|
| 1 | AD | 0.047 | NC | 0.471 | NC |
| 2 | AD | 0.096 | NC | 0.601 | AD |
| 3 | AD | 0.126 | NC | 0.463 | NC |
| 4 | NC | 0.895 | AD | 0.875 | AD |
| 5 | NC | 0.830 | AD | 0.456 | NC |

The figure shows one illustrative slice per patient; the probabilities
combine all of that patient's scans. These cases were selected for
high-confidence errors.

- **Case 1:** both models missed AD. The CNN was close to the decision
  threshold, while ConvNeXt strongly favoured NC.
- **Case 2:** the CNN identified AD, but ConvNeXt predicted NC with
  90.4% confidence.
- **Case 3:** both missed AD. ConvNeXt's 87.4% confidence exceeded the
  82% review threshold, so this error would be retained.
- **Case 4:** both incorrectly predicted AD with high probabilities.
  Agreement between the models would not flag this patient.
- **Case 5:** the CNN identified NC, but ConvNeXt predicted AD. Its
  83.0% confidence also passed the retention threshold.

### Slice and scan audit

The saved predictions were grouped by scan and patient to examine where
these errors arose. Recomputing the scan means and then the patient
means reproduced the saved patient probabilities.

| Case | Slices / scans | CNN wrong slices | ConvNeXt wrong slices | CNN wrong scans | ConvNeXt wrong scans |
|---|---:|---:|---:|---:|---:|
| 1 | 60 / 3 | 53.3% | 96.7% | 2/3 | 3/3 |
| 2 | 20 / 1 | 25.0% | 95.0% | 0/1 | 1/1 |
| 3 | 20 / 1 | 55.0% | 90.0% | 1/1 | 1/1 |
| 4 | 20 / 1 | 100.0% | 95.0% | 1/1 | 1/1 |
| 5 | 40 / 2 | 45.0% | 85.0% | 1/2 | 2/2 |

In Case 1, ConvNeXt assigned scan-level P(AD) values from 0.0017 to
0.1080 across three scans. All three missed AD. In Case 5, both scans
were incorrectly classified as AD, with probabilities of 0.8168 and
0.8431. These errors were already present across scans before patient
averaging. Cases 2, 3 and 4 each had one scan, so averaging multiple
scans cannot explain their patient-level errors either.

The CNN showed a different pattern. It correctly classified most slices
in Case 2 and identified the patient as AD. In Case 5, one scan was
correct and one was wrong; their mean of 0.4560 gave the correct NC
prediction. Case 4 was a shared failure: every CNN slice and 95% of
ConvNeXt slices were wrong.

Across the five ConvNeXt cases, 85.0%–96.7% of slices and every scan
were wrong. The main observed failure was a prediction that remained
wrong across many slices, rather than an isolated misleading slice.
The audit identifies this pattern but does not establish its anatomical
or acquisition-related cause.

Run `python analyse_failure_cases.py` to reproduce the audit from the
saved predictions. The case summaries and scan results are in
`results/failure_audit/case_summary.csv` and
`results/failure_audit/scan_details.csv`.

## Practical recommendation

I recommend ConvNeXt for further evaluation with human review. It
correctly classified 89 of 102 test patients versus 80 for the CNN.
The CNN remains useful where lower memory use and latency matter.

The confidence rule flagged seven of ConvNeXt's 13 errors, but left
six among retained patients, including four missed AD cases. Further
assessment should use an independent cohort and record reviewer
accuracy and time.

## Limitations

The experiments used one patient split and one seed per final model,
with 102 test patients, including 33 AD patients. Architecture,
pretraining and batch size all changed between models, so their effects
cannot be separated. ConvNeXt showed a training–validation gap, and CNN
validation scores fluctuated. No external validation was performed.

## Artificial Intelligence Usage Disclosure

I used OpenAI ChatGPT for technical explanations, implementation assistance, troubleshooting and report preparation. Its contribution included code drafts and suggested revisions.

AI assistance included:

- Discussing dataset indexing and patient-independent splitting, and helping draft the implementation. I ran the split audit and checked patient separation and label consistency.
- Helping draft and revise the CNN baseline, ConvNeXt integration, and training and evaluation code. I tested the code on Rangpur and checked the logs and metrics.
- Explaining Slurm commands and error messages. I applied the job settings, submitted jobs and checked their outputs.
- Helping write plotting, inference profiling and confidence-analysis code, and discussing the results. I examined the model comparisons and prediction errors to assess the findings.
- Helping organise the README and revise explanations of the methods, results and limitations.

I ran the experiments on the UQ Rangpur cluster and checked the outputs, including patient separation, reproduced validation results and prediction errors. I used these checks to assess the AI-assisted code and explanations. All reported numerical results came from the executed experiments and saved outputs.

## References

[1] Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T.,
and Xie, S. (2022). *A ConvNet for the 2020s*. CVPR, 11976–11986.
https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html

[2] PyTorch contributors. *Torchvision ConvNeXt implementation*.
The project uses Torchvision's implementation and pretrained weights.
https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py
