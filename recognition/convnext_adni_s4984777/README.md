# Alzheimer's Disease Classification from MRI Slices Using ConvNeXt-Tiny

**COMP3710 Pattern Recognition · University of Queensland**  
**Student ID:** s4984777 · **Difficulty:** Hard · **Framework:** PyTorch  
**Dataset:** ADNI brain MRI (course-provided, Rangpur) · **Task:** AD (1) vs Cognitively Normal / NC (0)

> **Research prototype, not a medical device.** The model predicts labels for individual 2D MRI slices. Its outputs are not patient diagnoses, and softmax confidence should not be interpreted as clinical certainty.

## 1. Project overview and engineering question

This project studies binary classification of preprocessed structural brain MRI slices into **Alzheimer's Disease (AD)** and **Cognitively Normal (NC)** categories. The intended context is a research or clinical-imaging workflow in which a machine-learning assistant might prioritize cases for specialist review, not independently diagnose patients. A four-block convolutional neural network (CNN) provides a simple baseline; the proposed hard-difficulty system replaces it with **ImageNet-pretrained ConvNeXt-Tiny**, adapted to a two-class output head. Models are compared on an identical, patient-independent test split.

**Engineering dilemma:** Does ConvNeXt improve not only classification accuracy but also the *reliability of predicted confidence* relative to a simple CNN? When errors can be made with high confidence, can a confidence-based reject/referral rule provide a defensible balance between automated coverage, missed AD slices, and clinical review effort?

**Main findings.** ConvNeXt-Tiny attained **81.42% slice-level test accuracy** on 4,440 held-out images versus **69.59%** for BaselineCNN, meeting the assignment's 80% target. However, ConvNeXt had **worse ten-bin expected calibration error** (ECE **0.1199** vs **0.0248**) and made high-confidence false negative and false positive predictions. A 90%-confidence acceptance rule retained **80.11%** of test slices at **87.07% accepted-case accuracy**, but still accepted **279 AD→NC errors**. These findings favor a research-stage, human-in-the-loop prototype rather than autonomous clinical use.

![Validation accuracy of the training experiments](figures/experiment_comparison.png)

*Figure 1. Validation accuracy across the experiments. Experimental checkpoints were selected by minimum validation loss, not by test performance.*

## 2. One-page feasibility review (midpoint submission)

*The following section preserves the substance and preliminary status of the one-page feasibility review presented for the COMP3710 project. Its initial statements about training being pending refer to the midpoint, not the completed project.*

### 2.1 User need, scope, and acceptance criteria

The prototype classifies structural brain MRI images into AD or NC. Its intended users are clinical researchers and medical imaging specialists who may benefit from automated screening assistance; it is intended to **support, not replace**, clinical decisions. The central challenge is whether ConvNeXt-Tiny provides more reliable classification and confidence estimates than a conventional CNN, particularly for ambiguous or misclassified images.

The proposed acceptance criteria were: (1) achieve at least **80% held-out test accuracy**; (2) ensure **zero patient overlap** across train, validation, and test sets; (3) compare accuracy, per-class precision/recall, macro-F1, AUROC, and confidence reliability with a baseline; (4) analyze **3–5 representative errors** and a confidence-based referral rule; and (5) measure model parameters, peak GPU memory, and inference latency.

### 2.2 Model choice and course concepts

The project uses the course-provided ADNI brain MRI dataset on Rangpur. A simple CNN serves as the baseline; **ConvNeXt-Tiny** is the proposed Hard-difficulty model. ConvNeXt modernizes convolutional image classification with depthwise convolutions, residual connections, and hierarchical feature extraction. The implementation uses PyTorch and ImageNet-pretrained weights as an initialization option. Both models are evaluated with the same patient-independent splits.

### 2.3 Preliminary feasibility evidence

At the midpoint, the dataset audit identified **30,520 JPEG slices from 680 distinct patients**. An initial inspection of the supplied folder division revealed **216 patients overlapping** between the original train/test directories, creating potential leakage if used as-is. A stratified **patient-level** split with random seed **42** was implemented, yielding 476/102/102 patients and 21,620/4,460/4,440 slices for train/validation/test, respectively. No patient was assigned to more than one split. The PyTorch loader produced batches of shape **(8, 3, 224, 224)**. At the review stage, final training and benchmarking had not yet been completed.

### 2.4 Risks, compute budget, and fallback

Risks identified at feasibility review included overfitting to patient-specific anatomy, unreliable confidence estimates, the gap between 2D slice accuracy and patient-level diagnosis, and ConvNeXt's higher computational cost. The plan was to test CUDA and data loading using a short A100 allocation on Rangpur, then train the baseline and ConvNeXt through Slurm jobs. The baseline CNN would remain a fallback if ConvNeXt proved computationally impractical. The preliminary conclusion supported continuing, subject to measured accuracy, calibration, and efficiency.

**Post-feasibility outcome:** Training and testing were completed; the 80% slice-level accuracy target was reached, while confidence calibration and AD false negatives remained substantial concerns.

## 3. Dataset, Preprocessing, and Leakage Controls

The project uses the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset provided by the COMP3710 course, containing preprocessed structural brain MRI slices stored as JPEG files under `/home/groups/comp3710/ADNI/AD_NC` on the University of Queensland Rangpur HPC cluster.

The classification task distinguishes between two categories:

- **Cognitively Normal (NC):** Label `0`
- **Alzheimer's Disease (AD):** Label `1`

### 3.1 Dataset Composition

The dataset contains **30,520 MRI slices from 680 unique patients**. Each patient contributes multiple slices, making patient-level separation essential to avoid data leakage.

| Split | Patients | MRI Slices | NC Slices | AD Slices |
|---|---:|---:|---:|---:|
| Training | 476 | 21,620 | 11,000 | 10,620 |
| Validation | 102 | 4,460 | 2,220 | 2,240 |
| Testing | 102 | 4,440 | 2,440 | 2,000 |
| **Total** | **680** | **30,520** | **15,660** | **14,860** |

The dataset was divided into approximately 70% training, 15% validation, and 15% testing by patient count.

### 3.2 Patient-Level Splitting and Leakage Prevention

A significant risk in medical imaging classification is **patient-level data leakage**, where MRI slices from the same patient appear in both training and evaluation datasets.

Because slices from the same patient share anatomical and acquisition characteristics, randomly splitting individual images could produce misleadingly optimistic evaluation results.

To mitigate this issue, `dataset.py` implements patient-stratified splitting using a fixed random seed of `42`.

The splitting procedure ensures that:

1. All MRI slices belonging to the same patient remain within one partition.
2. Training, validation, and testing sets contain mutually exclusive patient IDs.
3. Class representation is maintained through stratification.
4. Experiments use consistent dataset partitions for fair comparison.

The final dataset audit confirmed **zero patient-ID overlap** between training, validation, and testing.

The generated `splits.csv` manifest determines which partition each image belongs to. The original dataset directory structure does not determine the final split.

For example, an MRI image physically located inside the original `train/AD/` directory may belong to the final held-out test partition if its patient was assigned to that partition during patient-independent splitting.

### 3.3 Image Preprocessing and Data Augmentation

Images are converted to three-channel RGB representations and resized to **224 × 224 pixels** to satisfy the input requirements of ConvNeXt-Tiny.

The training and evaluation pipelines use different transformations.

**Training pipeline:**

1. Resize the image to `224 × 224`.
2. Apply random rotation within ±10 degrees.
3. Convert the image into a PyTorch tensor.
4. Normalize the tensor using ImageNet normalization parameters.

**Validation and testing pipeline:**

1. Resize the image to `224 × 224`.
2. Convert the image into a PyTorch tensor.
3. Apply the same ImageNet normalization.

The normalization parameters are:

| Channel | Mean | Standard Deviation |
|---|---:|---:|
| Red | 0.485 | 0.229 |
| Green | 0.456 | 0.224 |
| Blue | 0.406 | 0.225 |

The final input tensor has shape:

`(batch_size, 3, 224, 224)`

Random rotation is applied only during training to introduce image variability and reduce sensitivity to small orientation changes.

Validation and testing transformations remain deterministic, ensuring that reported evaluation results are not affected by random augmentation.

### 3.4 Data Loading

The `get_dataloaders()` function in `dataset.py` creates three PyTorch `DataLoader` objects:

- `train`: Uses training augmentation with shuffled samples.
- `val`: Uses deterministic preprocessing without shuffling.
- `test`: Uses deterministic preprocessing without shuffling.

The default DataLoader configuration is:

| Parameter | Value |
|---|---|
| Batch size | 32 |
| Number of workers | 4 |
| Image size | 224 × 224 |
| Training shuffle | Enabled |
| Validation/test shuffle | Disabled |
| Pin memory | Enabled when CUDA is available |
| Persistent workers | Enabled when worker count is greater than zero |

These values are the loader's defaults; individual training and evaluation commands may override them.

### 3.5 Dataset Limitations

Although the dataset contains 30,520 MRI slices, these observations are not statistically independent because multiple slices originate from the same patient.

Consequently, the reported performance metrics are **slice-level metrics**, not patient-level diagnostic metrics.

An accuracy of 81.42% on individual MRI slices cannot be interpreted as 81.42% patient-level diagnostic accuracy.

Furthermore, the model processes individual 2D slices rather than complete 3D MRI volumes, potentially losing contextual anatomical information between adjacent slices.

The prototype is therefore intended for experimental pattern recognition research rather than independent clinical diagnosis.



## 4. Models and working principles

### 4.1 BaselineCNN

The implemented baseline uses four repeated feature-extraction blocks: **3×3 convolution → batch normalization → ReLU → 2×2 max pooling**, with channels **3 → 32 → 64 → 128 → 256**. An adaptive global average pool reduces the representation to a 256-dimensional vector; dropout (0.3) and a fully connected layer produce **two logits** (NC and AD). Its relatively small size (**389,890 parameters**) provides a low-cost reference model.

### 4.2 ConvNeXt-Tiny

The advanced model uses `torchvision.models.convnext_tiny` with pretrained ImageNet weights and replaces the final 768-dimensional classifier with a **768 → 2** linear layer. ConvNeXt-Tiny extracts multiscale visual features through convolutional stages incorporating depthwise spatial mixing and residual connections. Its **27,821,666 parameters** provide substantially greater representational capacity, at higher storage and compute cost. Both models produce logits; softmax converts these to an NC/AD score distribution during inference. Training minimizes **cross-entropy loss** with AdamW optimization.

Transfer learning is an important difference between the models: the ConvNeXt result reflects the effects of **both** architecture and ImageNet pretraining, rather than a controlled architecture-only comparison.

### 4.3 Training variants

The investigation compared a baseline CNN against ConvNeXt with (a) full fine-tuning at two learning rates, (b) training only the final classifier head (frozen backbone), and (c) partial fine-tuning of the later ConvNeXt stage plus classifier. The training code and job configurations document the exact modes. The first attempt at partial fine-tuning unintentionally left the whole network trainable; the corrected run should be distinguished from that invalid configuration in any discussion of ablation results.

## 5. Implementation and Reproducibility

### 5.1 Directory Layout

The implementation is organized into modular Python scripts, separating model architecture, dataset handling, training, inference, evaluation, and experimental analysis.

```text
recognition/convnext_adni_s4984777/
├── README.md
├── requirements.txt
├── modules.py
├── dataset.py
├── train.py
├── predict.py
├── evaluate.py
├── plot_train.py
├── conf.py
├── error.py
├── baseline.sbatch
├── train.sbatch
├── exp2.sbatch
├── exp3.sbatch
├── exp4.sbatch
├── eval.sbatch
├── predict.sbatch
└── figures/
    ├── baseline_loss.png
    ├── convnext_loss.png
    ├── convnext_accuracy.png
    ├── experiment_comparison.png
    ├── baseline_confusion_matrix.png
    ├── convnext_confusion_matrix.png
    ├── convnext_roc.png
    ├── calibration.png
    ├── coverage_accuracy.png
    └── confidence_distribution.png
```

The main scripts have the following responsibilities:

| File | Purpose |
|---|---|
| `modules.py` | Defines BaselineCNN and ConvNeXt-Tiny model architectures. |
| `dataset.py` | Implements patient-level splitting, preprocessing, and DataLoaders. |
| `train.py` | Handles model training, validation, and checkpoint selection. |
| `predict.py` | Performs individual MRI slice inference and visualizes predictions. |
| `evaluate.py` | Computes held-out test metrics and evaluation figures. |
| `plot_train.py` | Generates training curves and experiment comparisons. |
| `conf.py` | Evaluates confidence calibration and referral thresholds. |
| `error.py` | Selects and analyzes representative false-positive and false-negative predictions. |

The `checkpoints/`, `results/`, and Slurm log files are generated during execution and are not intended for submission.

The repository excludes trained model weights and original ADNI images. Patient-identifying manifests should also not be made publicly accessible without appropriate authorization.

### 5.2 Environment and Dependencies

The project was implemented using **Python and PyTorch** on the University of Queensland Rangpur HPC cluster.

Model training and evaluation were performed on an **NVIDIA A100 GPU**.

The project was developed and evaluated on the University of Queensland Rangpur HPC cluster using the following environment:

| Component | Configuration |
|---|---|
| Python | 3.14.7 |
| Deep learning framework | PyTorch 2.14.1 |
| torchvision | 0.29.1 |
| CUDA build used by PyTorch | 13.0 |
| GPU | NVIDIA A100 |
| Environment | Rangpur HPC, Conda `base` |

The installed environment passed `python -m pip check`, reporting no broken package requirements.

The CUDA version refers to the CUDA version against which PyTorch was built, not necessarily the installed NVIDIA driver version. The reported package versions document the working environment; fresh-environment reproducibility has not yet been independently tested.

The following Python package versions were reported by the working environment:

| Package | Version |
|---|---|
| PyTorch | 2.14.1 |
| torchvision | 0.29.1 |
| NumPy | 2.5.4 |
| pandas | 3.0.6 |
| scikit-learn | 1.9.1 |
| Pillow | 12.3.0 |
| matplotlib | 3.11.2 |

The exact Python package dependencies are recorded in `requirements.txt`.

To install the required packages:

```bash
python -m pip install -r requirements.txt
```

The implementation additionally requires access to the COMP3710 ADNI dataset located on Rangpur:

```text
/home/groups/comp3710/ADNI/AD_NC
```

Dataset files and trained model checkpoints are not distributed with the repository.

Reproducing the trained-model evaluation requires either training a checkpoint with the provided scripts or obtaining an appropriately authorized checkpoint separately.

The software versions above document the original environment; installation and reproducibility on a fresh environment have not yet been independently verified.

### 5.3 Reproduction Commands

All commands below are executed from the project directory:

```bash
cd ~/convnext/PatternAnalysis-2026/recognition/convnext_adni_s4984777
```

#### Step 1 — Generate Patient-Level Dataset Splits

The patient-independent train, validation, and test partitions are generated by running:

```bash
python dataset.py
```

This command invokes the `generate_splits()` function and produces the `splits.csv` manifest required by the DataLoaders.

The generated partitions are configured to use a fixed random seed of `42`, enabling consistent patient-level dataset assignments when the same source data and splitting procedure are used.

**Note:** The command should only be rerun intentionally because it may overwrite the existing split manifest.

#### Step 2 — Check Python Scripts

Before training, basic syntax verification can be performed using:

```bash
python -m py_compile \
  modules.py \
  dataset.py \
  train.py \
  predict.py \
  evaluate.py \
  plot_train.py \
  conf.py \
  error.py
```

This checks Python syntax but does not establish that all dependencies, datasets, and GPU resources are available.

#### Step 3 — Train Baseline CNN

To train the lightweight baseline model using the provided Slurm configuration:

```bash
sbatch baseline.sbatch
```

Alternatively, within an appropriate GPU allocation:

```bash
python train.py \
  --model baseline \
  --epochs 5 \
  --batch-size 8 \
  --require-cuda
```

#### Step 4 — Train ConvNeXt-Tiny

To reproduce the primary ConvNeXt-Tiny experiment:

```bash
sbatch train.sbatch
```

Alternatively, within an appropriate GPU allocation:

```bash
python train.py \
  --model convnext \
  --epochs 5 \
  --batch-size 8 \
  --lr 1e-4 \
  --require-cuda
```

The primary configuration uses a learning rate of `1e-4` and a maximum of five training epochs.

Model checkpoints are selected according to the **lowest validation loss**.

The held-out test data is not used for model optimization or checkpoint selection.

#### Step 5 — Reproduce Additional Experiments

Three additional ConvNeXt configurations were evaluated:

| Experiment | Configuration | Slurm Command |
|---|---|---|
| Experiment 2 | Lower learning rate (`1e-5`) | `sbatch exp2.sbatch` |
| Experiment 3 | Frozen ConvNeXt backbone | `sbatch exp3.sbatch` |
| Experiment 4 | Partial fine-tuning | `sbatch exp4.sbatch` |

Each experiment uses the same patient-independent dataset partitions to support fair comparison.

#### Step 6 — Evaluate the Trained Model

To evaluate the selected ConvNeXt checkpoint on the held-out test partition:

```bash
sbatch eval.sbatch
```

Alternatively, within an appropriate GPU allocation:

```bash
python evaluate.py \
  --model convnext \
  --checkpoint checkpoints/convnext/best.pth \
  --output-dir results/convnext \
  --batch-size 8 \
  --require-cuda
```

The evaluation script generates classification metrics, confusion matrices, ROC curves, and per-image prediction records.

The primary reported ConvNeXt model achieved **81.42% test accuracy** on 4,440 held-out MRI slices.

This exceeded the assignment's 80% test accuracy threshold for the hard-difficulty ConvNeXt classification task.

#### Step 7 — Run Individual MRI Predictions

The `predict.py` script demonstrates how to load the trained model and predict the class of a single MRI image.

For example:

```bash
python predict.py \
  --model convnext \
  --checkpoint checkpoints/convnext/best.pth \
  --test-index 0 \
  --output results/prediction_convnext.png
```

A recorded example produced:

| Field | Result |
|---|---|
| Ground-truth class | AD |
| Predicted class | NC |
| NC probability | 78.53% |
| AD probability | 21.47% |
| Classification outcome | False Negative |

The prediction script prints the predicted probabilities and creates a visual representation of the selected input and corresponding classification result.

The test index refers to the ordering of the generated test partition, not necessarily the order of images in the original dataset directories.

#### Step 8 — Generate Training and Evaluation Analysis

Training histories can be processed using:

```bash
python plot_train.py
```

This generates training and validation accuracy/loss curves, along with a comparison of model configurations.

Confidence calibration and referral-threshold analysis can be reproduced using:

```bash
python conf.py
```

Representative misclassification analysis can be generated using:

```bash
python error.py
```

These scripts use previously generated training histories, evaluation predictions, and dataset metadata.

They do not retrain the classification models.

#### Step 9 — Output Artifacts

Generated results are stored in the local `results/` directory, including:

```text
results/
├── baseline/
├── convnext/
├── training_curves/
├── confidence_analysis/
├── error_analysis/
└── prediction_convnext.png
```

Selected reviewed figures are copied into the `figures/` directory for inclusion in this README.

Model checkpoints and generated datasets are excluded from the public repository in accordance with the project submission requirements.

## 6. Training Experiments and Model Selection

Five training configurations were evaluated using the same patient-independent training and validation partitions: one baseline CNN and four ConvNeXt-Tiny configurations. The experiments investigated whether learning rate selection and the extent of backbone fine-tuning affected model performance and generalization.

The primary ConvNeXt experiment used full fine-tuning with a learning rate of `1e-4`. A second experiment reduced the learning rate to `1e-5`, while subsequent experiments investigated frozen-backbone training and partial fine-tuning.

### 6.1 Experiment Comparison

| Model | Epochs Completed | Selected Checkpoint Epoch | Best Validation Loss | Validation Accuracy at Checkpoint | Highest Validation Accuracy |
|---|---:|---:|---:|---:|---:|
| Baseline CNN | 5 | 2 | 0.5189 | 73.52% | 74.96% |
| ConvNeXt (LR 1e-4) | 5 | 1 | **0.4679** | **83.05%** | **83.05%** |
| ConvNeXt (LR 1e-5) | 10 | 2 | 0.5115 | 80.22% | 81.43% |
| ConvNeXt (Frozen Backbone) | 7 | 4 | 0.5244 | 74.06% | 74.06% |
| ConvNeXt (Partial Fine-tuning) | 5 | 2 | 0.5332 | 74.98% | 77.26% |

All checkpoints were selected according to **minimum validation loss**, rather than maximum validation accuracy.

The distinction is important because several experiments achieved their highest validation accuracy at an epoch different from the selected checkpoint.

For example, partial fine-tuning reached 77.26% validation accuracy during training, but the checkpoint selected by minimum validation loss achieved only 74.98%.

### 6.2 Training and Validation Curves

![Baseline training and validation loss](figures/baseline_loss.png)

*Figure 2. Training and validation loss of the baseline CNN across five epochs.*

![ConvNeXt full fine-tuning loss](figures/convnext_loss.png)

*Figure 3. Training and validation loss of the primary ConvNeXt-Tiny configuration (learning rate 1e-4). The selected checkpoint corresponds to epoch 1.*

![ConvNeXt training and validation accuracy](figures/convnext_accuracy.png)

*Figure 4. Training and validation accuracy of ConvNeXt-Tiny during full fine-tuning.*

![Validation accuracy across experiments](figures/experiment_comparison.png)

*Figure 5. Validation accuracy across the baseline CNN and four ConvNeXt-Tiny configurations.*

### 6.3 Interpretation

**Full fine-tuning (learning rate 1e-4).** The primary ConvNeXt experiment achieved the highest validation accuracy at its selected checkpoint (83.05%) and the lowest validation loss (0.4679) among the five configurations. This suggests that adapting the full pretrained backbone was more effective than restricting the trainable layers under the tested settings.

**Lower learning rate (1e-5).** Reducing the learning rate produced 80.22% validation accuracy at the selected checkpoint. Although this configuration achieved reasonable performance, it did not outperform the original full fine-tuning experiment.

**Frozen backbone.** Training only the classification head achieved 74.06% validation accuracy. This result suggests that pretrained representations alone were insufficient for optimal discrimination in this particular MRI classification setting.

**Partial fine-tuning.** Unfreezing only part of the network achieved 74.98% validation accuracy at the selected checkpoint. Its maximum observed validation accuracy was 77.26%, indicating that the selected checkpoint and peak accuracy occurred at different stages of training.

**Generalization and overfitting.** In the primary ConvNeXt experiment, final training accuracy reached approximately 98.53%, while the selected checkpoint achieved 83.05% validation accuracy. The training curves also show that validation loss deteriorated after the first epoch, consistent with overfitting. Selecting the checkpoint with the lowest validation loss reduced reliance on later epochs with poorer validation behavior.

### 6.4 Model Selection

The primary ConvNeXt-Tiny model, trained with full fine-tuning and a learning rate of `1e-4`, was selected for held-out test evaluation.

Its checkpoint was saved at:

`checkpoints/convnext/best.pth`

This checkpoint corresponds to **epoch 1**, selected by minimum validation loss rather than the final training epoch.

The final test set was not used to select the checkpoint. The comparison is therefore based on validation performance before held-out test evaluation.

Because training settings were not identical across all experiments, the results should be interpreted as comparisons of the tested configurations rather than isolated causal effects of individual hyperparameters.

## 7. Held-out test benchmark

Both selected checkpoints were evaluated once on the same **4,440 test MRI slices** (2,440 NC; 2,000 AD).

| Metric | BaselineCNN | ConvNeXt-Tiny |
|---|---:|---:|
| Test accuracy | 69.59% | **81.42%** |
| Macro-F1 | 0.6860 | **0.8115** |
| ROC-AUC | 0.7675 | **0.8935** |
| NC precision | 69.52% | **81.88%** |
| NC recall | 79.55% | **85.00%** |
| NC F1 | 0.7420 | **0.8341** |
| AD precision | 69.72% | **80.81%** |
| AD recall | 57.45% | **77.05%** |
| AD F1 | 0.6299 | **0.7888** |
| Correctly classified slices | 3,090 | **3,615** |
| Incorrectly classified slices | 1,350 | **825** |

**Accuracy improvement:** **11.82 percentage points**, or 525 additional correctly classified slices. The improvement is not uniform across diagnostic risks: AD recall increased from 57.45% to 77.05%, but ConvNeXt still missed **459 of 2,000 AD-labeled slices**.

### 7.1 Confusion matrices

| Model | True NC → NC | True NC → AD (FP) | True AD → NC (FN) | True AD → AD |
|---|---:|---:|---:|---:|
| BaselineCNN | 1,941 | 499 | 851 | 1,149 |
| ConvNeXt-Tiny | 2,074 | 366 | 459 | 1,541 |

![Baseline confusion matrix](figures/baseline_confusion_matrix.png)

*Figure 6. BaselineCNN confusion matrix.*

![ConvNeXt confusion matrix](figures/convnext_confusion_matrix.png)

*Figure 7. ConvNeXt-Tiny confusion matrix.*

### 7.2 ROC-AUC

![ConvNeXt ROC curve](figures/convnext_roc.png)

*Figure 8. ConvNeXt ROC curve. AUROC measures ranking/discrimination across thresholds, not the calibration of predicted probabilities.*

### 7.3 Resource trade-offs

| Resource metric | BaselineCNN | ConvNeXt-Tiny |
|---|---:|---:|
| Parameters | 389,890 | 27,821,666 |
| Inference forward-pass latency | 0.425 ms/slice | 1.181 ms/slice |
| Inference throughput | 2,355 slices/s | 846 slices/s |
| Peak allocated GPU memory during evaluation | 0.112 GiB | 0.207 GiB |

These latency measures were recorded on the A100 using **batch size 8** and synchronized forward passes. They exclude image decoding, loading, and most preprocessing, so they are **not end-to-end clinical workflow latency**. Inference GPU allocation is not directly comparable with peak memory during training. ConvNeXt uses approximately 71× the parameters and 2.8× the measured forward-pass time per image, but also improves classification substantially.

## 8. Confidence calibration and selective referral

A classifier can discriminate well while remaining poorly calibrated: a prediction assigned 95% confidence need not be correct 95% of the time. We therefore examine **top-class confidence calibration**, probability scoring, and the trade-off between accepted predictions and referral for specialist review.

| Metric | BaselineCNN | ConvNeXt-Tiny |
|---|---:|---:|
| 10-bin ECE (lower is better) | **0.0248** | 0.1199 |
| Brier score for AD probability (lower is better) | 0.1966 | **0.1494** |
| Log loss (lower is better) | **0.5787** | 0.5812 |
| Mean confidence on correct predictions | 73.50% | 95.25% |
| Mean confidence on incorrect predictions | 66.33% | 85.35% |

ConvNeXt has markedly **higher accuracy and lower Brier score**, yet **higher ECE** under the adopted ten-bin computation. These are not inconsistent: Brier score combines discrimination and calibration effects, whereas ECE approximates reliability using discretized confidence bins. ECE changes with binning and sample composition; it should not be treated as a definitive proof of calibration quality. ConvNeXt's average confidence on **incorrect** predictions (85.35%) indicates an important operational risk.

![Confidence calibration comparison](figures/calibration.png)

*Figure 9. Predicted confidence compared with empirical accuracy. The deviation from the diagonal illustrates confidence miscalibration.*

### 8.1 Referral threshold illustration

A simple selective-classification rule accepts the model's predicted label only when the **maximum softmax probability** is at least threshold *t*; otherwise it refers the slice for human review. The figures below evaluate this rule retrospectively on the existing held-out test predictions.

| ConvNeXt threshold | Automated coverage | Referral rate | Accuracy among accepted | Accepted errors | Accepted AD→NC errors |
|---|---:|---:|---:|---:|---:|
| 50% | 100.00% | 0.00% | 81.42% | 825 | 459 |
| 70% | 92.09% | 7.91% | 84.13% | 649 | 373 |
| 80% | 87.34% | 12.66% | 85.48% | 563 | 327 |
| 90% | 80.11% | 19.89% | 87.07% | 460 | 279 |
| 95% | 72.05% | 27.95% | 89.15% | 347 | 215 |
| 99% | 52.45% | 47.55% | 93.09% | 161 | 115 |

![Coverage versus selective accuracy](figures/coverage_accuracy.png)

*Figure 10. Referral/coverage trade-off. Accuracy improves among accepted slices as acceptance becomes more selective, at the cost of more manual referrals.*

The **90% threshold is an illustrative operating point**, *not* a clinically safe recommendation: even at this threshold, **279 AD false negatives** remain among automated decisions. Because threshold values were explored using test predictions, any operational threshold must be selected using **validation data** and then assessed on an untouched independent cohort. In a medical screening context, a single global softmax cutoff may be insufficient; class-sensitive referral rules, calibration, and clinician oversight merit further investigation.

## 9. Representative Failure-Case Autopsy

The selected ConvNeXt-Tiny checkpoint made **825 incorrect predictions** across the 4,440 held-out MRI slices:

- **459 false negatives (AD → NC):** Alzheimer's disease slices incorrectly classified as cognitively normal.
- **366 false positives (NC → AD):** Cognitively normal slices incorrectly classified as Alzheimer's disease.

Five representative misclassifications were selected to investigate both error directions and differences in prediction confidence.

### 9.1 Selected Failure Cases

| Case | Ground Truth → Prediction | Confidence | Test Index | Failure Type |
|---|---|---:|---:|---|
| 1 | AD → NC | 99.94% | 3273 | High-confidence false negative |
| 2 | AD → NC | 50.10% | 3233 | Near-boundary false negative |
| 3 | NC → AD | 99.93% | 2595 | High-confidence false positive |
| 4 | NC → AD | 50.28% | 3870 | Near-boundary false positive |
| 5 | AD → NC | 99.93% | 960 | High-confidence false negative |

The five cases were confirmed to originate from **five distinct patients** using the `patient_id` field in `results/error_analysis/representative_errors.csv`.

Selecting distinct patients avoids presenting several slices from one patient as separate patient-level examples.

### Prediction-to-Image Alignment Verification

Before selecting representative failures, `error.py` validates the correspondence between prediction records and the held-out test dataset.

The script checks that the test DataLoader uses a sequential sampler, verifies prediction index consistency, and performs tensor-based comparisons using `torch.allclose()` to detect image-order mismatches. Detected inconsistencies cause the analysis to terminate with a `RuntimeError`.

The error-analysis script completed successfully, providing evidence that its implemented alignment checks passed. This supports the validity of the reported test indices and selected failure cases.

The five selected examples also originate from five distinct patients, as verified from the test manifest.

### 9.2 Qualitative Inspection of Individual Failure Cases

Visual inspection of the five selected MRI slices reveals variations in anatomical cross-section, the appearance of internal fluid-filled spaces, and the visibility of surrounding brain structures. These observations are descriptive and do not establish the clinical causes of the model's errors.

| Case | Error | Visual Observation | Interpretation |
|---|---|---|---|
| 1 | AD → NC (99.94%) | The slice shows a relatively compact internal dark space compared with several other selected examples, with a different visible cross-section of the surrounding brain tissue. | The model confidently predicts NC despite the AD label. The visible anatomy alone does not explain this error. |
| 2 | AD → NC (50.10%) | A prominent elongated dark central space is visible, and the slice shows a different anatomical cross-section from Case 1. | The prediction is close to the decision boundary, indicating model uncertainty despite the AD ground truth. |
| 3 | NC → AD (99.93%) | The slice contains a prominent dark internal space and clearly visible cortical folds. Its general appearance shares features with some AD-labelled examples in the selected set. | The model confidently predicts AD for an NC-labelled slice, illustrating that visible structural similarities do not necessarily correspond to the correct class. |
| 4 | NC → AD (50.28%) | The image also shows a prominent dark internal space, with differences in the surrounding tissue contours and apparent slice geometry compared with Case 3. | The near-50% confidence suggests weak separation between the two model output scores. |
| 5 | AD → NC (99.93%) | The slice shows visible cortical folding and an elongated internal dark region, with a different tissue outline and cross-section from Case 1. | The model makes another highly confident false-negative prediction, suggesting that confidence alone is insufficient for reliable screening. |

The examples illustrate substantial visual variation across both diagnostic categories. However, the available 2D slices do not permit a reliable determination of whether differences arise from pathology, slice position, acquisition parameters, preprocessing, or natural anatomical variability.

In particular, the five images should not be used to diagnose ventricular enlargement, cortical atrophy, or other clinical abnormalities without expert interpretation and access to the full MRI volumes.

The strongest conclusion supported by these cases is that **ConvNeXt-Tiny makes both uncertain and extremely confident classification errors across visually different MRI slices**.

The cases were deliberately selected to illustrate these failure modes, not to estimate their frequency in the wider population.

### 9.3 Near-Boundary Misclassifications

Cases 2 and 4 were incorrectly classified with confidence values of 50.10% and 50.28%, respectively.

Because the model produces probabilities for two classes, confidence values close to 50% indicate that neither class received substantially greater probability than the other.

These examples demonstrate the type of uncertainty that confidence-based referral mechanisms are intended to capture.

A referral rule could flag such predictions for human review instead of automatically accepting the classification.

However, the presence of high-confidence errors shows that confidence thresholding alone cannot eliminate all incorrect predictions.

### 9.4 Potential Sources of Failure

Several factors may contribute to the observed classification errors.

**Limited anatomical context.** The classifier processes individual 2D MRI slices rather than complete 3D MRI volumes. A single slice may not provide sufficient anatomical context to distinguish AD from NC reliably.

**Inter-patient variability.** Differences in anatomy, acquisition characteristics, and image appearance may influence classification.

**Ambiguous visual evidence.** Some slices may contain weak or insufficiently discriminative image features, particularly when the relevant anatomical changes are subtle.

**Confidence calibration limitations.** A model can produce sharply concentrated softmax probabilities even when its classification is incorrect.

These are plausible explanations rather than established causes of the five individual errors. The available images and predictions do not independently identify the anatomical or acquisition factors responsible for each misclassification.

### 9.5 Interpretation and Limitations

The five examples were deliberately selected to cover high-confidence and near-boundary errors. They are illustrative cases, not a random sample of model failures.

Consequently, their confidence values should not be used to estimate how frequently overconfident errors occur across the full dataset.

The quantitative confidence analysis in Section 8 provides a more appropriate basis for evaluating the overall relationship between confidence and classification correctness.

Furthermore, all reported errors are measured at the **MRI slice level**. They cannot be directly interpreted as patient-level diagnostic errors.

The results demonstrate that although ConvNeXt-Tiny improves classification performance over the baseline, it remains unsuitable for autonomous clinical decision-making without further validation, calibration, and human oversight.

### 9.6 Representative Failure Visualization

The analysis script `error.py` generated a five-case visualization at:

`results/error_analysis/representative_errors.png`

This figure contains medical images from the course-provided ADNI dataset. It should not be published to a public GitHub repository unless the applicable dataset permissions explicitly authorize redistribution.

If permitted, a reviewed and appropriately de-identified copy can be added to `figures/` and referenced in this report.

The table above preserves the relevant error classifications and confidence measurements without publishing the underlying MRI images.

## 10. Engineering recommendation and limitations

**Recommendation to the project manager:** Advance ConvNeXt-Tiny only as an **experimental, human-supervised research prototype**. Relative to the simple CNN, it substantially improves slice-level accuracy, AD recall, and AUROC for a manageable increase in A100 forward-pass compute. It therefore justifies *further investigation* as an image-level assistance tool. It does **not** justify autonomous triage or diagnosis at this stage.

The principal reasons are (1) **459 AD false negatives** across the held-out slices, (2) severe high-confidence mistakes, (3) the poor ConvNeXt ECE relative to baseline despite stronger discrimination, and (4) no independently established patient-level clinical validity. Even high-confidence acceptance policies leave nontrivial AD false negatives.

Before practical consideration, the next engineering steps should include a **patient-level aggregation study**, calibration (e.g., temperature scaling) fitted on validation data, threshold selection under explicit false-negative constraints, an untouched external dataset or site-independent test cohort, and appropriate specialist evaluation. Further development should verify subgroup sensitivity, robustness to acquisition shifts, interpretability, and whether class-specific rather than maximum-confidence referral improves safety. No clinical performance claims should be made from the present 2D-slice results alone.

## 11. Artificial Intelligence Usage Disclosure

**Tool used:** OpenAI ChatGPT. **Assistance:** discussion of PyTorch model/training and Slurm script structure; debugging inference, CSV label parsing, and checkpoint handling; drafting evaluation and visualization scripts; outlining possible calibration, referral, and error analyses; and assistance organizing this technical documentation.

**Verification and responsibility:** I reviewed code, ran experiments in the Rangpur environment, tested data loading and checkpoint inference, checked patient-disjoint splits, inspected actual training logs and saved CSV metrics, and compared reported figures with outputs from the scripts. 


## 12. References

1. Liu, Z., et al. (2022). **A ConvNet for the 2020s (ConvNeXt).** *CVPR 2022*. https://arxiv.org/abs/2201.03545
2. Alzheimer's Disease Neuroimaging Initiative (ADNI). **Study and dataset information.** https://adni.loni.usc.edu/
3. PyTorch / torchvision documentation. **ConvNeXt-Tiny pretrained model and transforms.** https://pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html
4. Guo, C., et al. (2017). **On Calibration of Modern Neural Networks.** *ICML 2017*. https://arxiv.org/abs/1706.04599
5. COMP3710/COMP7710 teaching team (2026). **Working Prototype Project in the Era of Artificial Intelligence, Final Assessment Specification v2.5.** University of Queensland.


