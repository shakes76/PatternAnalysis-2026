# HipMRI Anatomical Segmentation Using 2D and 3D U-Net Architectures

## 1. Introduction

### 1.1 Background and Motivation

The purpose of medical segmentation using computer vision is to easily identify areas of need to detect lesions, their size, location and relationship with nearby areas (Gao et al., 2025).

MRI is used because it excels at differentiating subtle variations in dense soft tissues—such as distinguishing gray matter, white matter, and cerebrospinal fluid in the brain, or mapping muscles, ligaments, and cartilage in joints. Moreover, it provides accurate 3D cross-sectional volumes across any spatial plane (Palazzo et al., 2024).

Even though MRI is capable of providing multiplanar anatomical information, there can still be areas that are hard to differentiate, with some planes having lower resolution than others.

**2D limitations.** The limitation with a 2D implementation is that each slice is processed independently. Additionally, it can see the cross-sectional area but not its extent into neighbouring slices. Moreover, it cannot directly determine whether a small region is the beginning of an organ, its end, or a completely different one with a similar appearance. It might also produce inconsistent boundaries between adjacent slices.

**3D opportunity.** The benefit of using a 3D U-Net is that it can process a block of neighbouring slices simultaneously, as well as learn how the organ's shape changes along the depth axis. Moreover, it learns continuity, curvature and spatial relationships across slices. It may produce more consistent anatomical segmentation.

### 1.2 Research Question

Does incorporating 3D spatial context improve multi-class MRI segmentation when compared to a standard 2D U-Net segmentation baseline, and does an improved 3D U-Net using residual connections provide enough improvement for the increased computational cost?

### 1.3 Hypothesis

**IF** a 3D U-Net uses spatial information from neighbouring MRI slices, **THEN** it may produce more accurate segmentation than a standard 2D U-Net, **BECAUSE** organs are three-dimensional structures whose cross-sectional shapes and boundaries change across depth. Learning these relationships may help identify anatomical boundaries that are ambiguous in individual 2D slices.

### 1.4 Evaluation Approach

Performance is evaluated using mean foreground Dice and per-class Dice scores on the validation dataset. Computational costs, including training time and model complexity, are also considered to determine whether any improvements in segmentation accuracy justify the additional computational resources required.

## 2. Dataset and Preprocessing

### 2.1 Dataset Overview

The HipMRI dataset contains 211 MRI volumes from 38 patients. Each volume has a corresponding segmentation, and the data is stored in NIfTI (`.nii.gz`) format. Volume dimensions are 256×256×128 or 256×256×144. There are six segmentation labels: background, bladder, body, bone, rectum and prostate. Generally, the bladder and body occupy relatively large regions, whereas the rectum and prostate are smaller regions. Furthermore, each structure has a different geometry, making segmentation difficulty vary between classes. Finally, these differences allow us to investigate whether 3D spatial context particularly benefits smaller or more complex anatomical structures.

### 2.2 Dataset Splitting

| Split | Patients | MRI Volumes |
|---|---:|---:|
| Training | 26 | 136 |
| Validation | 6 | 34 |
| Testing | 6 | 41 |
| **Total** | **38** | **211** |

**Key points:**

- Patient-level splitting prevents scans belonging to the same patient from appearing across different subsets, reducing the risk of data leakage.
- The split was generated once, before any training, by randomly permuting the 38 patient IDs with seed 3710 and assigning 15% of patients (6) to testing, 15% (6) to validation and the remaining 26 to training. It was then frozen in the manifest described in Section 9.2, and `dataset.load_manifest` checks its SHA-256 on every run so that all five models use exactly the same patients.
- The **training set** is used to optimise the model's parameters.
- The **validation set** is used to evaluate performance during training and select the best-performing model.
- The **test set** was reserved for independent evaluation, but a complete held-out evaluation was not obtained; all results reported here are validation results.

### 2.3 Preprocessing

#### 2D U-Net

- MRI volumes divided into individual 2D axial slices.
- Images paired with corresponding segmentation masks.
- Individual slices fed into the model.
- Predictions reassembled into volumes for validation.

#### 3D U-Net

- MRI volumes processed as volumetric data.
- Extract 96 × 96 × 96 voxel patches.
- Use class-balanced sampling with foreground probability 0.33.
- Patch sampling helps expose the network to less frequently represented structures.
- Use overlapping sliding-window inference to evaluate full volumes.

#### Intensity Normalisation

MRI intensity values were normalised separately for each volume. The values were clipped to the 0.5th and 99.5th percentiles of non-zero voxels, then standardised using the mean and standard deviation of the clipped non-zero region. Non-finite input values were replaced with zero before clipping, and a minimum standard deviation safeguard prevented division by zero.

## 3. Feasibility Review

### 3.1 Feasibility Review as Presented at the Consultation

The one-page review below is reproduced verbatim from the version prepared for the compulsory practical consultation (21 October 2026). Only blank lines and bold paragraph labels have been added for Markdown rendering. It describes the original 2D plan; the revised scope is explained in Section 3.2.

> **COMP3710 | Feasibility review**\
> HipMRI 2D prostate segmentation - Improved U-Net vs standard U-Net\
> Updated 21 October 2026. Status: real-data feasibility evidence; tutor check-off pending.
>
> **User need and scope.** An imaging specialist needs a prototype that drafts prostate contours for review. Compare normal-difficulty 2D Improved U-Net with standard 2D U-Net on the same held-out HipMRI patients. Investigate whether residual multi-scale features reduce difficult boundary errors, despite the absence of 3D continuity. This prototype is not approved for clinical use.
>
> **Testable acceptance criteria (proposed).** (1) Zero patient overlap and no reused files across partitions; reject misaligned pairs and unknown labels. (2) Held-out patient-macro prostate Dice >= 0.75, with pooled and per-class Dice/IoU also reported. (3) Review 3-5 real boundary failures, including apex/base and adjacent-organ leakage, using input/truth/prediction/FP-FN panels. (4) On the allocated GPU, target peak allocated memory <= 8 GiB and warmed forward latency <= 1 second per 256x256 slice at batch 1. These are proposed resource limits, not measured Rangpur results. (5) Compare both models using the identical split, preprocessing, seed and training budget.
>
> **Model choice and course concepts.** Use 2D Nifti slices, a verified fixed multi-class vocabulary, per-slice normalization and patient-level partitions. Convolutions, pooling/striding, skip connections, residual learning, dropout and gradient supervision connect course concepts to the architecture.
>
> **Preliminary evidence.** The downloaded data contains 12,660 aligned slice pairs from 38 patients. The supplied train/validation/test partitions contain 31/4/3 patients and 11,460/660/540 slices, with no patient overlap. Every pair passed shape, affine and finite-value checks. Labels 0-5 match the published order: background, bladder, body, bone, rectum and prostate. Prostate occupies 1,766,549 of 415,088,640 pixels (0.426%), confirming severe imbalance. Most slices are 256x128; 60 training slices are 256x144. Visual checks of positive examples from every split showed the label-5 contour aligned with the prostate. Both models also completed a one-epoch CPU pipeline check on 114 real slices, including checkpoint reload, native-grid evaluation and prediction export.
>
> **Risks, budget and fallback.** Main risks are small-cohort variance, resize-related boundary loss, extreme class imbalance and 2D continuity failures. The next experiment is a short representative optimization run on the full training partition, followed by one 50-epoch run per model at width 16, 256x256, batch 8; measure a real epoch before reserving GPU hours. Use the same reduced batch size for both if memory is insufficient. Retain the baseline if the improved model lacks meaningful boundary benefit; discuss any difficulty change with the tutor. No superiority or deployment recommendation is justified yet. Present this review at the compulsory practical consultation.

### 3.2 Changes After the Consultation

Following the feasibility review consultation, the tutor suggested investigating the 3D aspect of MRI segmentation, which I had not initially considered. My original plan was to compare a standard 2D U-Net with an improved 2D U-Net using residual connections. Based on this feedback, I revised the project to use the 2D U-Net as a baseline, followed by a standard 3D U-Net and an improved 3D U-Net with residual connections. This revised approach allows me to investigate whether incorporating spatial information across neighbouring MRI slices improves segmentation performance, while also evaluating the effect of residual connections.

| | Feasibility review (original plan) | Final project |
|---|---|---|
| Comparison | Standard 2D U-Net vs improved 2D U-Net | 2D U-Net baseline vs standard 3D U-Net vs residual 3D U-Net |
| Data | Supplied 2D slices, 31/4/3 patient partitions | 3D NIfTI volumes (2D model uses their axial slices), frozen 26/6/6 patient-level split |
| Improvement tested | Residual multi-scale features in 2D | 3D spatial context, then residual blocks in 3D |

## 4. Methodology

Three U-Net-based architectures were used to investigate the effect of spatial context and architectural improvements on multi-class MRI segmentation: a standard 2D U-Net, a standard 3D U-Net, and an improved 3D U-Net incorporating residual connections. The 2D model serves as the baseline, while the 3D models investigate whether learning spatial relationships between neighbouring MRI slices improves segmentation performance.

### 4.1 Standard 2D U-Net

The initial baseline uses a standard 2D U-Net architecture consisting of an encoder, bottleneck, decoder, and skip connections.

The encoder progressively extracts higher-level features through convolutional blocks while reducing the spatial resolution of the input image. Each convolutional block contains two 3×3 convolutions, each followed by batch normalisation and ReLU activation. Max pooling is used for downsampling.

The bottleneck processes the encoded features at the lowest spatial resolution, allowing the network to learn more complex representations of the input.

The decoder progressively restores the original spatial resolution using transposed convolutions. Skip connections concatenate features from the corresponding encoder stages with decoder features, helping recover spatial details lost during downsampling. This is particularly important for segmentation, where accurately identifying anatomical boundaries requires both high-level context and local information.

Two configurations were implemented:

| Model | Encoder Channels | Bottleneck Channels | Trainable Parameters |
|---|---|---:|---:|
| 2D U-Net 256 | 32, 64, 128, 256 | 256 | 5,140,166 |
| 2D U-Net 512 | 32, 64, 128, 256 | 512 | 7,762,630 |

Both configurations process individual axial MRI slices as single-channel images and produce six output channels corresponding to the segmentation classes. A final 1×1 convolution maps the decoder features to the six output logits.

Although this approach can learn spatial relationships within each slice, it cannot directly use anatomical information from neighbouring slices. Consequently, structures that change in shape or size across the MRI volume may be more difficult to identify using individual cross-sections alone.

### 4.2 Standard 3D U-Net

To investigate whether volumetric information improves segmentation, the 2D architecture was extended into a standard 3D U-Net.

The 3D model follows the same encoder-decoder structure but replaces two-dimensional convolutions, pooling, and transposed convolutions with their three-dimensional equivalents. This allows the network to extract features across the depth, height, and width of the input volume rather than processing each MRI slice independently.

The model therefore has access to neighbouring slices within each input patch and can learn how anatomical structures change across depth. This may be beneficial for identifying curved anatomical boundaries and maintaining spatial continuity between consecutive slices.

Two bottleneck configurations were implemented:

| Model | Encoder Channels | Bottleneck Channels | Trainable Parameters |
|---|---|---:|---:|
| 3D U-Net 256 | 32, 64, 128, 256 | 256 | 14,975,238 |
| 3D U-Net 512 | 32, 64, 128, 256 | 512 | 22,578,438 |

Both configurations produce six output logits per voxel using a final 1×1×1 convolution. The network uses four downsampling stages, requiring input spatial dimensions divisible by 16.

#### 3D Patch Sampling

Instead of processing entire MRI volumes during training, the model uses patches of 96 × 96 × 96 voxels.

Patch-based training reduces the memory requirements associated with full-volume 3D convolution and allows multiple training samples to be extracted from each MRI scan.

Class-balanced patch sampling was used with a foreground sampling probability of 0.33 to increase the representation of anatomical structures that may otherwise appear infrequently in uniformly sampled patches.

A preliminary patch-sampling analysis was conducted using 5,000 patches per configuration. With class-balanced sampling, prostate presence increased from 36.8% in 64³ patches to 64.5% in 96³ patches, while rectum presence increased from 35.9% to 63.7%.

Therefore, 96³ patches were selected to provide greater spatial context and more frequent inclusion of these anatomical structures. However, this comparison measured class presence rather than segmentation performance, and no 64³ model was trained.

#### Full-Volume Validation

During validation, predictions are generated for entire MRI volumes using sliding-window inference.

The volume is divided into overlapping 3D patches, each of which is processed by the model. Predictions from overlapping regions are combined using uniform averaging to reconstruct the complete volumetric segmentation.

An overlap of 50% was used to provide coverage across the volume and reduce the dependence of predictions on individual patch boundaries.

This allows the model to be trained using manageable patches while still being evaluated on complete MRI volumes.

### 4.3 Improved Residual 3D U-Net

An improved 3D U-Net architecture was developed to investigate whether residual connections can further improve segmentation performance.

The standard U-Net uses skip connections between encoder and decoder stages to preserve spatial information. Residual connections operate differently: they introduce shortcut paths within convolutional blocks, allowing the block's input features to be combined with its transformed output.

The residual operation can be expressed as:

$$
y = \operatorname{ReLU}(F(x) + S(x))
$$

where $F(x)$ represents the convolutional transformation and $S(x)$ represents an identity or projection shortcut used to match feature dimensions.

These connections provide an additional pathway for information and gradients to propagate through the network, potentially making optimisation more effective.

Each residual block contains two 3D convolution operations with batch normalisation and ReLU activation. Where input and output feature dimensions differ, a 1×1×1 convolution followed by batch normalisation is used to project the input before residual addition.

The improved architecture replaces all nine standard convolution blocks with residual blocks while retaining the original U-Net encoder-decoder structure, volumetric input format, six output classes, and long skip connections.

#### Controlled Architectural Comparison

For the improved experiment, the 256-channel bottleneck configuration was selected to limit computational cost and allow a direct comparison with the standard 3D U-Net 256.

| Model | Bottleneck Channels | Trainable Parameters |
|---|---:|---:|
| Standard 3D U-Net 256 | 256 | 14,975,238 |
| Residual 3D U-Net 256 | 256 | 15,194,278 |

The residual model introduces 219,040 additional trainable parameters, an increase of approximately 1.5%.

Both models use the same patient-level dataset split, patch size, sampling strategy, loss function, optimiser, training budget, and validation procedure.

By keeping these experimental settings consistent, the comparison aims to investigate whether replacing standard convolutional blocks with residual blocks improves segmentation performance. However, because the residual architecture also introduces additional parameters, any observed difference cannot be attributed exclusively to the shortcut connections.

The effectiveness of this modification is assessed through validation Dice scores and computational cost, including training time and GPU memory usage.

## 5. Experimental Setup

All experiments were conducted using PyTorch, with the 3D models trained on an NVIDIA L4 GPU with approximately 23 GiB of usable VRAM. The experiments compared segmentation performance across the 2D U-Net, standard 3D U-Net and improved Residual 3D U-Net architectures.

The same patient-level dataset splits and evaluation metrics were used across the experiments to maintain consistency.

### 5.1 Training Configuration

All models were trained using the AdamW optimiser with a combined Dice and Cross-Entropy loss. A fixed training budget of 20 epochs was planned for each experiment, without learning-rate scheduling or early stopping.

The key training settings are summarised below.

| Hyperparameter | 2D U-Net | Standard 3D U-Net | Residual 3D U-Net |
|---|---|---|---|
| Optimiser | AdamW | AdamW | AdamW |
| Learning rate | 0.0003 | 0.0003 | 0.0003 |
| Weight decay | 0.00001 | 0.00001 | 0.00001 |
| Planned epochs | 20 | 20 | 20 |
| Loss function | 0.5 Dice + 0.5 CE | 0.5 Dice + 0.5 CE | 0.5 Dice + 0.5 CE |
| Random seed | 3710 | 3710 | 3710 |
| Training input | Individual 2D slices | 96³ patches | 96³ patches |
| Patch sampling | Not applicable | Class-balanced | Class-balanced |
| Foreground sampling probability | Not applicable | 0.33 | 0.33 |
| Batch size | 8 | 2 | 2 |
| Evaluation batch size | 16 slices | 1 window | 1 window |
| Sliding-window overlap | Not applicable | 50% | 50% |
| Mixed precision | Enabled | Enabled | Enabled |

The 3D experiments used 1,291 training patches per epoch, representing a volume-equivalent sampling budget across the 136 training volumes. Both standard 3D variants used identical training settings, allowing the effect of bottleneck capacity to be investigated.

The improved Residual 3D U-Net was configured using the same training parameters as the standard 3D U-Net 256, with the principal difference being the introduction of residual convolution blocks.

The standard 3D U-Net 256 experiment was interrupted after 19 completed epochs due to a disconnected terminal session. Its best model checkpoint had already been saved at epoch 15. The 3D U-Net 512 and residual 3D U-Net 256 each completed all 20 epochs.

### 5.2 Loss Function

A weighted combination of Dice loss and Cross-Entropy loss was used for model optimisation:

$$
\mathcal{L} = 0.5\,\mathcal{L}_{\text{Dice}} + 0.5\,\mathcal{L}_{\text{CE}}
$$

Cross-Entropy loss penalises incorrect voxel or pixel classification, while Dice loss encourages overlap between the predicted segmentation and its corresponding ground-truth mask.

Combining these losses allows the models to optimise both classification accuracy and segmentation overlap.

The same loss formulation was retained across all experiments to avoid introducing loss-function differences into the architectural comparisons.

### 5.3 Evaluation Metrics

The primary evaluation metric was the mean foreground Dice Similarity Coefficient (DSC), excluding the background class.

For an individual class:

$$
\text{DSC} = \frac{2\,TP}{2\,TP + FP + FN}
$$

where $TP$ represents true positives, $FP$ represents false positives and $FN$ represents false negatives.

The mean foreground Dice score was calculated across the five anatomical classes: bladder, body, bone, rectum and prostate.

Two complementary measurements were recorded:

- **Global mean foreground Dice:** Calculated by aggregating predictions and ground-truth labels across the validation volumes.
- **Per-volume Dice:** Calculated for individual MRI volumes and summarised using the mean and standard deviation.

Per-class Dice scores were also recorded to investigate whether volumetric context affected segmentation performance differently across anatomical structures.

### 5.4 Validation and Model Selection

Model performance was evaluated on the validation dataset after each training epoch.

For the 2D models, predictions from individual slices were reassembled into complete volumes before calculating validation metrics.

For the 3D models, full-volume predictions were generated using sliding-window inference with a patch size of 96³ and 50% overlap. Predictions in overlapping regions were combined using uniform averaging.

The checkpoint with the highest global mean foreground validation Dice was selected as the best-performing model for each architecture.

### 5.5 Computational Cost

In addition to segmentation accuracy, computational efficiency was considered when evaluating the architectures.

The following measurements were recorded where available:

- Number of trainable model parameters.
- Training time and time per epoch.
- GPU memory allocation during training.
- Best validation Dice relative to computational requirements.

These measurements allow the project to investigate whether improvements in segmentation accuracy justify the additional computational cost associated with 3D convolutions, increased bottleneck capacity and residual connections.

Because the 2D and 3D models use different input representations and training workloads, their training times are not perfectly controlled measures of architectural efficiency. Computational comparisons are therefore interpreted alongside differences in data processing and validation procedures.

### 5.6 Reproducibility

A fixed random seed of 3710 was used for the experiments, and the same patient-level data split was maintained across all models.

Training configurations, epoch histories and validation scores were recorded. The best-performing checkpoints were saved for subsequent analysis.

Only one training run per configuration was conducted, meaning the reported results describe performance under the selected experimental conditions rather than an estimate of variability across random seeds.

## 6. Results

This section presents the segmentation performance of the 2D and 3D U-Net architectures, comparing validation Dice scores, class-specific performance, model complexity and computational cost.

### 6.1 Overall Segmentation Performance

The best validation mean foreground Dice scores for each model are summarised below.

| Model | Trainable Parameters | Best Validation Dice | Best Epoch | Completed Epochs |
|---|---:|---:|---:|---:|
| 2D U-Net 256 | 5,140,166 | 0.9131 | 15 | 20 |
| 2D U-Net 512 | 7,762,630 | 0.9113 | 12 | 20 |
| 3D U-Net 256 | 14,975,238 | 0.9226 | 15 | 19 |
| 3D U-Net 512 | 22,578,438 | 0.9247 | 20 | 20 |
| **Residual 3D U-Net 256** | **15,194,278** | **0.9298** | **18** | **20** |

The standard 3D U-Net models achieved higher mean foreground validation Dice scores than their corresponding 2D configurations.

The 3D U-Net 256 achieved a Dice score of 0.9226 compared with 0.9131 for the 2D U-Net 256, an absolute increase of approximately 0.95 percentage points.

Similarly, the 3D U-Net 512 achieved 0.9247 compared with 0.9113 for the 2D U-Net 512, an increase of approximately 1.34 percentage points.

The highest validation Dice among the five experiments was achieved by the Residual 3D U-Net 256 (0.9298 at epoch 18), approximately 1.67 percentage points above the 2D U-Net 256 baseline.

### 6.2 Effect of Bottleneck Capacity

The two bottleneck configurations produced different results for the 2D and 3D architectures.

| Comparison | Difference in Validation Dice | Additional Parameters |
|---|---:|---:|
| 2D 512 vs 2D 256 | −0.0018 | +2,622,464 |
| 3D 512 vs 3D 256 | +0.0021 | +7,603,200 |

Increasing the bottleneck capacity from 256 to 512 did not improve performance for the 2D U-Net, with the larger model achieving a slightly lower validation Dice.

For the 3D U-Net, increasing bottleneck capacity produced a small improvement of approximately 0.21 percentage points, despite increasing the number of trainable parameters by approximately 51%.

These results indicate that increasing model capacity does not necessarily produce a proportional improvement in segmentation performance.

### 6.3 Per-Class Segmentation Performance

The best 2D U-Net 256 checkpoint achieved the following class-specific Dice scores.

| Anatomical Class | 2D U-Net 256 |
|---|---:|
| Bladder | 0.9835 |
| Body | 0.9133 |
| Bone | 0.9563 |
| Rectum | 0.8750 |
| Prostate | 0.8373 |
| **Mean Foreground Dice** | **0.9131** |

The prostate and rectum achieved lower Dice scores than several of the other anatomical classes, indicating that these structures were more challenging to segment under the baseline configuration.

For comparison, the 3D U-Net 256 produced the following values at epoch 19:

| Anatomical Class | 3D U-Net 256 — Epoch 19 |
|---|---:|
| Bladder | 0.9802 |
| Body | 0.9072 |
| Bone | 0.9508 |
| Rectum | 0.8704 |
| Prostate | 0.8672 |
| **Mean Foreground Dice** | **0.9152** |

These epoch-19 results show a higher prostate Dice than the best 2D U-Net 256 checkpoint. However, the values are not taken from the best 3D checkpoint, which occurred at epoch 15.

The class-specific scores for the best 3D U-Net 256 checkpoint (epoch 15) have not been extracted here, so the epoch-19 figures should not be treated as a best-checkpoint comparison. The residual model's best-checkpoint class-specific results are shown below.

| Anatomical Class | Residual 3D U-Net 256 — Best Epoch 18 |
|---|---:|
| Bladder | 0.9832 |
| Body | 0.9202 |
| Bone | 0.9626 |
| Rectum | 0.8913 |
| Prostate | 0.8916 |
| **Mean Foreground Dice** | **0.9298** |

The residual model's prostate validation Dice of 0.8916 is higher than the 2D U-Net 256 best-checkpoint prostate Dice of 0.8373 by approximately 5.43 percentage points. This is an observed comparison, not evidence that the gain was caused exclusively by residual connections or 3D context.

### 6.4 Training Progression

The standard 3D U-Net 256 reached its highest validation Dice of 0.9226 at epoch 15. Training was interrupted by a terminal disconnection after epoch 19, but the best-performing checkpoint had already been saved.

The standard 3D U-Net 512 completed its planned 20 epochs and achieved its highest validation Dice of 0.9247 at epoch 20. The Residual 3D U-Net 256 also completed 20 epochs, reaching its highest validation Dice of 0.9298 at epoch 18.

Both models showed substantial improvements in validation performance during their initial training epochs, followed by smaller fluctuations as training progressed.

The 3D U-Net 256 also experienced an unusually low validation Dice of 0.4185 at epoch 10, despite achieving 0.9058 at epoch 9 and recovering to 0.9134 at epoch 11. This indicates an isolated period of validation instability, although its underlying cause was not established.

The training scripts write per-epoch loss and Dice curves (`loss_curves.png` and `dice_curves.png`) to each run directory, as described in Section 9.4. These figures are not reproduced in this report because the run directories remain on the UQ GPU Lab and could not be retrieved before submission. The training progression is therefore summarised from the recorded epoch values:

| Model | Best validation Dice (epoch) | Notable progression |
|---|---|---|
| 2D U-Net 256 | 0.9131 (15) | Best checkpoint reached before the end of the 20-epoch budget |
| 2D U-Net 512 | 0.9113 (12) | Best checkpoint reached earliest of the five models |
| Standard 3D U-Net 256 | 0.9226 (15) | 0.9113 at epoch 8; isolated drop to 0.4185 at epoch 10; run stopped after epoch 19 |
| Standard 3D U-Net 512 | 0.9247 (20) | Best result at the final epoch |
| Residual 3D U-Net 256 | 0.9298 (18) | 0.9197 at epoch 8, ahead of the standard 3D U-Net 256 at the same epoch |

### 6.5 Computational Cost

Model complexity was compared using the number of trainable parameters.

| Model | Trainable Parameters |
|---|---:|
| 2D U-Net 256 | 5.14 million |
| 2D U-Net 512 | 7.76 million |
| 3D U-Net 256 | 14.98 million |
| 3D U-Net 512 | 22.58 million |
| Residual 3D U-Net 256 | 15.19 million |

The 3D architectures contained substantially more parameters than their corresponding 2D variants because three-dimensional convolutional kernels operate across an additional spatial dimension.

The standard 3D U-Net 256 required an average of 345.5 seconds per completed epoch (6,565.2 seconds across 19 epochs), compared with 56.8 seconds per epoch (1,136.2 seconds across 20 epochs) for the 2D U-Net 256.

These timing values are descriptive rather than a controlled speed benchmark, since the 2D and 3D models used different training inputs, sampling procedures and validation workloads.

The recorded training runtimes and peak GPU memory allocations are summarised below. Training duration excludes the initial data-preload step; the 3D U-Net 256 duration was reconstructed from its 19 completed epoch records because the interrupted run did not produce a final `summary.json`.

| Model | Completed epochs | Recorded training time | Mean epoch time | Peak GPU allocated memory |
|---|---:|---:|---:|---:|
| 2D U-Net 256 | 20 | 1,136.2 s (18.9 min) | 56.8 s | 837.7 MiB |
| 2D U-Net 512 | 20 | 1,136.6 s (18.9 min) | 56.8 s | 875.7 MiB |
| 3D U-Net 256 | 19 | 6,565.2 s (109.4 min) | 345.5 s | 2,040.0 MiB* |
| 3D U-Net 512 | 20 | 6,571.5 s (109.5 min) | 328.6 s | 2,148.3 MiB |
| Residual 3D U-Net 256 | 20 | 7,498.3 s (125.0 min) | 374.9 s | 2,608.9 MiB |

\*The 3D U-Net 256 memory figure is from its recorded per-epoch training history, rather than a completed run summary. The initial data-preload times were approximately 241–254 seconds across the five experiments.

Peak memory is the peak allocated PyTorch GPU memory during training (batch size 8 for 2D and 2 for 3D, mixed precision). The 2D and 3D measurements were not taken under identical batch sizes, so they describe each training configuration rather than a per-sample cost.

**Inference latency was not measured.** No per-slice or per-volume inference timing was recorded for the trained checkpoints, so inference runtime is missing from this resource profile. No estimate is given here.

The standard 3D U-Net 256 also recorded 3,800.7 seconds of patch-planning activity across its 19 epochs (200.0 seconds per epoch on average). This identifies patch preparation as a substantial computational component, although the planning timer should not be added to the total epoch duration without checking whether the measurements overlap. Overall, the 3D models attained higher validation Dice at considerably greater recorded training time and GPU memory usage under these pipelines.

### 6.6 Improved Residual 3D U-Net

The improved Residual 3D U-Net 256 was designed to investigate whether residual convolution blocks improve segmentation performance relative to the standard 3D U-Net 256.

Both models use the same bottleneck capacity, patient-level data splits, training settings, loss function and validation procedure.

| Metric | Standard 3D U-Net 256 | Residual 3D U-Net 256 |
|---|---:|---:|
| Parameters | 14,975,238 | 15,194,278 |
| Best validation Dice | 0.9226 | **0.9298** |
| Best epoch | 15 | 18 |
| Completed epochs | 19 | 20 |
| Recorded training time | 6,565.2 s (19 epochs) | 7,498.3 s (20 epochs) |

The residual model improved mean foreground validation Dice by approximately 0.72 percentage points relative to the standard 3D U-Net 256 (0.9298 versus 0.9226), while adding approximately 1.5% more trainable parameters. It also exceeded the larger standard 3D U-Net 512 by approximately 0.51 percentage points on the validation set. All comparisons are from individual training runs.

The interpretation of these results is presented in Section 7.

### 6.7 Summary of Findings

Across the completed baseline experiments, the standard 3D U-Net models achieved higher validation mean foreground Dice scores than the 2D architectures.

Among the five models, the Residual 3D U-Net 256 achieved the highest validation mean foreground Dice (0.9298). The standard 3D U-Net 512 achieved 0.9247, while the standard 3D U-Net 256 achieved 0.9226 with substantially fewer parameters than the 512-channel variant.

These findings are consistent with the hypothesis that volumetric spatial context may improve anatomical segmentation. However, the experiments do not independently isolate spatial dimensionality from differences in parameter count and training procedures.

The residual architecture improved validation Dice relative to both standard 3D models under the tested configuration, although further runs would be needed to establish the reliability of this difference.

### 6.8 Qualitative Failure Analysis (Incomplete)

A qualitative analysis of 3–5 representative failure cases was planned: image, ground-truth and predicted-mask panels from the final checkpoints, focusing on the prostate base and apex and the prostate–rectum and prostate–bladder interfaces. It was not completed. The final checkpoints are stored on the UQ GPU Lab, and prediction overlays from them were not generated within the project time frame. Overlays from earlier one-epoch smoke tests are not representative of the final models and are deliberately not used here.

As a result, this report does not identify where the models fail anatomically. The lower rectum and prostate Dice scores in Section 6.3 suggest that these smaller structures are where errors are concentrated. However, this is an inference from the aggregate scores and has not been confirmed visually. `predict3d.py` and `predict.py` (Section 9.5) produce the overlays and per-scan Dice needed for this analysis.

## 7. Discussion and Limitations

### 7.1 Comparison of 2D and 3D U-Net Performance

The experimental results indicate that the standard 3D U-Net architectures achieved higher mean foreground validation Dice scores than their corresponding 2D baselines.

The 2D U-Net 256 achieved a best validation Dice of 0.9131, compared with 0.9226 for the standard 3D U-Net 256, representing an absolute improvement of approximately 0.95 percentage points. Similarly, the 3D U-Net 512 achieved a Dice score of 0.9247, compared with 0.9113 for the 2D U-Net 512, an improvement of approximately 1.34 percentage points.

These results are consistent with the primary hypothesis that incorporating neighbouring MRI slices may improve anatomical segmentation. Unlike the 2D U-Net, which processes individual cross-sectional slices, the 3D architecture can learn spatial relationships across depth, height, and width. This additional information may help the model recognise anatomical continuity and structures whose shapes change across successive slices.

However, the observed improvement cannot be attributed exclusively to the use of three-dimensional spatial information. The 3D architectures also contain more trainable parameters than their corresponding 2D models and use different input representations and training procedures. Consequently, the results demonstrate an improvement under the implemented experimental configurations rather than proving that volumetric context alone caused the difference.

### 7.2 Effect of Bottleneck Capacity

Increasing the bottleneck capacity from 256 to 512 produced different outcomes for the 2D and 3D architectures.

For the 2D U-Net, the 256-channel model achieved a higher validation Dice of 0.9131 compared with 0.9113 for the 512-channel model. This suggests that increasing model capacity did not improve segmentation performance under the selected training configuration.

In contrast, the 3D U-Net 512 achieved a validation Dice of 0.9247, compared with 0.9226 for the 3D U-Net 256. This represents an improvement of approximately 0.21 percentage points, despite the larger 3D model containing approximately 51% more trainable parameters.

The relatively small improvement suggests diminishing returns from increasing bottleneck capacity in the tested 3D configurations. Although the 512-channel model achieved the highest validation Dice of the two standard 3D models, the 256-channel model provided similar segmentation performance with substantially fewer parameters.

Therefore, the 256-channel configuration was selected for the improved residual experiment to investigate whether an architectural modification could provide additional benefits without substantially increasing model complexity.

### 7.3 Effect of Residual Connections

The improved Residual 3D U-Net was developed to investigate whether incorporating residual connections into convolutional blocks could improve segmentation performance.

Residual connections provide additional pathways for information and gradients to propagate through the model. This may support more effective optimisation and feature learning compared with conventional convolutional blocks.

The improved 256-channel model contains 15,194,278 trainable parameters, approximately 1.5% more than the standard 3D U-Net 256, which contains 14,975,238 parameters.

Both models use the same dataset split, input patch dimensions, loss function, optimiser, sampling procedure, and validation method. This provides a relatively controlled comparison of the residual architecture against the standard 3D model.

The residual model achieved a best mean foreground validation Dice of 0.9298 at epoch 18, compared with 0.9226 at epoch 15 for the standard 3D U-Net 256. This is an increase of approximately 0.72 percentage points with a 1.5% increase in parameter count. It also exceeded the standard 3D U-Net 512 result of 0.9247.

The residual model reached 0.9197 validation Dice by epoch 8, compared with 0.9113 for the standard 3D U-Net 256 at epoch 8. This suggests faster early improvement in validation Dice in this particular run, but it does not independently establish faster convergence in computation time or generalise to repeated runs.

Any observed difference should be interpreted cautiously because only one training run was conducted for each architecture. Furthermore, the residual architecture introduces a small increase in trainable parameters, so any performance difference cannot be attributed exclusively to the shortcut connections.

### 7.4 Computational Cost and Performance Trade-offs

Although the 3D models achieved higher validation Dice scores, they also required greater computational resources.

The standard 2D U-Net 256 contains approximately 5.14 million trainable parameters, while the standard 3D U-Net 256 contains approximately 14.98 million. The increase reflects the additional spatial dimension used by three-dimensional convolutional operations.

The standard 3D U-Net 512 contains approximately 22.58 million parameters, making it the largest architecture evaluated in this project.

Training the 3D models also required substantially more time under the implemented experimental setup. The 3D U-Net 256 recorded 6,565.2 seconds across 19 completed epochs, compared with 1,136.2 seconds for the 2D U-Net 256 across 20 epochs, excluding dataset preloading. The 3D U-Net 512 and residual 3D U-Net 256 required 6,571.5 and 7,498.3 seconds respectively across their completed 20-epoch runs.

However, these runtime measurements are not directly comparable as isolated measures of computational efficiency. The 2D models were trained on individual slices, while the 3D models processed volumetric patches and performed full-volume sliding-window validation. Differences in data preparation, hardware execution, and validation workload also affect the measured times.

The results therefore demonstrate an accuracy–computation trade-off. The 3D architectures produced approximately 0.95–1.34 percentage points higher mean foreground validation Dice than their corresponding 2D baselines, but required more parameters and longer training under the tested configurations.

Whether this improvement justifies the additional computational cost depends on the intended application. For applications where segmentation accuracy is particularly important, the additional computation may be worthwhile. However, where processing speed, memory consumption, or hardware availability are constrained, the 2D architecture may remain a practical alternative.

The small performance difference between the standard 3D 256 and 512 models also suggests that increasing model capacity should be evaluated against its computational requirements rather than assuming that larger architectures will produce substantial improvements.

### 7.5 Limitations

Several limitations should be considered when interpreting the experimental results.

**Limited experimental repetitions:** Each architecture was evaluated using a single random seed. Consequently, the reported differences may be affected by random initialisation, patch sampling, and training variability. Repeated experiments would be required to assess the consistency of the results.

**Validation cohort size:** The validation dataset contains six patients and 34 MRI scans. Although patient-level separation reduces the risk of data leakage, the relatively small number of validation patients limits confidence in how well the observed performance differences generalise to unseen patients.

**Differences in model complexity:** The 2D and 3D architectures differ in trainable parameter counts as well as spatial dimensionality. Therefore, the experiments cannot isolate the effect of volumetric context independently from differences in model capacity.

**Training interruption:** The standard 3D U-Net 256 completed 19 of its planned 20 epochs after its terminal session was disconnected. Its best validation checkpoint had already been saved at epoch 15. The missing final epoch introduces a small difference in training and checkpoint-selection budgets compared with the 20-epoch models.

**Patch-size selection:** The 96³ patch size was selected based on spatial context and anatomical class-presence analysis. Although 64³ and 96³ patches were compared during sampling analysis, no model was trained using 64³ patches. Therefore, the effect of patch size on actual segmentation accuracy was not independently evaluated.

**Computational comparisons:** Training times reflect the specific hardware, data-processing pipeline, and experiment configurations used. They should not be interpreted as universally applicable runtime estimates.

**Dataset generalisability:** All experiments were conducted using a single MRI dataset. Performance on images acquired using different scanners, imaging protocols, or patient populations has not been established.

**Held-out test evaluation:** A held-out test evaluation was attempted but did not produce a verified, complete comparison of all five models. Consequently, the reported Dice scores represent validation performance rather than an independent estimate of generalisation to unseen test patients. Although the validation and test sets contain different patients, the repeated use of validation results for model selection may introduce selection bias.

**Boundary failure analysis:** Dice is a volume-overlap measure and can hide local boundary errors, such as at the prostate base and apex or at the rectum and bladder interfaces. A slice-level qualitative analysis of individual failure cases was not completed, so these results do not establish where the models fail anatomically.

### 7.6 Future Work

Several extensions could be investigated to build on the findings of this project.

First, repeating the experiments across multiple random seeds would provide a more reliable estimate of model performance and help determine whether the observed Dice differences are consistent.

Second, the improved residual architecture could be extended to a 512-channel bottleneck to investigate whether increased model capacity provides additional benefits when combined with residual connections.

Third, different volumetric patch sizes and sampling strategies could be evaluated through controlled training experiments. This would help determine whether the greater anatomical coverage observed with 96³ patches translates into improved segmentation accuracy.

Further investigations could also include anatomically appropriate MRI data augmentation, alternative loss functions for class imbalance, and optimisation of the data-loading pipeline to reduce computational overhead.

### 7.7 Engineering Recommendation

Based on these validation results, the Residual 3D U-Net 256 is the best candidate to carry forward. It gave the highest validation mean foreground Dice (0.9298) and the highest prostate Dice (0.8916) of the evaluated models, for about 1.5% more parameters than the standard 3D U-Net 256 and fewer parameters than the standard 3D U-Net 512. The cost is roughly 6.6 times the recorded training time of the 2D U-Net 256 (7,498.3 s versus 1,136.2 s) and about 3.1 times its peak allocated GPU memory (2,608.9 MiB versus 837.7 MiB), which fits comfortably on the 23 GiB GPU used here.

Increasing the standard 3D bottleneck to 512 channels is not recommended: it added about 51% more parameters for a 0.21 percentage-point gain. The 2D U-Net 256 remains a reasonable fallback where training time or GPU memory is the main constraint.

This recommendation is limited to validation performance from one run per model. Before the model is used to draft contours for specialist review, it should be evaluated once on the held-out test patients, repeated across seeds, and checked visually for boundary failures at the prostate base and apex and at the rectum and bladder interfaces. These results do not support clinical use.

## 8. Conclusion

This project investigated whether using neighbouring MRI slices through a 3D U-Net improved multi-class anatomical segmentation over a conventional 2D U-Net, and whether residual convolution blocks provided an additional improvement.

Both standard 3D variants achieved higher best mean foreground validation Dice than their corresponding 2D baselines. The 2D U-Net 256 scored 0.9131 compared with 0.9226 for the standard 3D U-Net 256, while the 2D U-Net 512 scored 0.9113 compared with 0.9247 for the standard 3D U-Net 512. These observations are consistent with the hypothesis that additional volumetric context can benefit segmentation, although the models also differ in parameter count and training procedure.

The Residual 3D U-Net 256 achieved the highest validation Dice of 0.9298 at epoch 18, exceeding the standard 3D U-Net 256 by approximately 0.72 percentage points while adding only approximately 1.5% more parameters. Its prostate Dice was 0.8916, compared with 0.8373 for the 2D U-Net 256 best checkpoint. These results suggest a promising benefit from the tested residual 3D configuration, without establishing clinical significance or causation from individual design choices.

These conclusions are limited to the validation cohort and one training run per model. A complete held-out test evaluation was not obtained, and the additional costs of volumetric training must be weighed against the observed gains. Future work could examine repeated seeds, external and held-out test datasets, and other patch sizes to assess whether the improvements generalise.

## 9. Usage and Reproducibility

All commands are run from this folder (`recognition/HipMRI_UNet_hetmehta0/`).

### 9.1 Dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` lists minimum versions. The exact versions used are:

| Environment | Python | PyTorch | Other packages |
|---|---|---|---|
| Local tests (macOS, CPU) | 3.11.17 | 2.14.1 | numpy 2.4.6, pandas 3.0.6, nibabel 5.4.2, matplotlib 3.11.2, pytest 9.1.1 |
| GPU training (UQ GPU Lab) | 3.10 | 2.14.0+cu130 | — |

### 9.2 Data

The scans are the HipMRI 3D volumes on Rangpur (`/home/groups/comp3710/HipMRI_Study_open`, images in `semantic_MRs`, masks in `semantic_labels_only`). The scripts read a manifest CSV (`--metadata`) with one row per scan and the columns `patient_id`, `scan_id`, `image_path`, `mask_path` and `split`. The manifest is not committed. It was built with the patient-level procedure in Section 2.2, using a helper from my earlier feasibility-stage project that is not part of this folder. `dataset.load_manifest` rejects a manifest with patient overlap between splits, a split other than 26/6/6 patients, or a split SHA-256 other than `4da18222…7dd7886` (`dataset.EXPECTED_SPLIT_SHA256`).

### 9.3 Tests

```bash
pytest                                               # CPU tests on synthetic data
HIPMRI_METADATA=/path/to/metadata.csv pytest         # also runs the real-data tests
```

### 9.4 Training

```bash
# 2D baselines: unet2d_256 or unet2d_512
python train.py --model unet2d_256 --metadata /path/to/metadata.csv \
    --out-dir runs/unet2d_256_seed3710_e20

# 3D models: unet3d_256, unet3d_512, resunet3d_256 or resunet3d_512
python train3d.py --model resunet3d_256 --metadata /path/to/metadata.csv \
    --out-dir runs/resunet3d_256_p96_seed3710_e20 --eval-batch-size 1
```

Defaults match Section 5.1. `train3d.py --memory-probe` measures GPU memory on random data without loading the dataset, and `--overfit-patches 4` (3D) or `--overfit-subset 8` (2D) runs an overfitting sanity check.

Each run directory contains `history.csv`, `summary.json`, `best.pt`, `last.pt`, `loss_curves.png` and `dice_curves.png`. `best.pt` is the epoch with the highest global mean foreground validation Dice. The test split is never loaded during training.

### 9.5 Evaluation and Example Predictions

```bash
python predict.py --checkpoint runs/unet2d_256_seed3710_e20/best.pt \
    --metadata /path/to/metadata.csv --split val --out-dir predictions/unet2d_256_val

python predict3d.py --checkpoint runs/resunet3d_256_p96_seed3710_e20/best.pt \
    --metadata /path/to/metadata.csv --split val --out-dir predictions/resunet3d_256_val
```

**Hardware.** The 2D models can run on any CUDA GPU with a few GiB of memory. The 3D runs used an NVIDIA L4 (about 23 GiB usable) and peaked at about 2.6 GiB of allocated memory at batch size 2 with mixed precision. Training preloads all training and validation volumes into RAM, which took about 4 minutes on the GPU Lab machine (30 GiB RAM). The CPU test suite runs without a GPU or the dataset.

`--split` has no default, so the test split is only used when requested. Each script prints per-scan Dice, global and per-volume mean foreground Dice, writes `metrics.json`, and saves image / ground-truth / prediction overlays for the first `--num-overlays` volumes (default 3) on the axial slice with the most foreground.

## 10. Artificial Intelligence Usage Disclosure

**Tools used.** ChatGPT, OpenAI Codex and Claude.

**What AI assisted with.**

- Routine implementation and engineering work, such as scaffolding data loading, training, inference and plotting code.
- Writing automated tests that run before models are trained and evaluated.
- Debugging.
- Setting up and configuring the Rangpur computing cluster and UQ GPU Lab.
- Report structuring, Markdown formatting and technical wording.

**My role and how AI output was verified.** I directed the investigation, chose the architectures and experiments, reviewed the implementations, interpreted the results and approved every change. AI-suggested code was reviewed before it was committed, and automatic commits were not enabled. The test suite was run before training, the code was checked on the real data, and I recorded the Dice scores from the training outputs myself. When the AI implemented something incorrectly, I identified it and steered the correction.

## 11. References

Gao, Y., Jiang, Y., Peng, Y., Yuan, F., Zhang, X., & Wang, J. (2025). Medical image segmentation: A comprehensive review of deep learning-based methods. *Tomography, 11*(5), Article 52. https://doi.org/10.3390/tomography11050052

Palazzo, S., Zambetta, G., & Calbi, R. (2024). An overview of segmentation techniques for CT and MRI images: Clinical implications and future directions in medical diagnostics. *Medical Imaging Process & Technology, 7*(1), Article 7227. https://doi.org/10.24294/mipt7227
