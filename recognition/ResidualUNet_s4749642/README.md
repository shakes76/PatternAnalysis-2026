# Prostate Segmentation with Baseline and Residual 2D U-Nets

**Author:** Andre Vasquez  
**Student ID:** s4749642  
**Course:** COMP3710  
**Proposed difficulty:** Open-Ended Project 2.2, Normal

## 1. Project Overview

This project investigates prostate segmentation from 2D HipMRI slices through a baseline U-Net, and an improved 2D U-Net model. The improved U-Net model was constructed using an Isensee-inspired residual encoder adaptation. The prototype generates suggested prostate outlines for specialist review in radiotherapy planning.

The models are compared using Dice, IoU, HD95, computational resource use during prediction, and qualitative failure analysis. Particular emphasis was given to whether overlap conceals local boundary score - boundary score is important in the context of prostate segmentation.

Due to the nature of radiotherapy, an accurate boundary is preferable over a model that overshoots its overlap. The aim is to determine whether the architectural modifications provide useful improvements that justify their additional computing cost.

### Project Development Process (And Its Relation to Repo Structure)

This project was originally developed in notebooks, which can be found in the development folder of the repo. These notebooks do not necessarily reflect the desired final structure of the codebase (some aspects of prediction were not separated from the training notebooks). 

Regardless, they were included as they were an foundational part of the development process and initial experimentation (including a smoke-test initial residual block implementation). These functionalities were refactored to better suit the .py format.

The final intended functionality can be found in the main repository (.py files).

## 2. Feasibility Review

### User Need, Scope & Acceptance Criteria

The prototype will suggest prostate outlines on 2D HipMRI slices for an imaging specialist to review and correct. It will compare standard and modified 2D U-Nets to investigate whether architectural changes improve difficult boundaries, particularly at the prostate base and apex, sufficiently to justify additional computing costs.

Acceptance criteria are to:

- Achieve prostate Dice over **0.75** on a patient-independent test set.
- Compare per-class Dice, IoU and 95th-percentile Hausdorff distance (HD95) on identical test data, assessing base and apex slices separately where identifiable.
- Examine **3–5 incorrect outlines** using ground-truth, prediction and error overlays.
- Complete initial experiments within a provisional A100 GPU-hour budget, recording peak GPU memory and inference time per slice.

### Model Choice & Course Concepts

The project will use HipMRI `keras_slices_data`, with normalised images, verified prostate labels and patient-separated training, validation and test sets. A modified 2D U-Net is proposed for Normal difficulty, using standard U-Net as the baseline.

Course concepts supporting this choice include convolutional feature extraction, hierarchical learning and receptive fields: early layers detect local details, while deeper layers capture broader anatomical features. Isensee et al. introduce residual blocks, strided convolutions, instance normalisation, leaky ReLU and upsampling followed by convolution. These modifications will be considered for 2D adaptation, initially focusing on residual blocks.

### Preliminary Feasibility Evidence

The data audit identified **12,660 paired slices and masks across 38 patients**, split into training (31), validation (4) and test (3), with no missing pairs or overlapping patient IDs. One inspected pair had matching dimensions of **256 × 128**.

A four-level baseline U-Net, adapted from GeeksforGeeks’ *Image Segmentation Using U-Net* into PyTorch, and a residual variant each completed **200 updates on 32 training slices**, using batch size 4 on an A100 40 GB GPU. Mean cross-entropy loss decreased from 1.7609 to 0.3589 for the baseline and from 1.6477 to 0.2194 for the residual variant between the first and last five updates. Peak PyTorch-allocated memory was 0.167 and 0.182 GiB, respectively, with mean update times of 6.71 and 7.70 ms after warm-up. These results support both pipelines’ feasibility; the residual variant achieved lower training loss with additional computing cost.

When comparing the training slices, the residual variant achieved higher Dice and IoU for labels 0–3. Dice increased from 0.952 to 0.972 for label 0, 0.907 to 0.940 for label 1, and 0.166 to 0.593 for label 2. Label 3 showed a smaller overlap improvement overall: Dice increased from 0.488 to 0.523 and IoU from 0.445 to 0.476. However, its mean HD95 worsened from 17.145 to 24.352 pixels, suggesting that better overlap did not consistently produce more accurate boundaries.

### Risks, Budget & Fallback

When considering risks, under-segmentation and over-segmentation should be distinguished, including which model exhibits each more frequently and their potential implications for radiotherapy planning.

Initial experiments will use batch size 16 within the provisional budget, revised after timing complete epochs. The next experiment will investigate different encoder residual connections, which learn corrections to existing features and provide direct gradient paths, alongside further reading of Isensee’s implementation and verification of prostate label identification.

Validation performance and resource use will be compared with the baseline. If unsuccessful, upsampling followed by convolution offers an alternative modification drawn from the Isensee paper. Standard U-Net remains a fallback at the Easy difficulty ceiling.

## 3. Dependencies and Reproducibility

Experiments were conducted using Python **3.10.12** on an
**NVIDIA A100-SXM4-40GB** GPU through Lambda. PyTorch was built with
CUDA **12.8**.

### Dependencies

| Package | Exact version |
|---|---|
| torch | 2.7.0 |
| numpy | 1.26.4 |
| nibabel | 5.4.2 |
| pandas | 1.3.5 |
| matplotlib | 3.5.1 |
| scipy | 1.13.1 |
| tqdm | 4.70.1 |
| nbformat | 5.11.1 |


### Reproducibility

Python, NumPy and PyTorch random seeds were set to `42`. All models used identical patient-independent splits: 31 training patients,
four validation patients and three test patients. Images were normalised per slice and centrally padded to 256 × 144 pixels. Mask padding was excluded from loss and metric calculations.

Experiments were conducted on an NVIDIA A100 40 GB GPU using Lambda. Both the baseline and modified models used batch size 16, Adam with a learning rate of `1e-3`, and cross-entropy loss. Training was limited to 20 epochs, with a one-hour soft limit checked after each completed epoch. The checkpoint with the highest validation prostate Dice was selected for evaluation.

Python, NumPy and PyTorch random seeds were set to `42`. These settings improve reproducibility, although GPU operations may still introduce numerical variation. All models use the same patient-independent data splits and preprocessing.

## 4. Example Inputs, Outputs, and Visual Plots

### Inputs

An example of an input is the 12 MRI images slices for patient 004 displayed below:


![Input MRI](images/input_mri.png)

### Labels

The label for MRI slice 40 for this patient is displayed below:

![MRI Labels](images/label_slice_40.png)

### Visual Plots of Outputs

#### Untrained

A result of an output of a model that has not been trained is shown in the prediction plot and shown below:

![Untrained Output](images/untrained.png)

#### Trained

After training the model, an overlay of the prediction image on top of the original image shows the following images:

![Untrained Output](images/trained.png)



## 5. Pre-processing, Training, Validation, And Testing Splits

The data set are in the form of files that are already segregated for training, validations, and testing sets. Initial data processing just involves matching the different files for each patient and collating the different slices and the corresponding masks of labels for each slice. Upon doing this, the following are the resulting number of training, validation, and testing sets collated. 

```
Split           Pairs   Patients
train           11460         31
validate          660          4
test              540          3
All image–mask pairs matched; no patient overlap.
```

With 16 image pairs for each batch, the following are the batch sizes for training and validation.

```
Training batches: 717
Validation batches: 42
```

After the data loader is constructed with organised and paired input images and its corresponding labels of masks for the segmentation task, the main substantive pre-processing of the images is mainly the normalisation of the images in the form of the code snippet located in the `load_pair` function through the line `image = (image - image.mean()) / (image.std() + 1e-8)`.

Initially, 50 is the number of epochs chosen for the training, however, upon training, this hyper-parameter showed that the threshold DICE was achieved and the improvement of this metric slows down significantly from epoch 20 and onwards. It is due to this that 20 as the target epoch number that is set for training.

## 6. The Open Research Dilemma

### 6.1 Quantitative Benchmarking

All models were evaluated on the same unfamiliar test set of 540 slices sourced from three patients. Best checkpoints were selected using validation prostate Dice.

| Model | Prostate Dice | Prostate IoU | Mean finite HD95 (pixels) | Completely missed prostate slices | False prostate detections on prostate-absent slices |
|---|---:|---:|---:|---:|---:|
| Baseline U-Net | 0.8642 | 0.7608 | 3.2206 | 16 | 2 |
| Improved U-Net | 0.8549 | 0.7466 | 3.3788 | 14 | 4 |

The baseline achieved slightly stronger prostate segmentation (0.8642 Dice baseline vs. 0.8549 Dice improved). IoU was also higher by 1.42 percentage points. Both models exceeded the stated Dice threshold of 0.75 using the pooled test-set calculation.

Despite the baseline model winning out in Dice, the improved model reduced completely missed prostate slices from 16 to 14, suggesting a small benefit in detecting prostate tissue on some difficult slices. However, false detections on prostate-absent slices increased from two to four. This indicates a trade-off between avoiding complete misses and incorrectly predicting prostate where none is labelled.

The improved model did not demonstrate a consistent quantitative advantage, nor did they present an obvious degradation in performance. Qualitative observations should be used in conjunction with these results to determine model feasibility.

### 6.2 Resource Profiling

Inference latency was measured at batch size 1 after warm-up, including the forward pass and class selection. Disk access, preprocessing and data transfers were excluded. Runtime is reported per 2D slice.

| Model | Parameters | Peak training/validation allocated VRAM (GiB) | Peak inference allocated VRAM (GiB) | Mean inference latency (ms/slice) | Training duration (min) |
|---|---:|---:|---:|---:|---:|
| Baseline U-Net | 2,158,502 | 0.727 | 0.129095 | 1.932159 | 43.1 |
| Improved U-Net | 2,204,008 | 0.846| 0.078977 | 3.188301 | 41.5 |

The improved U-Net used approximately 2.1% more parameters and 16.4% more peak training/validation VRAM than the baseline. Its mean inference latency increased from 1.93 to 3.19 ms per slice: approximately 65% slower. 

The improved model showed promising resource benefits, with approximately 39% lower recorded inference memory allocation and 3.7% shorter training time. These suggest potential advantages where GPU memory or training time is constrained.

Despite this, the additional computational cost was not immediately justified by prostate segmentation performance. The improved model reduced complete misses at the cost of lower Dice and IoU and more false positives. I believe that reducing complete misses is a valid result for the trade off of performance in other areas

The baseline, however, provides the stronger balance of overall prostate accuracy and inference speed. 

These findings align with Isensee's research concerning 3D U-Nets. Implementing similar improvements did not yield a substantial increase in performance measured through quantitative metrics alone.

### 6.3 Qualitative Error Autopsies

To analyse the worst-case errors of the models qualitatively, an initial filter was used to identify the worst performing cases through the chosen quantitative benchmarks.

#### Case 1: Missed Prostate Tissue

First, the case where the most missed tissue was examined. This occurred in Case 42, Week 0, Slice 17 (case_042_week_0_slice_17.nii.gz). In this slice, the baseline model has the largest number of reference prostate pixels missing from its prediction. These pixels appear blue in the error overlay.

The table below shows the quantitative performance metrics of the different models at this slice.

| Model | Dice | IoU | HD95 (pixels) | Missed pixels | Extra pixels |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.7404 | 0.5878 | 6.2278 | 208 | 1 |
| Improved | 0.8374 | 0.7203 | 5.0149 | 130 | 16 |

The improved model increased Dice from 0.7404 to 0.8374 and IoU from 0.5878 to 0.7203, indicating closer agreement with the reference prostate mask. Missed pixels decreased from 208 to 130, showing reduced under-segmentation, although extra pixels increased from 1 to 16. HD95 also decreased from 6.2278 to 5.0149 pixels, indicating a reduction in the 95th-percentile boundary distance. Overall, the improved model captures more of the prostate in this slice, but its outline still requires correction.

Upon visual inspection of the slice, a plausible trigger for this error could be limited contrast in the image between the prostate boundary and surrounding tissue, which could contribute to this conservative prediction.

The operational implication would be that a specialist would need to review and expand the predicted outline where prostate tissue has been ommitted. The prediction in this case should be treated as a draft requiring correction. Interestingly, this slice is neither an apex slice nor a base slice.

#### Case 2: Incorrectly Included Tissue

The case at the opposite end of the spectrum was also examined - the most incorrectly included tissue. This occurred in Case 41, Week 0, Slice 39 (case_041_week_0_slice_39.nii.gz). In this slice, the baseline model, despite showing missing the most prostate tissue in the worst case, also predicts the most prostate pixels outside the reference mask. These appear red in the error overlay.

The table below shows the quantitative performance metrics of the different models at this slice.

| Model | Dice | IoU | HD95 (pixels) | Missed pixels | Extra pixels |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.4764 | 0.3127 | 7.2111 | 1 | 210 |
| Improved | 0.5315 | 0.3619 | 6.7082 | 0 | 171 |

The improved model increased Dice from 0.4764 to 0.5315 and IoU from 0.3127 to 0.3619, indicating closer agreement with the reference mask. Extra pixels decreased from 210 to 171, while missed pixels decreased from 1 to 0. This shows reduced over-segmentation while retaining all reference prostate tissue. HD95 also decreased from 7.2111 to 6.7082 pixels. However, the remaining 171 extra pixels and relatively low overlap scores indicate that substantial over-segmentation persists, requiring specialist correction.

Despite the improved performance metrics, it is important to contextualise the results through observing the predictions and comparing them to the slices themselves. Upon observation, both the baseline and improved models drastically overshoot the segmentation area of the prostate. In the radiotheraphy contxt, this would undoubtedly cause the bombardment of radiation in unrelated tissue.

It is important to note that this failure occurs on the outskirts of the prostate, as noted by the high slice number (39). This could be evidence of a failure mode at superior and inferior slices where only tiny slivers of the prostate are visible on both ends.

In the operational context, accounting for this could manifest in leaving apex and base slices for manual operators to label - both for real-life operations and for continued training on those types of slices in particular.

#### Case 3: Largest Improved Model Dice Gain

To the cases in which the improved U-Net performed best and worst, the cases that showed the largest difference (both positive and negative) in Dice between the baseline and Isensee model were examined as well. The largest gain occurred in Case 40, Week 5, Slice 23 (case_040_week_5_slice_23.nii.gz). The Isensee-inspired model’s prostate Dice exceeds the baseline’s by the largest amount among the remaining eligible slices.

| Model | Dice | IoU | HD95 (pixels) | Missed pixels | Extra pixels |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.0000 | 0.0000 | ∞ | 104 | 0 |
| Improved | 0.7262 | 0.5701 | 3.0000 | 43 | 3 |

Although this may seem like a trivial case wherein the baseline did not identify any pixels (when there were reference pixels present), it still gives insight into the behaviours of these two models.

Upon visual inspection of this slice, it was noted that the true area of the prostate in the MRI was blank / black. This exists in contrast with other slices, which show some form of content or colour in the places where the prostate existed. The improved model managed to identify some central pixels of the small prostate mass in the slice, though the actual coverage of pixels were quite small, only spanning a very miniscule portion of the image.

In an operational context, if the baseline model was employed, specialists may have to manually review slices where the number of prostate pixels suddenly drops to zero rather than a gradual decrease, and if the improved model were employed - a similar manual screening process may be used in cases where the number of pixels is extremely small.

If a greater freedom of manual screening is allowed (and the greater operational cost can be spared), perhaps the category of slices that require manual review may also extend to those slices which have black 'voids' present under the constantly white bladder. The bladder (and the relative position of a void) could act as a possible determinant for the prostate area, though this requires further testing.

#### Case 4: Largest Improved Model Dice Loss

The case that displayed the largest loss in Dice when the improved model was used occurred in Case 40, Week 4, Slice 17 (case_040_week_4_slice_17.nii.gz), where tIsensee-inspired model’s prostate Dice falls below the baseline’s by the largest amount among the remaining eligible slices.

| Model | Dice | IoU | HD95 (pixels) | Missed pixels | Extra pixels |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.6705 | 0.5043 | 3.1623 | 57 | 0 |
| Improved | 0.0000 | 0.0000 | ∞ | 115 | 0 |

This case appears to be similar to Case 3, only in the opposite direction: the improved model failed to identify any pixels where they existed. Upon visual inspection of this slice, the same 'void' appeared where the reference prostate pixels existed, suggesting that this is a weakness pattern in both models, a failure mode.

This also ties into the concern about apex and base slices, where these voids could exist in greater numbers (slices where only the edge of the prostate exists, a relatively small area in the larger part of the body that resides under the bladder).

In the operational context, this only draws more attention to the base and apex slices, suggesting that these models are particularly weak in these areas in general.

### 6.4 Engineering Recommendation

Under the tested training configuration, the architectural modifications did not consistently improve prostate segmentation over the baseline. The baseline is therefore the preferred starting point for further prostate-contouring development, subject to the resource comparison and visual failure analysis above.

The Isensee-inspired adaptation offers some benefits, including fewer complete prostate misses and stronger rectum segmentation, but these gains must be weighed against its weaker prostate overlap, additional false detections and measured computational cost. All models still require specialist review and correction in operation.

The qualitative analysis nevertheless identified cases where the improved model reduced substantial baseline errors. In the worst under-segmentation case, missed pixels decreased from 208 to 130; in the worst over-segmentation case, extra pixels decreased from 210 to 171. These examples suggest that the improved model can mitigate some severe local failures despite weaker overall prostate overlap, possibly pointing towards the model being more consistent and middling where the baseline model would fail.

The current results support further development as a specialist-reviewed tool. The baseline provides the stronger starting point because it achieves higher test prostate Dice and IoU with lower inference latency. The improved model offers fewer complete prostate misses and better results in some difficult examples, but these benefits are offset by additional false detections and weaker overall prostate overlap. In general, extra care must be taken in superior and inferior slices (apex and base), as the models are still incapable of consistently identifying tiny segments of prostate in these slices.

I believe that contextually, it is important to prioritise minimising prostate misses, and to offer better results in more complicated situations. The choice can depend on the specialist: some might prefer a tool that aids guide in ambiguous slices, while some prefer a tool that can help in the more trivial cases so human judgment can be used in more difficult circumstances.

Neither model is ready for routine clinical deployment. Complete misses, omitted tissue and incorrectly included tissue persist, so generated outlines require specialist review and correction. Before deployment, evaluation should be expanded beyond three test patients. A practical workflow would have to support manual correction and validation within the intended clinical environment.


## 7. Artificial Inteligence Usage Disclosure

ChatGPT was used in generating early image visualisation helper functions when sifting through data so that a faster avenue to initial understanding of the dataset could be established.

ChatGPT was used in creating a rudimentary U-Net model with a residual block for an initial smoke-test (unused in final testing) and for initial impressions for choosing model values.

ChatGPT was used in refactoring plotting scripts. The slices were verified manually.

ChatGPT was used to help transfer concepts in the Isensee paper (originally about 3D MRI Volumes) into a 2D U-Net model. To verify this code, I scrutinised each line, using online resources to ensure I understand each section of code. To aid in my understanding and demonstrate my knowledge of the code's funcitonality, I added a plethora of comments throughout the code that explains the functionality and my understanding of how each line of code contributes to the model.