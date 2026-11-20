# Improved 2D UNet on 2D HipMRI Data

## 1.0 Introduction

This project focuses on training an Improved 2D UNet on 2D slices from the HipMRI dataset, focusing on identifying the prostate. 

The Improved 2D UNet model is a variation of the original UNet by Ronneberger et al. A UNet is a CNN designed for segmentation. It features a typical encoder-decoder construction, with the addition of skip connections between layers of corresponding depth in the encoder and decoder (`2d_unet_architecture.png`). This allows the model to identify high-level patterns while preserving spatial resolution, which is usually lost down the contraction path. The improved version adds pre-activation residual blocks, dropout layers, and instance normalisation, which improve context propagation, reduce overfitting, and stabilise training (Isensee et al., 2018). Furthermore, deep supervision is used in the expansion path, providing a path for the gradient from the final loss to deeper layers. This improves supervision for earlier layers, enhances convergence speed, and improves model accuracy, as intermediate features become more discriminative.

This model is being trained on 2D MRI slices of the hip to assist with medical identification of the prostate for radiotherapy. Targeted, external-beam radiotherapy requires highly precise identification of the prostate volume, as it is closely surrounded by multiple other important organs, like the bladder. Overidentification can result in unnecessary radiation damage to healthy tissue, while under-identification risks reoccurrence of tumours. While regular UNet is fairly successful at identifying the prostate in wide, central slices (Dice coefficient $> 0.75$), it often struggles at local boundaries, where organs are often much smaller and edges lack clarity. An Improved UNet, with all of its benefits, may be able to improve both overall and boundary accuracy, while requiring relatively similar training resources and time.

---

## 1.1 Model Architecture

The Improved UNet has 5 resolution levels in both the encoder (context pathway) and the decoder (localisation pathway), with 16, 32, 64, 128, and 256 channels respectively. All convolutions are $3 \times 3$ unless stated otherwise. The input is first convolved with stride 1 to produce 16 channels, then passed through a context module. Each context module is a residual block of two $3 \times 3$ convolutions, each preceded by instance normalisation and a leaky ReLU (negative slope $0.01$), with dropout ($p = 0.3$) between the two convolutions to reduce overfitting. A stride-2 convolution then halves the resolution and doubles the channel count, and the result goes into another context module. This pattern repeats down to the 256-channel bottleneck.

In the localisation pathway, each stage upsamples the features by a factor of 2 (nearest neighbour) and applies a $3 \times 3$ convolution to halve the channel count. The result is concatenated with the matching encoder features, and a $3 \times 3$ convolution followed by a $1 \times 1$ convolution fuses them. The final stage omits the $1 \times 1$ convolution, leaving 32 channels. For deep supervision, $1 \times 1$ convolutions produce segmentation maps from the outputs of the three final localisation stages. These are upsampled and summed element-wise, and the result is the final prediction. The network outputs raw logits, and softmax is applied inside the loss function. This is more clearly shown in `2d_improved_unet_architecture.png`.

---

## 2.0 Training

### Training Parameters

#### Data and Input
| Parameter | Value |
| :--- | :--- |
| **Input size** | $1 \times 256 \times 128$ (single-channel slice, centre-cropped) |
| **Output** | 6 class logits per pixel (background, body, bone, bladder, rectum, prostate) |
| **Training / validation / test slices** | 11,460 / 660 / 540 |

#### Optimisation
| Parameter | Value |
| :--- | :--- |
| **Optimiser** | Adam |
| **Learning rate** | $1 \times 10^{-3}$, constant (no scheduler, warm-up, or decay) |
| **Batch size** | 16 |
| **Epochs** | 30 |
| **Batch ordering** | Training set reshuffled every epoch; validation and test in fixed order |
| **Precision** | Full 32-bit floating point (no mixed precision) |
| **Weight initialisation** | PyTorch defaults (no custom initialisation) |
| **Random seed** | 42 |

### Loss Function

The loss is an equally weighted sum of soft Dice loss and cross-entropy, with weight $0.5$ on each (`ce_weight = 0.5`):

$$L = 0.5 \cdot (1 - \text{mean Dice over foreground classes}) + 0.5 \cdot \text{CE}$$

This loss function was used for both models for consistency. The averaging excludes the background class (class 0), so the Dice term concentrates on the five anatomical structures, while the cross-entropy term is unweighted and covers all six classes. Per-batch losses are weighted by batch size when averaged over the epoch, so the smaller final batch isn’t over-counted.

---

## 3.0 Dependencies

All code was written in Python 3.11.16. The following packages are required, and results were produced with the exact versions listed:

| Package | Version | Used For |
| :--- | :--- | :--- |
| **PyTorch** | 2.14.0+cu130 | Models, training, inference |
| **torchvision** | 0.29.0+cu130 | Rotation augmentation (`transforms.functional`) |
| **NumPy** | 2.4.6 | Array handling, seeding |
| **nibabel** | 5.4.2 | Reading `.nii.gz` slices |
| **matplotlib** | 3.11.0 | Training curves and prediction figures |
| **tqdm** | 4.70.0 | Data-loading progress bar |

The remaining imports (`os`, `glob`, `random`, `math`, `time`) were from the Python standard library. Training and predicting used CUDA 13.0 with cuDNN 9.24.0 on an NVIDIA A100-PCIE-40GB GPU.

The data is the HipMRI 2D slice dataset (`keras_slices_data`), which contains the six folders `keras_slices_{train,validate,test}` and `keras_slices_seg_{train,validate,test}`. The dataset root is set by `DATA_ROOT` in `train.py` and `predict.py`.

---

### 3.1 Reproducibility

Each model is trained with `python train.py` and evaluated with `python predict.py`. For the Improved UNet, `train.py` must construct `ImprovedUNet2D` and save to `improved_unet2d.pth`, and `predict.py` must construct `ImprovedUNet2D` and load the same file. For the baseline, use `UNet2D` and `unet2d.pth` in both.

All random number generators are seeded with 42 at the start of training: Python's `random` (used by the augmentation), NumPy, and PyTorch on both CPU and GPU. cuDNN is set to deterministic mode with benchmarking disabled. The data loaders use `num_workers=0`, so the augmentation draws from the single seeded generator in a fixed order, and the shuffled order of training batches is also fixed by the seed.

Results may not be identical for the following reasons:
1. Results can differ across GPU models, CUDA/cuDNN versions, and PyTorch versions, even with identical seeds.
2. The saved checkpoint is the epoch with the best mean foreground validation Dice, so a small change in training can change which epoch is selected.
3. Inference time depends on the hardware and is measured without GPU synchronisation, so it is only indicative.

---

## 4.0 Feasibility Review

### 4.1 User Need, Scope & Acceptance Criteria
The intended users for such a program would be medical imaging specialists for a variety of different body parts. For example, the data used in the project would be for imaging the prostate and the brain.

The purpose of the prototype would be to annotate images received from black and white MRI scans with contours. This would save time and effort for an imaging specialist, who could review and adjust the prototype result instead of needing to manually annotate themselves. They would then make the appropriate recommendations.

The key acceptance criteria would be the DICE score, checking the network’s result’s overlap with a manually specified baseline. However, there are restrictions to the resources allowed to the project; GPU VRAM usage and per-case inference time would need to be within specified limits for the prototype to be considered viable.

### 4.2 Model Choice & Course Concepts
The dataset used would be the 2D HipMRI data, which already has specified training, test, and validation sets.

The baseline would be a standard 2D UNet, with an Improved 2D UNet as the model under study. Thus, the chosen difficulty band is Normal.

A UNet model is ideal, as the problem is segmentation, which UNet (a CNN) is designed to solve. Furthermore, the skip paths between layers allow for spatial resolution to be preserved, which is important within the context, as boundary leakage could result in cancer reoccurrence or radiation proctitis/cystitis (in the case of clinical radiotherapy). An Improved UNet further refines the model, reducing overfitting, improving training stability, etc.

### 4.3 Preliminary Feasibility Evidence
The data contains MRI slices on the z-axis (from feet to head), with folders already dividing the dataset into training, testing, and validation subsets. The data is of NIfTI format, which can be converted into PyTorch tensors.

### 4.4 Risks, Budget & Fallback
Key technical risks include training within a reasonable time due to memory bottlenecks associated with skip connections, overfitting due to a small training set, and poor accuracy on border slices where there is less data.

The computing budget is limited by the availability of the Rangpur cluster, which is under unpredictable, varying loads but does have newer, high-end GPUs for training. The alternative, a virtual GPU environment, is limited to 26 hours.

The next planned experiment would be a 3D UNet, which could further improve spatial resolution and mitigate some of the issues with 2D analysis. If implementation of the Improved UNet on the HipMRI data failed, a fallback could be the OASIS Brain 2D dataset, which would be easier to train for, with fewer sections and easier-to-detect borders.

---

## 5.0 Data Pipeline

While most of the training data had a size of $256 \times 128$ pixels, a small minority instead had size $256 \times 144$. All slices (train, validation, and test) and their masks were thus centre-cropped to $256 \times 128$. Masks were one-hot encoded. Augmentation was applied to the training set only, in 3 ways: rotation, mirroring, and gamma variation. Each slice had an independent $\frac{1}{2}$ chance to be rotated, horizontally mirrored, and gamma varied.

Slices were mirrored across the y-axis, as axial slices are roughly symmetrical and the augmented data would be plausibly possible. Rotation magnitudes were chosen randomly between a minor $-10^\circ$ and $+10^\circ$ (for plausible scanner variation), and interpolated bilinearly for images and using nearest neighbour for masks (to preserve integer labels). Empty space was simply filled with background. Each image was min-max scaled between 0 and 1, raised to a power $\gamma$ sampled log-uniformly between $0.7$ and $1.5$, and rescaled to its original intensity range, to simulate realistic variances in contrast and brightness between scans.

Vertical flips were not considered, as the resulting slice was considered too different from the original slices and could interfere with the accuracy of training. Compared to the original paper, the data pipeline was simplified, excluding complicated transformations like elastic deformation in order to isolate the change in models and reduce potential artifacts introduced by interpolation.

The data was split based on the 6 folders in `keras_slices_data`, which already organised slices into training, validation, and test sets. These consisted of `keras_slices_test`, `keras_slices_train`, `keras_slices_validate`, and their segmentation mask counterparts: `keras_slices_seg_test`, `keras_slices_seg_train`, and `keras_slices_seg_validate`. Each folder was used for its named purpose (e.g., `keras_slices_test` and `keras_slices_seg_test` for testing the trained model).

---

## 6.0 Training and Trained Model Metrics

### General Resource & Performance Metrics

| Metric | Regular UNet | Improved UNet |
| :--- | :--- | :--- |
| **Peak GPU VRAM** | 3266 MB | 1358 MB |
| **Parameter count** | 31,036,806 | 3,181,986 |
| **Inference runtime (per slice)** | 0.70 ms | 0.93 ms |
| **Training time** | 28.7 min | 17.6 min |

### Class Segmentation Accuracy (Dice & IoU)

| Class | Regular UNet (Dice) | Improved UNet (Dice) | Regular UNet (IoU) | Improved UNet (IoU) |
| :--- | :--- | :--- | :--- | :--- |
| **Background** | 0.9965 | 0.9966 | 0.9930 | 0.9933 |
| **Body** | 0.9848 | 0.9857 | 0.9701 | 0.9718 |
| **Bone** | 0.9387 | 0.9413 | 0.8845 | 0.8891 |
| **Bladder** | 0.9665 | 0.9631 | 0.9352 | 0.9288 |
| **Rectum** | 0.8307 | 0.8669 | 0.7105 | 0.7651 |
| **Prostate** | 0.8688 | 0.8748 | 0.7680 | 0.7775 |

---

### 6.1 Training Performance

Training and validation loss were similar across both the baseline (`training_curves_unet.png`) and the Improved UNet (`training_curves-improved.png`), with training loss sharply decreasing within the first few epochs before slowing down, and validation loss reaching a plateau at around $0.17$ within the first few epochs. Dice score for the validation set only marginally increased as epochs increased.

---

### 6.2 Result Discussion

Both trained models had very similar Dice and IoU, with background, body, bone, and bladder classes performing exceptionally well, all being above $0.93$. This was somewhat expected, as these areas are the largest, appear most consistently in all slices, and were the easiest to identify. Rectum and prostate performed slightly worse, at around $0.85$, but were still above the required threshold of $0.75$.

The Improved UNet was marginally ahead on background, body, bone, and prostate, and marginally behind on bladder. The improved model had a not insignificant increase in Dice for the rectum, potentially due to reduced overfitting from fewer parameters. These differences were small enough to be within run-to-run variation. Both models used the same batch size ($16$), optimiser, learning rate, and number of epochs, with no learning-rate scheduler, isolating the architecture. The Improved UNet’s much lower parameter count (about $10\times$ fewer) and memory use come from its smaller base width of 16 channels, against 64 for the baseline. A scheduler, a gentler learning rate, or longer training could allow the Improved UNet to show a greater benefit.

---

### 6.3 Failure Cases

Four samples with the worst Dice scores and the largest areas of incorrect prostate identification were selected. `slice_case_042_week_0_slice_28.png` shows the results from a central slice, for reference. Slices with no ground-truth prostate had no Dice score, so the false-positive cases were ranked by predicted area instead. `false_positive_case_040_week_4_slice_15.png` and `false_positive_case_040_week_4_slice_35.png` are distinct false positive cases, where the model wrongly predicted prostate location when it was not there.

The former is a small artifact sandwiched by bone and bladder, while the latter is a comparatively larger, oval-shaped prediction between the bladder and rectum. The failures both share similar traits, having a similar intensity to and low contrast with surrounding tissue, and being located next to a significantly brighter bladder, which potentially contributes to this failure.

On the other hand, `low_dice_case_040_week_1_slice_39.png` and `low_dice_case_042_week_0_slice_12.png` are false negatives, where the model predicts that certain pixels do not include the prostate, even though they do. Both slices’ prostates are fairly dark and right next to a bright bladder, which likely stands out to the model and could be a reason why the failure occurred. `low_dice_case_040_week_1_slice_39.png` is a small sliver closer to the rectum, which shares a similar intensity, potentially leading to mistaken grouping. `low_dice_case_042_week_0_slice_12.png` is a much larger, circular chunk, and has a similar shape and intensity to other unidentified artifacts located to the right of the bladder, which appears to be the most probable reason for the failure.

Looking at both false positive and false negative cases, a trend is apparent: even between different patients, slices with poor performance happen between slices 10–20 and 30–40. Inspecting the dataset more closely, it is understood that these slices are located where the prostate starts and ends, with slices $< 10$ and $> 40$ generally not having any prostate pixels. These results seem to align with the context from the open research question, where even though overall Dice score is acceptable, the model over- and under-predicts edge slices.

Both probable causes for false positive and negative cases relate to misidentification of when the prostate gland begins and ends along the slice axis. This information is absent from a 2D slice, which is also consistent with the research question’s hypothesis that slice-to-slice context matters.

---

## 7.0 Recommendations and Conclusion

The overall recommendation is to use the Improved 2D UNet, but potentially with some changes. While the improved model performed similarly to the baseline, it still appears to be marginally better. Furthermore, given the significantly faster training time and lower computing power usage, the Improved UNet could be retrained on an even larger dataset with modified parameters, such as a gentler learning rate, scheduler, or more epochs, which would allow the benefits of the improved model to become more apparent.

2D UNet inherently struggles with identifying the borders of the prostate but can achieve acceptable results in more central slices. Therefore, the use case of the model has to be considered. The model can be used and is of benefit in cases where the prostate is large, clearly visible, and not immediately bordering other organs. Other cases, such as at the apex and base of the prostate, or where the cross-sectional area of the prostate is minimal, have the model performing poorer, are less suitable, and it is much less advisable to rely on the model in these circumstances.

3D models, like CAN or UNet, can potentially improve identification using additional context available in the higher dimension, and should be considered for future research into deep learning for medical imaging.

---

## References

1. O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional networks for biomedical image segmentation,” in *Proc. MICCAI*, 2015, pp. 234–241.
2. F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein, “Brain tumor segmentation and radiomics survival prediction: Contribution to the BRATS 2017 challenge,” in *Brainlesion: Glioma, Multiple Sclerosis, Stroke and TBI Models*, 2018, pp. 287–297.
3. krn0209. (n.d.). unet2d [Source code]. Kaggle. https://www.kaggle.com/code/krn0209/unet2d/notebook

---

## Artificial Intelligence Usage Disclosure

Gemini was used for concept lookup and explanation. Claude was used for debugging code, looking up syntax, creating code frameworks for files, generating data plotting helpers, helping write comments and documentation, and generating specific blocks of code with unknown parameter sizes, such as correctly importing the dataset and ensuring slice dimensions were suitable.

All AI-generated code was first inspected, line by line, and questionable blocks were queried and appropriately adjusted before being run with a small sample size. Results were then re-verified against the AI, with necessary discerning questions being asked based on course knowledge, before finally being executed on primary code.