# 2D Improved U-Net for Multi-Organ Pelvic and Prostate MRI Segmentation

**Student Name:** Yizhuo Wu  
**Student ID:** 49732823  
**Course:** COMP3710 - Pattern Recognition (2026)  
**Track:** 2.2 Segmentation — Anatomical Contouring and Radiotherapy Planning  
**Target Model:** 2D Improved U-Net (Normal Difficulty)  
**Baseline Model:** Standard 2D U-Net  
**Execution Environment:** UQ Rangpur HPC Cluster (NVIDIA A100 GPU)

---

## 1. Project Description & Working Principles

This project studies automated multi-organ pelvic segmentation and prostate anatomical contouring from MRI scans (`HipMRI` dataset) to investigate technical challenges in radiotherapy planning workflows. In this engineering study, dataset segmentation masks are treated as anatomical label benchmarks rather than clinically validated Clinical Target Volumes (CTV), and no clinical outcome claims (such as recurrence prevention or toxicity reduction) are made.

This project implements a 2D adaptation inspired by the architecture of Isensee et al. (2018) and compares it with a 2D U-Net baseline based on Ronneberger et al. (2015). While Isensee et al. (2018) proposed a 3D network trained with multi-class Dice loss, our implementation adapts the core design to 2D axial slices with the following specifications:
- **Residual Context Modules:** Two $3 \times 3$ convolutions with Instance Normalization, LeakyReLU ($\alpha = 0.01$), spatial Dropout ($0.3$), and residual identity skip connections using element-wise addition.
- **Concatenation Skip Connections & Localization Units:** Encoder and upsampled decoder features are concatenated along the channel dimension (`torch.cat`), then refined via $3 \times 3$ and $1 \times 1$ convolutions.
- **Deep Supervision:** Auxiliary segmentation heads at multiple decoder scales are upsampled and combined via element-wise addition into the final output logits.
- **Compound Loss:** Training uses an equally weighted combination of class-weighted cross-entropy and soft Dice losses:
  $$L = 0.5 L_{\mathrm{weighted\ CE}} + 0.5 L_{\mathrm{weighted\ Dice}}$$
  with fixed class weights of $[0.2, 0.5, 1.0, 1.5, 2.0, 3.0]$ across the 6 classes to mitigate foreground class imbalance.

```text
Input (1x256x256) -> [Conv 16] --------------------------------------> [Concat] -> [SegOut 1x]
                         |                                                ^             |
                  [Context 32] ------------------------> [Concat] -> [Loc]/       [Upsample 2x]
                         |                                  ^         |                 |
                  [Context 64] ----------> [Concat] -> [Loc]/    [SegOut 2x] --------> (+) -> Softmax (6 Classes)
                         |                    ^         |                               ^
                  [Context 128] -> [Bottleneck]    [SegOut 4x] -> [Upsample 4x] --------/
```

---

## 2. Data & Setup

- **Dataset Source:** Pre-sliced HipMRI study cohort provided via the COMP3710 course cluster (`/home/groups/comp3710/HipMRI_Study_open/keras_slices_data`), adapted using loading conventions from the COMP3710 Project Specification Appendix B.
- **Data Partitioning & Patient Isolation Note:**
  * The primary execution path in `dataset.py` loads pre-partitioned directories (`keras_slices_train`, `keras_slices_validate`, `keras_slices_test`), comprising 11,460 training slices, 660 validation slices, and 540 held-out test slices (reported as 30 / 4 / 4 patients in course documentation).
  * *Code Audit Note:* When reading directly from these pre-split directories, `dataset.py` returns the file lists directly and does not execute programmatic patient-ID disjointness assertions. Programmatic set-disjoint assertions (`assert len(set(train) & set(test)) == 0`) and the 70%/15%/15% automatic patient split (which yields 26 / 5 / 7 patients across 38 cases) reside only in the fallback branch for unsplit directories. Verifying zero patient leakage on the pre-split directory path requires auditing extracted filename prefixes across the three folders to confirm $S_{\mathrm{train}} \cap S_{\mathrm{val}} = \emptyset$, $S_{\mathrm{train}} \cap S_{\mathrm{test}} = \emptyset$, and $S_{\mathrm{val}} \cap S_{\mathrm{test}} = \emptyset$.
- **Preprocessing:** Per-slice Z-score normalization ($I' = \frac{I - \mu}{\sigma + 10^{-8}}$) and 6-class label mapping (0: Background, 1: Body, 2: Bone, 3: Bladder, 4: Rectum, 5: Prostate).
- **Framework & Environment:** `torch==2.5.1+cu118`, `torchvision`, `nibabel==5.2.1`, `numpy==1.26.4`, `matplotlib==3.9.2` (developed with Python 3.11; Slurm job script configured with fallback `module load python/3.10`).

### Reproduction Commands
```bash
# 1. Train 2D Improved U-Net (40 Epochs on A100)
python train.py --model improved --epochs 40 --batch_size 16 \
    --data_dir /home/groups/comp3710/HipMRI_Study_open/keras_slices_data --output_dir ./results

# 2. Evaluate on held-out test set and generate failure case autopsies
python predict.py --model improved \
    --data_dir /home/groups/comp3710/HipMRI_Study_open/keras_slices_data \
    --weights_path ./results/best_improved_unet.pth --output_dir ./predictions
```

---

## 3. Feasibility Report & Plan Evolution

### 3.1 User Need, Scope & Acceptance Criteria
- **Intended User & Context:** Radiation oncologists and medical physicists exploring automated pelvic organ contouring from MRI scans.
- **Open Dilemma Addressed:** Investigating the *Spatial Context vs. Boundary Utility Dilemma*—evaluating how a 2D slice-wise segmentation architecture performs across large pelvic organs versus small cross-sectional prostate slices.
- **Acceptance Criteria & Verification Outcome:**
  1. **Primary Segmentation Accuracy (Not Met on Test Split):** Target held-out test prostate $\mathrm{DSC} \ge 0.75$ for 2D slices. The trained 2D Improved U-Net achieved a validation prostate DSC of **0.7955**, but a held-out test prostate DSC of **0.7432**, falling short of the preset test threshold by **0.0068**. Validation performance does not substitute for held-out test acceptance.
  2. **Boundary & Overlap Metric:** Target test $\mathrm{IoU} \ge 0.65$ (originally $\ge 0.60$ in the standalone midpoint review); reported test prostate IoU is **0.6857** under batch-weighted evaluation.
  3. **Hardware & VRAM Constraints:** Peak training memory footprint $\le 12.0\text{ GB}$ VRAM on UQ Rangpur nodes (achieved **2.66 GB**).
  4. **Evaluation Throughput:** Average evaluation time $< 15\text{ ms / slice}$ on a single GPU (achieved **5.58 ms / slice** end-to-end evaluation time).

### 3.2 Scope Evolution from Standalone Midpoint Feasibility Report
During the project midpoint review, the standalone `feasibility report.md` proposed an initial 3D volumetric target with a 2D fallback plan. Following our staged risk-mitigation strategy, we executed the 2D Improved U-Net fallback on `keras_slices_data`. The table below documents the exact changes between the initial standalone proposal and the final executed 2D setup:

| Specification Item | Standalone Midpoint Feasibility Report | Final Executed Pipeline (README) | Reason for Adjustment |
| :--- | :--- | :--- | :--- |
| **Target Architecture** | 3D Improved U-Net (Hard) with 2D fallback | 2D Improved U-Net (Normal) vs. 2D U-Net Baseline | Executed planned 2D fallback to ensure a complete, verified pipeline within project schedule. |
| **Prostate IoU Threshold** | $\ge 0.60$ | $\ge 0.65$ | Tightened overlap threshold when transitioning from 3D volume target to 2D slice evaluation. |
| **Runtime Target** | $< 15\text{ seconds}$ per 3D patient volume | $< 15\text{ ms}$ per 2D axial slice | Adapted metric unit from volumetric scan latency to slice-wise evaluation throughput. |
| **Partition Plan** | Approx. 26 / 6 / 6 (unsplit auto-partition) | 30 / 4 / 4 (pre-partitioned `keras_slices_data`) | Adopted the pre-partitioned directory structure provided on the course HPC storage. |

---

## 4. Quantitative Results & Benchmarking

### 4.1 Model Comparison & Resource Profile (Rangpur A100 GPU)

> **Measurement & Aggregation Protocol:**
> 1. **Batch-Weighted Metric Aggregation:** In `evaluate()`, intersection and union counts are summed across all slices within each batch to compute a batch-level DSC and IoU, which are then averaged across batches weighted by batch slice count. This differs from patient-level volumetric aggregation, unweighted per-slice averaging, or global dataset-wide pixel summation. Because the nonlinear relation $\mathrm{IoU} = \frac{\mathrm{DSC}}{2 - \mathrm{DSC}}$ holds for a single confusion matrix (where $\mathrm{DSC}=0.7432$ corresponds to $\mathrm{IoU} \approx 0.5913$) but does not hold linearly after averaging across batches with varying foreground sizes, both metrics are reported directly as output by the batch-aggregated evaluation loop.
> 2. **Channel Width Configuration:** `BaselineUNet2D` uses 32 base channels (7.76M parameters), whereas `ImprovedUNet2D` uses 16 base channels (3.19M parameters). Parameter, VRAM, and speed differences therefore reflect the combined effect of channel width and architectural modules rather than residual blocks or deep supervision alone.
> 3. **Timing Definition:** The reported `ms / slice` values represent **average end-to-end evaluation pipeline time per slice** (including DataLoader fetching, host-to-device transfer, forward pass, and metric calculation without isolated CUDA synchronization warm-up), rather than pure forward-pass inference latency.

| Model Architecture | Base Channels | Parameters | Peak GPU VRAM | Avg Eval Pipeline Time | Best Val Prostate DSC | Test Prostate DSC | Test Prostate IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard 2D U-Net (Baseline)** | 32 | 7.76 M | 3.82 GB | 8.42 ms / slice | 0.7180 | 0.6920 | 0.5840 |
| **2D Improved U-Net (Ours)** | 16 | **3.19 M** | **2.66 GB** | **5.58 ms / slice** | **0.7955** | **0.7432** | **0.6857** |

### 4.2 Per-Class Performance on Held-Out Test Split (540 Slices)

| Anatomical Class | Sørensen-Dice (DSC) | Intersection-over-Union (IoU) |
| :--- | :---: | :---: |
| Background (0) | 0.9956 | 0.9913 |
| Body Outline (1) | 0.9833 | 0.9672 |
| Pelvic Bone (2) | 0.9255 | 0.8623 |
| Bladder (3) | 0.9523 | 0.9102 |
| Rectum (4) | 0.8513 | 0.7727 |
| **Prostate (5)** | **0.7432** | **0.6857** |

![Learning Curves](results/improved_learning_curves.png)

---

## 5. Investigation of the Open Dilemma: Failure Case Autopsy

To examine local failure modes behind the aggregate test prostate DSC (`0.7432`), we sorted the 540 held-out test slices by per-slice prostate DSC and inspected five lowest-scoring slices (`DSC = 0.0000`). Note that `Slice` numbers below denote **test DataLoader traversal indices** rather than verified anatomical $z$-coordinates, and a DSC of `0.0000` indicates zero true-positive overlap ($\mathrm{TP} = 0$), which can occur either from complete under-segmentation ($\mathrm{FN} > 0, \mathrm{FP} = 0$) or from non-overlapping false-positive predictions:

| Case | Test Loader Index | Visual Observation in Diagnostic Plot | Prostate DSC | Observed Error Pattern & Possible Contributing Factors |
| :--- | :---: | :--- | :---: | :--- |
| **#1** | `Slice 512` | Small ground-truth prostate mask adjacent to rectum | `0.0000` | Zero overlap ($\mathrm{TP}=0$, false negative in target region). Small target cross-section and low local contrast near the rectal boundary may hinder 2D feature discrimination. |
| **#2** | `Slice 484` | Small ground-truth prostate mask near prominent bladder | `0.0000` | Zero overlap ($\mathrm{TP}=0$). High contrast of the adjacent bladder and faint prostate boundary may contribute to foreground miss. |
| **#3** | `Slice 250` | Compact ground-truth target region | `0.0000` | Zero overlap ($\mathrm{TP}=0$). Limited 2D spatial extent provides sparse texture cues for positive classification. |
| **#4** | `Slice 513` | Small ground-truth target region | `0.0000` | Zero overlap ($\mathrm{TP}=0$). Consecutive index in test traversal following `Slice 512`; verifying whether these represent adjacent anatomical slices requires mapping back to source NIfTI filenames. |
| **#5** | `Slice 188` | Indistinct border between prostate mask and surrounding tissue | `0.0000` | Zero overlap ($\mathrm{TP}=0$). Low soft-tissue contrast along the target perimeter may prevent clear boundary separation. |

![Autopsy Case 1](predictions/autopsy_case_1_slice512_dice_0.000.png)
![Autopsy Case 2](predictions/autopsy_case_2_slice484_dice_0.000.png)
![Autopsy Case 3](predictions/autopsy_case_3_slice250_dice_0.000.png)
![Autopsy Case 4](predictions/autopsy_case_4_slice513_dice_0.000.png)
![Autopsy Case 5](predictions/autopsy_case_5_slice188_dice_0.000.png)

### Failure Mode Discussion & Hypotheses
The model processes slices independently and therefore does not explicitly use through-plane context. This may contribute to segmentation errors where the target occupies a small cross-sectional area. However, the selected low-Dice examples alone do not establish the cause of failure or exclude annotation uncertainty. Confirming this explanation requires anatomical localisation of the cases (recording exact patient IDs, sequence IDs, ground-truth/predicted pixel counts, and TP/FP/FN breakdowns) and a controlled comparison with a model that incorporates neighbouring slices.

---

## 6. Engineering Recommendations for Future Validation

1. **Evaluate Semi-Automated Drafting Potential in Follow-Up Studies:** While the 2D Improved U-Net shows promising test overlap on large pelvic structures (Bladder DSC `0.9523`, Pelvic Bone `0.9255`, Rectum `0.8513`) with modest memory usage (`2.66 GB` VRAM), it did not meet the preset $\ge 0.75$ test DSC criterion on the prostate class (`0.7432`). Further validation is required before considering clinical workflow integration.
2. **Investigate Candidate Area Thresholds for Quality-Control Flagging:** Because zero-DSC failures frequently coincide with small ground-truth target cross-sections in our visual inspection, future work should evaluate a candidate pixel-area threshold on the validation split to flag uncertain slices for manual review. Converting any pixel-count rule into physical units (such as $\mathrm{mm}^2$) would first require preserving NIfTI voxel-spacing metadata in the data loader.
3. **Conduct Controlled 2D vs. 2.5D/3D Ablation Studies:** Rather than assuming 3D networks are strictly necessary, a controlled experiment comparing matched-capacity 2D, 2.5D multi-slice, and 3D models under identical patient splits is recommended to test whether through-plane context reduces small-target dropouts.

---

## 7. Artificial Intelligence Usage Disclosure

- **AI Tools Used:** OpenAI ChatGPT and Google Gemini.
- **Assisted Tasks:** Scaffolding PyTorch network modules in `modules.py`, data loader structure in `dataset.py`, training/evaluation loops in `train.py`, 4-panel matplotlib diagnostic plots in `predict.py`，reinforce `README.md` final report and declaration.
- **Verification & Auditing Caveats:** Code scripts and markdown documentation were reviewed by the author. Note that while `dataset.py` includes a patient-level disjointness assertion in its automatic splitting branch, the primary execution path on Rangpur reads pre-partitioned directories (`keras_slices_train/validate/test`) without triggering that assertion.

---

## 8. References

1. **Ronneberger, O., Fischer, P., & Brox, T. (2015).** U-Net: Convolutional Networks for Biomedical Image Segmentation. In *Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015* (pp. 234–241). Springer International Publishing.
2. **Isensee, F., Kickingereder, P., Wick, W., Bendszus, M., & Maier-Hein, K. H. (2018).** Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge. In *Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries* (pp. 287–297). Springer International Publishing.
3. **COMP3710 Teaching Team (2026).** *COMP3710 Pattern Recognition Final Assessment Specification (Version 2.0), Appendix A (HipMRI Study Dataset) & Appendix B (Data Loading Reference).* School of Electrical Engineering and Computer Science, The University of Queensland.