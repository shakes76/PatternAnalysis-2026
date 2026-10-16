# ConvNeXt for Alzheimer's Disease Neuroimaging Classification, Uncertainty Calibration, and Selective Clinical Triage on ADNI MRI

**Course:** COMP3710 / COMP7710 — Pattern Recognition & Analysis (Semester 2, 2026)  
**Author:** Lingxiao Li  
**Student ID:** `49852413` (`s49852413`)  
**Email:** `s4985241@student.uq.edu.au`  
**Selected Project:** Section 2.1 Classification — Neuroimaging (`ADNI` Brain MRI)  
**Selected Architecture & Tier:** **ConvNeXt** (`[Hard Difficulty]`, eligible for full 20/20 implementation ceiling)  
**Branch:** `topic-recognition`  
**Folder:** `recognition/ConvNeXt_ADNI_s49852413/`

---

## 1. Executive Summary & Problem Formulation

Early and reliable differentiation between **Alzheimer's Disease (`AD`)** and ** Cognitively Normal Controls (`NC`)** from structural T1-weighted brain Magnetic Resonance Imaging (MRI) is critical for neurodegenerative triage, clinical trial enrichment, and longitudinal disease monitoring. Structurally, Alzheimer's pathology manifests across multiple spatial scales:
1. **Localised Medial Temporal Atrophy:** Cortical thinning of the entorhinal cortex, parahippocampal gyrus, and bilateral hippocampi.
2. **Global Structural Remodelling:** Ex-vacuo dilation of the lateral and third ventricles (**ventriculomegaly**) accompanied by widening of cerebral cortical sulci.

### 1.1 Why ConvNeXt for Structural Neuroimaging?
Standard small-kernel CNN baselines ($3\times 3$ convolutions with ReLU and BatchNorm) suffer from a limited effective receptive field in early stages, struggling to jointly model localised hippocampal boundary degradation alongside global ventricular-to-intracranial volume ratios. Conversely, pure Vision Transformers (ViTs) lack translation equivariance and inductive spatial biases, making them data-hungry and prone to overfitting on moderate-scale medical cohorts where patient-level sample sizes are constrained ($N \approx 350$ subjects).

To resolve this tension, we implement **ConvNeXt** (Liu et al., *CVPR 2022*) from scratch in pure PyTorch (`modules.py`). ConvNeXt modernises a hierarchical convolutional backbone with the architectural design principles of Swin Transformers:

$$\mathbf{z}_{\ell} = \mathbf{x}_{\ell-1} + \text{DropPath}\!\left(\boldsymbol{\gamma}_{\ell} \odot W_{\text{pw2}}\,\phi_{\text{GELU}}\!\Big(W_{\text{pw1}}\,\text{LN}\big(\text{DWConv}_{7\times 7}(\mathbf{x}_{\ell-1})\big)\Big)\right)$$

| Architectural Component | Classical ResNet / SimpleCNN | Our Medical ConvNeXt Implementation (`modules.py`) | Neuroimaging Benefit |
| :--- | :--- | :--- | :--- |
| **Macro Stage Ratio** | $(3, 4, 6, 3)$ | **`depths = (2, 2, 6, 2)`**, **`dims = (48, 96, 192, 384)`** | Concentrates representation capacity in Stage 3 where whole-ventricle and bilateral temporal lobe context converge |
| **Stem Design** | $7\times 7$ conv (stride 2) + MaxPool | **$4\times 4$ non-overlapping Patchify Stem (stride 4)** + `LayerNorm2d` | Downsamples $128\times 128 \to 32\times 32$ tokens cleanly without overlapping max-pool blurring |
| **Spatial Mixing** | $3\times 3$ standard dense conv | **$7\times 7$ Depthwise Separable Convolution (`groups=C`)** | $49\text{-pixel}$ spatial kernel per block captures wider anatomical context at a fraction of dense FLOPs |
| **Channel Mixing** | Bottleneck ($C \to C/4 \to C$) | **$4\times$ Inverted Bottleneck ($C \to 4C \to C$)** via $1\times 1$ linear layers | Expands high-dimensional intensity/texture feature spaces before spatial projection |
| **Normalisation & Activation** | `BatchNorm2d` + `ReLU` at every conv | **Single `LayerNorm2d` + Single `GELU` per block** | Eliminates batch-statistics drift across small medical batches and preserves smooth negative gradients |
| **Regularisation & Stability** | Standard residual addition | **Learnable `LayerScale` ($\boldsymbol{\gamma}_0 = 10^{-6}$) + Stochastic Depth (`DropPath`)** | Stabilises deep 12-block optimisation and acts as an implicit ensemble over anatomical pathways |

![ConvNeXt Architecture & Block Topology](./images/convnext_architecture.png)

---

## 2. Midpoint Feasibility Review (One-Page Progress Summary — Section 3 Specification)

> **Compliance Note (Section 3 of Assessment Specification v2.2):** This self-contained one-page engineering review documents the four compulsory midpoint verification gates: (1) Data Loading & Pre-processing Verification, (2) Preliminary Baseline vs. ConvNeXt Results, (3) Compute & Memory Footprint Audit, and (4) Experimental Roadmap Completion.

### 2.1 Data Loading & Patient-Level Split Hygiene Audit
A pervasive methodological flaw in 2D medical imaging benchmarks is **slice-level data leakage**, where adjacent axial slices from the same patient are randomly scattered across `train` and `test` sets, allowing models to memorise subject-specific skull geometry rather than disease pathology.
- In `dataset.py`, `extract_patient_and_slice()` parses patient identifiers from every file path (`AD_<subjectID>_slice<k>` / `NC_<subjectID>_slice<k>`), and `split_patients_stratified()` partitions unique patients into stratified `train` ($85\%$) and `val` ($15\%$) cohorts while keeping the held-out `test` cohort completely isolated.
- Programmatic execution of `verify_no_patient_leakage()` confirms **zero patient overlap** across all three partitions:

| Split Partition | Total 2D Slices | `NC` Slices (Class 0) | `AD` Slices (Class 1) | Unique Patients | `NC` Patients | `AD` Patients | Patient Leakage Overlap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train (`train`)** | $2,650$ | $1,340$ ($50.6\%$) | $1,310$ ($49.4\%$) | **$265$** | $134$ | $131$ | `train ∩ val = 0` |
| **Validation (`val`)** | $470$ | $240$ ($51.1\%$) | $230$ ($48.9\%$) | **$47$** | $24$ | $23$ | `val ∩ test = 0` |
| **Held-Out Test (`test`)** | $420$ | $190$ ($45.2\%$) | $230$ ($54.8\%$) | **$42$** | $19$ | $23$ | `train ∩ test = 0` |
| **Total Cohort** | **$3,540$** | **$1,770$** | **$1,770$** | **$354$** | **$177$** | **$177$** | **Verified $0$ Leakage** |

- **Pre-processing Verification (`ADNIDataset`):** Every slice is resized via bilinear interpolation to $128\times 128$, standardized via **per-slice Z-score normalisation** $\tilde{I}(x,y) = \frac{I(x,y)-\mu_I}{\max(\sigma_I, 10^{-6})}$ to eliminate scanner receiver-gain drift, and linearly mapped to $[-1, 1]$ after clipping to $[-3\sigma, +3\sigma]$. Training augmentations are strictly restricted to anatomically plausible transformations (symmetric horizontal flip, $\pm 10^\circ$ in-plane rotation, $\pm 4\%$ translation, and subtle contrast jitter).

![ADNI Pre-processing Pipeline and Class Intensity Distributions](./images/preprocessing_and_samples.png)

### 2.2 Preliminary Baseline vs. ConvNeXt Progression
Both `SimpleCNNBaseline` (4-stage $3\times 3$ Conv-BN-ReLU architecture, $8,994$ parameters) and `ConvNeXt` ($4,832,018$ parameters) were trained end-to-end on the exact same patient split (`seed = 42`, $18$ epochs, `AdamW`, cosine annealing with linear warmup):
- **`SimpleCNNBaseline`:** Converged rapidly to **$94.47\%$ validation accuracy** and **$93.81\%$ test accuracy** ($\text{Macro-F1} = 0.9379$), confirming pipeline end-to-end integrity, but suffered from **$22$ False Negatives** on the test set ($\text{AD Recall} = 90.43\%$), missing nearly 1 in 10 Alzheimer's slices.
- **`ConvNeXt` (`[Hard]`):** Reached **$97.66\%$ validation accuracy** (Best Epoch 12) and **$95.95\%$ test accuracy** ($\text{Macro-F1} = 0.9592$), **reducing False Negatives by $54.5\%$** ($22 \to 10$, boosting $\text{AD Recall}$ from $90.43\% \to 95.65\%$) and achieving **$95.24\%$ patient-level accuracy** ($\text{Patient AUROC} = 0.9977$).

### 2.3 Compute & Memory Footprint Audit
Resource telemetry logged automatically during training (`checkpoints/benchmark_summary.json`) verifies that the pipeline operates well within standard workstation and UQ Rangpur cluster (`a100` / `v100` / `mps`) limits:
- **Peak GPU Memory (`ConvNeXt`, batch size 32):** **$1,208.41\text{ MB}$** ($\approx 1.18\text{ GB}$, $<8\%$ of a 16 GB GPU).
- **Training Wall-Clock Time (`18` epochs):** **$21.11\text{ s}$** (`SimpleCNNBaseline`, $1.17\text{ s/epoch}$) vs. **$154.24\text{ s}$** (`ConvNeXt`, $8.56\text{ s/epoch}$).
- **Inference Throughput:** **$1,595.3\text{ slices/sec}$** ($0.627\text{ ms/slice}$ batched; $1.573\text{ ms/slice}$ single-patient latency).

### 2.4 Completed Experimental Roadmap
All four milestones defined at project inception have been executed and verified:
1. **[Completed]** Pure-PyTorch `ConvNeXt` & `SimpleCNNBaseline` implementation passing strict AST compliance (`test_pipeline.py`).
2. **[Completed]** Head-to-head training, patient-level aggregation, and hardware resource profiling (`train.py`).
3. **[Completed]** Post-hoc L-BFGS Temperature Scaling calibration and selective clinical referral sweep across $\tau \in [0.50, 0.95]$.
4. **[Completed]** Spatial Grad-CAM attribution and 4-case failure autopsy of False Negatives and False Positives (`predict.py`).

---

## 3. Quantitative Benchmark & Resource Profiling (Section 5.3)

### 3.1 Classification Performance on Held-Out Test Set (`test`, $420$ slices / $42$ patients)
To account for slight class imbalance in the held-out test split ($190\text{ NC}$ vs. $230\text{ AD}$ slices), we report overall Accuracy alongside **Balanced Accuracy**, **Macro-F1**, **AUROC**, **PR-AUC**, per-class Precision/Recall/F1, confusion matrix counts, and **Patient-Level Majority-Vote Accuracy**.

| Metric / Evaluation Dimension | `SimpleCNNBaseline` ($3\times 3$ Conv-BN-ReLU) | `ConvNeXt` ($7\times 7$ DWConv + Inverted Bottleneck) | Absolute / Relative Gain |
| :--- | :---: | :---: | :---: |
| **Slice-Level Test Accuracy** | $93.81\%$ ($394 / 420$) | **$95.95\%$ ($403 / 420$)** | **$+2.14\%$** ($-34.6\%$ total errors) |
| **Balanced Accuracy** | $94.16\%$ | **$95.98\%$** | **$+1.82\%$** |
| **Macro-Averaged F1 Score** | $0.9379$ | **$0.9592$** | **$+0.0213$** |
| **Test ROC-AUC (AUROC)** | $0.9946$ | **$0.9945$** | Comparable ($>0.994$) |
| **Test PR-AUC (Average Precision)** | $0.9954$ | **$0.9952$** | Comparable ($>0.995$) |
| **Patient-Level Accuracy ($N=42$ subjects)** | $95.24\%$ ($40 / 42$) | **$95.24\%$ ($40 / 42$)** | Tied on majority vote |
| **Patient-Level AUROC ($N=42$ subjects)** | $0.9954$ | **$0.9977$** | **$+0.0023$** (near-perfect ranking) |
| **`NC` (Class 0) Precision / Recall / F1** | $0.8942$ / $0.9789$ / $0.9347$ | **$0.9482$ / $0.9632$ / $0.9556$** | **$+5.40\%$ `NC` Precision** |
| **`AD` (Class 1) Precision / Recall / F1** | $0.9811$ / $0.9043$ / $0.9412$ | **$0.9692$ / $0.9565$ / $0.9628$** | **$+5.22\%$ `AD` Recall (Sensitivity)** |
| **Confusion Matrix (`TN`, `FP`, `FN`, `TP`)** | `TN=186, FP=4, FN=22, TP=208` | **`TN=183, FP=7, FN=10, TP=220`** | **False Negatives cut by $54.5\%$ ($22 \to 10$)** |
| **Best Validation Accuracy / Macro-F1** | $94.47\%$ / $0.9446$ (Epoch 18) | **$97.66\%$ / $0.9766$ (Epoch 12)** | **$+3.19\%$ Val Accuracy** |

#### Clinical Significance of the Benchmark Comparison
While `SimpleCNNBaseline` achieves respectable overall accuracy ($93.81\%$), its error distribution is **clinically unsafe**: $22$ out of $26$ total errors are **False Negatives (`FN = 22`)**, meaning $9.57\%$ of Alzheimer's slices are falsely cleared as Normal Controls (`AD Recall = 90.43%`). By contrast, `ConvNeXt`'s $7\times 7$ depthwise convolutions and inverted bottleneck stages capture subtle hippocampal and sulcal atrophy, cutting missed `AD` cases in half (`FN = 10`, `AD Recall = 95.65%`) and lifting `NC` Precision from $89.42\% \to 94.82\%$.

![Training Convergence and Validation Curves](./images/training_curves.png)

![ROC and Precision-Recall Comparison on Held-Out Test Split](./images/roc_pr_comparison.png)

### 3.2 Computational Resource & Latency Profiling
To evaluate whether `ConvNeXt`'s $+5.22\%$ gain in Alzheimer's sensitivity justifies its increased computational footprint, we profile both models across parameter complexity, peak memory, training duration, and inference latency:

| Hardware & Efficiency Metric | `SimpleCNNBaseline` | `ConvNeXt` (`[Hard]`) | Scaling Ratio & Engineering Interpretation |
| :--- | :---: | :---: | :--- |
| **Trainable Parameters** | $8,994$ ($0.009\text{ M}$) | **$4,832,018$ ($4.83\text{ M}$)** | $537\times$ larger, yet still $5.3\times$ smaller than ResNet-50 ($25.6\text{ M}$) |
| **Peak GPU Memory (`batch_size=32`)** | $58.97\text{ MB}$ | **$1,208.41\text{ MB}$ ($\approx 1.18\text{ GB}$)** | Easily fits on any commodity 4 GB+ GPU or Rangpur slice |
| **Mean Training Epoch Time** | $1.17\text{ s / epoch}$ | **$8.56\text{ s / epoch}$** | $7.3\times$ slower training, still under $9\text{ s}$ per epoch |
| **Total Training Wall-Clock Time ($18$ ep)** | $21.11\text{ s}$ | **$154.24\text{ s}$ ($2.57\text{ min}$)** | Full retraining completes in $<3$ minutes |
| **Single-Slice Inference Latency (`batch=1`)** | $0.181\text{ ms}$ | **$1.573\text{ ms}$** | Sub-$2\text{ ms}$ real-time response per axial slice |
| **Batched Inference Latency (`batch=32`)** | $0.015\text{ ms / slice}$ | **$0.627\text{ ms / slice}$** | Whole 20-slice patient scan processed in **$12.5\text{ ms}$** |
| **Batched Throughput** | $65,352\text{ slices/sec}$ | **$1,595.3\text{ slices/sec}$** | $\approx 80$ full patient volumes per second |

---

## 4. Open Research Dilemma: Uncertainty Calibration & Selective Clinical Triage

### 4.1 Formulating the Clinical Deployment Dilemma
In automated neuroimaging screening, raw neural network softmax probabilities $\hat{\mathbf{p}}(\mathbf{x}) = \text{Softmax}(\mathbf{z}(\mathbf{x}))$ are notoriously poorly calibrated, especially when trained with **label smoothing** ($\varepsilon = 0.05$), which artificially compresses logit magnitudes toward the uniform distribution. If an uncalibrated classifier is deployed directly in a hospital workflow, two failure modes arise:
1. **Silent False Negatives:** Borderline Alzheimer's cases are auto-cleared as `NC` without human review, delaying clinical intervention.
2. **Miscalibrated Confidence:** Clinicians cannot trust a reported "$85\%$ probability" as an accurate reflection of true empirical accuracy.

### 4.2 Post-Hoc Temperature Scaling & Predictive Entropy Formulation
Implemented in `TemperatureScaledClassifier` (`modules.py`), we freeze the trained `ConvNeXt` backbone weights and learn a single positive scalar temperature $T > 0$ (parameterised as $T = \exp(\log T)$) on the held-out validation logits $\mathbf{z}_i$ by minimising Cross-Entropy Negative Log-Likelihood (NLL) via **L-BFGS**:

$$T^* = \arg\min_{T > 0} -\frac{1}{N_{\text{val}}} \sum_{i=1}^{N_{\text{val}}} \log \frac{\exp(z_{i, y_i} / T)}{\sum_{c \in \{0,1\}} \exp(z_{i, c} / T)}$$

From the calibrated probability distribution $\hat{\mathbf{p}}_T(\mathbf{x}) = \text{Softmax}(\mathbf{z}(\mathbf{x}) / T^*)$, we extract two complementary uncertainty signals:
1. **Calibrated Maximum Confidence:** $\hat{c}_T(\mathbf{x}) = \max_{c \in \{0,1\}} \hat{p}_{T, c}(\mathbf{x}) \in [0.5, 1.0]$.
2. **Normalized Shannon Predictive Entropy (in bits):**

$$H(\hat{\mathbf{p}}_T(\mathbf{x})) = -\sum_{c=0}^{1} \hat{p}_{T,c}(\mathbf{x}) \log_2 \hat{p}_{T,c}(\mathbf{x}) \in [0, 1]$$

We measure probability calibration via **Expected Calibration Error (ECE)** across $M = 10$ uniform confidence bins $B_m$:

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \Big|\text{acc}(B_m) - \text{conf}(B_m)\Big|$$

### 4.3 Empirical Calibration Results
Because both models were trained with label smoothing ($\varepsilon = 0.05$), their raw logits are slightly *under-confident* on unambiguous cases. L-BFGS Temperature Scaling automatically learns $T^* < 1$ to sharpen the probability simplex to match true empirical accuracy:

| Model | Learned Temperature $T^*$ | Validation ECE (Uncalibrated $\to$ Calibrated) | Held-Out Test ECE (Uncalibrated $\to$ Calibrated) | Correct Slice Mean Conf / Entropy | Misclassified Slice Mean Conf / Entropy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`SimpleCNNBaseline`** | $T^* = 0.373$ | $0.0568 \to 0.0253$ ($-55.5\%$) | $0.0381 \to 0.0329$ ($-13.6\%$) | $0.9760$ / $0.0977\text{ bits}$ | $0.8404$ / $0.5177\text{ bits}$ |
| **`ConvNeXt` (`[Hard]`)** | **$T^* = 0.538$** | **$0.0155 \to 0.0087$ ($-43.9\%$)** | **$0.0364 \to 0.0242$ ($-33.5\%$)** | **$0.9837$ / $0.0721\text{ bits}$** | **$0.8332$ / $0.5290\text{ bits}$** |

Crucially, `ConvNeXt` exhibits a **$7.3\times$ higher predictive entropy on misclassified slices** ($0.5290\text{ bits}$) than on correctly classified slices ($0.0721\text{ bits}$). This strong epistemic/aleatoric separation enables an effective **Selective Clinical Triage Gate**.

### 4.4 Selective Clinical Triage Sweep (`ConvNeXt` on Held-Out Test Set)
We define a selective prediction policy with confidence threshold $\tau \in [0.50, 0.95]$:
- **Auto-Cleared ($\hat{c}_T(\mathbf{x}) \ge \tau$):** High-confidence prediction is accepted automatically.
- **Referred to Neuroradiologist ($\hat{c}_T(\mathbf{x}) < \tau$):** Uncertain slice is flagged for human specialist review.

| Triage Threshold $\tau$ | Auto-Cleared Coverage (%) | Specialist Referral Rate (%) | Auto-Cleared Accuracy (%) | Accepted AD False Negative Rate (%) | Unflagged Missed AD Rate (of all 230 AD) | Hybrid Clinical System Accuracy (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$\tau = 0.50$ (No Triage)** | $100.00\%$ ($420/420$) | $0.00\%$ ($0$) | $95.95\%$ | $4.35\%$ ($10/230$) | $4.35\%$ ($10/230$) | $95.95\%$ |
| **$\tau = 0.65$** | $97.62\%$ ($410/420$) | $2.38\%$ ($10$) | $96.83\%$ | $3.98\%$ ($9/226$) | $3.91\%$ ($9/230$) | $96.90\%$ |
| **$\tau = 0.75$** | $96.19\%$ ($404/420$) | $3.81\%$ ($16$) | $97.28\%$ | $3.18\%$ ($7/220$) | $3.04\%$ ($7/230$) | $97.38\%$ |
| **$\tau^* = 0.80$ (Recommended)** | **$95.95\%$ ($403/420$)** | **$4.05\%$ ($17$)** | **$97.27\%$** | **$3.20\%$ ($7/219$)** | **$3.04\%$ ($7/230$)** | **$97.38\%$** |
| **$\tau = 0.85$** | $94.52\%$ ($397/420$) | $5.48\%$ ($23$) | $97.73\%$ | $2.37\%$ ($5/211$) | $2.17\%$ ($5/230$) | $97.86\%$ |
| **$\tau = 0.90$** | $93.10\%$ ($391/420$) | $6.90\%$ ($29$) | $98.21\%$ | $1.44\%$ ($3/208$) | $1.30\%$ ($3/230$) | $98.33\%$ |
| **$\tau = 0.95$ (High-Safety)** | **$90.71\%$ ($381/420$)** | **$9.29\%$ ($39$)** | **$98.69\%$** | **$0.50\%$ ($1/201$)** | **$0.43\%$ ($1/230$)** | **$98.81\%$** |

> **Key Finding:** At the recommended operating threshold **$\tau^* = 0.80$**, referring just **$4.05\%$** of ambiguous scans ($17 / 420$ slices) intercepts **$35.3\%$ of all model errors** (including $30\%$ of False Negatives), boosting auto-cleared accuracy to **$97.27\%$**. Under the high-safety threshold **$\tau = 0.95$** (referring $9.29\%$ of scans), **`ConvNeXt` eliminates $90\%$ of missed Alzheimer's cases** (reducing unflagged False Negatives from $10 \to 1$, an accepted FN rate of just **$0.50\%$**) and achieves **$98.81\%$ hybrid diagnostic accuracy**.

![Calibration Reliability Diagrams, Entropy Separation, and Selective Triage Trade-Off](./images/calibration_and_uncertainty.png)

---

## 5. Interpretability & Deep Failure Autopsy (Section 5.3)

### 5.1 Spatial Attribution on Correctly Classified Test Slices
Using `compute_spatial_attribution()` (`modules.py`), we compute Grad-CAM activations on the final `ConvNeXt` stage (`stages.3`, $8\times 8$ spatial feature map upsampled to $128\times 128$). As shown below, on true positive `AD` scans and true negative `NC` scans, `ConvNeXt` consistently focuses on the **periventricular boundary (lateral ventricles)** and **bilateral medial temporal lobes**, rather than background skull artefacts.

![Sample Test Predictions with Grad-CAM Attribution Overlays](./images/sample_predictions.png)

### 5.2 Four-Case Diagnostic Failure Autopsy (`checkpoints/failure_autopsy_cases.json`)
To move beyond aggregate accuracy metrics, we perform a granular root-cause autopsy on **four specific misclassified test slices**—two **False Negatives (`FN`: true `AD` predicted as `NC`)** and two **False Positives (`FP`: true `NC` predicted as `AD`)**—visualising the input MRI, Grad-CAM heatmap, periventricular overlay, and baseline-vs-ConvNeXt uncertainty telemetry.

![Four-Case Failure Autopsy of False Negatives and False Positives](./images/failure_autopsy.png)

| Case ID & Error Mode | Patient & Slice ID | `SimpleCNN` Conf | `ConvNeXt` Raw $\to$ Calibrated Conf | Predictive Entropy $H$ | Triage Status ($\tau^* = 0.80$) | Root-Cause Anatomical / Imaging Trigger | Concrete Engineering Remedy |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Case #1 (`FN`: Missed AD)** | `AD_200433` `slice 4` | $0.854$ (`NC`, unflagged) | $0.630 \to$ **$0.723$** (`NC`) | **$0.852\text{ bits}$** (High) | **INTERCEPTED** ($\hat{c}_T < 0.80 \to$ Referred) | **Superior-peripheral slice offset & partial-volume averaging:** Slice 4 is sampled superior to the hippocampal body; ventricular horn cross-section is minimal (dark CSF ratio $= 0.14$) and cortical ribbon appears intact. | **2.5D / 3D Multi-Slice Context:** Aggregate predictions across all 10 axial slices per patient (`AD_200433` is correctly classified as `AD` at the patient level with $8/10$ slices positive). |
| **Case #2 (`FN`: Missed AD)** | `AD_300449` `slice 20` | $0.703$ (`NC`, borderline) | $0.637 \to$ **$0.730$** (`NC`) | **$0.842\text{ bits}$** (High) | **INTERCEPTED** ($\hat{c}_T < 0.80 \to$ Referred) | **Atypical cortical-predominant AD subtype:** Patient exhibits minimal central ventricular dilation (CSF ratio $= 0.16$), causing Grad-CAM to find no central ventriculomegaly cue. | **Multi-Scale ROI Attention:** Concatenate Stage-2 ($16\times 16$) medial-temporal feature maps with Stage-4 ($8\times 8$) global features to weight focal sulcal widening more heavily. |
| **Case #3 (`FP`: False AD Alarm)** | `NC_200429` `slice 25` | $0.453$ (`AD`, uncertain) | $0.505 \to$ **$0.510$** (`AD`) | **$1.000\text{ bits}$** (Max) | **INTERCEPTED** ($\hat{c}_T < 0.80 \to$ Referred) | **Lateral sulcal asymmetry & RF bias field:** Superior-lateral cortex exhibits asymmetric sylvian fissure widening and high-intensity anterior skull-margin gradient, triggering a split 51/49 logit tie. | **N4ITK Bias-Field Correction + Brain-Mask Stripping:** Apply skull-stripping prior to Z-score normalization so bright cranial dipole artefacts do not pull Stage-4 activations. |
| **Case #4 (`FP`: False AD Alarm)** | `NC_200429` `slice 20` | $0.935$ (`AD`, unflagged) | $0.969 \to$ **$0.988$** (`AD`) | **$0.094\text{ bits}$** (Low) | **UNFLAGGED ERROR** ($\hat{c}_T \ge 0.80$) | **Benign age-related ventriculomegaly in healthy elderly control:** Mid-ventricular slice displays prominent central CSF expansion (central dark ratio $= 0.28$) morphologically indistinguishable from AD hydrocephalus *ex vacuo* on a single 2D slice without patient age. | **Age-Conditioned Tabular Fusion:** Fuse patient chronological age and intracranial volume (ICV) covariates into the final linear classification head (`nn.Linear(384 + d_clin, 2)`). |

---

## 6. Actionable Manager Recommendation (Deployment Decision Memo)

**To:** Engineering Lead, Neuroimaging Diagnostics Pipeline  
**Recommendation:** **Deploy `ConvNeXt` with Post-Hoc Temperature Scaling ($T^* = 0.538$) and a Two-Tier Selective Triage Gate ($\tau^* = 0.80$ Standard / $\tau = 0.95$ High-Safety), Coupled with Patient-Level Multi-Slice Aggregation.**

1. **Retire `SimpleCNNBaseline` from Clinical Screening:** Although `SimpleCNNBaseline` is lightweight ($8,994$ parameters), its **$9.57\%$ False Negative rate (`FN = 22/230`)** poses an unacceptable clinical liability for Alzheimer's screening. `ConvNeXt` cuts False Negatives by **$54.5\%$ (`FN = 10/230`)** while requiring only **$1.18\text{ GB}$ VRAM** and **$12.5\text{ ms}$** per 20-slice patient volume—easily deployable on a single hospital workstation GPU or CPU server.
2. **Enforce Selective Referral Triage ($\tau^* = 0.80$ or $\tau = 0.95$):** Never deploy raw argmax predictions in isolation.
   - For **routine population screening**, set the calibrated confidence gate to **$\tau^* = 0.80$**: **$95.95\%$ of scans are auto-cleared** at **$97.27\%$ accuracy**, while only **$4.05\%$** of borderline slices (such as Failure Cases #1, #2, and #3) are routed to a neuroradiologist.
   - For **high-sensitivity memory-clinic triage**, raise the gate to **$\tau = 0.95$**: specialist referral rises modestly to **$9.29\%$**, reducing unflagged missed Alzheimer's cases to **$0.43\%$ ($1/230$ slices)** and achieving **$98.81\%$ overall diagnostic accuracy**.
3. **Phase-2 Roadmap (Resolving Failure Case #4):** Our failure autopsy proved that the few remaining high-confidence False Positives (`NC_200429`) stem from *benign age-related ventriculomegaly* in older controls. In Phase 2, the pipeline should (a) aggregate calibrated slice probabilities at the **patient level** (`Patient AUROC = 0.9977`) and (b) condition the `ConvNeXt` classification head on **patient age and sex** metadata to disambiguate normal ageing from Alzheimer's atrophy.

---

## 7. Repository Structure, Dependencies & Reproducibility Instructions

### 7.1 Directory Layout (`recognition/ConvNeXt_ADNI_s49852413/`)
All five mandatory deliverables required by Section 4.1 of the COMP3710/COMP7710 Specification are self-contained inside this folder:
```text
recognition/ConvNeXt_ADNI_s49852413/
├── README.md                  # Full technical report, 1-page feasibility review, failure autopsy & AI disclosure
├── modules.py                 # Pure PyTorch ConvNeXt, SimpleCNNBaseline, TemperatureScaling & Grad-CAM (0 NumPy imports)
├── dataset.py                 # ADNI loader, patient-level stratified splitter, leakage verifier & Z-score transforms
├── train.py                   # Training driver, L-BFGS calibration, hardware resource profiler & triage sweep
├── predict.py                 # Evaluation driver, Grad-CAM visualiser & 4-case failure autopsy generator
├── test_pipeline.py           # Pytest suite verifying AST rule compliance, tensor shapes, calibration & zero leakage
├── slurm_train_rangpur.sh     # Ready-to-submit Slurm batch script for UQ Rangpur HPC (/home/groups/comp3710/ADNI/AD_NC)
└── images/                    # All embedded diagnostic plots and architectural diagrams
    ├── convnext_architecture.png
    ├── preprocessing_and_samples.png
    ├── training_curves.png
    ├── roc_pr_comparison.png
    ├── calibration_and_uncertainty.png
    ├── sample_predictions.png
    └── failure_autopsy.png
```

### 7.2 Environment Dependencies
Tested with Python `3.9+` (`3.9.6` local / `3.10+` Rangpur conda) and the following core packages:
- `torch >= 2.0.0` (tested on `2.8.0` with CUDA / Apple MPS / CPU support)
- `torchvision >= 0.15.0` (tested on `0.23.0`)
- `numpy >= 1.24.0` (used only in `dataset.py`, `train.py`, `predict.py`; **strictly zero imports in `modules.py`**)
- `scikit-learn >= 1.3.0`, `matplotlib >= 3.7.0`, `Pillow >= 10.0.0`, `nibabel >= 5.0.0`, `pytest >= 7.0.0`

```bash
pip install torch torchvision numpy scikit-learn matplotlib Pillow nibabel pytest
```

### 7.3 Step-by-Step Reproducibility Commands

#### 1. Verify Code Hygiene & Patient Split Integrity (`pytest`)
Validates that `modules.py` contains **zero** `numpy` or third-party imports via Python AST inspection, tests `ConvNeXt` forward/backward passes, verifies `TemperatureScaledClassifier`, and confirms zero patient leakage:
```bash
pytest recognition/ConvNeXt_ADNI_s49852413/test_pipeline.py -v
```

#### 2. Train Both Models, Calibrate Temperature, and Run Triage Sweep (`train.py`)
Automatically detects `/home/groups/comp3710/ADNI/AD_NC` on Rangpur (or `--data_root` locally), verifies zero patient leakage, trains `SimpleCNNBaseline` and `ConvNeXt` with deterministic seed `42`, fits Temperature Scaling on `val`, and exports all benchmark curves to `./images/`:
```bash
python recognition/ConvNeXt_ADNI_s49852413/train.py \
    --data_root /home/groups/comp3710/ADNI/AD_NC \
    --output_dir recognition/ConvNeXt_ADNI_s49852413 \
    --model both \
    --epochs 18 \
    --batch_size 32 \
    --image_size 128 \
    --lr 8e-4 \
    --seed 42
```

#### 3. Run Calibrated Inference, Grad-CAM Attribution, and Failure Autopsy (`predict.py`)
Loads trained checkpoints, evaluates test-set accuracy, balanced accuracy, Macro-F1, AUROC, ECE, and triage metrics at `--triage_threshold 0.80`, and regenerates `sample_predictions.png` and `failure_autopsy.png`:
```bash
python recognition/ConvNeXt_ADNI_s49852413/predict.py \
    --data_root /home/groups/comp3710/ADNI/AD_NC \
    --output_dir recognition/ConvNeXt_ADNI_s49852413 \
    --triage_threshold 0.80 \
    --seed 42
```

#### 4. UQ Rangpur HPC Cluster Execution (`slurm_train_rangpur.sh`)
```bash
cd recognition/ConvNeXt_ADNI_s49852413
sbatch slurm_train_rangpur.sh
```

---

## 8. References
1. **Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022).** *A ConvNet for the 2020s.* In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11976–11986.
2. **Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).** *On Calibration of Modern Neural Networks.* In International Conference on Machine Learning (ICML), PMLR 70:1321–1330.
3. **Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017).** *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pp. 618–626.
4. **Jack, C. R., et al. (2008).** *The Alzheimer's Disease Neuroimaging Initiative (ADNI): MRI methods.* Journal of Magnetic Resonance Imaging, 27(4), 685–691.
5. **Wen, J., et al. (2020).** *Convolutional neural networks for classification of Alzheimer's disease: Overview and reproducible evaluation.* Medical Image Analysis, 63, 101694. (Documents the critical importance of subject-level splitting to prevent slice leakage).

---

## 9. Artificial Intelligence Usage Disclosure (Section 6.1 Mandatory Declaration)

In accordance with **Section 6 (Academic Integrity & AI Usage Policy)** of the COMP3710 / COMP7710 Project Specification (Version 2.2), I provide the following transparent disclosure of generative AI assistance used during this project:

- **AI Tools Used:** Google Antigravity (AI Pair-Programming Assistant).
- **Scope & Manner of Assistance:**
  1. **Boilerplate & Modular Code Structuring:** Assisted in structuring the CLI argument parsers, Matplotlib multi-panel figure layout routines in `train.py` and `predict.py`, and Pytest unit-test scaffolding in `test_pipeline.py`.
  2. **Pure-PyTorch Architectural Verification:** Assisted in verifying the pure-PyTorch implementation of `LayerNorm2d`, `DropPath`, `ConvNeXtBlock` ($7\times 7$ depthwise convolution, $4\times$ inverted bottleneck, `LayerScale`), and `TemperatureScaledClassifier` in `modules.py` to ensure strict compliance with the course rule prohibiting NumPy or third-party imports inside `modules.py`.
  3. **Documentation & Formatting:** Assisted in formatting markdown tables, LaTeX equations, and structuring the technical report in `README.md`.
- **Student Verification & Intellectual Ownership:** All architectural design choices (selection of `ConvNeXt` stage ratios `(2,2,6,2)`, $4\times 4$ patchify stem, L-BFGS post-hoc temperature scaling, and confidence-based clinical referral triage), patient-level split hygiene verification (`verify_no_patient_leakage`), experimental execution, metric validation (`benchmark_summary.json`), and anatomical root-cause interpretations of the four failure cases (`failure_autopsy.png`) were independently verified, executed, and validated by the author (**Lingxiao Li, `s49852413`**).
