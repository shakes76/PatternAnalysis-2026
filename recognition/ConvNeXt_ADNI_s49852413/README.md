# Uncertainty-Aware ConvNeXt for Alzheimer's Disease T1-MRI Screening & Automated Clinical Triage (ADNI)

**Author:** Lingxiao Li  
**Student ID:** 49852413 (`s49852413`)  
**Course:** COMP3710 / COMP7710 — Pattern Recognition and Analysis (2026)  
**Project Domain:** Section 2.1 Classification — Medical Diagnosis (Neuroimaging: ADNI Brain MRI)  
**Model & Difficulty Tier:** `ConvNeXt` (`[Hard Difficulty]`, eligible for 20/20 implementation ceiling)

---

## 1. Project Overview & Problem Statement

Alzheimer's Disease (AD) is an irreversible, progressive neurodegenerative disorder characterized anatomically on T1-weighted structural magnetic resonance imaging (MRI) by medial temporal lobe (hippocampal and entorhinal) atrophy, cortical grey matter ribbon thinning, and compensatory ventricular enlargement (hydrocephalus *ex vacuo*). Early, reliable differentiation between Alzheimer's Disease (`AD`) and Cognitively Normal (`CN` / `NC`) controls is essential for clinical triage and therapeutic intervention. However, automated 2D slice-based screening faces two acute engineering challenges: (1) high inter-subject anatomical variability in head geometry and normal aging sulcal widening, which easily confounds local convolutional filters, and (2) severe risk of **cross-slice patient data leakage** when adjacent 2D slices from the same 3D volume are split randomly across training and evaluation sets.

To address these challenges and resolve the domain's **Confidence and Ambiguity Dilemma in Automated Triage**, this project implements a modernized convolutional backbone—**ConvNeXt** ([Liu et al., CVPR 2022](https://arxiv.org/abs/2201.03545))—from scratch in pure PyTorch, paired with **post-hoc Temperature Scaling calibration** and a **Selective Reject-Option Triage Policy**. We benchmark `ConvNeXt` against a standard convolutional baseline (`SimpleCNNBaseline`) under strict patient-level separation, audit edge-case failure modes on borderline scans, and establish an evidence-based confidence threshold that safely automates high-certainty cases while routing ambiguous boundary scans to neuroradiologists.

---

## 2. Feasibility Review (Section 3 Compulsory Midpoint Check-Off)

### 2.1 User Need, Scope & Testable Acceptance Criteria
* **Intended User & Operational Purpose:** Neuroradiologists and clinical triage coordinators in a memory-clinic screening workflow. The prototype serves as an **automated first-pass screening and uncertainty-routing assistant** that classifies incoming T1-weighted brain MRI slices into `AD` vs. `NC`, automatically clearing high-confidence normal/pathological cases while flagging low-confidence or borderline scans for human specialist review.
* **Testable Acceptance Criteria:**
  1. **Zero Patient Leakage (Data Hygiene):** $100\%$ patient-level separation across training, validation, and testing splits ($\lvert \mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{val}} \rvert = \lvert \mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{test}} \rvert = \lvert \mathcal{P}_{\text{val}} \cap \mathcal{P}_{\text{test}} \rvert = 0$), verified programmatically prior to training.
  2. **Diagnostic Accuracy Target:** Held-out test accuracy $\ge 0.80$ ($80\%$) and macro-F1 $\ge 0.80$ on `AD` vs. `NC` classification, outperforming the implemented `SimpleCNNBaseline` on identical patient splits.
  3. **Uncertainty Calibration & Triage Utility:** Post-calibration Expected Calibration Error ($\text{ECE} \le 0.05$) and a calibrated reject-option threshold $\tau^*$ that reduces false-negative rate by $\ge 40\%$ on accepted automated predictions at $\ge 70\%$ cohort coverage.
  4. **Operational Resource Constraint:** Peak training/inference GPU memory footprint $< 4.0\text{ GB}$ VRAM and single-slice inference latency $< 15\text{ ms/slice}$ ($> 65\text{ slices/sec}$), enabling real-time clinical workstation deployment without multi-GPU hardware.

### 2.2 Model Choice, Difficulty Band & Course Concepts
* **Dataset Configuration:** Preprocessed ADNI T1-weighted 2D brain MRI dataset (`/home/groups/comp3710/ADNI/AD_NC`), comprising axial/coronal brain slices (`256 × 240` JPEG) across `AD` and `NC` cohorts with subject/scan identifiers encoded as `<scan_id>_<slice_idx>.jpeg`.
* **Chosen Architecture (`ConvNeXt` — `[Hard Difficulty]`):** Standard CNNs (`SimpleCNNBaseline`) use small $3 \times 3$ kernels with limited receptive fields, struggling to relate ventricular enlargement in the central brain to peripheral cortical thinning. Vision Transformers capture global context via self-attention but suffer from quadratic complexity and weak inductive bias on moderate-sized medical cohorts. **ConvNeXt** ([Liu et al., 2022](https://arxiv.org/abs/2201.03545)) modernizes the pure ConvNet using Vision Transformer design principles covered in COMP3710:
  * A $4 \times 4$ non-overlapping **patchify stem**;
  * **$7 \times 7$ depthwise separable convolutions** expanding the spatial receptive field to capture cross-regional ventricular-to-cortical morphology efficiently;
  * **Inverted bottleneck blocks** ($d \to 4d \to d$ via $1 \times 1$ pointwise convolutions) with **GELU** activations and **LayerNorm2d**;
  * **LayerScale** ($\gamma \in \mathbb{R}^C$) and **Stochastic Depth (`DropPath`)** for regularization and feature diversity.

### 2.3 Preliminary Feasibility Evidence
* **Data Audit:** Inspection of the ADNI directory structure confirms multiple sequential 2D slices per subject scan (`<scan_id>_<slice_idx>.jpeg`). Naive slice-wise random splitting leaks almost identical adjacent slices across splits, artificially inflating validation accuracy to $>97\%$. Grouping by `<scan_id>` yields clean, disjoint patient cohorts.
* **Smoke-Test & Resource Estimate:** An initial forward/backward smoke test of our custom `ConvNeXt-Tiny/Small` medical variant ($C = [48, 96, 192, 384]$, depths $[2, 2, 6, 2]$, $\sim 3.4\text{M}$ parameters) at resolution $192 \times 192$ with batch size $32$ consumes $< 1.2\text{ GB}$ peak VRAM, processes an epoch in $< 4\text{ s}$ on GPU/MPS, and converges smoothly without gradient instability.

### 2.4 Technical Risks, Computing Budget & Workable Fallback
* **Key Technical Risks:** (1) Overfitting to subject-specific skull/intensity artifacts when training under strict patient-level splits; (2) Overconfident softmax probabilities on out-of-distribution peripheral slices (extreme superior/inferior slices with heavy partial volume effects).
* **Computing Budget:** $< 15\text{ minutes}$ total wall-clock training time ($30\text{ epochs}$) on a single Rangpur NVIDIA A100 GPU (or Apple M-series MPS GPU locally).
* **Mitigation & Workable Fallback:** Enforce per-slice Z-score intensity normalization, label smoothing ($\varepsilon = 0.05$), `DropPath` ($p = 0.15$), and post-hoc **Temperature Scaling** on the held-out validation patient split. If deeper `ConvNeXt` stages overfit on patient-separated splits, fall back to our compact 4-stage `ConvNeXt-Med` configuration with cosine weight decay and patient-level ensemble aggregation.
