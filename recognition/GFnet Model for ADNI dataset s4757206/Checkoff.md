# Feasibility Review — GFNet for Alzheimer's Disease Classification

**COMP3710 Open-Ended Project | Domain: Medical Image Classification | Difficulty: Hard**

### 1. User Need, Scope & Acceptance Criteria

This project aims to develop a deep learning prototype to assist medical reviewers in distinguishing Alzheimer's Disease (AD) from Cognitively Normal (CN) cases using 2D T1-weighted brain MRI slices from the ADNI dataset. The system will compare a conventional CNN baseline with a Global Filter Network (GFNet), while investigating prediction confidence and identifying uncertain cases requiring expert review.

**Acceptance criteria:**
- Achieve at least 80% test accuracy for AD versus CN classification.
- Compare GFNet against a CNN baseline using accuracy, per-class precision, recall, macro-F1, and AUROC.
- Prevent patient-level data leakage across training, validation, and test sets.
- Evaluate prediction confidence and investigate 3–5 representative failure cases.
- Establish an empirical confidence threshold for referring uncertain predictions to human reviewers, and report inference latency and GPU memory usage.

### 2. Model Choice & Course Concepts

The project uses the ADNI dataset, comprising 2D MRI slices labelled AD or CN. A simple CNN will serve as the baseline, while GFNet is the selected advanced architecture (Hard Difficulty).

GFNet replaces conventional self-attention with global frequency-domain filtering using the 2D Fast Fourier Transform (FFT), learnable frequency-domain filters, and inverse FFT. This approach enables global spatial information mixing. Relevant course concepts include convolutional neural networks, Fourier transforms, backpropagation, classification metrics, and model generalisation.

Both models will be trained and evaluated using the same patient-independent data partitions.

### 3. Preliminary Feasibility Evidence

Initial data inspection identified 10,400 AD and 11,121 CN training slices, together with 4,460 AD and 4,540 CN test slices. Images are grayscale with a resolution of 256 × 240 pixels.

Metadata analysis identified 680 unique subjects, with 216 subjects appearing in both the supplied training and test partitions. All examined image identifiers were successfully mapped to metadata, and no conflicting AD/CN subject labels were detected.

This demonstrates a significant patient-level leakage risk in the supplied split. A new stratified, subject-level 70/15/15 training, validation, and test partition is proposed. Model training and computational feasibility testing remain pending.

### 4. Risks, Budget & Fallback

**Key risks:** Patient-level leakage, limited GPU availability, GFNet training cost, overfitting, and unreliable confidence estimates.

**Mitigation:** Use patient-independent partitions, reduced image resolution where appropriate, smaller training batches, early stopping, and reproducible experiments.

**Computing budget:** Google Colab GPU resources, with peak VRAM, epoch duration, and inference latency to be measured during initial experiments.

**Next experiment:** Construct the patient-independent data splits, train a small CNN baseline, and benchmark GFNet training feasibility on a limited subset.

**Fallback:** If the full GFNet architecture exceeds available resources, use a reduced-width GFNet retaining its core Fourier filtering mechanism, and report the computational trade-offs.