## 1. Project Description
This project investigates automated anatomical contouring of the prostate gland from pelvic MRI scans (HipMRI dataset) for targeted radiotherapy planning. The objective is to evaluate whether 3D volumetric context (3D Improved U-Net) significantly resolves slice-transition artifacts and clinical boundary leakage (into adjacent rectal wall and bladder mucosa) compared to a slice-wise 2D U-Net baseline, under strict patient-level separation.

## 2. Data & Setup
- **Dataset:** HipMRI Study (`/home/groups/comp3710/HipMRI_Study_open`)
- **Framework:** Python

## 3. Feasibility Report

### 3.1 User Need, Scope & Acceptance Criteria
* **Intended User & Clinical Context:** Radiation oncologists and medical physicists performing external-beam radiotherapy dose planning. The tool serves as an interactive clinical contouring assistant that automatically proposes prostate target volumes to minimize manual contouring time while preventing lethal under-contouring or severe radiation proctitis/cystitis.
* **Open Dilemma Addressed:** Resolving the *Spatial Context vs. Clinical Boundary Utility Dilemma*—mitigating severe boundary leakage and slice-transition disconnections at the anatomical base (superior) and apex (inferior) where standard 2D slice-wise methods fail.
* **Testable Acceptance Criteria:**
  1. **Primary Segmentation Accuracy:** Achieve a patient-level mean Sørensen-Dice Similarity Coefficient (DSC) $\ge 0.70$ (Hard 3D target) on the held-out test set (Fallback: $\ge 0.75$ for 2D prostate slices).
  2. **Boundary & Overlap Utility:** Achieve an Intersection-over-Union (IoU) $\ge 0.60$ with dedicated error reporting on base and apex slices.
  3. **Hardware & VRAM Constraints:** Model peak training memory footprint must strictly remain $\le 12.0\text{ GB}$ VRAM per GPU on UQ Rangpur compute nodes (utilizing sub-volume patch cropping or mixed precision).
  4. **Inference Latency:** Complete full-volume 3D inference in $< 15\text{ seconds}$ per patient scan on a single GPU.

### 3.2 Model Choice & Course Concepts
* **Dataset & Partitioning:** The UQ HipMRI study dataset (`/home/groups/comp3710/HipMRI_Study_open`), containing 38 patients and 211 3D MRI volumes with expert-annotated labels (prostate, bladder, rectum, bones).
  * *Data Hygiene:* A strict **patient-level split** (approx. 26 train, 6 validation, 6 test patients) is enforced to prevent inter-slice data leakage.
* **Proposed Architecture (Hard Difficulty):** 3D Improved U-Net (Isensee et al., 2018). It incorporates 3D volumetric convolutions to capture inter-slice anatomical continuity, context residual units to prevent vanishing gradients, and deep supervision across multi-scale decoders.
* **Baseline Architecture (Normal/Easy Baseline):** A standard 2D U-Net / 2D Improved U-Net operating on pre-extracted axial slices (`keras_slices_data`) to serve as an empirical benchmark.
* **Theoretical Justification:** 2D networks ignore out-of-plane z-axis correlation, producing jagged boundaries at the prostate apex/base. Volumetric residual blocks and combined Dice-Cross-Entropy loss directly address extreme class imbalance (prostate $< 2\%$ volume) while maintaining anatomical topology.

### 3.3 Preliminary Feasibility Evidence
* **Data Audit & Modality:** HipMRI NIfTI volumes inspected via Nibabel specifications:
  * Dimensions: 2D axial slices ($256 \times 256$) from `keras_slices_data`; volumetric MRs downsampled/cropped to fixed grid sizes ($128 \times 128 \times 64$).
  * Class Imbalance: Prostate target foreground constitutes approximately $1.2\% - 1.8\%$ of total pelvic voxel volume, confirming the theoretical necessity of weighted Dice loss over pure cross-entropy.
* **Empirical Smoke Test (Local RTX 4060 GPU):**
  * Baseline 2D U-Net (7.76M params) and Normal-tier 2D Improved U-Net (3.19M params, 58.9% reduction) successfully compiled in pure PyTorch.
  * Verified forward/backward convergence: Peak GPU VRAM allocated is **695.63 MB** (well below the 8.0 GB ceiling), with an inference latency of **4.41 ms / slice**.

### 3.4 Technical Risks, Compute Budget & Fallback Plan
* **Primary Technical Risks:**
  1. *GPU Out-Of-Memory (OOM):* Full-resolution 3D volumetric convolutions risk exceeding memory buffers; mitigated via sub-volume patches ($64 \times 64 \times 32$) and mixed precision.
  2. *Severe Class Imbalance:* Risk of trivial empty-mask convergence; addressed via `CombinedLoss`.
* **Compute Budget & Environment:**
  * Training executed on local NVIDIA RTX 4060 GPU or UQ Rangpur HPC (NVIDIA V100/A100).
  * **Measured / Estimated Run Time:** ~0.5 hours for 2D prototype (40 epochs); estimated 2–4 GPU hours for 3D volumetric model (100 epochs) with in-memory caching.
* **Workable Fallback Plan (Graceful Degradation):**
  * The 2D Improved U-Net (Normal Difficulty, target Dice $\ge 0.75$) serves as the fully functional, verified deliverable. If 3D volumetric memory constraints or schedule limits arise, the project safely anchors on this 2D prototype, fulfilling all marking criteria without risk of non-completion.