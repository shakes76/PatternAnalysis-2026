# Synthetic Prostate MRI Generation with WGAN-GP (HipMRI 2D)

This project compares a DCGAN baseline model with two different WGAN-GP models. Once WGAN-GP features the exact same architecture (latent space, num generator features, etc) and identical data pre-processing, while the other model adjusts data pre-proessing and increases both latent space a generator feautres. The aim was to see if a GAN could produce realistic, sharable synthetic images without memorising patient scans (with a SSIM > 0.6). The final tuned WGAN-GP model attainted an average SSIM of 0.547, with 11.1% of generated images surpassing SSIM 0.6. At this stage, the final model cannot be recommended for deployment, however additional tuning to the final model presented in this report (i.e. changing to a ResNet architecture) may meet the SSIM acceptance criteria. No models appeared to memorise patient data and all models were under the computational requirements suggested, therefore with additional tuning it is suggested that a model capable of generatic non-memorised, high-quality synthetic images is plausible.
> [!IMPORTANT]
> **Please review** the two notes within the **4.2 Pre-Processing** section to understand the difference in displayed results for
> `dcgan` and `wgan_gp` versus `wgan_gp_wide`.

<div align="center">

![WGAN-GP-Tuned Training](images/wgan_gp_wide/wgan_gp_wide_anim.gif)
### Figure 1: Animation of Final WGAN - 250 Epochs of Training

</div>

## Contents
1. [Feasibility Review](#1-feasibility-review)
   - [1.1 User Need, Scope and Acceptance Criteria](#11-user-need-scope-and-acceptance-criteria)
   - [1.2 Model Choice and Course Concepts](#12-model-choice-and-course-concepts)
   - [1.3 Preliminary Feasibility Evidence](#13-preliminary-feasibility-evidence)
   - [1.4 Risk, Budget and Fallback](#14-risk-budget-and-fallback)
2. [How It Works](#2-how-it-works)
   - [2.1 DCGAN (Baseline)](#21-dcgan-baseline)
   - [2.2 WGAN-GP](#22-wgan-gp)
   - [2.3 Architecture](#23-architecture)
3. [Usage](#3-usage)
   - [3.1 Dependencies](#31-dependencies)
   - [3.2 Hyperparameters](#32-hyperparameters)
   - [3.3 Reproducing Results](#33-reproducing-results)
4. [Data](#4-data)
   - [4.1 Dataset & Data Audit](#41-dataset--data-audit)
   - [4.2 Pre-Processing](#42-pre-processing)
   - [4.3 Splits](#43-splits)
5. [Experiments](#5-experiments)
   - [5.1 Training Metrics](#51-training-metrics)
   - [5.2 Generated Samples](#52-generated-samples)
   - [5.3 Tuned WGAN-GP](#53-tuned-wgan-gp)
6. [Results](#6-results)
   - [6.1 Quantitative Benchmark](#61-quantitative-benchmark)
   - [6.2 Inverse SSIM](#62-inverse-ssim)
   - [6.3 Memorisation Audit](#63-memorisation-audit)
   - [6.4 Latent Space Interpolation](#64-latent-space-interpolation)
   - [6.5 Resource Profiling](#65-resource-profiling)
7. [Failure Analysis](#7-failure-analysis)
8. [Recommendation](#8-recommendation)
   - [8.1 Limitations](#81-limitations)
   - [8.2 Future Work](#82-future-work)
9. [AI Use Disclosure](#9-ai-use-disclosure)
10. [References](#10-references)

## 1. Feasibility Review
**Topic 2.4 - Image Generation: Synthetic Medical Cohorts and Privacy**
### 1.1 User Need, Scope and Acceptance Criteria
The purpose of this research is to provide additional medical imaging resources to medical researchers investigating 2D pelvic MRI slices. Through WGAN generation, new slices could be synthesised which replicate realistic tissue structures and remove the risk of violating patient privacy laws such as HIPPA and GDPR. Final intentions for this project are to determine whether the model can produce synthetic images that are safe, structurally sound and could plausibly be used for training. To address the fidelity vs memorisation vs privacy dilemma, testable acceptance criteria of the final model include:
- Structural Similarity Index of greater than 0.6 (SSIM > 0.6) across synthesised cohorts against reference anatomical slices.
- WGAN results are better than those of the baseline model.
- Nearest-neighbour distance audit and latent space interpolation, sample distribution should not be closer than test distribution.
- Peak VRAM < 4 GB and < 6 hours GPU training.
### 1.2 Model Choice and Course Concepts
The chosen model is the WGAN-GP which is Normal difficulty, on HipMRI 2D slices which are already split into test, train and validation sets. The baseline model is a DCGAN which is identical apart from the loss and Lipschitz constraint.
- Must implement Earth-Mover distance. This is part of what makes it a WGAN, providing a continuous distance (differentiable almost everywhere) that allows the critic to be trained towards optimality. We can then train with low likelihood of mode collapse and image generation improving for higher number of iterations.
- Critic must have Lipschitz constraint. Based on the paper this is the other requirement that allows the critic to have a continuous loss function to be optimised.
- To be a WGAN-GP rather than just a WGAN, the loss function for the critic must include a gradient penalty scaled by some lambda (this penalty directly incorporates the Lipschitz constraint).
- Starting parameters for the system are based on those noted in the WGAN-GP paper, α=0.0001,λ=10,n_CRITIC=5 where these are learning rate, gradient penalty lambda and number of critic iterations per generator iteration, respectively. Batch size is adjusted from 64 to 128.
- Builds directly on course concepts involving DCGANs and WGANs (both covered in Week 9 lecture content) as well as loss/activation functions, hyperparameters and model evaluation methods.
### 1.3 Preliminary Feasibility Evidence
An initial data audit was completed using `data_audit.py`, which can be summarised into this table:

<div align="center">

| **Measure** | **Train** | **Validate** | **Test** |
| --- | :---: | :---: | :---: |
| Slices | 11,460 | 660 | 540 |
| Patients | 31 | 4 | 3 |
| Slice Shapes | 256x128 (11,400), 256x144 (60) | 256x128 | 256x128 |
| Intensity Range (Sampled) | 0 – 1,150 | 0 – 565  | 0 – 617 |
| Median Per-Slice | 387.5 | 373 | 424 |
| Near-Empty Slices | 0% | 0% | 0% |

</div>

An analysis of this preliminary data audit can be found in **Section 4: Data**.
### 1.4 Risk, Budget and Fallback
The key technical risks associated with the project are:
- WGAN has a risk of unstable training results if a momentum-based optimiser (Adam) or high learning rates are used, placing some restrictions on model architecture and tuning parameters.
- Rangpur script delays due to high student traffic approaching the due date.
- Chosen parameters exceed GPU memory or provided training time, which may not become apparent until training failure occurs.
- Test set only has 3 patients which limits strong statements surrounding how well generalised the data is.
The next planned experiment is memory/time estimations followed by baseline model (DCGAN) implementation. In the case where proposed risks begin to impact the deliverables of the project, the DCGAN baseline can instead become the main project with a basic autoencoder becoming the new baseline. This is a different normal difficulty option under the same topic.


## 2. How It Works
Exact information about hyperparameters/how to run can be found in **Section 3: Usage**. This section instead summarises the general architecture of the system and the differences between the baseline and real model. For tabular summary of architecture, skip to **Section 2.3**.
### 2.1 DCGAN (Baseline)
For the baseline implementation a DCGAN was created using only the standard hyperparameters from the DCGAN paper. The generator goes through 4 layers of convolution, starting from latent space vector *z* and outputting a 128x64 image, matching the training images (see **Section 4.2** about pre-process downsampling). Batch normalisation is done between convolutions with ReLU activation functions being used until after the final convolution, which is a Tanh activation. The discriminator follows the same architecture in reverse, taking in a 128x64 image, and outputting a real or fake label. The key difference between the two is in the activation functions, as the discriminator uses LeakyReLU with a Sigmoid after the final convolution. BCELoss was utilised in the baseline model.
### 2.2 WGAN-GP
WGAN-GP maintains most of the same architecture as the DCGAN in its generator and critic, with the main differences being in the critic, which no longer has a final Sigmoid activation function and uses group normalisation instead of batch. We no longer use the sigmoid because we want the raw critic score rather than the probability. As for the normalisation, WGAN-GP uses gradient penalty, which is based on one image at a time, but batch normalisation normalises each image by the mean and variance of the whole batch, which breaks the gradient penalty.

Other major changes between the DCGAN and the WGAN-GP surround the loss function. While DCGAN uses BCELoss, WGAN-GP *must* use Wasserstein loss (which has separate equations for the loss of the critic vs loss of the generator). In addition, the critic loss also includes the gradient penalty component (the GP portion of WGAN-GP). Lipschitz constraint is obeyed through the usage of the gradient penalty, making this model a WGAN.

### 2.3 Architecture
**Generator** (identical in DCGAN/WGAN-GP)

<div align="center">

| Layer | Operation | Output shape | Parameters |
|---|---|---|---|
| Input | Latent vector z ~ N(0, I) | 100 × 1 × 1 | – |
| 1 | ConvTranspose (8×4, stride 1) → BatchNorm → ReLU | 512 × 8 × 4 | 1,639,424 |
| 2 | ConvTranspose (4×4, stride 2) → BatchNorm → ReLU | 256 × 16 × 8 | 2,097,664 |
| 3 | ConvTranspose (4×4, stride 2) → BatchNorm → ReLU | 128 × 32 × 16 | 524,544 |
| 4 | ConvTranspose (4×4, stride 2) → BatchNorm → ReLU | 64 × 64 × 32 | 131,200 |
| 5 | ConvTranspose (4×4, stride 2) → Tanh | **1 × 128 × 64** | 1,024 |
| | | **Total** | **4,393,856** |

</div>

**Critic** (different in DCGAN/WGAN-GP)

<div align="center">

| Layer | Operation | Output shape | Parameters |
|---|---|---|---|
| Input | Real or generated slice | 1 × 128 × 64 | – |
| 1 | Conv (4×4, stride 2) → LeakyReLU(0.2) | 64 × 64 × 32 | 1,024 |
| 2 | Conv (4×4, stride 2) → Norm* → LeakyReLU(0.2) | 128 × 32 × 16 | 131,328 |
| 3 | Conv (4×4, stride 2) → Norm* → LeakyReLU(0.2) | 256 × 16 × 8 | 524,800 |
| 4 | Conv (4×4, stride 2) → Norm* → LeakyReLU(0.2) | 512 × 8 × 4 | 2,098,176 |
| 5 | Conv (8×4, stride 1) | **1 × 1 × 1** | 16,384 |
| | | **Total** | **2,771,712** |

</div>

\* **Norm** is `BatchNorm2d` in the DCGAN discriminator and `GroupNorm(1, C)` (layer
normalisation) in the WGAN-GP critic. The DCGAN discriminator additionally ends with a
Sigmoid to output a probability; the WGAN-GP critic outputs an unbounded score.

## 3. Usage
### 3.1 Dependencies
All code was run with **Python 3.13** on Windows 11 with an **NVIDIA GeForce RTX 3080** (CUDA 13.1).

Third-party packages:

<div align="center">

| Package | Version | Used by |
|---|---|---|
| torch | 2.14.0+cu130 | all model, training and evaluation code |
| torchvision | 0.29.0+cu130 | `train.py`, `predict.py` (saving image grids) |
| torchmetrics | 1.9.0 | `utils.py` (SSIM), `train.py` (FID) |
| torch-fidelity | 0.4.0 | required by torchmetrics' FID (Inception weights) |
| scipy | 1.18.1 | required by torchmetrics' FID |
| nibabel | 5.4.2 | `dataset.py`, `data_audit.py` (loading NIfTI slices) |
| numpy | 2.5.2 | `dataset.py`, `data_audit.py` (preprocessing / audit only) |
| matplotlib | 3.11.2 | `train.py` (loss, metric and histogram plots) |

</div>

Imports per file (standard-library modules in *italics*):

| File | Purpose | Imports |
|---|---|---|
| `modules.py` | Generator, DCGAN discriminator, WGAN-GP critic, weight initialisation | `torch.nn` only |
| `dataset.py` | `HipMRIDataset` (crop or downsample, clipping, scaling), `load_data` (DataLoader), `load_split` (whole split as a tensor) | *os*, *glob*, torch, `torch.utils.data`, `torch.nn.functional`, nibabel, numpy |
| `utils.py` | Shared evaluation helpers: `load_generator`, `invert_ssim`, `compute_fid`, `nearest_train`, `to_rgb_01`; `BEST_CHECKPOINTS`, `NUM_SAMPLES` | torch, `modules`, torchmetrics (SSIM) |
| `train.py` | Training (DCGAN, WGAN-GP), checkpoint validation, final testing, loss and metric plots; run configurations in `CONFIGS` | *os*, *json*, *time*, *argparse*, *pathlib*, torch (`nn`, `optim`), torchvision.utils, matplotlib, torchmetrics (FID), `dataset`, `modules`, `utils` |
| `predict.py` | Example usage of a trained model: samples, inversion reconstructions, nearest-neighbour pairs, latent interpolation, printed results | *json*, *argparse*, *pathlib*, torch, torchvision.utils, `dataset`, `train`, `utils` |
| `data_audit.py` | Standalone data audit (Section 1.3) | *os*, *re*, *collections*, nibabel, numpy |

### 3.2 Hyperparameters
Three configurations are compared; each is an entry in `CONFIGS` in `train.py`. First appearance of
difference in configuration of hyperparameters are bolded within the table.

| Hyperparameter | DCGAN (`dcgan`) | WGAN-GP standard (`wgan_gp`) | WGAN-GP tuned (`wgan_gp_wide`) |
|---|---|---|---|
| **Data & architecture** | | | |
| Preprocessing | Central 128×64 crop | Central 128×64 crop | **Whole slice, downsampled to 128×64** |
| Image size | 1 × 128 × 64 | 1 × 128 × 64 | 1 × 128 × 64 |
| Latent dimension (nz) | 100 | 100 | **128** |
| Generator / critic base width (ngf / ndf) | 64 / 64 | 64 / 64 | **128** / 64 |
| Generator parameters | 4,393,856 | 4,393,856 | **15,210,240** |
| Generator normalisation | BatchNorm | BatchNorm | BatchNorm |
| Discriminator / critic normalisation | BatchNorm | **GroupNorm(1, C) (layer norm)** | GroupNorm(1, C) (layer norm) |
| Discriminator / critic output | Sigmoid (probability) | **Linear (unbounded score)** | Linear (unbounded score) |
| Weight initialisation | N(0, 0.02) | N(0, 0.02) | N(0, 0.02) |
| **Objective** | | | |
| Loss | BCE (non-saturating G loss) | **Wasserstein + gradient penalty** | Wasserstein + gradient penalty |
| Gradient penalty weight (λ) | – | **10** | 10 |
| Critic updates per G update (n_critic) | 1 | **5** | 5 |
| **Optimisation** | | | |
| Optimiser | Adam | Adam | Adam |
| Learning rate (G and D) | 2 × 10⁻⁴ | **1 × 10⁻⁴** | 1 × 10⁻⁴ |
| Adam (β₁, β₂) | (0.5, 0.999) | **(0.0, 0.9)** | (0.0, 0.9) |
| Batch size | 128 | 128 | 128 |
| **Training length** | | | |
| Epochs | 50 | **250** | 250 |
| Generator updates per epoch | 90 | **18** | 18 |
| Total generator updates | 4,500 | 4,500 | 4,500 |
| Total critic / discriminator updates | 4,500 | **22,500** | 22,500 |
| Selected checkpoint (lowest validation FID) | epoch 50 | epoch 240 | epoch 250 |

The tuned configuration changes two things relative to the standard WGAN-GP: a wider generator
(more capacity) and downsampled data instead of the central-cropped data. Its results are therefore evaluated 
against equally downsampled test data and are not directly comparable with the centre-crop models.

### 3.3 Reproducing Results
**Data.**

Adjust `ROOT` in `dataset.py` to match the location of your base HipMRI folder. Ensure that 
`HipMRI/keras_slices_train`, `HipMRI/keras_slices_validate` and `HipMRI/keras_slices_test` all exist 
within the main HipMRI folder you direct `ROOT` to.

**Commands.**

Run from the project folder. `--run` selects a configuration from `CONFIGS`. The `CONFIGS` dictionary can be adjusted 
for chosen hyperparameters and can take an additional entry of `resume=[checkpoint_file]` if it desired to continue
training a previously trained model from a checkpoint.
All outputs go to `outputs/<run>/`.

| Step | Command | Output |
|---|---|---|
| Data audit | `python data_audit.py` | Printed split counts, shapes, intensities, patient overlap |
| Train | `python train.py --run dcgan --mode train`<br>`python train.py --run wgan_gp --mode train`<br>`python train.py --run wgan_gp_wide --mode train` | `checkpoints/` (every 25 epochs), `samples/epoch_NNN.png`, loss/metric plots, `resources.txt` |
| Validate (choose checkpoint) | `python train.py --run <run> --mode validate` | Validation FID per checkpoint, best checkpoint printed; `val_results_<run>.json` |
| Test (final values) | `python train.py --run <run> --mode test` | Test FID (with real-data floor), inversion SSIM, nearest-neighbour statistics; `test_results.json`, `nn_hist_<run>.png` |
| Visualise | `python predict.py --run <run>` | Loads the run's saved generator weights and the test split, then saves seeded sample grids, a latent interpolation, GAN-inversion reconstructions (with SSIM) and nearest-neighbour pairs.<br>Also prints the run's `test_results.json`. |

After validating, set the chosen file for each run in `BEST_CHECKPOINTS` (`utils.py`) before
testing or running `predict.py`.

**Notes on reproduction.**
- Evaluation is seeded to ensure fair comparisons between models.
- To visualise/generate outputs for a certain saved model, go into `utils.py` and adjust the checkpoint
  for the chosen model inside `BEST_CHECKPOINTS`.
- The results in this report were produced with the pre-restructure versions of these scripts
  (see redudnancy folder and commit history). New files adjusted to meet the spec requirements
  all received short test runs to ensure identical functionality.

## 4. Data
### 4.1 Dataset & Data Audit
The provided dataset is already split into separate training, validation and testing sets. In this case, an audit was done on the provided sets to ensure they were appropriately split with usable data (i.e. no patient leakage between sets and no near-empty/dark images) as seen in **Section 1.3**. The audit found no patient leakage between sets as well as 0 near-empty slices in the provided files, suggesting the test/train/split was valid. For this reason, there is no leakage detection/prevention functionality within `dataset.py` In addition to validating the dataset's usability for the task, the audit results were used to help determine what kind of pre-processing should be completed on the dataset.
### 4.2 Pre-Processing
- **Centre Crop:** In order to avoid dropping any data from the training set, a centre crop was performed to resize the 60 outlying 256x144 images that were found during the data audit.
> [!NOTE]
> Due to a bug in `dataset.py`, a 128x64 crop was applied directly to the dataset images for the `dcgan` and `wgan_gp` runs instead of cropping to 256x128 and then downsampling. Comparison will be completed between these two models as they were trained on the same type of pre-processed data, but note that only `wgan_gp_wide` had cropping applied as described above.
- **Scaling to [-1, 1]:** Due to the end of the generator architecture using a tanh activation function, intensity values must be min-max scaled to a range of [-1, 1] to be able to pass through it.
- **99.5th-Percentile Clipping:** The data audit revealed the training data featured intensity values up to 1,150, however, the median highest-intensity value seen in training images was only 387.5. Since the data is min-max scaled from [-1, 1], any training scans that feature outlier high-intensity values will define the top of the scaling range for that slice. This would mean that the other lower-intensity pixels would be scaled down and squashed into a narrower, darker band of intensity values. To prevent this, intensity values for images are clipped to the 99.5th percentile.
- **Downsampling to 128x64:** The network architecture in the DCGAN/WGAN papers was designed around a 64x64, so using the full resolution image would require an additional layer in both the generator and critic networks. Additionaly, downsampling reduces the total pixel count of the images, lowering expected memory usage and training time. While the 2:1 aspect ratio is preserved, it is expected that this pre-processing decision will prevent the generator from learning some of the finer details of the data.
> [!NOTE]
> Due to a bug in `dataset.py`, only `wgan_gp_wide` has a downsample applied from 256x128 to a 128x64 image (see Centre Crop note for additional details).

### 4.3 Splits
**Train (91%) / Validate (5%) / Test (4%) Split**
- **Train:** For fitting the generator and critic (learning the weights).
- **Validate:** FID verification is done against saved checkpoints for each model to find the best performing checkpoint (which will be used for final evaluation).
- **Test:** Used with the final model, checking SSIM, FID and memorisation audit (nearest neighbours) acceptance criteria are met.

## 5. Experiments
### 5.1 Training Metrics
All three models received 4,500 generator updates. Note that the x-axes differ: the DCGAN
plot is in generator iterations (one discriminator update per generator update), while the
WGAN-GP plots are in critic iterations (five critic updates per generator update, so 22,500
critic iterations = 4,500 generator updates).

<div align="center">

![DCGAN Losses](images/dcgan//dcgan_losses.png)
### Figure 2: DCGAN Baseline Losses per Iteration

</div>

After a very unstable first 100 or so iterations, the generator loss settles around roughly 4, while the
discriminator loss falls to around 0.2; this indicates the discriminator increasingly
dominates. Instability spikes appear to occur every 150–350 iterations, which are cases of 
where the discriminator is briefly fooled and both losses jump before recovering.
Both losses are flat after around 1,500 iterations, so there is no further indication of whether sample quality
is still improving.

<div align="center">

![WGAN-GP standard training curves](images/wgan_gp/wgan_gp_curves.png)
### Figure 3: WGAN-GP Standard Losses per Iteration

</div>

The gradient penalty falls from 30 to approximately 0.5 within the first few hundred critic iterations
and stays there for the rest of training, showing the Lipschitz constraint is being enforced.
The Wasserstein estimate rises sharply to around 50 as the critic first learns to separate real from
generated slices, drops to roughly 10 as the generator catches up, and then rises to a plateau of 15–18
from ~4,000 critic iterations onward. No instability spikes comparable to DCGAN occur. However,
the W estimate does not decrease steadily over training as reported by Gulrajani et al., suggesting
there are no further improvements in sample quality.

<div align="center">

![WGAN-GP tuned training curves](images/wgan_gp_wide/wgan_gp_wide_curves.png)
### Figure 4: WGAN-GP Tuned Losses per Iteration

</div>

The gradient penalty behaves very similary to the standard WGAN run, dropping quickly to a low penalty and remaining there for the duration of training. The Wasserstein estimate starts much higher, then falls and settles into a narrower band than the standard run. While the W values cannot be directly compared since the critic's scale depends on the data, it suggests that the critic estimates are steadier.

### 5.2 Generated Samples
All grids below are uncurated, and display the first 32 samples produced from random seed 1, with no selection. Additional seed 2 and 3 images are available within the respective model subfolder of `images/`.

<div align="center">

| Real Test Slices (Centre Crop) | DCGAN |
|---|---|
| ![Real Seeded Images](images/wgan_gp/grid_real_test.png) | ![DCGAN Seed 1](images/dcgan/grid_dcgan_seed1.png) |
### Figure 5: Real Images vs Generated DCGAN Images

</div>

The DCGAN image generations appear to have relatively sharp and clean boundaries, with nice contrasting edges between bright/dark areas. Additionally, the bladders are smooth, with similarity in shape and brightness to the real images. However, the global anatomical layout is lacking, with no consistent arrangement of organs and little symmetry between the left and right sides of the pelvis. Some images appear to be mostly texture (row 2, column 1-3) with no distinct layout at all.

<div align="center">

| Real Test Slices (Centre Crop) | WGAN Standard |
|---|---|
| ![Real Seeded Images](images/wgan_gp/grid_real_test.png) | ![WGAN-GP Seed 1](images/wgan_gp/grid_wgan_gp_seed1.png) |
### Figure 6: Real Images vs Generated Standard WGAN-GP Images

</div>

The standard WGAN-GP is able to reproduce global intensity and contrast, and places a bright bladder-like structure to the right in approximately half of its samples. However, it is not able to reproduce anatomical structure, as the bladder boundaries are irregular and surrounding tissue can have swirling patterns instead of distinct muscles/organs. Some samples contain no recognisable structure, but all samples vary noticably in layout with no obviously similar compositions. Compared to the DCGAN, this generator produces blurrier, streaky edges and a more irregular bladder appearance, suggesting that the baseline model has outperformed the standard WGAN-GP implementation.

<div align="center">

| Real Test Slices (Downsampled) | WGAN-GP (Tuned) |
|---|---|
| ![Real Full Seeded Images](images/wgan_gp_wide/grid_real_test.png) | ![WGAN-GP Tuned Seed 1](images/wgan_gp_wide/grid_wgan_gp_wide_seed1.png) |
### Figure 7: Real Images vs Generated Tuned WGAN_GP Images

</div>

The global layout of the generated samples are improved in the tuned WGAN-GP, with roughly the right body width, and a bladder centred in the pelvis, which is a direct improvement when compared to the other two models. Additionally, the layouts are diverse, with relatively plausible bladders that vary in brightness and size. There are still some aspects missing, including a need for clearer structure (tissue between the outline and the bladder is streaky) and a lack of contrast and sharpness (some images appear fuzzy).


### 5.3 Tuned WGAN-GP
After the first two models were trained using a cropped version of the dataset, the final version of the WGAN was
trained using images cropped to 256x128 and then downsampled to 128x64. Additionally, the latent dimension and the generator feature 
widith were both increased to 128, as it was believed that more dimensions and channels would allows the generator to more easily 
separate aspects of a slice (i.e. bladder size, position, tissue) and more easily represent distinct features, respectively.

## 6. Results
### 6.1 Quantitative Benchmark
All metrics are computed on the held-out test split (540 slices, 3 patients) using each run's
checkpoint selected by lowest validation FID. 540 samples are generated per model with the same
latent noise (seed 0). Each model is evaluated against test and training data preprocessed the
same way it was trained.

| Metric | DCGAN | WGAN-GP (standard) | WGAN-GP (tuned) |
|---|:---:|:---:|:---:|
| Data | Centre crop | Centre crop | Whole slice, downsampled |
| Checkpoint (generator updates) | Epoch 50 (4,500) | Epoch 240 (4,320) | Epoch 250 (4,500) |
| **Fidelity** | | | |
| FID ↓ | **177.5** | 271.4 | 204.0 |
| Real-data FID floor | 79.8 | 79.8 | 70.4 |
| FID above floor ↓ | **97.7** | 191.7 | 133.6 |
| Inversion SSIM ↑ (mean ± std) | 0.374 ± 0.072 | 0.364 ± 0.078 | **0.547 ± 0.037** |
| Test slices with inversion SSIM > 0.6 | 0% (0 / 540) | 0% (0 / 540) | **11.1% (60 / 540)** |
| Meets SSIM > 0.6 target (mean) | ✗ | ✗ | ✗ |
| **Memorisation audit** | | | |
| Median NN RMSE to training set (generated) | 0.467 | 0.445 | 0.333 |
| Median NN RMSE to training set (real test) | 0.356 | 0.356 | 0.315 |
| Generated / real NN ratio | 1.31 | 1.25 | 1.06 |
| Samples closer than 99% of real test slices (expected ≈ 1% by chance) | 0.2% (1 / 540) | 0.4% (2 / 540) | 2.4% (13 / 540) |

Between the two centre-crop models, DCGAN achieves a much better FID than the standard WGAN-GP (97.7 versus 191.7 above real data floor), but both achieve a very similar SSIM. This indicates that neither generator can properly represent unseen patient anatomy well. While both models were compared against test images cropped in the same way, a likely cause of the very low SSIM score is the cropping of the images. Meanwhile the tuned WGAN-GP reaches an average SSIM of 0.547, with 11.1% of slices exceeding 0.6. While this also does not meet the requirements for generated data, it is much closer in proximity to the > 0.6 requirement, with only changes in image cropping and latent space/feature size changes having been made. This suggests that there is a likely chance further tuning and exploration of the WGAN-GP model could meet the requirements for generated patient images.

### 6.2 Inverse SSIM
MRI images are too different in structure (varied anatomy) to obtain a valid SSIM score by comparing a generated image with a set of test images and returning the maximum SSIM score. Due to this, GAN inversion is used to "reconstruct" test slices. This is achieved by optimising a latent vector z so that G(z) matches the slice, and then the SSIM is computed between the slice and its reconstruction.

<div align="center">

![DCGAN Inversion](images/dcgan/inversion_DCGAN.png)


![Standard WGAN-GP Inversion](images/wgan_gp/inversion_wgan_gp.png)

![Tuned WGAN-GP Inversion](images/wgan_gp_wide/inversion_wgan_gp_wide.png)
### Figure 8: DCGAN (Top) Standard-WGAN (Middle) Tuned WGAN-GP (Bottom) Inversion

</div>

The reconstruction (test slices on top, inversions on bottom), show that all generators can recover a coards structure of bladder position, brightness levels, and primary dark/bright regions. The tuned model accurately reproduces the the body outline, background and bladder, which is a likely cause of the higher SSIM score, but the internal tissue is smoothed out and low in contrast. DCGAN's reconstructions are visibly sharper than those of the standard WGAN-GP, but the achieved SSIM is similar, suggesting that DCGAN may face more misalignments in texture despite being sharper in appearance.

### 6.3 Memorisation Audit
The memorisation images below show the 8 generated samples (top row) from each model that were closest to images in the training set (bottom row), illustrating the worst cases of memorisation discovered.

<div align="center">

![DCGAN NN](images/dcgan/nn_pairs_DCGAN.png)


![Standard WGAN-GP NN](images/wgan_gp/nn_pairs_wgan_gp.png)

![Tuned WGAN-GP NN](images/wgan_gp_wide/nn_pairs_wgan_gp_wide.png)
### Figure 9: DCGAN (Top) Standard-WGAN (Middle) Tuned WGAN-GP (Bottom) NN Slices

</div>

In all 3 cases, matches appear to come from the dominant shape being similar, rather than because the internal anatomy of the patients is directly copied. More specifically, each image features a large, bright bladder in the same position, making it dominating feature across the image. Thus, "similar" generated/real pairs are those where the generated bladder happens to be large, rather than due to copied anatomy across the whole image (i.e. generated tissue, muscles and bones differ).

The tuned model has the highest percentage of samples below the memorisation threshold (2.4%, when the expected would be 1% with random chance). To properly analyse why this may be, we can take a look at the histogram produced by the `test` function in `train.py`. The histogram compares 540 real test slices to 540 generated images, and for each generated image it finds the most similar training slice and measures how different they are with average pixel difference. A small number is located on the left-hand side, and indicates that an image is very similar to a training slice, while the right-hand side means they are less similar.

<div align="center">

![Tuned WGAN-GP NN Histogram](images/wgan_gp_wide/nn_hist_wgan_gp_wide.png)
### Figure 10: Tuned WGAN-GP NN Histogram

</div>

The generated distribution overlaps the real distribution but is centred slightly to the right of it, suggesting that generated images are actually slightly less like any real slice. There are still very few images are to the left of the percentile line marker, and they are only more similar to train images by a small margin. Moreover, the generated images actually have more images farther right than the test images, so it is likely that the increased similarites occured due to very similar global anatomy, rather than due to any direction training memorisation.

### 6.4 Latent Space Interpolation

<div align="center">

![DCGAN Interp](images/dcgan/interpolation_dcgan.png)


![Standard WGAN-GP Interp](images/wgan_gp/interpolation_wgan_gp.png)

![Tuned WGAN-GP Interp](images/wgan_gp_wide/interpolation_wgan_gp_wide.png)
### Figure 11: DCGAN (Top) Standard-WGAN (Middle) Tuned WGAN-GP (Bottom) Interpolations

</div>

All 3 images show smooth changes from one side to the other, indicating that the generator in all cases has managed to learn a continuous latent space rather than memorising individual training slices. No frames degrade into noise or artefacts midway though interpolation, and can be considered "plausible" transitions within the limitations of each model (i.e. ignoring the overall anatomical quality/correctness of the images). DCGAN has the largest structural change of the models, and while the transition is smooth and the image is symmetrical, it breaks up the tissue into an anatomically inplausible structure on the right side. Meanwhile the standard WGAN-GP maintains a consistent bladder shape throughout, without the tissue surround the bladder shifting with it accordingly. In this instance, the tuned WGAN-GP images are rather low in detail with less obvious constrast; a bladder appears to be form in the correct location towards the right side. While the tuned model does not perform particularly well in this instance, this is only a single interpolation and does not illustrate the overall quality of the generator (it mainly demonstrates the lack of memorisation and a continuous latent space).

### 6.5 Resource Profiling

<div align="center">

| | DCGAN | WGAN-GP (standard) | WGAN-GP (tuned) |
|---|:---:|:---:|:---:|
| Generator parameters | 4,393,856 | 4,393,856 | 15,210,240 |
| Discriminator / critic parameters | 2,771,712 | 2,771,712 | 2,771,712 |
| Peak training VRAM | 616 MB | 914 MB | 1,148 MB |
| Epochs | 50 | 250 | 250 |
| Generator updates per epoch | 90 | 18 | 18 |
| Mean epoch time | 24.1 s | 26.2 s | 32.4 s |
| Time per generator update | 0.27 s | 1.46 s | 1.80 s |
| Total training time (4,500 generator updates) | ~20 min | 109 min | 135 min |
| Time to generate 64 images | 2.5 ms | 2.3 ms | 6.4 ms |

</div>

Per-epoch, each generator takes relatively the same amount of time, however due to the fact that the WGAN-GP generator only updates for every 5 critic runs, it must run a significantly larger number of epochs to achieve the same amount of generator iterations. There is a clear increased in peak VRAM across the 3, as the gradient penalty's second backward pass in WGAN-GP requires more memory than the single BCELoss in the DCGAN. Tuned WGAN-GP sees another (anticipated) increase as it adds additional features and latent space to the generator. Overall all models fall far below the expected maximum allowable memory and time from the feasibility report (4GB VRAM and 6h training).

## 7. Failure Analysis
**Observation:** Centre-crop samples contain realistic elements (bladder-like shapes, tissue boundaries) but arrange them without anatomical coherence. There is no left-right symmetry or consistent organ layout and some sample are almost all texture (see Section 5.3 DCGAN/Standard WGAN-GP for images).
**Diagnosis:** The central 128x64 image crop removes landmarks that anchor the pelvis layout such as the body outline and fermoral heads. Low inversion SSIM (approximately 0.37) in both confirms they cannot accurately represent unseen anatomy.
**Evidence of Cause:** The tuned model with whole slices better reproduces the global layout and raises inversion SSIM to 0.55 (although increase in generator features may also be a contributor).

**Observation:** In the inversion figures across all models, slices with a large, smooth bladder were reconstructed well, while those with smaller, complex structures became blurred/streaked with less clear boundaries and texture (see Section 6.2).
**Diagnosis:** The generators have all been able to learn large-scale patterns better than the fine anatomy.
**Mitigation:** Higher image resolution usage, adaption in architecture (to support larger images or introduce ResNets to preserve fine image details).

**Observation:** Tuned samples have a plausable outline, but the top and bottom margins are not sharp, and many images have low contrast internally with mid-grey tissue and anatomical features, sometimes appearing nearly featureless (see Section 5.3).
**Diagnosis:** Downsampling halves resolution, so fine internal structures are underrepresented in the training data. The highly contrasting outlines are learned more easily due to their size even at lower resolution.
**Implication:** Global layout is now correct, so the remaining fidelity gap lies in the internal anatomy. Architecture, capacity and resolution improvements should be aimed at improving this aspect of generated images.

## 8. Recommendation
**Verdict**

At this stage, it cannot be recommended to use WGAN-GP to create a synthetic cohort of patient data as it did not meet the required evaluation metric (> 0.6 SSIM). However, more time should be spent tuning the WGAN-GP as it shows promise. Directly swapping from a standard DCGAN to a WGAN-GP is not recommended if further tuning is not done on the WGAN-GP model due to the higher time/memory computation budget.

<div align="center">

| Acceptance criterion (Section 1.1) | Result | Status |
|---|---|---|
| Inversion SSIM > 0.6 on test slices | Best: tuned WGAN-GP, 0.547 mean (11.1% of slices > 0.6) | ✗ Not met |
| WGAN-GP outperforms the DCGAN baseline | Standard: worse FID, equal SSIM. Tuned: higher SSIM (different preprocessing) | ~ Partially |
| No memorisation (NN audit) | No near-copies in any model; tuned closest (2.4% below threshold, layout-level similarity only) | ✓ Met |
| Peak VRAM < 4 GB, training < 6 h | Max 1,148 MB, max 135 min | ✓ Met |

</div>

At an equal budget of 4,500 generator updates, the WGAN-GP trained more stably than the DCGAN, with no instability spikes in the critic/generator loss and a stable gradient penalty over time. Before applying additional latent space/generator features, the WGAN-GP actually performed worse in FID, and had a relatively equal SSIM. The improvement in loss alone with the standard WGAN-GP cannot justify changing from the base model.

No model shows memorisation of training patient data, with the most similar cases occurring due to the same global layout rather than any direct copy of internal anatomty. This suggests that, with further improvements to the quality of image generation, the model may be able to be shared/utilised once all acceptance criteria are cleared.

### 8.1 Limitations
- **Small evaluation sets:** Validation and test contain only 4 and 3 patients, limiting how far
  results generalise and making FID and the memorisation threshold noisy.
- **Confounded tuned model:** Generator width and preprocessing changed together, so the tuned
  model's improvement cannot be attributed to either alone.
- **Different preprocessing:** The tuned model is evaluated on downsampled whole slices and the
  others on the central crop, so their metrics are not directly comparable.
- **Reduced data:** 2D slices only, at 128×64: the centre crop loses peripheral anatomy, and
  downsampling loses fine detail.
- **Metric limitations:** Inversion SSIM depends on the optimisation settings and mainly rewards large-scale
  structure (sharper DCGAN reconstructions scored no higher than blurrier WGAN-GP ones).
- **Checkpoint selection by FID only:** Other criteria (e.g. validation inversion SSIM) might
  select different epochs.


### 8.2 Future Work
- **Isolate the tuned model's gains:** Train a standard-width WGAN-GP on downsampled whole slices.
- **Stronger architectures:** Consider implementing a ResNet on the Generator/Critic to see if there are further improvements in generation quality (this was utilised in the WGAN-GP paper).
- **More robust evaluation:** KID alongside FID, repeated runs with different seeds, larger test
  sets, and checkpoint selection on multiple metrics.


## 9. AI Use Disclosure
This project was completed with assistance from **Claude (Anthropic)**, used as a tutor, code
reviewer and writing assistant throughout. I am responsible for all code, results and claims in
this report; every AI-assisted component was run, inspected and checked by me as described below.

**How AI was used**

| Area | AI contribution | My contribution and verification |
|---|---|---|
| **Concepts** | Explanations of WGAN/WGAN-GP theory (Wasserstein distance, Lipschitz constraint, gradient penalty, why BatchNorm is removed from the critic), DCGAN architecture, SSIM/FID, and PyTorch details (tensor shapes, `unsqueeze`, intensity scaling, BatchNorm train/eval behaviour). | Cross-checked against Arjovsky et al. (2017), Gulrajani et al. (2017) and Radford et al. (2016); used to make and justify my own design decisions. |
| **Feasibility & data audit** | Initial data-audit script and a draft structure for the feasibility review. | Rewrote the audit as `data_audit.py`, ran it on the real data, and wrote the review from its results. |
| **Data pipeline** | `HipMRIDataset` verification of code. | Through this, I identified that my crop setting was cropping every slice rather than only the oversized ones (documented in Section 4.2). |
| **Models** | Bug Identification. | The DCGAN is adapted from the PyTorch DCGAN tutorial; I wrote the WGAN-GP changes following Algorithm 1 of Gulrajani et al. and fixed the identified bugs; verified shapes and parameter counts. |
| **Training code** | Reviewed my training loops; helped debug errors. | Ran every training configuration myself; checked loss curves, samples and checkpoints; restructured the code into the required files and verified it on the real data. |
| **Evaluation** | Proposed and assisted in writing evaluation methods: best-match SSIM with a real-data ceiling check, GAN-inversion SSIM (after course staff guidance on paired SSIM), FID with a real-data floor, the nearest-neighbour memorisation audit, and latent interpolation. | Ran all evaluations, chose checkpoints on validation only, and inspected every figure; the ceiling check (real slices scoring 0.137) is my own measured result, which changed how SSIM was used. |
| **Report** | Suggested the README structure, formatted tables (architecture, hyperparameters, results, resources) from my numbers, suggested key content for sections. | Supplied all numbers and figures; verified suggested key content before I wrote my report sections. |

**Verification approach**
- All code was executed by me on the real HipMRI data; AI-debugged functions were also tested by the
  AI on synthetic data before use (e.g. planted near-copies for the memorisation audit, reachable
  vs unreachable targets for inversion).
- All reported numbers come from my own runs (`resources.txt`, `val_results_*.json`,
  `test_results.json`).


## 10. References
Arjovsky et al. 2017; Gulrajani et al. 2017; Radford et al. 2016;
PyTorch DCGAN tutorial;
