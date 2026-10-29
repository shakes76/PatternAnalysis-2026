# Improved 2D U-Net for Prostate Segmentation
Task 2.2.4 - Normal Difficulty

## Author
William Foley (47434785)
COMP3710 - Pattern Recognition and Analysis
The University of Queensland

## Overview
This project investigates automatic prostate segmentation from magnetic resonance imaging (MRI) using a baseline and advanced U-Net models. The implementation of a standard 2D U-net (based on Ronneberger et al. (2015)) and an improved 2D U-net (based on Isensee et al. (2018)) were compared. Both models are trained on the provided HipMRI dataset and evaluated using segmentation metrics, with particular attention to prostate Dice score and boundary errors.

The problem to be analysed is that automated segnementation trained purely on global dice scores can mask critical boundary failure modes. These failure modes may impact radiotherapy planning and therefore treatment accruacy. The model comparison will identify whether architectural improvements reduce segmentation failure cases sufficiently to justify increases in computational cost. Both models must also achieve a Target Dice ≥ 0.75 on prostate label the test set.

## How Algorithm Works
The U-Nets are convolutional neural networks designed for pixel-level image segmentation. The models use an encoder to reduce image resolution and extract features. The decoder upsamples these features to reconstruct a segmentation mask. Skip connections are used to recover spacial detail from the encoder. The improved model uses this structure but also introduces pre-activation residual blocks, strided convolutions, InstanceNorm, LeakyReLU and dropout to support feature learning and reduce overfitting. Refer to key features for more information. 

The image below shows an example of the 2D Improved U-Net model architecture developed by Isensee et al. (2018). The improved U-NET model implements most of this architecture. It differs to the reference by using billinear interpretation for upsampling instead of feature repetition followed by 3x3 convolution. 

![Original 3D U-Net architecture from Isensee et al. (2018)](Figures/isensee_architecture.png)


## Key Features 
| Feature | Baseline 2D U-Net | Improved 2D U-Net |
|---|---|---|
| Feature extraction | Double convolution blocks | Pre-activation residual blocks |
| Downsampling | Max pooling | Strided convolutions |
| Normalisation | BatchNorm | InstanceNorm |
| Activation | ReLU | LeakyReLU (slope = 0.01) |
| Regularisation | No dropout | Dropout2d (p = 0.3) |
| Upsampling | Bilinear interpolation | Bilinear interpolation |
| Skip connections | Encoder-decoder connections | Encoder-decoder and residual connections |
| Output | Six segmentation classes | Six segmentation classes |

## Feasibility Review

### User Need, Scope & Acceptance Criteria

Medical specialists require precise prostate contouring on MRI images for effective radiotherapy procedures. Automated segmentation trained purely on global dice scores can mask critical boundary failure modes. These failure modes can include contours extending into adjacent organs which can potentially compromise treatment planning.

This project will compare an improved 2D U-Net against a standard 2D U-Net baseline line model. The comparison will identify whether architectural improvement reduces segmentation failure cases sufficiently to justify increases in computational cost.

Acceptance Citeria:

Achieve prostate dice coefficient of at least 0.75 on the test data set.

Compare dice coefficients and Intersection-over-Union between models on the same test data set.

Calculate mean dice coefficient for prostate base, middle and apex slices separately. Report the difference in values to identify potential failure modes for both baseline and improved model.

Document 3-5 segmentation failures, showing ground-truth comparison, quantitative error measurement and anatomical significance

Compare peak GPU VRAM & per-case inference time for the baseline and improved 2D U-Net model.

### Model Choice & Course Concepts

The project uses the supplied HipMRI keras_slices_data dataset which provides separate training, validation & test folders containing compressed NifTi MRI slices and segmentation masks.

The proposed model is a 2D Improved U-Net [Normal Difficulty], following the architectural principles of Isensee et al. (2018), with standard 2D U-Net (Ronneberger et al., 2015) as the baseline.

Both models will utilise backpropagation, convolutional feature extraction, receptive field and hierarchical representation.

### Preliminary Feasibility Evidence

An initial data audit confirmed 12,660 MRI slices comprising of 11,460 training, 660 validation and 540 test images (90.5% train, 5.2% validation, 4.3% test) all with corresponding segmentation masks. No patient overlap between the three partitions.

Loading of a small MRI sample using the supplied NifTI-loading approach was successful.

The data set contains six segmentation labels (0-5) and all images besides 60 training images have dimension of 256x128 pixels. The 60 training images have dimensions of 256x144 pixels.

From data inspection label 0 appears correspond to outside the body, label 1 the soft tissue region, label 2 the pelvic bones, label 3 the bladder, label 4 the rectum and label 5 the prostate.

### Risks, Budget & fallback

Risks: The data set contain 60 images with inconsistent dimensions, requiring standardisation without compromising mask alignment. Both models lack interslice context so improvement in dice score may not translate to between clinical boundary accuracy. Label pixel occupation imbalance may reduce training effectiveness if all pixels are treated equally. Want to prioritise prostate label pixel accuracy.

Computing Budget: Training will use UQ's Rangpur GPU cluster.

The next experiment will be to standardise image dimension and run a short standard U-Net training experiment to verify decreasing loss and measure resource requirements.

If improved U-Net fails to improve boundary accuracy, the standard U-Net will be recommended as the more efficient alternative, supported by the experimental results.

## Dependencies
The project was developed in Python using PyTorch for neural network training and inference. Training was conducted on the University of Queensland's Rangpur computing cluster using GPU acceleration and a Conda environment named torch.

| Library | Version | Purpose |
|---|---|---|
| Python | 3.11.15 | Programming language |
| PyTorch | 2.13.0 | Neural network training and inference |
| NumPy | 2.4.6 | MRI preprocessing and numerical operations |
| NiBabel | 5.4.2 | Loading NIfTI MRI images and segmentation masks |
| Matplotlib | 3.11.2 | Training curves and segmentation visualisations |
| tqdm | 4.70.1 | Progress bars |
| CUDA | 13.0 | GPU acceleration |

All Python dependencies, including supporting CUDA libraries, are recorded with exact versions in `requirements.txt`.

### Project Structure
```text
2D_U_net_47434785/
├── dataset.py             # Dataset loading and preprocessing
├── modules.py             # Baseline and Improved U-Net architectures
├── train.py               # Model training and validation
├── predict.py             # Test evaluation and visualisations
├── train_unets.sbatch     # Rangpur GPU training script
├── requirements.txt       # Exact Python dependency versions
├── Figures/               # Architecture diagrams
└── README.md              # Project documentation
```

Model checkpoints and generated results are saved under `runs/baseline/` and `runs/improved/`.

### Installation, Training & Predicting

Python 3.11 and a compatible GPU environment are recommended. The project was trained using a Conda environment named `torch`.

Miniconda must first be installed. On Rangpur, where Miniconda is installed in the user's home directory, Conda can be initialised using

```bash
source ~/miniconda3/bin/activate
```

Create and activate the `torch` environment with Python 3.11.15:

```bash
conda create -n torch python=3.11.15 -y
conda activate torch
```

Install the Python Dependencies:

```bash
python -m pip install -r requirements.txt
```


The project uses the HipMRI dataset available on Rangpur at:

```text
/home/groups/comp3710/HipMRI_Study_open/keras_slices_data
```

The `DATA_ROOT` variable in `train.py` and `predict.py` must point to the dataset location.

The dataset must contain the following directories:

```text
keras_slices_data/
├── keras_slices_train/
├── keras_slices_seg_train/
├── keras_slices_validate/
├── keras_slices_seg_validate/
├── keras_slices_test/
└── keras_slices_seg_test/
```

To train the models on Rangpur, navigate to the project directory and submit the Slurm batch script:

```bash
cd ~/2D_U_net_47434785
sbatch train_unets.sbatch
```
The batch script automatically activates the existing `torch` environment and executes `train.py` using the allocated GPU resources.

**Note:** The batch script assumes an environment named `torch` already exists. If a newly created environment uses a different name, update the `conda activate torch` line in `train_unets.sbatch` accordingly.

Alternatively, training can be executed directly on a local computer or allocated GPU compute node within an activated environment using:

```bash
python train.py
```

The best-performing checkpoint for each model is saved according to validation prostate Dice:

```text
runs/baseline/best_model.pt
runs/improved/best_model.pt
```

To evaluate the trained checkpoints:

```bash
python predict.py
```

The scripts save model checkpoints, training curves and test predictions to their respective directories under `runs/`.
The initial feature channel settings in `predict.py` must match those used to generate each saved checkpoint.

### Reproducibility

Both models used the same predefined, patient-independent HipMRI dataset partitions: 11,460 training slices, 660 validation slices and 540 test slices.

The following hyperparameters were used:

| Parameter | Baseline U-Net | Improved U-Net |
|---|---|---|
| Training epochs | 30 | 30 |
| Initial feature channels | 32 | 32 |
| Batch size | 8 | 8 |
| Optimiser | Adam | Adam |
| Learning rate | 0.0001 | 0.0001 |
| Loss function | Cross-entropy + Dice | Cross-entropy + Dice |
| Data augmentation | Brightness and contrast | Brightness and contrast |
| Model selection | Best validation prostate Dice | Best validation prostate Dice |

All MRI slices were Z-score normalised and centre-padded to 256 × 144 pixels. Random brightness and contrast adjustments were applied only during training, with no augmentation during validation or testing.

The best checkpoint was selected using validation prostate Dice to avoid selecting models based on test-set performance.

Random seeds and deterministic CUDA operations were not explicitly configured in the recorded training scripts. Consequently, results may vary between training runs due to random initialisation, data shuffling, augmentation and GPU computation. Exact reproduction of the reported Dice scores is therefore not guaranteed, even when using the same dependencies and hyperparameters.


## Example Input, Output & Training plots

### Input
The models take a single-channel, Z-score normalised 2D MRI slice as input. 

![Example Model Input](Figures/Example_input.png)

### Output

The figures below show the predicted segmentations produced by the baseline and improved U-Nets on the same held-out test image. Each figure displays the original MRI, the ground-truth segmentation and the corresponding model prediction. Purple area is what the model is predicting as the prostate.

![Example Baseline Model Output](Figures/Example_baseline_output.png)

![Example Improved Model Output](Figures/Example_improved_output.png)

### Training & Validation Dice Scores & Loss
The following graphs show the prostate Dice coefficient and the combined cross-entropy and Dice loss across training epochs for both models. Training was performed on the Rangpur GPU cluster using an NVIDIA A100-PCIE-40GB GPU.

| Baseline U-Net | Improved U-Net |
|---|---|
| ![Baseline Dice curve](Figures/baseline_dice.png) | ![Improved Dice curve](Figures/improved_dice.png) |



| Baseline U-Net | Improved U-Net |
|---|---|
| ![Baseline loss curve](Figures/baseline_loss.png) | ![Improved loss curve](Figures/improved_loss.png) |

Both models demonstrate successful optimisation. To elaborate dice scores increase and training loss decreases throughout the 30 epochs. 

Both architectures achieved comparable validation performance, suggesting that the additional complexity of the improved U-Net did not produce a substantial improvement in generalisation.

The increasing gap between training and validation performance suggests some degree of overfitting. This may be influenced by the relatively small number of independent training patients (31), despite the large number of MRI slices.

 

## DataSet and Preprocessing
The dataset refers to the HipMRI keras_slices_data dataset, provided by UQ at /home/groups/comp3710/HipMRI_Study_open/keras_slices_data on the Rangpur computing cluster. 

### Train/Validation/Test Split
The dataset contains 12660 MRI slices and a pre-defined split as follows:
- Training: 11,460 slices
- Validation: 660 slices
- Test: 540 slices

This approximates to a split of 90/5/5 for training, validation,test for 2D slices. This maximises the data available for model training however limits the generalisability of results as the validation and test sets contain only four and three patients respectively. 

This split maintains patient independence (no patient overlap between data split) which mitigates data leakage as the model is less likely to encounter highly similar patient anatomy during training and testing. 

The dataset contains six segmentation labels (0-5). Corresponding segmentation masks were provided for each MRI slice. 

### Data Loading
load_data_2D() function adapted from Appendix B of the COMP3710 assignment (The University of Queensland, 2026) was used to load the provided NifTI files as NumPy arrays. 

Images are matched with their segmentation mask using their filenames. 

The HipMRIDataset class loads individual MRI images and their corresponding masks when requested, rather than loading the entire dataset into memory. The NumPy arrays are converted into PyTorch tensors for use in neural network training.

The make_dataloaders() function creates separate PyTorch DataLoaders for training, validation and testing (PyTorch contributors, n.d.). Training images are shuffled, while validation and testing images are not.

### Preprocessing 
MRI Intensity Normalisation: Each MRI slice undergoes Z-score normalisation using the method provided in Appendix B of the COMP3710 assignment (The University of Queensland, 2026). This standardises MRI intensity values to approximately zero mean and unit standard deviation, reducing intensity variations between scans. 

Image Dimension Standardisation: The dataset contains 12,600 images measuring 256 × 128 pixels and 60 measuring 256 × 144 pixels. All images and corresponding masks are centre-padded to 256 × 144 pixels to ensure consistent dimensions while preserving anatomical proportions and mask alignment.

Segmentation Mask Encoding: The to_channels() function adapted from Appendix B converts segmentation masks into six-channel one-hot representations. A fixed number of six channels is used because individual slices do not always contain every anatomical label.

Data Augmentation: Small random brightness and contrast adjustments are applied to training MRI images to encourage generalisation and reduce overfitting. These adjustments are not applied to segmentation masks or to validation and test images.

## Investigation
Both models were evaluated on the same heldout HipMRI test split using identical preprocessing and evalutation procedures. 

### Quantitative Results
| Class | Baseline Dice | Improved Dice | Baseline IoU | Improved IoU |
|---|---:|---:|---:|---:|
| 0 | 0.9968 | 0.9967 | 0.9936 | 0.9934 |
| 1 | 0.9843 | 0.9847 | 0.9691 | 0.9699 |
| 2 | 0.9329 | 0.9360 | 0.8742 | 0.8798 |
| 3 | 0.9668 | 0.9642 | 0.9358 | 0.9309 |
| 4 | 0.8633 | 0.8605 | 0.7594 | 0.7552 |
| **5 – Prostate** | **0.8598** | **0.8607** | **0.7540** | **0.7554** |
| **Mean foreground Dice** | **0.9214** | **0.9212** | — | — |

Both models satsify the target prostate dice coeeficient of 0.75. The improved U-NET achieved a marginally higher prostate dice score of 0.8607 compares to 0.8598. However, the baseline achieved a slightly higher mean foreground dice (0.9214 instead of 0.9212). The improved model also had a slightyly higher prostate IoU of 0.7554 instead of 0.7540. The results above show that the additional architectural complexity did not produce a consistent or significant improvement across segmentation classes. 

The small prostate dice and IoU difference is not significant enough to establish superiority of the improved model. Further evaluation across additional patients would be beneficial to see if the marginal difference is consistent. 

### Computational Efficiency

| Resource | Baseline U-Net | Improved U-Net | Percentage Increase |
|---|---:|---:|---:|
| Parameter count | 7,849,190 | 12,727,066 | 62.1% |
| Median inference latency (ms/slice) | 4.774 | 8.473 | 77.5% |
| Peak allocated GPU memory (MiB) | 64.3 | 93.9 | 46.0% |
| Peak reserved GPU memory (MiB) | 80.0 | 120.0 | 50.0% |

Computational efficiency was evaluated locally using an NVIDIA RTX 3060 Laptop GPU, with Pytorch 2.13.0 and CUDA 13.0. Inference latency was measured using a single 2D MRI slice, with 10 warm-up passes followed by 30 CUDA-synchronised forward passes.

The measurements represent the computational requirements of applying trained models to MRI slices not while training them. The measurements exclude data loading, backpropagation and model optimisation. 

The improved U-Net required 62.1% more parameters, 46.0% more peak allocated GPU memory and 77.5% longer inference time compared with the baseline. Despite this increase in computational requirements, the improved model achieved only a marginal increase in prostate Dice (0.8598 to 0.8607) and IoU (0.7540 to 0.7554).

### Failure Autopsies
#### Lowest average prostate dice in both Models
![Case 1 - Complete detection failure](Figures/case_1_case_040_week_1_slice_39.png)
 

#### Worst Baseline prostate dice in comparison to Improved
![Case 2 - Improved model recovery](Figures/case_2_case_040_week_1_slice_21.png)


#### Worst Improved prostate dice in comparison to Baseline
![Case 3 - Improved model under-segmentation](Figures/case_3_case_040_week_0_slice_36.png)

#### Most False Positives
![Case 4 - Prostate over-segmentation](Figures/case_4_case_041_week_0_slice_39.png)
#### Most False Negatives
![Case 5 - Prostate boundary under-segmentation](Figures/case_5_case_042_week_0_slice_20.png)

The failure autopsies support why aggregate Dice coefficients are not a sufficient indicators of automatic prostate identification reliability. Both models satisfy the target dice of 0.75 however this masks the failure modes identified above. The analysis identified three failure modes, complete prostate omission, over-segmentation and under-segmentation. Smaller prostate regions appeared particularly challenging as highlighted by case 1 & 2. Although, larger regions still constituted signficiant boundary errors indicating segmentation issues are not only dictated by size.

The improved U-Net sucessfully identified prostate tissue missed byt the baseline in Case 2 and showed less over-segmentation in case 4. However, the baseline U-Net identified more prostate in case 3 and showed less oversegmentation in case 5. This supports that additional architectural complexity does not consistently improve boundary reliability. Aswell, the autopsy highlights the the limitations of relying on aggregate overlap metrics. 


### Engineering Recommendation 
Neither model is recommended for unsupervised clinical use. Failure cases indicate that automatically generated contoours should still undergo expert review. 

Both models exceed the required prostate dice threshold. The improved U-NET archieved marginally better results (+0.0009 Dice, +0.0014 IoU) at the cost of 46% increase in peak GPU VRAM and 77.5% increase in inference latency. Both models exhibited significant boundary innaccuracies depsite achieving relatively high dice scores. Therefore, the baseline 2D U-Net model is recommended for further development due to similar performance with higher computational efficiency. Future development should prioritise small region detection, validation on a larger number of individual patients and implementing boundary senstive loss functions.
## References
Isensee, F., Kickingereder, P., Wick, W., Bendszus, M., & Maier-Hein, K. H. (2018). Brain tumor segmentation and radiomics survival prediction: Contribution to the BRATS 2017 challenge. In A. Crimi, S. Bakas, H. Kuijf, B. Menze, & M. Reyes (Eds.), Brainlesion: Glioma, multiple sclerosis, stroke and traumatic brain injuries (pp. 287–297). Springer. https://doi.org/10.1007/978-3-319-75238-9_25

PyTorch contributors. (n.d.). Datasets & DataLoaders. PyTorch. https://docs.pytorch.org/docs/stable/data.html

The University of Queensland. (2026). COMP3710 pattern recognition: Assignment specification (Appendix B) [Unpublished course material].

## Ai Usage Disclosure
OpenAi ChatGPT-6 was used for assisting the interpretation of baseline and improved U-Net architectures, planning data auditing, developing and debugging python code, diagnosing Slurm errors, planning the project structure and refining README documentation. AI assisted code and recommendations were reviewed against the assignment requirements and the referenced U-Net architecture. Ai generated content was verified by carefully reading each Ai generated output, prompting for justification/explanation, local smoke tests for each modified file to test individual functionality and inspection of model generated results (e.g. segmentation predictions). The student takes responsibility for the final implementation, reported results and conclusions. 