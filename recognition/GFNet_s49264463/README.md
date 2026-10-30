
# GFNet for AD vs CN Classification (ADNI)

## Section 1 - Problem Overview

The problem is to classify the ADNI data set which contains images of patients brain slices  into AD or NC (Alzheimer's Disease or Normal Cognitive) , this classifier uses a GFnet (Global Filter Network)  with the goal of achieving a test accuracy of >0.8 with a baseline CNN to compare the results with.

## The workings of a GFnet as a AD/NC Classifier:
A Global Filter Network works like a vision transformer-style classifier , that uses a simple frequency-domain filter instead of self-attention to share information across the whole image.It flows like this -> it takes each slice in its spatial form, cuts it into 16×16 patches, and turns each patch into a short list of 192 numbers (a token), giving a 14×14 grid of tokens. Inside each block, the grid for every feature is converted from the spatial domain to its frequency domain using a 2D FFT transform, where each value now describes how much of a particular wave pattern, from broad shapes to fine detail, the slice contains. These frequency values are multiplied by a learned filter that decides which patterns to keep, shrink or boost, and an inverse Fourier transform converts the result back into the spatial domain. Because every frequency covers the whole slice, each patch now carries information from every other patch after a single layer. A small per-patch network then recombines the features, and this block is repeated six times. Finally, the tokens are averaged into one vector and a linear layer outputs two scores (NC and AD), which a softmax turns into the confidence score.  

<img width="2055" height="1083" alt="gfnet_architecture" src="https://github.com/user-attachments/assets/2e782f6e-cc3c-4255-8d9b-a2329d56b71f" />


*Figure 1: GFNet as implemented. Left: the overall network. Right: one block. Own diagram, following Rao et al. (2023).*

## Section 2 - Feasibility Review

### 1 - User Need, Scope & Acceptance Criteria
The intended user for our product will mainly fall in the medical field ,with clinical researchers and diagnostic healthcare providers being the main focus of the intended user.The product is not a replacement for diagnostic health care providers but a tool they can use to help understand the urgency of which cases to review and diagnose .
The prototype classifies the ADNI data set which are MRI slices as AD or CN using a global filter network as the networks architecture . The prototype will report a confidence score with each prediction and ideally route low confidence cases to a human source. The prototype includes a baseline CNN to test whether GFNet’s extra complexity is justified, comparing accuracy, confidence, reliability, parameters, peak VRAM and inference time. It is not designed to grade Alzheimer’s severity or to process 3D scans.An open dilemma to address  is whether the GFnet is over confident in ambiguous cases and what threshold safely separates automated decisions from human review. Acceptance Criteria : 1) The GFNet achieves ≥ 0.80 test accuracy on AD vs CN . 2) Among test predictions that have a confidence score of 85% or greater the GFNet achieves a 90% test accuracy , 3)  peak training VRAM fits on a ColabPro GPU (≤ 15 GB), and inference takes ≤ 50 ms per image, 4) a confidence threshold exists that sends ≤ 30% of test cases to human review while catching ≥ 70% of GFNet’s errors.


### 2 - Model Choice & Course Concepts
The model chosen for this task is the Global Filter Network, a Hard difficulty architecture , compared against a baseline simple CNN . The model is a justified choice for this task because Alzheimer's appears as large-scale changes across the brain , and its global filters let every patch influence every other patch in a single layer whereas a CNN needs many stacked layers. It also uses 2D
FFT token mixing at roughly O(N log N) cost. This suits the VRAM and inference time criteria . These are the expected benefits right now and the comparison with the baseline CNN will test them.The proposed dataset is the preprocessed ADNI brain MRI collection (AD vs CN), split at the patient level so no patient appears in more than one set.Each MRI slice is cut into patches and projected into tokens and are passed through N stacked Global Filter Blocks.Each block applies a 2D FFT, multiplies the spectrum by a learnable complex-valued global filter, and applies a 2D IFFT, followed by an MLP with LayerNorm and residual connections, as in a ViT.final pooling layer and linear classifier give a softmax confidence score for the human-review threshold.


### 3 - Preliminary Feasibility Evidence
Preliminary Feasibility Evidence:An audit of the preprocessed ADNI data found 21,520 training slices from 1,076 patients and 9,002 test slices from 450 patients, with exactly 20 slices per patient. File names encode the patient ID and the provided train and test folders share no patient IDs, so the supplied split is patient-level and shows no leakage. No patient is labelled both AD and NC. The classes are close to balanced, at 48% AD and 52% NC in training and 50% AD and 50% NC in test, so accuracy is a meaningful metric here.


### 4 - Risks, Budget & Fallback
The main technical risks are patient-level data leakage, overfitting, and overconfidence. Slices from the same patient could end up in both training and test sets and inflate accuracy, so I will split by patient ID and check that no patient appears in more than one set. Transformer style models like GFnet can overfit on these smaller medical datasets so I will tackle this through data augmentation on training sets only. The GFnet may also be over confident on ambiguous scans which is the dilemma this project investigates, so I will measure accuracy within confidence bands and compare against the baseline CNN.If GFNet cannot reach 0.80 test accuracy or does not fit the resource limits, the fallback is a smaller configuration (fewer blocks or a smaller embedding dimension) with stronger regularisation. If it still falls short, I will report the result honestly and compare it with the baseline CNN, which is the simpler option.The next step is training the baseline CNN on the patient-level split

## 3. Dependencies and Reproducibility

### Environment

| Item | Version / value |
|---|---|
| Hardware | Google Colab Pro, NVIDIA Tesla T4 (15,360 MiB), driver 580.82.07 |
| Python | 3.13.16 |
| PyTorch | 2.11.0 (CUDA 13.0 build, `2.11.0+cu130`) |
| torchvision | 0.26.0 (`0.26.0+cu130`) |
| Pillow | 11.3.0 |
| matplotlib | 3.10.0 |
| scikit-learn | 1.6.1 (used only for the evaluation metrics, not for training) |

No other libraries are needed. `modules.py` uses only PyTorch.

### Data

The data is the course's preprocessed ADNI collection (AD and NC folders; NC corresponds to CN in the assignment specification). It is copied to Colab's local disk at `/content/AD_NC`, because reading it directly from Google Drive was far too slow. The data and model weights are not committed to the repository.

### Reproducibility

- **Validation split:** deterministic. Patients are grouped by nearby IDs (gap 50) and shuffled with seed 42, giving the same 915 training and 161 validation patients every time.
- **Training:** no random seed was set for weight initialisation, data order or augmentation, so repeated runs differ. Repeating one configuration changed test accuracy from 0.722 to 0.731 (Section 6), so differences of about one to two points are noise.
 **Code versions:** the reported baseline CNN and plain GFNet were trained with an earlier version of `train.py` (plain Adam, 30 epochs; see the commit history). The current `train.py` is the regularised setup (for GFNet: AdamW with weight decay 0.05, dropout 0.1, stochastic depth 0.3, cosine learning-rate schedule, 40 epochs), so rerunning it reproduces the regularised GFNet and not the plain runs. The model is chosen with `model_name` at the top of the file.
- **Checkpoints:** the checkpoint with the best validation accuracy was kept for each run. The first regularised run's checkpoint was lost, so Section 6 benchmarks the second run.

### Running the code

See Section 8 for the commands.

## Section 4 -  Data - Information And Preprocessing :

### 1 - Data Audit: 

The data originally started with the training set containing 1,076 patients with 21,520 slices totally , and the test set contained 450 patients with 9,002 slices totally, which is around 20 slices per patient with one patient having 22 slices.The provided data split was 70/30 with no leakage from patients in the training set to the test set. 

The image ( a slice ) is a greyscale image , containing one channel and the images are 256×240. 

Here is a example of a AD and NC greyscale image slice :

<img width="256" height="240" alt="218391_78" src="https://github.com/user-attachments/assets/33165666-1b88-4f99-8911-feb5989500b3" />

*IMAGE OF AD BRAIN SLICE*

<img width="256" height="240" alt="808819_88" src="https://github.com/user-attachments/assets/2dc61e74-e774-46a4-835c-c55e18bab589" />

*IMAGE OF NC BRAIN SLICE*


### 2 - Data split :
As mentioned before an audit of the file names showed that no patient ID appears in both sets and that no patient carries both labels, so the supplied split is already separated by patient and the test set can be trusted as an independent measure of performance. The classes are close to balanced (about 48% AD in training and 50% in test).

There is no validation set in the provided dataset , I need more feedback in training to make better choices . I cant figure out parts in the training such as , which epochs weights to keep , when to stop training and how strong the regularisation should be and where to set the confidence threshold that routes uncertain cases to a human reviewer
I created the validation set from the training set , it was separated from the remaining training data at the same strictness the test set  was .each patient contributes 20 near-identical slices and a slice-level split would let the model be validated on a brain it has already seen. The patient ID is read from each file name (218391_78.jpeg is slice 78 of patient 218391), and all of a patient’s slices are grouped together. The IDs in the data suggest that one person can appear under several numerically close IDs (probably repeat scans), so patients whose IDs are within 50 of each other are merged into one group and are never separated. The groups are shuffled with a fixed seed (42), and whole groups are moved into validation until about 15% of the patients are held out. This gives 915 training patients (18,300 slices) and 161 validation patients (3,220 slices) . 

There was no augmentation applied to the validation set.

This was how the validation set was split with the data- leakage prevention step:

<img width="686" height="558" alt="Screenshot 2026-10-28 at 3 26 18 PM" src="https://github.com/user-attachments/assets/adb144a0-24f5-4ddb-8b15-ab638acd336f" />

The assertion to check whether there are no shared patients between the validation and training set :

<img width="632" height="42" alt="Screenshot 2026-10-28 at 3 30 31 PM" src="https://github.com/user-attachments/assets/5f1b77df-f6fd-4c3d-a58d-de453ae4e6f6" />




### 3 -  Preprocessing and augmentation:

Every slice, in the training, validation and test sets, goes through the same three steps.

First - The slice is resized , 256x240 becomes 224x224 , the model needs one fixed size for its inputs. It slightly squashed the aspect ratio , which is applied equally to every set . 

Pixel values 0-225 become  0.0-1.0 , shape (1 , 224 , 224) , PyTorch needs tensors. The 1 is the single greyscale channel as mentioned in the data audit . 
 
Normalise :  (mean 0.5, std 0.5)	Values become -1 to 1. A pixel of 0.8 becomes (0.8 - 0.5) / 0.5 = 0.6
Centred inputs make training faster and steadier.

Slices that belong only to the training set had the following  augmentations applied to them :
Random rotation: up to ±15°
Random shift: up to 10% in each direction
Random zoom: a scale factor between 0.9 and 1.1
Brightness and contrast jitter: ±0.1

These mimic natural variation in head position, head size and scanner settings, so the model cannot simply memorise exact pixels. Horizontal flips were deliberately left out, because the left and right sides of the brain cannot be assumed to be interchangeable. The validation and test slices received no augmentation.

<img width="2055" height="791" alt="preprocessing_table" src="https://github.com/user-attachments/assets/d0fe5c57-131c-4b26-a60d-de0210861b8d" />

Summary table of the preprocessing and augmentation applied to the dataset*









## Section 5 - Models:


## 1 - Baseline CNN:


  
### Introduction : 

The baseline for comparison with the GFnet is a convolutional neural network which I adapted from the MNIST practical in COMP3710 . It takes a 224x224 greyscale slice and passes it through four identical blocks (Added two more blocks for a better test accuracy to compare with the GFnet), each combining a 3×3 stride-2 convolution, batch normalisation, a ReLU and light dropout.The convolutions shrink the slice from 224×224 to 13×13 while the number of feature maps grows to 256, so that later blocks respond to larger structures. Global average pooling then collapses the feature maps into a 256-number summary of the slice, and a linear layer produces two logits, which a softmax turns into the NC and AD probabilities used as the confidence score. 

### Why is this a suitable baseline? 

I found that the structure of the CNN was the closest to the GFnet overall structure , they take the same 224×224 slice, pass it through a stack of repeated blocks, and end with global average pooling and a linear layer that outputs two logits and a softmax confidence score.The main difference is how information is shared across the slice , the CNN’s small 3×3 filters connect distant regions only after several layers, whereas GFNet’s global filter connects every patch to every other patch in one layer. Comparing them therefore tests whether global mixing, and GFNet’s extra parameters, are worth the cost for AD vs NC classification.

### Architecture

A 1 × 224 × 224 greyscale slice passes through four identical blocks, each made of a 3×3 convolution with stride 2, batch normalisation, ReLU and dropout (0.1). Global average pooling then reduces the final 256 feature maps to 256 numbers, and a linear layer outputs two logits (NC, AD). The model has 961,666 parameters.


| Stage | Output shape |
|---|---|
| Input slice | 1 × 224 × 224 |
| Block 1 | 64 × 111 × 111 |
| Block 2 | 128 × 55 × 55 |
| Block 3 | 256 × 27 × 27 |
| Block 4 | 256 × 13 × 13 |
| Global average pooling | 256 |
| Linear | 2 logits |

### Design choices

Stride-2 convolutions shrink the slice without separate pooling layers, so later blocks see a larger part of the brain. Global average pooling followed by one linear layer keeps the model small and less prone to overfitting than flattening the feature maps into a large fully connected layer. The design follows the course MNIST CNN so that the baseline stays simple and recognisable.

### Training Of The Baseline CNN:


The baseline was trained from scratch on the patient-level training split. After every epoch it was evaluated on the validation set, and the checkpoint with the best validation accuracy was kept. The test set was not used to choose the checkpoint.

| Setting | Value |
|---|---|
| Task | Binary classification, NC (0) vs AD (1), per slice |
| Input | 1 × 224 × 224 greyscale slice, normalised (mean 0.5, std 0.5) |
| Training data | 18,300 slices (915 patients) |
| Validation data | 3,220 slices (161 patients), grouped by nearby patient ID (seed 42) |
| Augmentation | Training set only: rotation ±15°, shift 10%, zoom 0.9–1.1, brightness/contrast ±0.1 |
| Loss function | Cross-entropy |
| Optimiser | Adam, learning rate 1e-3, no weight decay, no learning-rate schedule |
| Batch size | 32 (572 steps per epoch) |
| Epochs | 30 |
| Regularisation | Dropout 0.1, batch normalisation, augmentation |
| Weight initialisation | PyTorch defaults |
| Checkpoint rule | Best validation accuracy |
| Hardware | Google Colab Pro, NVIDIA T4 (15 GB) |
| Software | Python 3.13.16, PyTorch 2.11.0, torchvision 0.26.0 |
| Training time | 21.6 minutes |
| Peak GPU memory | 0.59 GB |
| Parameters | 961,666 |

### Training Behaviour:

<img width="1246" height="586" alt="cnn_clean_curves" src="https://github.com/user-attachments/assets/82db6f21-763e-4412-be3f-b5e342d44942" />



### Baseline results

| Model | Parameters | Test accuracy (per slice) | Test loss | Training time | Peak VRAM |
|---|---|---|---|---|---|
| CNN (4 blocks) | 961,666 | 0.716 | 0.642 | 21.6 min | 0.59 GB 



## 2 -  Global Filter Network

This section covers the implemented architecture, its design choices and how it was trained.
GFNet's working principle (patch tokens, 2D FFT, learned frequency-domain filter, inverse FFT) is described in Section 1.

### Architecture :


<img width="2055" height="1083" alt="gfnet_architecture" src="https://github.com/user-attachments/assets/0190e60d-08ae-43b6-9dcd-f39bbbe9a077" />
* Image of the GFnet Architecture*

| Stage | Output shape |
|---|---|
| Input slice | 1 × 224 × 224 |
| Patch embedding (16×16 conv, stride 16) + positional embedding | 196 tokens × 192 |
| 6 × GFNet block | 196 tokens × 192 |
| LayerNorm + average over tokens | 192 |
| Linear | 2 logits (NC, AD) |

Each block computes `x = x + MLP(LayerNorm(GlobalFilter(LayerNorm(x))))`: LayerNorm, global filter (2D FFT, learned complex filter, inverse FFT), LayerNorm, an MLP (192 → 768 → 192) and one residual connection around the whole block.

| Component | Parameters |
|---|---|
| Patch embedding | 49,344 |
| Positional embedding | 37,632 |
| 6 blocks (each: filter 43,008 + MLP 295,872 + LayerNorms 768) | 2,037,888 |
| Final LayerNorm + classifier | 770 |
| **Total** | **2,125,634** |


### Design Choices

I chose A small configuration ( 192 - tokens and 6 global filtered blocks) as the training set is only 18,000 slices from 915 patients.
A large transformer-style model like this can overfit easily when trained from scratch ,The filter is a separate learned 14×8 complex array for each of the 192 features, so features are filtered independently and are mixed across features only by the MLP.  Stochastic depth (drop-path 0.1) was used as light regularisation, and tokens are averaged before the classifier so that every patch contributes. Position information comes from a learned positional embedding.

#### Training

| Setting | Value |
|---|---|
| Loss / optimiser | Cross-entropy / Adam, learning rate 3e-4, no weight decay, no schedule |
| Epochs / batch size | 30 / 32 |
| Regularisation | Stochastic depth 0.1, augmentation (training set only) |
| Data and augmentation | As for the baseline (Section 4) |
| Checkpoint rule | Best validation accuracy |
| Hardware | Google Colab Pro, NVIDIA T4 |
| Training time / peak VRAM | 20.3 min / 0.48 GB |
| Parameters | 2,125,634 |

#### Results

| Model | Parameters | Test accuracy (per slice) | Test loss | Training time | Peak VRAM |
|---|---|---|---|---|---|
| CNN (4 blocks) | 961,666 | 0.716 | 0.642 | 21.6 min | 0.59 GB |
| GFNet | 2,125,634 | 0.713 | 0.653 | 20.3 min | 0.48 GB |
| GFNet (regularised) | 2,125,634 | 0.722 | 0.680 | 27.2 min | 0.51 GB |

My GFNet has about 2.2 times the parameters of the baseline but does not improve on its test accuracy, and the three results are within the noise expected from 450 test patients. The regularised run (AdamW with weight decay 0.05, dropout 0.1, stochastic depth 0.3, cosine schedule, 40 epochs) closed the gap between training and validation loss but left test accuracy essentially unchanged. The further experiments, full metrics, the confidence analysis and the comparison are in Section 6.




## Section 6 - Results - The Open Research Dilemma

The question that was proposed at the beginning of the report was whether the GFNet's extra complexity is justified over a simple CNN,
and whether either model's confidence can be trusted in a way that it can decide which cases need a human reviewer. The accuracies are all per slice unless stated otherwise. The target we set out for the  GFNet was 0.80 / 80% test accuracy .

I did not reach the 0.80 test accuracy target for GFNet. Every model I trained, including the baseline CNN, scored about 0.71 to 0.73 on the test set, while validation accuracy was consistently higher, at about 0.82 to 0.86. Rather than stopping there, I ran a series of experiments across several strategies to try to raise test accuracy and to understand why it stayed where it was.



### Experiments :
| # | Strategy | Experiment | Val acc | Test acc | Test loss |
|---|---|---|---|---|---|
| 1 | Capacity | CNN, 2 blocks (75,138 parameters)* | 0.630 | 0.622 | 0.648 |
| 2 | Capacity | CNN, 4 blocks (961,666 parameters)* | 0.860 | 0.728 | 0.624 |
| 3 | Baseline | GFNet, light augmentation* | 0.848 | 0.713 | 0.890 |
| 4 | Data | GFNet, stronger augmentation* | 0.823 | 0.731 | 0.715 |
| 5 | Data pipeline | Validation leak fixed, CNN retrained | 0.834 | 0.716 | 0.642 |
| 5 | Data pipeline | Validation leak fixed, GFNet retrained | 0.822 | 0.713 | 0.653 |
| 6 | Regularisation | GFNet: AdamW, dropout, stochastic depth, cosine schedule, 40 epochs | 0.818 | 0.722 | 0.680 |

### Experiment 1: Increasing CNN depth
**Motivation.** The first two-block CNN underfitted: training and validation loss were still falling at epoch 15.
**Change.** I extended it to four blocks, lowered the dropout from 0.3 to 0.1 and trained for 30 epochs.
**Result.** Test accuracy rose from 0.622 to 0.728.
**Takeaway.** The baseline needed more capacity. Depth, dropout and epochs changed together, so the gain cannot be attributed to depth alone.

### Experiment 2: GFNet with light augmentation
**Motivation.** To test whether GFNet's global filter beats the stronger baseline.
**Result.** GFNet reached 0.713 test accuracy, with 2.2 times the CNN's parameters. Its training loss fell to 0.21 while validation loss stalled near 0.40, so it was memorising the training patients, and its test loss (0.890) was far higher than the CNN's.
**Takeaway.** GFNet did not beat the baseline, and it was overfitting.

### Experiment 3: Stronger augmentation
**Change.** Rotation from ±10° to ±15°, shift from 5% to 10%, and zoom from 0.95-1.05 to 0.9-1.1.
**Result.** The train-to-validation loss gap fell from about 0.20 to about 0.09, test loss fell from 0.890 to 0.715 and test accuracy rose slightly to 0.731.
**Takeaway.** Augmentation reduced memorisation and made the model less overconfident on test patients, but accuracy barely changed.

### Experiment 4: Fixing leakage in the validation split
**Motivation.** Validation accuracy (0.82-0.86) was well above test accuracy (about 0.72).
**Finding.** A nearest-neighbour similarity check showed that 16.8% of validation slices had a near-identical training slice (cosine similarity above 0.99), against 0.4% for test versus training. The matching patients had numerically close IDs, which suggests one person appears under several IDs. An ID-equality check could not detect this.
**Change.** Patients with IDs within 50 of each other are now kept in the same group, and whole groups are assigned to validation. The near-duplicate rate fell to 0.6%. I then retrained both models.
**Result.** Test accuracy was 0.716 (CNN) and 0.713 (GFNet), essentially unchanged, and validation accuracy was still about 0.82-0.83.
**Takeaway.** The leak was real and is fixed, so validation is more trustworthy, but it was not the main cause of the validation-to-test gap.

### Experiment 5: Regularising GFNet
**Change.** AdamW with weight decay 0.05, dropout 0.1, stochastic depth 0.3 and a cosine learning-rate schedule, trained for 40 epochs, on the fixed split.
**Result.** The gap between training and validation loss closed (0.376 against 0.404), but test accuracy was 0.722 and test loss 0.680, essentially unchanged.
**Takeaway.** Overfitting to the training patients is not what limits test accuracy.

### Overall Experiment Takeaway
Across all runs, test accuracy stayed within about two points of 0.72 while the training-validation gap changed substantially, and the CNN and GFNet performed about the same. This suggests the limit comes from how the test patients differ from the training patients and not from model size or training recipe, although I was not able to confirm the cause.


### Section 6.1 Benchmark:



All models were evaluated on the same 9,002 test slices (450 patients). AD is treated as the positive class, and all accuracies are per slice unless stated otherwise. Confidence intervals resample patients, not slices, because the 20 slices of one patient are not independent.


| Model | Test acc (95% CI) | Macro-F1 | AUROC | NC precision / recall | AD precision / recall | Patient-level acc |
|---|---|---|---|---|---|---|
| CNN (4 blocks) | 0.716 (0.682–0.749) | 0.710 | 0.795 | 0.675 / 0.841 | 0.784 / 0.587 | 0.731 |
| GFNet | 0.713 (0.682–0.741) | 0.707 | 0.792 | 0.670 / 0.846 | 0.787 / 0.577 | 0.764 |
| GFNet (regularised) | 0.731 (0.701–0.760) | 0.727 | 0.799 | 0.689 / 0.850 | 0.800 / 0.610 | 0.773 |

**Confusion matrices** (rows: true class; columns: predicted NC, AD)

| Model | True NC | True AD |
|---|---|---|
| CNN (4 blocks) | 3,820 / 720 | 1,841 / 2,621 |
| GFNet | 3,843 / 697 | 1,889 / 2,573 |
| GFNet (regularised) | 3,860 / 680 | 1,739 / 2,723 |

From the figures you can see that the three Models are statistically indistinguishable . Their test accuracies differ by under two points , their confidence intervals overlap almost completely , and their AUROCS are all 0.79 to 0.80 . This clearly shows that the GFnet does not outperform the baseline CNN , and none of the models come close to reaching the 0.80 test accuracy target.

The three models are also biased towards predicting NC . Around 84 to 85% of NC slices are classified correctly, but for AD it's only 58 to 61% of AD slices , so it's around 40% of the AD slices are missed . This is a serious concern in this case as for a screening tool a missed AD case is more of a serious error. I have included an investigation on this Issue in section 6.4 .

Averaging each patient's 20 slice probabilities gives patient-level accuracies of 0.731 (CNN), 0.764 (GFNet) and 0.773 (regularised GFNet). This is a different quantity from per-slice accuracy, and with only 450 patients the differences between models are not reliable.


### 6.2 Resource profile

| Model | Parameters | Peak VRAM | Training time | Inference, ms/image (batch 32 / batch 1) |
|---|---|---|---|---|
| CNN (4 blocks) | 961,666 | 0.59 GB | 21.6 min | 0.61 / 0.96 |
| GFNet | 2,125,634 | 0.48 GB | 20.3 min | 0.54 / 2.89 |
| GFNet (regularised) | 2,125,634 | 0.51 GB | 28.7 min (40 epochs) | 0.53 / 3.10 |

Training used a Google Colab Pro NVIDIA T4, and inference times were measured on the same GPU. GFNet has 2.2 times the CNN's parameters but uses less peak memory and trains in a similar time ,the regularised run took longer only because it ran for 40 epochs. Its throughput at batch 32 is slightly better, but a single image takes about three times as long as the CNN's (about 3 ms against 1 ms), probably because its six blocks launch many small GPU operations, although I did not profile this. All models are fast enough for interactive use, so the extra cost is small in absolute terms, but it buys no accuracy on this data.


### 6.3 Confidence analysis and reject threshold

<img width="3000" height="880" alt="fig_roc_reliability_coverage" src="https://github.com/user-attachments/assets/65edeeec-6fc3-4399-b3a1-722997f6f864" />



 All three models are noticeably overconfident on the test set: their mean confidence (0.81 to 0.85) is higher than their accuracy (0.71 to 0.73), and their expected calibration error (ECE) is about 0.10 to 0.12. The regularised GFNet is the most overconfident (mean confidence 0.853 against accuracy 0.731, ECE 0.122), even though it is the most accurate. Regularisation therefore did not improve calibration here. This answers the first half of the dilemma: neither model's raw confidence can be taken at face value.

The confidence is still informative - >  Accuracy rises steadily as the least confident slices are rejected. For the CNN, the slices kept at a confidence of at least 0.7, 0.8 and 0.9 are 0.768, 0.792 and 0.841 accurate, covering 76%, 61% and 39% of slices. The two GFNet runs behave similarly (0.830 and 0.812 at a threshold of 0.9, covering 37% and 51%). So confidence does rank predictions usefully, but at a cost in coverage.

I chose the threshold on the validation set as the lowest confidence at which the kept slices reach 0.90 accuracy, then applied it unchanged to the test set.

| Model | Threshold (from validation) | Val kept accuracy | Test coverage | Test kept accuracy | Sent to a human |
|---|---|---|---|---|---|
| CNN | 0.64 | 0.902 | 83.1% | 0.754 | 16.9% |
| GFNet | 0.73 | 0.900 | 69.5% | 0.767 | 30.5% |
| GFNet (regularised) | 0.79 | 0.900 | 69.8% | 0.782 | 30.2% |

The reject rule did not hold up on the test set. I picked each threshold on the validation set so that the slices it kept were 0.90 accurate, but on the test set the same threshold only gave 0.75 to 0.78. I think this is because the validation set is easier than the test set, although I couldn’t confirm that. It shows why the dilemma matters: a threshold that looks safe on validation data can fail on new patients. I also tried fixed thresholds on the test set to see what it would take. Even at a confidence of 0.9, the kept slices were only 0.81 to 0.84 accurate, and I had to reject 49 to 63% of slices to get there. So none of my models reached 0.90 accuracy on the test set at any threshold I tried.

### 6.4 Failure case autopsy

The most serious error for a screening tool is a missed AD case, and all three models miss about 40% of AD slices (Section 6.1). I examined the missed AD slices of the regularised GFNet in four ways: across patients, across slice position, by confidence, and by looking at individual examples.

<img width="2800" height="760" alt="fig_missed_ad" src="https://github.com/user-attachments/assets/21b670bc-62e7-47e4-9b3c-5f773579e7ea" />

*Figure 6: Left: share of each AD patient's slices that were missed. Centre: miss rate by slice position. Right: confidence of missed and correctly caught AD slices.*
*Figure 7: Top: confidently missed AD slices from four patients (all 20 of each patient's slices were missed). Bottom: a caught AD slice from a different patient at a similar slice number.*

I noticed that the misses are not spread evenly. Of the 223 AD test patients, 21% had no slice missed, 41% had fewer than half missed, and 38% had half or more missed, including 12% (about 27 patients) whose 20 slices were all missed. For these patients the model sees nothing that looks like AD in any slice, so averaging a patient's slices cannot recover them. This probably explains why patient-level accuracy (0.773) is only a few points above slice-level accuracy (0.731).

The miss rate rises with slice number: 0.27 for slices 75-89, 0.31 for 89-96, 0.46 for 96-103 and 0.58 for 103-114. The model therefore misses AD much more often in the later slices. I did not check what these slice numbers correspond to anatomically, or whether patients differ in where their slices start, so I treat this as an observation and not an explanation.

 Missed AD slices are not low-confidence. Their median confidence is 0.86, almost the same as the 0.87 of correctly caught AD slices, and 61% of them have a confidence above 0.8. The model is confidently wrong on these slices, so a reject threshold cannot catch them. This agrees with Section 6.3, where the reject rule did not reach the target accuracy on the test set.

Figure N+1 shows the four most confidently missed AD slices, each from a different patient, next to a correctly caught AD slice from a different patient at a similar slice number.

<img width="2600" height="1360" alt="fig_missed_ad_examples" src="https://github.com/user-attachments/assets/84a0852f-437f-4872-966f-f708446d8817" />

Figure N+1: Top: confidently missed AD slices from four patients (all 20 of each patient's slices were missed). Bottom: a caught AD slice from a different patient at a similar slice number.</em></p>

| Case | Patient / slice | True / predicted | Confidence | Slices missed | What I observe |
|---|---|---|---|---|---|
| 1 | 1223029 / 109 | AD / NC | 1.00 | 20/20 | Clean scan; the ventricle looks moderate in size and the cortical folds look well preserved, with no obvious atrophy compared with the caught AD slices |
| 2 | 1240482 / 109 | AD / NC | 1.00 | 20/20 | Same: clean scan, moderate ventricle, well-defined folds |
| 3 | 1398870 / 107 | AD / NC | 1.00 | 20/20 | Same: clean scan, moderate ventricle, well-defined folds |
| 4 | 985197 / 102 | AD / NC | 1.00 | 20/20 | Atypical image: speckled texture, unusual intensities and part of the brain cut off, which looks like a preprocessing or registration problem |

In three of the four cases the scan looks clean and its anatomy looks pretty ordinary to me. In the caught AD slices the ventricle looks larger and the sulci wider. This suggests these patients may have AD that is not visible as marked atrophy on these slices, so the model has little to go on and answers NC with full confidence. The fourth case is an atypical image with noise and missing brain, which suggests that some scans are affected by preprocessing problems. In all four cases the confidence was 1.00, so no reject threshold would have caught them.

To summarise it ,  The missed AD slices are concentrated in a minority of patients, are more common in later slices, and are made with high confidence. This matters for the dilemma: the errors that matter most for screening are the ones a confidence-based reject rule is least able to catch.


### 6.5 Recommendation to the project manager

*Is the GFnet worth Advancing on?*
On this Data I would say that the GFnet is not worth advancing on, the extra complexity is not justified over the CNN 4 block architecture. It has  about 2.2 times the parameters (2,125,634 versus  961,666) and reaches the same test accuracy (0.713 against 0.716 per slice, with overlapping 95% confidence intervals and AUROCs of 0.79 to 0.80). Its inference time for a single image is about three times longer (about 3 ms versus 1 ms), although its training time and memory use are similar. Strengthening the augmentation, adding regularisation and changing the training recipe did not change this (As Mentioned in Experiments). Neither model reached the 0.80 target

*Can either model's confidence be trusted?** Wouldn't say on its own. Both models are overconfident on unseen patients (mean confidence 0.81 to 0.85 against accuracy 0.71 to 0.73, ECE about 0.10 to 0.12). A reject threshold chosen on validation so that kept slices were 0.90 accurate gave only 0.75 to 0.78 on the test set (Section 6.3). Many of the most serious errors are also made with full confidence: the missed AD slices have a median confidence of 0.86, and the most confident misses reach 1.00, so a confidence threshold cannot catch them (Section 6.4).

**What I would and would not do**

| Use | Recommendation |
|---|---|
| Automated diagnosis | No. About 40% of AD slices are missed, and the 0.80 target was not met |
| Triage aid with human review | Possible only as a supporting tool. The CNN with a threshold of 0.64 would send about 17% of slices to a reviewer, but the remaining slices were only 0.75 accurate on test |
| Choice of model | Prefer the CNN. It is smaller, simpler and performs the same |
| Setting the threshold | Re-estimate it on a held-out set of new patients before any real use, because validation overstated test performance |

**Limitations of this recommendation.** All results are per slice from a single run of each configuration, so differences of about two points are within noise. The 0.80 target was not met, and I did not establish why validation accuracy (about 0.82) was higher than test accuracy (about 0.72). The most likely explanation is that the test patients differ from the training patients, but I could not confirm this. The validation split was built with a heuristic (grouping nearby patient IDs) checked by a similarity test. Further limitations and ideas for improvement are in the next section.

## Section 7 -  Limitations and Ideas For Improvement :

### 7.1 - Limitations:
- **Per-slice results.** The models classify one slice at a time and every accuracy in this report is per slice. A clinician needs one answer per patient. Patient-level accuracy (0.73 to 0.77) was computed by averaging each patient's 20 slice probabilities, but it is a different quantity and was not optimised.
- **Target not met.** No model reached the 0.80 test accuracy target. All scored about 0.71 to 0.73.
- **Unexplained validation-to-test gap.** Validation accuracy (about 0.82) was consistently higher than test accuracy (about 0.72). Fixing the validation leak did not close it. I think the test patients differ from the training patients, but I did not confirm this (for example, by comparing scanners, sites or patient ages).
- **Test set seen during development.** I looked at test accuracy after most runs. Checkpoints were chosen on validation, but test results informed which experiments I tried next, so the test numbers are less independent than they look.
- **Missed AD cases.** About 40% of AD slices were missed, many with confidence near 1.00 (Section 6.4). I did not establish why, and my reading of the example slices is an observation, not a clinical judgement.
- **Training from scratch on a small dataset.** About 18,000 training slices from 915 patients, with no pretrained weights. Architecture and recipe were not tuned beyond the experiments reported.

### 7.2 Ideas For Improvement:

The most urgent problem I noticed is that about 40% of the AD slices were missed, many of these were with a confidence close to 1.00.
The AUROC is higher (0.79) than the accuracy at the default 0.5 cutoff suggests,I'd say that the model probably rank the slices better than they classify them , so choosing the decision threshold on the validation set to trade some false alarms for more caught AD cases is the cheapest thing to try. For a screening tool, that is also the more sensible direction to move in.

Another thing  is that everything I trained plateaued at about 0.72 on the test set, whether it was a CNN or GFNet and however it was regularised. That makes me think the limit lies in the data, not in the architecture. The first thing I would do with more time is investigate the gap between validation accuracy (about 0.82) and test accuracy (about 0.72), by comparing image statistics and any available metadata to see whether the test patients were acquired differently, because until that is understood, it is hard to trust any other improvement. A related step is to build the validation split from image-similarity clusters rather than nearby patient IDs, so that validation is as hard as the test set. If pretrained weights are allowed, they are the most likely way to get past the plateau, since features learned on much larger datasets often help when only about 18,000 training slices are available.

Finally, the models were overconfident (ECE about 0.10 to 0.12), so calibration methods such as label smoothing or temperature scaling should make a reject threshold more reliable. To judge any of these changes properly, I would run several seeds per configuration, since a repeat of one run moved test accuracy by about a point, and I would finish the strategies I planned but did not run: mixup, per-slice normalisation, weight averaging, a smaller GFNet and attention in place of the FFT filter.

## 8. Example Usage

### Setup

```
git clone https://github.com/RishabhSrikanth06/PatternAnalysis-2026.git
cd PatternAnalysis-2026
git checkout topic-recognition
cd recognition/GFNet_s49264463
```

### Training

```
python train.py
```

Set `model_name` at the top of `train.py` to `"gfnet"` or `"cnn"`. Checkpoints and curves are saved to Google Drive (`/content/drive/MyDrive/gfnet_runs`), not to the repository.

### Prediction

```
python predict.py --model gfnet --ckpt /content/drive/MyDrive/gfnet_runs/best_gfnet.pt --threshold 0.79 --out predictions.png
```

**Example output**

```
1003730_100.jpeg: true AD, predicted NC, confidence 0.76 -> SEND TO HUMAN
...(the other 11 lines)...
416324_86.jpeg: true AD, predicted AD, confidence 0.97 -> auto
9/12 correct on this sample; 3 flagged for human review; figure saved to predictions.png
```

**Example output**

<img width="1920" height="1620" alt="predictions" src="https://github.com/user-attachments/assets/1caf1b19-8643-4722-9d7c-810a8a99f38e" />

/content/PatternAnalysis-2026/recognition/GFNet_s49264463
1003730_100.jpeg: true AD, predicted NC, confidence 0.76 -> SEND TO HUMAN
1162188_86.jpeg: true AD, predicted AD, confidence 0.77 -> SEND TO HUMAN
1196891_98.jpeg: true NC, predicted NC, confidence 0.97 -> auto
1231978_108.jpeg: true AD, predicted AD, confidence 0.93 -> auto
1259263_98.jpeg: true AD, predicted NC, confidence 1.00 -> auto
1281631_108.jpeg: true NC, predicted NC, confidence 0.95 -> auto
1320573_96.jpeg: true NC, predicted NC, confidence 0.70 -> SEND TO HUMAN
1344288_108.jpeg: true NC, predicted NC, confidence 0.99 -> auto
1387539_96.jpeg: true NC, predicted NC, confidence 0.81 -> auto
388630_86.jpeg: true AD, predicted NC, confidence 0.83 -> auto
395983_93.jpeg: true AD, predicted AD, confidence 0.87 -> auto
416324_86.jpeg: true AD, predicted AD, confidence 0.97 -> auto

In this sample, 3 of 12 predictions are wrong, all of them missed AD cases. Only one is flagged for human review; the other two have confidence 0.83 and 1.00 and would be accepted automatically. This illustrates the finding in Section 6.4 that the errors that matter most for screening are the ones a confidence threshold is least able to catch. With only 12 slices, it illustrates the pattern and does not prove it.

## 9. Artificial Intelligence Usage Disclosure

**Tools used.** I used Claude (Anthropic) throughout this project as an assistant for explanations, drafting and debugging. 

**What AI assisted with**

| Area | How AI was used |
|---|---|
| Understanding | Explaining the Fourier transform, GFNet, residual connections, validation sets, patient-level leakage, calibration and the assessment spec |
| Data pipeline (`dataset.py`) | Drafted the patient grouping, the train/validation split, the transforms, the `Dataset` class and the data loaders. I later wrote the similarity check approach with Claude after noticing the validation gap |
| Debugging | Colab, Google Drive, Git and shell problems, including slow data loading and corrupted copied files |
| Report |   edited README sections, tables and figure scripts,|

**How I verified AI-generated work**

- Ran all code myself and checked outputs against expectations: tensor shapes, parameter counts (the GFNet count matched a hand calculation of 2,125,634), a one-epoch smoke test before each full run, and agreement between validation and test results.
- Added an assertion that no patient is shared between the training and validation sets, and ran a nearest-neighbour similarity check that found the leak an ID check had missed. I fixed the split and retrained both models.
- Checked every number in the report against the printed logs and saved outputs, and removed or hedged claims I could not support (for example, the cause of the validation-to-test gap).
- All the experiments were done by me and The Idea behind the different experiments were mine

**Limits of my verification.** The AI-drafted code and text were not independently reviewed by anyone else. Some AI suggestions contained errors that I found only when running them, so I treated all output as unverified until tested. I take responsibility for every line of code, equation and claim in this submission.

## 10. References

1. Y. Rao, W. Zhao, Z. Zhu, J. Lu and J. Zhou, "Global filter networks for image classification," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 45, no. 6, pp. 7837–7850, 2023. (Earlier version: NeurIPS 2021, arXiv:2107.00645.)
2. Y. Rao et al., GFNet official implementation, https://github.com/raoyongming/GFNet. Used as a structural reference for the block layout; the code in this repository is my own implementation.
3. COMP3710 course practical: MNIST CNN classification notebook, The University of Queensland, 2026. Adapted for the baseline CNN and the training loop.
4. A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," *ICLR*, 2021. Background for patch tokens and the block layout.
5. Alzheimer's Disease Neuroimaging Initiative (ADNI), http://adni.loni.usc.edu. Source of the MRI data, supplied preprocessed through COMP3710 (The University of Queensland).
6. A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," *NeurIPS*, 2019.
7. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011. Used for evaluation metrics.

