# ConvNeXt for Alzheimer's Screening
## What is ConvNeXt?
ConvNeXt is a modern Convolutional Neural Network (CNN) architecture introduced to bridge the performance and design gap between traditional CNNs and the newer transformers. While transformers dominate computer vision benchmarks due to advanced design choices, they often lack the efficiency, simplicity, and inductive biases of convolutional networks. ConvNeXt bridges this gap by proving that traditional ConvNets can be improved to match or exceed transformer performance while remaining simple and efficient.

## How Does it Work?
ConvNeXt is a highly optimised CNN that processes image data through a series of convolutional stages. It starts off with a stem that groups pixels into patches, passing them through feature blocks where inverted bottlenecks expand and contract channel dimensions to maintain representational power. To capture broader spatial context, ConvNeXt uses large 7x7 depthwise convolutions to isolate spatial filtering from channel mixing. It goes through multiple stages of blocks between downsampling, and integrates design choices such as replacing traditional batch normalisation with layer normalisation and using the GELU activation function, and removes excessive activation layers. This streamlined pipeline enables ConvNeXt to efficiently extract features and maintain a high spatial awareness through standard convolution operations. This makes it ideal for medical imaging classification and triage. See [the original paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf) for more details.

<img width="2089" height="598" alt="image" src="https://github.com/user-attachments/assets/037cf03d-c7c0-466c-b0a1-80009d7c2bda" />

The architecture of ConvNeXt (figure by [Lu et al.](https://xiahepublishing.com/m/2310-8819/JCTH-2025-00631))


## Feasibility Review
### 1. User Need, Scope & Acceptance Criteria
This prototype is for medical practitioners/researchers to reliably classify MRIs to determine whether the patients have Alzheimer's, and to provide the level of uncertainty on ambiguous boundary cases.
The acceptance criteria are as follows:
- No data leakage – all MRI slices from one patient must be in the same split
- Peak GPU VRAM of no more than 8GB
- Median inference latency of no more than 100ms per MRI slice
- Per class precision, recall, macro-F1, and AUROC – ConvNeXt should improve on these when compared to the baseline CNN

### 2. Model Choice & Course Concepts
The dataset will be split into 70% training, 15% validation, and 15% testing, ensuring that there is no data leakage between the sets. Data leakage is important to avoid, as Fung et al. (https://arxiv.org/html/1906.04231v1) identified substantial differences in performance with and without patient-level splitting. 
The baseline model will be a small CNN with the following architecture: 
- Input: Single-channel (grayscale) 2D image tensor.
-  Block 1: Conv2d (1 -> 16 channels, 3x3 kernel, padding 1), batch normalisation and ReLU activation, with 2x2 max pooling. Halves spatial dimensions.
-  Block 2: Conv2d (16 -> 32 channels, 3x3 kernel, padding 1), batch normalisation and ReLU activation, with 2x2 max pooling. Halves spatial dimensions again.
-  Block 3: Conv2d (32 -> 64 channels, 3x3 kernel, padding 1), batch normalisation and ReLU activation, with 2x2 max pooling and then adaptive average pooling. Forces spatial dimensions to 1x1 globally.
-  Classifier: convert the resulting 64x1x1 tensor into a 64-element feature vector and output the classification logits.

ReLU activation was chosen because it introduces non-linearity, and is simple and efficient (lecture). Using batch normalisation because it enables faster and more stable training (http://arxiv.org/abs/1805.11604). 

The ConvNeXt prototype will have the following architecture (based on the architecture in “A ConvNet for the 2020s” https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf) :
- Input: Single-channel (grayscale) 2D image tensor.
-  Stem: Conv2d (1 -> 32 channels, 4x4 kernel, stride 4) and group normalisation. Halves spatial dimensions.
-  Stage 1 (32 channels): One ConvNeXtBlock module (using depth-wise convolution with a 7x7 kernel, layer normalisation, 1x1 channel expansion by 2, GELU activation, and 1x1 projection, with a residual connection).
-  Downsample 1: Conv2d (32 -> 64 channels, 2x2 kernel, stride 2) and group normalisation. Halves spatial dimensions.
-  Stage 2 (64 channels): One ConvNeXtBlock module.
-  Downsample 2: Conv2d (64 -> 128 channels, 2x2 kernel, stride 2) and group normalisation. Halves spatial dimensions.
-  Stage 3 (128 channels): One ConvNeXtBlock module.
-  Downsample 3: Conv2d (128 -> 256 channels, 2x2 kernel, stride 2) and group normalisation. Halves spatial dimensions.
-  Stage 4 (256 channels): One ConvNeXtBlock module, followed by adaptive average pooling forcing spatial dimensions to 1x1 globally.
-  Classifier: linear layer to output the classification logits.

Using AdamW optimiser for both models because of the consistent regularisation and better generalisation (lectures).

The chosen difficulty band is hard. 

### 3. Preliminary Feasibility Evidence
Initial data audit: 
- Number of participants: 1526
- Scans per class show that the classes are pretty well balanced:
AD in the training/validation class: 10400
NC in the training/validation class: 11120
AD in the test class: 4460
NC in the test class: 4540
- Verified that there is no data leakage between the training/validation set and the test set
- Image dimensions: 256x240
- Missing or invalid data: None
- The mean and standard deviation of the training set is 0.1156 and 0.2229, respectively. This will be used to normalise the data. 

### 4. Risks, Budget & Fallback
Technical risks: limited generalisation if it learns dataset-specific patterns, if data leakage then inflated test accuracy from same patient appearing in multiple splits.

Computing budget:
- Maximum 8GB peak VRAM
- Batch size 32
- Training time: up to 2 hours
- Total initial computation: up to 4 hours

Next planned experiment: establish whether ConvNeXt has a measurable improvement on baseline CNN

Workable fallback: simplify training by reducing batch size to 16 or 8, if calibration is poor, refer more cases for specialist review

## Final Model Architecture:
-  Input: Single-channel (grayscale) 2D image tensor.
-  Stem: Conv2d (1 -> 32 channels, 4x4 kernel, stride 2, padding 1) and group normalisation. Halves spatial dimensions.
-  Stage 1 (32 channels): Two consecutive ConvNeXtBlock modules (using depth-wise convolution with a 7x7 kernel, layer normalisation, 1x1 channel expansion by 4, GELU activation, and 1x1 projection, with a residual connection).
-  Downsample 1: Conv2d (32 -> 64 channels, 2x2 kernel, stride 2) and group normalisation. Halves spatial dimensions.
-  Stage 2 (64 channels): Two consecutive ConvNeXtBlock modules.
-  Downsample 2: Conv2d (64 -> 128 channels, 2x2 kernel, stride 2) and group normalisation. Halves spatial dimensions.
-  Stage 3 (128 channels): Three consecutive ConvNeXtBlock modules.
-  Downsample 3: Conv2d (128 ->256 channels, 2x2 kernel, stride 2) and group normalisation. Halves spatial dimensions.
-  Stage 4 (256 channels): Two consecutive ConvNeXtBlock modules, followed by adaptive average pooling forcing spatial dimensions to 1x1 globally.
-  Classifier: Dropout (with rate 0.5) followed by a linear layer to output the classification logits.

This architecture was the result of many intermediate models. Expansion was increased from 2 to 4, stride was reduced from 4 to 2 to keep a higher spatial resolution initially, the data was augmented (it wasn't to begin with), added a drop path, added label smoothing to the cross entropy loss criterion to force the network to remain generalisable, and weight decay was increased from 0.001 to 0.03. The number of ConvNeXt blocks in each stage was also increased.

## Dependencies Required
These are the dependencies required to reproduce the results:
- Python (v3.11.16)
- PyTorch (v2.14.0)
- Torchvision (v0.29.0)
- NumPy (v2.4.6)
- Scikit-learn (v1.9.1)
- Matplotlib (v3.11.2)
- Pillow (v12.3.0)

#### Reproducibility of Results
The results are reproducible if you have access to the ADNI dataset, and those files are split into the same test and training folders as the ones used in this report. Through the use of random_state=42 in the train/validation split, these results are reproducible.


## Example Inputs, Outputs, and Visual Plots
### Inputs
Input data consists of 1 channel grey-scale 256x240 JPEG MRI slice images such as the following:

<img width="256" height="240" alt="374536_95" src="https://github.com/user-attachments/assets/b37af1bd-925f-457c-8875-fc29b126e5e8" />

### Visual Plots
#### Training and Validation Accuracy for ConvNeXt
<img width="2100" height="1200" alt="convnext_accuracy_curves" src="https://github.com/user-attachments/assets/734cbe84-d32d-49c7-9e59-e30183e8492a" />

#### Confusion Matrix for ConvNeXt
<img width="1500" height="1200" alt="convnext_confusion_matrix" src="https://github.com/user-attachments/assets/a99a7889-b95a-455b-bcc4-cc47c07cffc9" />

#### Test Set Class Probability Distributions for ConvNeXt
<img width="3000" height="1200" alt="convnext_probability_distributions" src="https://github.com/user-attachments/assets/0e6368d2-c556-4a57-8620-4f93877e2ebb" />

#### Coverage Risk of ConvNeXt
<img width="1800" height="1200" alt="convnext_coverage_risk" src="https://github.com/user-attachments/assets/dca4bacf-6790-453f-a12a-1ec9f12a4a4c" />


## Preprocessing of Data
• Description of any specific pre-processing used, with references if any. Justify your training, validation, and testing splits of the data.
To prepare the ADNI MRI slices for the ConvNeXt model, the following data pre-processing and augmentation pipeline was used: 
- Standardise the spatial dimensions of the MRI slices to 224x224 to match the input size expected by standard ConvNeXt architectures
- Data Augmentation (RandomRotation, RandomAffine, ColorJitter) to introduce minor geometric variations (slight rotations and translations up to 5%) and photometric adjustments (brightness and contrast) to simulate minor differences in head positioning, scanner calibration, and slice angles, helping prevent overfitting and improving model robustness (note: standard medical imaging classification pipelines commonly use these to adapt convolutional networks, as seen in the paper by [Basheera et al.](https://alz-journals.onlinelibrary.wiley.com/doi/10.1016/j.trci.2019.10.001) which utilises ADNI pre-processed subsets)
- Normalise(mean=0.1156, std=0.2229): Centres pixel distribution using training data mean and standard deviation to stabilise gradient descent and speed up convergence.


I used the 70% training, 15% validation, and 15% testing split (ensuring that it was split patient-wise to avoid data leakage) because it strikes an optimal balance for deep learning pipelines in medical imaging. By allocating 70% of the data for training, it allows the ConvNeXt model to sufficiently learn complex structural biomarkers of cognitive decline across a diverse patient pool. 15% validation provides a suitably sized independent subset to monitor loss and implement early stopping to prevent overfitting. This leaves 15% for testing, which is a suitable size for a completely unseen and unbiased set to evaluate the model's generalistion.

## Investigation of the Open Research Dilemma
The following table compares the benchmark CNN model performance against the ConvNeXt model under identical evaluation splits:
||CNN baseline model|ConvNeXt|
|---|---|---|
|Test accuracy (%)|69.14|76.57|
|Parameters|23650|1760226|
|Time taken to train (s)|754.3|2335.6|
|Macro F1|0.69|0.75|
|AD precision|0.67|0.69|
|AD recall|0.78|0.96|
|CN precision|0.73|0.93|
|CN recall|0.61|0.57|
|AUROC|0.76|0.81|

It can be seen that ConvNeXt performs better than the CNN baseline model. It does take much longer to train, however, that is expected as it is a more complex model. It still is over-predicting AD and classifying a large number of NC cases as AD, but overall, it performs much better than the baseline. 

#### Resource Profiling Table
||ConvNeXt|
|---|---|
|Peak GPU VRAM (MiB|2459|
|Parameter count|1760226|
|Inference latency (ms)|1.88|

While being more computationally intensive than the baseline, it is still relatively efficient for a complex model.


– In-depth qualitative error autopsies analyzing 3–5 representative failure cases from the dataset (diagnosing root triggers and failure modes).
### Failure Cases
#### Case #1: False Positive
<img width="256" height="240" alt="1331222_92" src="https://github.com/user-attachments/assets/609e94a8-b4da-477a-8238-3b7c5a680ea6" />

This was misclassified as AD with a confidence of 96.36%. The MRI slice shows relatively prominent ventricular spaces, particularly the lateral ventricle, alongside visible cortical sulci and CSF spaces. The overall brain structure remains distinguishable, without an obviously severe pattern of widespread cortical atrophy. Root Trigger/Failure Mode: Overreliance on Ventricular Enlargement. The model may interpret the prominent ventricular spaces as evidence of AD-related cerebral atrophy. This reliance on a single structural feature, without sufficiently considering surrounding anatomical context, may have lead to the false positive classification.

#### Case #2: False Negative
<img width="256" height="240" alt="1252573_103" src="https://github.com/user-attachments/assets/57562d72-9e81-4b41-a467-b782c8be5b5d" />

This was misclassified as NC with a confidence of 96.38% when the patient actually has AD. Looking at the scan, it has unclear artifacts/uneven brightness on the right, meaning that the model probably has trouble classifying it and wrongly classified as NC despite having AD. Root Trigger/Failure Mode: Domain Shift & Intensity Bias. The model relies heavily on low-level pixel intensity distribution rather than pure high-level geometry, confusing artifact-induced contrast variations with pathology.


#### Case #3: False Positive
<img width="256" height="240" alt="1188738_96" src="https://github.com/user-attachments/assets/0ff94726-7678-4f04-ac83-0cdbc395bddb" />

This was misclassified as AD with a low confidence of 50.18% The MRI scan shows relatively prominent CSF spaces, particularly around the central and posterior brain regions. The cortical folds remain relatively well-defined, with no obvious widespread cortical thinning. Root Trigger/Failure Mode: Sensitivity to CSF Space Enlargement. The model may interpret the prominent dark CSF-filled region near the brainstem and cerebellum as an indicator of abnormal brain volume loss. However, this appearance may reflect normal anatomical variation rather than AD-related atrophy. The relatively low confidence (50.18%) also suggests that the extracted features provide an ambiguous distinction between AD and NC.

### Engineering Recommendation and Trade-Off
The evaluation reveals that while ConvNeXt achieves better aggregate metrics (higher AUROC and Macro-F1) compared to the CNN baseline, it suffers from uncalibrated overconfidence on borderline neuroimaging cases. Specifically, when handling ambiguous structural scans, the model frequently outputs softmax probabilities > 0.90 even when incorrect.

Operational Reject-Option Rule (The Safety Threshold): To prevent unflagged false negatives in clinical triage, implement an empirical confidence threshold of 0.85, where if it is > 0.85, automatically route through clinical screening pipelines. If the confidence is low/ambiguous (< 0.85), automatically flag and route for human expert review (neurologist validation).

Impact: Utilising our risk coverage curve data (see Coverage Risk of ConvNeXt in Visual Plots), rejecting the bottom 20% lowest-confidence predictions drives clinical error rates down to almost zero.

#### Final Recommendation & Trade-Off
Architecture Choice: Deploy ConvNeXt. Despite its higher parameter count and higher computational overhead, its superior feature representation gives a higher AUROC, which is vital for clinical safety margins. Couple ConvNeXt with the proposed uncertainty-based reject-option rule to prevent unflagged false negatives.

Project Manager Action Item: Approve production integration with human-in-the-loop fallback for any inference falling below 85% confidence, ensuring patient safety while leveraging automation for clear-cut cases. Before deployment, the over-predicting of AD needs to be fixed.


## Artificial Intelligence Usage Disclosure
ChatGPT was used for helping scaffold the data loading functions. I verified the accuracy by verifying that there was no data leakage (see screenshot below).
<img width="661" height="402" alt="image" src="https://github.com/user-attachments/assets/1c5156eb-aa1e-464e-9cd8-beac8a35cbc5" />

Gemini and ChatGPT were used for debugging syntax. I used my knowledge of what the code should look like to see if the suggested code made sense.
Gemini was used to help write docstrings. I read them to make sure that they were accurate.
Gemini was used to help with failure cases. I made sure that they made sense and were accurate before including them.
