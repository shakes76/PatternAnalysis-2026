# ConvNeXt for classification of Alzheimer's Diseases and Congitive Normal on 2D MRI brain slice scans

## Project Description

Alzheimer’s Disease (AD) causes irreversible neurode generation marked by progressive hippocampal and cortical atrophy with ventricular enlargement.
Automated classification of T1-weighted structural brain MRIs into AD versus Cognitively Normal (CN) aids early screening. 
The primary challenge is extracting discriminative anatomical biomarkers amidst significant natural head shape and age variation between patients, while strictly avoiding patient data leakage across multi-slice extractions.

## User Need and Scope

The intended users are neurologists and radiologists who analyse MRI brain scans to detect and diagnose the onset of Alzheimer's disease.
Automated classifications of T1-weighted structural brain MRIs into AD vs CN will help these users in early screening and diagnosis of Alzheimer's disease in patients.

This project will allow the intended users to have an automated assistance that classifies only high confidence cases. Ambiguous cases are re-routed to the specialists for human review.

# The Model

## ConvNeXt

This project implements ConvNeXt (Liu et al., 2022). ConvNeXt is a modernised pure CNN architecture that integrates elements from Vision Transformers.  It modernises a standard ResNet model towards a Vision Transformer model, however is constructed entirely from standard ConvNet modules.

The key innovations that led to ConvNeXt are as follows:
- **Patchify Stem**: replaces 7x7 convolutions with 4x4 convolutions (with stride = 4) to create non-overlapping patches to improve efficiency
- **Stagewise Architecture**: 4 stage structure where feature maps are processed at progressively lower resolutions
- **Inverted Bottleneck Blocks**: expansion occurs before depthwise convolutions. Each block expands hidden dimensions using pointwise linear convolutions.
- **Layer Normalisation** instead of batch normalisation
- **GELU Activation**
- **Depthwise Separable Convolutions**: reduces the number of parameters, which reduces computation and allows the network to be deeper and wider

ConvNeXt architectures have been proven to match or surpass standard vision transformers on accuracy, while retaining the computational efficiency and hardware optimisations of CNNs.

The model used was the convnext-tiny variant.

### Architecture

<img width="409" height="204" alt="Screenshot 2026-10-30 164945" src="https://github.com/user-attachments/assets/e72b6ce8-e558-464a-914b-5913160ea721" />   
  
*Figure 1: ConvNeXt architectural flow*

Figure 1 shows the ConvNeXt architecture. MRI brain scans were passed into the network as input. It was then prepared for deeper processing by passing it through a 2D convolutional layer and layer normalisation (4x4 non-overlapping patchify convolution). 
It was then passed through 4 ConvNeXt blocks, each designed to extract increasingly complex features. Downsampling layers in between each ConvNeXt block allow the model to find higher level patterns by reducing spatial dimensions and increasing feature channels. Finally, global average pooling and a linear classifier produce the final output.
 
Instead, of duplicating the grayscale MRI images across 3 channels (as was done in the ImageNet trained ConvNeXt), my implementation modies the patch stem to accept 1 input channel. This reduces the number of parameters and memory usage. 

# Project Data

The ADNI dataset was used in this project. 

The provided dataset (on the Rangpur cluster) has been preprocessed and contains 2 classes: Alzheimer's Disease (AD) and Cognitive Normal (CN). Each patient is associated with 20 2D slices of their MRI brain scan.
Filenames in the dataset are <patient_id>_<slice_id>.jpeg. The patient id was used to split data and verify there no leakage between the preprocessed training and testing sets (`verify_leakage.py`).

The training set contains 1076 patients. This will be split so that the training set contains 80%: 860 patients. The validation set will then have 216 patients. The testing set contains 450 patients. Since the sample size is quite small, augmentations were be performed, only on the training dataset.

Early experiements showed high validation accuracy (up to 95%) while test accuracy reamined in the low 70s.
To ensure uniform distribution when splitting training and validation sets, pixel intensity was analysed at the patient level. The training set was split into 5 bins and the validation set was created by randomly sampling from each bin.

## Data Preprocesing

Augmentations were performed on the traininng data only. These included:
- RandomRotation and RandomAffine to simulate natural variations in patient head positioning
- ColorJitter to simulate tissue contrast between different scanners
- RandomErasing to act as a dropout, and ensure the network did not overfit to localised regions

Images were also normalised to have a mean and standard deviation of 0.5, to ensure pixel intensities were scaled to a consistent range. This stabilises training and allows the model to converge more efficiently.

## Acceptance Criteria

**The Dilemma**: models frequently achieve deceptively high aggregate test accuracy while concealing catastrophic overconfidence on borderline, ambiguous, or out-of-distribution cases.

To address this, the following acceptance criteria were defined:
- **Zero Data Leakage**: 100% patient level separation across data splits
- **Classification Accuracy**: 80% classification accuracy of AD and CN images.
- **Operational Feasibility**: Training completes in half a day with peak GPU VRAM limited to 16GB

## Risks

Risks associated with the project include:
- Data leakage between training sets
- Class imbalance may cause bias towards majority class
- Unequal distribution between datasets may cause inflated accuracy readings

The project has been implemented in a manner that mitigates these risks.

Fallback Model: Since ConvNeXt is build on a ResNet Architecture, a suitable fallback would be ResNet-18 or ResNet-50.

# Training and Inference

## Final Hyperparameters

The model was trained using the following:
- Learning rate: 0.0005
- Batch size: 16
- Epochs: 100
- Weight decay: 0.05 on AdamW optimiser
- Cross entropy loss with 0.1 label smoothing
- Cosine Annealing learning rate scheduler with linear warmup (10 epochs) 
- Drop path: 0.1

## Training Results

<img width="4200" height="1500" alt="convnext_training_curves" src="https://github.com/user-attachments/assets/5f25c17e-5da2-43cf-97e9-e60bdac6aea6" />
    
*Figure 2: Cross-entropy loss (left) and classification accuracy (right) across 100 training epochs. The red dashed line denotes final test set performance.*

Figure 2 shows loss and accuracy for training, validation and testing.

Training loss drops to 0.23 from 0.69 and validation loss follows closely, dropping to 0.18 from 0.61. The variations in the validation loss can be explained by the batch size of 16, which causes noisy weight updates.
Validation loss is lower than training loss because a dropout of 0.1 was used during training only.

Training and validation accuracy increase with minimal divergence. This suggests that the model was not overfitting.

## Test Results

The model recieved a final test accuracy of **77.18%**, with a patient level test accuracy of **78.22%**.
Although this is much lower than validation accuracy and loss, it can be explained by a few possible reasons:
- The test set may have a different distribution to the testing and validation sets
- There may be a different class imbalance distribution between the test and training sets

<img width="1650" height="1440" alt="convnext_test_cm" src="https://github.com/user-attachments/assets/39679a69-fd67-4b7b-b62b-66d447d55cb7" />
<img width="1800" height="1500" alt="convnext_test_roc" src="https://github.com/user-attachments/assets/576deda1-c102-4762-828c-1e6cff515b9d" />
    
*Figure 3: Test confusion matrix (above) displaying high AD precision alongside conservative AD recall; ROC curve (below) demonstrating an AUROC of 0.8251.*

**Classification Report**
| Metric / Class | Precision | Recall (Sensitivity) | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| NC | 0.7027 | 0.9491 | 0.8075 | 4,540 slices |
| AD | 0.9195 | 0.5913 | 0.7197 | 4,460 slices |
| Slice-Level Accuracy | — | — | 77.18%*| 9,000 slices |
| Patient-Level Accuracy | — | — | 78.22% | 450 patients |
| Test AUROC | — | — | 0.8251 | — |

- AD precision refers to the number of scans with AD out of all scans the model flagged as having AD
- AD recall refers to the number of scans identified as having AD (out of all scans that were AD)
- NC recall refers to number of scans identified as NC (out of all NC scans)
- NC precision refers to the number of scans with NC out of all scans the model flagged as having NC
- F1 score balances recall and precision to a value between 0 and 1

AUROC tests the model across all possible decision cutoffs from 0% to 100%.  
A high AUROC score means that the model correctly ranks an Alzheimer's scan as higher risk than a healthy scan 82.5% of the time.

The figures above show that the model is biased twoards predicting NC over AD, which is undesirable in clinical screening.
**Future work** could include lowering the decision threshold for the AD class from 0.5 down to 0.3. 

## Experimental iterations

- Early runs used 3-channel replicated grayscale images and 40 epochs. While competitive, 3-channel input tripled memory bandwidth without adding new information. 
- Testing label smoothing (0.1), DropPath (0.1, 0.2, 0.3), and spatial transforms demonstrated that ConvNeXt scales gracefully without overfitting over 300 epochs
- Modifying the patch stem to 1 native grayscale channel eliminated redundant parameters
- Adding linear warmup (3, 10 epochs) followed by cosine annealing decay stabilized feature learning. However, using batch size 128 consumed 17.7 GB VRAM and caused generalization degradation
- Dropping batch size to 16 provided beneficial stochastic gradient noise, boosting slice accuracy by +2.54% while reducing GPU memory to just 2.5 GB
- Testing an aggressive learning rate (0.01) led to divergence where loss flattened at 0.693 and the model collapsed to predicting 100% NC. Using a learning rate of 0.0005 established the optimal results (75.76% slice acc, 76.89% patient acc)
- Partitioning patients into 5 volumetric intensity quintiles completely eliminated train-validation distribution skew

# The Open Research Dilemma

Models frequently achieve deceptively high aggregate test accuracy while concealing overconfidence on borderline or ambiguous cases.
In clinical screening, missing an early Alzheimer's case (false negative) is far more damaging than a routine false positive.
  
## CNN vs ConvNeXt

The ConvNeXt architecture was benchmarked against a simple CNN architecture, under identical splits and hyperparameters.
- Both models ran for 40 epochs and were trained using AdamW optimiser, cross entropy loss criterion (no label smoothing), a learning rate of 0.0001 and a batch size of 32.

| Metric | CNN | ConvNeXt | Performance |
|---|---:|---:|---|
| Training Runtime | 9.7 mins | 51.8 mins | ConvNeXt took 5.3× longer to train |
| Slice-level Test Accuracy | 73.41% | 73.62% | Similar performance, with a +0.21% gain for ConvNeXt under early 40-epoch training |
| Test Loss | 1.6831 | 1.1283 | ConvNeXt achieved 33% lower test loss |
| AD Precision | 86% | 82% | CNN is more conservative |
| AD Recall | 56% | 60% | ConvNeXt achieved a 4% gain |
| NC Recall | 91% | 87% | CNN achieved a 4% gain |

ConvNeXt is significantly larger than the CNN, so 40 epochs is insufficient for full convergence. 

When trained to full convergence (the current model) ConvNeXt performs significantly better:
- Slice Accuracy: Reaches 77.18% 
- Patient-Level Accuracy: Reaches 78.22%: 
- Test AUROC: Reaches 0.8251 
- AD Precision: Reaches 91.95%, dramatically reducing false alarms 

## Resource Profiling

The final model was profiled and results were as follows.
| Metric | Value |
|---|---:|
| Trainable Parameters | 27,818,594 |
| Peak GPU VRAM | 2,523.50 MB |
| Training Runtime | 7,621.16 s (127 min) |
| Inference Latency | 1.30 ms/slice (26 ms/patient scan) |

## Error Autopsies
  
<p align="center">
 <img width="4560" height="2400" alt="convnext_sample_predictions" src="https://github.com/user-attachments/assets/833e3251-fd26-4396-8c64-427b2fca9829" />
 <img width="4560" height="2400" alt="convnext_misclassified_slices" src="https://github.com/user-attachments/assets/017ea31d-0c77-406f-9f60-97a3a6b84189" />
</p>    
  
*Figure 4: Model failure cases (below) and sucessful predictions (above)*
  
Figure 4 above compares the model’s failure cases (false positives and false negatives) against its successful predictions.
* Catastrophic Overconfidence on Incorrect Cases: In every misclassified case, the model predicted the incorrect label with extreme confidence (90.2% to 94.6% on False Positives; 93.6% to 95.0% on False Negatives). Standard softmax outputs completely fail to reflect diagnostic uncertainty on out-of-distribution slices.
* False Positive Triggers (NC misdiagnosed as AD): Healthy elderly patients frequently exhibit normal age-related ventricular dilation (ex vacuo ventricular enlargement) and widened sulcal spaces. ConvNeXt keyed heavily into enlarged ventricular boundaries as an AD proxy, falsely categorizing these elderly normal cases as diseased with >94% confidence.
* False Negative Triggers (AD misdiagnosed as NC): Slices taken from peripheral cranial cuts (extreme lateral or superior planes) lack the hippocampal formation and medial temporal lobes. Without visible atrophy signatures in these outer slices, ConvNeXt defaulted to NC with >93% confidence.
* Scanner Inhomogeneity & Partial Volume Effects: Variations in white-to-gray matter contrast ratios misled the patch stem tokenization on boundary slices.

## Actionable Engineering recommendation

For clinical screening, based on AD probability:
- High confidence NC cases should be logged without specialist intervention
- High confidence AD cases should be re-routed for specialist confirmation
- Borderline cases should be rerouted for specialist review

# Dependencies
All experiements were conducted on GPUs on the Rangpur cluster.

Software versions used to run the experiements were

```python >= 3.10
torch >= 2.0.1
torchvision >= 0.15.2
numpy >= 1.24.3
scikit-learn >= 1.3.0
matplotlib >= 3.7.1
pillow >= 9.5.0
```

## Reproducibility

To ensure reproducibility, a fixed random seed (42) was used..

To run the model:

1. Clone the repository, and ensure all dependencies are intalled
2. Navigate to the convenext_47496390 directory
3. To run ConvNeXt with the final hyperparameters:
        `python train.py --model convnext --epochs 100 --batch_size 16 --lr 5e-4 --weight_decay 0.05 --data_root /home/groups/comp3710/ADNI/AD_NC` 
4. To run iference using trained weights and visualise results:
        `python predict.py --model convnext --weights convnext_best.pth --data_root /home/groups/comp3710/ADNI/AD_NC --batch_size 16 --save_dir ./diagnostic_plots`

The inference script (`predict.py`) generates:
- `convnext_test_cm.png`: Annotated confusion matrix heatmap.
- `convnext_test_roc.png`: Receiver Operating Characteristic curve with AUROC calculation.
- `convnext_misclassified_slices.png`: 8-sample qualitative failure case autopsy grid.
- `convnext_sample_predictions.png`: 8-sample grid of correct diagnostic predictions.

# ADNI ConvNeXt Feasibility Review
some sections here may duplicate other parts of this README {see [#259](https://edstem.org/au/courses/38242/discussion/3688339)}

## User Need and Scope
The intended users are neurologists and radiologists who analyse MRI brain scans to detect and diagnose the onset of Alzheimer's disease.
Automated classifications of T1-weighted structural brain MRIs into AD vs CN will help these users in early screening and diagnosis of Alzheimer's disease in patients.

This project will allow the intended users to have an automated assistance that classifies only high confidence cases. Ambiguous cases are re-routed to the specialists for human review.

## Acceptance Criteria
**The Dilemma**: models frequently achieve deceptively high aggregate test accuracy while concealing catastrophic overconfidence on borderline, ambiguous, or out-of-distribution cases.

The following acceptance critera is defined to address this dilemma:
- 100% patient level separation across data splits
- 80% classification accuracy of AD and CN images
- Ambiguous MRI scans are re-routed for human review
- Training runtime is 60 seconds per epoch (Reyes & Sánchez, 2024)

## Model Choice and Course Concepts
**Chosen Difficulty**: Hard

**Dataset Configuration**:
The dataset has been preprocessed and contains 2 classes: Alzheimer's Disease (AD) and Cognitive Normal (CN).
The dataset has been preprocessed and contains brain scans that are 2D slices.

Filenames in the dataset are <patient_id>_<slice_id>.jpeg. The patient id will be used to split data and verify there is no leakage between the preprocessed training and testing sets.

Normalisation and augmentations will be performed only on the training dataset to prevent overfitting and make the model robust and generalisable.

**Model Architecture**:
The ConvNeXt architecture was introduced by Meta AI and UC Berkley. It modernises a standard ResNet model towards a Vision Transformer model, however is constructed entirely from standard ConvNet modules.

**Course Concepts**:
- Data splitting and preventing data leakage
- Data processing, including normalisation and augmentation on training set
- Training and Validation of model
- CNNs, ResNets and Transformers
- Testing/evaluation of model performance

## Preliminary Feasibility Evidence

**Data Audit**:
The data available on Rangpur has already been split into training and testing sets. This split was analysed (using `verify_leakge.py`) and there is no data leakage.
The training set will be further split into training and validation sets. Split will be done by patient ID to ensure no data leakage.

The training set contains 1076 patients. This will be split so that the training set contains 80%: 860 patients. The validation set will then have 216 patients. 
The testing set contains 450 patients.
Since the sample size is quite small, augmentations will be performed, only on the training dataset.

The dataset contains 2D slice images. This can be loaded into a tensor with 3 channels (to match the input for ConvNeXt). The images are grayscale, so the same value will be duplicated three times.

Image sizes were found to be 256x240 pixels, using `file ADNI/AD_NC/train/AD/218391_78.jpeg`. This size will be kept to ensure that resizing doesn't remove important features or distort the image.

### Baseline smoke-test experiment
A baseline experiment was conducted using a simple CNN architecture. The model was trained for 10 epochs.
Training and validation over 10 epochs, with validation every second epoch took 167 seconds on the Rangpur cluster.
Test accuracy with the CNN architecture was only 70%.

## Risks, Budget and Fallback

**Risks**:
- Data leakage between training sets
- Class imbalance may cause bias towards majority class

**Computing Budget**:
Experiements and training will be performed on Rangpur GPU nodes. Training will target completion in 24hours.

**Fallback Model**: 
Since ConvNeXt is build on a ResNet Architecture, a suitable fallback would be ResNet-18 or ResNet-50.

**Next planned experiment**:
Next steps are to create the ConvNext model and run the current training, testing and validation procedures on it.
Based on the results, each hyperparameter will be tuned.
The model's detection of false positives and negatives will also be examined.

# References
GeeksForFeeks. (2025, July 15). ConvNeXt. Retrieved from GeeksForFeeks: https://www.geeksforgeeks.org/computer-vision/convnext/  
Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition , (pp. 11966–11976).  
Mayo Clinic Staff. (2026, August 1). Alzheimer's disease. Retrieved from Mayo Clinic: https://www.mayoclinic.org/diseases-conditions/alzheimers-disease/symptoms-causes/syc-20350447  
Pytorch. (2025, April 23). Vision. Retrieved from GitHub: https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py  
Reyes, D., & Sánchez, J. (2024). Performance of convolutional neural networks for the classification of brain tumors using magnetic resonance imaging. Heliyon, 10(3), e25468. https://doi.org/10.1016/j.heliyon.2024.e25468   
  
# AI Disclosure
Google Gemini (3.8) was used to assit with the project.
Gemini was used to assist in building the dataloaders, ensure a patient level split and create a validation set uniformly by creating bins of the training set. It was then used to adapt the PyTorch ConvNeXt model and ensure alignment with the ConvNeXt paper.
Gemini was also used to create plotting functions, evaluate accuracy at the patient level, determine computational profiling and implement argument parsing.
Gemini was used to understand the slice images and autopsy them. Gemini was also used to create markdown tables.
All AI generated code was verified.
