# ADNI ConvNeXt Feasibility Review

## Project Description
Alzheimer’s Disease (AD) causes irreversible neurode generation marked by progressive hippocampal and cortical atrophy with ventricular enlargement.
Automated classification of T1-weighted structural brain MRIs into AD versus Cognitively Normal (CN) aids early screening. 
The primary challenge is extracting discriminative anatomical biomarkers amidst significant natural head shape and age variation between patients, while strictly avoiding patient data leakage across multi-slice extractions.

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

## References
- https://www.sciencedirect.com/science/article/pii/S2405844024014993
- https://share.gemini.google/OlLE9OZGsMfJ

