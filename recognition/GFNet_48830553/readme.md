# GFNET Project

Created by Yoshihito Sugiyama  
ID: 48830553  

## Files

### dataset.py

Contains functions to preprocess, split, and load data.

### modules.py

Contains classes of the models and helper functions that will be used in other scripts.

### predict.py

Includes the testing pipeline to analyse models for performance in accuracy and speed.  
Flags:  

- --model ("RN18", "GFNET")
- --version ("BEST", "LATEST")

### train.py

Includes the training pipeline to train and save models.  
Flags:  

- --model ("RN18", "GFNET")

## Usage

### Requirements

- python=3.14.7
- torch=2.14.1
- torchvision=0.29.1
- torcheval=0.0.7
- matplotlib=3.11.2

### Instructions

After installing the dependencies within a python environment, run "train.py" with appropriate flags ("RN18" for ResNet18, "GFNET" for GFNet) to train either a ResNet18 model or a GFNet model. Run "predict.py" with appropriate flags ("RN18" for ResNet18, "GFNET" for GFNet, "BEST" for model with highest validation accuracy, "LATEST" for latest saved model)to test a ResNet18 model or a GFNet model to analyse performance.

#### Training

```bash
python train.py --model "GFNET"
```

#### Inference

```bash
python predict.py --model "GFNET" --version "BEST"
```

## Introduction

### Problem

This report will focus on the overconfidence dilemma of machine learning models in automated triage and determine whether advance machine learning models are suitable for MRI classification. The classification will be based on human conditions between Alzheimer's (AD) and Cognitively Normal (NC) on the ADNI dataset.

### Model

The chosen baseline model is Residual Neural Network 18-Layers (ResNet18) and the chosen advanced model is Global Filter Network (GFNet). ResNet18 is chosen as a baseline model for it's overall performance and inference speed in classification [1]. GFNet is chosen due to it's fast O(nlog(n)) run time complexity with the accuraccy of a normal transformer model. GFNet has a run time O(nlog(n)) because the model replaces the self-attention layer with a global filter in the Fourier domain, which is an element-wise multiplicaiton, with the only drawback being the O(nlog(n)) complexity of the Fast Fourier and Inverse Fast Fourier Transformations [2].

### Evaluation Process

To answer the dilemma, confidence of each winning class prediction will be analysed to determine whether there is a significant difference between confident right answers and unconfident wrong answers. Furthermore, to answer whether an advanced machine learning models are suitable for MRI classification, inference time, precision, recall, and accuracy (0-1 loss)  will be measured. A machine learning model should have high accuracy to be useful for operational use to filter hard classification that can be handed to professionals instead. Moreover, high precision is required to reduce the extra work for professions in the event of overwhelming number of false positives, and high recall is required to ensure patients are not mislabeled as NC and left untreated [3].

### Data Processing

The data is stored as NC classes and AD classes for both provided training and testing datasets. The training dataset is further splitted into validation (20%) and training (80%) data loaders. Furthermore, during the data split, the datasets are splitted with respect to the patients to ensure no data leakage occurs. Moreover, the data is splitted with even distribution of classes in the validation and training data loaders. Furthermore, the AD classes will be classified as a 1 (true) and NC classes as 0 (false) in a binary classification.

## Process

Firstly, data processing and loading functions are created in the "dataset.py" file. Secondly, the main training loop function is constructed in the "train.py" file. Thirdly, the main testing function is created in the "predict.py" file. All repeated functions are saved in the "modules.py" file.

### Problems

#### Memory

Due to the memory restriction of a singular Nvidia A100, the max batch size had to be reduced.

#### Epoch

The initial max epoch was set to a small number (5) to determine feasibility, but was increased (250) once feasibility was validated.

#### Weight Decay

Weight decay is important to control overfitting; however, too much weight decay caused the model to not learn and was adjusted accordingly. Not all parameters such as positional embedding, bias, LayerNorm, and LayerScale required weight decay as those parameters were not capturing complex features and was removed from decay.

#### Droppaths

To improve regularisation and decrease overfitting, droppaths were introduced. Dropout and droppaths both improve regularisation, but dropout deactivates specific neurons while droppaths deactivates entire layer/blocks.

#### Data Augmentation

Data augmenation in the form of jitter, flips, affine transformation, erasing, and guassian blur are introduced to increase the effective training set with variation to reduce overfitting. With further investigation, random erasing is suspected to introduce too much variation and decreased performance of the model and was removed.

#### Learning rate

Learning rate was adjusted to a small number to allow smoother learning with smaller chance to overshoot minimas. Furthermore, warm up was introduced to allow the model to learn the appropriate directions of parameter changes.

#### Initial values

Initial values of parameters were adjusted for faster and stable convergence.

#### Patch size

Reduced embedding patch size of the GFNet for finer and granular patching to improve local feature extraction.

#### CUDA and Hardware

CUDA and Nvidia A100 related code are implemented to imporve training time by utilising the specific hardware capabilities

### Results

After training and testing using "train.py" and "predict.py", results were saved. Furthermore, graphs of the losses from each model and spread of the winning/predicted class confidences/probabilites were saved.

#### ResNet18

True positive confidence: 0.78 (Var: 0.0132, Std: 0.1147)  
True negative confidence: 0.68 (Var: 0.0133, Std: 0.1153)  
False positive confidence: 0.71 (Var: 0.0141, Std: 0.1186)  
False negative confidence: 0.63 (Var: 0.0142, Std: 0.1193)  
True confidence: 0.75 (Var: 0.0155, Std: 0.1243)  
False confidence: 0.70 (Var: 0.0149, Std: 0.1221)  
Precision: 0.59  
Recall: 0.88  
Train Accuracy: 0.89 | Val Accuracy: 0.91 |Test Accuracy: 0.64  
Train Loss: 0.38 | Val Loss: 0.40 | Test Loss: 0.68  
Inference time: 9.38  
Confusion Matrix: tensor([[1828., 2712.], [ 535., 3925.]], device='cuda:0')  

Losses:  
![Image not Found](/Images/BEST_RN18_graph-1.png)  

Confidence Spread:  
![Image not Found](/Images/BEST_RN18_confidence_spread-1.png)  

#### GFNet

True positive confidence: 0.87 (Var: 0.0093, Std: 0.0965)  
True negative confidence: 0.84 (Var: 0.0212, Std: 0.1455)  
False positive confidence: 0.81 (Var: 0.0159, Std: 0.1261)  
False negative confidence: 0.77 (Var: 0.0238, Std: 0.1543)  
True confidence: 0.86 (Var: 0.0140, Std: 0.1181)  
False confidence: 0.80 (Var: 0.0176, Std: 0.1328)  
Precision: 0.64  
Recall: 0.88  
Train Accuracy: 0.92 | Val Accuracy: 0.94 |Test Accuracy: 0.70  
Train Loss: 0.33 | Val Loss: 0.34 | Test Loss: 0.73  
Inference time: 9.52  
Confusion Matrix: tensor([[2366., 2174.], [ 514., 3946.]], device='cuda:0')  

Losses:  
![Image not Found](/Images/BEST_GFNET_graph-1.png)  

Confidence Spread:  
![Image not Found](/Images/BEST_GFNET_confidence_spread-1.png)  

#### Predictions for GFNet

False Positives:  
![Image not Found](/Results/GFNET/Wrong_Medium/sample_1087_true0_pred1_conf0.8405_1445758_96.jpeg)
![Image not Found](/Results/GFNET/Wrong_Medium/sample_1573_true0_pred1_conf0.8406_1387622_92.jpeg)

True Negatives:  
![Image not Found](/Results/GFNET/Correct_High/sample_1212_true0_pred0_conf0.9966_1186737_110.jpeg)
![Image not Found](/Results/GFNET/Correct_High/sample_1419_true0_pred0_conf0.9970_1446679_96.jpeg)  

False Negatives:  
![Image not Found](/Results/GFNET/Wrong_High/sample_4685_true1_pred0_conf0.9998_985197_94.jpeg)
![Image not Found](/Results/GFNET/Wrong_Low/sample_6469_true1_pred0_conf0.5001_399798_95.jpeg)  

True Positives:  
![Image not Found](/Results/GFNET/Correct_Medium/sample_4861_true1_pred1_conf0.9067_391567_94.jpeg)
![Image not Found](/Results/GFNET/Correct_Medium/sample_5525_true1_pred1_conf0.9067_1122616_106.jpeg)  

### Discussion/Analysis

Firstly, the ResNet18 had a test accuracy of around 64% with an inference time of 12 seconds, while GFNet had a test accuracy of 70% with an inference time of 9 seconds. Furthermore, the GFNet had slightly better precision of 0.64 than ResNet18 with a precision of 0.59. Overall, GFNet slightly performed better than ResNet18 with respect to test accuracy, inference time, and precision. When analysing the loss curves for both models, the validation and training accuracy were almost equally high, which means low overfitting; however, the test accuracy does not reflect. This mismatch between validation and testing results may be the result of data distribution mismatch between the provided training dataset and testing dataset.  

Secondly, the GFNet was not able to attain the 80% test accuracy minimum performance. Furthermore, the GFNet had a precision of 0.64 and recall of 0.88. The low precision indicates the model has a tendency to identify NC classes as AD classes. Recall of the model is high, but is not to standard of medical professions [3], which means the model still has a relatively high false negative rate. The high false negative rate is critical as medical professions will not tolerate false negatives as much as false positives [3]. Therefore, despite all the optimisations made to make the model generalise well, to not overfit, and reliably train, the model is not suitable for MRI classifications for AD and NC labels in the current state.  

Thirdly, by analysing the confidence spreads in the graph, it is clear that there is significant overlap between true predictions and false confidences. Therefore, it is not possible to cleanly differentiate wrong confidences/probabilities and correct confidences/probabilities. The overconfidence nature of machine learning models is not solvable with a simple confidence threshold based on this current GFNet model.  

Forthly, when analysing the false predictions with the true predictions, it appears the GFNet model may not be able to analyse the grooves of the brain in great detail. This inability to analyse the grooves may be caused by the global nature of the filters in the GFNet model, which lacks the ability to analyse local features. Furthermore, when analysing the false negatives, a completely different MRI scan type was identified in the test dataset, which could have contributed to the overall lower performance of the GFNet and ResNet18 models. Furthermore, transformers may perform worse in small datasets which could be the case for the provided training dataset [4].

## Conclusion

The GFNet model did outperform the ResNet18 model, even if it is by a small margin. Furthermore, due to the overall low test accuracy, low precision, and low recall, the GFNet in it's current state with the provided dataset is not suitable for MRI classifications. Moreover, the overconfidence of machine learning models cannot be simply adjusted by placing a confidence threshold. Finally, there is suspected data distribution mismatch between the provided test and training datasets that led to the overall performance degradation in both ResNet18 and GFNet models.

## Feasibility Report

*Name: Yoshihito Sugiyama*
*ID: 48830553*

### User Need, Scope & Acceptance Criteria

#### User

Patients for quicker and more reliable cancer checkups; doctors to reduce workload and allow focus on more difficult analysis

#### Purpose

To demonstrate that a model can reliably determine cancer/normal organs with an appropriate level of confidence

#### Acceptance Criteria

- If there is a significant difference between correct confident answers and wrong uncertain answers, a threshold can be placed to direct uncertain cases to doctors.
- If there are significant and reliable answers, the model may prove to be useful (0-1 Error and PR).
- If the inference time is shorter than the average doctor’s time to determine anomalous brain structure.

### Model Choice & Course Concepts

- By using FFT, GFNet can have the accuracy of a transformer model with the speed of FFTs
- No self-attention layer (replaced by global filter)
- No cross-attention layer (Classification problem)

### Preliminary Feasibility Evidence

#### ResNet18 (20 Epochs max)

- Final Train Acc: 0.7298
- Final Val Acc: 0.7220
- Final Test Acc: 0.6326
- Training Time: 809.85 seconds (13.50 minutes)

#### GFNet (20 Epochs max)

- Final Train Acc: 0.8356
- Final Val Acc: 0.7310
- Final Test Acc: 0.6258
- Training Time: 841.66 seconds (14.03 minutes)

### Risks, Budget & Fallback

- Less than 1 hour of training on the whole dataset with 20 epochs
- A100 GPU with 40GB - 80GB of memory
- Access to EAIT Labs computers, AWS GPUs, and Rangpur GPUs

## References

[1] T. Shahriar, "Comparative Analysis of Lightweight CNNs for Resource-Constrained Devices: Predictive Performance, Efficiency Trade-offs, and Initialization Effects," *arXiv preprint arXiv:2505.03303*, 2025. [Online]. Available: [https://arxiv.org/abs/2505.03303](https://arxiv.org/abs/2505.03303)

[2] Y. Rao, W. Zhao, Z. Zhu, J. Lu, and J. Zhou, "Global Filter Networks for Image Classification," *arXiv preprint arXiv:2107.00645*, 2021. [Online]. Available: [https://arxiv.org/abs/2107.00645](https://arxiv.org/abs/2107.00645)

[3] Duponte, "False negatives: The hidden risk in rapid diagnostics," Duponte, Nov. 2025. [Online]. Available: [https://duponteadn.com/en/false-negatives-the-hidden-risk-in-rapid-diagnostics/](https://duponteadn.com/en/false-negatives-the-hidden-risk-in-rapid-diagnostics/)

[4] N. Park and S. Kim, "How Do Vision Transformers Work?," *arXiv preprint arXiv:2202.06709*, 2022. [Online]. Available: [https://arxiv.org/abs/2202.06709](https://arxiv.org/abs/2202.06709)

## AI Reference

Gemini 3.8 Low, Medium, and High was used to create the code and suggest improvements on the training loop and models.

Gemini Auto was used to create the reference list.
