# HipMRI Improved 2D U-Net Prostate Segmentation

Prototype Improved 2D U-Net (normal difficulty) model for pelvic/hip area segmentation with a 
focus on the prostate segmentation specifically. A basic 2D U-Net (easy difficulty) is utilized 
as a baseline to identify if this model results in any identifiable improvement over the standard
model. The models are trained on A100s on the Rangpur Cluster

---

Student ID: s4648169  
Full Name: Matthew Zhang

## Introduction

---

Radiotherapy requires millimeter-level delineation of the prostate clinical target volume while 
tightly sparing adjacent critical Organs at Risk (OARs: bladder, rectum, femoral heads). 
Under-contouring risks tumor recurrence; over-contouring irradiates rectal mucosa and bladder wall, 
causing severe radiation toxicity.

A single global Dice score can mask critical boundary failures, on the surface an acceptable Dice 
score might appear to correctly predict large central slices of the target organ while failing 
catastrophically in the apex and base slices. Either leaking into the bladder and/or the rectum, 
causing severe radiation proctitis or cystitis. Or under-contouring, which risks cancer recurrence.
Both of these scenarios can incur costs on the medical institution.

This project aims to segment the HipMRI 2D slices into six distinct classes, labeled 0 to 5 (with
label 5 representing the prostate). The goal is to determine if an improved 2D U-Net can give a
better segmentation than the standard 2D U-Net (Ronneberger et al., 2015)[1] at a better or
satisfactory computing cost. Since only a tiny portion of each slice contains the prostate, 
and a small portion of images contain the prostate at all. The model needs to be able to cope
with a severe class imbalance. To show this we want the prostate label's Dice score to be at 
least 0.75 on the test dataset.

### The Model

---

The Improved 2D U-Net is based off that of Isensee et al. (2017) [2], converted from 3D to 2D. It 
uses two pre-activation 3x3 convolutions with dropuout 0.3, followed by strided downsampling, 
InstanceNorm2D and a LeakyReLU with a negative slope of 0.01. It also uses context modules,
a bilinear Upsampler2D followed by two more convolutions a 3x3 then a 1x1 and three summed
segmentation heads which are then Upscaled as well (Deep Supervision).

This is a visual representation of the U-Net structure in Isensee et al. (2017)[2]:
![Isensee UNet.png](segmentation%20UNET%2046481698/plots/Isensee%20UNet.png)

### Files

---

- dataset.py 
  - This file loads the NIFTI files, pairs the images and the segmentation masks, one-hot encodes
  the masks and crops mismatching images sizes to 256x128. Builds Dataloaders for train, test and
  validation datasets
- modules.py
  - Creates the network blocks and sets up the model architecture for the Basic and Improved
  2D U-Nets. As well as methods to calculate Dice Score/Loss and Combined CrossEntropy and Dice
  loss
- predict.py
  - Prediction evalution of the trained models with dice scores per label and over-contouring and 
  under-contouring percentages.
- train.py
  - Training/validation loops, optimising and learning rate scheduling, saving the best model by
  combined Cross Entropy + Dice loss

## Feasibility Report
### User Need, Scope & Acceptance

---

The intended users of this model would be Imaging Specialists and radiotherapy professionals that
require precise millimeter-level delineation of the prostate volume while avoiding Organs at Risk (OaR).
By building this improved U-Net model and comparing it with a basic U-Net I hope to dtermine weather the
extra complexity and prediction accuracy of the model outweighs its computational cost.

#### Testable acceptance criteria:

1. Have the target prostate Dice score >= 0.75
2. Have each epoch run in less than one minute
3. Keep GPU usage under 16GB
4. The improved model draws reasonable boundaries

### Model Choice

---

The models to be used will be a basic 2D U-Net (easy difficulty) used as a baseline for comparison and an improved
2D U-Net (normal difficulty). These models uses course ideas such as convolution, skip connections, normalisation and
validation splits.


### Preliminary Findings

---

There are no issues loading the data from the pairs of images and segmentation masks. The HipMRI dataset contains
31 training, 4 validation, and 6 test patients in the data

### Risks

---

There is a high class imabalance and a high number of slices containing low or no amount of the target. Due to the 
nature of the images they have a high noise and thus boundaries between classes are unclear visually.

### Dataset

---
The HipMRI 2D slices dataset is split into three sets (train, validation, test) with is no patient overlap within 
the three splits. A read only root at: /home/groups/comp3710/HipMRI_Study_open/keras_slices_data. 
They contain (31, 4, 3) defined patients over (11460, 660, 540) slices. Of all the files in the dataset, 12,600 files have
the resolution 128x128, with the remaining 60 having the resolution 128x144. To rectify this those 60 files have been
center-cropped to fall in line with the other files in the dataset.

#### Class Distribution and Balance

<div align="center">

  |  Image/Mask Pairs  | Train | Validation | Test | Total Pairs |
  |:------------------:|:-----:|:----------:|:----:|:-----------:|
  | Including Prostate | 4152  |    254     | 195  |    4601     |
  | Excluding Prostate | 7308  |    406     | 345  |    8059     |
  |    Total Pairs     | 11460 |    660     | 540  |    12660    |

</div>

All the patients in each of the sets are all separate and disjoint from the others, all slices from the same week lie
within their respective sets. The data is checked everytime get_dataloaders in run in dataset.py

## Comparison

---

Figures 1 and 2, The training curves for the Basic 2D U-Net and Improved 2D U-Net respectively

|Basic U-Net|Improved U-Net|
|---|---|
|![training_curves.png](segmentation%20UNET%2046481698/plots/Basic%20UNet/training_curves.png)|![training_curves.png](segmentation%20UNET%2046481698/plots/Improved%20UNet/training_curves.png)|

Through the two graphs above, There are two major differences in the training of the models.
While the basic 2D U-Net has a steady curve, so the training is steady. Compared to the Improved U-Net which has a
very steep initial 

The Dice score is calculated by 2TP/(2TP+FP+FN) and IoU is TP/(TP+FP+FN) from logits during training.

#### Basic 2D U-Net

  |     Label      | Dice Score |  IoU   |
  |:--------------:|:----------:|:------:|
  | 0 (Background) |   0.9968   | 0.9936 |
  |    1 (Body)    |   0.9833   | 0.9672 |
  |    2 (Bone)    |   0.9302   | 0.8695 |
  |  3 (Bladder)   |   0.9614   | 0.9256 |
  |   4 (Rectum)   |   0.8430   | 0.7286 |
  |  5 (Prostate)  |   0.8536   | 0.7447 |

Mean Epoch Time: 110.3s

Peak GPU Usage: 3.00GB


#### Improved 2D U-Net

|     Label      | Dice Score |  IoU   |
|:--------------:|:----------:|:------:|
| 0 (Background) |   0.9971   | 0.9943 |
|    1 (Body)    |   0.9842   | 0.9689 |
|    2 (Bone)    |   0.9356   | 0.8791 |
|  3 (Bladder)   |   0.9630   | 0.9286 |
|   4 (Rectum)   |   0.8286   | 0.7075 |
|  5 (Prostate)  |   0.8484   | 0.7367 |

Mean epoch time: 111.2s

Peak GPU Usage: 2.14GB

It appears that both the U-Nets fail to meet the time requirements in this scenario. However, the datasets are
not cached, thus resulting in the dataset having to decompress the files each time Dataloader[i] asks for a slice
This is repeated every epoch, increasing the required time to process each epoch. Perhaps if the data were cached,
then maybe the models would fulfill the time requirements. 

Otherwise, it appears that the improved 2D U-Net only takes a few seconds more training per epoch than the basic U-Net.

Unfortunately it appears that my improved 2D model performs worse than the basic U-Net but its peak GPU usage is 1GB
less than the Basic U-Net. This may make the model much better computational much better

## Prostate Failure Cases

Figure 3, Improved U-Net Model Performance on Slices
![class_balance_test.png](segmentation%20UNET%2046481698/plots/Improved%20UNet/class_balance_test.png)

As seen in the first row of Figure 3, the most accurate prediction occurs during a middle slice where the prostate has
the most pixels it will have. There is still some over-contouring and under-contouring resulting in the prostate prediction
retaining its size. However, it still overlaps slightly with the bladder, meaning any user of the model will still have to
personally audit any prediction the model has to make.

The worst predictions come when the slice has a high class imbalance against the prostate, where the prostate has only a
few pixels in the slice. In this case the model will be unable to predict any pixels to be the prostate. This means that
users will have to also audit all slices even if the model does not determine there is any prostate to check for any 
false negatives.

Finally, there's a case where there is no prostate, in which the model successfully predicts there to be no Prostate in
the image.

From the figure, this shows that the lack of any 3-dimensional context can lead to under contouring. The vertical axis
could provide more information which might allow a 3D model to predict the prostate where the 2-dimensional models
may struggle.


## Conclusion

---

Although the project has some flaws, lower Dice score than the standard U-Net there are some benefits to the improved U-Net too.
One such benefit is a lower computational peak cost allowing the model to be run on lower tier hardware. This makes the model
much more open. Allowing many more people to use it even in situations where higher spec hardware may be harder to come by.
In that respect I do tacitly recommend this model in this specific circumstance. Otherwise the standard U-Net is still more
accurate in its predictions than my model.

## References

---

[1] O. Ronneberger, P. Fischer, and T. Brox, “U-Net: Convolutional networks for biomedical image segmentation,” in Proc. MICCAI, 2015, pp. 234–241.

[2] F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein, “Brain tumor segmentation and radiomics survival prediction: Contribution to the BRATS 2017 challenge,” in Brainlesion: Glioma, Multiple Sclerosis, Stroke and TBI Models, 2018, pp. 287–297.