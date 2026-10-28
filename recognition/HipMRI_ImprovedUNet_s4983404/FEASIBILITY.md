# HipMRI segmentation: feasibility review

Student: s4983404 | Draft for tutor discussion

**1. Aim and success criteria**

I aim to produce prostate outlines that an imaging specialist can review and correct. I will compare a basic U-Net with an Improved U-Net to see whether the extra complexity gives better boundaries at a reasonable computing cost.

- Keep all images from the same patient in one data split.
- Target prostate test Dice >= 0.75 and report Dice and IoU for each label.
- Compare the models and inspect 3-5 examples of boundary errors.
- Keep GPU memory below 16 GB and record training and prediction time.

**2. Model and data**

I will use HipMRI 2D slices and their segmentation masks. A basic 2D U-Net (Easy) is the baseline. A 2D Improved U-Net (Normal), based on Isensee et al. (2018), is the main model. This uses course ideas including convolution, skip connections, normalization and validation. Both models will use the same data split and preprocessing.

**3. Preliminary evidence**

The image and mask pairs can be loaded together. The supplied split separates 31 training, 4 validation and 3 test patients. A short training trial has run successfully on Rangpur. These checks support feasibility, but do not establish reliability on new patients.

**4. Risks and next steps**

The test group is small, and some boundaries are unclear. I need to confirm the label numbers and slice direction. Next, I will review the model comparison, local errors and computing cost. Runs are budgeted at up to 20 minutes each. The basic U-Net is the fallback if the Improved model gives little benefit. 3D work is optional if time allows.

References: Ronneberger et al. (2015), U-Net; Isensee et al. (2018), BRATS 2017 contribution. Data: HipMRI Study.
