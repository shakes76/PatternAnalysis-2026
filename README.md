# Pattern Analysis

I will attempt the easy target and increase the difficulty as i complete them, the line below will be the final version difficulty that i submit
Target Difficulty: Normal

Setup -------------------
The HIP MRI data should be in a file structure like
Target2.2_s47694413\Data\HIP MRI\keras_slices_data
This file should then contain all the keras_slices_seg_** and keras_slices_**
I can only 100% confidently say that latest version of train.py and predict.py are the most accurate versions of the code
however if needed to test earlier versions of the U-net you can possible download the modules.py file on the github commit and
run that with the train.py and predict.py, with a few modifications to train.py.
Uncomment lines 105 and 106 and comment lines 109-113 to test deep vision

Architecture: Improved U-Net

Title: Anatomical Contouring and Radiotherapy Planning.

Radiotherapy planning relies on identifying anatomical structures in medical images. Manually outlining these structures can be time-consuming, 
and inaccurate boundaries may affect treatment planning, especially near organs at risk. This project investigates whether a convolutional neural 
network can automatically segment anatomical regions in MRI images to assist radiotherapy professionals.

The intended users are radiotherapists and other clinicians involved in contouring and treatment planning. The model is a research prototype, 
not a clinically validated or autonomous diagnostic tool. Because it is still technically an AI its predictions would require an expert to review
any outputs it produces before any clinical use. However it could help achieve a cut down in time for potential experts in almost
immediately identifying key parts of the scan.

The dataset is treated as a six-class semantic segmentation problem: each image pixel is assigned one of six label IDs (0–5). 

------------------- Data Processing ---------------------------
The project uses HipMRI images and corresponding segmentation masks which are used for verification on training and testing. 
The model expects a single-channel image and predicts six class logits per pixel as mentioned before.
It should also be noted that alot of the images aren't the same dimension so alot were either forced to
scale up or down its vertical/horizontal axis to make sure that the input channels remain the same for
all cases.

------------------- Basic U-Net Architecture: -----------------
The baseline is an encoder–decoder convolutional network with three downsampling stages, a bottleneck, and three upsampling stages.
Each encoder stage applies two 3 × 3 convolutions with ReLU activations followed by 2 × 2 max pooling.
Decoder stages use transposed convolutions to upsample features, concatenate matching encoder features through 
skip connections, and apply two further convolutions.
A final 1 × 1 convolution generates six output logits per pixel.
The baseline feature widths are 32 → 64 → 128 → 256 at the bottleneck, then 128 → 64 → 32 through the decoder.
The predicted segmentation is obtained with argmax over the six output channels.
------------------- Training and Evaluation --------------------------
The models are trained using 2D MRI images and integer segmentation masks. 
The discussed training code uses CrossEntropyLoss, backpropagation and an optimizer to update model weights.
During training, the mean validation dice is used as a performance check and if the mean is higher
than previous epochs, then it saves the weights.
The learning rate is 0.001 and the training is run on 10 epochs.
------------------- Results ----------------------
Training basic U-Net:
Parameter 1 (Dice Scores):
Validation Dice:
  Class 0: 0.9953
  Class 1: 0.9823
  Class 2: 0.9328
  Class 3: 0.8810
  Class 4: 0.8778
  Class 5: 0.8208
Mean Validation Dice: 0.9150

Running the weights gathered from the training data and running the weights against all the test
images produced the following dice scores
The score from each individual class was the following

===== Dice Scores =====
Dice 0: 0.9972
Dice 1: 0.9846
Dice 2: 0.9351
Dice 3: 0.9631
Dice 4: 0.8365
Dice 5: 0.8449
Average inference time per image: 1.19 ms
Average Dice Score: 0.9269
Average Dice Percentage: 92.69%
Peak VRAM: 674.42 MB
Parameters ran on validation: 1925190

Parameter 2 (Run Time):
Running these results fresh gave a processing time of 1.19ms

Parameter 3 (Peak VRam):
Running the results gave a constant Peak VRAM of 674.42 MB of data per run of predict.py

Improved U-Net Training
-------------------- First Modification -------------------------
NOTE: Due to some mistakes the individual scores of the First and Second Modifications on their own are wrong, However the scores showing both of them
combined and onwards are correct.
The first improvement was trying to add residuals to the convolution layer. The convolutional block adds its transformed features to a shortcut 
projection of the original input, helping gradient flow through the network.

Epoch 20/20, Loss: 0.0252, Time: 64.87s
Validation Dice:
  Class 0: 0.9956
  Class 1: 0.9830
  Class 2: 0.9353
  Class 3: 0.9087
  Class 4: 0.8658
  Class 5: 0.8195
Mean Validation Dice: 0.9180
Upon Testing it it showed that the test set it was not as good either
Average inference time per image: 1.44 ms

===== Dice Scores =====
Dice 0: 0.9970
Dice 1: 0.9843
Dice 2: 0.9346
Dice 3: 0.9595
Dice 4: 0.8392
Dice 5: 0.8271

Average Dice Score: 0.9236
Average Dice Percentage: 92.36%
Peak VRAM: 738.76 MB
Parameters ran on validation: 2013106

With increase VRAM and lower Dice Scores it's clear the residual setup currently implemented is not the best

----------- Experiment Modification 1 and modification 2 together -----------------
With both modifications together it actually showed an improvement over the base U-Net architecture.
Auxiliary segmentation heads at intermediate decoder stages contribute additional loss terms with each
convolution upsample step 
during training, encouraging useful predictions at multiple scales.
    Validation Dice:
    Class 0: 0.9961
    Class 1: 0.9824
    Class 2: 0.9336
    Class 3: 0.8795
    Class 4: 0.8823
    Class 5: 0.8243
    Mean Validation Dice: 0.9164

===== Dice Scores =====
    Dice 0: 0.9971
    Dice 1: 0.9840
    Dice 2: 0.9336
    Dice 3: 0.9625
    Dice 4: 0.8393
    Dice 5: 0.8471
    Average inference time per image: 1.44 ms
    Average Dice Score: 0.9273
    Average Dice Percentage: 92.73%
    Peak VRAM: 738.76 MB
    Parameters ran on validation: 2013106


----------- Modification 3 with Modification 2 and 1 all together
This is probably one of the more harder modifications to understand but this one essentially has
a sort of map to it called the attention gate which is another sort of learning feature that is
multiplied with a feature that can help spatially tell which part that the feature map to focus on.
Validation Dice:
  Class 0: 0.9953
  Class 1: 0.9825
  Class 2: 0.9356
  Class 3: 0.8554
  Class 4: 0.8808
  Class 5: 0.8315
Mean Validation Dice: 0.9135

===== Dice Scores =====
Dice 0: 0.9970
Dice 1: 0.9849
Dice 2: 0.9366
Dice 3: 0.9621
Dice 4: 0.8591
Dice 5: 0.8572
Average inference time per image: 1.60 ms
Average Dice Score: 0.9328
Average Dice Percentage: 93.28%
Peak VRAM: 802.86 MB
Parameters ran on validation: 2035397

------------- Failure Cases -----------------------
The failure case images can be found in the image folders
The images labled "Case # 'Model'" look at each different class
found in only 1 particular image to look at each failed pixel for each class
and the "'model' Image #" are used just to look at the ground truth and the
prediction model for different images

To investigate the limitations of the segmentation models, several MRI slices were visually 
examined using their original images, ground-truth segmentation masks, predicted masks and 
class-specific error overlays. The analysis focuses on differences between the standard 
U-Net and the improved architecture incorporating residual connections, deep supervision and attention gates.

        Case 1 — Small Anatomical Regions

Both architectures successfully identify the larger anatomical regions, producing segmentation
masks that visually resemble the ground truth. However, differences are visible in the smaller
central structures, particularly around their boundaries and relative positions.
These results suggest that smaller anatomical regions may be more sensitive to 
architectural changes than large, continuous structures.

        Case 2 — Boundary Accuracy

The baseline and improved models both identify the major anatomical regions, but their predicted
boundaries differ from the ground truth in several locations. These errors may arise from the 
difficulty of distinguishing neighbouring tissues with similar MRI intensity characteristics.

        Case 3 — Class Confusion

The predictions show differences in how pixels are assigned between neighbouring anatomical 
classes, particularly around the central structures. These differences may represent false-positive
or false-negative segmentation errors, depending on the selected class.
This case highlights the importance of examining individual anatomical classes rather than relying exclusively on mean Dice.

        Case 4 — Small-Region Detection

A small region visible in the ground-truth mask is absent or differently located in the predicted
segmentation. This demonstrates a limitation of the model when identifying small anatomical regions. 
Such errors may contribute relatively little to the overall segmentation score while being important
for the accuracy of individual classes.

        Overall Findings

The visual analysis suggests that both architectures are capable of identifying the major anatomical regions.
However, errors remain around boundaries and smaller structures. The architectural modifications change the
predicted segmentation in several locations, although the images alone do not establish that one model is
consistently more accurate. Further class-specific Dice measurements and quantitative false-positive/false-negative
comparisons are required to determine whether the additional model complexity produces meaningful improvements.



3. Feasibility review — draft, expand to approximately one page
3.1 Technical Feasibility

This project investigates the feasibility of using a 2D U-Net neural network to automatically segment anatomical structures in 
MRI images for radiotherapy planning. The objective is to reduce the manual effort required for anatomical contouring while 
providing predictions that clinicians can review and correct.
The baseline U-Net achieved a mean validation Dice score of 91.50% and a reported test Dice score of 92.69%. These results 
demonstrate that the base model can learn meaningful anatomical features quite well.
Three architectural modifications were investigated: residual connections, deep supervision and attention gates. The final 
architecture incorporating all three achieved a reported test Dice score of 93.28%, an improvement of 0.59 percentage points 
over the baseline. However, its mean validation Dice was 91.35%, slightly below the baseline's 91.50%.

3.2 Computational Feasibility

The baseline U-Net contained 1,925,190 parameters, with a measured average forward-pass inference time of 1.19 
milliseconds per image and peak GPU memory allocation of 674.42 MB.
The final improved architecture contained 2,035,397 parameters, required 1.60 milliseconds per image, and reached 
802.86 MB of peak GPU memory allocation.
Compared with the baseline, this represents approximately:
      - 5.7% more parameters
      - 34.5% longer measured inference time per image
      - 19.0% greater peak GPU memory allocation
Note the reported inference times represent model forward-pass measurements and do not include the complete processing 
workflow, such as loading images, preprocessing, visualization and clinician review. Consequently, these results should 
not be interpreted as total clinical processing times.

3.3 Clinical Feasibility

The intended application is to assist radiotherapy professionals by generating preliminary anatomical contours that can be 
reviewed before use in treatment planning.
Although the experimental Dice scores are promising, they do not establish that the system is clinically safe. Segmentation 
errors involving small anatomical structures which were class 4 and 5, which had a dice score of 0.82 - 0.88 show the greatest
risk and could potentially have serious consequences without having an experts opinion.

The model also processes individual 2D slices rather than complete 3D MRI volumes (hard task). This reduces computational complexity but 
limits its ability to incorporate anatomical continuity between adjacent slices.
As stated earlier, before any real-world clinical application, the system would require independent validation and mandatory review of its 
predictions by qualified professionals before any operations could be performed. So strictly speaking this tool
should only be used as a time saving tool and not a replacement for the professional in this case.

3.4 Overall Feasibility

The results suggest that a 2D U-Net is computationally practical and technically promising as a research 
prototype for MRI anatomical segmentation.
The final improved model achieved the highest reported test Dice, but this improvement was relatively small 
and accompanied by increased inference time, parameter count and GPU memory requirements. Furthermore, 
the improved model did not outperform the baseline on mean validation Dice while training.
Therefore, the additional architectural complexity cannot yet be considered conclusively beneficial.
The standard U-Net remains a strong candidate because of its competitive segmentation performance and 
lower computational requirements.
Overall, the project demonstrates the potential of automated MRI segmentation as a clinical assistance tool, 
but further validation and failure-case analysis are necessary before its suitability for clinical deployment can be established.

4. Engineering recommendation — provisional
Current recommendation: retain the standard 2D U-Net as the preferred baseline until the improved architectures demonstrate a reproducible 
advantage on clinically relevant structures or boundaries. In the recorded validation comparison, the baseline achieved a slightly higher 
mean Dice (0.915) while using less GPU memory than the combined modified models. Attention gates produced small improvements in some classes 
but increased measured peak memory.

This recommendation is provisional. A final decision should use held-out patient-level test results, per-class Dice/IoU, boundary error 
inspection, corrected inference latency, memory and parameter count. If the improved model consistently reduces important boundary errors, 
the additional computational cost may be justified even without a higher mean Dice.

5. AI Disclosure
The AI that has been used quite a bit in this is the visual studio code auto complete AI and it has been able to
auto complete alot of my Dataloading file and has helped with the module aswell although i did look up the
general architecture that a 2D U-net should follow. 
As for testing the AI code specifically it mostly comes from testing the algorithim all together
Chatgpt was used for coming up with the ideas for the improved models and explaining the way they worked
and what changes to the code would be required. It also produced the code for visualising the failure case plots
and alot of the summary for the README.md file


6. References
"UNet - Line by Line Explanation" Jeremy Zhang 2019
https://medium.com/data-science/unet-line-by-line-explanation-9b191c76baf5

