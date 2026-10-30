# 3D improved u-net for HipMRI Segmentation

## Introduction
1. Introduction
    1.1 Project Background
        In this project, we are using medical image segmentation to determine the prostate using the
        given images. Medical image segmentation is the process of partitioning a 2D or 3D medical 
        scan into distinct, meaningful regions or extracting specific regions. 

    1.2 Project Objectives
        By creating an improved 3D U-Net and using downsampled HipMRI 3D data to achieve a target 
        dice >= 0.7 on the test set. This improved 3D U-Net is later then compared with a normal 
        3D U-Net as a baseline.

2. Dataset and Preprocessing
    2.1 HipMRI Dataset 
    Firstly, the Dataset is found inside the Rangpur that contains the MRI images. A total of 211 
    images has been used in the testing. The MRI images size are 128x128x64 and is split between 6 
    segmentation classes, labeled from 0 to 5 where class 5 being the prostate.

    2.2 Data Splitting
    The dataset is divided into training, validation and testing subsets using a patient-level 
    split to prevent data leakage between the subsets. The number of volumes in each split are 
    145 for training, 31 for validation and 35 for testing.

    2.3 Preprocessing
    Before training, the MRI volumes were preprocessed to ensure that it have the same input 
    dimensions and intensity. Firstly, the MRI intensities were normalized using zscore. This was done
    to standardize the intensity values. Secondly, the MRI volumes were resized to 128 * 128 * 64 using 
    trilinear interpolation. This reduces the computational cost and GPU memory. Lastly, the 
    segmentation masks were resized using nearest-neighbour interpolation instead of trilinear 
    interpolation because nearest-neighbour interpolation preserves the original integer class labels.

3. Model Architecture
    3.1 Baseline: Standard 3d U-Net
    The baseline model uses a standard 3D U-Net architecture consisting of an encoder and a decoder. 
    The encoder is responsible for extracting features through reducing spatial dimensions while the
    decoder is responsible for reconstructing the segmentation to the original resolution. Meanwhile,
    skip connections are used to transfer info from the encoder to the decoder. 

    3.2 Intermediate: Residual 3D U-Net
    The intermediate model improves upon the standard 3D U-Net by replacing the standard convolutional
    blocks with residual blocks. Residual connections allos information to bypass the convolutional areas
    acting like a shortcut. A 1x1x1 convolution is used when the number of inputs and outputs are different
    and letting them add together as a tensor. This architecture is used to improve training stability.

    3.3 Advanced: Residual 3D U-Net with Deep Supervision
    The advanced model extends the intermediate Residual #d U-Net by introducing deep supervision. Deep 
    supervision involves calculating additional losses from the intermediate decoder outputs. 
    Two auxiliary segmentations heads were added to produce segmentation predictions at different 
    solutions. During training, the auxiliary predictions are used together with main loss to become 
    the total loss. During inference, your model uses only the final segmentation output. The 
    auxiliary outputs are used for training supervision, not for generating the final prediction.

4. Training Methodology
    4.1 Training Configuration
    All three models were trained using the AdamW Optimizer with a learning rate of 0.0001. The batch 
    size was set to 1 and each model was trained for 20 epochs. Training was performed on both A100
    and Comp 3710 GPU partition using the Rangpur computing cluster whenever which one is available. 
    The models were configured to predict 6 segmentation classes.

    4.2 Loss Function
    The models were trained using a combination of Cross-Entropy Loss and Dice Loss. Cross-entropy Loss
    is used to measure the difference between the predicted segmentation classes and the ground-truth 
    labels for each voxel. Meanwhile, Dice Loss is used to improve the overlap between the predicted 
    segmentation and the ground-truth regions.For the advanced model, deep supervision introduces two 
    additional auxiliary losses from intermediate decoder layers.

    4.3 Model Selection
    During training, the models were evaluated using the validation dataset. The Dice score for Class 5, 
    which represents the prostate, was used to determine the best-performing model. The model checkpoint 
    with the highest validation prostate Dice score was saved and later used for testing. This ensures 
    that model selection is based on validation performance rather than training loss alone.

5. Experimental Results
    5.1 Segmentation Performance
    All three models were evaluated using the same test dataset containing 35 MRI volumes. Dice score 
    was used to measure the overlap between the predicted segmentation and the ground-truth segmentation.
    All three models achieved the target Dice score of 0.70. The intermediate model achieved the highest 
    prostate Dice score of 0.7880, followed by the advanced model at 0.7699 and the baseline model at 0.7304.

    5.2 Dice and IOU Comparison
    The Dice score and Intersection over Union (IoU) were calculated for each segmentation class. For the 
    baseline model, the Dice scores for Classes 0 to 5 were 0.9972, 0.9771, 0.8844, 0.9375, 0.6649 and 0.7304 
    respectively. For the intermediate model, the Dice scores were 0.9969, 0.9787, 0.8840, 0.9538, 0.7996 and 
    0.7880. Meanwhile, the advanced model achieved Dice scores of 0.9975, 0.9770, 0.8485, 0.9539, 0.7854 and 
    0.7699. The intermediate model achieved IoU scores of 0.9938, 0.9583, 0.7921, 0.9117, 0.6660 and 0.6502 
    for Classes 0 to 5 respectively. The advanced model achieved IoU scores of 0.9950, 0.9551, 0.7369, 0.9119, 
    0.6466 and 0.6259. Based on these results, the intermediate model achieved better prostate segmentation 
    performance than the advanced model in terms of both Dice score and IoU. 

    5.3 Computational Performance
    Computational Performance for both models were evaluated by comparing their parameter counts, GPI memory
    usage and inference times. The intermediate model contained 5996614 parameters and the advanced
    model have an additional 588 parameters compared to the intermediate model placing him at 5997202 parameters.
    Both model recorded approximately 0.756GB of peak allocated GPU. The advanced model have 25.99 milliseconds 
    per volume and the intermediate have an average of 25.16 milliseconds. This shows that both have quite the 
    similar computational requirements. It should be noted that the intermediate checkpoint was evaluated 
    using the advanced model implementation with auxiliary heads instantiated. Therefore, the GPU memory and 
    inference time measurements should be considered approximate comparisons rather than fully independent 
    architectural benchmarks.

    5.4 Training and validation curves
    Training and validation curves were generated to monitor the learning performance of the models throughout 
    the 20 training epochs. THe training loss represents how well the model learns from the training dataset.
    These curves can be used to investigate whether the models converged during training and whether there 
    were signs of overfitting. A decreasing training loss generally indicates that the model is learning 
    from the training data, while changes in validation loss and Dice score help determine whether this learning 
    improves performance on unseen data.

6. Failure Analysis
    6.1 Identification of the Failure Class
    To investigate the limitations of the advanced model, the prostate Dice score was calculated for each 
    individual test volume. The three volumes with the lowest prostate Dice scores were selected for further 
    analysis. Volume 17 achieved a Dice score of 0.4762, Volume 12 achieved 0.5757, and Volume 13 achieved 
    0.5899. These scores were considerably lower than the overall prostate Dice score of 0.7699 achieved by 
    the advanced model. This indicates that the model performed less accurately on certain MRI volumes, even 
    though its overall test performance exceeded the assignment target.

    6.2 Visualization of Segmentation Errors
    To examine the segmentation errors, visualisations were generated for the three selected failure cases. 
    Each visualisation contains the original MRI image, the ground-truth prostate segmentation, the predicted 
    prostate segmentation and an error overlay. In the error overlay, false positives are highlighted in red, 
    representing regions incorrectly predicted as prostate. False negatives are highlighted in blue, 
    representing prostate regions that the model failed to identify. These visualisations allow the predicted 
    segmentation to be compared directly with the ground truth and help identify where segmentation errors 
    occur. The generated images for Volumes 17, 12 and 13 should be inserted into this section.

7. Discussion
    7.1 Effect of residual connections
    The intermediate Residual 3D U-Net achieved a prostate Dice score of 0.7880, compared with 0.7304 for 
    the standard 3D U-Net. This improvement suggests that introducing residual connections was beneficial for 
    prostate segmentation in this experiment.Residual connections allow information to bypass convolutional 
    layers through shortcut connections. These connections can improve gradient flow during backpropagation 
    and help the network learn features more effectively. As a result, residual connections may improve 
    training stability and segmentation accuracy. The results indicate that the intermediate model was able 
    to achieve better prostate segmentation performance than the standard baseline model under the tested 
    conditions.

    7.2 Effect of Deep Supervision
    The advanced model introduced deep supervision by adding auxiliary segmentation heads to intermediate 
    decoder layers. The purpose of deep supervision was to provide additional training signals and encourage 
    the model to learn useful features at different resolutions. However, the advanced model achieved a 
    prostate Dice score of 0.7699, which was lower than the intermediate model's score of 0.7880. This indicates 
    that the addition of deep supervision did not improve prostate segmentation accuracy in this experiment.
    One possible explanation is that the auxiliary losses influenced the learning process in a way that did not 
    benefit the final segmentation output. The selected auxiliary loss weights may also require further 
    adjustment. Additional experiments would be necessary to determine whether different weights or training 
    configurations could improve performance.

8. Feasibility Review
    The results show that 3D prostate MRI segmentation is computationally feasible using the Rangpur GPU resources. 
    Both models required approximately 0.756 GB of GPU memory during evaluation, with inference times of around 
    25–26 milliseconds per volume. The intermediate model achieved the highest prostate Dice score of 0.7880 while 
    using fewer parameters than the advanced model. Therefore, the intermediate model is recommended due to its 
    better accuracy and similar computational requirements. However, further testing and clinical validation are 
    required before practical medical use.

9. Reproducibility
    9.1 Project files.
    The project was implemented using Python and PyTorch. The main source files are modules.py, dataset.py, 
    train.py and predict.py. The modules.py file contains the neural network architectures and their components. 
    The dataset.py file handles loading and preprocessing the MRI images and segmentation masks. The train.py 
    file performs model training and validation, while predict.py is responsible for testing the trained models, 
    calculating evaluation metrics and generating segmentation visualisations.

    9.2 Dataset
    The HipMRI dataset was accessed through the UQ Rangpur computing cluster at 
    /home/groups/comp3710/HipMRI_Study_open. The dataset was divided into training, validation and testing subsets 
    containing 145, 31 and 35 MRI volumes respectively. Before being passed into the models, the MRI volumes were 
    normalised using Z-score normalization and resized to 128 × 128 × 64. The segmentation masks were resized 
    using nearest-neighbour interpolation to preserve their class labels.

    9.3 Training Procedure
    Training was performed using PyTorch on the Rangpur computing cluster. The models were trained using the 
    AdamW optimizer with a learning rate of 0.0001, a batch size of one and a total of 20 epochs. During each epoch, 
    the models were trained using the training dataset and evaluated using the validation dataset. The checkpoint 
    with the highest validation Dice score for Class 5 was saved as the best-performing model. For the advanced 
    model, deep supervision was implemented by combining the main segmentation loss with two weighted auxiliary 
    losses.

    9.4 Testing Procedure
    After training, the saved model checkpoints were loaded and evaluated using the testing dataset containing 
    35 MRI volumes. The predict.py script was used to calculate Dice and IoU scores for the segmentation classes. 
    The script was also used to measure computational performance and generate visualisations of the segmentation 
    errors. To fully reproduce the experiments, the required Python dependencies, software versions and exact 
    commands used for training and testing should also be documented.

10. Conclusion
    This project investigated prostate MRI segmentation using three different 3D U-Net architectures, consisting 
    of a standard 3D U-Net, an intermediate Residual 3D U-Net and an advanced Residual 3D U-Net with deep 
    supervision.All three models successfully achieved the required prostate Dice score of 0.70. The baseline 
    model achieved 0.7304, the intermediate model achieved 0.7880 and the advanced model achieved 0.7699. The 
    intermediate Residual 3D U-Net achieved the highest prostate segmentation accuracy, demonstrating the 
    potential benefits of residual connections. Although the advanced model introduced deep supervision to 
    improve feature learning, it did not achieve better segmentation accuracy than the intermediate model.
    Overall, the results suggest that residual connections can improve the performance of 3D U-Net models for 
    prostate MRI segmentation. However, further improvements and additional testing would be required to increase 
    segmentation reliability and assess the model's suitability for clinical applications.

11. AI Usage Disclosure
    ChatGPT was used to assist with understanding technical concepts related to 3D U-Net architectures, residual 
    connections, deep supervision and segmentation evaluation metrics. It was also used to provide programming 
    guidance, assist with debugging, interpret experimental results and improve the structure and wording of the 
    project documentation. The final implementation and experimental results should be checked against the 
    submitted source code and recorded outputs to ensure accuracy.
