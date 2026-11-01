1. User Need Scope & Acceptance: 
This model will help radiotherapists to plan their radiotherapy by outlining the exact, precise segments of the prostate where there is a tumor, that will need to be irradiated. The model will mark segments to irradiate given a scan. I am to create a model with a dice score >= 0.75, patient-level splitting to avoid data leakage and compare the performance of the model to the improved U-net model implemented by "Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge". The model must have a fast inference time since it is expected to be used in real-time to irradiate tumors, and its must be compact enough to be used on an MRI machine

2. Model Choice and Course Concepts: 
I will use a CNN architechture since it is best suited towards segmentation tasks according to the course lectures. I will use a dice loss function for the same reason as recommended by the lectures.I intend to implement the medium difficulty 2D improved Unet task. So, since I am working with 2D images, I will use 1 channel. I will split the datasets per patient to avoid data leakage which would train the model on a specific patient, preventing generalization

3. Preliminary Feasibility Evidence: An initial data audit, baseline smoke-test experiment, or
measured resource estimate. A fully trained final model is not required.


4. Risks, Budget & Fallback: Key technical risks, computing budget, the next planned experiment,
and a workable fallback.
