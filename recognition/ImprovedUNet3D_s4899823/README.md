# 3D Improved U-Net for Prostate and Organ-at-Risk Segmentation on HipMRI 

Keith Emmanuel Arputham (s4899823) 

## 1 Problem and algorithm 

### 1.1 Problem 

External-beam radiotherapy for prostate cancer needs the prostate and the nearby organs at risk, the bladder and rectum, to be contoured on every planning scan. Contouring by hand is slow. Errors also have direct clinical consequences, under  contouring the prostate risks the tumour coming back, and over contouring irradiates the rectum wall or bladder. The task is to label every voxel of a 3D pelvic MRI as one of six classes. Two things make it hard. First, the prostate is tiny, about 0.1% of the voxels in a scan. Second, its superior (base) and inferior (apex) ends have weak soft-tissue contrast against the bladder and pelvic floor. The PROMISE12 challenge identified the base and apex as the hardest parts of the prostate to segment (Litjens et al., 2014). 

### 1.2 The algorithm 

The Improved U-Net(Isensee et al., 2018) is an encoder & decoder convolutional network.It builds on the the U-Net (Ronneberger et al., 2015) and its 3D extension.The encoder repeatedly halves the spatial size with 2 stride concolutions and doubles the number of feature channels . Each level includes a pre activation residual block of two 3X3 convolution with drpout.These modules learn the image. The decoder upsamples back to full resolution . At each level, the encoder features are concated in through skip connections. so the decoder also knows where things are. 

A localisation module then mixes the two. Deep supervision adds segmentation outputs from three decoder levels together, which gives the deeper layers a direct training signal. Every convolution uses instance normalisation (Ulyanov et al., 2016) and leaky ReLU.My implementation takes a dims argument: with dims=3 every layer is 3D (Conv3d, InstanceNorm3d, trilinear upsampling) and the network sees whole volumes; with dims=2 the identical design runs on single axial slices. This makes the 2D model a controlled baseline: the only difference between the two models is 3D context. 

## 2 Feasibility review 

### 2.1 User need and acceptance criteria 

The radiotherapist who manually outlines the prostate, bladder and rectum in 2D on MRI scans of the pelvis region. The annotation assistant that generates the 3D outline for the radiotherapist to check and refine.It is not an autonomous system. 

The open dilemma is even if global Dice is high, there may be mistakes in the prostate edges and base and apical regions, and possible spillage of the contour into rectum/bladder. Does 3D context help fix these boundary issues compared to 2D? 

Acceptance criteria : 

1. The prostate Dice >= 0.7 with per-class Dice and iou reported for all 6 classes. 

2. The prostate Dice on the base and apex slices 20% of top and bottom of the slices that contains prostate. 

3. The fraction of predicted prostate that is really bladder or rectum. 

4. Trains on one Rangpur Gpu within a 6h job.Peak VRAm, parameter count and inference time are reported for both models. 

### 2.2 Model choice course concepts: 

Data : HipMRI 3D volumes from the course copy on Rangpur. 211 scans from 38 patients, 6 classes, downsampled ×0.5. 

Model : 3D Improved U-Net (Isensee et al., 2018), with residual context modules, instance normalisation, leaky ReLU and deep supervision. 3D convolutions see the continuity between slices, which is exactly what the base and apex need. 

Baseline: he same architecture in 2D,train on axial slices of the same volumes with the same patient split. 

Loss : Dice + cross entropy because of the extreme class imbalance.The checkpoint is selected by validation prostate Dice 

### 2.3 Preliminary feasibility evidence 

Data audit. The split is patient-level (seed 42), and disjointness is asserted in code: train 26 patients / 136 scans, val 6 / 48, test 6 / 27. Class voxel fractions: background 61.7%, body 34.1%, bone 3.4%, bladder 0.6%, rectum 0.15%, prostate 0.10%. Prostate is about 1 voxel in 1,000, so accuracy is meaningless as a metric and plain cross-entropy would ignore it. 

2D baseline, fully trained on Rangpur (A100-40GB, 100 epochs, batch 16): best val prostate Dice 0.815; bladder / rectum 0.926 / 0.853; 3.19 M params; peak VRAM 0.72 GB; 16.8 s/epoch (29 min). 

### 2.4 Potential risk 

1. 3D runs out of gpu memory 

2. 3D epoch too slow for the 6hr limit 

3. prostate under segmentation 

## 3 Pre-processing and splits 

### 3.1 Pre processing 

All the pre-processing steps occur in dataset.py.First, all volumes are down-sampled by 0.5 to 128 × 128 × 64 from 256 × 256 × 128. The images are interpolated using trilinear interpolation while labels use nearest neighbour interpolation to ensure that all the labels become an integer value for a particular class. Down-sampling helps follow the approach followed in the assignment, where down-sampled volumes were used, and all volumes can be accommodated in the GPU memory. Since MRI scans have no standard scale, they need to be normalized to zero mean and unit variance. 

For training only, I use slight augmentation. First, each volume is flipped left–right with a probability of 0.5, which makes sense because the pelvis is symmetric. The intensity values are further scaled randomly from 0.9 to 1.1 and offsetted by ±0.1. These manipulations emulate variations in scanning procedures. No augmentation is used for validation and test sets. 

There was one volume which required special treatment. Volume K019_Week1 consists of 144 slices compared to 128 slices in others, and hence after downsampling, it becomes 128 x 128 x 72 rather than 128 x 128 x 64. It creates two issues in 3D modelling. Firstly, the network reduces the volume by 4 times on every axis. As a result, 72 slices become 72 to 36 to 18 to 9 to 5 and the decoding step where 5 is upsized to 10 no longer matches skip connection consisting of 9 slices. Secondly, all volumes within a single batch need to have the same shape. In case of 3D training, I simply crop out 128 x 128 x 64 from each volume. With probability 0.5 the cropping is centered on the prostate and with probability 0.5 the cropping location is random. For the other 210 volumes which already have the correct dimensions (128 x 128 x 64), the crop operation does nothing. So for K019_Week1 volume the net sees slightly different 64 slices each epoch. At test time, I do not apply crop. Instead, I pad each axis to the nearest multiple of 16, run the network, and crop out the prediction.The 2D baseline is trained on every axial slice of the same pre-processed volumes 

### 3.2 Train/Validation/Test split 

Data has been split by patient, not by scan or slice (patient_split(), seed 42): 26 patients (136 scans) for training, 6 (48) for validation and 6 (27) for testing, about 70/15/15% of the 38 patients. Weekly scans of the same patient show almost identical anatomy. A random scan- or slice-level split would therefore put near-copies of test scans into training and inflate the scores. Assertions in the code check that no patient appears in two splits, and the split is saved to outputs/split.json. Validation is used only to pick the best epoch (by validation prostate Dice). The test set is used once, by predict.py, after training. I didn’t use the pre-made keras_slices_data, because it has its own split: slicing the same 3D volumes ensures the 2D and 3D models are tested on exactly the same 27 volumes, so they can be compared volume by volume. 

## 4 Method 

### <u>4.1 Architecture</u> 

|Component|Implementation|Source|
|---|---|---|
|get_layers(dims)|returns (Conv,InstanceNorm,Dropout) for<br>2D or 3D|lets one code path build<br>both models|
|ContextModule|IN to LeakyReLU to 3×3 conv to<br>Dropout(0.3) to IN to LeakyReLU to 3×3<br>conv, plus identityskip|Isensee et al. (2018); pre-<br>activation order from He et<br>al.(2016)|
|Downsampling|3×3 conv,stride 2(channels ×2)|Isensee et al.(2018)|
|UpsampleModule|nearest-neighbour ×2, then 3×3 conv<br>(channels ÷2)|Isensee et al. (2018)|
|LocalizationModule|3×3 conv,then 1×1 conv(channels ÷2)|Isensee et al.(2018)|
|Deep supervision|1×1 segmentation heads at 3 decoder levels,<br>upsampled(bi/trilinear)and summed|Isensee et al. (2018)|
|Normalisation /<br>activation|InstanceNorm (affine), LeakyReLU(0.01)|Ulyanov et al. (2016);<br>Isensee et al.(2018)|
|Parameters|2D: 3,187,330 · 3D: 9,463,714|printed bypython<br>modules.py|



### 4.2 Loss functions 



where p is the softmax output, g is the one-hot label and ϵ = 10<sup>−5</sup> . 

Soft Dice is indifferent to class proportions, so the 0.1% of prostate matters as much as the rest of the body (Milletari et al., 2016). Cross entropy gives smooth voxel-level gradients and makes the initial training stable. Difference from Isensee et al.: They used only multi-class Dice loss; I used cross entropy too, like nnU-Net does (Isensee et al., 2021). 

### 4.3 Training 

Both models use identical training settings except the input . 

|Setting|2D baseline|3D model|
|---|---|---|
|Input|128×128 axial slices|128×128×64 volumes|
|Batch size|16 slices|2 volumes|
|Optimiser|Adam,lr 5×10⁻⁴,weight decay10⁻⁵|same|
|LR schedule|×0.985per epoch(exponential)|same|



|Setting|2D baseline|3D model|
|---|---|---|
|Epochs|100|100|
|Checkpoint kept|best validationprostate Dice|same|



The optimiser and schedule follow Isensee et al. (2018). All runs used one NVIDIA A100-40GB on Rangpur 

### 4.4 Evaluation matrics 

Every metric is computed per test volume and reported as mean ± standard deviation over the 27 test volumes. If a class is absent from both the prediction and the ground truth of a volume, its score is undefined (NaN) and that volume is left out of that class’s mean. 

For each class, with P the predicted voxels and G the ground-truth voxels: 



Boundary metrics (the dilemma) 

- Base / apex Dice. Let Z be the axial slices that contain prostate in the ground truth, ordered from inferior to superior, and k = max(1,round(0.2|Z|)). The base is the top k slices (towards the bladder) and the apex is the bottom k (towards the pelvic floor). Prostate Dice is computed using only those slices. 

- Leakage into an organ at risk O (bladder or rectum): the fraction of the predicted prostate that is really that organ: 



Why these are needed on top of global Dice. Global Dice is dominated by the central slices, where the prostate is largest and easiest to segment. The base and apex hold only a small share of the prostate voxels, so even a total failure there barely moves the global score. For example, on T014_Week2 the 3D model’s global prostate Dice is 0.772, which looks acceptable, but its apex Dice is 0.000: the inferior end of the gland was missed entirely. Global Dice also treats every false-positive voxel the same, whether it lands in fat or in the rectal wall. Leakage measures where the errors go, and that decides the clinical harm. On J026_Week1, about 16% of the predicted prostate is actually rectum, so that contour would irradiate rectal wall. 

Limitation: base/apex Dice is computed only on slices where the ground truth contains prostate. A prediction that extends beyond the true apex or base is not penalised there; it only lowers global Dice. 

Resources and inference time 

Peak GPU memory (torch.cuda.max_memory_allocated), parameter count, and inference time per volume. Here, inference means turning one pre-processed 128×128×64 volume into a 3D label map: 

- 2D: the 64 axial slices are passed through the model in 2 batches of 32, and the per-slice predictions are stacked back into a volume. 

- 3D: the volume is zero-padded to a multiple of 16, passed through the model once, and the padding is cropped off. 

Timing starts once the pre-processed volume is in CPU memory (so the copy to the GPU is included) and stops when the label map is back on the CPU. One warm-up volume is run first, and torch.cuda.synchronize() is called before both timestamps, because GPU calls are asynchronous. File loading and pre-processing are excluded. Per volume is the fair unit: a planner contours whole scans, and timing 2D per slice would make it look about 64× faster without any real difference. 

#### Paired comparison 

Both models are tested on the same 27 volumes, so compare_2d_3d.py compares each volume with itself (3D better / same / worse). It then repeats this per patient, after averaging each patient’s volumes, because weekly scans of one patient are not independent. 

### 4.5 Design decision and why 

Same architecture for the baseline. In the baseline, I also used the same Improved U-Net in 2D, not just a plain 2D U-Net or keras_slices_data, which is already prepared for me. In other words, my research question concerns only the 3D context, so I need to keep all other factors, including the architecture, the same. With a plain U-Net, both the architecture and the dimensionality of the network would change, making it impossible to determine what makes them different from each other. Training of the 2D model on slices of the same volumes also ensures that both the 3D and 2D models will be evaluated using the same exact 27 volumes, which allows to compare them volume by volume. The price of that is the inability of this study to show the value of the Improved U-Net’s individual components (context modules and deep supervision).Dice + crossentropy loss. The prostate is only about 0.1% of the voxels. With cross-entropy alone, the model can get a very low loss by predicting background and body everywhere and almost ignoring the prostate. Soft Dice is computed per class and then averaged, so the prostate counts as much as the body, whatever its size. I kept cross-entropy as well, because it gives a smooth signal for every voxel and makes early training more stable. The original paper used Dice alone; nnU-Net uses the same combination I did. Both models reached a prostate Dice above 0.8 on the test set. 

Choosing the best epoch by validation prostate Dice, not validation loss. For the 2D model, the validation loss stayed at about 0.34 while the training loss fell to about 0.05. That looks like overfitting, but validation prostate Dice levelled off at about 0.80 instead of dropping. The high loss comes from computing soft Dice on batches of slices: many slices contain no prostate or rectum, and those give noisy, inflated loss values. Validation loss was therefore a misleading signal, so I kept the checkpoint with the best validation prostate Dice. That’s also the metric the project is about. 

One dims switch instead of separate 2D and 3D models. Every layer in modules.py is built by get_layers(dims), which returns either the 2D or 3D version of the convolution, normalisation and dropout. This guarantees the two models are identical apart from their dimensionality, which keeps the comparison fair. I also kept the 2D layer names unchanged during the refactor, so the 2D checkpoint I had already trained still loaded. I checked this by confirming the 2D parameter count stayed at exactly 3,187,330. 

Entire volume at half resolution. I used half-resolution training data (128×128×64) for my 3D model with a batch size of 2, without splitting into patches at full resolution. This ensured that my pipeline remained straightforward, easily fitting into memory (peak memory: 4.22 GB), while the model got the whole pelvis as input every time during forward pass. However, the tradeoff here was reduced resolution, where each voxel becomes double as large along every axis, thereby reducing the detail of thin boundaries such as the prostate-rectum boundary. 

### 5 Command to run on Rangpur 

Step 1: Install the dependencies 

pip install -r requirements.txt 

Versions used: Python 3.11.16, PyTorch 2.14.1 (CUDA 13.0), nibabel 5.4.2, NumPy 2.4.6, Matplotlib 3.11.2. 

Step 2: Link the data 

mkdir -p data/rangpur ln -s /home/groups/comp3710/HipMRI_Study_open/semantic_MRs data/rangpur/semantic_MRs_anon ln -s /home/groups/comp3710/HipMRI_Study_open/semantic_labels_only data/rangpur/semantic_labels_anon 

Step 3: Train both models 

sbatch train_2d.slurm      # 2D baseline, about 30 min sbatch train_3d.slurm      # 3D model, about 40 min 

This saves the best checkpoint for each model in outputs/best_model_2d.pt and outputs/best_model_3d.pt. 

Step 4: Evaluate on the test set 

sbatch predict_2d.slurm sbatch predict_3d.slurm 

This prints the per-class Dice and IoU, the base/apex Dice, leakage and inference time. It saves outputs/test_results_2d.json and outputs/test_results_3d.json, the failure-case figures and the training curves in images/. 

#### Step 5: Compare the two models 

python compare_2d_3d.py --dir outputs 

This prints the paired 2D vs 3D comparison used in Section 7. 

The seed is fixed (42), so the patient split is the same on every run. GPU training isn’t exactly deterministic, so a rerun may differ slightly (around ±0.01 Dice). 

### 6 Results 

#### 6.1 Training curves 



<!-- Start of picture text -->
2D loss 2D validation Dice<br>10 SS<br>— train<br>05 — val<br>os<br>3 606<br>Bos 5<br>+ ct<br>$ 2B 0.44 — body<br>So2 s — bone<br>— bladcer<br>0.2] — rectum<br>oa 00 —o> targetprostate 0.7<br>3 Zo ry 6 80 100 3 20 ry 6 80 100<br>Epoch Epoch<br><!-- End of picture text -->

#### 2D training curves 



<!-- Start of picture text -->
3D loss 3D validation Dice<br>— train 10 ———————=<br>on — vt Sone<br>os 2<br>Bos $ /<br>3 Bos<br>t F}<br>fsgo4 Boa3 —— bodybone<br>— bladder<br>oz oz — rectum<br>00 —o-= prostatetarget 07<br>3 Zo ry 6 80 100 ry 20 ry 6 80 100<br>Epoch Epoch<br><!-- End of picture text -->

#### 3D training curves 

Both models learn fast, then level off. In the 2D case, prostate Dice validation score reaches about 0.80 within the first 10 epochs and doesn’t change much after that. The 3D model starts with zero score (it predicts no prostate at all for epoch 1), passes 0.75 around epoch 6 and stabilizes around 0.85 at epoch 30. It takes longer to warm up, but ends up better. In both cases, the prostate and rectum are the most difficult classes to predict and body – the easiest. The 3D closes the gap to some extent, since its scores for prostate and rectum reach about 0.85–0.88 by the end, while the prostate score of 2D stays around 0.80. Neither of them overfits on Dice score, because the validation Dice score never falls during further training. The loss graphs are completely different, though.The 2D validation loss stays around 0.44 until about epoch 26, then 

drops to about 0.34 and stays there,while its training loss drops to around 0.05. This is largely an artifact: soft Dice is calculated in each batch of slices, and there are many slices that have no prostate or rectum in them, which leads to a high and noisy loss. It is not overfitting since the validation Dice for 2D network remains flat during the same epochs. The 3D network calculates the loss on entire volumes, and the validation loss follows the training loss very closely (0.13 vs 0.055). 

#### <u>6.2 Per-class test results (27 volumes, 6 unseen patients, mean ± std)</u> 

|Class|2D Dice|3D Dice|2D IoU|3D IoU|
|---|---|---|---|---|
|background|0.996 ± 0.002|0.998 ± 0.001|0.992 ± 0.004|0.996 ± 0.002|
|body|0.983 ± 0.004|0.987 ± 0.003|0.967 ± 0.007|0.974 ± 0.005|
|bone|0.920 ± 0.013|0.919 ± 0.011|0.853 ± 0.022|0.851 ± 0.019|
|bladder|0.935 ± 0.046|0.946 ± 0.029|0.881 ± 0.073|0.899 ± 0.049|
|rectum|0.848 ± 0.053|0.840 ± 0.075|0.739 ± 0.076|0.730 ± 0.102|
|prostate|0.801 ± 0.089|0.839 ± 0.052|0.676 ± 0.117|0.726 ± 0.077|



Both models clear the assignment target (prostate Dice ≥ 0.70) on the test set. 

#### <u>6.3 Boundary and leakage metrics</u> 

|Metric|2D|3D|Change|
|---|---|---|---|
|Prostate base Dice|0.748 ± 0.104|0.797 ± 0.093|+0.049|
|Prostate apex Dice|0.630 ± 0.244|0.698 ± 0.205|+0.068|
|Leakage into bladder|0.045 ± 0.063|0.013 ± 0.021|3.5× lower|
|Leakage into rectum|0.002 ± 0.006|0.011 ± 0.035|5× higher(still about 1%)|



#### <u>6.4 Resource profile</u> 

||2D|3D|3D / 2D|
|---|---|---|---|
|Parameters|3,187,330|9,463,714|3.0×|
|Peak trainingVRAM|0.72 GB|4.22 GB|5.9×|
|Timeper epoch|16.8 s|21.2 s|1.3×|
|Total trainingtime(100 epochs)|29 min|40 min|about 1.4×|
|Inference timeper volume|24 ms|27 ms|1.1×|
|Peak inference VRAM|0.37 GB|0.72 GB|1.9×|



All measured on one NVIDIA A100-40GB. Inference time is per volume for both models (2D: 64 slices in 2 batches of 32; 3D: one forward pass), excluding file loading and pre-processing; see Section 4.4 for exactly what is timed. Per slice, 2D takes about 0.4 ms (24 ms ÷ 64), but a planner contours whole volumes, so per-volume time is the fair comparison. Both are far below the 10 s acceptance limit. 

The 3D model is more accurate where it matters (Sections 6.2–6.3), so the question is whether its extra cost would stop the team from using it. It would not. The largest increase is training memory, about 6× (0.72 to 4.22 GB). In absolute terms, though, 4.22 GB fits on almost any modern GPU, and training is a one-off cost: about 40 minutes instead of 29. At deployment, where the cost repeats for every patient, the difference is very small. Inference takes 27 ms instead of 24 ms per volume and needs 0.72 GB instead of 0.37 GB, both far below the 10 s acceptance limit. A planner would not notice the difference. The 3× parameter count only makes the saved model larger (about 38 MB vs 13 MB), which also doesn’t matter in practice. The one place cost would become a real constraint is scaling up: memory grows with the volume size, so training at full resolution (8× more voxels) would need patch-based training. At the resolution used here, the extra cost of 3D is small compared with its accuracy gain. 

### 7 Investigating the dilemma : Does 3D context fix boundary error? 

#### 7.1 Comparison 

Both models were tested on the same 27 volumes from 6 patients, so each volume is compared with itself. The patient column averages each patient’s volumes first, because weekly scans of one patient are not independent. “Same” means the difference is below 0.001. 

|Metric|3D better / same / worse (27<br>volumes)|3D better / same / worse (6<br>patients)|
|---|---|---|
|Prostate Dice|20 / 0 / 7|3 / 1 / 2|
|Base Dice|20 / 0 / 7|4 / 0 / 2|
|Apex Dice|14 / 1 / 12|4 / 0 / 2|
|Bladder Dice|18 / 1 / 8|4 / 0 / 2|
|Rectum Dice|14 / 1 / 12|1 / 0 / 5|
|Leakage into<br>bladder|18 / 5 / 4|6 / 0 / 0|
|Leakage into rectum|0 / 21 / 6|0 / 3 / 3|



Per patient (mean over that patient’s test volumes): 

|Patient|Test<br>scans|Prostate Dice 2D<br>to 3D|Base 2D to<br>3D|Apex 2D to<br>3D|Verdict|
|---|---|---|---|---|---|
|T014|8|0.691 to 0.803|0.697 to<br>0.805|0.471 to<br>0.632|3D much better|
|T009|1|0.871 to 0.912|0.784 to<br>0.848|0.574 to<br>0.751|3D better|
|J026|8|0.819 to 0.851|0.735 to<br>0.784|0.739 to<br>0.810|3D better|
|V027|8|0.872 to 0.872|0.791 to<br>0.815|0.701 to<br>0.671|same overall; apex<br>slightly worse|
|K008|1|0.826 to 0.768|0.833 to<br>0.808|0.402 to<br>0.378|3D worse|
|S028|1|0.866 to 0.768|0.790 to<br>0.626|0.759 to<br>0.803|3D worse (at the base)|



3D is better on 20 of 27 volumes” looks compelling; however, 27 volumes are not 27 separate tests. Scans of the same patient every week show almost identical anatomy, so they are rather repeated measures of the same case. 24 out of 27 volumes were obtained by just 3 patients (8 scans each), and 15 out of 20 wins were made by just 2 patients – T014 (8/8) and J026 (7/8). The 2 patients in which 3D was worse, K008 and S028, have only 1 scan each and thus contribute to volume count insignificantly. The real measure is the number of cases in which 3D was better – it is 3, the same 1 and worse 2 out of 6. This is the question that needs to be answered by the manager: Will 3D be helpful to the next patient? With only 6 patients, the sample is not large enough, and I report both measures, considering the second as the principal one. 

#### 7.2 Where 3D context helps 

Finding 1: 3D rescues the patient that 2D fails on. 

- Evidence: all four of 2D’s worst volumes are from patient T014 (Weeks 2, 5, 6 and 7; 2D Dice 0.61–0.65). With 3D they rise to 0.77–0.82. T014 accounts for most of the overall gain (0.801 to 0.839). 

- Two-dimensional model decides on each slice separately. For instance, when a certain slice is ambiguous (low contrast, peculiar gland shape), the model can fail to locate the prostate on this slice, although the previous and subsequent slices have high contrast. All four of 2D’s poorest performing volumes belong to T014, for different weeks, which may imply that there is something particular to the anatomy of this patient that makes each separate slice hard to interpret. Three-dimensional model considers neighboring slices 

altogether, thus carrying over the shape of the gland through the ambiguous slices. This explains the improvements at both ends of T014’s gland: base 0.697 to 0.805, apex 0.471 to  0.632 

- Limit: for patient V027, who was already segmented well, 3D gave no overall gain. 

Finding 2: bladder leakage falls for every patient. 

- Evidence: 0.045 to 0.013 overall, and lower for 6 of 6 patients. Per volume it is lower in 18, the same in 5 and slightly higher in 4. This is the most consistent result in the study. 

- On the single axial slice around the base, there is a small interface between the prostate and the bladder neck; they may look alike, so that the 2D approach has to estimate the place of the boundary between these structures. On the other hand, the situation becomes clear on slices - the bladder is a huge, uninterrupted structure located above the prostate, and the 3D model is able to distinguish voxels belonging to the bladder and define the boundary of the prostate. That is the reason of bladder leakage reduction for all 6 patients 

- Limit: T014 still has the highest bladder leakage in 3D (up to 0.094 for Week 2). The base is improved, not solved. 

Finding 3: the apex improves on average, but not consistently. 

- Evidence: mean apex Dice 0.630 to 0.698, with large gains for T014 (0.471 to  0.632), J026 and T009. But per volume it is better in 14, the same in 1 and worse in 12, almost a coin flip. V027’s apex got slightly worse (0.701 to 0.671). 

- The neighbouring slices would be helpful for the apex since the gland tapers off gradually and the 3D model can follow the tapering off in slices where it is still obvious. However, after 0.5x downsampling, every slice becomes twice thicker, hence only a few slices represent the apex. Dice score for the apex is calculated from the bottom 20% slices of the prostate gland. Thus, for a gland which occupies roughly 10 slices, the number of slices in the apex region become 2 slices with very little number of voxels. With such few voxels, a small mistake changes the Dice value greatly. This is the reason for the high variability of the apex results, almost random (14 better / 1 same / 12 worse), with high standard deviation (0.205). 

Finding 4: fewer bad drafts. 

   - Evidence: the standard deviation of prostate Dice fell from 0.089 to 0.052. The worst 3D volume (0.734) is better than 2D’s worst four (0.61–0.65). 

   - The planner goes through each draft. A "pretty good" tool takes a reliable few minutes for editing per patient, and a tool that is generally "excellent" yet sometimes "very wrong" is less useful. Sometimes, a poor draft must be completely redrafted; a wrongly drawn yet convincing-looking contour might go unnoticed in the process of reviewing. With radiotherapy, the worst-case scenario determines the acceptability of the tool rather than the average one. It is important that the 3D model has a lower standard deviation (0.089 to 0.052) and a better worst-case scenario (0.734) than the 2D’s worst four. 

- 7.3 Where 3D context does not help 

Finding 5: 3D is slightly but consistently worse on the rectum, and it amplifies an existing leak. 

- Evidence: rectum Dice 0.848 to 0.840, and worse for 5 of 6 patients. Prostate leakage into the rectum is never lower with 3D (0 better / 21 same / 6 worse). Most of it comes from two scans of patient J026, which already leak in 2D, but much more in 3D: 

|Scan<br>2D leak to rectum|3D leak to rectum|
|---|---|
|J026_Week1 0.025|0.159 (about 6×)|
|J026_Week0 0.021|0.097 (about 5×)|



Every other volume is 0.000 in 2D, and these two scans account for about 85% of all 3D rectum leakage. 

- j026 In weeks 0 and 1, the prostate is already leaking into the rectum in 2D (0.025 and 0.021), so the trigger must be some feature about these two images that cannot possibly be introduced by 3D. The most plausible reason for this problem is the lack of clarity of the interface between the prostate and the rectum in these two images. This interface is very thin, consisting of 1-2 voxels at 0.5x resolution, while the shape of the rectum and its filling varies from week to week. In 2D images, the error on the boundary persists only within the corresponding slices. The purpose of the 3D model is to create contours that are continuous from one slice to another, so once it makes the decision to place some part of the prostate in the rectal wall, it extends the error to neighbouring slices 5-6 times. 

Finding 6: 3D is worse on two single-scan patients. 

   - Evidence: S028_Week0 0.866 to 0.768 (base 0.790 to 0.626) and K008_Week0 0.826 to 0.768. These are now 3D’s 2nd- and 3rd-worst volumes. 

   - Another potential explanation could be the quantity of the training data. While both models have access to the same number of training volumes – 136, the 2D model processes it as 8,700 slices per epoch, whereas the 3D model receives it as 136 scans with 3× as many parameters. With such a small number of 26 training patients, the 3D model had less exposure to rare anatomy cases and hence would generalise poorly to a patient that looks different from those in the training set. 

- 7.4 Is it worth the cost? 

The assignment brief warns that 3D might “demand 5–10× the GPU memory and training time”. Measured: training VRAM 5.9×, but training time only about 1.2–1.3× and inference time 1.1× (Table 6.4). 

Both models process exactly the same amount of voxels in each epoch – all 136 training images. The only distinction lies in their batch organization. 2D model distributes the voxels into roughly 8,700 slices divided into 545 batches of 16, whereas 3D model uses 68 batches of 2 images. Each 3D convolution operation requires 3 times more computational power per voxel (27 weight coefficients instead of 9). However, GPU works best with big tensors and there is always a 

certain overhead per batch (data loading, operation launch, optimizer operation). Fewer but bigger batches compensate for the additional calculations, and therefore a 3D epoch costs 21.2 seconds against 16.8 seconds for 2D one. Additional memory consumption is real (4.22 GB against 0.72 GB), however, 4.2 GB fit any modern GPU and thus are not an issue for this team. 

Statistical caveat. There are only 6 test patients and one training seed per model. The patientlevel counts in 7.1 are the honest measure of consistency, and a difference for 1–2 patients could be noise. 

7.5 Acceptance-criteria outcome 

|Criterion (Section 2.1)|Result|Met?|
|---|---|---|
|1. Prostate Dice ≥ 0.70, all<br>classes reported|3D 0.839 (2D 0.801); Table 6.2|Yes|
|2. 3D beats 2D at base and<br>apex|Base +0.049 (4/6 patients); apex +0.068 on<br>average but only 14/27 volumes|Yes on<br>average|
|3. Leakage not worse than 2D|Bladder: better for 6/6 patients. Rectum: worse<br>(0.002 to 0.011, mostly J026)|Partly|
|4. ≤ 6 h training on one GPU;<br>< 10 s inference|About 40min; 27 ms per volume|Yes|



## 8. Failure-case autopsies 

How to read the figures. predict.py picks the 4 test volumes with the lowest prostate Dice. For each one it shows the worst base slice (top row) and the worst apex slice (bottom row): the MR image, the ground truth, the prediction, and a prostate difference map (green = correct, blue = missed prostate, red = false-positive prostate). In these figures the patient’s back is at the top: the rectum (orange) sits above the prostate (red), and the bladder (light blue) below it. 

Each case follows the same steps: observation (what the figure shows), hypothesis (why it happened), evidence (numbers, 2D vs 3D, other weeks) and conclusion (is this case safe to automate?). 

Case 1: T014_Week2, the apex is missed completely by both models 

2D: 



<!-- Start of picture text -->
— —_ a —_<br>ti (sf ( e<br>— —_— = —<br>~— — ld —<br><!-- End of picture text -->

#### 3D: 



<!-- Start of picture text -->
— —_ — —_<br>EE EEE — hh <=<br>i }<br>Stig) [ont iz) [iw<br>~—— ld —— —<<br><!-- End of picture text -->

||Prostate Dice|Base|Apex|Leak to bladder|
|---|---|---|---|---|
|2D|0.626|0.763|0.000|—|
|3D|0.772|0.885|0.000|0.094 (highest of all 27)|



- Observation. Apex (z = 35): the true prostate is a tiny spot just below the rectum. Both models predict rectum there but no prostate at all, so the whole gland on this slice is blue (missed). The 2D and 3D predictions are almost identical. Base (z = 45): 2D draws the gland too large on every side (a red ring around a green core, slice Dice 0.72); 3D’s outline is much tighter (slice Dice 0.85). 

- Hypothesis. At the apex the gland is only a few voxels wide at 0.5× resolution, and it is surrounded by pelvic-floor muscle of similar darkness. There is too little signal for either model to detect it, so 3D context cannot help. A structure that is barely visible on a slice is barely visible on its neighbours too. 

- Evidence. The prostate spans about 11 slices here (z = 35–45), so the apex region is only 2 slices. 3D fixed T014’s base (0.763 to 0.885) and its other weeks (for example Week 6 apex 0.286 to 0.697), but not this apex. The 3D bladder leakage of 0.094 does not appear on the base slice shown, which has no bladder in it. So it most likely comes from the prediction extending above the true base into the bladder, which base Dice cannot see (Section 4.4, limitation). 

Conclusion. A global Dice of 0.772 looks acceptable, yet the apex is entirely missed. This is the dilemma in a single case. Apex slices are not safe to automate 

Case 2: J026_Week1, prostate leaks into the rectum 



<!-- Start of picture text -->
( oa ) (aio seisNA : i ) -Ss)sis é ) ae )<br>« ‘ e ~ e “ « ‘<br><!-- End of picture text -->

||Prostate Dice|Base|Apex|Leak to rectum|
|---|---|---|---|---|
|2D|0.800|0.580|0.841|0.025|
|3D|0.734|0.547|0.816|0.159 (highest of all 27)|



   - Observation. Base (z = 37): the rectum is very large and round on this scan, much bigger than in the other cases, and the true prostate is a small region just below it. The prediction makes the prostate about twice as big, growing upward towards the rectum: the red falsepositive area sits between the green core and the rectum. The model also predicts a small patch of bladder below the prostate where the ground truth has none. Apex (z = 28): mostly correct (slice Dice 0.76). 

   - Hypothesis. The enlarged rectum presses against the back of the prostate and changes the usual shape of the interface. The model’s learned expectation of “a prostate of normal size and shape” then draws the gland into the rectal wall. A 3D model enforces continuity across slices, so once it makes this mistake, it tends to repeat it on the neighbouring slices too. That is why 3D leaks about 6× more than 2D here. 

   - Evidence. The same weakness exists in 2D (0.025), so the trigger is in this scan, not in the 3D model alone. J026_Week0 also leaks (2D 0.021 to 3D 0.097), while J026 Weeks 2–7 are about 0. Two scans account for about 85% of all 3D rectum leakage. 

   - Conclusion. About 16% of the predicted prostate is rectal wall, which would be irradiated (a risk of proctitis). Scans with an unusually distended rectum need full manual review of the prostate–rectum boundary. 

- Case 3: S028_Week0, the 3D model draws the gland past its true base 



<!-- Start of picture text -->
Overlays: ‘5028 _WeekO: prostate Dice 0.768 (base 0.626, apex 0.803)<br>Base slice z=41:; imagebone yellow, bladder blue,Groundrectumtruth orange, prostate red | Difference: Predictiongreen correct, blue missed, redProst te f a lse positivedifference (slice Dice 0.28)<br>Apex sice s=e8 mage round te Prediction Prostate dtference tarce Dice 0.70<br><!-- End of picture text -->

||Prostate Dice|Base|Apex|
|---|---|---|---|
|2D|0.866|0.790|0.759|
|3D|0.768|0.626|0.803|



- Observation. Base (z = 41): the bladder (light blue) is large and full. The ground truth shows only a thin sliver of prostate on its upper edge. The 3D model predicts a much larger block of prostate in the tissue between the bladder and the rectum, so most of it is red (false positive), with only a little green and blue (slice Dice 0.28). Apex: fine (0.70). 

- Hypothesis. This slice is right at the top of the gland, where the prostate is ending. The tissue just above the base (bladder neck and seminal vesicles, which are labelled as body here) looks similar to prostate. The 3D model continues the gland’s shape upward from the slices below, instead of stopping where it really ends: the same continuity bias as in Case 2, this time at the base. 2D judges each slice separately and was less affected (base 0.790 vs 0.626). 

- Evidence. Leakage into the bladder is low (0.005), so the false-positive tissue is mostly unlabelled tissue, not bladder or rectum. The leakage metric doesn’t catch this error, but base Dice does. S028 is a single-scan test patient, so there are no other weeks to compare. 

- Conclusion. Over-extension at the base is a 3D-specific failure. The top slices of the gland need review, especially when the bladder is full. 

Case 4: K008_Week0, the apex is missed again 



<!-- Start of picture text -->
santasSenerwage heme leider Ses008GroundWeek0:ecto uthsineeprostatepretend Dice 0.768 (haze| Moment 0.808,Predictionpearapex coc 0376) Uhre ellieprostate palitaifference (slice Dice 0.77)<br>===<br>ZG 1< Ke<br>- =e L Sage Le -<br><!-- End of picture text -->

|Prostate Dice|Base<br>Apex|
|---|---|
|2D 0.826|0.833 0.402|
|3D 0.768|0.808 0.378|



- Observation. Base (z = 41): good (slice Dice 0.77). The outline is close, with a thin red edge towards the rectum and a thin blue edge towards the bladder. Apex (z = 28): the 

same pattern as Case 1. The true prostate is a small spot below the rectum, and the prediction contains rectum but no prostate, so the whole spot is blue. 

- Hypothesis. The same as Case 1: at the apex the gland is too small and too similar to the surrounding muscle at 0.5× resolution. 

- Evidence. The apex is poor in both models (0.402 and 0.378), so this is not a 3D problem. It is a resolution and contrast problem that neither model solves. 

- Conclusion. Missing the apex is the most common failure across the cases. It leads to under-treatment of the apex, which risks recurrence. Apex slices always need human review. 

Summary of failure modes 

|Failure mode|Cases|Fixed by 3D?|Safe to automate?|
|---|---|---|---|
|Apex missed<br>completely (too small,<br>low contrast)|T014_Week2,<br>K008_Week0|No (2D fails the<br>same way)|No: always review the apex|
|Prostate leaking into a<br>distended rectum|J026_Week1 (and<br>Week0)|No: 3D makes it<br>about 6× worse|No: review the rectum<br>boundary, especially on<br>distended rectums|
|Gland drawn past the<br>true base|S028_Week0|No: 3D-specific|No: review the top slices|
|Gland drawn too large<br>at the base|T014 (2D)|Yes (slice Dice<br>0.72 to 0.85)|Mostly, with a check|
|Leakage into the<br>bladder|all patients|Yes (6/6 patients<br>improved)|Mostly, with a check|



Overall pattern. 3D context helps when the neighbouring slices are informative: it tightens the base and keeps the prostate out of the bladder. It does not help when the structure is too small to see (the apex). And its built-in continuity can spread an error across slices (the rectum in J026) or carry the gland past where it really ends (the base in S028). 

## 9. Recommendation to the project manager 

Recommendation: adopt the 3D Improved U-Net instead of the 2D baseline as a contour-drafting assistant, with mandatory human review of three specific regions. Do not use it as an autonomous contouring tool. 

9.1 Why 3D instead of 2D 

   - More accurate where it matters Prostate Dice improves from 0.801 to 0.839, base Dice from 0.748 to 0.797 and apex Dice from 0.630 to 0.698 (Section 6). 

   - Protect  the bladder. Leakage of the prostate into the bladder decreases from 4.5% to 1.3% of the contour area, and it applies to all 6 test patients (Section 7.1). This is the most consistent result in the study. 

   - Fewer bad drafts. The spread of prostate Dice almost halves (std 0.089 to 0.052), and it rescues the patient the 2D model failed on (T014: 0.691 to 0.803). For a planner, fewer bad drafts means less time redrawing contours from scratch. 

   - Costs almost nothing more to compute. Inference time increases from 24 ms to 27 ms per volume and memory consumption – from 0.37 GB to 0.72 GB of GPU RAM. The increased cost is mostly 6× more memory during training (4.22 GB), which is a one-off and fits on any modern GPU (Section 7.4 

- 9.2 What the planner must always review 

However, the gains are uneven. The following three regions should be checked on every case, because the autopsies in Section 8 show the model can fail there without warning: 

   - The apex (the bottom sections of the gland). Apex Dice is at 0.70 ± 0.21 at present, while apex was missed completely in T014_Week2 and K008_Week0 by both models (Cases 1 and 4). At this resolution, apex is made up of 2-3 slices of a few voxels, hence 3D context is unable to find it. Missing of the apex implies under-dose and a likelihood of recurrence. 

   - The prostate-rectum border. 3D did not contribute to the correct segmentation of rectum: rectum Dice score was worse for 5 out of 6 patients. In J026_Week1, 16% of the predicted prostate belonged to the rectal wall (Case 2). Over contouring in the region leads to proctitis during radiation. 

   - The top sections of the gland (the base). On S028_Week0, 3D model has extended the gland beyond its true end (Case 3 It doesn’t show up in the bladder or rectum leakage numbers, so it could easily be missed in a quick review 

- 9.3 Cases that need extra caution 

From the failure autopsy findings, the following scenarios are recommended for comprehensive manual assessment instead of fast examination: 

- Highly distended rectum (as seen in J026 Weeks 0-1). 3D magnified the 2D mistake by almost 6 times in these images. 

- Patient with fully distended bladder causing pressure on the base (as seen in S028). The model extended the gland too far at the base 

- New patients whose anatomy is different from the training data set. In single-scan test patients (S028 and K008), 3D was inferior to 2D. The model was trained on just 26 patients 

#### 9.4 Before clinical use 

These results come from 6 test patients and one training run per model. Before this tool is used on real patients, I recommend: 

- Validating on more patients, ideally with patient-level cross-validation, so the per-patient conclusions are not based on 6 people. 

- Adding a boundary-distance metric (for example HD95), so millimetre-level boundary errors are measured directly, not only through overlap. 

- Testing at full resolution (for example with patch-based 3D training). The 0.5× downsampling limits how thin a boundary can be drawn, and this is the most likely cause of the missed apices and the rectum leakage. 

In one sentence: the 3D model is a better drafting assistant than 2D for almost no extra running cost, but the apex, the prostate–rectum boundary and the top of the gland must always be checked by a human. 

## 4. Reference 

- Dowling, J. & Greer, P. (2021). Labelled weekly MR images of the male pelvis (v2) [Data set]. CSIRO. https://doi.org/10.25919/45t8-p065 

- He, K., Zhang, X., Ren, S. & Sun, J. (2016). Identity mappings in deep residual networks. In Proc. ECCV, pp. 630–645. 

- Isensee, F., Kickingereder, P., Wick, W., Bendszus, M. & Maier-Hein, K. H. (2018). Brain tumor segmentation and radiomics survival prediction: Contribution to the BRATS 2017 challenge. In Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries (BrainLes 2017), LNCS 10670, pp. 287–297. Springer. 

- Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J. & Maier-Hein, K. H. (2021). nnUNet: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 18(2), 203–211. 

- Kingma, D. P. & Ba, J. (2015). Adam: A method for stochastic optimization. In Proc. ICLR. 

- Litjens, G., Toth, R., van de Ven, W., et al. (2014). Evaluation of prostate segmentation algorithms for MRI: The PROMISE12 challenge. Medical Image Analysis, 18(2), 359– 373. 

- Milletari, F., Navab, N. & Ahmadi, S.-A. (2016). V-Net: Fully convolutional neural networks for volumetric medical image segmentation. In Proc. 3DV, pp. 565–571. 

- Ronneberger, O., Fischer, P. & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. In Proc. MICCAI, pp. 234–241. 

- Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T. & Ronneberger, O. (2016). 3D U- Net: Learning dense volumetric segmentation from sparse annotation. In Proc. MICCAI, pp. 424–432. 

- Ulyanov, D., Vedaldi, A. & Lempitsky, V. (2016). Instance normalization: The missing ingredient for fast stylization. arXiv preprint arXiv:1607.08022. 

## 5. Ai disclosure 

I used Claude throughout this project as a learning and support tool. I mainly used it to learn and understand the U-Net and Improved U-Net architecture before implementing each part myself, and to help debug my code. For example, it pointed out that my sub-modules were not receiving the dims argument.I also used it to navigate the Rangpur cluster (writing Slurm scripts, fixing job errors and monitoring jobs), to explain evaluation metrics such as Dice, IoU and the base/apex and leakage measures, and to suggest improvements and relevant references. Claud also help with the structure of the report and editing my words. The idea and knowledge is from me while claude gives me suggestion. 

