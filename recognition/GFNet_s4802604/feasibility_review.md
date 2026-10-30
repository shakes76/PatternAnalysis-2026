# Feasibility Review: Alzheimer's MRI Classification using GFNet

Sasha Borislavov, 48026048

## 1. User Need, Scope, and Acceptance Criteria

This project will build a research prototype (NOT A DIAGNOSTIC TOOL!) that classifies brain MRI scans as Alzheimer's disease (AD) or cognitively normal (CN), for researchers exploring AI-assisted scan analysis. The key engineering question is whether the model's confidence reflects its actual reliability, so that uncertain predictions can be flagged for human review. The project succeeds if:

1. It reaches at least 80% scan-level test accuracy on a subject-independent test set, with slice-level accuracy also reported.
2. No patient leakage: scans from the same patient never appear across training, validation and test sets, enforced by an assertion in the data-loading code.
3. Confidence calibration is evaluated with a reliability diagram and expected calibration error (ECE), comparing GFNet against a simple CNN baseline.
4. Confidence-based referral is evaluated by reporting accuracy on accepted scans and the percentage referred for human review, using a threshold selected on the validation set.
5. Training completes within an estimated 10 A100 GPU-hours, reporting peak GPU memory, parameter count and inference time.

## 2. Model Choice and Course Concepts

I will use GFNet (Hard tier) on the ADNI brain MRI dataset to classify scans as AD or CN. GFNet applies the Fast Fourier Transform (FFT) to move image features into the frequency domain, where a learned filter mixes information across the whole image without the costly self-attention mechanism. By the convolution theorem, this filtering is equivalent to a global circular convolution, so each block captures global patterns at FFT cost rather than quadratic attention cost. This suits MRI because Alzheimer's-related changes, such as hippocampal shrinkage and enlarged ventricles, may involve spatially distributed patterns. Following Rao et al. (2023), I will compare GFNet against a small CNN baseline on identical subject-level splits.

## 3. Preliminary Feasibility Evidence

A data audit confirmed a leakage issue: 216 of the 331 subjects in the provided test set also appear in the provided training set. I built a clean subject-level split (475 training, 102 validation and 103 test subjects) from the 680 subjects, 1526 scans and 20 slices per scan. PyTorch runs on an A100 GPU (40 GB). The short-test partition starts jobs within seconds but limits runs to 10 minutes, while the main A100 partition has queue times of many hours. A 12-epoch baseline CNN (97,890 parameters, about 27 seconds per epoch) reached 0.82 scan-level and 0.81 slice-level accuracy on the clean test set (220 scans), so the 80% target looks realistic. A single run on a small test set is noisy, and calibration and GFNet performance are still to be evaluated.

## 4. Risks, Budget, and Fallback

- **Poor generalisation or overfitting:** use the clean split, select the best checkpoint by validation loss, and compare against the CNN baseline.
- **Poor calibration:** evaluate ECE and reliability diagrams, then assess confidence-based referral.
- **Compute access:** main A100 queues may delay long GFNet runs, so I will start early and use checkpointing.
- **Limited time or compute:** prioritise GFNet training and the essential evaluation within budget.

**Budget:** approximately 10 A100 GPU-hours, revised after initial GFNet profiling. Peak VRAM, parameter count, inference time and total GPU-hours will be recorded.

**Next experiment:** train GFNet on the clean split and compare its scan-level accuracy, calibration and resource use with the CNN baseline.

**Fallback:** if GFNet exceeds the budget or cannot be trained reliably, I will use the CNN baseline to complete the accuracy, calibration and referral evaluation, and report any unmet acceptance criteria transparently.

## 5. Reference List

Y. Rao, W. Zhao, Z. Zhu, J. Lu, and J. Zhou, "Global filter networks for image classification," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 45, no. 6, pp. 7837-7850, 2023.
