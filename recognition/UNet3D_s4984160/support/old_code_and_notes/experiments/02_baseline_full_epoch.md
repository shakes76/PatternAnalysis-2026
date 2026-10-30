# Phase 2: Standard 3D U-Net — Full Epoch Feasibility Test

## 1. Objective
Verify the complete training and validation pipeline on all
HipMRI training and validation samples. Identify potential
limitations before full baseline training.

## 2. Experimental Configuration
- Date: 2026-10-29
- Slurm Job ID: 712581
- GPU: NVIDIA A100 40GB
- Model: Standard 3D U-Net
- Input shape: [1, 1, 64, 128, 128]
- Number of classes: 6
- Split: 145 train / 39 validation / 27 test
- Batch size: 1
- Epochs: 1
- Optimizer: Adam
- Learning rate: 0.001
- Loss: Cross-Entropy

## 3. Results
- Slurm status: COMPLETED
- Exit code: 0:0
- Elapsed time: 12 minutes 58 seconds
- Training batches: 145
- Validation batches: 39
- Training loss: 1.3084
- Validation loss: 0.8643
- Peak allocated GPU memory: 0.537 GB
- Error log: Empty

### Validation Metrics

| Label | Anatomy | Dice | IoU |
|---|---|---:|---:|
| 0 | Background | 0.9726 | 0.9482 |
| 1 | Body | 0.9199 | 0.8565 |
| 2 | Bones | 0.5251 | 0.3623 |
| 3 | Bladder | 0.0000 | 0.0000 |
| 4 | Rectum | 0.0000 | 0.0000 |
| 5 | Prostate | 0.000022 | 0.000011 |

Label definitions were verified using the official
HipMRI BriefDataDescription.txt.

These are validation metrics after one training epoch,
not held-out test-set performance.

## 4. Findings
The complete training and validation pipeline ran successfully.

Background and Body achieved high validation Dice, while
Bladder, Rectum and Prostate were poorly segmented.

The strong class imbalance observed during dataset auditing
is a primary concern for the current Cross-Entropy training
configuration.

## 5. Next Experiments
1. Implement combined Dice and Cross-Entropy loss.
2. Improve per-case Dice/IoU reporting and predicted label counts.
3. Train the baseline for multiple epochs and save the best checkpoint.
4. Compare against 3D Improved U-Net on identical patient splits.

## 6. Reproducibility
- Source: train.py
- Slurm script: train_full_epoch.sbatch
- Training history: results/baseline_full_epoch/history.csv
- Checkpoint: results/baseline_full_epoch/best_model.pt

Checkpoints and raw logs are retained on Rangpur and are not
committed to GitHub.
