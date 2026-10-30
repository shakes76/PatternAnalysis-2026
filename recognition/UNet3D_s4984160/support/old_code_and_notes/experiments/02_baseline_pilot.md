# Phase 2: Baseline 3D U-Net Pilot Training

## Experiment Setup
- Date: 2026-10-29
- Slurm Job ID: 712336
- GPU: NVIDIA A100 40GB
- Model: Standard 3D U-Net
- Input shape: [1, 1, 64, 128, 128]
- Data split: 145 train / 39 validation / 27 test
- Optimizer: Adam, learning rate 0.001
- Loss: Cross-Entropy
- Training: 1 epoch, 2 training batches, 2 validation batches

## Results
- Job status: COMPLETED (ExitCode 0:0)
- Training loss: 2.0204
- Validation loss: 1.8632
- Validation Dice: [0, 0, 0, 0, 0, 0.00106]
- Validation IoU: [0, 0, 0, 0, 0.00053]
- Peak allocated GPU memory: 0.537 GB
- Slurm elapsed time: 47 seconds
- Error log: Empty

## Conclusion
The baseline training pipeline successfully completed training,
validation, metric calculation, and checkpoint saving.

The pilot is too small to evaluate segmentation performance.

## Next Step
Run one complete epoch using all 145 training scans
and 39 validation scans.
