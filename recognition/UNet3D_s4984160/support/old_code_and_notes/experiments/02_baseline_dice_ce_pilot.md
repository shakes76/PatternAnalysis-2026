# Baseline Dice + CE Pilot

## Configuration
- Date: 2026-10-29
- Job ID: 712801
- Model: Standard 3D U-Net
- Loss: 0.5 Cross-Entropy + 0.5 foreground Soft Dice
- Optimizer: Adam, learning rate 0.001
- Input: [1, 1, 64, 128, 128]
- Training: 2 batches
- Validation: 2 batches
- Split: 145 train / 39 validation / 27 test

## Results
- Status: COMPLETED (0:0)
- Elapsed: 36 seconds
- Train loss: 1.4816
- Validation loss: 1.3982
- Prostate validation Dice: 0.00106
- Prostate validation IoU: 0.00053
- Peak allocated GPU memory: 0.585 GB

## Metric Verification
- Six-class Dice/IoU calculation: PASS
- Per-case CSV generation: PASS
- GT/predicted voxel counts: PASS
- Checkpoint saving: PASS
- Error log: Empty

## Finding
The model predicted all 2,097,152 voxels across two
validation cases as Prostate (class 5).

This pilot validates the implementation, not segmentation
quality. It is not comparable to the full-epoch CE result.

## Next Step
Run multi-epoch baseline training on the complete training
and validation sets using the combined loss.
