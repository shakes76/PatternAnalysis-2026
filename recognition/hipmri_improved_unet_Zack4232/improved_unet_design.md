# Improved U-Net 2D adaptation

Reference: Isensee et al. (2018), *Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge*, Section 2.1. https://arxiv.org/abs/1802.10508

## Implemented design

The implementation adapts residual context blocks, instance normalization, LeakyReLU (0.01), dropout (0.3), strided downsampling, nearest-neighbor upsampling, and summed segmentation outputs from the reference to 2D.

Project-specific choices are explicit:

- One input channel and six output classes.
- Five encoder levels with 32, 64, 128, 256, and 512 channels.
- Decoder widths of 256, 128, 64, and 32. Each localization module uses a 3x3 convolution at twice its output width, then a 1x1 reduction.
- Pre-activation normalization and activation for feature convolutions, except the input stem. Segmentation heads are plain 1x1 convolutions.
- Affine instance normalization with no running statistics; element-wise dropout.
- Heads at full, half, and quarter resolution. Logits are summed through successive nearest-neighbor upsampling. One CE+Dice objective is applied to the combined logits; there are no separate auxiliary losses.
- Input dimensions must be divisible by 16, with more than one bottleneck pixel.

This is a 2D adaptation, not an exact reproduction of the 3D system. It does not add inter-slice context. The baseline classes remain unchanged.

## Experiment plan

Compare against the standard U-Net with CE+Dice using the same data normalization, splits, batch size 8, Adam learning rate 0.001, seed 42, and 20 epochs. Use no augmentation for this comparison. Select checkpoints by validation prostate Dice. These settings differ from the paper's training procedure.

This comparison measures the full architecture change, not the effect of each component alone. Parameter counts are not matched and must be reported with runtime and memory.

## Verification status

Local syntax and baseline class AST comparison passed. GPU execution is pending. The provided smoke test reads two training slices at widths 128 and 144, checks gradients and all segmentation heads, verifies an optimizer update, checks evaluation repeatability, reloads the state dictionary, and loads the previous baseline checkpoint strictly. It does not measure trained-model quality.
