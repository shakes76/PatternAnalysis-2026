### Per-class test results (mean over test volumes)

| Class | Dice Improved 3D U-Net | Dice 3D U-Net (baseline) | IoU Improved 3D U-Net | IoU 3D U-Net (baseline) | HD95 Improved 3D U-Net | HD95 3D U-Net (baseline) | ASD Improved 3D U-Net | ASD 3D U-Net (baseline) |
|---|---|---|---|---|---|---|---|---|
| body | 0.988 | 0.981 | 0.976 | 0.962 | 1.00 | 1.81 | 0.21 | 0.42 |
| bone | 0.923 | 0.910 | 0.857 | 0.836 | 1.14 | 1.24 | 0.44 | 0.52 |
| bladder | 0.949 | 0.934 | 0.905 | 0.880 | 1.19 | 3.06 | 0.36 | 0.55 |
| rectum | 0.857 | 0.811 | 0.754 | 0.687 | 2.22 | 2.71 | 0.58 | 0.83 |
| prostate | 0.853 | 0.846 | 0.746 | 0.735 | 1.47 | 1.42 | 0.56 | 0.59 |

HD95 and ASD are in voxels of the resized volume (lower is better). They are averaged only over volumes where both a ground-truth and a predicted structure exist.

### Structures missed entirely (ground truth present, nothing predicted)

| Class | Improved 3D U-Net | 3D U-Net (baseline) |
|---|---|---|
| body | 0/27 | 0/27 |
| bone | 0/27 | 0/27 |
| bladder | 0/27 | 0/27 |
| rectum | 0/27 | 0/27 |
| prostate | 0/27 | 0/27 |

### Prostate Dice by position along the gland

| Position | Improved 3D U-Net | 3D U-Net (baseline) | Difference |
|---|---|---|---|
| third 1 | 0.787 | 0.732 | 0.055 |
| third 2 (middle) | 0.909 | 0.895 | 0.014 |
| third 3 | 0.847 | 0.839 | 0.008 |

Thirds are taken along the slice axis of the ground-truth prostate extent. Decide which end is the base and which is the apex from the images before labelling them in the report.

### Computational cost

| Metric | Improved 3D U-Net | 3D U-Net (baseline) |
|---|---|---|
| Trainable parameters | 8,280,066 | 5,646,470 |
| Training peak VRAM (GB) | 1.14 | 0.70 |
| Inference peak VRAM (GB) | 0.76 | 0.66 |
| Inference time (s/volume) | 0.022 | 0.017 |
| Best epoch | 68 | 33 |

Add total training time from your Slurm logs.

### Paired comparison of prostate Dice over test volumes

- Volumes compared: 27 (improved minus baseline)
- Mean difference: +0.0072
- Improved model better on 16/27 volumes; worse on 11
- Wilcoxon signed-rank p = 0.0772

Volumes from the same patient are not independent, and this is a single training run per model, so treat any p-value as indicative only.
