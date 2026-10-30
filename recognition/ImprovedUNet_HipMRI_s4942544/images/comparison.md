Mean ± standard deviation over random seeds (2D U-Net (baseline): 3, 2D Improved U-Net: 3, 3D Improved U-Net: 3, 3D Improved U-Net (half resolution): 3), held-out test split (3 patients, 540 slices).

**Prostate accuracy and clinical error modes**

| Metric | 2D U-Net (baseline) | 2D Improved U-Net | 3D Improved U-Net | 3D Improved U-Net (half resolution) |
|---|---|---|---|---|
| Prostate Dice (all slices) | 0.873 ± 0.003 | 0.871 ± 0.008 | 0.881 ± 0.004 | 0.872 ± 0.005 |
| Prostate IoU | 0.775 ± 0.005 | 0.771 ± 0.013 | 0.787 ± 0.006 | 0.773 ± 0.007 |
| Prostate Dice: central slices | 0.906 ± 0.001 | 0.909 ± 0.004 | 0.913 ± 0.004 | 0.908 ± 0.003 |
| Prostate Dice: edge slices | 0.741 ± 0.016 | 0.713 ± 0.028 | 0.749 ± 0.004 | 0.714 ± 0.012 |
| Prostate boundary F-score (2 px) | 0.920 ± 0.006 | 0.915 ± 0.012 | 0.937 ± 0.003 | 0.927 ± 0.004 |
| Under-contouring (% of prostate missed) | 12.2 ± 2.5 | 12.7 ± 0.7 | 9.0 ± 1.1 | 10.5 ± 1.2 |
| Leak into bladder (% of prostate area) | 0.92 ± 0.38 | 1.17 ± 0.31 | 1.04 ± 0.39 | 0.49 ± 0.26 |
| Leak into rectum (% of prostate area) | 0.26 ± 0.34 | 0.16 ± 0.04 | 0.03 ± 0.02 | 0.01 ± 0.02 |
| False-positive prostate-free slices (%) | 0.6 ± 0.8 | 0.6 ± 0.3 | 0.7 ± 0.4 | 0.3 ± 0.3 |

**Review-band rule (flag the slices around each end of the predicted prostate extent)**

| Metric | 2D U-Net (baseline) | 2D Improved U-Net | 3D Improved U-Net | 3D Improved U-Net (half resolution) |
|---|---|---|---|---|
| Slices flagged for manual review (%) | 13.3 ± 0.0 | 13.3 ± 0.0 | 13.3 ± 0.0 | 13.3 ± 0.0 |
| Missed prostate area caught by the flag (%) | 28.3 ± 4.4 | 36.0 ± 4.5 | 40.6 ± 2.7 | 45.7 ± 4.6 |
| False-positive area caught by the flag (%) | 18.8 ± 1.0 | 16.6 ± 1.4 | 13.9 ± 1.7 | 13.2 ± 2.5 |
| Whole-slice omissions (count) | 13.3 ± 2.3 | 17.7 ± 3.1 | 11.0 ± 2.0 | 13.3 ± 3.1 |
| Whole-slice omissions caught (%) | 95.1 ± 4.3 | 92.6 ± 1.9 | 87.7 ± 5.3 | 72.1 ± 2.6 |
| Prostate Dice on auto-drafted (unflagged) slices | 0.895 ± 0.002 | 0.895 ± 0.005 | 0.902 ± 0.004 | 0.898 ± 0.006 |

**Per-class Dice and IoU**

| Metric | 2D U-Net (baseline) | 2D Improved U-Net | 3D Improved U-Net | 3D Improved U-Net (half resolution) |
|---|---|---|---|---|
| body Dice | 0.980 ± 0.000 | 0.981 ± 0.000 | 0.981 ± 0.001 | 0.974 ± 0.000 |
| body IoU | 0.961 ± 0.001 | 0.963 ± 0.001 | 0.962 ± 0.003 | 0.950 ± 0.000 |
| bone Dice | 0.938 ± 0.000 | 0.940 ± 0.001 | 0.938 ± 0.003 | 0.922 ± 0.004 |
| bone IoU | 0.882 ± 0.001 | 0.887 ± 0.001 | 0.883 ± 0.006 | 0.855 ± 0.006 |
| bladder Dice | 0.964 ± 0.002 | 0.964 ± 0.001 | 0.960 ± 0.001 | 0.946 ± 0.005 |
| bladder IoU | 0.930 ± 0.004 | 0.931 ± 0.002 | 0.923 ± 0.002 | 0.897 ± 0.009 |
| rectum Dice | 0.852 ± 0.010 | 0.858 ± 0.020 | 0.851 ± 0.003 | 0.833 ± 0.032 |
| rectum IoU | 0.742 ± 0.016 | 0.752 ± 0.030 | 0.741 ± 0.005 | 0.714 ± 0.047 |
| prostate Dice | 0.873 ± 0.003 | 0.871 ± 0.008 | 0.881 ± 0.004 | 0.872 ± 0.005 |
| prostate IoU | 0.775 ± 0.005 | 0.771 ± 0.013 | 0.787 ± 0.006 | 0.773 ± 0.007 |

**Resource profile**

| Metric | 2D U-Net (baseline) | 2D Improved U-Net | 3D Improved U-Net | 3D Improved U-Net (half resolution) |
|---|---|---|---|---|
| Parameters (M) | 7.76 ± 0.00 | 3.19 ± 0.00 | 9.46 ± 0.00 | 9.46 ± 0.00 |
| Peak training VRAM (GB) | 0.89 ± 0.00 | 0.67 ± 0.00 | 7.66 ± 0.00 | 1.08 ± 0.00 |
| Training time per epoch (s) | 34.5 ± 0.1 | 38.3 ± 0.3 | 25.4 ± 0.2 | 5.8 ± 0.0 |
| Total training time (min) | 29.2 ± 0.1 | 32.3 ± 0.2 | 43.2 ± 0.3 | 20.7 ± 0.2 |
| Inference per slice (ms; 3D: volume time / slices) | 1.13 ± 0.11 | 0.92 ± 0.02 | 1.54 ± 0.03 | 0.40 ± 0.01 |
| Inference per scan (s) | 0.07 ± 0.01 | 0.06 ± 0.00 | 0.09 ± 0.00 | 0.02 ± 0.00 |
| Peak inference VRAM (GB) | 0.97 ± 0.00 | 0.59 ± 0.00 | 2.06 ± 0.00 | 0.37 ± 0.00 |
| Latency for one input, batch of 1 (ms; 3D: one volume) | 2.62 ± 0.01 | 5.98 ± 0.03 | 55.18 ± 0.13 | 7.60 ± 0.01 |
