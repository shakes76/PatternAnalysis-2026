# Results summary (test set: 9 scans, 3 patients, full native resolution)

| Metric | 2D U-Net (baseline) | 3D Improved U-Net (main) |
|---|---|---|
| Prostate Dice (overall) | 0.863 | 0.859 |
| Prostate Dice per scan (mean ± std) | 0.866 ± 0.017 | 0.863 ± 0.030 |
| Prostate Dice, boundary slices | 0.707 | 0.778 |
| Prostate Dice, middle slices | 0.898 | 0.903 |
| Rectum Dice | 0.851 | 0.818 |
| Bone Dice | 0.933 | 0.896 |
| Bladder Dice | 0.966 | 0.945 |
| Parameters | 31.0M | 9.5M |
| Training time (M2 Pro, MPS) | 4.1 h (30 epochs) | 12.1 h (100 epochs) |
| Inference per scan (M2 Pro, MPS) | 740 ms | 840 ms |

3D prostate Dice at its training resolution (128x128x64): 0.874.
Boundary slices = first/last 20% of prostate-containing slices in each scan.