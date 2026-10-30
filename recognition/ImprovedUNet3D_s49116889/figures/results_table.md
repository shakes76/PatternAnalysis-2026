| Class | 3D Improved U-Net Dice | 3D Improved U-Net IoU | 2D U-Net (baseline) Dice | 2D U-Net (baseline) IoU |
|---|---|---|---|---|
| background | 0.999 | 0.998 | 0.998 | 0.995 |
| body | 0.988 | 0.977 | 0.985 | 0.970 |
| bone | 0.929 | 0.868 | 0.925 | 0.861 |
| bladder | 0.952 | 0.910 | 0.950 | 0.906 |
| rectum | 0.864 | 0.764 | 0.855 | 0.753 |
| prostate | 0.870 | 0.773 | 0.822 | 0.700 |
| **foreground mean** | **0.921** |  | **0.907** |  |

| Prostate region | 3D Improved U-Net | 2D U-Net (baseline) |
|---|---|---|
| extremity slices (base + apex) | 0.840 | 0.778 |
| central slices | 0.907 | 0.889 |

| Resource | 3D Improved U-Net | 2D U-Net (baseline) |
|---|---|---|
| Parameters | 9,462,722 | 7,762,630 |
| Peak training VRAM (GB) | 2.77 | 0.92 |
| Peak inference VRAM (GB) | 0.76 | 0.45 |
| Inference time per volume (ms) | 269 | 260 |
| Training time (min) | 32 | 15 |
| Best epoch | 120 | 33 |
