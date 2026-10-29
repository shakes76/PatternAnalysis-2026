## Test prostate Dice (mean over the 28 test volumes, per run)

| Model | Runs | Mean ± std over runs |
|---|---|---|
| 2D U-Net (baseline) | 0.748, 0.682, 0.739 | 0.723 ± 0.036 (n=3) |
| 2D Improved U-Net | 0.702, 0.664, 0.714 | 0.693 ± 0.026 (n=3) |
| 3D Improved U-Net (matched settings) | 0.768, 0.737, 0.797 | 0.767 ± 0.030 (n=3) |
| 3D Improved U-Net (lr 5e-4, 50 epochs) | 0.761 | 0.761 (1 run) |
| 2D Improved U-Net (lr 5e-4) | 0.699 | 0.699 (1 run) |

## Prostate Dice per test patient (mean over runs)

| Model | B006 | G021 | K018 | S035 | T009 | W029 |
|---|---|---|---|---|---|---|
| 2D U-Net (baseline) | 0.795 | 0.697 | 0.779 | 0.807 | 0.903 | 0.643 |
| 2D Improved U-Net | 0.725 | 0.671 | 0.718 | 0.801 | 0.887 | 0.644 |
| 3D Improved U-Net (matched settings) | 0.862 | 0.794 | 0.797 | 0.776 | 0.918 | 0.667 |
| 3D Improved U-Net (lr 5e-4, 50 epochs) | 0.864 | 0.800 | 0.803 | 0.756 | 0.915 | 0.636 |
| 2D Improved U-Net (lr 5e-4) | 0.752 | 0.722 | 0.699 | 0.803 | 0.913 | 0.623 |

3D (matched) beats the 2D baseline on 5 of 6 test patients.

## Dice by prostate third, rectum/bladder leakage and inference time (from predict.py --analyse)

| Model | Base | Middle | Apex | Leak rectum % | Leak bladder % | s / volume |
|---|---|---|---|---|---|---|
| 2D U-Net (baseline) | 0.575 ± 0.086 (n=3) | 0.790 ± 0.031 (n=3) | 0.795 ± 0.030 (n=3) | 0.31 | 1.88 | 0.17 |
| 2D Improved U-Net | 0.539 ± 0.070 (n=3) | 0.765 ± 0.022 (n=3) | 0.785 ± 0.027 (n=3) | 0.58 | 1.62 | 0.20 |
| 3D Improved U-Net (matched settings) | 0.694 ± 0.018 (n=3) | 0.826 ± 0.030 (n=3) | 0.783 ± 0.045 (n=3) | 0.30 | 0.31 | 0.15 |
| 3D Improved U-Net (lr 5e-4, 50 epochs) | 0.698 (1 run) | 0.808 (1 run) | 0.796 (1 run) | 0.24 | 0.81 | 0.16 |
| 2D Improved U-Net (lr 5e-4) | 0.567 (1 run) | 0.755 (1 run) | 0.798 (1 run) | 0.31 | 0.93 | 0.19 |

## Per-class Dice / IoU on the test split (mean over the runs analysed)

| Model | bladder | rectum | prostate | bone | body | background |
|---|---|---|---|---|---|---|
| 2D U-Net (baseline) | 0.900 / 0.825 | 0.820 / 0.702 | 0.723 / 0.577 | 0.911 / 0.836 | 0.980 / 0.960 | 0.996 / 0.992 |
| 2D Improved U-Net | 0.861 / 0.772 | 0.798 / 0.669 | 0.693 / 0.541 | 0.911 / 0.837 | 0.976 / 0.953 | 0.994 / 0.989 |
| 3D Improved U-Net (matched settings) | 0.886 / 0.813 | 0.836 / 0.723 | 0.767 / 0.634 | 0.914 / 0.842 | 0.984 / 0.968 | 0.998 / 0.996 |
| 3D Improved U-Net (lr 5e-4, 50 epochs) | 0.922 / 0.863 | 0.832 / 0.716 | 0.761 / 0.625 | 0.920 / 0.852 | 0.985 / 0.971 | 0.998 / 0.996 |
| 2D Improved U-Net (lr 5e-4) | 0.876 / 0.791 | 0.842 / 0.731 | 0.699 / 0.545 | 0.917 / 0.848 | 0.981 / 0.963 | 0.996 / 0.992 |
