# 2D U-Net Baseline for Prostate Segmentation on HipMRI

# SEE 3D IMPROVED UNET FOR BREAKDOWN, THIS IS BASELINE
This folder contains the **2D U-Net baseline** for the 3D Improved U-Net project. The full report
(problem description, feasibility review, results, failure analysis and recommendation) is in
[`../ImprovedUNet3D_HipMRI_s4883582/README.md`](../ImprovedUNet3D_HipMRI_s4883582/README.md).

The baseline is a standard 2D U-Net [1] trained slice-by-slice on the HipMRI 2D slices
(`keras_slices_data`) with a combined cross-entropy + Dice loss. It is adapted from my own
COMP3710 Demo 2 U-Net.

## Files
| File         | Purpose                                                                             |
|--------------|-------------------------------------------------------------------------------------|
| `modules.py` | 2D U-Net architecture, Dice loss and combined CE + Dice loss                        |
| `dataset.py` | Nifti slice loading, normalisation, image/mask pairing and patient leakage check    |
| `utils.py`   | Per-class Dice and IoU accumulated over a whole split                               |
| `train.py`   | Training, validation-based checkpointing, loss/Dice plots and final test evaluation |
| `predict.py` | Test evaluation, resource profiling and failure case figures                        |

## Usage
1. Install `python-dotenv` (`pip install python-dotenv`) and create a `.env` file in this folder
   pointing at the HipMRI dataset (it is git-ignored, so each machine keeps its own):
   ```
   HIPMRI_ROOT=/home/groups/comp3710/HipMRI_Study_open
   ```
   A local copy must keep the same `keras_slices_data/keras_slices_*` folder structure.
   Setting `HIPMRI_ROOT` as a normal environment variable also works.
2. Train:
   ```bash
   python train.py
   ```
   This saves `outputs/unet2d_best.pt` and `outputs/unet2d_training_curves.png`.
3. Evaluate and visualise:
   ```bash
   python predict.py
   ```
   This prints per-class Dice/IoU, parameter count, latency and peak VRAM, and saves
   `outputs/failure_*.png` for the worst prostate slices.

## References
1. O. Ronneberger, P. Fischer, T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," MICCAI, 2015.
