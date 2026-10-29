#!/bin/bash
#SBATCH --job-name=predict
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --output=outputs/%x_%j.out
#SBATCH --error=outputs/%x_%j.err
# Analyse the three first-run models on the test split and save the four-panel failure figures.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
cd ~/PatternAnalysis-2026/recognition/ImprovedUNet_HipMRI_s4895678
python -u predict.py --model unet2d --analyse --figures W029_Week5 G021_Week5 G021_Week7 T009_Week0
python -u predict.py --model improved2d --analyse --figures W029_Week5 G021_Week5 G021_Week7 T009_Week0
python -u predict.py --model improved3d --analyse --figures W029_Week5 G021_Week5 G021_Week7 T009_Week0
