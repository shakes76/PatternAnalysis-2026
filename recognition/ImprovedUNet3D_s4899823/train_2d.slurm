#!/bin/bash
#SBATCH --job-name=unet2d
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
eval "$(conda shell.bash hook)"
conda activate comp3710
cd ~/PatternAnalysis-2026/recognition/ImprovedUNet3D_s4899823
nvidia-smi
python -u train.py --dims 2 --root data/rangpur --batch-size 16 --epochs 100 --num-workers 4
