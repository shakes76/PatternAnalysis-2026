#!/bin/bash
#SBATCH --job-name=analyse_all
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:40:00
#SBATCH --output=outputs/%x_%j.out
#SBATCH --error=outputs/%x_%j.err
# Score every repeat run on the test split (base/middle/apex Dice, leakage, timing).
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
cd ~/PatternAnalysis-2026/recognition/ImprovedUNet_HipMRI_s4895678
for spec in "unet2d s1" "unet2d s2" "improved2d s1" "improved2d s2" "improved2d lr5e4" \
            "improved3d matched_s1" "improved3d matched_s2"; do
    set -- $spec
    echo "=== $1 $2 ==="
    python -u predict.py --model $1 --tag $2 --analyse
done
