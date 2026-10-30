#!/bin/bash
#SBATCH --job-name=GFNETTe
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu
#SBATCH --time=00:30:00
#SBATCH --output=Outputs/Test_GFNET.out
#SBATCH --error=Errors/Test_GFNET.err

# Activate your conda environment
source $HOME/miniconda3/bin/activate
conda activate Main_Env

python $HOME/PatternAnalysis-2026/recognition/GFNet_48830553/predict.py --model "GFNET" --version "LATEST"