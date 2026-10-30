#!/bin/bash
#SBATCH --job-name=RN18Te
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu
#SBATCH --time=00:30:00
#SBATCH --output=Outputs/Test_RN18.out
#SBATCH --error=Errors/Test_RN18.err

# Activate your conda environment
source $HOME/miniconda3/bin/activate
conda activate Main_Env

python $HOME/PatternAnalysis-2026/recognition/GFNet_48830553/predict.py --model "RN18" --version "LATEST"