#!/bin/bash
#SBATCH --job-name=RN18Tr
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu
#SBATCH --time=05:00:00
#SBATCH --output=Outputs/Train_RN18.out
#SBATCH --error=Errors/Train_RN18.err

# Activate your conda environment
source $HOME/miniconda3/bin/activate
conda activate Main_Env

python $HOME/PatternAnalysis-2026/recognition/GFNet_48830553/train.py --model "RN18"