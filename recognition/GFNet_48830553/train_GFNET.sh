#!/bin/bash
#SBATCH --job-name=GFNETTr
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu
#SBATCH --time=05:00:00
#SBATCH --output=Outputs/Train_GFNET.out
#SBATCH --error=Errors/Train_GFNET.err

# Activate your conda environment
source $HOME/miniconda3/bin/activate
conda activate Main_Env

python $HOME/PatternAnalysis-2026/recognition/GFNet_48830553/train.py --model "GFNET"