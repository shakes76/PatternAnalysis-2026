#!/bin/bash
#SBATCH --job-name=adni
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Usage: sbatch train.sh <model> [epochs] [lr]
#   e.g. sbatch train.sh convnext 30 1e-3
MODEL=${1:-convnext}
EPOCHS=${2:-30}
LR=${3:-1e-3}

source $HOME/miniconda3/bin/activate
conda activate torch

nvidia-smi
python train.py \
  --model "$MODEL" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --data_root /home/groups/comp3710/ADNI/AD_NC \
  --num_workers 4 \
  --out_dir outputs
