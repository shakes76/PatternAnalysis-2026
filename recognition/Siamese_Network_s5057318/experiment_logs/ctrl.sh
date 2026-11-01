#!/bin/bash
#SBATCH --job-name=ctrl
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=ctrl_%j.out
#SBATCH --error=ctrl_%j.err
source $HOME/miniconda3/bin/activate torch
cd $HOME/ctrl_code
for m in siamese baseline; do
  python train.py $HOME/data/isic2020 --model $m --epochs 30 --num_workers 3 --seed 42 --split_seed 42 --out_dir outputs/ctrl > train_$m.log 2>&1
done
