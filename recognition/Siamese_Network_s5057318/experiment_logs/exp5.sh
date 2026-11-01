#!/bin/bash
#SBATCH --job-name=exp5
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=exp5_%j.out
#SBATCH --error=exp5_%j.err
source $HOME/miniconda3/bin/activate torch
DATA=$HOME/data/isic2020
cd $HOME/exp_code
for d in ctrl ep60; do
  python diagnose.py $DATA --seed 42 --split_seed 42 --out_dir outputs/$d > outputs/$d/diagnose.txt 2>&1
done
