#!/bin/bash
#SBATCH --job-name=exp7
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=exp7_%j.out
#SBATCH --error=exp7_%j.err
source $HOME/miniconda3/bin/activate torch
DATA=$HOME/data/isic2020
cd $HOME/exp7_code
mkdir -p sweeps
for d in ctrl ep60 ep100 ep100lr3e4; do
  python sweep_k.py $DATA --seed 42 --split_seed 42 --out_dir $HOME/exp_code/outputs/$d > sweeps/$d.txt 2>&1
done
for d in m20 same; do
  if [ -d $HOME/exp6_code/outputs/$d ]; then
    python sweep_k.py $DATA --seed 42 --split_seed 42 --out_dir $HOME/exp6_code/outputs/$d > sweeps/$d.txt 2>&1
  fi
done
