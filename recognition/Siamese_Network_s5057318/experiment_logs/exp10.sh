#!/bin/bash
#SBATCH --job-name=exp10
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --output=exp10_%j.out
#SBATCH --error=exp10_%j.err
source $HOME/miniconda3/bin/activate torch
DATA=$HOME/data/isic2020
cd $HOME/exp7_code
mkdir -p outputs
for d in ep100 ep100lr3e4; do
  nd=outputs/${d}_proto
  rm -rf $nd
  cp -r $HOME/exp_code/outputs/$d $nd
  python predict.py $DATA --seed 42 --split_seed 42 --out_dir $nd --score prototype --n_refs 0 > $nd/predict.txt 2>&1
done
