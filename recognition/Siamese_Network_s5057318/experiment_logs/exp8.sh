#!/bin/bash
#SBATCH --job-name=exp8
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=exp8_%j.out
#SBATCH --error=exp8_%j.err
source $HOME/miniconda3/bin/activate torch
DATA=$HOME/data/isic2020
cd $HOME/exp7_code
mkdir -p outputs
for s in 7 123; do
  for cfg in "knn 200" "prototype 0" "knn 0"; do
    set -- $cfg
    d=outputs/lab_s${s}_${1}_n${2}
    rm -rf $d
    cp -r $HOME/seeds_lab/seed$s $d
    python predict.py $DATA --seed $s --split_seed 42 --out_dir $d --score $1 --n_refs $2 > $d/predict.txt 2>&1
  done
done
