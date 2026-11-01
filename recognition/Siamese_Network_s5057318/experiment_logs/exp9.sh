#!/bin/bash
#SBATCH --job-name=exp9
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00
#SBATCH --output=exp9_%j.out
#SBATCH --error=exp9_%j.err
source $HOME/miniconda3/bin/activate torch
DATA=$HOME/data/isic2020
cd $HOME/exp7_code
mkdir -p outputs
for s in 7 123; do
  d=outputs/ep100_s$s
  mkdir -p $d
  for m in siamese baseline; do
    python train.py $DATA --model $m --epochs 100 --num_workers 3 --seed $s --split_seed 42 --out_dir $d > $d/train_$m.log 2>&1
  done
  python predict.py $DATA --seed $s --split_seed 42 --out_dir $d --score knn --n_refs 200 > $d/predict_knn.txt 2>&1
  python predict.py $DATA --seed $s --split_seed 42 --out_dir $d --score prototype --n_refs 0 > $d/predict_prototype.txt 2>&1
  b=outputs/bl_lr3e4_s$s
  mkdir -p $b
  python train.py $DATA --model baseline --epochs 100 --lr 0.0003 --num_workers 3 --seed $s --split_seed 42 --out_dir $b > $b/train_baseline.log 2>&1
  cp $d/siamese_best.pt $b/siamese_best.pt
  python predict.py $DATA --seed $s --split_seed 42 --out_dir $b --score prototype --n_refs 0 > $b/predict.txt 2>&1
done
