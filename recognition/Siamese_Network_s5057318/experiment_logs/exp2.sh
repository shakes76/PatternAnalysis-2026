#!/bin/bash
#SBATCH --job-name=exp2
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=exp2_%j.out
#SBATCH --error=exp2_%j.err
source $HOME/miniconda3/bin/activate torch
DATA=$HOME/data/isic2020
cd $HOME/exp2_code
mkdir -p outputs/pre
for m in siamese baseline; do
  python train.py $DATA --model $m --epochs 30 --num_workers 3 --seed 42 --split_seed 42 --out_dir outputs/pre --preprocess grayworld > outputs/pre/train_$m.log 2>&1
done
python predict.py $DATA --seed 42 --split_seed 42 --out_dir outputs/pre --preprocess grayworld > outputs/pre/predict.txt 2>&1
