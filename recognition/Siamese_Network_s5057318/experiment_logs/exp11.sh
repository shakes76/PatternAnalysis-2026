#!/bin/bash
#SBATCH --job-name=exp11
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=exp11_%j.out
#SBATCH --error=exp11_%j.err
source $HOME/miniconda3/bin/activate torch
DATA=$HOME/data/isic2020
cd $HOME/exp7_code
mkdir -p outputs/bl_lr3e4_s42
cp $HOME/exp_code/outputs/ep100/siamese_best.pt outputs/bl_lr3e4_s42/siamese_best.pt
cp $HOME/exp_code/outputs/ep100lr3e4/baseline_best.pt outputs/bl_lr3e4_s42/baseline_best.pt
python predict.py $DATA --seed 42 --split_seed 42 --out_dir outputs/bl_lr3e4_s42 --score prototype --n_refs 0 > outputs/bl_lr3e4_s42/predict.txt 2>&1
