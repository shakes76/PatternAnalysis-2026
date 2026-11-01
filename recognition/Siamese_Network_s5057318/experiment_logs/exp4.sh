#!/bin/bash
#SBATCH --job-name=exp4
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=exp4_%j.out
#SBATCH --error=exp4_%j.err
source $HOME/miniconda3/bin/activate torch
DATA=$HOME/data/isic2020
cd $HOME/exp_code
run() {
  name=$1; seed=$2; shift; shift
  mkdir -p outputs/$name
  for m in siamese baseline; do
    python train.py $DATA --model $m --epochs 30 --num_workers 3 --seed $seed --split_seed 42 --out_dir outputs/$name "$@" > outputs/$name/train_$m.log 2>&1
  done
  python predict.py $DATA --seed $seed --split_seed 42 --out_dir outputs/$name > outputs/$name/predict.txt 2>&1
}
run ctrl_s7 7
run ctrl_s123 123
run ep60_s7 7 --epochs 60
run ep60_s123 123 --epochs 60
