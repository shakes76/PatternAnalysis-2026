#!/bin/bash
#SBATCH --job-name=exp3
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00
#SBATCH --output=exp3_%j.out
#SBATCH --error=exp3_%j.err
source $HOME/miniconda3/bin/activate torch
DATA=$HOME/data/isic2020
cd $HOME/exp_code
for cfg in "1 200" "3 200" "10 200" "20 200" "5 400" "10 400"; do
  set -- $cfg
  d=outputs/ep60_k$1_n$2
  rm -rf $d
  cp -r outputs/ep60 $d
  python predict.py $DATA --seed 42 --split_seed 42 --out_dir $d --k $1 --n_refs $2 > $d/predict.txt 2>&1
done
run() {
  name=$1; shift
  mkdir -p outputs/$name
  for m in siamese baseline; do
    python train.py $DATA --model $m --epochs 30 --num_workers 3 --seed 42 --split_seed 42 --out_dir outputs/$name "$@" > outputs/$name/train_$m.log 2>&1
  done
  python predict.py $DATA --seed 42 --split_seed 42 --out_dir outputs/$name > outputs/$name/predict.txt 2>&1
}
run ep100 --epochs 100
run ep100lr3e4 --epochs 100 --lr 0.0003
