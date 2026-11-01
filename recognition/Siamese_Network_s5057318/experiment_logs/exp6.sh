#!/bin/bash
#SBATCH --job-name=exp6
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=exp6_%j.out
#SBATCH --error=exp6_%j.err
source $HOME/miniconda3/bin/activate torch
DATA=$HOME/data/isic2020
cd $HOME/exp6_code
mkdir -p outputs
for src in ctrl ep60; do
  for cfg in "knn 0" "prototype 0" "prototype 200"; do
    set -- $cfg
    d=outputs/${src}_$1_n$2
    rm -rf $d
    cp -r $HOME/exp_code/outputs/$src $d
    python predict.py $DATA --seed 42 --split_seed 42 --out_dir $d --score $1 --n_refs $2 > $d/predict.txt 2>&1
  done
done
trainv() {
  name=$1; shift
  mkdir -p outputs/$name
  python train.py $DATA --model siamese --epochs 60 --num_workers 3 --seed 42 --split_seed 42 --out_dir outputs/$name "$@" > outputs/$name/train_siamese.log 2>&1
  cp $HOME/exp_code/outputs/ep60/baseline_best.pt outputs/$name/baseline_best.pt
  python predict.py $DATA --seed 42 --split_seed 42 --out_dir outputs/$name > outputs/$name/predict.txt 2>&1
}
trainv m20 --margin 2.0
trainv same --pair_source same
