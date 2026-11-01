#!/bin/bash
#SBATCH --job-name=exp1
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=exp1_%j.out
#SBATCH --error=exp1_%j.err
source $HOME/miniconda3/bin/activate torch
DATA=$HOME/data/isic2020
cd $HOME/exp_code
mkdir -p outputs
cp -r $HOME/ctrl_code/outputs/ctrl outputs/ctrl
python predict.py $DATA --seed 42 --split_seed 42 --out_dir outputs/ctrl > outputs/ctrl/predict.txt 2>&1
run() {
  name=$1; shift
  mkdir -p outputs/$name
  for m in siamese baseline; do
    python train.py $DATA --model $m --epochs 30 --num_workers 3 --seed 42 --split_seed 42 --out_dir outputs/$name "$@" > outputs/$name/train_$m.log 2>&1
  done
  python predict.py $DATA --seed 42 --split_seed 42 --out_dir outputs/$name > outputs/$name/predict.txt 2>&1
}
run aug --augment strong
run lr3e4 --lr 0.0003
run ep60 --epochs 60
