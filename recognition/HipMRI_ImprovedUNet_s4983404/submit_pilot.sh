#!/bin/bash
#SBATCH --job-name=hipmri_pilot
#SBATCH --partition=a100-test
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=pilot_%j.out
set -eu
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2
source "$HOME/venv/bin/activate"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -u smoke_gpu.py
python -u train.py --model basic --run runs/basic_pilot --epochs 3 --cache --workers 0
python -u train.py --model improved --run runs/improved_pilot --epochs 3 --cache --workers 0
