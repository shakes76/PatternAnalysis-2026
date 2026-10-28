#!/bin/bash
#SBATCH --job-name=hipmri_smoke
#SBATCH --partition=a100-test
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --output=smoke_%j.out
set -eu
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2
source "$HOME/venv/bin/activate"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -u smoke_gpu.py
python -u train.py --model basic --run runs/basic_smoke --epochs 2 --limit 128 --val-limit 64 --workers 0
python -u train.py --model improved --run runs/improved_smoke --epochs 2 --limit 128 --val-limit 64 --workers 0
python -u predict.py --checkpoint runs/basic_smoke/best.pt --output runs/basic_preview --count 3
python -u predict.py --checkpoint runs/improved_smoke/best.pt --output runs/improved_preview --count 3
