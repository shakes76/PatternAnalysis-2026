#!/bin/bash
#SBATCH --job-name=unet-ce-dice
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=train_ce_dice_%j.out
#SBATCH --error=train_ce_dice_%j.err

set -euo pipefail
source "$HOME/miniconda3/bin/activate"
conda activate torch
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

run_dir="runs/ce_dice_${SLURM_JOB_ID}"
mkdir -p "$run_dir"
mkdir "$run_dir/source"
cp dataset.py modules.py metrics.py sampling.py losses.py train.py check_losses.py train_ce_dice.sh "$run_dir/source/"
git rev-parse HEAD > "$run_dir/git_commit.txt"
git status --short > "$run_dir/git_status.txt"
git diff HEAD -- dataset.py modules.py metrics.py sampling.py losses.py train.py train_ce_dice.sh > "$run_dir/uncommitted.patch"
python -m pip freeze > "$run_dir/environment.txt"
nvidia-smi > "$run_dir/gpu_info.txt"

python -u "$run_dir/source/check_losses.py"
python -u "$run_dir/source/train.py" \
    --loss ce_dice --dice-weight 1.0 \
    --epochs 20 --batch-size 8 --learning-rate 0.001 --seed 42 \
    --output "$run_dir/results"
