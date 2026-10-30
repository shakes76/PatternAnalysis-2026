#!/bin/bash
#SBATCH --job-name=improved-train
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=train_improved_%j.out
#SBATCH --error=train_improved_%j.err

set -euo pipefail
source "$HOME/miniconda3/bin/activate"
conda activate torch
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

run_dir="runs/improved_${SLURM_JOB_ID}"
mkdir -p "$run_dir"
mkdir "$run_dir/source"
cp modules.py dataset.py sampling.py metrics.py losses.py train.py predict.py train_improved.sh "$run_dir/source/"
git rev-parse HEAD > "$run_dir/git_commit.txt"
git status --short > "$run_dir/git_status.txt"
git diff HEAD -- modules.py dataset.py sampling.py metrics.py losses.py train.py predict.py train_improved.sh > "$run_dir/uncommitted.patch"
python -m pip freeze > "$run_dir/environment.txt"
nvidia-smi > "$run_dir/gpu_info.txt"

python -u "$run_dir/source/train.py" \
    --model ImprovedUNet2D --loss ce_dice --dice-weight 1.0 \
    --epochs 20 --batch-size 8 --learning-rate 0.001 --seed 42 \
    --output "$run_dir/results"

python -u "$run_dir/source/predict.py" \
    --checkpoint "$run_dir/results/best.pt" \
    --output "$run_dir/validation_inspection"
