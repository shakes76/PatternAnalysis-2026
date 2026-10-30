#!/bin/bash
#SBATCH --job-name=improved-overfit
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --output=overfit_improved_%j.out
#SBATCH --error=overfit_improved_%j.err

set -euo pipefail
source "$HOME/miniconda3/bin/activate"
conda activate torch
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

run_dir="runs/overfit_improved_${SLURM_JOB_ID}"
mkdir -p "$run_dir"
mkdir "$run_dir/source"
cp modules.py dataset.py metrics.py losses.py overfit_improved.py overfit_improved.sh "$run_dir/source/"
git rev-parse HEAD > "$run_dir/git_commit.txt"
git status --short > "$run_dir/git_status.txt"
python -m pip freeze > "$run_dir/environment.txt"

python -u "$run_dir/source/overfit_improved.py" \
    --steps 400 \
    --output "$run_dir/results"
