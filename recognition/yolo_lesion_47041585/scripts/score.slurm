#!/bin/bash
#SBATCH --job-name=yolo-score
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu
#SBATCH --time=00:30:00
#SBATCH --output=yolo-score-slurm-%x-%j.out

# Usage, from the project folder:
# sbatch scripts/score.slurm [score.py options]
# e.g. sbatch scripts/score.slurm --tau-m 0.07

module load cuda/11.1 # as in train.slurm
set -euo pipefail

[[ -f pyproject.toml ]] || { echo "Run sbatch from the project folder, not $PWD" >&2; exit 1; }

export PATH="$HOME/.local/bin:$PATH" # where uv usually installs, and batch shells may not have it
nvidia-smi
uv run --frozen python scripts/score.py --device 0 "$@"
