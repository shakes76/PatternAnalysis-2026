#!/bin/bash
#SBATCH --job-name=yolo-latency
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu
#SBATCH --time=00:30:00
#SBATCH --output=yolo-latency-slurm-%x-%j.out

# Usage, from the project folder:
# sbatch scripts/latency.slurm [profile_latency.py options]
# e.g. sbatch scripts/latency.slurm --checkpoint runs/A0-s47041585/weights/best.pt

module load cuda/11.1 # as in train.slurm
set -euo pipefail

[[ -f pyproject.toml ]] || { echo "Run sbatch from the project folder, not $PWD" >&2; exit 1; }

export PATH="$HOME/.local/bin:$PATH" # where uv usually installs, and batch shells may not have it
nvidia-smi
uv run --frozen python scripts/profile_latency.py --device 0 "$@"
