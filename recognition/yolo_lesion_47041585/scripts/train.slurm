#!/bin/bash
#SBATCH --job-name=yolo-train
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu
#SBATCH --time=00:30:00
#SBATCH --output=yolo-train-slurm-%x-%j.out

# Usage, from the project folder:
# sbatch scripts/train.slurm configs/{config}.yaml [--pilot] [--seed N]
# e.g. sbatch scripts/train.slurm configs/a0.yaml --pilot --seed 47041585

module load cuda/11.1
set -euo pipefail
CONFIG="${1:?usage: sbatch scripts/train.slurm <condition config> [--pilot] [--seed N]}"

[[ -f pyproject.toml ]] || { echo "Run sbatch from the project folder, not $PWD" >&2; exit 1; }

export PATH="$HOME/.local/bin:$PATH" # where uv usually installs, and batch shells may not have it
nvidia-smi
uv run --frozen train --config "$CONFIG" "${@:2}"
