#!/bin/bash
#SBATCH --job-name=yolo-final
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu
#SBATCH --time=00:30:00
#SBATCH --output=yolo-final-slurm-%x-%j.out

# Usage, from the project folder:
# sbatch scripts/final_test.slurm freeze --candidate A0 --baseline B0 --seed 47041585 --tau-m 0.07 --final

# sbatch scripts/final_test.slurm test --split val      # dry run on validation
# sbatch scripts/final_test.slurm test                  # the final test-set evaluation

# The first word is the final_test.py command. The test split is evaluated once: `test`
# refuses unless the record is final, and will not overwrite an existing result.

module load cuda/11.1 # as in train.slurm
set -euo pipefail

[[ -f pyproject.toml ]] || { echo "Run sbatch from the project folder, not $PWD" >&2; exit 1; }

export PATH="$HOME/.local/bin:$PATH" # where uv usually installs, and batch shells may not have it
nvidia-smi
uv run --frozen python scripts/final_test.py "$@" --device 0
