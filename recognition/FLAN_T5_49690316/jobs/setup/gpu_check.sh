#!/bin/bash
#SBATCH --job-name=flan-gpu-check
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
# Requests one GPU. gres means "generic resources"



#SBATCH --time=00:05:00 


#SBATCH --output=gpu_check_%j.out
# Sends standard output (prints) to a file. Slurm replaces %j with the job id

#SBATCH --error=gpu_check_%j.err
# Sends errors and warnings to a separate file

set -euo pipefail
# Enables stricter error handling.
# -e exits on most unhandled command failures; -u errors on an unset variable
# pipefail makes a pipeline report failure if one of its commands fail


source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate torch

cd "$HOME/comp3710/flan_t5"

nvidia-smi
python -m scripts.diagnostics.check_gpu