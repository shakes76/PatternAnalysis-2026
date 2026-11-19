#!/bin/bash
# Model B (bottleneck 512). Submit from the project directory:
#     METADATA=/path/to/metadata.csv sbatch slurm/train_unet2d_512.sbatch
# Extra arguments are passed to train.py, e.g. the overfit check:
#     METADATA=... RUN_NAME=overfit_unet2d_512 sbatch slurm/train_unet2d_512.sbatch --overfit-subset 8 --epochs 300
#
# Resource requests below are ESTIMATES, not measurements. Confirm the
# partition/QOS with slurm/check_resources.sh and revise --mem/--time after
# the overfit run reports peak RSS and the pilot reports epoch time.
#SBATCH --job-name=unet2d_512
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x-%j.out

set -euo pipefail
MODEL=unet2d_512
SEED=3710
EPOCHS="${EPOCHS:-20}"
: "${METADATA:?Set METADATA=/path/to/metadata.csv}"
RUN_NAME="${RUN_NAME:-${MODEL}_seed${SEED}_e${EPOCHS}}"
RUN_DIR="runs/${RUN_NAME}"

# Optional: VENV=/path/to/venv to activate a virtual environment.
if [[ -n "${VENV:-}" ]]; then source "${VENV}/bin/activate"; fi

echo "job ${SLURM_JOB_ID} on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python train.py --model "${MODEL}" --metadata "${METADATA}" --out-dir "${RUN_DIR}" \
    --epochs "${EPOCHS}" --seed "${SEED}" "$@"
echo "finished at $(date)"
