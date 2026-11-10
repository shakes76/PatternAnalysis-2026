#!/bin/bash
#SBATCH --job-name=sd-r1
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=01:30:00
#SBATCH --output=logs/r1_train_%j.out
#SBATCH --error=logs/r1_train_%j.err
# S2 of docs/r1-seed-replicates-protocol.md: E2's or E3's exact command with training
# seed 4710, validation seed 3710, epoch-10 weights only.
# Usage: sbatch tools/r1_train.sbatch e2   (or e3)
# No --mem directive: on Rangpur one makes the job pend.
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
model="${1:?model required: e2 or e3}"
# Free space needed: E2 writes a 1.7 GB checkpoint, E3 a 1.6 MB adapter plus logs.
case "$model" in
    e2) recipe=(); need_mib=2048 ;;
    e3) recipe=(--trainable lora --lora-rank 4 --lr 1e-4); need_mib=200 ;;
    *) echo "unknown model $model" >&2; exit 2 ;;
esac
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
.venv-train/bin/python -c "import shutil; assert shutil.disk_usage('.').free > $need_mib * 1024**2, 'Need $need_mib MiB free'"
.venv-train/bin/python train.py --output "outputs/r1-$model-$SLURM_JOB_ID" --time-budget-minutes 75 \
    --exclude-defective-scans --expect-excluded 2 40 \
    --seed 4710 --val-seed 3710 --save-epochs 10 "${recipe[@]}"
