#!/bin/bash
#SBATCH --job-name=hipmri-recovery-R1
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# TRAIN/VAL only: policy fixtures, isolated smokes, then fixed C/D eight-epoch arms.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$SLURM_SUBMIT_DIR/artifacts/recovery_mpl_${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
.venv/bin/python -m unittest discover -s tests -p test_recovery.py -v
.venv/bin/python -m unittest discover -s tests -p test_evaluate_recovery.py -v
for variant in clean_adapt polarity_rehearsal; do
  .venv/bin/python -u experiments/train_recovery.py \
    --protocol recovery_protocol.json --variant "$variant" --device cuda \
    --smoke-output "artifacts/recovery_smoke_${SLURM_JOB_ID}/$variant"
done
for variant in clean_adapt polarity_rehearsal; do
  .venv/bin/python -u experiments/train_recovery.py \
    --protocol recovery_protocol.json --variant "$variant" --device cuda
done

.venv/bin/python -u experiments/evaluate_recovery.py \
  --protocol recovery_protocol.json --manifest artifacts/cache64/manifest.json \
  --output-dir "artifacts/recovery_evaluation_R1_${SLURM_JOB_ID}" --device cuda
