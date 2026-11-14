#!/bin/bash
#SBATCH --job-name=b3-epochs
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/b3_epochs_%j.out
#SBATCH --error=logs/b3_epochs_%j.err
# Post-hoc: does B3's replication grow with training? Compares its two saved
# checkpoints (epochs 49 and 98):
#   1. loss-based membership scores of epoch 49 against validation slices
#      (rehearsal mode; epoch 98's are in outputs/final-mia-b3-rehearsal-672913);
#   2. the first 128 seeds per class from each checkpoint (batch 16), and
#      their nearest training slices (tools/nearest_training.py), with the
#      final batch-1 epoch-98 cohort on the same seeds as a check.
# Submitted from the final run directory. Reads no test-split data.
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
source tools/final_models.sh
python=.venv-train/bin/python
e49=../b3-20261015/outputs/b3-668934/weights/unet-ema-epoch-49.safetensors
e98="${WEIGHTS[b3]}"
out="outputs/b3-epochs-$SLURM_JOB_ID"
mkdir -p "$out"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$python tools/membership.py --model b3-epoch49 --weights "$e49" --unet-init scratch \
    --exclude-scans "$EXCLUDED_SCANS" --mode rehearsal --output "$out/mia-epoch49"
for epoch in 49 98; do
    weights=$e49
    [ "$epoch" = 98 ] && weights=$e98
    $python predict.py --weights "$weights" --unet-init scratch \
        --output "$out/samples-epoch$epoch" --count 128 --batch-size 16
done
$python tools/nearest_training.py --real outputs/final-real-rehearsal-672908 \
    --cohort "epoch49=$out/samples-epoch49" --cohort "epoch98=$out/samples-epoch98" \
    --cohort "epoch98-final-batch1=outputs/final-b3-s37200-673092" \
    --seed-limit 37328 --output "$out/nearest"
