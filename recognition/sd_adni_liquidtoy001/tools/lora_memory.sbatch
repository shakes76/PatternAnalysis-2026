#!/bin/bash
#SBATCH --job-name=sd-lora-memory
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/lora_memory_%j.out
#SBATCH --error=logs/lora_memory_%j.err
# Post-hoc memory check of E3 (tools/lora_memory.py). Submitted from the E3 run
# directory so it uses the modules.py, environment and model E3 trained with.
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# Pass 1: every variant with cuDNN autotuning, as trained. Pass 2: full and
# LoRA without it.
for variant in full lora lora-bf16base frozen; do
    .venv-train/bin/python tools/lora_memory.py --variant "$variant" \
        --output "outputs/lora-memory-$SLURM_JOB_ID" || echo "variant $variant failed"
done
for variant in full lora; do
    .venv-train/bin/python tools/lora_memory.py --variant "$variant" --no-benchmark \
        --output "outputs/lora-memory-$SLURM_JOB_ID" || echo "variant $variant failed"
done
