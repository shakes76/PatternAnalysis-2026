#!/bin/bash
#SBATCH --job-name=sd-mem-phases
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/train_memory_phases_%j.out
#SBATCH --error=logs/train_memory_phases_%j.err
# Post-hoc: peak memory of each train.py phase (tools/train_memory_phases.py),
# for the E3 configuration (MODE=lora) or E2's (MODE=unet), 50 steps of the
# E3 probe setting; NOBENCH=1 turns cuDNN autotuning off while sampling.
# Submitted from the E3 run directory so it uses E3's code. Reads the train
# and validation splits only. Run output goes to job scratch.
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mode="${MODE:?set MODE=lora or MODE=unet}"
extra=()
if [ "$mode" = lora ]; then extra=(--trainable lora --lora-rank 4 --lr 1e-4); fi
tag=""
flags=()
if [ "${NOBENCH:-0}" = 1 ]; then tag="-nobench"; flags=(--sampling-no-benchmark); fi
.venv-train/bin/python tools/train_memory_phases.py "${flags[@]}" \
    --report "outputs/train-memory-phases-$SLURM_JOB_ID/$mode$tag.json" -- \
    --output "${TMPDIR:?}/run-$mode" --weights-dir "$TMPDIR/weights-$mode" \
    "${extra[@]}" --max-steps 50 --val-images 256 --save-epochs \
    --exclude-defective-scans --expect-excluded 2 40
