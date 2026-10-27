#!/bin/bash
# submit_final.sh - submit the four headline runs (zeroshot, LoRA r=8/16, full FT).
# Author : Trung Bao Truong (Brian Truong), student ID 49846627 (s4984662) | COMP3710 Project 2.6
# Usage  : bash slurm/submit_final.sh   (from the project folder on Rangpur)
# Skips a run when metrics.json exists, or when a Slurm job with that name is already queued/running.
# Learning rates: LoRA 3e-3, full 3e-4 (the best of the development-set sweep, see the README).
# Effective batch 16 (micro-batch 8, grad_accum 2); wall time 2 h zero-shot, 5 h per fine-tune.

set -euo pipefail

LORA_LR=3e-3
FULL_LR=3e-4
BATCH="--batch_size 8 --grad_accum 2"

if [[ ! -f train.py || ! -f slurm/train.sh ]]; then
  echo "Run this from recognition/FLAN-T5_LaySumm_s4984662/ (need train.py and slurm/train.sh)" >&2
  exit 1
fi

mkdir -p runs

submit_one() {
  local name="$1" wall="$2"
  shift 2
  if [[ -f "runs/${name}/metrics.json" ]]; then
    echo "skip (already finished): ${name}"
    return 0
  fi
  if [[ -n "$(squeue --me -h -n "rrs-${name}" 2>/dev/null)" ]]; then
    echo "skip (already queued/running): ${name}"
    return 0
  fi
  rm -rf "runs/${name}"
  local jid
  jid=$(sbatch --parsable \
    --job-name="rrs-${name}" \
    --time="${wall}" \
    --output="runs/${name}_%j.out" \
    --error="runs/${name}_%j.err" \
    slurm/train.sh \
    --output_dir "runs/${name}" \
    ${BATCH} \
    "$@")
  echo "submitted ${name}  job ${jid}  wall ${wall}"
}

submit_one "zeroshot"              "02:00:00" --mode zeroshot
submit_one "lora_r8_lr${LORA_LR}"  "05:00:00" --mode lora --lora_r 8  --lr "${LORA_LR}"
submit_one "lora_r16_lr${LORA_LR}" "05:00:00" --mode lora --lora_r 16 --lr "${LORA_LR}"
submit_one "full_lr${FULL_LR}"     "05:00:00" --mode full --lr "${FULL_LR}"

echo
squeue --me
