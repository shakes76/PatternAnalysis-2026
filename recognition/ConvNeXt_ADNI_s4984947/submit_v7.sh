#!/bin/bash
set -euo pipefail
previous=""
for stage in 1 2 3 4 5; do
  if [ -z "$previous" ]; then
    result=$(sbatch --parsable run_v7_finetune.sbatch)
  else
    result=$(sbatch --parsable --dependency="afterok:${previous}" run_v7_finetune.sbatch)
  fi
  previous="${result%%;*}"
  echo "V7 stage ${stage}/5: Job ${previous}"
done
squeue -u "$USER"
