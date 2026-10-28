#!/bin/bash
set -euo pipefail
last=""
for stage in 1 2 3 4 5 6; do
  if [ -z "$last" ]; then
    job=$(sbatch --parsable run_convnext_v3_long.sbatch)
  else
    job=$(sbatch --parsable --dependency="afterok:${last}" run_convnext_v3_long.sbatch)
  fi
  last="${job%%;*}"
  echo "Stage $stage/6: Job $last"
done
squeue -u "$USER"
