#!/bin/bash
set -euo pipefail

last=""

for stage in 1 2 3 4; do
    if [ -z "$last" ]; then
        result=$(sbatch --parsable run_v4_aug.sbatch)
    else
        result=$(sbatch --parsable \
          --dependency="afterok:${last}" \
          run_v4_aug.sbatch)
    fi

    last="${result%%;*}"
    echo "V4 Stage $stage/4: Job $last"
done

squeue -u "$USER"
