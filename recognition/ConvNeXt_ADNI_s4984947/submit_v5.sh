#!/bin/bash
set -euo pipefail

last=""

for stage in 1 2 3 4; do
    if [ -z "$last" ]; then
        result=$(sbatch --parsable run_v5_smoothing.sbatch)
    else
        result=$(sbatch --parsable \
          --dependency="afterok:${last}" \
          run_v5_smoothing.sbatch)
    fi

    last="${result%%;*}"
    echo "V5 Stage $stage/4: Job $last"
done

squeue -u "$USER"
