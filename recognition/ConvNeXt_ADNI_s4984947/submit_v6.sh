#!/bin/bash
set -euo pipefail

last=""

for stage in 1 2 3 4 5; do
    if [ -z "$last" ]; then
        result=$(sbatch --parsable run_v6_12slices.sbatch)
    else
        result=$(sbatch --parsable \
          --dependency="afterok:${last}" \
          run_v6_12slices.sbatch)
    fi

    last="${result%%;*}"
    echo "V6 Stage $stage/5: Job $last"
done

squeue -u "$USER"
