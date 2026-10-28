#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [ -f runs/phase4_benchmark/benchmark.json ]; then
    echo "Baseline/V7 benchmark already exists; will not rerun."
    benchmark=""
else
    benchmark=$(sbatch --parsable run_phase4_benchmark.sbatch)
    benchmark="${benchmark%%;*}"
    echo "Benchmark job: $benchmark"
fi
if [ -f runs/phase4_benchmark/review_rule.json ]; then
    echo "Validation-derived review rule already exists; will not rerun."
elif [ -n "$benchmark" ]; then
    policy=$(sbatch --parsable --dependency="afterok:$benchmark" run_phase4_review_policy.sbatch)
    echo "Review-rule job: ${policy%%;*} (depends on $benchmark)"
else
    policy=$(sbatch --parsable run_phase4_review_policy.sbatch)
    echo "Review-rule job: ${policy%%;*}"
fi
squeue -u "$USER"
