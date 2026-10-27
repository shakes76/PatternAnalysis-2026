#!/bin/bash
#SBATCH --job-name=hipmri-independent-prepare
#SBATCH --account=comp3710
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --time=00:25:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPYCACHEPREFIX="$SLURM_SUBMIT_DIR/artifacts/pycache"
export HIPMRI_V1_PROJECT_ROOT=/home/Student/s4927782/comp3710-demo3
DEMO3_PYTHON="$HIPMRI_V1_PROJECT_ROOT/.venv/bin/python"
for fixture in test_v2_pipeline.py test_small_fit.py test_qa_v2.py test_compare_common_grid.py test_real_pipeline_check.py test_infer_v2.py; do
  "$DEMO3_PYTHON" -m unittest discover -s . -p "$fixture" -v
done
"$DEMO3_PYTHON" -u v2_pipeline.py --mode prepare \
  --protocol protocol_v2.json --project-root "$HIPMRI_V1_PROJECT_ROOT" \
  --manifest "$HIPMRI_V1_PROJECT_ROOT/artifacts/cache64/manifest.json" \
  --raw-root /home/groups/comp3710/HipMRI_Study_open \
  --output-dir artifacts/cache_v2
