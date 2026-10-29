#!/bin/bash
#SBATCH --job-name=adni-c2
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=c2-slurm-%j.out
set -euo pipefail
cd /home/Student/s4982104/comp3710-code/ConvNeXt_ADNI_s4982104
PYTHON_BIN="${1:?Pass the absolute path of your existing cluster Python interpreter}"
[[ "$PYTHON_BIN" = /* && -x "$PYTHON_BIN" ]] || { echo 'Python must be an absolute executable path'; exit 2; }
export OMP_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
EVIDENCE_DIR="$HOME/comp3710-data/adni/c2_pilot_${SLURM_JOB_ID}"
mkdir "$EVIDENCE_DIR"
printf '%s\n' "$SLURM_JOB_ID" > "$EVIDENCE_DIR/job_id.txt"
echo "Evidence: $EVIDENCE_DIR"
"$PYTHON_BIN" -m unittest discover -s tests -v > "$EVIDENCE_DIR/test_results.txt" 2>&1
"$PYTHON_BIN" train.py \
  --manifest "$HOME/comp3710-data/adni/split_seed42_60_20_20/split_samples.csv" \
  --summary "$HOME/comp3710-data/adni/split_seed42_60_20_20/split_summary.json" \
  --output-dir "$EVIDENCE_DIR/training" \
  --device cuda --epochs 3 --batch-size 16 --learning-rate 0.001 --seed 42
