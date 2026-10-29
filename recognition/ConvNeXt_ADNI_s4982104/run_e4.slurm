#!/bin/bash
#SBATCH --job-name=adni-e4-b42
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=e4-slurm-%j.out
set -euo pipefail
cd /home/Student/s4982104/comp3710-code/ConvNeXt_ADNI_s4982104
PYTHON_BIN="${1:?Pass absolute cluster Python path}"
[[ "$PYTHON_BIN" = /* && -x "$PYTHON_BIN" ]] || exit 2
A42_DIR="${2:-$HOME/comp3710-data/adni/e3_A42_694924/run}"
export OMP_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
EVIDENCE_DIR="$HOME/comp3710-data/adni/e4_B42_${SLURM_JOB_ID}"
mkdir "$EVIDENCE_DIR"
printf '%s\n' "$SLURM_JOB_ID" > "$EVIDENCE_DIR/job_id.txt"
echo "Evidence: $EVIDENCE_DIR"
"$PYTHON_BIN" -m unittest discover -s tests -v > "$EVIDENCE_DIR/test_results.txt" 2>&1
"$PYTHON_BIN" run_e4.py \
 --manifest "$HOME/comp3710-data/adni/split_seed42_60_20_20/split_samples.csv" \
 --summary "$HOME/comp3710-data/adni/split_seed42_60_20_20/split_summary.json" \
 --a42-dir "$A42_DIR" --output-dir "$EVIDENCE_DIR/run" --device cuda
