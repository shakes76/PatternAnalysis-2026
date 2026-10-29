#!/bin/bash
#SBATCH --job-name=adni-c3
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --time=00:20:00
#SBATCH --output=c3-slurm-%j.out
set -euo pipefail
cd /home/Student/s4982104/comp3710-code/ConvNeXt_ADNI_s4982104
PYTHON_BIN="${1:?Pass the absolute cluster Python interpreter path}"
[[ "$PYTHON_BIN" = /* && -x "$PYTHON_BIN" ]] || exit 2
TRAIN_DIR="${2:-$HOME/comp3710-data/adni/c2_pilot_694112/training}"
export OMP_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
EVIDENCE_DIR="$HOME/comp3710-data/adni/c3_validation_${SLURM_JOB_ID}"
mkdir "$EVIDENCE_DIR"
printf '%s\n' "$SLURM_JOB_ID" > "$EVIDENCE_DIR/job_id.txt"
echo "Evidence: $EVIDENCE_DIR"
"$PYTHON_BIN" -m unittest discover -s tests -v > "$EVIDENCE_DIR/test_results.txt" 2>&1
"$PYTHON_BIN" predict.py \
 --manifest "$HOME/comp3710-data/adni/split_seed42_60_20_20/split_samples.csv" \
 --summary "$HOME/comp3710-data/adni/split_seed42_60_20_20/split_summary.json" \
 --checkpoint "$TRAIN_DIR/best.pt" --training-summary "$TRAIN_DIR/training_summary.json" \
 --output-dir "$EVIDENCE_DIR/prediction" --device cuda --batch-size 16
