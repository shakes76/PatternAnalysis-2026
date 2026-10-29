#!/bin/bash
#SBATCH --job-name=adni-n5
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=n5-slurm-%j.out
#SBATCH --error=n5-slurm-%j.err
set -euo pipefail
cd /home/Student/s4982104/comp3710-code/ConvNeXt_ADNI_s4982104
PYTHON_BIN="${1:?Pass absolute existing cluster Python path}"
[[ "$PYTHON_BIN" = /* && -x "$PYTHON_BIN" ]] || exit 2
TRAIN_SEED="${2:-42}"
case "$TRAIN_SEED" in 42|43|44) ;; *) echo "Seed must be 42,43,44" >&2; exit 2;; esac
export OMP_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
EVIDENCE_DIR="$HOME/comp3710-data/adni/convnext_formal_seed${TRAIN_SEED}_${SLURM_JOB_ID}"
mkdir "$EVIDENCE_DIR"
printf '%s\n' "$SLURM_JOB_ID" > "$EVIDENCE_DIR/job_id.txt"
trap 'command_exit_code=$?; printf "%s\n" "$command_exit_code" > "$EVIDENCE_DIR/exit_code.txt"' EXIT
"$PYTHON_BIN" -m unittest discover -s tests -v > "$EVIDENCE_DIR/test_results.txt" 2>&1
"$PYTHON_BIN" run_n5.py \
 --manifest "$HOME/comp3710-data/adni/split_seed42_60_20_20/split_samples.csv" \
 --summary "$HOME/comp3710-data/adni/split_seed42_60_20_20/split_summary.json" \
 --weights "$HOME/comp3710-data/pretrained/convnext_tiny-983f1562.pth" \
 --weights-summary "$HOME/comp3710-data/pretrained/convnext_tiny_weights_summary.json" \
 --reference-dir "$HOME/comp3710-data/adni/convnext_stability_lr3e5_702225/run" \
 --seed "$TRAIN_SEED" \
 --output-dir "$EVIDENCE_DIR/run" --device cuda
