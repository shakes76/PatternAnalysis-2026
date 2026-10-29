#!/bin/bash
#SBATCH --job-name=adni-r3-conv42
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=r3-convnext-%j.out
#SBATCH --error=r3-convnext-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
PYTHON_BIN="${1:?Pass absolute cluster Python path}"
[[ "$PYTHON_BIN" = /* && -x "$PYTHON_BIN" ]] || exit 2
export OMP_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
DATA_ROOT="$HOME/comp3710-data/adni"
EVIDENCE_DIR="$DATA_ROOT/r3_convnext_seed42_${SLURM_JOB_ID}"
mkdir "$EVIDENCE_DIR"
finish() {
 result=$?
 trap - EXIT
 set +e
 printf '%s\n' "$result" > "$EVIDENCE_DIR/exit_code.txt"
 cp "r3-convnext-${SLURM_JOB_ID}.out" "r3-convnext-${SLURM_JOB_ID}.err" "$EVIDENCE_DIR/"
 tar --exclude='*.pt' --exclude='*.pth' --exclude='*.partial' -czf "${EVIDENCE_DIR}_evidence.tar.gz" -C "$DATA_ROOT" "$(basename "$EVIDENCE_DIR")"
 exit "$result"
}
trap finish EXIT
printf '%s\n' "$SLURM_JOB_ID" > "$EVIDENCE_DIR/job_id.txt"
mkdir -p "$EVIDENCE_DIR/source_snapshot/tests"
cp ./*.py "$EVIDENCE_DIR/source_snapshot/"
cp tests/*.py "$EVIDENCE_DIR/source_snapshot/tests/"
cp -R reference_sources "$EVIDENCE_DIR/source_snapshot/"
cp "${BASH_SOURCE[0]}" "$EVIDENCE_DIR/source_snapshot/run_r3_convnext.slurm"
"$PYTHON_BIN" -m pip freeze > "$EVIDENCE_DIR/pip_freeze.txt"
"$PYTHON_BIN" -m unittest discover -s tests -v > "$EVIDENCE_DIR/test_results.txt" 2>&1
"$PYTHON_BIN" run_r3_convnext.py \
 --seed 42 --device cuda \
 --manifest "$DATA_ROOT/split_seed42_60_20_20/split_samples.csv" \
 --summary "$DATA_ROOT/split_seed42_60_20_20/split_summary.json" \
 --weights "$HOME/comp3710-data/pretrained/convnext_tiny-983f1562.pth" \
 --weights-summary "$HOME/comp3710-data/pretrained/convnext_tiny_weights_summary.json" \
 --n5-reference "$DATA_ROOT/convnext_formal_seed42_702352/run/training" \
 --r2-reference "$DATA_ROOT/r2_diagnostic_704838/run/F" \
 --output-dir "$EVIDENCE_DIR/run" > "$EVIDENCE_DIR/run_stdout.txt" 2> "$EVIDENCE_DIR/run_stderr.txt"
