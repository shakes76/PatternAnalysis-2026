#!/bin/bash
#SBATCH --job-name=adni-r2
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=r2-slurm-%j.out
#SBATCH --error=r2-slurm-%j.err
set -euo pipefail
# Submit from the project directory: Slurm copies this script into its spool.
cd "${SLURM_SUBMIT_DIR:?Submit using sbatch from the project directory}"
PYTHON_BIN="${1:?Pass absolute cluster Python path}"
DATA_ROOT="${R2_DATA_ROOT:-$HOME/comp3710-data/adni}"
WEIGHTS_DIR="${R2_WEIGHTS_DIR:-$HOME/comp3710-data/pretrained}"
N4_REFERENCE="${2:-$DATA_ROOT/convnext_stability_lr3e5_702225/run/training/training_summary.json}"
[[ "$PYTHON_BIN" = /* && -x "$PYTHON_BIN" ]] || { echo 'Invalid Python path' >&2; exit 2; }
[[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || { echo 'Missing Slurm job ID' >&2; exit 2; }
for input in run_r2.py "$DATA_ROOT/split_seed42_60_20_20/split_samples.csv" \
 "$DATA_ROOT/split_seed42_60_20_20/split_summary.json" \
 "$WEIGHTS_DIR/convnext_tiny-983f1562.pth" \
 "$WEIGHTS_DIR/convnext_tiny_weights_summary.json" "$N4_REFERENCE"; do
 [[ -r "$input" ]] || { echo "Missing/unreadable input: $input" >&2; exit 2; }
done
export OMP_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
EVIDENCE_DIR="$DATA_ROOT/r2_diagnostic_${SLURM_JOB_ID}"
mkdir "$EVIDENCE_DIR" # Deliberately refuse overwriting a previous run.
finish() {
 command_exit_code=$?
 trap - EXIT
 set +e
 printf '%s\n' "$command_exit_code" > "$EVIDENCE_DIR/exit_code.txt"
 for log in "r2-slurm-${SLURM_JOB_ID}.out" "r2-slurm-${SLURM_JOB_ID}.err"; do
  [[ ! -f "$log" ]] || cp "$log" "$EVIDENCE_DIR/"
 done
 tar --exclude='*.pt' --exclude='*.pth' --exclude='*.partial' \
  -czf "${EVIDENCE_DIR}_evidence.tar.gz" -C "$DATA_ROOT" "$(basename "$EVIDENCE_DIR")"
 archive_exit_code=$?
 if [[ "$command_exit_code" -eq 0 && "$archive_exit_code" -ne 0 ]]; then
  command_exit_code=$archive_exit_code
  printf '%s\n' "$command_exit_code" > "$EVIDENCE_DIR/exit_code.txt"
 fi
 exit "$command_exit_code"
}
trap finish EXIT
printf '%s\n' "$SLURM_JOB_ID" > "$EVIDENCE_DIR/job_id.txt"
printf '%s\n' "$PYTHON_BIN" > "$EVIDENCE_DIR/python_path.txt"
printf '%s\n' "$N4_REFERENCE" > "$EVIDENCE_DIR/n4_reference_path.txt"
mkdir "$EVIDENCE_DIR/source_snapshot" "$EVIDENCE_DIR/source_snapshot/tests"
cp ./*.py "$EVIDENCE_DIR/source_snapshot/"
cp ./tests/*.py "$EVIDENCE_DIR/source_snapshot/tests/"
cp "${BASH_SOURCE[0]}" "$EVIDENCE_DIR/source_snapshot/run_r2.slurm"
[[ ! -f requirements.txt ]] || cp requirements.txt "$EVIDENCE_DIR/source_snapshot/"
cp "$N4_REFERENCE" "$EVIDENCE_DIR/n4_training_summary.json"
"$PYTHON_BIN" -m pip freeze > "$EVIDENCE_DIR/pip_freeze.txt"
"$PYTHON_BIN" -m unittest discover -s tests -v > "$EVIDENCE_DIR/test_results.txt" 2>&1
"$PYTHON_BIN" run_r2.py \
 --manifest "$DATA_ROOT/split_seed42_60_20_20/split_samples.csv" \
 --summary "$DATA_ROOT/split_seed42_60_20_20/split_summary.json" \
 --weights "$WEIGHTS_DIR/convnext_tiny-983f1562.pth" \
 --weights-summary "$WEIGHTS_DIR/convnext_tiny_weights_summary.json" \
 --n4-reference "$N4_REFERENCE" \
 --output-dir "$EVIDENCE_DIR/run" --device cuda \
 > "$EVIDENCE_DIR/run_stdout.txt" 2> "$EVIDENCE_DIR/run_stderr.txt"
