#!/bin/bash
#SBATCH --job-name=adni-r4-eff-v2
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=r4-efficiency-%j.out
#SBATCH --error=r4-efficiency-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
PYTHON_BIN="${1:?Pass absolute cluster Python path}"
[[ "$PYTHON_BIN" = /* && -x "$PYTHON_BIN" ]] || exit 2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
DATA_ROOT="$HOME/comp3710-data/adni"
EVIDENCE_DIR="$DATA_ROOT/r4_efficiency_v2_${SLURM_JOB_ID}"
mkdir "$EVIDENCE_DIR"
finish() {
 result=$?
 trap - EXIT
 set +e
 printf '%s\n' "$result" > "$EVIDENCE_DIR/exit_code.txt"
 nvidia-smi -q > "$EVIDENCE_DIR/nvidia_smi_after.txt" 2>&1
 cp "r4-efficiency-${SLURM_JOB_ID}.out" "r4-efficiency-${SLURM_JOB_ID}.err" "$EVIDENCE_DIR/"
 tar --exclude='*.pt' --exclude='*.pth' --exclude='*.partial' -czf "${EVIDENCE_DIR}_evidence.tar.gz" -C "$DATA_ROOT" "$(basename "$EVIDENCE_DIR")"
 exit "$result"
}
trap finish EXIT
printf '%s\n' "$SLURM_JOB_ID" > "$EVIDENCE_DIR/job_id.txt"
mkdir -p "$EVIDENCE_DIR/source_snapshot/tests"
cp ./*.py "$EVIDENCE_DIR/source_snapshot/"
cp tests/*.py "$EVIDENCE_DIR/source_snapshot/tests/"
cp r4_frozen_bests.json run_r4_efficiency.slurm R4_RUN_GUIDE.md "$EVIDENCE_DIR/source_snapshot/"
"$PYTHON_BIN" -m pip freeze > "$EVIDENCE_DIR/pip_freeze.txt"
nvidia-smi -q > "$EVIDENCE_DIR/nvidia_smi_before.txt"
"$PYTHON_BIN" -m unittest discover -s tests -v > "$EVIDENCE_DIR/test_results.txt" 2>&1
"$PYTHON_BIN" run_r4_efficiency.py \
 --data-root "$DATA_ROOT" \
 --lock "$PWD/r4_frozen_bests.json" \
 --output-dir "$EVIDENCE_DIR/run" \
 > "$EVIDENCE_DIR/run_stdout.txt" 2> "$EVIDENCE_DIR/run_stderr.txt"
