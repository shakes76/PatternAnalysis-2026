#!/bin/bash
#SBATCH --job-name=adni-r6-final
#SBATCH --account=comp3710
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --output=r6-final-%j.out
#SBATCH --error=r6-final-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
PYTHON_BIN="${1:?Pass absolute cluster Python path}"
[[ "$PYTHON_BIN" = /* && -x "$PYTHON_BIN" ]] || exit 2
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
DATA_ROOT="$HOME/comp3710-data/adni"
EVIDENCE_DIR="$DATA_ROOT/r6_final_${SLURM_JOB_ID}"
mkdir "$EVIDENCE_DIR"
finish() {
 result=$?
 trap - EXIT
 set +e
 printf '%s\n' "$result" > "$EVIDENCE_DIR/exit_code.txt"
 nvidia-smi -q > "$EVIDENCE_DIR/nvidia_smi_after.txt" 2>&1
 cp "r6-final-${SLURM_JOB_ID}.out" "r6-final-${SLURM_JOB_ID}.err" "$EVIDENCE_DIR/" 2>/dev/null || result=1
 tar --exclude='*.pt' --exclude='*.pth' --exclude='*.partial' -czf "${EVIDENCE_DIR}_evidence.tar.gz" -C "$DATA_ROOT" "$(basename "$EVIDENCE_DIR")" || result=1
 printf '%s\n' "$result" > "$EVIDENCE_DIR/exit_code.txt"
 exit "$result"
}
trap finish EXIT
printf '%s\n' "$SLURM_JOB_ID" > "$EVIDENCE_DIR/job_id.txt"
"$PYTHON_BIN" -m pip freeze > "$EVIDENCE_DIR/pip_freeze.txt"
nvidia-smi -q > "$EVIDENCE_DIR/nvidia_smi_before.txt"
"$PYTHON_BIN" -m unittest tests.test_final_evaluation tests.test_r5_freeze tests.test_r6_final > "$EVIDENCE_DIR/test_results.txt" 2>&1
"$PYTHON_BIN" run_r6_final.py --data-root "$DATA_ROOT" --output-dir "$EVIDENCE_DIR/run" > "$EVIDENCE_DIR/run_stdout.txt" 2> "$EVIDENCE_DIR/run_stderr.txt"
