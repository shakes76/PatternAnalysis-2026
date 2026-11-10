#!/bin/bash
#SBATCH --job-name=final-gen
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=logs/final_generate_%j.out
#SBATCH --error=logs/final_generate_%j.err
# Stage 1 of docs/final-evaluation-protocol.md: COUNT seeds per class from
# FIRST_SEED (default 37200). Diffusion models sample one image at a time, the
# path every earlier cohort used: the rehearsal showed batched sampling changes
# a few images beyond rounding. Their 512 seeds per class run in chunks that fit
# this partition and are joined by tools/merge_cohorts.py.
# Usage: sbatch tools/final_generate.sbatch MODEL COUNT [FIRST_SEED]
#   MODEL: sd e0 e2 r1e2 e3 r1e3 b3 b1 b2 b2b
set -euo pipefail
umask 077
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
source tools/final_models.sh
name="${1:?model required}"
count="${2:?seeds per class required}"
first="${3:-37200}"
out="outputs/final-$name-s$first-$SLURM_JOB_ID"
python=.venv-train/bin/python
case "$name" in
    b1)
        [ "$first" = 37200 ] || { echo "B1 seeds always start at 37200" >&2; exit 2; }
        $python tools/sample_vae.py --checkpoint "$B1_CHECKPOINT" --output "$out" --count "$count"
        ;;
    b2) $python tools/baseline_gan.py generate --checkpoint "$B2_CHECKPOINT" --output "$out" \
        --first-seed "$first" --count "$count" ;;
    b2b)
        # Only the classes whose GAN produced a result (the B2b protocol).
        classes=()
        [ -f "$B2B_AD" ] && classes+=(--class-checkpoint "AD=$B2B_AD")
        [ -f "$B2B_NC" ] && classes+=(--class-checkpoint "NC=$B2B_NC")
        [ "${#classes[@]}" -gt 0 ] || { echo "no B2b class has a result" >&2; exit 2; }
        $python tools/baseline_gan.py generate "${classes[@]}" --allow-partial \
            --output "$out" --first-seed "$first" --count "$count"
        ;;
    *)
        if [ "$name" = e0 ] && [ ! -f "${WEIGHTS[e0]}" ]; then
            $python tools/export_e0_adapter.py --checkpoint "$E0_CHECKPOINT" --output "${WEIGHTS[e0]}"
        fi
        diffusion_args "$name"
        $python predict.py "${model_args[@]}" --output "$out" --first-seed "$first" \
            --count "$count" --batch-size 1
        ;;
esac
