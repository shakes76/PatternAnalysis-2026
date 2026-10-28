#!/bin/bash
# Run many small training runs at once on ONE GPU (each run needs well under 1 GB, so
# they share an A100). Usage: sbatch slurm_pack.sh <command-file>
# Each line of the command file is:  <out-dir> <model> <seed> [extra train.py args...]
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

CMDFILE=${1:?usage: sbatch slurm_pack.sh <command-file>}
DATA_DIR=${DATA_DIR:-$HOME/COMP3710/LOB_project/data}
export OMP_NUM_THREADS=1   # one CPU thread per process; the runs share the allocation

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source $HOME/miniconda3/bin/activate torch
nvidia-smi -L

while read -r outdir model seed extra; do
    [ -z "$outdir" ] && continue
    mkdir -p "$outdir"
    python3 -u train.py --model "$model" --data-dir "$DATA_DIR" --out-dir "$outdir" \
        --ae-epochs "${AE_EPOCHS:-30}" --sup-epochs "${SUP_EPOCHS:-30}" \
        --epochs "${EPOCHS:-100}" --eval-every 5 --seed "$seed" $extra \
        > "$outdir/train.log" 2>&1 &
done < "$CMDFILE"
wait
echo "all runs finished"
