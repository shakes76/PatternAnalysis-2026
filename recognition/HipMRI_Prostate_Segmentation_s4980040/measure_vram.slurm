#!/bin/bash
# Slurm job: measure peak GPU memory (VRAM) and inference time on a Rangpur A100.
#   1. Inference: runs predict.py with the trained checkpoints (outputs/best_*.pt)
#   2. Training:  1-epoch runs of both models, saved to outputs_vram/ so the real
#                 checkpoints and results are never overwritten
# Submit with:  sbatch measure_vram.slurm
# Track with:   squeue -u $USER      Live log:  tail -f logs/vram_<jobid>.out

#SBATCH --job-name=vram
#SBATCH --partition=comp3710        # the course's A100 partition
#SBATCH --gres=gpu:1                # 1 GPU
#SBATCH --cpus-per-task=4           # CPU cores for data loading
#SBATCH --time=02:00:00             # time limit
#SBATCH --output=logs/%x_%j.out     # %x = job name, %j = job ID

source ~/miniforge3/etc/profile.d/conda.sh
conda activate comp3710
cd ~/PatternAnalysis-2026/recognition/HipMRI_Prostate_Segmentation_s4980040
set -e  # stop at the first error

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('GPU:', torch.cuda.get_device_name(0))"

echo "=== Inference VRAM and timing (trained checkpoints, full test set) ==="
python predict.py --model unet2d
python predict.py --model improved3d

echo "=== Training VRAM (1-epoch runs, separate output folder) ==="
python train.py --model unet2d --epochs 1 --preload --out-dir outputs_vram
python train.py --model improved3d --epochs 1 --out-dir outputs_vram

echo "=== Summary ==="
grep -h "Peak GPU\|Inference time\|^Model:" logs/vram_${SLURM_JOB_ID}.out || true
