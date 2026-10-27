#!/bin/bash
#!/bin/bash
#SBATCH --job-name=adni_cnn
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=cnn_%j.out
#SBATCH --error=cnn_%j.err

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate torch

cd ~/PatternAnalysis-2026/recognition/ConvNeXt_ADNI_YangjiaYan

echo "===== GPU Information ====="
hostname
nvidia-smi

echo "===== PyTorch ====="
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo "===== Start CNN Baseline Training ====="
python train.py \
    --model cnn \
    --epochs 15 \
    --batch-size 64 \
    --workers 4 \
    --lr 0.0001 \
    --seed 42 \
    --output-dir ~/data/comp3710_a3_runs_lr1e4


