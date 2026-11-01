#!/usr/bin/env bash

# train.sh
# Authored by: Gemini 3.8
# 
# Small bash script wrapper for automating ldm pipeline training

set -eo pipefail

DATA_DIR="${1:-ADNI}"
TRAIN_LOG="training.log"

echo "========================================================" | tee "$TRAIN_LOG"
echo " Starting LDM Pipeline Training:"                        | tee -a "$TRAIN_LOG"
echo " Dataset Path: $DATA_DIR"                                | tee -a "$TRAIN_LOG"
echo " Start Time:   $(date)"                                  | tee -a "$TRAIN_LOG"
echo "========================================================" | tee -a "$TRAIN_LOG"

# 1. Train VAE
echo -e "\n[$(date +'%T')] [1/2] Training VAE (100 Epochs)..." | tee -a "$TRAIN_LOG"
python -u train.py "$DATA_DIR" --vae 2>&1 | tee -a "$TRAIN_LOG"
echo "[$(date +'%T')] VAE Training Complete!" | tee -a "$TRAIN_LOG"

# 2. Train UNet
echo -e "\n[$(date +'%T')] [2/2] Training UNet (300 Epochs)..." | tee -a "$TRAIN_LOG"
python -u train.py "$DATA_DIR" --unet 2>&1 | tee -a "$TRAIN_LOG"
echo "[$(date +'%T')] UNet Training Complete!" | tee -a "$TRAIN_LOG"

echo -e "\n========================================================" | tee -a "$TRAIN_LOG"
echo " All Training Finished Successfully at: $(date)"          | tee -a "$TRAIN_LOG"
echo "========================================================" | tee -a "$TRAIN_LOG"
