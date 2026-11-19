#!/bin/bash
# Read-only Rangpur resource check. Run on the login node:
#     bash slurm/check_resources.sh [partition] > resources.txt 2>&1
# Nothing here submits jobs or changes allocations.

PARTITION="${1:-comp3710}"
DATA_DIR="/home/groups/comp3710/HipMRI_Study_open"

section() { printf '\n===== %s =====\n' "$1"; }

section "User / host"
whoami; hostname; date

section "All partitions (summary)"
sinfo -s

section "Partition ${PARTITION}: config (MaxTime, AllowGroups/QOS, TRES limits)"
scontrol show partition "${PARTITION}"

section "Partition ${PARTITION}: nodes, state, CPUs, memory (MB), GPUs (GRES)"
sinfo -N -p "${PARTITION}" -o "%N %T %c %m %G"

section "My Slurm associations (account, partition, QOS)"
sacctmgr -n -P show assoc user="$USER" format=Account,Partition,QOS,GrpTRES,MaxTRES,MaxJobs,MaxSubmit

section "QOS limits (per-user GPU/job caps decide whether two jobs can run at once)"
sacctmgr -n -P show qos format=Name,MaxTRESPU,MaxJobsPU,MaxSubmitPU,MaxWall,GrpTRES

section "My queued / running jobs"
squeue -u "$USER" -o "%i %P %j %T %M %l %D %R"

section "Partition ${PARTITION}: current queue"
squeue -p "${PARTITION}" -o "%i %u %j %T %M %l %b %R" | head -40

section "Dataset readable?"
ls -ld "${DATA_DIR}" "${DATA_DIR}/semantic_MRs" "${DATA_DIR}/semantic_labels_only"
echo "images: $(ls "${DATA_DIR}/semantic_MRs" | wc -l)  masks: $(ls "${DATA_DIR}/semantic_labels_only" | wc -l)"

section "Disk quota / home usage"
quota -s 2>/dev/null || echo "quota command unavailable"
df -h "$HOME"

section "GPU visible on this node? (usually none on the login node)"
nvidia-smi 2>/dev/null || echo "nvidia-smi not available here"

section "Python / PyTorch (current environment)"
python --version 2>&1
python -c "import torch; print('torch', torch.__version__, 'cuda build', torch.version.cuda)" 2>&1
