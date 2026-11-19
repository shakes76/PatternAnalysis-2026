#!/bin/bash
# Submit an experiment group to Slurm: every model x seed x setting.
#   ./run_experiments.sh transform      # return-transform ablation (zscore/clip/quantile)
#   ./run_experiments.sh step           # grid step size (0.5/1/5/10 s)
#   ./run_experiments.sh loss           # TimeGAN loss weights, feature discriminator, cosine LR
#   ./run_experiments.sh ladder         # price ladder as distances vs gaps between levels
#   ./run_experiments.sh combined       # best changes from the other groups, together
#   ./run_experiments.sh depth          # difference moment-matching loss for depth persistence
#   ./run_experiments.sh gatesup        # hold gate supervised with the real change pattern
#   ./run_experiments.sh improve        # longer windows and the hold-gate decoder (SBATCH_TIME=04:00:00)
# Results land in outputs/<group>/<setting>/<model>_s<seed>/ (gitignored).
export PATH=/opt/slurm/bin:$PATH
GROUP=${1:?usage: run_experiments.sh <transform|step|loss|depth|ladder|combined>}
SEEDS=${SEEDS:-"0 1 2"}

# Runs are packed: one Slurm job per (group, seed) runs all its settings and models at once
# on a single GPU (slurm_pack.sh). DRYRUN=1 only writes the command files.
submit() {  # submit <setting-name> <model> <seed> <extra train.py args...>
    local setting=$1 model=$2 seed=$3; shift 3
    echo "outputs/$GROUP/$setting/${model}_s$seed $model $seed $*" >> "$CMDFILE"
}

for seed in $SEEDS; do
  CMDFILE=outputs/$GROUP/cmds_s$seed.txt
  mkdir -p outputs/$GROUP && : > "$CMDFILE"
  for model in timegan baseline; do
    # loss/depth settings only change TimeGAN training, so the baseline is skipped
    [[ $model == baseline && ( $GROUP == loss || $GROUP == depth ) ]] && continue
    case $GROUP in
      transform)
        for t in zscore clip quantile; do submit $t $model $seed --return-transform $t; done ;;
      step)
        for st in 0.5 1 5 10; do submit step$st $model $seed --step $st; done ;;
      loss)
        submit default  $model $seed
        submit mom10    $model $seed --w-moment 10
        submit mom0     $model $seed --w-moment 0
        submit sup10    $model $seed --w-sup 10
        submit featadv  $model $seed --feature-adv
        submit cosine   $model $seed --lr-schedule cosine --epochs 150 ;;
      depth)
        submit diff10   $model $seed --w-diff 10
        submit diff100  $model $seed --w-diff 100 ;;
      ladder)
        for l in levels increments; do submit $l $model $seed --ladder $l; done ;;
      combined)
        BEST="--return-transform quantile --ladder increments"
        submit q_inc $model $seed $BEST
        if [[ $model == timegan ]]; then   # TimeGAN-only options
            ALL="$BEST --feature-adv --w-diff 10"
            submit all       $model $seed $ALL
            submit all_step5 $model $seed $ALL --step 5
            submit all_step05 $model $seed $ALL --step 0.5
        fi ;;
      improve)
        BEST="--return-transform quantile --ladder increments"
        ALL="$BEST --feature-adv --w-diff 10"
        if [[ $model == timegan ]]; then
            submit len120      $model $seed $ALL --seq-len 120
            submit gate24      $model $seed $ALL --hold-gate
            submit gate120     $model $seed $ALL --hold-gate --seq-len 120
        else
            submit len120      $model $seed $BEST --seq-len 120
        fi ;;
      gatesup)   # hold gate trained to open only where real features change
        ALL="--return-transform quantile --ladder increments --feature-adv --w-diff 10 --hold-gate"
        if [[ $model == timegan ]]; then
            submit gate24 $model $seed $ALL
            submit gate60 $model $seed $ALL --seq-len 60
        fi ;;
      *) echo "unknown group $GROUP"; exit 1 ;;
    esac
  done
  if [ -z "$DRYRUN" ]; then
      sbatch --parsable ${SBATCH_TIME:+--time=$SBATCH_TIME} --job-name=${GROUP:0:4}_s$seed slurm_pack.sh "$CMDFILE"
  fi
done
