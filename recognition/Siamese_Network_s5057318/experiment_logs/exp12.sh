   #!/bin/bash
   #SBATCH --job-name=exp12
   #SBATCH --partition=comp3710
   #SBATCH --account=comp3710
   #SBATCH --gres=gpu:1
   #SBATCH --cpus-per-task=4
   #SBATCH --time=00:20:00
   #SBATCH --output=exp12_%j.out
   #SBATCH --error=exp12_%j.err
   source $HOME/miniconda3/bin/activate torch
   DATA=$HOME/data/isic2020
   cd $HOME/PatternAnalysis-2026
   git fetch origin -q
   rm -rf $HOME/chk; mkdir -p $HOME/chk/old $HOME/chk/old2 $HOME/chk/new
   git archive origin/topic-recognition recognition/Siamese_Network_s5057318 | tar -x -C $HOME/chk/old
   git archive origin/topic-recognition recognition/Siamese_Network_s5057318 | tar -x -C $HOME/chk/old2
   git archive origin/train-metrics recognition/Siamese_Network_s5057318 | tar -x -C $HOME/chk/new
   for v in old old2 new; do for m in baseline siamese; do
     python $HOME/chk/$v/recognition/Siamese_Network_s5057318/train.py $DATA --model $m --debug --out_dir $HOME/chk/$v/out > $HOME/chk/${v}_$m.txt 2>&1
     grep -E "^epoch|^test loss" $HOME/chk/${v}_$m.txt | cut -d'|' -f1-3 > $HOME/chk/${v}_${m}_loss.txt
   done; done
   for m in baseline siamese; do
     diff -q $HOME/chk/old_${m}_loss.txt $HOME/chk/old2_${m}_loss.txt > /dev/null && echo "$m: old run repeats itself" || echo "$m: old run does NOT repeat itself (test not valid)"
     diff -q $HOME/chk/old_${m}_loss.txt $HOME/chk/new_${m}_loss.txt > /dev/null && echo "$m: new = old (identical losses)" || echo "$m: new DIFFERENT from old"
   done > $HOME/chk/result.txt