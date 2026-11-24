# Recorded 2D training run 716367

This directory preserves the original training CSV, configuration, patient
manifest, summary and curve from the completed cluster job. Training stopped
after 13 epochs; epoch 5 had the highest validation mean scan prostate Dice,
**0.8341295702**, and was frozen before testing.

![Recorded loss and validation Dice](training_plot.png)

After epoch 5, training loss continued to fall while validation loss and Dice
did not improve beyond the selected value. This supports stopping rather than
assuming more epochs would help; the curve alone does not establish a unique
cause for the generalization gap. Loss is the valid-pixel-weighted average
of batch CE/soft-Dice objectives, not whole-dataset pooled Dice.

The original `training_summary.json` records `test_evaluated=false` because
training did not use held-out test data. The subsequent independent evaluation
is documented in [the test result record](../test_results/README.md).
