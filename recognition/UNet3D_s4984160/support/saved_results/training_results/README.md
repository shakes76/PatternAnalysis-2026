# 3D training curves from the recorded histories

![Recorded 3D training objective and validation prostate Dice](training_curves.png)

The figure combines 15 initial epochs and 11 refinement epochs. It was drawn
from the original CSV files without training or model inference. The selected
checkpoint is refinement epoch 11 / cumulative epoch 26, with validation mean
prostate Dice **0.7559718763**.

Training objective values are separated by phase because class weights and
optimizer settings changed. Validation used the same patient split and resized
grid. The initial 15 epochs had zero prostate Dice even though overall training
loss fell; this is why lower training loss alone did not establish useful
prostate segmentation. Later improvement cannot be assigned to only one
training change without an ablation.

`curve_data.csv` records the plotted points; `scratch_history.csv` and
`refine_history.csv` preserve their sources; `curve_metadata.json` records
source hashes and the selected epoch. The copied refinement CSV matches the
original test archive's training history byte-for-byte. No validation-loss
series was logged, so no validation-loss curve is fabricated.
