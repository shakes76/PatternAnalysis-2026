# Assessment requirement check

This check compares the HipMRI project with the supplied **draft** COMP3710/COMP7710 brief. It records evidence available in this repository, not an awarded grade.

| Brief requirement | Evidence | Status |
| --- | --- | --- |
| Approved project and difficulty | HipMRI medical image analysis; hard 3D Improved U-Net | Implemented |
| Baseline and comparison | Full-field 2D U-Net shares preprocessing with 3D; a stronger cropped 2D checkpoint was independently selected by validation Dice. All three use the same patient split and native test masks | Measured; 3D benefit is mixed |
| Modular `modules.py`, `dataset.py`, `train.py`, `predict.py` | All files present; `modules.py` uses PyTorch and standard-library `math` only | Checked |
| Patient-level train/validation/test split | 27/6/5 disjoint patients; 154/34/23 visits; saved `split.json` | Checked |
| Nifti alignment, normalization, labels | All 211 native image/mask pairs passed shape, affine, and integer-label audit; course tutor confirmed IDs 5/3/4 | Checked |
| Training augmentation | Aligned left-right flip option in `dataset.py`/`train.py` | Implemented; off in measured runs |
| Dice target ≥0.70 and per-class Dice/IoU | Selected 3D internal test: resampled patient mean 0.757 and scan mean 0.737; native patient mean 0.694 versus 0.698 for the independently selected cropped 2D model. Background/prostate Dice and IoU recorded | Brief's downsampled 3D target met; native Dice advantage over tuned 2D not shown |
| Boundary and edge-slice analysis | Native patient means: 3D apex/base Dice 0.348/0.636 versus tuned 2D 0.388/0.645; HD95 8.17 versus 9.11 mm | Measured; edge-slice overlap does not favor 3D |
| Bladder/rectum leakage | Native patient mean false-positive fractions: 3D bladder/rectum 0.97%/0.261%; tuned 2D 1.16%/0.221% | Measured; mixed organ results |
| Parameters, memory, inference latency | 3D: 93,358 parameters and 74.0 ms model-only CPU inference; tuned 2D: 30,782 parameters and 52.1 ms | Target-GPU memory and latency unmeasured |
| Three to five failure autopsies | Five distinct-patient full-field 2D/3D panels, with observed false positives/negatives and written case reviews; panels do not display tuned cropped 2D | Completed for common-configuration pair; no specialist adjudication |
| One-page feasibility review | Dedicated README section with need, goals, risks, budget, and fallback | Completed |
| README report, figures, versions, references, AI disclosure | Completed README, paired-patient Dice figures, five error panels, code map, exact CPU configuration, and measured tables | Present; verify against final submission |
| Progressive commit history | Local history contains milestone commits | Assess progression and authorship from the actual Git history |
| Midpoint tutor consultation | Brief calls for a 3–5 minute review/check-off | Attendance is not evidenced in this repository |
| Pull request to upstream `topic-recognition` | No course pull request is evidenced in this checkout | Complete through the course workflow |
| Turnitin PDF | A rendered PDF of the completed README is supplied separately | Verify final PDF submission |
| Respond to later code review | Depends on teaching-team review after submission | Future event |

## Local verification

- The automated suite has **24 passing tests** covering split disjointness, singleton 4D Nifti support, alignment, tensor shapes, gradient flow, native-grid metrics, suffix-aware pairing, patient-balanced aggregation, opt-in/one-time test gates, case-selection diversity, matching splits for the independent baseline, and checked cross-run case pairing. Synthetic scores are never presented as HipMRI performance.
- The native audit passed for 211 pairs from 38 patients. A train/validation-only crop check confirmed all 188 prostates lie within a central 128×128 crop; the crop was selected for the 2D comparator, while the 3D model retained the full field because its validation Dice was higher there.
- Four candidates were compared using validation data only. The selected full-field weight-20 run reached 3D validation Dice 0.684; the independently selected cropped 2D model reached validation Dice 0.708. One recorded test evaluation of each selected checkpoint on the same 23 visits gave native patient-balanced Dice **0.694 for 3D and 0.698 for tuned 2D**. The paired five-patient 3D-minus-2D bootstrap interval was **-0.075 to +0.077**, so a 3D Dice advantage is not established.
- All five common-configuration case panels were inspected. They include a patient for whom the full-field 2D model outperformed 3D and apex omissions despite good whole-volume overlap. They are not visual evidence for the cropped 2D comparator.
- The **earlier one-epoch pilot exposed this same test split** before the validation-only gate existed. All subsequent scores on it are internal exploratory comparisons. A genuinely blind estimate requires a separate independent cohort.

## Remaining course or scientific gates

1. For a full-resolution claim, run a separately selected higher-resolution experiment on a suitable GPU and measure target-GPU memory and latency. The reported CPU result belongs to a coarse grid and small model.
2. For generalization and clinical usefulness, evaluate an unseen patient cohort, obtain contour specialist adjudication, and assess native-space dosimetric impact. The present evidence does not support automatic clinical use.
3. Complete the tutor check-off, document individual implementation and history accurately, submit the course pull request and Turnitin PDF, and respond to later review in the required timeline.
