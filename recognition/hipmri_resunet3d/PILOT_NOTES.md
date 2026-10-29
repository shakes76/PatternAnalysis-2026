# Exploratory native-HipMRI CPU pilot

**This is a software and resource check, not the final experiment.** It used one training epoch, a `32×64×64` grid, four base channels per network, and CPU-only PyTorch. Native label IDs 5/3/4 for prostate/bladder/rectum were initially inferred from anatomy and subsequently confirmed by the course tutor. The patient-disjoint split contained 154 train, 34 validation, and 23 test scan visits. The downloaded archive contains 211 paired visits from 38 patients, and all pairs passed the image/mask alignment audit.

The first pilot ran before `--validation-only` was added, so it opened the test split. Its test results and figures are **exploratory**; they must not be presented as an untouched final benchmark. The test set was not used to select its checkpoints. Later resource and hyperparameter runs use validation only.

| Resampled-grid pilot measure | 2D U-Net | 3D Improved U-Net |
| --- | ---: | ---: |
| Validation mean prostate Dice, by scan | 0.129 | 0.000 |
| Exploratory test mean prostate Dice, by scan | 0.207 | 0.000 |
| Test mean apex Dice, by scan | 0.508 | 0.000 |
| Test mean base Dice, by scan | 0.563 | 0.000 |

The 3D pilot model produced no visible prostate contour on the displayed apex, middle, or base slices in the five selected test visits; the blue error overlays show missed expert-mask voxels. The 2D baseline produced visible contours but tended to exceed the small expert apex region. The following observations describe only the displayed resampled slices:

| Visit | Observed 2D error on displayed slices | Observed 3D error |
| --- | --- | --- |
| Case_005_Week0 | Apex contour is too broad; middle contour is larger and shifted; base is fairly close. | Misses the displayed gland at apex, middle, and base. |
| Case_019_Week0 | Exceeds the tiny apex region; tracks the middle more closely; extends beyond the small base region. | Misses the displayed gland at all three levels. |
| Case_022_Week0 | Exceeds the apex; is larger and shifted at middle; extends below and to the side of the base. | Misses the displayed gland at all three levels. |
| Case_041_Week0 | Over-contours the small apex and is broader than the expert contour at middle and base. | Misses the displayed gland at all three levels. |
| Case_042_Week0 | Greatly over-contours the narrow apex; is closer at middle; broadens the base. | Misses the displayed gland at all three levels. |

These observations do not identify a clinical or image-level cause. A full run needs native-grid and patient-balanced scores, a preselected evaluation protocol, and a new specialist review of the final model's cases. The reused test split should be reported as an internal comparison because this pilot already exposed it. CPU runs cannot supply the target-GPU peak-memory measurement.
