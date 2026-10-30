# Final test comparison

Evidence: original school-side JSONs in final_results_backup.zip have now been imported and checked. Both best-weight SHA-256 values match the training summaries and final evaluations, and the current central model/train/evaluate/predict code matches the executed snapshots. Consolidated verified values are in `verified_results.json`; `test_results_reported.json` preserves the earlier terminal transcript parsing.

## Dataset and fixed protocol

The same 595-image decontaminated subset of the official ISIC 2017 test set was used for both models. Five of the original 600 test images were excluded before performance scoring: two duplicates of training images, one duplicate of a validation image, and two duplicates within the test set. Remaining classes: melanoma 117, seborrheic keratosis 90, nevus 388. This is not the untouched official 600-image benchmark.

Both models were trained for 50 epochs from COCO-pretrained YOLOv8n at their respective resolutions, using the same 1,999-image training and 150-image validation inventories and configured seed. The confidence of 0.15 was fixed after exploratory validation analysis, before test evaluation. No threshold sweep was run on test data. No patient mapping or near-duplicate guarantee is claimed. Experiment configuration files remain the authority for exact training details.

Environment: NVIDIA A10G (22590 MiB reported by CUDA), PyTorch 2.14.0+cu130, Ultralytics 8.4.191, Python 3.10.21.

## AP results

| Class / metric | 640 baseline | 960 comparison |
|---|---:|---:|
| Overall mAP50 | **0.6257** | 0.5675 |
| Overall mAP50:95 | **0.3721** | 0.3321 |
| Melanoma AP50 | **0.4621** | 0.4348 |
| Melanoma AP50:95 | **0.2478** | 0.2221 |
| Seborrheic keratosis AP50 | **0.5713** | 0.4353 |
| Seborrheic keratosis AP50:95 | **0.3480** | 0.2762 |
| Nevus AP50 | **0.8437** | 0.8326 |
| Nevus AP50:95 | **0.5205** | 0.4981 |

The higher-resolution model's small validation AP50 gain did not persist on this test subset. Relative to 640, 960 lost 0.0582 absolute mAP50 (5.82 percentage points) and 0.0399 mAP50:95 (3.99 points). These are single-seed results without confidence intervals; they do not prove that higher resolution is generally worse.

## Operating results: fixed confidence 0.15

The following are from the saved single-class-per-candidate inference pipeline and class-aware one-to-one matching, not the validator's independently selected P/R summary point. Precision and recall must not be mixed between those protocols. AP remains the framework's metric.

| IoU >= 0.5 | 640 | 960 |
|---|---:|---:|
| Overall TP / FP / FN | 448 / 306 / 147 | 409 / 332 / 186 |
| Overall precision (micro) | 0.5942 | 0.5520 |
| Overall recall (micro) | 0.7529 | 0.6874 |
| All-class false positives per image | 0.5143 | 0.5580 |
| Melanoma TP / FP / FN | 57 / 72 / 60 | 61 / 102 / 56 |
| Melanoma precision | 0.4419 | 0.3742 |
| Melanoma recall | 0.4872 | 0.5214 |

At this operating point, 960 detected four additional annotated melanoma lesions but produced 30 additional melanoma-labelled false positives. These aggregate differences do not imply that the two models' true positives were nested; individual paired predictions would be needed to establish which cases changed. Recall is reported as a measured outcome, without an additional numerical pass criterion absent from the assignment.

| IoU >= 0.8 | 640 | 960 |
|---|---:|---:|
| Overall TP / FP / FN | 285 / 469 / 310 | 234 / 507 / 361 |
| Overall precision (micro) | 0.3780 | 0.3158 |
| Overall recall (micro) | 0.4790 | 0.3933 |
| Melanoma TP / FP / FN | 33 / 96 / 84 | 30 / 133 / 87 |
| Melanoma recall | 0.2821 | 0.2564 |

Here recall means the fraction of ground-truth lesions matched with both the correct class and sufficient IoU; it is not average IoU or classification accuracy. Increasing the match threshold changes which existing boxes count as matches, not the number of boxes emitted. These results do not establish that a separate average-IoU >= 0.8 requirement is met. If the assignment expects that aggregation, compute it separately with matching and missed-detection handling explicitly defined.

## Compute and trade-off

Reported training time: 1.006 hours for 640 and 1.170 hours for 960, about 16.3% more. The final validator reported inference components of 1.280 and 2.317 ms/image respectively. These are framework batched evaluation timings, not robust single-image end-to-end deployment FPS. Preprocessing, NMS, disk IO, batching, warm-up and synchronization must be stated when reporting speed. Peak VRAM and final parameter information should be extracted from the actual training JSONs; smoke-run GPU memory is not a substitute for the 50-epoch run measurements.

## Engineering conclusion

Within this experiment, the 640 baseline provides the stronger overall detection/localization results with lower training cost. The 960 alternative offers a small melanoma recall increase at the fixed operating point, but a lower melanoma precision, more false positives, and no strict-localization advantage. The evidence does not justify claiming the higher resolution is an overall improvement.

Neither configuration supports a reliable automated melanoma exclusion claim. Report the assignment's localization target with explicit denominators and the five validation error cases rather than further tuning on test performance. Any future changes (class balancing, different architecture, calibration or postprocessing) require a new development/evaluation protocol rather than reuse of this test set as a tuning set.
