# Feasibility Review: Skin Lesion Detection (DRAFT)

Student: [complete] | Model: YOLOv8n | Difficulty: Normal

## User need, scope and acceptance criteria

Build a research prototype that localizes and classifies annotated dermoscopic lesions for reviewer assistance. Investigate the sensitivity versus false-alarm trade-off rather than accuracy alone. Dataset scope is limited to images with matched masks and verified diagnosis labels; this is not a validated clinical screening system.

Proposed acceptance criteria: (1) reproducible, patient-disjoint train/validation/test partitions with no exact image duplicates across partitions; (2) report mAP50, mAP50:95, precision, recall and per-class localization IoU, with the assignment's IoU >= 0.8 target (confirm its aggregation with tutor); (3) compare baseline and proposed change under identical splits and budget; (4) select confidence thresholds using validation data and report recall and false positives per image; (5) complete inference profiling and attempt a configuration within the available 4GB VRAM, with a documented fallback if it fails. Performance targets are not achieved results.

## Model choice and course concepts

Use ISIC 2017/2018 with matched image, mask and diagnostic labels. Convert the foreground-mask extent to a bounding box. Start with COCO-pretrained YOLOv8n as the standard detector baseline; evaluate a controlled resolution/augmentation change. CNN feature extraction, detection losses, transfer learning, IoU and precision-recall analysis support this approach. Confirm with tutor that the implementation and comparison depth satisfy the course expectations for pretrained detectors.

## Preliminary evidence — incomplete

Hardware audit: local NVIDIA RTX 3050 Ti Laptop GPU, 4096 MiB VRAM. School GPU Lab provided an NVIDIA L40S; PyTorch 2.14.0+cu130 and Ultralytics 8.4.191 imported successfully, and a CUDA tensor operation returned the expected result (student-provided terminal evidence). Training/validation diagnosis CSVs were audited: 2000/150 records, three classes, no overlapping image IDs. One training sample per class was matched to an official 2017 mask; dimensions and mask-to-box overlays passed inspection. The CSVs and three sampled API records lack patient identifiers, so patient independence remains unverified. Baseline training, resource profiling and full image/mask/diagnosis matching are PENDING. This draft must be updated with the agreed split policy and measured training evidence; it does not assert model performance.

## Risks, budget and fallback

Risks: incomplete patient metadata or label overlap; class imbalance; hair/marker false positives; faint/small lesions; VRAM exhaustion. Start with a small smoke test at image size 416 and batch 2, then estimate full-run time from measurements. Restrict the initial comparison to baseline plus one controlled change. Fallback: lower batch/image size, or use Rangpur if access is available. Next experiment: audit matched records and visualize boxes, then run a short baseline training job. Tutor questions: acceptable patient-disjoint split if metadata are unavailable, exact IoU aggregation, and expected depth of pretrained-model adaptation.

References: assessment Sections 2.3 and 3; https://docs.ultralytics.com/models/yolov8/ ; https://challenge.isic-archive.com/data/

Update, 2026-10-29: the student reports confirmation that official ISIC splits may be used. This resolves the proposed split protocol; it does not provide patient identifiers for independent programmatic verification. The implementation preserves official membership and checks decoded duplicates. Nine software logic tests and one synthetic CPU training/save/load/inference/evaluation integration check passed. No real ISIC model performance results are available yet.
