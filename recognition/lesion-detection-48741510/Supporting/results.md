**BASELINE YOLOv8n**
Split: test, imgsz: 640, device: cpu
No detection: 0/600

=== Localisation (top-confidence box vs mask-derived box) ===
Mean IoU, all images (misses = 0): 0.703
Mean IoU, detected only (600 images): 0.703
Images with IoU > 0.5: 85.0%
Images with IoU >= 0.8: 41.2%
Target (mean IoU >= 0.8) met: False
Mean IoU, best detection per image: 0.775
Split evaluated: test

=== Per-class localisation (grouped by TRUE class) ===
melanoma: mean IoU 0.655, IoU > 0.5: 80.3%, IoU >= 0.8: 35.9%, n=117
seborrheic keratosis: mean IoU 0.615, IoU > 0.5: 68.9%, IoU >= 0.8: 33.3%, n=90
nevus: mean IoU 0.737, IoU > 0.5: 90.1%, IoU >= 0.8: 44.5%, n=393
Ultralytics 8.4.190  Python-3.14.6 torch-2.14.1+cpu CPU (13th Gen Intel Core i7-13620H)
Model summary (fused): 73 layers, 3,133,193 parameters, 0 gradients, 8.6 GFLOPs
val: Fast image access  (ping: 0.0±0.0 ms, read: 1802.2±502.8 MB/s, size: 11061.4 KB)
val: Scanning C:\Users\zakts\CODE\PatternAnalysis-2026\recognition\lesion-detection-48741510\data\…: 100% ━━━━━━━━━━━━ 600/600 15.9Mit/s 0.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 38/38 4.5s/it 2:50
                   all        600        600      0.477      0.614      0.546      0.288
Speed: 2.1ms preprocess, 90.1ms inference, 0.0ms loss, 0.5ms postprocess per image
Results saved to C:\Users\zakts\CODE\PatternAnalysis-2026\recognition\lesion-detection-48741510\outputs\test_final\yolo_val

=== Detection metrics (Ultralytics val, test split) ===
Overall: mAP50 0.546, mAP50-95 0.288, P 0.477, R 0.614
AP at IoU 0.8 (mean over classes): 0.173
melanoma: P 0.376, R 0.462, mAP50 0.354, mAP50-95 0.156
seborrheic keratosis: P 0.353, R 0.567, mAP50 0.473, mAP50-95 0.268
nevus: P 0.701, R 0.813, mAP50 0.811, mAP50-95 0.442

=== Classification (top-confidence box per image) ===
Confusion matrix (rows = true, last column = no detection):
 [[ 55  15  47   0]
 [  4  62  24   0]
 [ 34  28 331   0]]
melanoma: precision 0.591, recall 0.470, F1 0.524
seborrheic keratosis: precision 0.590, recall 0.689, F1 0.636
nevus: precision 0.823, recall 0.842, F1 0.833
Overall accuracy: 0.747
Melanoma vs rest: sensitivity 0.470, specificity 0.921, AUC 0.764

**YOLOv8, IMGSZ = 1024**
No detection: 0/600

=== Localisation (top-confidence box vs mask-derived box) ===
Mean IoU, all images (misses = 0): 0.755
Mean IoU, detected only (600 images): 0.755
Images with IoU > 0.5: 89.7%
Images with IoU >= 0.8: 52.5%
Target (mean IoU >= 0.8) met: False
Mean IoU, best detection per image: 0.803
Split evaluated: test

=== Per-class localisation (grouped by TRUE class) ===
melanoma: mean IoU 0.721, IoU > 0.5: 88.0%, IoU >= 0.8: 49.6%, n=117
seborrheic keratosis: mean IoU 0.705, IoU > 0.5: 82.2%, IoU >= 0.8: 44.4%, n=90
nevus: mean IoU 0.776, IoU > 0.5: 91.9%, IoU >= 0.8: 55.2%, n=393
Ultralytics 8.4.190  Python-3.14.6 torch-2.14.1+cpu CPU (13th Gen Intel Core i7-13620H)
Model summary (fused): 73 layers, 3,133,193 parameters, 0 gradients, 8.6 GFLOPs
val: Fast image access  (ping: 0.0±0.0 ms, read: 1841.4±409.6 MB/s, size: 11532.7 KB)
val: Scanning C:\Users\zakts\CODE\PatternAnalysis-2026\recognition\lesion-detection-48741510\data\…: 100% ━━━━━━━━━━━━ 600/600 83.9Mit/s 0.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 38/38 4.5s/it 2:53
                   all        600        600      0.614      0.613      0.633      0.356
Speed: 2.0ms preprocess, 77.4ms inference, 0.0ms loss, 0.5ms postprocess per image
Results saved to C:\Users\zakts\CODE\PatternAnalysis-2026\recognition\lesion-detection-48741510\outputs\yolo_val

**YOLOv8: IMGSZ = 1024, box = 12, dfl = 2.0, cls = 0.3**
Split: test, imgsz: 1024, device: cpu
No detection: 0/600

=== Localisation (top-confidence box vs mask-derived box) ===
Mean IoU, all images (misses = 0): 0.753
Mean IoU, detected only (600 images): 0.753
Images with IoU > 0.5: 89.3%
Images with IoU >= 0.8: 54.0%
Target (mean IoU >= 0.8) met: False
Mean IoU, best detection per image: 0.805
Split evaluated: test

=== Per-class localisation (grouped by TRUE class) ===
melanoma: mean IoU 0.714, IoU > 0.5: 85.5%, IoU >= 0.8: 47.0%, n=117
seborrheic keratosis: mean IoU 0.694, IoU > 0.5: 78.9%, IoU >= 0.8: 51.1%, n=90
nevus: mean IoU 0.777, IoU > 0.5: 92.9%, IoU >= 0.8: 56.7%, n=393
Ultralytics 8.4.190  Python-3.14.6 torch-2.14.1+cpu CPU (13th Gen Intel Core i7-13620H)
Model summary (fused): 73 layers, 3,133,193 parameters, 0 gradients, 8.6 GFLOPs
val: Fast image access  (ping: 0.0±0.0 ms, read: 2422.5±547.4 MB/s, size: 14280.3 KB)
val: Scanning C:\Users\zakts\CODE\PatternAnalysis-2026\recognition\lesion-detection-48741510\data\…: 100% ━━━━━━━━━━━━ 600/600 64.5Mit/s 0.0s
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 38/38 3.7s/it 2:20
                   all        600        600      0.551      0.547      0.564      0.331
Speed: 1.7ms preprocess, 68.4ms inference, 0.0ms loss, 0.3ms postprocess per image
Results saved to C:\Users\zakts\CODE\PatternAnalysis-2026\recognition\lesion-detection-48741510\outputs\test_final\yolo_val

=== Detection metrics (Ultralytics val, test split) ===
Overall: mAP50 0.564, mAP50-95 0.331, P 0.551, R 0.547
AP at IoU 0.8 (mean over classes): 0.249
melanoma: P 0.439, R 0.354, mAP50 0.372, mAP50-95 0.188
seborrheic keratosis: P 0.434, R 0.522, mAP50 0.480, mAP50-95 0.302
nevus: P 0.780, R 0.766, mAP50 0.840, mAP50-95 0.503

=== Classification (top-confidence box per image) ===
Confusion matrix (rows = true, last column = no detection):
 [[ 41  24  52   0]
 [  6  61  23   0]
 [ 33  34 326   0]]
melanoma: precision 0.512, recall 0.350, F1 0.416
seborrheic keratosis: precision 0.513, recall 0.678, F1 0.584
nevus: precision 0.813, recall 0.830, F1 0.821
Overall accuracy: 0.713
Melanoma vs rest: sensitivity 0.350, specificity 0.919, AUC 0.735

Examples saved to C:\Users\zakts\CODE\PatternAnalysis-2026\recognition\lesion-detection-48741510\outputs\test_final/examples
