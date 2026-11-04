# Optional scan-consistent input preprocessing

`--preprocessing none` (the default) converts each slice to grayscale, resizes
it to the requested size if needed and maps `uint8 / 255` to [-1, 1]. Three
optional profiles standardise brightness and head position per scan:

| Profile | Brightness | Geometry |
|---|---|---|
| `none` | fixed pixel mapping | resize if needed |
| `scan_intensity` | foreground percentile mapping shared by the scan | resize if needed |
| `scan_crop` | fixed pixel mapping | scan-consistent integer crop / black padding |
| `scan_intensity_crop` | foreground percentile mapping shared by the scan | scan-consistent integer crop / black padding |

1. Foreground is the head proxy `uint8 > 16` (not a brain mask; no pretrained
   segmentation is used).
2. Intensity profiles pool the foreground pixels of all slices of one scan,
   map its 1st-99th percentile range linearly to [0, 255] and set the
   background to 0. Each scan uses only its own pixels and no labels.
3. Crop profiles cut one fixed window, centred on the scan's foreground box,
   from every slice of the scan, without rescaling. The window size is the only
   setting fitted across cases, and only on training scans (largest foreground
   extent plus margins, rounded up to a multiple of 32). A held-out scan whose
   foreground would not fit is rejected instead of being clipped.
4. Training augmentation runs after this deterministic step; evaluation never
   augments.

Earlier fold-1 runs did not show a gain from these profiles (see
[EXPERIMENTS.md](EXPERIMENTS.md)), so the main recipe keeps `none`.

## Visual check without training

```bash
python audit_preprocessing.py --splits-dir "$HOME/comp3710/adni_splits_v1" \
    --data-root /home/groups/comp3710/ADNI --output "$HOME/comp3710/runs/preprocessing_review" \
    --fold 1 --preprocessing scan_intensity_crop --max-images 6
```

This exports original/processed previews and per-scan statistics for the
training and validation patients only (the crop is fitted on training scans); no model is trained and no held-out
patient is shown.
