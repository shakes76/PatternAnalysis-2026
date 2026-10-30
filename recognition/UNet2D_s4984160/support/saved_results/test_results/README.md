# Completed 2D test and paired 3D comparison

Frozen evaluation job **716656** finished at 04:00:05 AEST on 2026-10-30.
Both completion markers were present, all five figures were saved, and the
displayed error log was empty. The accounting query returned no row; success
is evidenced by the output files and completion markers, not an accounting
state inferred from an empty query.

Epoch 5 was selected using validation mean prostate Dice 0.8341295702.
The held-out test set comprises **27 scans / six patients / 1620 slices**.
No weights were updated and no postprocessing or threshold tuning was applied.

## Native 2D task: all 60 original valid slices per scan

| Prostate metric | Result |
| --- | ---: |
| Mean scan Dice ± SD | **0.825359 ± 0.063976** |
| Mean patient Dice | 0.790862 |
| Mean scan IoU | 0.707511 |
| Pooled precision / recall | 0.844047 / 0.814109 |
| Pooled predicted/GT volume ratio | 0.964530 |
| Mean native-band HD95 | 4.9023 mm |
| Scans below Dice 0.75 | 4 / 27 |

The mean meets the 0.75 target, not every scan. Native HD95 uses original 3D
affine spacing on the supplied partial band; it is not a full-volume measure.
Do not compare native voxel counts or this native-resolution score directly
with downsampled full-volume 3D results.

## Shared-grid comparison: D64 × H128 × W30

| Prostate metric | Standard 2D | Standard 3D |
| --- | ---: | ---: |
| Mean scan Dice | **0.823198** | 0.773216 |
| Mean patient Dice | 0.787652 | 0.767818 |
| Mean shared HD95 (approx. mm) | 5.1784 | 6.4995 |
| Pooled precision | 0.842072 | 0.714726 |
| Pooled recall | 0.809284 | **0.841597** |
| Pooled predicted/GT volume ratio | 0.961062 | 1.177509 |
| Prostate FP in GT bladder | 110 | **89** |
| Prostate FP in GT rectum | 400 | **21** |

All 27 GT grids matched exactly. Re-inferred 3D TP/FP/FN reproduced the original
full/shared counts for every class and scan. Shared distances and organ-confusion
counts were independently measured in this job. One 2D prediction (H007 Week0)
touches the shared-band W boundary; GT and 3D predictions do not. ROI clipping
can affect surfaces and distances and is disclosed in `paired_cases.csv`.

2D had higher Dice in **21/27 scans**. The mean difference is +4.9982 percentage
points per scan, but only +1.9834 points after equal patient weighting; 4/6
patients favour 2D. These are descriptive comparisons with repeated weeks,
not a significance test on 27 independent patients. Models differ in capacity
(2D: 1,942,374 versus 3D: 351,206 parameters), normalization, resolution,
sampling, training and inference precision. These results do not isolate the
causal effect of 2D versus 3D convolution.

## Five fixed paired cases

Columns are MRI, GT, 3D prediction, 2D prediction, and 2D prostate TP/FP/FN.
Green/red/blue are TP/FP/FN; cyan/orange/magenta are bladder/rectum/prostate.
Gray shading is outside the evaluated band, not predicted background. Extent
rows are GT-based proxies, not independently annotated clinical landmarks.
Case identities and view positions were fixed from the earlier 3D review.

### B037 Week0: improved average overlap, but superior omission remains

![B037 paired predictions](figures/B037_Week0_LFOV.png)

Shared Dice rises from 0.6413 to 0.7195 and HD95 falls from 14.65 to 12.49 mm.
The 2D middle-plane core has green overlap with visible blue missed edges;
the superior D=39 GT focus remains blue/missed. Volume ratios are 0.6481 for
2D and 0.6010 for 3D: both undersegment. A better average has not repaired this
specific extent failure.

### L011 Week2: less excess prediction, residual superior error

![L011 paired predictions](figures/L011_Week2_LFOV.png)

Shared Dice improves from 0.7088 to 0.8458. The middle-plane 2D contour is less
expanded than 3D, consistent with the volume ratio falling from 1.6787 to
1.1880. HD95 improves from 8.13 to 4.71 mm; bladder invasion falls from 32 to
3 voxels. The small superior D=44 focus is still missed, so improvement is
not uniform across boundaries.

### H017 Week3: better precision with remaining undercoverage

![H017 paired predictions](figures/H017_Week3_LFOV.png)

Shared Dice improves from 0.7215 to 0.8158 and HD95 from 6.72 to 3.36 mm.
The 2D prediction is smaller overall (volume ratio 0.8130), but its precision
is higher (0.9096 versus 0.7246). Core overlap improves while blue errors
remain at small extents. Bladder invasion is zero for 2D versus ten for 3D;
rectal invasion is eight versus zero. Volume agreement alone does not rank
the models, and improved overlap does not remove every organ-confusion error.

### K042 Week0: 3D retains a small Dice advantage

![K042 paired predictions](figures/K042_Week0_LFOV.png)

Shared Dice is 0.8378 for 2D versus 0.8525 for 3D. 2D has lower recall
(0.7607 versus 0.8106) and a smaller volume ratio (0.8158 versus 0.9017),
consistent with undercoverage despite a mostly matching core. Its HD95 is
slightly lower (5.68 versus 6.26 mm), illustrating that Dice and HD95 need
not produce the same ranking.

### H007 Week0: important 2D failure and rectal confusion

![H007 paired predictions](figures/H007_Week0_LFOV.png)

Shared Dice falls from 0.7888 for 3D to 0.6686 for 2D; HD95 worsens from 4.47
to 9.06 mm. The middle-plane 2D error map has red extension toward the GT
rectal region, while blue extent misses persist. Whole shared-volume counts
confirm **191 rectal-invasion voxels for 2D versus five for 3D**; these counts
are not limited to the displayed slice. This one case supplies 47.75% of all
2D rectal-invasion voxels. It also touches the shared ROI boundary. This
failure is retained rather than hidden by the better 2D average.

## Interpretation and next step

Both standard models satisfy their chosen mean-Dice targets. The 2D pipeline
has better average overlap/precision/HD95, while 3D has higher pooled recall
and markedly less rectal intrusion. Neither result establishes clinical
acceptability without review. The examples do not demonstrate a specific
causal imaging trigger, because no controlled resolution/capacity/context
ablation was performed. Preserve both frozen models and report this trade-off;
do not tune against these test cases to make either model win.

This paired evaluation kept both models co-resident and did not itself profile
isolated latency/memory. The subsequent [resource job 717015](../../../../UNet3D_s4984160/support/saved_results/resource_results/README.md)
now supplies separate-model measurements on the same GPU, without changing
these segmentation scores. The final course report still needs report assembly,
feasibility review, references and full AI-use disclosure.

## Files and provenance

- `summary.json`: original three-scope metrics and checkpoint/source hashes.
- `native_2d_per_case_metrics.csv`, `shared_2d_per_case_metrics.csv`,
  `reference_3d_shared_metrics.csv`: original six-class scan rows.
- `paired_cases.csv`: all 27 paired cases and geometry/contact checks.
- `evaluation_config.json`, `patient_split.json`: exact evaluation settings.
- `figures/manifest.json`: fixed figure identities and saved filenames.
- `evaluation_complete.txt`, `figures/plots_complete.txt`: stage completion.
- [Training history, configuration and curve](../training_results/README.md)
  are archived separately from test evaluation.

2D checkpoint SHA256:
`3a666e057827897d294d9df9604ac325617fa874bf541402011537847024174b`.
3D checkpoint SHA256:
`77ecc9f051d5d1213c2e62474d582050fe5b425a08182e3d2becc73540737156`.
Raw MRI, checkpoint weights and private inference arrays are excluded from Git.
