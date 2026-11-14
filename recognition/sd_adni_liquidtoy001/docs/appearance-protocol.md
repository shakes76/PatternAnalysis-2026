# Appearance diagnostic v1 — 7 October 2026

Freeze this descriptive protocol before opening real-reference results. It extends
the exploratory 192-image local check; it is not a preregistration of that earlier
check. Read validation only: one deterministic middle sorted slice per patient,
using the existing patient_selection routine (102 patients, 33 AD / 69 NC).
Generated cohorts retain all 64 samples per model. Report AD/NC separately because
the generated and real groups have different label proportions. Slice position
and reference sample size remain confounders; no significance claim is intended.

- Color: mean per-pixel max(R,G,B)-min(R,G,B) at native resolution, scaled [0,1].
  Real grayscale and the single-channel VAE have zero by construction. This cannot
  establish anatomical quality or attribute SD color artifacts to a specific cause.
- Border: mean grayscale in the outer eight pixels after bilinear resize to128.
  Real images are centered and black-padded to square first, matching training.
  This is a fixed spatial proxy, not foreground/background segmentation. Padding
  affects it; retained full images and case panels are necessary context.
- Edge: mean absolute horizontal and vertical neighboring-pixel difference at128.
  Lower values can reflect smoothing; higher values can reflect noise or artifacts.
  It is not a quality ranking or anatomical metric. Native resolutions still differ.

Summarize mean, median, range and 5th/95th percentiles. Verify every input SHA256.
Keep per-image records private. No test image is read. No invented pass thresholds,
no automatic retraining decision, and no inference of privacy safety.

Run from the project root:

```sh
.venv-train/bin/python tools/audit_appearance.py --real --output outputs/appearance-real-20261007
```

Local generated-only execution omits --real and explicitly records the absence of
real-reference evidence. This stage does not deliver a foreground mask, learned
feature evaluation, or a causal ablation; those remain separate follow-up work.

On Rangpur, supply --baseline ../runs/baseline128-20261005/outputs/baseline-vae-632570 to use the frozen baseline run.
