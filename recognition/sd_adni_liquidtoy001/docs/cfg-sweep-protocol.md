# CFG sweep — 7 October 2026

Question: can intermediate guidance reduce residual color/background artifacts
without losing useful structure? This is sampling only; no training or test data.

Hold the step20000 checkpoint, prompts,256 resolution, DDIM30, eta0, aligned
float32 scheduler-input boundary, and seeds37200-37231 fixed. Generate both AD/NC
for CFG3,5,7.5:192 images total. Retain every output, including failure seeds.

Regenerate7.5 in the same job to avoid assuming an unchanged execution environment.
After completion compare all64 output hashes to cohort632557; report mismatches
instead of silently treating historical and new images as identical. Verify
initial-noise hashes across all guidance levels and complete factorial coverage.

Compute appearance-v1 color, border and edge diagnostics, paired changes, and
per-label summaries. AD/NC share seed noise:32 seed blocks, not64 independent
replicates. Do not treat edge as anatomical quality or optimize toward a single
real-reference mean. Inspect complete grids and known failures. Choosing CFG on
these diagnostic seeds is exploratory; later validation needs fresh seeds.

No optimal setting, clinical validity, privacy safety, or retraining need is
assumed. Job completion alone is not result acceptance. A single20-minute GPU
job is requested; prior throughput suggests roughly9 minutes of execution,
excluding queue time. Keep patient data and generated artifacts private.
