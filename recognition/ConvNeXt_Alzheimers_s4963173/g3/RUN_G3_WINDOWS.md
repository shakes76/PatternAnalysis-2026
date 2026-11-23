# Run G3 on Windows PowerShell 5.1

28 October 2026. **Control fixed: REUSE_G2. Run one new G3 seed42 attempt only.**
No U3 command is provided because a fresh uniform control is unnecessary and
not permitted by this frozen plan. No command launches G0/G1/G2, another seed,
calibration, test inference, promotion or a successor experiment.

Run these foreground steps in order. Each block is independent and restores all
paths/variables; do not reuse old G012 `$entry` or `$commonArgs`. No package
installation or shell-policy change is needed. Preserve the console and all
study evidence. The supplied pinned environment is required; no fresh environment
or retraining success is claimed. Codex has NOT executed steps3–11 on real data.

## 1. Verify committed source and plan (read-only)

Expected `G3_COMMITTED_IMPLEMENTATION_VERIFIED`. A mismatch stops; do not edit
frozen hashes or replace evidence to make it pass.

```powershell
$ErrorActionPreference = 'Stop'
$py = 'C:/Users/Angus/COMP3710-private/.venv/Scripts/python.exe'
$assignment = 'C:/Users/Angus/Documents/Codex/2026-10-27/independently-audit-my-comp3710-alzheimer-s/work/scan-attention-code/recognition/ConvNeXt_Alzheimers'
$entry = Join-Path $assignment 'g3/run.py'
$study = 'C:/Users/Angus/COMP3710-private/scan-attention-g3'
Set-Location -LiteralPath $assignment
$commonArgs = @(
    '--data-root', 'C:/Users/Angus/COMP3710-private/data/ADNI',
    '--splits-dir', 'C:/Users/Angus/COMP3710-private/patient-splits-20261022-155230',
    '--study-root', $study,
    '--historical-root', 'C:/Users/Angus/COMP3710-private/scan-supervision-g012',
    '--historical-source', 'C:/Users/Angus/Documents/Codex/2026-10-27/independently-audit-my-comp3710-alzheimer-s/work/scan-supervision-code/recognition/ConvNeXt_Alzheimers'
)
& $py -B $entry verify
if ($LASTEXITCODE -ne 0) { throw 'G3 verify stopped. Preserve evidence; inspect the blocker before continuing.' }
```

## 2. Run synthetic tests only

Expected 50 tests / OK (15 G3 +35 inherited G012). All tensors/rows are artificial.

```powershell
$ErrorActionPreference = 'Stop'
$py = 'C:/Users/Angus/COMP3710-private/.venv/Scripts/python.exe'
$assignment = 'C:/Users/Angus/Documents/Codex/2026-10-27/independently-audit-my-comp3710-alzheimer-s/work/scan-attention-code/recognition/ConvNeXt_Alzheimers'
$entry = Join-Path $assignment 'g3/run.py'
$study = 'C:/Users/Angus/COMP3710-private/scan-attention-g3'
Set-Location -LiteralPath $assignment
$commonArgs = @(
    '--data-root', 'C:/Users/Angus/COMP3710-private/data/ADNI',
    '--splits-dir', 'C:/Users/Angus/COMP3710-private/patient-splits-20261022-155230',
    '--study-root', $study,
    '--historical-root', 'C:/Users/Angus/COMP3710-private/scan-supervision-g012',
    '--historical-source', 'C:/Users/Angus/Documents/Codex/2026-10-27/independently-audit-my-comp3710-alzheimer-s/work/scan-supervision-code/recognition/ConvNeXt_Alzheimers'
)
& $py -B -m unittest g3.test_g3 g012.test_g012 -v
if ($LASTEXITCODE -ne 0) { throw 'Synthetic tests failed; do not proceed to MRI execution.' }
```

## 3. Prepare and verify the historical control (no MRI inference)

Creates only the NEW private study destination. Checks approved manifests
(test metadata only for hashes/grouping/disjointness), old source/receipts,
saved validation arrays and all30 schedules; constructs fresh random CPU E0.
At least12 GiB free on C: is required. A bound existing preparation is verified;
conflicting/incomplete destinations fail without replacement. Expected
`G3_PREPARED_REUSE_G2_NO_IMAGES_DECODED`, then
`G3_CONTROL_VERIFIED_REUSE_G2_NO_NEW_INFERENCE`. Existing G2 replay is reused.

```powershell
$ErrorActionPreference = 'Stop'
$py = 'C:/Users/Angus/COMP3710-private/.venv/Scripts/python.exe'
$assignment = 'C:/Users/Angus/Documents/Codex/2026-10-27/independently-audit-my-comp3710-alzheimer-s/work/scan-attention-code/recognition/ConvNeXt_Alzheimers'
$entry = Join-Path $assignment 'g3/run.py'
$study = 'C:/Users/Angus/COMP3710-private/scan-attention-g3'
Set-Location -LiteralPath $assignment
$commonArgs = @(
    '--data-root', 'C:/Users/Angus/COMP3710-private/data/ADNI',
    '--splits-dir', 'C:/Users/Angus/COMP3710-private/patient-splits-20261022-155230',
    '--study-root', $study,
    '--historical-root', 'C:/Users/Angus/COMP3710-private/scan-supervision-g012',
    '--historical-source', 'C:/Users/Angus/Documents/Codex/2026-10-27/independently-audit-my-comp3710-alzheimer-s/work/scan-supervision-code/recognition/ConvNeXt_Alzheimers'
)
& $py -B $entry prepare @commonArgs
if ($LASTEXITCODE -ne 0) { throw 'G3 prepare stopped. Preserve evidence; inspect the blocker before continuing.' }
& $py -B $entry control @commonArgs
if ($LASTEXITCODE -ne 0) { throw 'G3 control stopped. Preserve evidence; inspect the blocker before continuing.' }
```

## 4. User-run non-candidate G3 training-only smoke

**Real MRI execution starts here; this was not run by Codex.** Uses three
predetermined complete TRAINING scans, a pair and single tail (two updates).
Checks initial uniformity/normalisation, finite loss/gradients, updates of
backbone/head/attention, strict checkpoint roundtrip and available GPU headroom.
Reports peak allocated/reserved/device-free samples and its separate elapsed
time. No validation/test images or accuracy tuning. The full attempt starts
fresh; smoke weights never initialise it. Expected
`G3_SMOKE_PASSED_NON_CANDIDATE`. A failed/incomplete smoke blocks training;
preserve it and investigate rather than replacing it or switching control mode.

```powershell
$ErrorActionPreference = 'Stop'
$py = 'C:/Users/Angus/COMP3710-private/.venv/Scripts/python.exe'
$assignment = 'C:/Users/Angus/Documents/Codex/2026-10-27/independently-audit-my-comp3710-alzheimer-s/work/scan-attention-code/recognition/ConvNeXt_Alzheimers'
$entry = Join-Path $assignment 'g3/run.py'
$study = 'C:/Users/Angus/COMP3710-private/scan-attention-g3'
Set-Location -LiteralPath $assignment
$commonArgs = @(
    '--data-root', 'C:/Users/Angus/COMP3710-private/data/ADNI',
    '--splits-dir', 'C:/Users/Angus/COMP3710-private/patient-splits-20261022-155230',
    '--study-root', $study,
    '--historical-root', 'C:/Users/Angus/COMP3710-private/scan-supervision-g012',
    '--historical-source', 'C:/Users/Angus/Documents/Codex/2026-10-27/independently-audit-my-comp3710-alzheimer-s/work/scan-supervision-code/recognition/ConvNeXt_Alzheimers'
)
& $py -B $entry smoke @commonArgs --variant G3
if ($LASTEXITCODE -ne 0) { throw 'G3 smoke stopped. Preserve evidence; inspect the blocker before continuing.' }
```

## 5. Uniform control decision — no new run

Skip U3. Historical G2 selected20/completed28 is the designated uniform control,
already completed and replayed. Its recovered disk stop is not a reason to rerun
it. If a new correctness issue emerges, stop for a documented protocol revision;
do not silently create a different control.

## 6. Run the single G3 seed42 attempt

Requires bound prepare/control/smoke. One attempt ID, max30/min12/patience8 and
the frozen patient selection rule. Each complete epoch emits `COMMITTED_EPOCH`.
Normal finish emits `G3_RUN G3 COMPLETED_TRAINING`; selected replay remains
required. `ALREADY_COMPLETED_VERIFIED_SKIPPED` means it did not train again.

Cap10800 supervised active seconds INCLUDING selected replay; training reserves
300 seconds and therefore stops at cumulative10500. Repeated partial work and
crash leases remain charged across resumptions. These are limits, not estimates.
No budget extension or batch/precision change. Historical G2 time is separate.

```powershell
$ErrorActionPreference = 'Stop'
$py = 'C:/Users/Angus/COMP3710-private/.venv/Scripts/python.exe'
$assignment = 'C:/Users/Angus/Documents/Codex/2026-10-27/independently-audit-my-comp3710-alzheimer-s/work/scan-attention-code/recognition/ConvNeXt_Alzheimers'
$entry = Join-Path $assignment 'g3/run.py'
$study = 'C:/Users/Angus/COMP3710-private/scan-attention-g3'
Set-Location -LiteralPath $assignment
$commonArgs = @(
    '--data-root', 'C:/Users/Angus/COMP3710-private/data/ADNI',
    '--splits-dir', 'C:/Users/Angus/COMP3710-private/patient-splits-20261022-155230',
    '--study-root', $study,
    '--historical-root', 'C:/Users/Angus/COMP3710-private/scan-supervision-g012',
    '--historical-source', 'C:/Users/Angus/Documents/Codex/2026-10-27/independently-audit-my-comp3710-alzheimer-s/work/scan-supervision-code/recognition/ConvNeXt_Alzheimers'
)
& $py -B $entry run @commonArgs --variant G3
if ($LASTEXITCODE -ne 0) { throw 'G3 run stopped. Preserve evidence; inspect the blocker before continuing.' }
```

## 7. Check status after a foreground command exits

Reports charged seconds and terminal/recoverable status. While another process
holds the OS lock, `STUDY_ACTIVE_LOCKED` is expected; inspect append-only
`C:/Users/Angus/COMP3710-private/scan-attention-g3/G3/execution.log` instead of
opening live atomic JSON in an editor. Do not launch concurrent training.

```powershell
$ErrorActionPreference = 'Stop'
$py = 'C:/Users/Angus/COMP3710-private/.venv/Scripts/python.exe'
$assignment = 'C:/Users/Angus/Documents/Codex/2026-10-27/independently-audit-my-comp3710-alzheimer-s/work/scan-attention-code/recognition/ConvNeXt_Alzheimers'
$entry = Join-Path $assignment 'g3/run.py'
$study = 'C:/Users/Angus/COMP3710-private/scan-attention-g3'
Set-Location -LiteralPath $assignment
$commonArgs = @(
    '--data-root', 'C:/Users/Angus/COMP3710-private/data/ADNI',
    '--splits-dir', 'C:/Users/Angus/COMP3710-private/patient-splits-20261022-155230',
    '--study-root', $study,
    '--historical-root', 'C:/Users/Angus/COMP3710-private/scan-supervision-g012',
    '--historical-source', 'C:/Users/Angus/Documents/Codex/2026-10-27/independently-audit-my-comp3710-alzheimer-s/work/scan-supervision-code/recognition/ConvNeXt_Alzheimers'
)
& $py -B $entry status @commonArgs
if ($LASTEXITCODE -ne 0) { throw 'G3 status stopped. Preserve evidence; inspect the blocker before continuing.' }
```

## 8. Resume the same interrupted/disk-blocked attempt from a new terminal

Use only after an interruption or a `DISK_BLOCKED` stop has been resolved.
Free space manually to >=12 GiB; this command never deletes old evidence.
Restores the latest readable COMPLETE epoch, model/optimiser/RNG/history/best
selection and schedules; repeats any partial epoch and keeps all charged time.
No replacement attempt. `WALL_LIMIT_INCONCLUSIVE`, OOM/headroom/nonfinite/
integrity blockers require investigation and do not permit continued training.
A wall-limited incomplete run stays inconclusive; use compare/bundle to report it.

```powershell
$ErrorActionPreference = 'Stop'
$py = 'C:/Users/Angus/COMP3710-private/.venv/Scripts/python.exe'
$assignment = 'C:/Users/Angus/Documents/Codex/2026-10-27/independently-audit-my-comp3710-alzheimer-s/work/scan-attention-code/recognition/ConvNeXt_Alzheimers'
$entry = Join-Path $assignment 'g3/run.py'
$study = 'C:/Users/Angus/COMP3710-private/scan-attention-g3'
Set-Location -LiteralPath $assignment
$commonArgs = @(
    '--data-root', 'C:/Users/Angus/COMP3710-private/data/ADNI',
    '--splits-dir', 'C:/Users/Angus/COMP3710-private/patient-splits-20261022-155230',
    '--study-root', $study,
    '--historical-root', 'C:/Users/Angus/COMP3710-private/scan-supervision-g012',
    '--historical-source', 'C:/Users/Angus/Documents/Codex/2026-10-27/independently-audit-my-comp3710-alzheimer-s/work/scan-supervision-code/recognition/ConvNeXt_Alzheimers'
)
& $py -B $entry resume @commonArgs --variant G3
if ($LASTEXITCODE -ne 0) { throw 'G3 resume stopped. Preserve evidence; inspect the blocker before continuing.' }
```

## 9. Replay G3's frozen selected checkpoint on validation

Run after normal completion. Same10800-second ledger; no fresh budget. Expected
`G3_FULL_VALIDATION_REPLAY_PASSED`, then `G3_RUN G3 COMPLETED_TRAINING`.
All slice/scan/patient probability differences <=2e-5, identical classes,
attention difference <=2e-5 and unchanged slice order are required.
`ALREADY_REPLAYED_VERIFIED` reuses bound saved evidence without inference.
An interrupted replay is rerun using THIS same command and cumulative ledger,
not a new attempt or a training resume. A wall limit is terminal/inconclusive.
No G2 full-validation replay command is needed.

```powershell
$ErrorActionPreference = 'Stop'
$py = 'C:/Users/Angus/COMP3710-private/.venv/Scripts/python.exe'
$assignment = 'C:/Users/Angus/Documents/Codex/2026-10-27/independently-audit-my-comp3710-alzheimer-s/work/scan-attention-code/recognition/ConvNeXt_Alzheimers'
$entry = Join-Path $assignment 'g3/run.py'
$study = 'C:/Users/Angus/COMP3710-private/scan-attention-g3'
Set-Location -LiteralPath $assignment
$commonArgs = @(
    '--data-root', 'C:/Users/Angus/COMP3710-private/data/ADNI',
    '--splits-dir', 'C:/Users/Angus/COMP3710-private/patient-splits-20261022-155230',
    '--study-root', $study,
    '--historical-root', 'C:/Users/Angus/COMP3710-private/scan-supervision-g012',
    '--historical-source', 'C:/Users/Angus/Documents/Codex/2026-10-27/independently-audit-my-comp3710-alzheimer-s/work/scan-supervision-code/recognition/ConvNeXt_Alzheimers'
)
& $py -B $entry replay @commonArgs --variant G3
if ($LASTEXITCODE -ne 0) { throw 'G3 replay stopped. Preserve evidence; inspect the blocker before continuing.' }
```

## 10. Compare all cached evidence and export aggregate results

No image inference. Includes retained A1, historical G2 and new G3, complete
histories and fixed last-five summaries, the complete initial screen and NLL
warnings, both classes/paired changes, attention/resource/replay summaries.
Below-screen and incomplete outcomes remain in the comparison. No automatic
promotion or new experiment follows a pass. Expected
`G3_AGGREGATE_COMPARISON_EXPORTED` and a content-addressed aggregate directory.

```powershell
$ErrorActionPreference = 'Stop'
$py = 'C:/Users/Angus/COMP3710-private/.venv/Scripts/python.exe'
$assignment = 'C:/Users/Angus/Documents/Codex/2026-10-27/independently-audit-my-comp3710-alzheimer-s/work/scan-attention-code/recognition/ConvNeXt_Alzheimers'
$entry = Join-Path $assignment 'g3/run.py'
$study = 'C:/Users/Angus/COMP3710-private/scan-attention-g3'
Set-Location -LiteralPath $assignment
$commonArgs = @(
    '--data-root', 'C:/Users/Angus/COMP3710-private/data/ADNI',
    '--splits-dir', 'C:/Users/Angus/COMP3710-private/patient-splits-20261022-155230',
    '--study-root', $study,
    '--historical-root', 'C:/Users/Angus/COMP3710-private/scan-supervision-g012',
    '--historical-source', 'C:/Users/Angus/Documents/Codex/2026-10-27/independently-audit-my-comp3710-alzheimer-s/work/scan-supervision-code/recognition/ConvNeXt_Alzheimers'
)
& $py -B $entry compare @commonArgs
if ($LASTEXITCODE -ne 0) { throw 'G3 compare stopped. Preserve evidence; inspect the blocker before continuing.' }
```

## 11. Create and locate the allowlisted ZIP to return

Expected `G3_RESULTS_BUNDLE`. **Return only
`COMP3710_G3_Seed42_Results.zip` at the exact path printed below**, plus any
blocker console message if execution stopped. Full local evidence stays private.
The dynamic directory is read from the recorded export receipt; no unknown path
placeholder is used. The ZIP excludes identifiers/private paths, per-case
predictions, attention arrays, MRI, model/optimiser states and old provenance
appendices. Do not manually zip the whole private study tree.

```powershell
$ErrorActionPreference = 'Stop'
$py = 'C:/Users/Angus/COMP3710-private/.venv/Scripts/python.exe'
$assignment = 'C:/Users/Angus/Documents/Codex/2026-10-27/independently-audit-my-comp3710-alzheimer-s/work/scan-attention-code/recognition/ConvNeXt_Alzheimers'
$entry = Join-Path $assignment 'g3/run.py'
$study = 'C:/Users/Angus/COMP3710-private/scan-attention-g3'
Set-Location -LiteralPath $assignment
$commonArgs = @(
    '--data-root', 'C:/Users/Angus/COMP3710-private/data/ADNI',
    '--splits-dir', 'C:/Users/Angus/COMP3710-private/patient-splits-20261022-155230',
    '--study-root', $study,
    '--historical-root', 'C:/Users/Angus/COMP3710-private/scan-supervision-g012',
    '--historical-source', 'C:/Users/Angus/Documents/Codex/2026-10-27/independently-audit-my-comp3710-alzheimer-s/work/scan-supervision-code/recognition/ConvNeXt_Alzheimers'
)
& $py -B $entry bundle @commonArgs
if ($LASTEXITCODE -ne 0) { throw 'G3 bundle stopped. Preserve evidence; inspect the blocker before continuing.' }
$latest = Get-Content -LiteralPath (Join-Path $study 'latest_aggregate.json') -Raw | ConvertFrom-Json
$export = Join-Path (Join-Path $study 'aggregate_exports') $latest.directory
$zip = Join-Path $export 'COMP3710_G3_Seed42_Results.zip'
Get-Item -LiteralPath $zip | Select-Object FullName, Length
Get-FileHash -LiteralPath $zip -Algorithm SHA256
```

Stop here and return the ZIP for the separate analysis task. The existing A1
selection, manuscript results and unresolved0.80 test requirement remain intact.

Historical implementation inspection/replay uses the preserved
`C:/Users/Angus/Documents/Codex/2026-10-27/independently-audit-my-comp3710-alzheimer-s/work/scan-supervision-code`
worktree and its RUN_G012_WINDOWS.md. The G3 branch separately binds its optional
engine extension; do not rewrite the old G012 plan to accept new source hashes.
