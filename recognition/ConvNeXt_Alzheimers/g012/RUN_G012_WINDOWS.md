# Run G0/G1/G2 seed 42 on Windows — user execution only

**REAL EXPERIMENTS HAVE NOT RUN.** Codex implemented and synthetically checked
these commands; no real-data smoke, training or new MRI replay was executed.
Run the numbered blocks yourself in Windows PowerShell 5.1 or later. Every
standalone block reconstructs its variables, quotes paths, uses the supplied
interpreter and checks native exit codes. No execution-policy change or install
is required. Do not paste the whole document into a shell; copy one code block
at a time and inspect its result. Keep this source fixed while the study runs.

The branch is `codex/scan-supervision-g012`, derived from submission commit
`585152470540e98fca97d0b96e876a9cad8014cb`. Implementation/plan commit:
`484acdf5fabc0d08608ed5be6c397480f127d95e`. Later command-only documentation does
not change the checkpoint source bindings. A1 remains selected; no automatic
adoption, test query, calibration/threshold fitting or additional seed is allowed.

Run G0, then G1, then G2, each once at seed 42. One study lock prevents concurrent
GPU runs. Repeating a completed command verifies and skips its evidence. Recovery
uses the same attempt and budget. Do not choose a new output namespace after a
failure. The original source folder named `test` is not the eligibility rule:
the approved regrouped training manifest determines training membership.

Caps are 10,800 active seconds per variant and 32,400 total, including separately
commanded selected-checkpoint replay, initialization, validation, I/O and repeated
partial work. Non-candidate smoke/preparation and downtime are excluded and
separately scoped. Five-second crash reservations are conservative, not exact
optimization-only timing. There is a 12 GiB free-disk guard. OOM/unsafe sampled
headroom (less than max(1 GiB,10% device capacity) free) is a blocker; 4.5 GiB
allocated is only a warning. Do not lower bag size, change precision or extend
caps to finish. Report wall-limited runs as inconclusive.

## 1. Verify interpreter, branch, committed identities and synthetic tests

```powershell
$ErrorActionPreference = 'Stop'
$py = 'C:/Users/Angus/COMP3710-private/.venv/Scripts/python.exe'
$worktree = 'C:/Users/Angus/Documents/Codex/2026-10-27/independently-audit-my-comp3710-alzheimer-s/work/scan-supervision-code'
$project = "$worktree/recognition/ConvNeXt_Alzheimers"
$entry = "$project/g012/run.py"
$data = 'C:/Users/Angus/COMP3710-private/data/ADNI'
$splits = 'C:/Users/Angus/COMP3710-private/patient-splits-20261022-155230'
$study = 'C:/Users/Angus/COMP3710-private/scan-supervision-g012'
$commonArgs = @('--data-root', $data, '--splits-dir', $splits, '--study-root', $study)
Set-Location -LiteralPath $project
if (-not (Test-Path -LiteralPath $py -PathType Leaf)) { throw 'Pinned interpreter missing.' }
& $py -B -c "import sys,torch,numpy,PIL,sklearn; assert sys.version_info[:3]==(3,11,9); assert str(torch.__version__)=='2.7.1+cu126'; assert numpy.__version__=='2.2.6'; assert PIL.__version__=='11.2.1'; assert sklearn.__version__=='1.6.1'; print(sys.executable); print('PINNED_RUNTIME_OK',torch.__version__,'CUDA_AVAILABLE',torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw 'Interpreter/version verification failed; do not install or alter the training environment.' }
$branch = & git -C $worktree branch --show-current
if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect experiment branch.' }
if ($branch.Trim() -ne 'codex/scan-supervision-g012') { throw 'Wrong branch; do not reset or rewrite it.' }
& git -C $worktree merge-base --is-ancestor 585152470540e98fca97d0b96e876a9cad8014cb HEAD
if ($LASTEXITCODE -ne 0) { throw 'Worktree is not descended from the submission baseline.' }
$planHash = (Get-FileHash -LiteralPath "$project/g012/G012_PLAN.json" -Algorithm SHA256).Hash.ToLowerInvariant()
if ($planHash -ne 'b824cbe7502f023b24d85ff38c8dfa4eedcf0060c119b1d231623fc6dba07133') { throw 'Frozen plan bytes changed.' }
& $py -B $entry verify
if ($LASTEXITCODE -ne 0) { throw 'G012 verify stopped; preserve evidence and inspect the message above.' }
& $py -B "$project/g012/test_g012.py"
if ($LASTEXITCODE -ne 0) { throw 'Synthetic tests failed; do not run the real smokes.' }
```

Expected: `PINNED_RUNTIME_OK`, `G012_COMMITTED_IMPLEMENTATION_VERIFIED` and 35 passing synthetic tests. This step runs no MRI inputs. Existing matplotlib/pyparsing deprecation warnings and a deliberately single-class fixture warning are not claimed to be package/environment repairs.

## 2. Prepare this fixed study from the existing approved manifests

```powershell
$ErrorActionPreference = 'Stop'
$py = 'C:/Users/Angus/COMP3710-private/.venv/Scripts/python.exe'
$worktree = 'C:/Users/Angus/Documents/Codex/2026-10-27/independently-audit-my-comp3710-alzheimer-s/work/scan-supervision-code'
$project = "$worktree/recognition/ConvNeXt_Alzheimers"
$entry = "$project/g012/run.py"
$data = 'C:/Users/Angus/COMP3710-private/data/ADNI'
$splits = 'C:/Users/Angus/COMP3710-private/patient-splits-20261022-155230'
$study = 'C:/Users/Angus/COMP3710-private/scan-supervision-g012'
$commonArgs = @('--data-root', $data, '--splits-dir', $splits, '--study-root', $study)
Set-Location -LiteralPath $project
$a1 = 'C:/Users/Angus/COMP3710-private/v2-scratch_augmented-20261024-161044-951159'
$priorReplay = 'C:/Users/Angus/COMP3710-private/submission-readiness/clean-verification-v2/prediction'
& $py -B $entry prepare @commonArgs --a1-summary "$a1/v2_summary.json" --a1-predictions "$a1/PRIVATE_validation_predictions.json" --a1-history "$a1/history.csv" --a1-checkpoint "$a1/PRIVATE_best_model.pt" --a1-replay-receipt "$priorReplay/prediction_receipt.json" --a1-replayed-predictions "$priorReplay/PRIVATE_predictions.json"
if ($LASTEXITCODE -ne 0) { throw 'G012 prepare stopped; preserve evidence and inspect the message above.' }
```

Expected: `G012_PREPARED_NO_IMAGES_DECODED`. This reads approved metadata and existing cached validation evidence and initializes private tensor/schedule digests. It does not regenerate partitions, decode images, train a model or infer predictions. Existing matching preparation is verified/completed; conflicting or unbound study directories are refused. It needs none of the old author pointer files.

## 3. Historical regression and three real-data TRAINING-ONLY GPU smokes

```powershell
$ErrorActionPreference = 'Stop'
$py = 'C:/Users/Angus/COMP3710-private/.venv/Scripts/python.exe'
$worktree = 'C:/Users/Angus/Documents/Codex/2026-10-27/independently-audit-my-comp3710-alzheimer-s/work/scan-supervision-code'
$project = "$worktree/recognition/ConvNeXt_Alzheimers"
$entry = "$project/g012/run.py"
$data = 'C:/Users/Angus/COMP3710-private/data/ADNI'
$splits = 'C:/Users/Angus/COMP3710-private/patient-splits-20261022-155230'
$study = 'C:/Users/Angus/COMP3710-private/scan-supervision-g012'
$commonArgs = @('--data-root', $data, '--splits-dir', $splits, '--study-root', $study)
Set-Location -LiteralPath $project
& $py -B $entry historical @commonArgs
if ($LASTEXITCODE -ne 0) { throw 'G012 historical stopped; preserve evidence and inspect the message above.' }
foreach ($variant in @('G0', 'G1', 'G2')) {
    & $py -B $entry smoke @commonArgs --variant $variant
    if ($LASTEXITCODE -ne 0) { throw "Smoke blocked for $variant; preserve evidence and stop. Do not change bag size or precision." }
}
```

Expected historical marker: `A1_HISTORICAL_REGRESSION_PASSED_REUSED_RECEIPT_NO_NEW_INFERENCE`.
This deliberately reuses the exact prior clean full-validation runtime receipt,
checks the checkpoint/source/manifest/preprocessing bindings, and re-compares its
cached slice/scan/patient probabilities. It is not new A1 MRI inference.

Each GPU smoke must print `G012_SMOKE_PASSED_NON_CANDIDATE`. It uses the same
three complete TRAINING scans (60 slices) and two updates including a one-scan
tail, with the full model and real preprocessing. Receipts are under
`smoke-G0`, `smoke-G1`, `smoke-G2`: `smoke.json`, `resources.json` and a PRIVATE
non-candidate checkpoint. No validation/test pixels or smoke accuracy selection.
Each real run initializes its model/RNG again. All three receipts and the
historical regression must pass before training. Actual GPU feasibility remains
unknown until you run this block; no old memory peak multiplier guarantees it.

## 4. Run G0, seed 42

```powershell
$ErrorActionPreference = 'Stop'
$py = 'C:/Users/Angus/COMP3710-private/.venv/Scripts/python.exe'
$worktree = 'C:/Users/Angus/Documents/Codex/2026-10-27/independently-audit-my-comp3710-alzheimer-s/work/scan-supervision-code'
$project = "$worktree/recognition/ConvNeXt_Alzheimers"
$entry = "$project/g012/run.py"
$data = 'C:/Users/Angus/COMP3710-private/data/ADNI'
$splits = 'C:/Users/Angus/COMP3710-private/patient-splits-20261022-155230'
$study = 'C:/Users/Angus/COMP3710-private/scan-supervision-g012'
$commonArgs = @('--data-root', $data, '--splits-dir', $splits, '--study-root', $study)
Set-Location -LiteralPath $project
& $py -B $entry run @commonArgs --variant G0
if ($LASTEXITCODE -ne 0) { throw 'G012 run stopped; preserve evidence and inspect the message above.' }
```

Watch `C:/Users/Angus/COMP3710-private/scan-supervision-g012/G0/execution.log` for `COMMITTED_EPOCH`. The parent prints `G012_RUN G0 COMPLETED_TRAINING` only after normal training completion. Selected-checkpoint replay is still required. Physical microbatch is one 20-slice scan, accumulation two scans, 531 updates/full epoch. No first-four-epoch equality with historical A1 is expected.

## 5. Run G1, seed 42

```powershell
$ErrorActionPreference = 'Stop'
$py = 'C:/Users/Angus/COMP3710-private/.venv/Scripts/python.exe'
$worktree = 'C:/Users/Angus/Documents/Codex/2026-10-27/independently-audit-my-comp3710-alzheimer-s/work/scan-supervision-code'
$project = "$worktree/recognition/ConvNeXt_Alzheimers"
$entry = "$project/g012/run.py"
$data = 'C:/Users/Angus/COMP3710-private/data/ADNI'
$splits = 'C:/Users/Angus/COMP3710-private/patient-splits-20261022-155230'
$study = 'C:/Users/Angus/COMP3710-private/scan-supervision-g012'
$commonArgs = @('--data-root', $data, '--splits-dir', $splits, '--study-root', $study)
Set-Location -LiteralPath $project
& $py -B $entry run @commonArgs --variant G1
if ($LASTEXITCODE -ne 0) { throw 'G012 run stopped; preserve evidence and inspect the message above.' }
```

Watch `C:/Users/Angus/COMP3710-private/scan-supervision-g012/G1/execution.log` for `COMMITTED_EPOCH`. The parent prints `G012_RUN G1 COMPLETED_TRAINING` only after normal training completion. Selected-checkpoint replay is still required. Physical microbatch is one 20-slice scan, accumulation two scans, 531 updates/full epoch. No first-four-epoch equality with historical A1 is expected.

## 6. Run G2, seed 42

```powershell
$ErrorActionPreference = 'Stop'
$py = 'C:/Users/Angus/COMP3710-private/.venv/Scripts/python.exe'
$worktree = 'C:/Users/Angus/Documents/Codex/2026-10-27/independently-audit-my-comp3710-alzheimer-s/work/scan-supervision-code'
$project = "$worktree/recognition/ConvNeXt_Alzheimers"
$entry = "$project/g012/run.py"
$data = 'C:/Users/Angus/COMP3710-private/data/ADNI'
$splits = 'C:/Users/Angus/COMP3710-private/patient-splits-20261022-155230'
$study = 'C:/Users/Angus/COMP3710-private/scan-supervision-g012'
$commonArgs = @('--data-root', $data, '--splits-dir', $splits, '--study-root', $study)
Set-Location -LiteralPath $project
& $py -B $entry run @commonArgs --variant G2
if ($LASTEXITCODE -ne 0) { throw 'G012 run stopped; preserve evidence and inspect the message above.' }
```

Watch `C:/Users/Angus/COMP3710-private/scan-supervision-g012/G2/execution.log` for `COMMITTED_EPOCH`. The parent prints `G012_RUN G2 COMPLETED_TRAINING` only after normal training completion. Selected-checkpoint replay is still required. Physical microbatch is one 20-slice scan, accumulation two scans, 531 updates/full epoch. No first-four-epoch equality with historical A1 is expected.

## 7. Replay all selected checkpoints and generate the comparison

```powershell
$ErrorActionPreference = 'Stop'
$py = 'C:/Users/Angus/COMP3710-private/.venv/Scripts/python.exe'
$worktree = 'C:/Users/Angus/Documents/Codex/2026-10-27/independently-audit-my-comp3710-alzheimer-s/work/scan-supervision-code'
$project = "$worktree/recognition/ConvNeXt_Alzheimers"
$entry = "$project/g012/run.py"
$data = 'C:/Users/Angus/COMP3710-private/data/ADNI'
$splits = 'C:/Users/Angus/COMP3710-private/patient-splits-20261022-155230'
$study = 'C:/Users/Angus/COMP3710-private/scan-supervision-g012'
$commonArgs = @('--data-root', $data, '--splits-dir', $splits, '--study-root', $study)
Set-Location -LiteralPath $project
foreach ($variant in @('G0', 'G1', 'G2')) {
    & $py -B $entry replay @commonArgs --variant $variant
    if ($LASTEXITCODE -ne 0) { throw "Replay blocked for $variant; preserve evidence. Use block 8 to export the incomplete outcome without GPU execution." }
}
& $py -B $entry compare @commonArgs
if ($LASTEXITCODE -ne 0) { throw 'G012 compare stopped; preserve evidence and inspect the message above.' }
```

Expected per completed variant: `G012_FULL_VALIDATION_REPLAY_PASSED`, or
`ALREADY_REPLAYED_VERIFIED`. Each replay uses the entire approved validation set,
with native G2 logit-mean scan aggregation, absolute 2e-5 probability tolerance
and unchanged slice/scan/patient decisions. G2 logits remain private. Replay
uses the SAME per-variant cumulative active budget; it never resets it.

`REPLAY_NOT_AVAILABLE_PROTOCOL_INCOMPLETE` is not a pass. A budget/GPU/integrity
failure stops the block. Block 8 independently exports the available evidence
and honestly marks incomplete/blocked outcomes; it performs no MRI inference.
`ALL_THREE_COMPLETE_AND_REPLAYED True` is required for complete protocol
reporting, not automatic candidate adoption. Selected epochs follow the original
accuracy/F1/NLL rule; do not replace them to satisfy a recall/screen condition.

## 8. Build and locate the allowlisted aggregate results ZIP

```powershell
$ErrorActionPreference = 'Stop'
$py = 'C:/Users/Angus/COMP3710-private/.venv/Scripts/python.exe'
$worktree = 'C:/Users/Angus/Documents/Codex/2026-10-27/independently-audit-my-comp3710-alzheimer-s/work/scan-supervision-code'
$project = "$worktree/recognition/ConvNeXt_Alzheimers"
$entry = "$project/g012/run.py"
$data = 'C:/Users/Angus/COMP3710-private/data/ADNI'
$splits = 'C:/Users/Angus/COMP3710-private/patient-splits-20261022-155230'
$study = 'C:/Users/Angus/COMP3710-private/scan-supervision-g012'
$commonArgs = @('--data-root', $data, '--splits-dir', $splits, '--study-root', $study)
Set-Location -LiteralPath $project
& $py -B $entry bundle @commonArgs
if ($LASTEXITCODE -ne 0) { throw 'G012 bundle stopped; preserve evidence and inspect the message above.' }
$pointer = Get-Content -LiteralPath "$study/aggregate_export_pointer.json" -Raw | ConvertFrom-Json
$directory = Join-Path $study $pointer.relative_directory
$zip = Join-Path $directory 'COMP3710_G012_Seed42_Results.zip'
if (-not (Test-Path -LiteralPath $zip -PathType Leaf)) { throw 'Results ZIP was not created.' }
Get-Item -LiteralPath $zip | Select-Object FullName, Length
Get-FileHash -LiteralPath $zip -Algorithm SHA256
```

The ZIP is stored at the printed absolute path under
`C:/Users/Angus/COMP3710-private/scan-supervision-g012/aggregate_exports/<content-derived directory>/COMP3710_G012_Seed42_Results.zip`.
The command resolves this future results-dependent directory automatically;
there is no path placeholder to fill in. Each changed aggregate snapshot gets
its own directory; previous exports are retained, never silently overwritten.
The same unchanged export is verified/skipped on repeat.

It contains the frozen public plan, sanitized configurations,
`G012_STUDY_SUMMARY.json`, A1/G0/G1/G2 aggregate history CSVs and two aggregate
plots. Replay/resource summaries are embedded in the study summary. It excludes
IDs, image paths, per-patient probabilities/logits, raw images, model/optimizer
states, logs and absolute private paths. It includes all variants and incomplete
outcomes. Return this ZIP with your separate analysis prompt. If blocked, also
provide the minimal terminal blocker/status text; keep all detailed evidence
local. Do not upload MRI data, manifests, checkpoints or per-patient predictions.

## 9. Show status from a new terminal

```powershell
$ErrorActionPreference = 'Stop'
$py = 'C:/Users/Angus/COMP3710-private/.venv/Scripts/python.exe'
$worktree = 'C:/Users/Angus/Documents/Codex/2026-10-27/independently-audit-my-comp3710-alzheimer-s/work/scan-supervision-code'
$project = "$worktree/recognition/ConvNeXt_Alzheimers"
$entry = "$project/g012/run.py"
$data = 'C:/Users/Angus/COMP3710-private/data/ADNI'
$splits = 'C:/Users/Angus/COMP3710-private/patient-splits-20261022-155230'
$study = 'C:/Users/Angus/COMP3710-private/scan-supervision-g012'
$commonArgs = @('--data-root', $data, '--splits-dir', $splits, '--study-root', $study)
Set-Location -LiteralPath $project
& $py -B $entry status @commonArgs
if ($LASTEXITCODE -ne 0) { throw 'G012 status stopped; preserve evidence and inspect the message above.' }
```

`STUDY_ACTIVE_LOCKED` means a process still owns the study; do not start a
second copy. Read the append-only execution log if needed. Do not keep editors
or PowerShell `Get-Content` monitors open on live atomic budget/state JSON files.

For an interrupted training run, use the matching standalone block below from
a new terminal. It restores the same complete-epoch state and cumulative ledger,
normally repeats a partial epoch, and never starts a replacement attempt. Disk
blockers may resume only after space is restored. WALL_LIMIT_INCONCLUSIVE,
BLOCKED_OOM and BLOCKED_HEADROOM are blockers to report, not permission to reset
or alter the experiment. Missing/unusable recovery evidence is also a blocker.

If interruption occurred during selected-checkpoint replay AFTER training,
repeat block 7 instead. That reuses the same selected weights and ledger;
already verified replays are skipped. Repeating a completed training resume
command verifies/skips it and does not reset its budget.

## 9a. Resume the original G0 attempt

```powershell
$ErrorActionPreference = 'Stop'
$py = 'C:/Users/Angus/COMP3710-private/.venv/Scripts/python.exe'
$worktree = 'C:/Users/Angus/Documents/Codex/2026-10-27/independently-audit-my-comp3710-alzheimer-s/work/scan-supervision-code'
$project = "$worktree/recognition/ConvNeXt_Alzheimers"
$entry = "$project/g012/run.py"
$data = 'C:/Users/Angus/COMP3710-private/data/ADNI'
$splits = 'C:/Users/Angus/COMP3710-private/patient-splits-20261022-155230'
$study = 'C:/Users/Angus/COMP3710-private/scan-supervision-g012'
$commonArgs = @('--data-root', $data, '--splits-dir', $splits, '--study-root', $study)
Set-Location -LiteralPath $project
& $py -B $entry status @commonArgs
if ($LASTEXITCODE -ne 0) { throw 'G012 status stopped; preserve evidence and inspect the message above.' }
& $py -B $entry resume @commonArgs --variant G0
if ($LASTEXITCODE -ne 0) { throw 'G012 resume stopped; preserve evidence and inspect the message above.' }
```

## 9b. Resume the original G1 attempt

```powershell
$ErrorActionPreference = 'Stop'
$py = 'C:/Users/Angus/COMP3710-private/.venv/Scripts/python.exe'
$worktree = 'C:/Users/Angus/Documents/Codex/2026-10-27/independently-audit-my-comp3710-alzheimer-s/work/scan-supervision-code'
$project = "$worktree/recognition/ConvNeXt_Alzheimers"
$entry = "$project/g012/run.py"
$data = 'C:/Users/Angus/COMP3710-private/data/ADNI'
$splits = 'C:/Users/Angus/COMP3710-private/patient-splits-20261022-155230'
$study = 'C:/Users/Angus/COMP3710-private/scan-supervision-g012'
$commonArgs = @('--data-root', $data, '--splits-dir', $splits, '--study-root', $study)
Set-Location -LiteralPath $project
& $py -B $entry status @commonArgs
if ($LASTEXITCODE -ne 0) { throw 'G012 status stopped; preserve evidence and inspect the message above.' }
& $py -B $entry resume @commonArgs --variant G1
if ($LASTEXITCODE -ne 0) { throw 'G012 resume stopped; preserve evidence and inspect the message above.' }
```

## 9c. Resume the original G2 attempt

```powershell
$ErrorActionPreference = 'Stop'
$py = 'C:/Users/Angus/COMP3710-private/.venv/Scripts/python.exe'
$worktree = 'C:/Users/Angus/Documents/Codex/2026-10-27/independently-audit-my-comp3710-alzheimer-s/work/scan-supervision-code'
$project = "$worktree/recognition/ConvNeXt_Alzheimers"
$entry = "$project/g012/run.py"
$data = 'C:/Users/Angus/COMP3710-private/data/ADNI'
$splits = 'C:/Users/Angus/COMP3710-private/patient-splits-20261022-155230'
$study = 'C:/Users/Angus/COMP3710-private/scan-supervision-g012'
$commonArgs = @('--data-root', $data, '--splits-dir', $splits, '--study-root', $study)
Set-Location -LiteralPath $project
& $py -B $entry status @commonArgs
if ($LASTEXITCODE -ne 0) { throw 'G012 status stopped; preserve evidence and inspect the message above.' }
& $py -B $entry resume @commonArgs --variant G2
if ($LASTEXITCODE -ne 0) { throw 'G012 resume stopped; preserve evidence and inspect the message above.' }
```

## Interpreting the fixed screen

Each variant needs >=86/102 correct, >=22/33 AD detections, macro-F1 >=
0.8012151782643586 and completed integrity/replay. G0 is eligible. A +0.02 ordinary
patient-NLL deterioration versus A1 or G0 is a review warning, not an extra hard
gate; NLL/Brier and both-class trade-offs are always reported. G1/G2 versus G0 is
reported separately; no two-patient G0 margin is required just to acknowledge a
useful candidate. G1–G0 isolates the loss in grouped batching; G2 also changes
scan aggregation. One seed and repeatedly used 102 validation patients cannot
establish repeatability or test performance. No model is adopted automatically.

STOP after these three fixed runs/replays and aggregate export. Do not add seeds,
fit policies/calibration, query test data or update the selected-model README.
Return the results for the separately authorized analysis task.
