# G3 export repair — replacement steps 10 and 11

The original exporter rejects its own aggregate boolean
`replay.checks.attention.slice_order_unchanged`: the frozen privacy guard forbids
the substring `slice_order`, intended to block private slice-path arrays. This
is an export defect, not evidence that training or replay failed.

The additive `export_adapter.py` renames only that verified `True` boolean to
`ordering_unchanged` in the newly constructed public comparison. It then invokes
the original unchanged privacy guard. False/non-boolean/conflicting checks and
private identifiers, paths, arrays and predictions still fail closed. Private
replay receipts, all 27 trained source bindings, frozen plan, checkpoint, metrics
and budgets are unchanged. Do not regenerate the plan or rerun training/replay.
The adapter separately requires its own exact source to be committed in HEAD.

Eight synthetic regression tests cover the actual replay-check shape, retained
privacy rejection, unchanged input receipt, scoped adaptation, output-free check,
the plan sharing the study name without comparison rows, and the complete frozen
comparison/plan/history/plot/ZIP pipeline, including repeated export. The first
adapter incorrectly assumed every object with this study name had rows; its
check-only run stopped before reaching the plan. That was inadequate verification.
The implementation-time synthetic fixture also omitted the real attention replay
check shape, allowing the original privacy-key defect to escape that suite.
Codex assisted the repair and verification. No inference or training is involved.

Use these replacement blocks after the successful selected-checkpoint replay.
Each works from a new PowerShell 5.1 terminal. They execute only saved-evidence
comparison/export; no successor experiment, model promotion or calibration.

## 10. Cached comparison with repaired public-key sanitisation

```powershell
$ErrorActionPreference = 'Stop'
$py = 'C:/Users/Angus/COMP3710-private/.venv/Scripts/python.exe'
$assignment = 'C:/Users/Angus/Documents/Codex/2026-10-27/independently-audit-my-comp3710-alzheimer-s/work/scan-attention-code/recognition/ConvNeXt_Alzheimers'
$entry = Join-Path $assignment 'g3/export_adapter.py'
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
if ($LASTEXITCODE -ne 0) { throw 'G3 cached comparison stopped; preserve evidence and inspect the error.' }
```

Expected `G3_AGGREGATE_COMPARISON_EXPORTED` and the adapter SHA. The original
step-10 command using `g3/run.py` remains frozen and will still encounter the
original defect; use this separate entry point for both replacement steps.

## 11. Bundle and locate the shareable aggregate ZIP

```powershell
$ErrorActionPreference = 'Stop'
$py = 'C:/Users/Angus/COMP3710-private/.venv/Scripts/python.exe'
$assignment = 'C:/Users/Angus/Documents/Codex/2026-10-27/independently-audit-my-comp3710-alzheimer-s/work/scan-attention-code/recognition/ConvNeXt_Alzheimers'
$entry = Join-Path $assignment 'g3/export_adapter.py'
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
if ($LASTEXITCODE -ne 0) { throw 'G3 aggregate bundle stopped; preserve evidence and inspect the error.' }
$latest = Get-Content -LiteralPath (Join-Path $study 'latest_aggregate.json') -Raw | ConvertFrom-Json
$export = Join-Path (Join-Path $study 'aggregate_exports') $latest.directory
$zip = Join-Path $export 'COMP3710_G3_Seed42_Results.zip'
Get-Item -LiteralPath $zip | Select-Object FullName, Length
Get-FileHash -LiteralPath $zip -Algorithm SHA256
```

Return that ZIP for the separate analysis task. Do not zip the private study tree.
