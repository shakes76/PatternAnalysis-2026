# A1 schedule endpoint experiment

This extension tests one scientific change: cosine decay reaches 1e-6 at epoch
12 rather than 30, holding the floor afterwards. It preserves the historical
source, A1 selection, CNN, manifests, checkpoints, pointers and earlier results.
No test inference, individual test predictions, calibration, threshold searches,
pretraining, multi-slice learning, architecture changes or automatic promotion.

## Fixed protocol

Seeds 42, 43, 44 run once each, sequentially, in private `schedule-only-end12/seed-N`.
Resume the same directory after interruption; never create replacement attempts.
All settings equal the matched A1 configuration: custom scratch ConvNeXt-Tiny,
unweighted hard-label cross-entropy, peak LR 3e-4, floor 1e-6, warmup three epochs,
training rotation +/-5 degrees per slice, batch 8, accumulation 4, AdamW decay
grouping 0.05, clipping 5, float32/strict deterministic settings, zero workers,
30 maximum epochs, 12 minimum epochs, patience 8. Accuracy, macro-F1 and negative
ordinary patient log loss select checkpoints lexicographically; earliest exact tie.

For epochs 1-3 use LR=3e-4*epoch/3. Thereafter use t=min(1,(epoch-4)/8) and
LR=1e-6+(3e-4-1e-6)*(1+cos(pi*t))/2. Epoch 4 retains the peak; epoch 12 reaches
the floor. Reuse the original model, optimizer, train_epoch, data and aggregation.

## Comparability declared before training

Compare the first four actual epochs to each matched archived A1 history.
Absolute tolerances: 1e-6 for training, validation slice and patient log loss;
1e-12 for training slice accuracy, patient accuracy, macro-F1, AUROC and class
recalls. Optimizer update counts must match exactly. Timing is excluded.
Check each epoch before committing its state. A failure stops the study and
records differences; do not relax tolerances or repeat those epochs as a new run.
The first-four epochs are part of the actual budget and experiment.

## Recovery and budget

Require 12 GiB free disk before and between epochs. A parent supervisor accounts
the entire child invocation, including model setup, validation, checkpoint I/O,
repeated partial epochs and final validation replay. This conservatively exceeds
optimization-only time. Exclude time between stopped invocations.

Each seed has 10,800 seconds; total at most 32,400 seconds. Before child launch,
and while it executes, atomically reserve up to five seconds ahead. A supervisor
crash retains uncertain reserved time; a clean exit refunds only demonstrably
unused reservation. The ledger never resets across resumptions. A Windows job
with kill-on-close prevents a child from continuing when its supervisor dies.
The child waits at a unique gate until assigned to that job. At the wall limit
the supervisor terminates it and preserves atomic completed-epoch states;
mark WALL_LIMIT_INCONCLUSIVE even if a partial training result looks good.
Unexpected errors stop for investigation. Never change batch size or budget.

State includes current/best weights, optimizer, RNG, history, patience,
configuration, schedule/loader state, manifest/source/environment/plan bindings
and first-four comparison receipts. Keep the previous complete state as fallback;
preserve unreadable latest files. No readable state for an initialized attempt
is a stopping condition, not authorization to restart from scratch.

## Verification and analysis

Commit this plan, JSON bindings, implementation and synthetic tests before real
training. Full validation replay of each selected exported checkpoint must match
saved slice probabilities within 2e-5, with unchanged slice/scan/patient classes.
Check exact tensor shapes, dtypes, finite values and provenance before loading.
Do not substitute an epoch to meet an acceptance condition.

Controls have 85/80/80 correct and 22/21/23 AD detections over the same 102
validation patients (33 AD). Recompute the exact control mean patient log loss
in the bound JSON plan. All four acceptance gates must pass:

1. Summed correct >=251, at least two seeds improve, none loses >1 correct.
2. Mean AD recall >= controls; seed42 >=85 correct and >=22 AD detections.
3. Mean ordinary patient log loss <= the exact three-control mean.
4. Integrity/loading/replay checks and all four comparability epochs pass.

Report all runs, selected epochs, resources and seed-matched metric differences.
For each control/candidate report the final five completed epochs' patient
accuracy and patient log loss: values, mean, population SD and range. These are
the last five epochs, not five selected favorable epochs. Three seeds reuse 102
patients; do not describe them as 306 independent subjects. Incomplete runs are
inconclusive. Stop after these three seeds; do not promote, recalibrate, test,
add seeds or launch other configurations.

## Commands

From recognition/ConvNeXt_Alzheimers using the supplied private Python:

```
python -B test_schedule_only.py
python -B train_schedule_only.py prepare --private-root C:\Users\Angus\COMP3710-private
# Review and commit the four new files before running:
python -B train_schedule_only.py run-all --private-root C:\Users\Angus\COMP3710-private
python -B train_schedule_only.py run --private-root C:\Users\Angus\COMP3710-private --seed 42
python -B train_schedule_only.py resume --private-root C:\Users\Angus\COMP3710-private --seed 42
python -B train_schedule_only.py compare --private-root C:\Users\Angus\COMP3710-private
```

run-all processes fixed seeds in order and reuses completed runs; individual
run/resume commands also enforce order and use the same original attempt.
No historical latest pointer or main README is written by these commands.
Codex assisted with implementation, tests and this plan. Author review and
disclosure remain required; passing tests is a scoped assurance, not proof.
