# Frozen final evaluation protocol

Recorded before final test-label release at **2026-10-18T12:59:32.985259+00:00** (UTC).
Approved Stage 7 revision: `e72784d295137db604932d63b08dd416fddf487e`.
No held-out test evaluation had occurred when this record was written.

Models, numeric preprocessing, seed-42 masks, architecture/training settings,
calibration and thresholds remain unchanged. Three new driver tests passed using
selection labels only before release. Core algorithm ASTs match Stage 7.

| Model | Epoch | Temperature | Threshold | Checkpoint SHA-256 |
| --- | ---: | ---: | ---: | --- |
| MLP | 148 | 0.9440649462009191 | 0.8765371476783451 | `0429ce4eb0ea381290e038d1438401a4b92a5a779a4285a71dcba37400678062` |
| GCN | 170 | 0.7836173067240391 | 0.6341986811730764 | `63ad1b19b7659ccbbc6618ef079a800166299f6e2b921e04f5295f199f1a6cd4` |

Both models: 128 -> 64 -> 4, ReLU, dropout 0.5, Adam lr 0.01, weight decay
0.0005, maximum 300 epochs, patience 30, min-delta 0.0001, seed/split seed 42.

Data SHA-256: `7a1b8a0687515e9859d1ca649dc276825f5db44568894dfbb6b55ebc1cf6a9c1`.
Policy SHA-256: `cda69812bdd3c6ad0227f5bfac03123e308173837f31ec26084d7dbd7cea05c2`.
The exact source/data/split/configuration manifest is retained locally at
`outputs/frozen_evaluation_manifest.json`, excluded from Git. The evaluator
checks every recorded identity before requesting labels, and rechecks afterward.

## Fixed procedures

- **classification:** argmax; per-class precision/recall/F1, macro-F1, macro OvR AUROC.
- **probabilities:** raw and saved-temperature softmax on CPU float64.
- **calibration:** existing ECE 15 equal-width bins, stable NLL, summed multiclass Brier.
- **referral:** existing >= saved threshold, whole ties; no search/refitting.
- **latency:** full graph; first call, 20 warmup then 100 synchronized calls.
- **failures:** same five-case ordering as Stage 6; training-labelled neighbors only.

No training, calibration fitting, threshold search, hyperparameter optimization
or source-method changes may follow test inspection. Diagnostic risk curves
are descriptive only; their alternative thresholds must not alter the frozen
operating point. Raw/scaled classification scores are both reported; AUROC is
macro one-vs-rest. Both models use identical 4,494 test IDs. Failure selection
uses the existing ordered high/low-confidence and model-disagreement rule,
with only training-labelled neighbor composition.

The CLI requires `--release-test-labels`; missing release permission or manifest
mismatch rejects evaluation before label access. Durable label-release audits
record manifest identity and UTC time. A deterministic rerun is allowed; keep
the first results and compare rather than selecting favorable outcomes.
