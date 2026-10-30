# G3 control comparability decision

**REUSE_G2 — fixed before any new G3 outcome. No fresh U3.** Machine decision:
G3_CONTROL_COMPARABILITY.json. G3 real-data smoke is pending and required; it
cannot change this choice silently. No real-data compatibility inference is
required because the original feature/head forward is retained without rewriting
its operations. Stop if smoke fails. A newly discovered issue needs a documented
revision before any fresh control is authorised/executed.

The preserved completed worktree is `work/scan-supervision-code`, branch
codex/scan-supervision-g012, revision500bfe84168af9b7f6d59634276802765fdacbc4.
The new branch derives from it and retains submission revision5851524.
Historical G2 is seed42, selected20/completed28, patience stop, correct81/102,
AD27/33. Its prior disk stop was recovered; final training and selected replay
completed inside the10800-second budget (8178.031 total charged seconds).
It is below the old A1 screen; low score does not invalidate a matched control.

| Requirement | Evidence and scope |
|---|---|
| Actual feature and head | modules.ConvNeXtTiny.forward executes each original downsample/stage, spatial mean, final_norm, head. G3 uses that exact method and captures its actual head input/output with a temporary forward hook. No added backbone pass or reordered LayerNorm. Original head is linear768→2. |
| Uniform path | Historical G2 scan_loss is negative log-softmax of mean slice logits; scan_probs is softmax of mean logits. No new uniform wrapper/attention constructor/optimiser path is introduced. Synthetic uniform feature/head algebra is a reference only. |
| Shared initialisation | Full seed42 original model reconstructed on CPU matches recorded digest0a1fa0bc64eb8d5cceee73d246f1272caccfa0fc98790802516532a94679bcfa. All shared tensors equal the additive G3 initial model. Old E0 was overwritten: this is digest consistency, not observed old tensor-by-tensor equality. New prepare saves immutable full G3 E0 privately. |
| RNG and stochastic depth | Isolated attention constructor/init seed4242 leaves Python/NumPy/Torch CPU/CUDA RNG unchanged. Full-model controlled-mask comparison tests exact original slice logits and post-forward stream, then shared gradients. Arbitrarily reordered batched dropout is not assumed equivalent. |
| Orders and rotation | All30 recorded complete scan orders and per-slice angle arrays recomputed from approved metadata and match exactly. The epoch transaction uses the same model_training seed independently of these generators. Original arrays are read-only. |
| Optimisation and validation | Same531 updates, actual two-scan/single-tail division, AdamW defaults/shared groups, globalclip5, float32, deterministic flags, zero workers, original epoch30 LR, maximum/minimum/patience, validation cadence and patient selection. G3 adds attention tensors to the same rank-based groups/global clip; those gradients can change clipping and trajectories. |
| Historical evidence | Frozen source/configuration/manifest bindings, completion hashes, all28 history rows/selected20/update counts, selected/replay checkpoint/prediction SHA bindings and recorded 2e-5 all-level replay were checked from saved evidence. Native mean-logit probabilities reconstructed from saved validation slice logits. No new MRI inference or G2 replay. |
| Synthetic compatibility | Full original ConvNeXt logits/features/loss/shared gradients; small deterministic accumulation/update reference; attention gradients and RNG; strict checkpoints; artificial epoch recovery; budget/OS-job/export checks. Exact executed counts and logs are in the implementation handback. Tolerances are declared in test_g3.py and frozen JSON, not loosened after failures. |

Hashes bind bytes and support receipt reuse; they do not establish chronology or
independently reproduce historical training. CPU synthetic agreement does not
prove identical CUDA learning trajectories or candidate benefit. G3 is expected
to diverge as weights learn; even initially uniform attention does not imply
identical later gradients/global clipping. No first-four-epoch equality gate.

The only G012 source extension in the NEW worktree is optional service injection
in engine.supervised/advance, reusing the same supervisor/epoch transaction.
Default G012 behaviour is covered by inherited tests. Original G012 bytes and
bindings remain untouched in the completed worktree. Its old plan is deliberately
not rewritten to bless the new engine: run historical commands in the preserved
G012 worktree. The new G3 plan separately binds its extended engine.

No G3 MRI smoke, training, inference or selected-checkpoint replay has been run
by Codex. The user must run the command guide and return the allowlisted ZIP.
