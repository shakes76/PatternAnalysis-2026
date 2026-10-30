# G3 seed-42 gated-attention scan-pooling plan

Declared 28 October 2026, before any G3 MRI execution or outcome. The accompanying
G3_PLAN.json binds the exact runtime/configuration, approved manifests, historical
G2 evidence, control decision and pinned environment. This is the separately
authorised successor to the negative G012 study. A1 remains selected.

## Fixed control and scientific comparison

**REUSE_G2.** Historical G2 seed 42, selected epoch 20/completed epoch 28, is the
designated matched uniform control. No U3 is authorised or exposed by this CLI.
G2's recovered disk interruption, completed patience stop and existing successful
full-validation replay are retained. Do not replay G2 routinely, recharge its time,
or select a control based on new outcomes. Failed G3 smoke/integrity checks STOP;
they do not switch the control mode. A later discovery requiring U3 needs a
documented protocol revision and separate user execution.

G3 versus G2 tests learned attention weights together with their native weighted
logit inference against uniform mean-logit pooling. Both use complete-scan CE.
G0 versus A1 had changed grouping/update structure; G1 failure did not disprove
attention. The negative G012 conclusion is unchanged. One seed cannot establish
repeatability or clinical benefit. Retain A1 seed42/epoch13 whatever the screen.

## Exact model and loss

Fresh original one-channel ConvNeXt-Tiny: 27,818,594 shared parameters, widths
96/192/384/768 and depths 3/3/9/3, stochastic-depth maximum 0.1. A temporary hook
captures the actual input/output of its existing linear head during the original
forward: h is the 768-vector after per-slice spatial mean and final LayerNorm.
The original head still computes z_i = W h_i + b exactly once per slice batch.
No moved/added normalisation, detached/cached features or duplicate backbone pass.

Attention: u_i=tanh(Vh_i+bV)*sigmoid(Uh_i+bU); s_i=w^T u_i;
a=softmax(s, dimension=slices); scan_logits=sum_i(a_i*z_i).
V/U are biased 768→128; w is bias-free 128→1. V/U truncated normal std .02,
biases zero, w zero. Seed 4242 isolates constructor AND initialisation RNG draws
from Python/NumPy/Torch CPU/CUDA training streams. Initial weights are uniform;
w can learn first, V/U score gradients initially zero and subsequently nonzero.
Added parameters: 196,992; total 28,015,586. No other attention hyperparameters.
Uniform G2 retains its actual mean(slice_logits) loss/inference computation;
algebraic head(mean_features) is a synthetic reference, not a production rewrite.

Hard-label unweighted CE on scan logits; no smoothing/auxiliary loss. Native scan
probabilities are softmax(scan_logits). Patients use ordinary mean scan AD
probabilities, threshold >=0.5, NC=0/AD=1. Per-slice softmax is diagnostic only.

## Inherited fixed training and selection

Approved training 476 patients / 1,061 scans / 21,220 slices; validation 102
patients (69 CN,33 AD) / 233 scans / 4,660 slices. All scans have 20 slices.
No partition changes, rebalancing, omissions or inferred physical anatomy.
Aspect-preserving grayscale preprocessing and independent ±5° training-only
slice rotation; zero workers; float32; inherited deterministic CUDA flags.
One complete scan per forward/backward, accumulate two scans, actual-length
single-scan tail division; 531 updates/completed epoch. Release each scan graph
after backward. Reuse all 30 original canonical scan orders/rotation arrays and
the per-epoch isolated stochastic-depth seed; no first-four-epoch score equality.

Original AdamW defaults (betas .9/.999, eps 1e-8, amsgrad false and unchanged
backend defaults). Preserve all shared group membership. Rank>1 decay .05,
bias/norm vectors decay zero. Added attention decay group contributions are
196,736 matrix parameters and 256 zero-decay bias parameters. Original head
groups remain original head groups; attention joins the corresponding non-head
groups. Global norm clip 5 after accumulation includes ALL trainable parameters.
Log every preclip norm (JSON array per history row), mean/max and clipped-update
fraction. Attention can change global clipping; separate clipping is forbidden.

LR epochs1–3: 3e-4*epoch/3; epoch4 peak3e-4; afterwards original cosine to1e-6
at epoch30. Max30/minimum run12/patience8; validate every complete epoch. Select
highest patient accuracy, then macro-F1, then lower ordinary raw patient NLL;
earliest exact tie. An epoch below12 may be selected. Never substitute an epoch
to satisfy a class/screen condition. No full training-set accuracy is invented.

## Execution, recovery and fixed stopping limits

Implementation work is synthetic/read-only only. User first runs committed source
verification, prepare/control metadata verification, then non-candidate G3 smoke.
Smoke uses the first two canonical CN training scans and first canonical AD scan
(sorted indices), exactly three unique complete scans, two optimiser steps with
tail, initial attention uniformity, finite loss/gradient, changed weights and
strict checkpoint roundtrip. Its evaluation passes use those same training
scans only. Smoke accuracy is not a tuning signal. The real worker constructs
fresh seed42/shared and seed4242/attention states; never uses smoke/trained G2.

Preparation retains immutable private full G3 E0 (including shared tensors), RNG,
its full/shared digests and SHA, and all matched schedules. Epoch recovery is
atomic, preserves previous complete state and restores model/optimiser/RNG,
history/best selection/custom schedule. One attempt ID; repeated partial work
remains charged. Conflicting outputs or source/input changes fail closed.

Exactly one new attempt: cap10,800 supervised active seconds INCLUDING selected
replay, total10,800 new-study seconds. Historical G2 charged8178.031 seconds is
reported separately. Reserve300 seconds: the same supervisor stops training
at cumulative10,500 seconds, retaining latest complete epoch; replay may use
the remaining cap. A training wall stop is inconclusive, cannot be resumed
past its terminal limit, and is not a successful completion. A replay cap stop
also prevents screen passage. No budget extension or replacement attempt.
The reused five-second prepaid leases conservatively preserve uncertain work
on parent crash; downtime excluded. Preparation/non-candidate smoke are separately
timed and reported. Complete-epoch loop time is a separate, narrower timing scope.

Use actual output-volume >=12 GiB free guard, inherited OS study lock, atomic
state writes and Windows kill-on-parent-close job. Require available CUDA
memory >=max(1GiB,10% total). >4.5GiB allocated warns, not a hardware ceiling;
historical ~2.55GiB is reference only. OOM/nonfinite/headroom/integrity failure:
stop and investigate, never change batch/precision/head. Disk pauses preserve
the original attempt and ledger. Incomplete smoke is not silently replaced.

## Measurements, screen and export

Save native selected scan logits, original slice logits/softmax and attention
weights with stable verified slice paths privately; no full feature maps. Save
complete histories, selected/recovery states, source/configuration/manifest/control
identity, resource/budget records. Derive attention diagnostics from existing
outputs: maximum weight, normalised entropy H/log(K), effective slices exp(H).
K=1 convention is normalised entropy1 and effective count1. Summarise distributions
at selection and train/validation means per epoch. No entropy thresholds; attention
is not a causal/anatomical explanation. Filename position is not verified anatomy.

Strict selected replay: every slice/scan/patient probability difference <=2e-5,
identical class decisions; attention weights <=2e-5 and identical stable ordering.
G3 schema/configuration/keys/shapes/dtypes/finiteness/source bindings are strict;
G2 cannot be loaded permissively as G3 or vice versa. Replay only validation.

Compare A1, historical G2 and G3, complete histories and fixed last five completed
epochs. Report correct/N, accuracy, both-class precision/recall, confusion counts,
macro-F1, AUROC, ordinary patient NLL/Brier, native scan and labelled slice metrics,
paired gained/lost correct by class, attention distributions, parameter/update
counts, active/loop/smoke/preparation time, allocator/device memory and interruptions.
Initial screen: >=86/102 correct, >=22/33 AD detections, macro-F1 >= exact
0.8012151782643586 and normal completion/integrity/selected replay. NLL/Brier are
trade-offs, not hard gates. Warn if NLL worsens >.02 versus A1 OR matched G2.
Explicitly report any loss against G2's27 AD detections. No arbitrary two-patient
margin, no extra seeds/promotion/calibration/test query following a screen pass.

Export only explicit aggregate allowlist in COMP3710_G3_Seed42_Results.zip:
sanitised plan/control decision, comparison (including configuration, histories,
resources/replay summaries), CSV histories, aggregate trajectory plot and results
summary. Never export identifiers, private paths, per-case probabilities/logits,
attention weight arrays, MRI, models or optimiser state. Historical evidence is
labelled historical, not newly reproduced. Keep full evidence local for analysis.

## Evidence and assistance

Authority: supplied 01_IMPLEMENT_G3_AND_RETURN_COMMANDS.md; actual frozen G012
source/evidence and completed G012 analysis; submission-readiness handback.
The supplied deep-research-report.md motivates attention with uncertainty. Its
parameter estimate assumes a different extra classifier; this study retains the
original shared classifier and uses the exact counts above. Original method:
Ilse, Tomczak & Welling, Attention-based Deep Multiple Instance Learning (2018),
https://arxiv.org/abs/1802.04712. No new literature verification is claimed.
Codex assisted implementation, tests, evidence checks and command documentation;
these are not claims of author personal review or MRI execution in this task.
