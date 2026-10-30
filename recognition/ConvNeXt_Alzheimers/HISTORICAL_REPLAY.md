# Historical source and replay

The scientific A1 selection remains scratch_augmented, seed 42, epoch 13, with
checkpoint SHA-256 `efde4d21993085119d3ef1bc0efb88f837954c187c95b036373cd00e59d4d5ef`.
The schedule endpoint experiment is a completed negative experiment, not A1.

The pre-submission implementation is Git revision
`b67bb1327c744fdd9a30fea439efc4a5005fca27`. Progressive history is retained.
`historical_runtime/` contains exact, unedited files needed to reconstruct the
retained A1 and regularized CNN checkpoints. `HISTORICAL_RUNTIME.json` records
their byte hashes and revision. The full original assignment source was also
backed up privately before editing dataset.py, train.py or predict.py.

The new `predict.py infer` route checks every source hash stored in a checkpoint
against either the current source or this frozen directory, then executes the
matching dataset/model/aggregation implementation in a fresh process. It does
not relabel edited files as historical files or change checkpoint expectations.
The model state undergoes exact key/shape/dtype/finite checks before strict loading.
The frozen source itself is never patched to make a mismatch pass.

For source inspection, from the repository root:

```powershell
git show b67bb1327c744fdd9a30fea439efc4a5005fca27:recognition/ConvNeXt_Alzheimers/dataset.py
git show b67bb1327c744fdd9a30fea439efc4a5005fca27:recognition/ConvNeXt_Alzheimers/train.py
```

A separate detached worktree at that revision can run the complete historical
implementation with its original private inputs. The original local checkout
and its pointer files were left intact during submission packaging. The old
source-hash checks still apply: historical analysis/evaluation commands are not
expected to accept edited submission files as their original code.

For an independent validation-only replay, use the README's `predict.py infer`
command with the retained checkpoint digest, explicit data/manifests and
`--source-dir historical_runtime`. It needs no previous CNN/report/test tree.
Legacy `freeze`/`evaluate` and `revised-*` commands remain separate with their
existing explicit-test and provenance guards; this task did not execute them.

No model weights, patient manifests, predictions or medical-image panels are
included here. The recorded checkpoint is available in the author's authorized
local private workspace; public redistribution permission is not established.
Someone without authorized access must use the documented training route on
authorized data. Epoch 13 and exact accuracy are recorded results, not promises
for a fresh training run or a different platform.
