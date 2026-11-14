# Rangpur and ADNI Audit — 2 October 2026

## Evidence and scope

Read-only checks were performed over SSH and inside short CPU allocations. The final reproducible dataset audit was job **620988**, on `vcpu-5`: **COMPLETED, exit 0:0, 22 seconds**. Its UTC timestamp corresponds to 03:24 on 2 October in Australia/Brisbane. See [aggregate JSON](audit-2026-10-02.json) and [audit source](../tools/audit_adni.py).

No GPU job was requested, no training/model loading occurred, no environment was modified, and no existing data or checkpoint was deleted. An existing unrelated GPU training job remained running. One small audit script was uploaded to a separate home directory. The JSON contains aggregates, not patient identifiers or images.

## Dataset findings

Source: `/home/groups/comp3710/ADNI/AD_NC`, with `meta_data_with_label.json` in the parent directory.

| Property | Observed value |
| --- | --- |
| JPEG slices | 30,520 |
| Total JPEG bytes | 191,363,711 (approximately 182.5 MiB) |
| Dimensions and mode | All headers: 256 × 240, grayscale `L` |
| Scans represented | 1,526; exactly 20 images per scan |
| Patients represented | 680, inferred from metadata raw-path patient identifiers |
| Metadata records | 2,189, all with parseable patient IDs; not all records have course JPEGs |
| Filename/metadata/header errors | None detected by this audit |

| Supplied split/class | Images |
| --- | ---: |
| train/AD | 10,400 |
| train/NC | 11,120 |
| test/AD | 4,460 |
| test/NC | 4,540 |

The provided folder labels map consistently to metadata values: **AD = 2, NC = 0**. Do not assume the opposite numeric encoding or treat every metadata record as an included image. No represented patient appears in both class folders.

## Critical split finding

The supplied training directory contains 1,076 scans from 565 patients. The test directory contains 450 scans from 331 patients. There are **zero shared scan IDs but 216 shared patient IDs**.

Consequently, the supplied split does not meet our patient-independent evaluation criterion. Repeated scans of the same patient can cross the directory boundary even though filenames differ. This audit establishes identity overlap; it does not measure the resulting metric inflation.

Proposed next step: build a manifest and a deterministic patient-grouped train/validation/test partition over the available images, stratifying by patient class where feasible. Preserve the original directories and record the original split as provenance. Confirm any mandated course split before finalizing the evaluation protocol. The new split must be shared by the simple baseline and all diffusion comparisons. Actual patient-level manifests stay outside the public repository.

## Environment

The existing course environment reports Python 3.11.15, torch package 2.13.0, torchvision 0.28.0, NumPy 2.4.6 and Pillow 12.3.0. Diffusers, Transformers, Accelerate, PEFT, safetensors and nibabel are absent from this interpreter's package metadata.

Package inventory is not an import/compatibility test. No GPU availability or model operation was tested in this CPU audit. Use a project environment and a pinned compatible dependency set; do not alter the environment supporting the other training project. Nibabel is not required just to read the supplied JPEG slices.

Non-login SSH shells initially lacked Slurm on PATH. `bash -lc` or sourcing `/etc/profile.d/slurm.sh` exposes `/opt/slurm/bin`.

## Storage

- Home filesystem: 16 GiB total, approximately 2.20 GiB free at the final audit.
- CPU job scratch: approximately 196.75 GiB free; ephemeral, not a durable checkpoint destination.
- Preliminary directory accounting: existing conda approximately 4.3 GiB; unrelated training project approximately 8.8 GiB; its venv approximately 233 MiB. These were observed, not cleanup targets.
- CPU node physical memory: approximately 46 GiB, 43 GiB available at the initial check. This is a shared node snapshot, not a reserved per-job allowance.

The dataset is modest in size, but free home space does not establish a safe SD model-cache/checkpoint budget. Before downloading models or starting adaptation, identify sufficient persistent storage, measure the expected checkpoint contents, and allow for overlapping files during atomic saves. Do not rely on scratch surviving a job. Do not delete another project's artifacts to make space.

## Scheduling findings

`comp3710` currently lists ten A100 nodes, partition `MaxTime=UNLIMITED`, default time 3 hours, and partition QoS `comp3710`. The **QoS** record sets `MaxWall=12:00:00` and `MaxJobsPU=1`. The `a100-test` partition also reports unlimited partition time, but its `test` QoS limits jobs to 20 minutes and one job per user.

Thus `sinfo` displaying `infinite` does not imply unlimited effective job duration. Account association reports the normal QoS; the running unrelated course job has `QOS=normal`. Effective restrictions combine partition and job/account policy; do not infer permission to bypass the course QoS from that field.

Slurm reports placeholder-like memory TRES (`mem=1M` per node) and unlimited partition memory defaults. These values do not describe physical RAM. The earlier failure to schedule explicit memory requests may be related, but causation was not verified. Confirm suitable memory-request practice before formal training.

## Reproduction

Transfer the audit script without publishing the development branch. In a login shell, run:

```bash
srun --partition=cpu --time=00:05:00 --cpus-per-task=1 --immediate=30 \
  "$HOME/miniconda3/envs/torch/bin/python" \
  "$HOME/comp3710-sd-audit/audit_adni.py"
```

Capture stdout as JSON and stderr separately. `--immediate=30` bounds the resource-allocation wait. Confirm completion with `sacct`; a log file alone does not establish success.

## Failures and limits

An initial shell command had nested PowerShell/SSH quoting errors. CPU probe job 620980 subsequently returned exit 127 because a CRLF heredoc terminator was interpreted as a command, after printing partial resource results. Corrected LF scripts succeeded in jobs 620981 and 620982. The final file-based audit succeeded in job 620988; the failed probe is not counted as a successful audit run.

Image headers were inspected, but full pixel decoding, duplicate-content hashing, orientation/clinical assessment, VAE reconstruction, model downloads, and GPU checks remain outstanding. Patient grouping relies on the supplied metadata naming convention. Dataset split creation and persistent-storage planning precede training.
