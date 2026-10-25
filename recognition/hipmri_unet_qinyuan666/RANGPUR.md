# Rangpur: midpoint data and training handoff

The immediate objective is access confirmation, correct data interpretation and
a small U-Net pilot if allocated resources and time permit. The midpoint does
not require a completed Improved U-Net comparison. Existing synthetic resource
measurements are preliminary engineering evidence only; they do not establish
HipMRI accuracy. The student must attend the tutor check-off personally.

## 1. Connect from local Windows PowerShell

The SSH account and host below were supplied by the student. The course PDF gives
the data path used here. Consult the authenticated
[UQ compute instructions](https://student.eait.uq.edu.au/infrastructure/compute/)
for the actual network and GPU-allocation rules; those rules could not be verified
without university access. No VPN, jump host, scheduler, partition or allocation
account is assumed in this guide.

```powershell
ssh s4906920@rangpur.eait.uq.edu.au
```

Use the university authentication flow. A password is not echoed while typing.
If SSH asks about an unfamiliar host key, compare its fingerprint with the UQ
instructions/support before accepting. Do not send passwords or authentication
codes to this chat.

Everything below runs in the **remote Bash terminal** after login. First confirm
the host and data-directory access; this does not read image contents:

```bash
whoami
hostname
python3 --version
ls -ld /home/groups/comp3710/HipMRI_Study_open/keras_slices_data
```

`Permission denied` needs course access support; `No such file or directory` needs
the current course data location. Do not attempt a recursive scan of shared storage.

## 2. Get this project's fork and branch

On the **first** Rangpur setup, run:

```bash
git clone --single-branch --branch topic-recognition https://github.com/qinyuan666/PatternAnalysis-2026.git "$HOME/PatternAnalysis-2026"
cd "$HOME/PatternAnalysis-2026/recognition/hipmri_unet_qinyuan666"
```

If that checkout already exists, update it instead:

```bash
cd "$HOME/PatternAnalysis-2026"
git status --short
git branch --show-current
git remote get-url origin
git pull --ff-only origin topic-recognition
cd recognition/hipmri_unet_qinyuan666
```

Only pull when the branch is `topic-recognition`, origin is the fork above and
the worktree is clean. If there are remote-local edits, preserve them and report
the status before updating. Do not reset or overwrite them. Read-only HTTPS
cloning of this public fork does not require a GitHub token.

## 3. Run the limited read-only inventory

Use **Python 3.10 or newer**. If `python3 --version` is older, activate the
course-provided Python environment according to the UQ instructions first.
The initial inventory needs no installed Python packages and runs no external
commands, GPU work or training. By default it samples at most 120 directory
entries, 20 per directory, and two directory levels; these are not dataset totals.

```bash
python3 inspect_rangpur.py --output runs/rangpur_inventory.json
cat runs/rangpur_inventory.json
```

`runs/` is ignored by Git. This report can contain private file/patient names;
keep it private and share only the access status, Python version, available
command names and de-identified directory layout. No patient names are needed
in the public repository. A detected `srun` or `sbatch` command is a hint about
installed software, not proof of the scheduler's allocation rules.

When the sampled directory tree misses the relevant subfolder, run the same
script with `--data-root` set to that **known** subfolder. Do not compensate by
scanning the entire shared group filesystem.

At this point obtain the accompanying dataset documentation (or ask the tutor)
to confirm all of the following:

- Which files are image slices and which are paired masks.
- Where the real patient/subject identifier comes from, including repeated scans.
- The complete raw-label dictionary: background, prostate, and other structures.
- Whether masks represent the whole prostate or separate regions, plus image
  orientation, slice ordering and spacing where supplied.

File-name patterns and numeric mask values alone cannot establish patient identity
or anatomical label meanings. Stop real-data setup if either remains unknown.
The current code expects one documented nonzero prostate label; separate-zone
masks require an explicitly documented conversion or a loader adjustment first.

After nibabel is available, optional small **explicitly identified** samples can
be inspected with `--nifti PATH_TO_IMAGE` and `--mask PATH_TO_KNOWN_MASK`. Replace
those tokens with real paths. At most three samples are accepted; files above
16 MiB are skipped and mask arrays above 1,048,576 voxels are not read. Raw values
remain anatomically unverified until confirmed against the label dictionary.

## 4. Prepare the course environment and a private manifest

Prefer the course's documented environment, including its GPU-compatible
PyTorch build. The tested environment uses Python 3.10.19 and the versions in
`requirements.txt`. Do not select a CUDA wheel or overwrite a shared environment
without the course's environment instructions.

In your **own** environment, after selecting Python 3.10+ and the documented
PyTorch installation method, install/check the project requirements:

```bash
python -m pip install -r requirements.txt
python -m pip check
python -c "import sys, torch, nibabel; print(sys.version); print('torch:', torch.__version__); print('CUDA build:', torch.version.cuda)"
```

If the course mandates a different PyTorch version, keep that prescribed build,
record the deviation, and ask for an environment-specific dependency command
before running the generic requirements installation. Installing packages may
have a course-approved node/location; follow that rule. Do not use `sudo`.

Create `runs/hipmri_pilot_manifest.csv` using verified patient metadata. Its header is:

```csv
image,mask,patient_id,split
```

Use real absolute image/mask paths (or paths relative to the manifest), a verified
patient ID, and `train`, `val` or `test`. Each patient belongs to exactly one
split, including all slices/repeated scans. The checked-in `manifest.example.csv`
uses fictional names and is **not runnable real data**. For a short pilot, choose
a small number of complete patient cases from the planned train/validation splits;
leave the test split unused. A pilot without test rows is acceptable for midpoint
feasibility, with that limitation recorded.

Record the metadata source, mapping, split seed/selection and preprocessing in
private notes. The code checks the CSV's declared patient isolation; it cannot
prove identities that were entered incorrectly. No automatic manifest is created
until the actual file layout and metadata have been inspected.

## 5. Audit, then run a small baseline on allocated resources

The following Bash prompts deliberately request the **verified raw values**;
there are no assumed label values in this guide. Use background `0` first and
include every documented mask label. The prostate value must be a member of that
list. Enter only the integer values, separated by spaces:

```bash
read -r -p 'Documented raw labels, background 0 first: ' -a HIPMRI_LABEL_VALUES
read -r -p 'Documented raw prostate label: ' HIPMRI_PROSTATE_LABEL
```

An audit reads every image/mask pair in its manifest. Run it only for the small
pilot manifest and in a course-approved compute session, not as a large login-node
job. It checks shape/alignment, raw values and declared patient isolation, and
saves a first-sample overlay for visual inspection:

```bash
python audit_data.py --manifest runs/hipmri_pilot_manifest.csv --label-values "${HIPMRI_LABEL_VALUES[@]}" --prostate-label "$HIPMRI_PROSTATE_LABEL" --output runs/hipmri_pilot_audit.json --sample-output runs/hipmri_pilot_sample.png
```

Review every warning and view the image/mask overlay before training. A successful
audit does not verify the anatomical label meanings or the provenance of IDs.

**GPU allocation is an external step:** obtain the exact interactive allocation
or submission command from the authenticated UQ/course instructions. Supply that
command to the assistant if help adapting it is needed; no partition, GPU type,
account, walltime or memory syntax can be safely filled in before it is known.
Being SSH-connected to Rangpur does not itself mean a GPU is allocated. Do not
run either model on the login node. After allocation, activate the same environment,
return to the project directory, and set the two label variables in that session.

Inside the allocated GPU session, check CUDA, then run two pilot epochs:

```bash
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'STOP: no accessible CUDA device')"
python train.py --model unet --manifest runs/hipmri_pilot_manifest.csv --label-values "${HIPMRI_LABEL_VALUES[@]}" --prostate-label "$HIPMRI_PROSTATE_LABEL" --image-size 128 --base-channels 16 --batch-size 2 --epochs 2 --device cuda --output runs/hipmri_pilot_unet
python predict.py --checkpoint runs/hipmri_pilot_unet/best.pt --manifest runs/hipmri_pilot_manifest.csv --split val --limit 4 --device cuda --output runs/hipmri_pilot_unet/predictions
```

Stop if CUDA is unavailable or the allocation/environment is uncertain. Two
epochs demonstrate data loading and training feasibility; they are not a fair
final accuracy comparison. Keep the generated audit, overlay, `summary.json`,
training curve and predictions private for the tutor discussion. Discuss any
actual early errors without claiming the final Dice target has been achieved.

For this midpoint, running Improved U-Net on real data is optional after the
baseline and audit work. The full, matched-budget comparison belongs to the
subsequent project stage.

## What to bring back for the midpoint

- Login/data access status and the course's exact GPU-allocation instructions.
- A de-identified data-layout description and authoritative patient/label mapping.
- If executed: pilot sample counts, warnings, actual hardware/resources, training
  result and privately viewed image/mask/prediction examples.
- If not executed: explicitly state that real-data access/audit/training remains
  pending; use only the clearly labelled synthetic engineering measurements.

Use the one-page feasibility document for the tutor discussion and update its
status only from observed evidence. Do not upload images, masks, weights, private
manifests or raw inventory reports in the small task commits.
