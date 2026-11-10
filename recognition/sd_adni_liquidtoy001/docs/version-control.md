# Local Development and Single-PR Workflow

Decision recorded: 2026-10-02. Keep project development local until final publication. Create one submission PR, then respond to review on that same PR.

## Separate operations

- `git commit` records a local milestone. It does not publish the work.
- `git push` publishes commits to the selected remote. The existing public fork is publicly readable before a PR is opened or merged.
- A pull request proposes merging the fork's branch into the course branch. Multiple commits can be submitted through one PR.

## Daily development

Continue on the existing `topic-recognition` branch, which descends from the course's actual topic branch. Keep changes inside `recognition/sd_adni_liquidtoy001/`. Inspect diffs, stage explicit paths, and commit genuine milestones. Do not squash the entire project into one submission commit or fabricate a retrospective history.

Do not push development commits to the public fork before the student explicitly requests final publication. The repository-local `push.default=nothing` setting prevents a bare `git push`; it does not block an explicit remote/branch push or all GUI actions. Review any sync/publish action before using it.

## Backup and Rangpur transfer

Use local Git bundles as snapshots of committed history, stored outside the repository on a second drive. A bundle does not contain uncommitted or ignored files. Verify each bundle with `git bundle verify`; maintain a separate backup of valuable experiment artifacts.

For Rangpur, transfer committed source or a verified Git bundle directly over SSH. Do not publish the development branch just to allow the cluster to clone it. Keep the local checkout as the source of truth and bring back measured results before committing the associated experiment record.

## Final publication

1. Complete the report and reproduction checks; inspect the entire diff and history for datasets, weights, credentials, and unrelated files.
2. Fetch the course branch and inspect upstream changes. Preserve genuine project commits while resolving any necessary integration changes.
3. With the student's explicit publication instruction, push the completed `topic-recognition` history to the existing public fork.
4. Open one PR from `liquidtoy001:topic-recognition` to `shakes76:topic-recognition` before the confirmed deadline. Submit the README PDF to Turnitin as required.
5. Address teaching-team feedback with new commits on the same branch and PR. One PR does not mean only one push or one commit.

The project becomes public at step 3, not at PR merge. Creating a private duplicate is optional for private remote backup, but is unnecessary for this local-first workflow. If used later, preserve common Git history and publish through the existing fork for the course PR.
