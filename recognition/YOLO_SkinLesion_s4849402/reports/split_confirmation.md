# Split protocol update

On 2026-10-29 the student reported that using the official ISIC splits had been confirmed. In response to the question of tutor approval versus a supplied patient-ID mapping, the student answered that the ISIC split could be used.

The project therefore supports official train/validation/test membership, checks membership and diagnosis against the official CSVs, and retains exact decoded-image duplicate checks. No patient-ID mapping was supplied and patient identity separation is not claimed to have been programmatically verified. The exported audit records the confirmation as `split_evidence` rather than inventing IDs.

This updates the previously documented unresolved split-policy question. The earlier label/sample audits remain historical records of what metadata was actually inspected.
