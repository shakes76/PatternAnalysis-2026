# Initial-noise interpolation — 7 October 2026

Freeze checkpoint step20000, aligned scheduler precision,256, DDIM30,eta0,CFG7.5.
Use endpoint seed pairs37200->37201 and37216->37217, alpha0,.25,.5,.75,1 for each
AD/NC prompt:20 images. Prompt remains fixed along each path. No real/test image
is encoded or read. This is diffusion initial-noise interpolation, not interpolation
between disease states, real-patient embeddings, or textual conditions.

Use z(alpha)=(1-alpha)z0+alpha*z1, preserving exact endpoint tensors. Interior
noise variance is lower than at endpoints; record initial norms and disclose this
confound. Smooth output variation cannot prove useful semantic coverage, absence
of memorization, anatomical validity, or AD/NC class fidelity.

Before interpretation verify20 unique outputs, all file hashes, endpoint noise
and PNG parity against the accepted64-image CFG7.5 cohort (eight endpoints).
Inspect all four paths including failures, not selected pleasant transitions.
Do not claim completion of feature-space evaluation from this diagnostic.
