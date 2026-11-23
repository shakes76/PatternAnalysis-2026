# Step 9 — predeclared confidence and referral method

This is a validation-development investigation, not another training experiment or
clinical deployment policy. Preserve B0, B1 and C1. Keep the preferred CNN from the
Step 8 patient macro-F1/log-loss selection rule. For the supplied results this is
B1, the regularized CNN. ConvNeXt remains the main Hard-Difficulty architecture.

## Inputs and safeguards

Use only the selected checkpoint's already-saved validation probabilities. Verify
current model/training source hashes, checkpoint hashes, all finalized manifest
hashes, original validation metrics, and the slice-to-scan-to-patient aggregation.
No neural-network training, forward passes, MRI reads or final-test predictions are
performed. Reading test CSV metadata to verify disjoint patient groups is not
final-test evaluation. All historic run directories are read-only.

The available aggregate Step 8 ZIP is insufficient to calculate calibration or
confidence-error relationships. The new helper therefore runs locally against
`PRIVATE_validation_predictions.json`; those records stay private.

## Calibration and diagnostic classification

Let p be a patient's mean of scan means of slice P(AD). Clip p to [1e-7,1-1e-7],
compute z=log(p/(1-p)), and transform to q=sigmoid(z/T). Fit one scalar T in
[0.5,10] by minimizing binary negative log likelihood; no intercept is fitted.
The minimization is convex in inverse temperature and uses deterministic bisection
with endpoint and identity checks. The range is a project choice, not a standard.

Keep the AD-versus-NC diagnosis cutoff at 0.5. A fixed positive T preserves these
class decisions and the confidence ordering (subject to numerical clipping/ties).
It cannot improve discrimination, fix misclassified scans, or turn accepted-subset
accuracy into the required overall test accuracy. Report NLL, Brier score and
five-equal-width-bin ECE for AD probability. ECE is bin-sensitive, especially with
102 patients. Record full-fit/in-sample and cross-fitted diagnostics separately.

## Cross-fitting, and its important limitation

Use five class-stratified patient folds with fixed seed 42. Within each class,
sort by SHA256 of the documented seed/class/subject string and allocate by rank
modulo five. Both models use the same folds. On four folds fit T and the referral
cutoff, then apply them once to the remaining fold. Pool the held-fold outcomes.
The final candidate T and cutoff are fitted using all validation patients.

These patients already influenced model checkpoint selection. Therefore this
cross-fitting is NOT an independent estimate of model generalisation and does not
remove model-selection optimism. It only avoids using a held-fold patient's label
in the temperature/cutoff fitting for that fold. Different fold temperatures can
change the pooled probability ordering, so no pooled cross-fit AUROC is reported.
The eventual untouched test must evaluate the frozen complete procedure.

## Confidence and referral

Confidence is max(q,1-q). Accept the existing 0.5-threshold diagnosis if confidence
>= tau, otherwise refer for review. Predeclared tau grid:
0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99.

Select maximum development coverage subject to all of:

* At least max(20,ceil(25% of fit patients)) accepted.
* Observed error among accepted patients <=10%.
* No more than 10% of ALL AD patients automatically labelled NC.

Break coverage ties by the lower cutoff. If no candidate meets all constraints,
refer everyone. Do not force a favourable cutoff or relax constraints after
seeing results. These are exploratory engineering targets, not clinical safety
standards, and no human reviewer performance is assumed.

Report coverage, referral burden, accepted error, the accepted confusion matrix,
AD automatically cleared, AD automatically detected and AD referred. If no cases
are accepted, accepted accuracy/error are not estimable, not 100%/0%.
'AD not automatically cleared' includes referrals and is NOT diagnostic recall.
Descriptive 95% Wilson intervals do not adjust for selection or cross-fit
correlations. High accepted-subset accuracy is NOT all-patient test accuracy.

## Scope and outputs

Analyse confidence versus mistakes and raw risk-coverage curves for both selected
models. Do not claim temperature improved the risk-coverage ordering. Save full-fit
candidate rules with selected checkpoint and manifest hashes for later review.
Keep private subject-linked records out of the ZIP, Git and cloud-synced outputs.
Use only the allowlisted aggregate results and plots for sharing/documentation.
A completed software run does not establish safe automation, attainment of 0.80,
or permission to inspect the final test before the evaluation plan is frozen.

## Source and disclosure

Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017,
https://arxiv.org/abs/1706.04599 . This references the method; no current web
verification is claimed. Implementation uses the previously installed NumPy,
scikit-learn and Matplotlib. The model implementation in `modules.py` is unchanged.
ChatGPT assisted with code, synthetic tests and documentation. Review it, run the
local checks and record the actual assistance and verification in the mandatory
Artificial Intelligence Usage Disclosure. Do not invent clinical verification.
