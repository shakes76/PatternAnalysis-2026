# R6b: private visual case review

The case export keeps the nine-case selection fixed: three highly confident AD misses, three highly confident NC false alarms, and three correct patients closest to the 0.5 boundary. It reads the accepted test prediction JSON and split manifest; it does not tune or evaluate a new model.

On the cluster login node, from `/home/Student/s4982104/comp3710-code/ConvNeXt_ADNI_s4982104`, run:

```bash
/home/Student/s4982104/comp3710-env/bin/python run_r6_case_review.py \
  --evaluation /home/Student/s4982104/comp3710-data/adni/r6_final_708475/run/evaluation.json \
  --manifest /home/Student/s4982104/comp3710-data/adni/split_seed42_60_20_20/split_samples.csv \
  --output /home/Student/s4982104/comp3710-data/adni/r6_case_review_708475
```

It creates nine PNG contact sheets, a private `case_index.csv` and `REVIEW_TEMPLATE.md`. Each sheet shows all 20 source JPEG slices in filename order; numbered tiles preserve that order. Move the directory into `final_report/adni_manifest_local/r6_final/r6b_analysis/case_review/` on the local computer. Keep the case index and images private; do not commit them.

For each case, inspect the 20 tiles and fill the template with observed blank/edge slices, crop, motion or other visible artifacts. Record which tile numbers support each observation. Compare the prediction probability and three seed probabilities in the existing private case index. An appearance cannot establish an Alzheimer's diagnosis or prove a causal explanation for an error. If no relevant defect is visible, say so. Preserve the original 0.5 and supplemental decisions.

Acceptance: nine aliases match `anonymous_cases.json`; each has exactly 20 numbered tiles and the selected patient/scan/label match the frozen split manifest; no missing source files; reviewer observations cite tile numbers and distinguish observation from hypothesis.
