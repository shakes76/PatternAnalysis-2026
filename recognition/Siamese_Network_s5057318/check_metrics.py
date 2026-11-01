"""Hand-calculated tests of the AUROC and PR-AUC functions in predict.py."""
import math

from predict import auroc, average_precision, select_threshold, wilson_interval

failures = 0


def check(name, got, expected, tol=1e-4):
    """Print PASS if `got` is within `tol` of `expected`, else FAIL, and count failures."""
    global failures
    ok = abs(got - expected) < tol
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name} -> {got:.4f} (expected {expected})")


a, y = [0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]
check("AUROC [0.1,0.4,0.35,0.8] / [0,0,1,1]", auroc(a, y), 0.75)
check("PR-AUC [0.1,0.4,0.35,0.8] / [0,0,1,1]", average_precision(a, y), 0.8333)
b = [0.1, 0.2, 0.8, 0.9]
check("AUROC [0.1,0.2,0.8,0.9] / [0,0,1,1]", auroc(b, y), 1.0)
check("PR-AUC [0.1,0.2,0.8,0.9] / [0,0,1,1]", average_precision(b, y), 1.0)
check("AUROC [0.1,0.4,0.35,0.8] / [1,1,0,0]", auroc(a, [1, 1, 0, 0]), 0.25)
check("AUROC all scores equal / [0,1,0,1]", auroc([0.5] * 4, [0, 1, 0, 1]), 0.5)
check("PR-AUC all scores equal / [0,1,0,1] (= prevalence)",
      average_precision([0.5] * 4, [0, 1, 0, 1]), 0.5)
print("AUROC with one class is nan:", math.isnan(auroc([0.1, 0.2], [0, 0])))

# Wilson 95% interval, hand-calculated, tolerance 0.002.
for k, n, (lo_exp, hi_exp) in ((9, 176, (0.0271, 0.0943)), (0, 10, (0.0, 0.2775))):
    lo, hi = wilson_interval(k, n)
    ok = abs(lo - lo_exp) < 0.002 and abs(hi - hi_exp) < 0.002
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}  Wilson({k} of {n}) -> ({lo:.4f}, {hi:.4f}) "
          f"(expected about ({lo_exp}, {hi_exp}))")

# Threshold selection, hand-calculated (rule "balanced" = (a), rule "recall_acc" = (b)).
def check_threshold(name, scores, labels, rule, expected_t, expected=None):
    """Compare select_threshold with a hand-calculated threshold and metrics.

    `expected_t=None` means no threshold should qualify; `expected` maps metric
    names to their expected values (tolerance 1e-4).
    """
    global failures
    t, m = select_threshold(scores, labels, rule)
    if expected_t is None:
        ok = t is None and m is None
        got = "none" if t is None else f"threshold {t}"
    else:
        ok = (t is not None and abs(t - expected_t) < 1e-9
              and all(abs(m[k] - v) < 1e-4 for k, v in expected.items()))
        got = "none" if t is None else f"threshold {t}, " + ", ".join(f"{k} {m[k]:.4f}" for k in expected)
    failures += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name} -> {got} (expected "
          f"{'none' if expected_t is None else f'threshold {expected_t}, {expected}'})")


s10 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
y10 = [0, 0, 0, 0, 0, 1, 0, 1, 1, 1]
check_threshold("rule (a) 4 points", [0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1], "balanced", 0.35,
                {"balanced_acc": 0.75})
check_threshold("rule (a) 10 points", s10, y10, "balanced", 0.6,
                {"recall": 1.0, "precision": 0.8, "accuracy": 0.9, "balanced_acc": 0.9167})
check_threshold("rule (b) 10 points", s10, y10, "recall_acc", 0.6,
                {"recall": 1.0, "precision": 0.8, "accuracy": 0.9, "f1": 0.8889})
check_threshold("rule (b) nothing reaches accuracy 0.80", [0.1, 0.9], [1, 0], "recall_acc", None)

print("\nALL PASSED" if not failures else f"\n{failures} FAILED")
raise SystemExit(1 if failures else 0)
