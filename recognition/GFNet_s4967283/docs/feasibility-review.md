# Feasibility review

[&larr; Back to the main README](../README.md)

---

*This is the one-page review I presented at the midpoint check-off, as Section 3
of the assignment specification requires.*

**Intended user.** A clinician who does early Alzheimer's screening. The
prototype does not replace the judgement of that clinician. It sorts scans into
two groups: cases the model is sure about, and cases it is not. The unsure cases
reach a specialist first.

**Purpose.** Put a 2D T1-weighted brain MRI slice in the AD class or the NC
class, and attach a confidence score that you can trust.

**Acceptance criteria.** I measured all of them on the held-out test split.

| # | Criterion | Target | Outcome |
|---|-----------|--------|---------|
| 1 | GFNet test accuracy, AD against NC | ≥ 0.80 | **Not met.** See Limitations |
| 2 | Baseline CNN trained on identical splits | reported | Met |
| 3 | Expected Calibration Error for both models | reported | Met |
| 4 | Reject-option rule that reaches ≥ 0.95 retained accuracy | ≤ 30% referral | **Not achievable.** See Calibration |
| 5 | Peak GPU memory and training time on one A100 | ≤ 16 GB, ≤ 2 h | Met. 2.32 GB, 18 min |

**Model choice.** GFNet (Rao et al., 2023) is a Hard-difficulty option. The
choice builds on the Fourier material in this course and on my own Demo 2 Part 1
work. There I measured a naive DFT at 34.2 s against an FFT at 0.13 ms for
N = 4096. The argument for GFNet against attention is the same argument one
level up.

**Baseline.** A plain four-block CNN written for this project, with 421,474
parameters. I did not tune it hard, so it stays an honest reference point and
not a weak target.

**Risks I found, and what happened.**

1. *Complexity of the GFNet filter block.* Closed by the FFT identity
   self-test, before I spent any GPU time.
2. *Cluster environment.* Confirmed to work. torch 2.14.0+cu130 on an
   A100-PCIE-40GB.
3. *Overfit on 915 training patients.* This risk happened. See the overfit
   section below for the diagnosis and the fix.

---
