"""Exploratory validation-only risk-coverage diagnostics; no test tuning.

Runs a single validation inference using the already locked ConvNeXt V7 checkpoint.
Reports the predeclared 90%-accuracy / >=20%-coverage failure, and the
highest attainable accepted-case validation accuracy within that coverage
constraint. NOT a deployable clinical triage rule.
"""
import json
import numpy as np
from pathlib import Path
from review_policy import eval_prob_on_validation, summary


def main():
    y, p = eval_prob_on_validation()
    grid = [summary(y, p, float(t)) for t in np.linspace(0.50, 0.99, 50)]
    candidates = [r for r in grid if r['accepted'] and r['coverage'] >= 0.20]
    best = max(candidates, key=lambda x: (x['accepted_accuracy'], x['coverage'])) if candidates else None
    fixed = [summary(y, p, t) for t in (0.50, 0.70, 0.80, 0.90, 0.95, 0.99)]
    meets = [r for r in candidates if r['accepted_accuracy'] >= 0.90]
    out = {
        'dataset': 'validation only; full 4860 slices; 102 patients',
        'preset_goal_accuracy': 0.90,
        'preset_min_coverage': 0.20,
        'passes_preset_goal': bool(meets),
        'selected_operational_threshold': None,
        'best_at_least_20pct_coverage_exploratory': best,
        'predeclared_threshold_examples': fixed,
        'warning': 'Exploratory slice-level diagnostics; no post-test tuning and no clinical recommendation.'
    }
    path=Path('runs/phase4_benchmark/review_rule_diagnostics.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,indent=2),flush=True)
    print('Saved:',path)


if __name__ == '__main__':
    main()
