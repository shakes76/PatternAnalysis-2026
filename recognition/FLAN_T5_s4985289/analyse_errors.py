"""Create a manual clinical-audit worksheet from *real* test predictions."""
import argparse
import csv
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--predictions', required=True, help='test predictions.jsonl from predict.py')
    p.add_argument('--output', default='outputs/clinical_audit.csv')
    p.add_argument('--cases', type=int, default=5)
    a = p.parse_args()
    with open(a.predictions) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError('No predictions')
    # Diverse positions, not a claim that these are confirmed failures.
    indices = sorted(set(round(i * (len(rows) - 1) / max(1, a.cases - 1))
                         for i in range(min(a.cases, len(rows)))))
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['case_index', 'source', 'reference', 'prediction',
                                          'hallucination_evidence', 'critical_omission_evidence',
                                          'certainty_change', 'readability', 'risk_level',
                                          'suggested_guardrail'])
        w.writeheader()
        for i in indices:
            row = rows[i]
            w.writerow({'case_index': i, 'source': row['source'],
                        'reference': row['reference'], 'prediction': row['prediction']})
    print(f'Created {out}: inspect each case manually and record exact evidence.')


if __name__ == '__main__':
    main()
