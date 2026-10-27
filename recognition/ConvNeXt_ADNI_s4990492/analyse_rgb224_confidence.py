"""CPU-only diagnostics from saved validation predictions; never reads test data."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def rate(a, b):
    return float(a / b) if b else None


def metrics(y, pred):
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    return dict(images=len(y), accuracy=rate(tn + tp, len(y)),
                nc_recall=rate(tn, tn + fp), ad_recall=rate(tp, tp + fn),
                nc_support=tn + fp, ad_support=tp + fn,
                nc_predicted_ad=fp, ad_predicted_nc=fn)


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=base / 'runs/local_rgb224_lr3e6_imagenet_20261026_201343_178908')
    args = parser.parse_args()
    source = args.run / 'validation_rgb224_evaluation'
    summary = json.loads((source / 'metrics.json').read_text(encoding='utf-8'))
    manifest = base.parents[1] / 'local_data/splits_local/val.csv'
    if summary['split'] != 'validation' or hashlib.sha256(manifest.read_bytes()).hexdigest() != summary['manifest_sha256']:
        raise ValueError('Validation identity mismatch.')
    with manifest.open(newline='', encoding='utf-8-sig') as f:
        records = list(csv.DictReader(f))
    with (source / 'predictions.csv').open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    y = np.array([int(r['target']) for r in rows])
    pred = np.array([int(r['prediction']) for r in rows])
    prob = np.array([float(r['prob_ad']) for r in rows])
    confidence = np.array([float(r['confidence']) for r in rows])
    if len(rows) != len(records) or len(rows) != summary['images'] or not rows:
        raise ValueError('Prediction count mismatch.')
    if [int(r['row_index']) for r in rows] != list(range(len(rows))) or not np.array_equal(y, [int(r['target']) for r in records]):
        raise ValueError('Prediction order or labels mismatch.')
    if not np.isin(y, [0, 1]).all() or not np.isin(pred, [0, 1]).all() or not np.isfinite(prob).all() or not ((prob >= 0) & (prob <= 1)).all():
        raise ValueError('Invalid probabilities or labels.')
    if not np.array_equal(pred, (prob > 0.5).astype(int)) or not np.allclose(confidence, np.maximum(prob, 1-prob), atol=1e-6, rtol=0):
        raise ValueError('Saved decisions/confidence do not match probabilities.')
    correct = y == pred
    if abs(float(correct.mean()) - summary['accuracy']) > 1e-10:
        raise ValueError('Summary accuracy mismatch.')
    bins = []
    for i in range(10):
        lo, hi = i/10, (i+1)/10
        mask = (confidence >= lo) & ((confidence < hi) if i < 9 else (confidence <= hi))
        bins.append(dict(lower=lo, upper=hi, images=int(mask.sum()),
                         accuracy=float(correct[mask].mean()) if mask.any() else None,
                         mean_confidence=float(confidence[mask].mean()) if mask.any() else None))
    referral = []
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]:
        keep = confidence >= threshold
        m = metrics(y[keep], pred[keep])
        referral.append(dict(confidence_threshold=threshold, coverage=float(keep.mean()),
                             referred=int((~keep).sum()), retained_error_rate=1-m['accuracy'] if m['accuracy'] is not None else None, **m))
    thresholds = [dict(ad_threshold=i/100, **metrics(y, (prob > i/100).astype(int))) for i in range(10,91)]
    best = max(thresholds, key=lambda r: (r['accuracy'], -abs(r['ad_threshold']-0.5), -r['ad_threshold']))
    ece = sum(b['images']/len(y)*abs(b['accuracy']-b['mean_confidence']) for b in bins if b['images'])
    report = dict(split='validation', evaluation_unit='slice', checkpoint_epoch=summary['checkpoint_epoch'],
                  manifest_sha256=summary['manifest_sha256'], checkpoint_sha256=summary['checkpoint_sha256'],
                  baseline=metrics(y,pred), exploratory_best_grid=best,
                  grid_definition='0.10 to 0.90 inclusive, step 0.01; rank accuracy, then proximity to 0.5, then lower threshold',
                  brier_score=float(np.mean((prob-y)**2)), ece_10_equal_width_bins=ece,
                  probabilities_calibrated=False,
                  interpretation='Exploratory validation diagnostics, not test results. Referral accuracy requires coverage; AD thresholds classify every slice. No operating threshold is selected automatically.',
                  threshold_rule='AD if prob_ad > threshold; otherwise NC. Referral retains confidence >= threshold.')
    output = args.run / 'validation_rgb224_confidence'
    output.mkdir(exist_ok=False)
    write_csv(output/'confidence_bins.csv', bins)
    write_csv(output/'referral_thresholds.csv', referral)
    write_csv(output/'ad_thresholds.csv', thresholds)
    (output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    fig, axes = plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    used = [b for b in bins if b['images']]
    axes[0].plot([0,1],[0,1],'--',color='gray')
    axes[0].plot([b['mean_confidence'] for b in used],[b['accuracy'] for b in used],'o-')
    axes[0].set(xlabel='Mean confidence',ylabel='Observed accuracy',title='Validation reliability',xlim=(0,1),ylim=(0,1))
    valid = [r for r in referral if r['images']]
    axes[1].plot([r['coverage'] for r in valid],[r['retained_error_rate'] for r in valid],'o-')
    axes[1].set(xlabel='Retained fraction (coverage)',ylabel='Retained error rate',title='Referral: fixed original predictions',xlim=(0,1),ylim=(0,1))
    for key in ['accuracy','nc_recall','ad_recall']:
        axes[2].plot([r['ad_threshold'] for r in thresholds],[r[key] for r in thresholds],label=key)
    axes[2].legend()
    axes[2].set(xlabel='AD probability threshold',ylabel='Fraction',title='All validation slices',ylim=(0,1))
    fig.savefig(output/'confidence_analysis.png',dpi=160)
    plt.close(fig)
    print(json.dumps(report,indent=2,allow_nan=False))
    print('Referral thresholds:',json.dumps(referral,allow_nan=False))
    print('Saved:',output)


if __name__ == '__main__':
    main()
