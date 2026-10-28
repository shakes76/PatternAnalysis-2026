"""Create training-only figures for the COMP3710 ADNI experiment.
No GPU, no access to held-out Test images, no model selection.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path('runs/phase2_final_analysis')
OUT.mkdir(parents=True, exist_ok=True)

CONFIG = [
    ('CNN baseline', 'runs/baseline_v3_15epochs/metrics.json'),
    ('ConvNeXt V3 scratch', 'runs/convnext_v3_60_newsplit/metrics.json'),
    ('ConvNeXt V4 augmentation', 'runs/convnext_v4_aug20_newsplit/metrics.json'),
    ('ConvNeXt V5 smoothing', 'runs/convnext_v5_smoothing20_newsplit/metrics.json'),
    ('ConvNeXt V6 more slices', 'runs/convnext_v6_12slices20_newsplit/metrics.json'),
    ('ConvNeXt V7 pretrained', 'runs/convnext_v7_pretrained20_newsplit/metrics.json'),
]

histories = {}
summary = []
for name, filename in CONFIG:
    path = Path(filename)
    if not path.is_file():
        print(f'SKIP missing log: {path}', flush=True)
        continue
    history = json.loads(path.read_text(encoding='utf-8'))['history']
    histories[name] = history
    best = max(history, key=lambda r: (
        r['val']['slice']['accuracy'],
        r['val']['slice']['macro_f1'],
        -r['val']['loss']))
    summary.append({
        'experiment': name,
        'epochs_completed': len(history),
        'best_epoch': best['epoch'],
        'val_slice_accuracy': best['val']['slice']['accuracy'],
        'val_macro_f1': best['val']['slice']['macro_f1'],
        'val_patient_accuracy': best['val']['patient']['accuracy'],
        'val_loss_at_best_accuracy': best['val']['loss'],
        'train_slices_per_epoch': best['train']['slices'],
        'val_slices_per_epoch': best['val']['validation_slices'],
    })

if not histories:
    raise RuntimeError('No experiment logs found; run from project folder.')

with (OUT / 'experiment_summary.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)

# 01: Accuracy trajectory across V7 epochs
if 'ConvNeXt V7 pretrained' in histories:
    h = histories['ConvNeXt V7 pretrained']
    x = [r['epoch'] for r in h]
    plt.figure(figsize=(8, 4.7))
    plt.plot(x, [r['train']['accuracy'] * 100 for r in h], label='Train', linewidth=2)
    plt.plot(x, [r['val']['slice']['accuracy'] * 100 for r in h], label='Validation', linewidth=2)
    best = max(h, key=lambda r: (r['val']['slice']['accuracy'], r['val']['slice']['macro_f1'], -r['val']['loss']))
    plt.scatter([best['epoch']], [100 * best['val']['slice']['accuracy']], marker='*', s=180, zorder=5,
                label=f"Selected epoch {best['epoch']}: {100 * best['val']['slice']['accuracy']:.2f}%")
    plt.axhline(80, linestyle='--', linewidth=1, label='80% reference')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)')
    plt.title('V7 pretrained ConvNeXt — train vs validation accuracy')
    plt.grid(alpha=0.22); plt.legend(fontsize=9); plt.tight_layout()
    plt.savefig(OUT / '01_v7_accuracy_curve.png', dpi=180); plt.close()

    plt.figure(figsize=(8, 4.7))
    plt.plot(x, [r['train']['loss'] for r in h], label='Train loss', linewidth=2)
    plt.plot(x, [r['val']['loss'] for r in h], label='Validation loss', linewidth=2)
    plt.axvline(best['epoch'], linestyle='--', linewidth=1, label=f"Selected epoch {best['epoch']}")
    plt.xlabel('Epoch'); plt.ylabel('Label-smoothed cross-entropy loss')
    plt.title('V7 pretrained ConvNeXt — loss curves')
    plt.grid(alpha=0.22); plt.legend(); plt.tight_layout()
    plt.savefig(OUT / '02_v7_loss_curve.png', dpi=180); plt.close()

# 03: Comparable primary metric for all experiments (6 selected slices per scan)
plt.figure(figsize=(9.8, 5))
short = ['CNN', 'V3', 'V4', 'V5', 'V6', 'V7']
order = ['CNN baseline', 'ConvNeXt V3 scratch', 'ConvNeXt V4 augmentation',
         'ConvNeXt V5 smoothing', 'ConvNeXt V6 more slices', 'ConvNeXt V7 pretrained']
rows = [(abbr, histories[name]) for abbr, name in zip(short, order) if name in histories]
vals = [100 * max(h, key=lambda r: (r['val']['slice']['accuracy'],
                                   r['val']['slice']['macro_f1'], -r['val']['loss']))['val']['slice']['accuracy']
        for _, h in rows]
bars = plt.bar([x for x, _ in rows], vals)
for bar, val in zip(bars, vals):
    plt.text(bar.get_x() + bar.get_width() / 2, val + 0.5, f'{val:.2f}%', ha='center', fontsize=9)
plt.axhline(80, linestyle='--', linewidth=1, label='80% reference')
plt.ylim(0, 95); plt.ylabel('Best validation slice accuracy (%)')
plt.title('Model development (same patient split and 6-slice validation protocol)')
plt.legend(); plt.tight_layout()
plt.savefig(OUT / '03_validation_accuracy_comparison.png', dpi=180); plt.close()

# 04: Validation accuracy trajectories
plt.figure(figsize=(9.5, 5.4))
for name in order:
    if name not in histories: continue
    h = histories[name]
    plt.plot([r['epoch'] for r in h], [100 * r['val']['slice']['accuracy'] for r in h],
             label=name, linewidth=1.5)
plt.axhline(80, linestyle='--', linewidth=1)
plt.xlabel('Epoch'); plt.ylabel('Validation slice accuracy (%)')
plt.title('Validation learning curves — all experiments')
plt.ylim(45, 90); plt.grid(alpha=0.22); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT / '04_all_validation_curves.png', dpi=180); plt.close()
print('Saved training figures and experiment_summary.csv to', OUT, flush=True)
