"""Draw report figures from frozen summaries and photos; no model inference."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
from PIL import Image


def build(photos):
    code = Path(__file__).resolve().parent
    assets = code/'assets'
    assets.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    provenance = {'new_inference': False, 'selection_changed': False, 'photos': {}, 'figures': {}}
    fig, ax = plt.subplots(figsize=(11, 2.05))
    ax.set(xlim=(0, 11), ylim=(0, 2.05)); ax.axis('off')
    labels = ['Image\nletterbox 640', 'Backbone\nimage features', 'Neck\nmultiscale fusion',
              'Detection head\nboxes + classes', 'NMS + score\nfiltering', 'Retained boxes\nand class scores']
    for i, label in enumerate(labels):
        x = 0.08 + i*1.84
        ax.add_patch(Rectangle((x, 0.65), 1.58, 0.9, facecolor='#f2f2f2', edgecolor='black', linewidth=.8))
        ax.text(x+.79, 1.1, label, ha='center', va='center', fontsize=12)
        if i < 5: ax.annotate('', xy=(x+1.81, 1.1), xytext=(x+1.6, 1.1), arrowprops={'arrowstyle': '->', 'lw': .8})
    ax.text(5.5, .2, 'Ultralytics YOLOv8 detector; one network predicts location and class', ha='center', fontsize=11)
    fig.tight_layout(pad=.2); fig.savefig(assets/'report_pipeline.png', dpi=160); plt.close(fig)

    summary = json.loads((code/'known_optimization_results_20261021.json').read_text())
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    for key, label, style, color in [('baseline_n', 'Original n', '--', '0.5'), ('n_geometry', 'n + light geometry', '-', '0.0')]:
        report = summary['reports'][key]; rows = report['grid']; selected = report['selected']
        ax.plot([100*r['image_alerts']['benign_fp']/124 for r in rows],
                [100*r['classes']['Malignant']['recall'] for r in rows], style, color=color,
                marker='.', markersize=4, lw=1.1, label=label)
        ax.scatter(100*selected['image_alerts']['benign_fp']/124,
                   100*selected['classes']['Malignant']['recall'], s=70, facecolors='white', edgecolors=color, zorder=4)
    ax.axvline(10, color='0.4', linestyle=':', lw=1, label='10% validation alert budget')
    ax.set(xlabel='Benign images with a malignant alert (%)', ylabel='Malignant detection recall (%)', xlim=(-1, 50), ylim=(0, 75))
    ax.grid(alpha=.18); ax.legend(frameon=False, fontsize=9, loc='lower right')
    fig.tight_layout(); fig.savefig(assets/'report_threshold_tradeoff.png', dpi=160); plt.close(fig)

    final = json.loads((code/'final_candidate_reassessment_20261022.json').read_text())
    for number, case in enumerate(final['cases'][:3], 1):
        paths = list(photos.glob(case['image_id']+'.*'))
        if len(paths) != 1: raise ValueError('Expected one original photo for '+case['image_id'])
        photo = paths[0]; provenance['photos'][photo.name] = hashlib.sha256(photo.read_bytes()).hexdigest()
        with Image.open(photo) as source: im = source.convert('RGB')
        w, h = im.size
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.65))
        def draw(ax, box, style):
            x1, y1, x2, y2 = box
            patch = Rectangle((x1*w, y1*h), (x2-x1)*w, (y2-y1)*h,
                              fill=False, edgecolor='black', lw=1.35, linestyle=style)
            patch.set_path_effects([pe.Stroke(linewidth=3.0, foreground='white'), pe.Normal()])
            ax.add_patch(patch)
        for ax in axes: ax.imshow(im); ax.axis('off')
        gt = 'Malignant' if case['class_id'] else 'Benign'
        axes[0].set_title(f"{case['image_id']} | GT: {gt}", fontsize=10)
        draw(axes[0], case['gt_box'], '--')
        labels=[]
        for pred in case['retained']:
            assert pred['confidence'] > final['threshold']
            label = 'M' if pred['class_id'] else 'B'
            labels.append(f"{label} {pred['confidence']:.3f}, IoU {pred['iou']:.3f}")
            draw(axes[1], pred['box'], '-.' if pred['class_id'] else '-')
        axes[1].set_title('Predictions: '+ '; '.join(labels), fontsize=10)
        fig.tight_layout(pad=.6)
        name=f'report_final_case_{number}.png';fig.savefig(assets/name,dpi=150);plt.close(fig)
    for name in ['report_pipeline.png', 'report_threshold_tradeoff.png', *[f'report_final_case_{i}.png' for i in range(1,4)]]:
        provenance['figures'][name]=hashlib.sha256((assets/name).read_bytes()).hexdigest()
    provenance['summary_sha256']=hashlib.sha256((code/'final_candidate_reassessment_20261022.json').read_bytes()).hexdigest()
    provenance['threshold_summary_sha256']=hashlib.sha256((code/'known_optimization_results_20261021.json').read_bytes()).hexdigest()
    provenance['builder_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (code/'report_figure_provenance_20261025.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Created pipeline, two-model trade-off and three frozen-prediction case figures')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--photos',type=Path,required=True)
    build(p.parse_args().photos)
