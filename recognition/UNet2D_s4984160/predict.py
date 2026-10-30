"""Paired fixed-case figures, with optional direct frozen-checkpoint inference.

`python predict.py evaluate <evaluate.py arguments>` loads the checkpoint via
the evaluator, infers once, then plots. `render --input-dir ... --output-dir ...`
plots already saved inference arrays without another inference pass.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np


def render(image, target, pred2d, pred3d, meta, path):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.colors import ListedColormap
    valid = pred2d != 255
    z = np.flatnonzero((target == 5).any(axis=(1,2)))
    if not len(z):
        raise ValueError('Fixed case has no GT prostate')
    planes = [('axial', int(z[0]), 'inferior GT extent proxy'),
              ('axial', int(z[len(z)//2]), 'mid-gland'),
              ('axial', int(z[-1]), 'superior GT extent proxy'),
              ('coronal', 64, 'shared coronal plane')]
    if meta['original_axcodes'][2] != 'S':
        planes[0] = ('axial', int(z[0]), 'low-array GT extent')
        planes[2] = ('axial', int(z[-1]), 'high-array GT extent')
    fig, axes = plt.subplots(4,5,figsize=(19,14),constrained_layout=True)
    low, high = np.percentile(image,[1,99])
    colors=ListedColormap(['cyan','orange','magenta'])
    error_colors=ListedColormap(['lime','red','dodgerblue'])
    for row,(kind,index,name) in enumerate(planes):
        def plane(array):
            return array[index].T if kind == 'axial' else array[:,:,index]
        x,gt,p2,p3,roi = (plane(a) for a in (image,target,pred2d,pred3d,valid))
        origin='upper' if kind == 'axial' else 'lower'
        errors=np.full(gt.shape,-1,dtype=np.int8)
        truth,pred=gt==5,p2==5
        errors[roi & truth & pred]=0
        errors[roi & ~truth & pred]=1
        errors[roi & truth & ~pred]=2
        for col,title in enumerate(['MRI','GT','3D prediction','2D prediction','2D TP / FP / FN']):
            ax=axes[row,col]
            ax.imshow(x,cmap='gray',vmin=low,vmax=high,origin=origin)
            if col in (1,2,3):
                labels={1:gt,2:p3,3:p2}[col]
                overlay=np.ma.masked_where(~(roi & np.isin(labels,[3,4,5])),labels.astype(int)-3)
                ax.imshow(overlay,cmap=colors,vmin=0,vmax=2,alpha=.55,origin=origin)
            if col==4:
                ax.imshow(np.ma.masked_less(errors,0),cmap=error_colors,
                          vmin=0,vmax=2,alpha=.7,origin=origin)
            ax.imshow(np.ma.masked_where(roi,np.ones(roi.shape)),
                      cmap=ListedColormap(['gray']),vmin=0,vmax=1,alpha=.45,origin=origin)
            ax.set_title(f'{title}\n{name}; {"D" if kind=="axial" else "W"}={index}',fontsize=9)
            ax.axis('off')
    fig.suptitle(f'{meta["case"]} | {meta.get("split", "held-out test")} | paired shared-grid view\n'
                 f'Native 2D Dice={meta["native_2d_dice"]:.4f}; '
                 f'shared 2D Dice={meta["shared_2d_dice"]:.4f}; '
                 f'shared 3D Dice={meta["shared_3d_dice"]:.4f}\n'
                 'Cyan: bladder; orange: rectum; magenta: prostate. '
                 'Green/red/blue: prostate TP/FP/FN.\n'
                 'Gray shading excludes the unobserved band; GT extents are proxies. '
                 'Both predictions are from frozen checkpoints.',fontsize=11)
    fig.savefig(path,dpi=140); plt.close(fig)


def render_all(input_dir, output_dir):
    inputs, output=Path(input_dir),Path(output_dir)
    manifest=json.loads((inputs/'manifest.json').read_text(encoding='utf-8'))
    output.mkdir(parents=True,exist_ok=False)
    figures=[]
    for row in manifest['cases']:
        with np.load(inputs/row['cache_file'],allow_pickle=False) as data:
            meta=json.loads(str(data['metadata'].item()))
            name=row['case'].replace('.nii.gz','.png')
            render(data['image'],data['target'],data['prediction_2d'],
                   data['prediction_3d'],meta,output/name)
        figures.append(dict(row,figure=name))
        print(f'Saved {output/name}',flush=True)
    (output/'manifest.json').write_text(json.dumps(dict(figures=figures),indent=2)+'\n')
    (output/'plots_complete.txt').write_text('Paired prediction figures completed.\n')


def main():
    if len(sys.argv)>1 and sys.argv[1]=='evaluate':
        from evaluate import main as evaluate
        out=evaluate(sys.argv[2:])
        render_all(out/'prediction_inputs',out/'figures')
        return
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['render'])
    p.add_argument('--input-dir',required=True)
    p.add_argument('--output-dir',required=True)
    args=p.parse_args()
    render_all(args.input_dir,args.output_dir)


if __name__=='__main__':
    main()
