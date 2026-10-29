"""Plain grayscale figures, drawn only from the validation comparison JSON."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
S=Path(__file__).resolve().parent
r=json.loads((S/'results.json').read_text())
assert r['status']=='completed','Wait for all predeclared candidates'
out=S/'assets';out.mkdir(exist_ok=True)
labels={'baseline_n':'Original n','baseline_s':'Original s','n_cls1':'n: cls=1.0',
        'n_geometry':'n: light geometry','n_long120':'n: 120 epochs'}
order=list(labels)
plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(8.5,4.6))
styles=[('black','o','-'),('.65','v',':'),('.25','s','--'),('.45','^','-.'),('.1','D',':')]
for n,(c,m,l) in zip(order,styles):
    d=r['reports'][n]
    x=[p['image_alerts']['benign_fp'] for p in d['grid']];y=[p['classes']['Malignant']['tp'] for p in d['grid']]
    ax.plot(x,y,color=c,marker=m,markersize=3,linestyle=l,label=labels[n],linewidth=1.2)
    p=d['selected']
    if p:ax.plot(p['image_alerts']['benign_fp'],p['classes']['Malignant']['tp'],marker=m,markersize=8,markerfacecolor='white',markeredgecolor=c,markeredgewidth=1.7)
ax.axvline(12.4,color='.3',linestyle='--',linewidth=.8)
ax.text(13.5,39,'10% budget',fontsize=8,va='top')
ax.set(xlabel='Benign images with a malignant alert (out of 124)',ylabel='Malignant detection TP (out of 41)',ylim=(-1,42),xlim=(-1,85))
ax.grid(alpha=.18);ax.legend(loc='lower right',frameon=False)
ax.set_title('Validation trade-off; outlined markers are selected operating points')
fig.tight_layout();fig.savefig(out/'threshold_tradeoff.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(8.5,3.8));x=np.arange(len(order));w=.35
for offset,name,color,hatch in [(-w/2,'Benign','.8',''),(w/2,'Malignant','.4','///')]:
    vals=[r['reports'][n]['selected']['classes'][name]['iou']['all_gt_class_correct']['mean'] for n in order]
    bars=ax.bar(x+offset,vals,w,label=f'{name} (N={124 if name=="Benign" else 41})',color=color,edgecolor='black',linewidth=.5,hatch=hatch)
    ax.bar_label(bars,fmt='%.3f',fontsize=8,padding=2)
ax.set_xticks(x,[labels[n] for n in order]);ax.set_ylim(0,1)
ax.set_ylabel('Mean IoU: all GT, correct class; missing = 0')
ax.legend(frameon=False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
fig.tight_layout();fig.savefig(out/'all_gt_iou.png',dpi=180);plt.close(fig)
print(out)
