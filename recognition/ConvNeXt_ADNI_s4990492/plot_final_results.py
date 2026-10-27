"""Generate aggregate report figures from frozen evaluations; no inference."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    base=Path(__file__).resolve().parent
    cnn=base/'runs/baseline_20261022_211305_056381'
    conv=base/'runs/local_rgb224_lr3e6_imagenet_20261026_201343_178908'
    a=json.loads((cnn/'final_baseline_evaluation/metrics.json').read_text())['test']
    b=json.loads((conv/'final_test_evaluation/metrics.json').read_text())['primary_threshold_0_50']
    out=base/'figures'
    out.mkdir(exist_ok=True)
    names=['SimpleCNN','ConvNeXt (ImageNet)']
    fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
    for ax,name,m in zip(axes,names,[a,b]):
        cm=np.array(m['confusion_matrix'])
        assert cm.sum()==4580
        ax.imshow(cm,cmap='Blues',vmin=0,vmax=2520)
        for i in range(2):
            for j in range(2): ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>1260 else 'black',fontsize=15)
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['NC','AD'],yticklabels=['NC','AD'],xlabel='Predicted',ylabel='True',title=f'{name}\nTest accuracy {m["accuracy"]:.2%}')
    fig.savefig(out/'final_test_confusion.png',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5),layout='constrained')
    identity=[]
    for name,path,m in [(names[0],cnn/'final_baseline_evaluation/test_predictions.csv',a),(names[1],conv/'final_test_evaluation/predictions.csv',b)]:
        with path.open(newline='') as f: rows=list(csv.DictReader(f))
        identity.append([(int(r['row_index']),int(r['target'])) for r in rows])
        y=np.array([int(r['target']) for r in rows]); p=np.array([float(r['prob_ad']) for r in rows])
        order=np.argsort(-p); y,p=y[order],p[order]
        ends=np.r_[p[:-1]!=p[1:],True]
        tpr=np.r_[0,np.cumsum(y==1)[ends]/sum(y==1)]
        fpr=np.r_[0,np.cumsum(y==0)[ends]/sum(y==0)]
        auc=float(np.trapezoid(tpr,fpr))
        assert abs(auc-m['auroc_ad_positive'])<1e-10
        ax.plot(fpr,tpr,label=f'{name}: AUROC {auc:.4f}')
    assert identity[0]==identity[1]
    ax.plot([0,1],[0,1],'--',color='gray')
    ax.set(xlabel='False positive rate (NC)',ylabel='True positive rate (AD)',title='Held-out slice-level ROC',xlim=(0,1),ylim=(0,1))
    ax.legend(loc='lower right')
    fig.savefig(out/'final_test_roc.png',dpi=180)
    plt.close(fig)
    profiles=[json.loads((cnn/'validation_baseline_profile/summary.json').read_text()),json.loads((conv/'validation_rgb224_profile/summary.json').read_text())]
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    fields=[('Batch-1 forward (ms)',[p['forward_only'][0]['median_forward_batch_ms'] for p in profiles]),('Batch-8 peak allocated (MiB)',[p['forward_only'][1]['peak_allocated_mib'] for p in profiles]),('Test accuracy (%)',[a['accuracy']*100,b['accuracy']*100])]
    for ax,(label,values) in zip(axes,fields):
        bars=ax.bar(['SimpleCNN','ConvNeXt'],values,color=['#567a99','#c46c38'])
        ax.set_title(label)
        ax.set_ylim(0,max(values)*1.2)
        ax.bar_label(bars,labels=[f'{v:.2f}' for v in values],padding=3)
    fig.suptitle('Observed performance and resource trade-off (separate profiling sessions)')
    fig.savefig(out/'final_resource_comparison.png',dpi=180)
    plt.close(fig)
    print('PASS: test row alignment, confusion totals and AUROC reproduction. Saved three aggregate figures.')


if __name__=='__main__':
    main()
