"""R5d quantitative grid: retained error risk versus coverage and remaining AD misses.

All three families and all four prespecified rates are shown; lines are guides
between measured operating points. No human outcomes or statistical intervals
are inferred. Python/matplotlib; 183-mm vector exports plus 600-dpi PNG preview.
"""
import csv
from pathlib import Path


def render(metrics_file,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from run_r5_ensemble import MODELS,require
    with Path(metrics_file).open(newline='') as stream:rows=list(csv.DictReader(stream))
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':8,'svg.fonttype':'none','pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    fig,axes=plt.subplots(1,2,figsize=(7.2047244,3.6),layout='constrained')
    colors=['#586875','#0072B2','#D55E00'];markers=['o','s','^']
    for model,color,marker in zip(MODELS,colors,markers):
        group=sorted([r for r in rows if r['model']==model],key=lambda r:float(r['requested_referral_rate']))
        require([float(r['requested_referral_rate']) for r in group]==[0,.05,.1,.2],'Incomplete operating points')
        require(all(int(r['full_n'])==136 for r in group),'Wrong plotted cohort')
        coverage=[100*float(r['coverage']) for r in group]
        risk=[100*float(r['retained_error_rate']) for r in group]
        referral=[100*float(r['actual_referral_rate']) for r in group]
        misses=[int(r['retained_fn']) for r in group]
        axes[0].plot(coverage,risk,color=color,marker=marker,label=model,linewidth=1.2,markersize=4)
        axes[1].plot(referral,misses,color=color,marker=marker,label=model,linewidth=1.2,markersize=4)
        # Ring the predeclared 10% rule (14/136 actually referred), not a chosen optimum.
        axes[0].scatter([coverage[2]],[risk[2]],s=65,facecolors='none',edgecolors=color,linewidths=1)
        axes[1].scatter([referral[2]],[misses[2]],s=65,facecolors='none',edgecolors=color,linewidths=1)
    axes[0].set(xlabel='Automatically handled patients (%)',ylabel='Error rate among automatically handled (%)',
                title='Risk–coverage',xlim=(77,101))
    axes[1].set(xlabel='Patients referred (%)',ylabel='AD misses among automatically handled (count)',
                title='Remaining AD misses',xlim=(-1,22))
    for i,ax in enumerate(axes):
        ax.set_ylim(bottom=0);ax.text(0,1.05,chr(97+i),transform=ax.transAxes,fontweight='bold')
    axes[0].legend(fontsize=7,loc='lower left')
    fig.suptitle('Batch referral at threshold 0.5 | development validation (n = 136)',fontsize=10)
    fig.supxlabel('Fixed rates: 0/5/10/20%. Rings: 14 referred (10.29%). Referral does not imply correction.',fontsize=7)
    output=Path(output)
    fig.savefig(output/'referral_analysis.svg',metadata={'Date':None})
    fig.savefig(output/'referral_analysis.pdf',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(output/'referral_analysis.png',dpi=600)
    plt.close(fig)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metrics',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();render(args.metrics,args.output)
