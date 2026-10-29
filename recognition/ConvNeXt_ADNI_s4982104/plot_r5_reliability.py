"""Plot complete R5b confidence bins; empty bins remain missing, not zero.

Figure contract: compare confidence versus empirical accuracy before/after
patient-level calibration, together with bin support. A quantitative grid uses
all 136 validation patients per family, equal weighting, no error bars or tests.
Three columns distinguish families; two rows separate reliability and counts.
Python/matplotlib only. Export editable SVG/PDF plus 600-dpi PNG at 183x135 mm.
Development validation fit is stated explicitly; no generalization claim.
"""
import csv
from pathlib import Path


def render(bin_file, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from run_r5_ensemble import MODELS, require

    with Path(bin_file).open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8,
        'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,
        'axes.spines.right':False,'legend.frameon':False,'axes.linewidth':.8})
    fig,axes=plt.subplots(2,3,figsize=(7.2047244,5.3149606),
                          gridspec_kw={'height_ratios':[2,1]},layout='constrained')
    colors={'raw':'#586875','calibrated':'#0072B2'}
    markers={'raw':'o','calibrated':'s'}
    for column,model in enumerate(MODELS):
        upper,lower=axes[:,column]
        upper.plot([0,1],[0,1],color='#999999',linestyle='--',linewidth=.8)
        for stage in ('raw','calibrated'):
            bins=sorted([r for r in rows if r['model']==model and r['stage']==stage],key=lambda r:int(r['bin']))
            require(len(bins)==10 and [int(r['bin']) for r in bins]==list(range(10)), 'Incomplete plot bins')
            counts=[int(r['count']) for r in bins]
            require(sum(counts)>0,'Empty plot cohort')
            # NaNs explicitly break curves at empty bins rather than invent points.
            xs=[float(r['mean_confidence']) if int(r['count']) else float('nan') for r in bins]
            ys=[float(r['accuracy']) if int(r['count']) else float('nan') for r in bins]
            upper.plot(xs,ys,marker=markers[stage],markersize=4,linewidth=1,
                       color=colors[stage],label=stage.capitalize())
            offset=-.02 if stage=='raw' else .02
            centers=[(float(r['left'])+float(r['right']))/2+offset for r in bins]
            lower.bar(centers,counts,width=.038,color=colors[stage],label=stage.capitalize())
        upper.set(title=model,xlabel='Mean confidence',xlim=(0,1),ylim=(0,1))
        lower.set(xlabel='Confidence bin',xlim=(0,1))
        lower.set_ylim(bottom=0)
        if column==0:
            upper.set_ylabel('Observed accuracy');lower.set_ylabel('Patient count')
            upper.legend(loc='upper left',fontsize=7)
        upper.text(0,1.07,chr(97+column),transform=upper.transAxes,fontweight='bold',va='bottom')
    fig.suptitle('Patient confidence calibration | development validation (n = 136)',fontsize=10)
    fig.supxlabel('10 equal-width bins on [0, 1]; empty bins omitted. Temperatures fitted on these same patients.',fontsize=7)
    output=Path(output)
    fig.savefig(output/'reliability_diagram.svg',metadata={'Date':None})
    fig.savefig(output/'reliability_diagram.pdf',metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(output/'reliability_diagram.png',dpi=600)
    plt.close(fig)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bins',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    render(args.bins,args.output)
