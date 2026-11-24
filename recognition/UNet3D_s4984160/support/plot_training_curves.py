"""Plot existing scratch/refinement histories without training or inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def read_history(path,phase,offset):
    with Path(path).open(newline='',encoding='utf-8') as stream:
        rows=list(csv.DictReader(stream))
    if not rows or [int(r['epoch']) for r in rows]!=list(range(1,len(rows)+1)):
        raise ValueError('Expected contiguous recorded epochs')
    return [dict(phase=phase,phase_epoch=int(r['epoch']),cumulative_epoch=offset+int(r['epoch']),
                 train_loss=float(r['train_loss']),
                 **{f'val_dice_{c}':float(r[f'val_dice_{c}']) for c in range(6)}) for r in rows]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scratch-history',required=True)
    p.add_argument('--refine-history',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--selected-refine-epoch',type=int,default=11)
    args=p.parse_args()
    scratch=read_history(args.scratch_history,'scratch',0)
    refine=read_history(args.refine_history,'refinement',len(scratch))
    selected=next(r for r in refine if r['phase_epoch']==args.selected_refine_epoch)
    if selected['val_dice_5']!=max(r['val_dice_5'] for r in refine):
        raise ValueError('Selected epoch does not match recorded best prostate Dice')
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    for group,label,color in [(scratch,'Initial training','#2864b4'),(refine,'Refinement','#d16b17')]:
        xs=[r['cumulative_epoch'] for r in group]
        axes[0].plot(xs,[r['train_loss'] for r in group],label=label,color=color)
        axes[1].plot(xs,[r['val_dice_5'] for r in group],label=label,color=color)
    for ax in axes:
        ax.axvline(len(scratch)+.5,color='gray',linestyle=':',label='Phase boundary')
        ax.set_xlabel('Cumulative epoch');ax.legend(fontsize=8)
    axes[0].set_ylabel('Recorded training objective')
    axes[0].set_title('Training loss (phase-specific class weights)')
    axes[1].set_ylabel('Mean validation prostate Dice');axes[1].set_ylim(-.02,1.)
    axes[1].set_title('Validation overlap on the same patient split')
    axes[1].axhline(.7,color='gray',linestyle='--')
    axes[1].scatter([selected['cumulative_epoch']],[selected['val_dice_5']],color='black',zorder=5)
    axes[1].annotate('Selected checkpoint',xy=(selected['cumulative_epoch'],selected['val_dice_5']),
                     xytext=(-95,18),textcoords='offset points',fontsize=8)
    fig.suptitle('Standard 3D U-Net: existing training records only\n'
                 'Class weights/optimizer change between phases; validation loss was not logged',fontsize=11)
    fig.savefig(out/'training_curves.png',dpi=160);plt.close(fig)
    rows=scratch+refine
    with (out/'curve_data.csv').open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for source,name in [(args.scratch_history,'scratch_history.csv'),(args.refine_history,'refine_history.csv')]:
        (out/name).write_bytes(Path(source).read_bytes())
    metadata=dict(scratch_epochs=len(scratch),refinement_epochs=len(refine),
                  selected_cumulative_epoch=selected['cumulative_epoch'],
                  selected_refine_epoch=selected['phase_epoch'],selected_val_dice=selected['val_dice_5'],
                  sources_sha256={name:hashlib.sha256(Path(s).read_bytes()).hexdigest()
                    for name,s in [('scratch_history.csv',args.scratch_history),('refine_history.csv',args.refine_history)]},
                  note='No validation-loss series exists; phase loss weights differ. No training performed.')
    (out/'curve_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2),flush=True)


if __name__=='__main__':
    main()
