"""Predict from a checkpoint and labelled MRI, or render existing cached outputs."""
import argparse
import json
from pathlib import Path

import numpy as np


def planes_for(target, geometry):
    z = np.flatnonzero((target==5).any(axis=(1,2)))
    if not len(z):
        z = np.array([0,target.shape[0]//2,target.shape[0]-1])
    chosen = [int(z[0]),int(z[len(z)//2]),int(z[-1])]
    names = ['low array extent','mid-gland','high array extent']
    if geometry['original_axcodes'][2]=='S':
        names = ['inferior GT extent (apex proxy)','mid-gland','superior GT extent (base proxy)']
    result = [('axial',d,f'{name}; D={d}') for d,name in zip(chosen,names)]
    w = geometry['shared_target_w_indices']
    index = w[len(w)//2] if w else target.shape[2]//2
    result.append(('coronal',index,f'coronal midpoint; W={index}'))
    return result


def render(image,target,prediction,meta,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    organ_colors = ListedColormap(['cyan','orange','magenta'])
    error_colors = ListedColormap(['lime','red','dodgerblue'])
    fig,axes = plt.subplots(4,4,figsize=(15,13),constrained_layout=True)
    low,high = np.percentile(image,[1,99])
    for row,(kind,index,label) in enumerate(planes_for(target,meta)):
        if kind=='axial':
            x,y,p = image[index].T,target[index].T,prediction[index].T
            origin = 'upper'
        else:
            x,y,p = image[:,:,index],target[:,:,index],prediction[:,:,index]
            origin = 'lower'
        for col in range(4):
            axes[row,col].imshow(x,cmap='gray',vmin=low,vmax=high,origin=origin)
            axes[row,col].axis('off')
        for col,mask in [(1,y),(2,p)]:
            overlay = np.ma.masked_where(~np.isin(mask,[3,4,5]),mask.astype(int)-3)
            axes[row,col].imshow(overlay,cmap=organ_colors,vmin=0,vmax=2,alpha=.55,origin=origin)
        truth,pred = y==5,p==5
        errors = np.full(truth.shape,-1)
        errors[truth & pred]=0; errors[pred & ~truth]=1; errors[truth & ~pred]=2
        axes[row,3].imshow(np.ma.masked_less(errors,0),cmap=error_colors,vmin=0,vmax=2,alpha=.7,origin=origin)
        for col,title in enumerate(['MRI','Ground truth','Prediction','Prostate TP / FP / FN']):
            axes[row,col].set_title(f'{title}\n{label}',fontsize=9)
    dice = meta['prostate_dice']; distance = meta['prostate_hd95_mm']
    fig.suptitle(f'{meta["case"]} | {meta["split"]} | prostate Dice={dice} | HD95={distance} mm\n'
                 f'Selection: {meta["selection_reason"]}\n'
                 'Organs: bladder cyan, rectum orange, prostate magenta. Errors: TP green, FP red, FN blue.\n'
                 'Resized grid; anatomical extents are GT-derived proxies.',fontsize=11)
    fig.savefig(path,dpi=140)
    plt.close(fig)


def render_cached(input_dir, output_dir):
    inputs,output = Path(input_dir),Path(output_dir)
    manifest = json.loads((inputs/'manifest.json').read_text())
    output.mkdir(parents=True,exist_ok=False)
    figures = []
    for row in manifest['cases']:
        with np.load(inputs/row['cache_file'],allow_pickle=False) as data:
            meta = json.loads(str(data['metadata'].item()))
            name = row['case'].replace('.nii.gz','')+'.png'
            render(data['image'],data['target'],data['prediction'],meta,output/name)
        figures.append(dict(row,figure=name))
        print(f'Saved {output/name}',flush=True)
    (output/'manifest.json').write_text(json.dumps(dict(
        checkpoint=manifest['checkpoint'],checkpoint_sha256=manifest['checkpoint_sha256'],
        split=manifest['split'],figures=figures),indent=2),encoding='utf-8')
    (output/'plots_complete.txt').write_text('Prediction figures completed successfully.\n')


def predict_case(checkpoint_path, mri_path, mask_path, output_dir,
                 device='auto', precision='auto', expected_sha256=None, threads=1):
    """Real checkpoint loading/inference using the exact training preprocessing.

    The example takes a labelled MRI so it can show GT/error overlays. It
    exports resized-grid labels, not an original-resolution clinical contour.
    """
    import torch
    import nibabel as nib
    from contextlib import nullcontext
    from dataset import HipMRIDataset
    from support.evaluation_utils import (load_model, checkpoint_sha256,
                                          geometry, class_metrics, hd95)
    if threads<1:
        raise ValueError('threads must be positive')
    torch.set_num_threads(threads)
    torch.manual_seed(42)
    if device!='cpu':torch.backends.cudnn.benchmark=True
    digest=checkpoint_sha256(checkpoint_path)
    if expected_sha256 and digest!=expected_sha256:
        raise ValueError('Checkpoint SHA256 mismatch')
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    if precision=='auto':
        precision='bf16' if device=='cuda' and torch.cuda.is_bf16_supported() else 'fp32'
    if precision=='bf16' and (device!='cuda' or not torch.cuda.is_bf16_supported()):
        raise ValueError('BF16 mode requires a supported CUDA GPU')
    model,checkpoint=load_model(checkpoint_path,device)
    shape=tuple(checkpoint['training_config']['target_shape'])
    if len(shape)!=3 or any(s<8 or s%8 for s in shape):
        raise ValueError('Checkpoint target shape must have three positive multiples of eight')
    mri_path,mask_path=Path(mri_path),Path(mask_path)
    raw_mask=np.asarray(nib.load(str(mask_path)).dataobj)
    raw_mri=np.asarray(nib.load(str(mri_path)).dataobj)
    if not np.isfinite(raw_mri).all() or not np.isfinite(raw_mask).all():
        raise ValueError('Non-finite MRI or labels')
    if not np.equal(raw_mask,np.rint(raw_mask)).all() or not np.isin(raw_mask,range(6)).all():
        raise ValueError('Mask must contain integer labels 0..5')
    image,target=HipMRIDataset([(mri_path.name.split('_')[0],mri_path,mask_path)],shape)[0]
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=False)
    context=torch.autocast('cuda',dtype=torch.bfloat16) if precision=='bf16' else nullcontext()
    with torch.inference_mode(),context:
        prediction=model(image[None].to(device)).argmax(1)[0].cpu().numpy().astype(np.uint8)
    truth=target.numpy().astype(np.uint8)
    geo=geometry(mri_path,shape)
    metrics=class_metrics(prediction,truth)
    try:
        distance,status=hd95(prediction==5,truth==5,geo['spacing_dhw_mm'])
    except ModuleNotFoundError as error:
        if error.name!='scipy' and not str(error.name).startswith('scipy.'):
            raise
        distance,status=None,'unavailable_scipy'
    if checkpoint_sha256(checkpoint_path)!=digest:
        raise ValueError('Checkpoint changed during inference')
    metadata=dict(case=mri_path.name,split='single-case inference',
                  selection_reason='Explicit user-specified case; no model selection',
                  checkpoint=str(checkpoint_path),checkpoint_sha256=digest,
                  epoch=checkpoint['epoch'],model_config=checkpoint['model_config'],
                  device=device,precision=precision,target_shape=list(shape),**geo,
                  prostate_dice=metrics[5]['dice'],prostate_hd95_mm=distance,
                  hd95_status=status,class_metrics=metrics,
                  note='Resized D,H,W labels; single-case result, not a new aggregate test estimate')
    np.save(output/'prediction_dhw.npy',prediction)
    render(image[0].numpy(),truth,prediction,metadata,output/'prediction.png')
    (output/'summary.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    (output/'prediction_complete.txt').write_text('Checkpoint inference and overlay completed.\n')
    print(json.dumps(dict(case=metadata['case'],checkpoint_sha256=digest,
                          prostate_dice=metrics[5]['dice'],hd95_mm=distance,
                          precision=precision,output_dir=str(output)),indent=2),flush=True)
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input-dir',help='Render existing evaluation caches; no new inference')
    source.add_argument('--checkpoint',help='Load weights and infer the specified labelled MRI')
    parser.add_argument('--mri');parser.add_argument('--mask')
    parser.add_argument('--expected-sha256')
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--precision',choices=['auto','fp32','bf16'],default='auto')
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    if args.input_dir:
        if args.mri or args.mask or args.expected_sha256:
            parser.error('MRI/mask/checkpoint hash arguments require --checkpoint')
        render_cached(args.input_dir,args.output_dir)
    else:
        if not args.mri or not args.mask:parser.error('--checkpoint requires --mri and --mask')
        predict_case(args.checkpoint,args.mri,args.mask,args.output_dir,
                     args.device,args.precision,args.expected_sha256,args.threads)


if __name__ == '__main__':
    main()
