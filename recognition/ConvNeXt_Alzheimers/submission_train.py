"""Explicit-input A1 entry points. Smoke is 32 training slices and one update.

Full training is available for reproduction but requires an explicit flag; it
is not run by the submission verification task. Historical evidence is not edited.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader,Subset
from dataset import ADNISliceDataset,load_finalized_splits,require,counts_for_records
import train_convnext_v2 as v2

class TrainingSubset(Subset):split='train'

def new_output(path,root,splits):
    path=Path(path).resolve()
    require(not path.exists(),'Use a NEW output directory; never overwrite an earlier run.')
    for protected in (Path(root).resolve(),Path(splits).resolve(),Path(__file__).resolve().parent):
        require(path!=protected and protected not in path.parents,'Output is inside protected data/source.')
    require(not any((p/'.git').exists() for p in (path,*path.parents)),'Keep training outputs outside Git.')
    path.mkdir(parents=True);return path

def smoke(root,splits,output,device_name):
    bundle=load_finalized_splits(root,splits);out=new_output(output,root,splits)
    config=v2.Config('scratch_augmented',device=device_name)
    device=v2.choose_device(device_name);v2.seed_everything(42)
    data=v2.WeightedDataset(bundle,config)
    indices=[]
    for label in (0,1):indices += [i for i,r in enumerate(data.records) if r.label==label][:16]
    require(len(indices)==32,'Smoke requires sixteen training slices per class.')
    subset=TrainingSubset(data,indices)
    loader=DataLoader(subset,batch_size=8,shuffle=True,num_workers=0,
                      generator=torch.Generator().manual_seed(43),pin_memory=device.type=='cuda')
    model=v2.make_model(config).to(device);optimizer=v2.make_optimizer(model,config)
    v2.reset_memory_peak(device);v2.set_epoch(model,optimizer,config,1)
    before=model.head.weight.detach().clone()
    result=v2.train_epoch(model,loader,optimizer,device,config)
    require(result['images']==32 and result['updates']==1 and not torch.equal(before,model.head.weight),
            'Smoke did not execute the expected single update.')
    model.eval();sample=data[indices[0]]['image'].unsqueeze(0).to(device)
    with torch.inference_mode():expected=model(sample).cpu()
    checkpoint={'scope':'TRAINING_SMOKE_NOT_A_CANDIDATE','configuration':asdict(config),
        'manifest_sha256':bundle.manifest_hashes,'source_code_sha256':v2.source_hashes(),
        'model_state':v2.tensor_state(model),'optimizer_state':optimizer.state_dict()}
    torch.save(checkpoint,out/'PRIVATE_smoke.pt')
    loaded=torch.load(out/'PRIVATE_smoke.pt',map_location='cpu',weights_only=True)
    clone=v2.make_model(config).to(device);clone.load_state_dict(loaded['model_state'],strict=True);clone.eval()
    with torch.inference_mode():actual=clone(sample).cpu()
    require(torch.equal(expected,actual),'Smoke checkpoint round trip differs.')
    receipt={'scope':'32_TRAINING_SLICES_ONE_UPDATE_NOT_A_COHORT_EPOCH','passed':True,
        'configuration':asdict(config),'manifest_sha256':bundle.manifest_hashes,
        'source_code_sha256':v2.source_hashes(),'train':result,'gpu_memory':v2.memory_info(device),
        'checkpoint_roundtrip_equal':True,'validation_images_decoded':0,'test_images_decoded':0,
        'candidate_selection_performed':False,'environment':v2.environment(device)}
    v2.atomic_json(out/'training_smoke.json',receipt);print(json.dumps(receipt,indent=2));return receipt

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('a1','a1-smoke'));p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--splits-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda');p.add_argument('--full-training',action='store_true')
    p.add_argument('--confirm-preprocessing-review',action='store_true')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args(argv)
    if a.command=='a1-smoke':
        require(not a.resume and not a.full_training,'Smoke is separate from full training/recovery.')
        return smoke(a.data_root,a.splits_dir,a.output_dir,a.device)
    require(a.full_training,'Full training requires --full-training; use a1-smoke for the bounded interface check.')
    require(a.confirm_preprocessing_review,'Confirm actual review of the unchanged preprocessing/augmentation.')
    config=v2.Config('scratch_augmented',device=a.device)
    from check_data_pipeline import run_check
    from training_utils import checked_bundle
    out=Path(a.output_dir).resolve()
    if not a.resume:
        out=new_output(out,a.data_root,a.splits_dir)
        # Fresh explicit-input preparation state; only training/validation pixels.
        run_check(a.data_root,a.splits_dir,out/'data-check',batch_size=8,num_workers=0)
        (out/'run').mkdir()
    else:require(out.is_dir() and (out/'run'/'PRIVATE_resume.pt').is_file(),'No existing run to resume.')
    report=out/'data-check'/'data_pipeline_summary.json'
    bundle=checked_bundle(a.data_root,a.splits_dir,report)
    with v2.run_lock(out/'run'):
        state=v2.train_run(out/'run',bundle,config,report,out,resume=a.resume)
    print('A1 training selection is validation accuracy, macro-F1, lower log loss; epoch 13 is not guaranteed.')
    return state

if __name__=='__main__':main()
