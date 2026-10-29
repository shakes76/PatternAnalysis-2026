"""N1/N2: pretrained ConvNeXt smoke check; no full training or test pixels.

Reference architecture: Liu et al., CVPR 2022, https://arxiv.org/abs/2201.03545
Oracle and weight source: https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py
See THIRD_PARTY_NOTICES.md. Local tests use synthetic random state dictionaries;
only the CLI pins the accepted real pretrained file's full SHA256.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import inspect
import json
import logging
from pathlib import Path
import platform
import random
import sys
import time

from PIL import Image, ImageDraw
import PIL
import torch
from torch import nn

from check_dataloader import require, sha256, code_version, make_loader
from dataset import (ConvNeXtSliceDataset, CONVNEXT_PREPROCESSING,
                     LABEL_MAP, convnext_crop, convnext_transform)
from modules import ConvNeXtTiny, build_model
from train import class_weight_config

OFFICIAL_SHA256 = '983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d'
WEIGHT_NAME = 'ConvNeXt_Tiny_Weights.IMAGENET1K_V1'
WEIGHT_URL = 'https://download.pytorch.org/models/convnext_tiny-983f1562.pth'


def verify_weights(path, summary_path, expected_sha256=OFFICIAL_SHA256):
    """Rehash every run, including existing files. Never download implicitly."""
    digest = sha256(path)
    require(digest == expected_sha256, 'Pretrained SHA256 mismatch')
    report = json.loads(Path(summary_path).read_text())
    require(report.get('status') == 'passed' and report.get('strict_cpu_load') is True,
            'Weight preparation has not passed strict CPU loading')
    require(report.get('sha256') == digest and report.get('size_bytes') == Path(path).stat().st_size,
            'Preparation summary/file mismatch')
    require(report.get('weights') == WEIGHT_NAME and report.get('url') == WEIGHT_URL
            and report.get('original_classifier_outputs') == 1000, 'Unexpected weight identity')
    return digest


def compare_official(model, state, probe):
    """Use the same 1000-class state and inputs to test independent implementations."""
    from torchvision.models import convnext_tiny
    reference = convnext_tiny(weights=None).eval()
    reference.load_state_dict(state, strict=True)
    model.eval()
    differences = []
    with torch.no_grad():
        for inputs in (probe, torch.randn_like(probe)):
            expected, actual = reference(inputs), model(inputs)
            require(bool(torch.isfinite(actual).all()), 'Nonfinite oracle comparison')
            require(torch.allclose(actual, expected, rtol=1e-5, atol=1e-6),
                    'Self-implemented model differs from official model')
            differences.append(float((actual-expected).abs().max()))
    return dict(passed=True, device='cpu', inputs=['validation smoke batch','seeded random tensor'],
                rtol=1e-5, atol=1e-6, maximum_absolute_differences=differences,
                classifier_outputs=1000)


def validate_batch(batch, records):
    images, labels = batch['image'], batch['label']
    require(tuple(images.shape) == (len(labels),3,224,224)
            and images.dtype == torch.float32 and bool(torch.isfinite(images).all()),
            'Invalid ConvNeXt input')
    require(labels.dtype == torch.int64 and tuple(labels.shape) == (len(images),)
            and bool(((labels==0)|(labels==1)).all()), 'Invalid labels')
    by_path = {row['file_path']:row for row in records}
    for i,path in enumerate(batch['file_path']):
        require(path in by_path, 'Sample outside requested split')
        row = by_path[path]
        require(int(labels[i]) == LABEL_MAP[row['label']], 'Batch label mismatch')
        require(all(batch[key][i] == row[key] for key in ('patient_id','image_id')),
                'Batch metadata mismatch')


def train_step(model, optimizer, images, labels, weights):
    """Two-class weighted CE; require finite gradients and both head/stem updates."""
    require(images.ndim == 4 and tuple(images.shape[1:]) == (3,224,224)
            and bool(torch.isfinite(images).all()), 'Invalid/nonfinite input')
    require(labels.dtype == torch.int64 and tuple(labels.shape) == (len(images),)
            and bool(((labels==0)|(labels==1)).all()), 'Invalid labels')
    require(tuple(weights.shape) == (2,) and bool(torch.isfinite(weights).all())
            and bool((weights>0).all()), 'Invalid class weights')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    before_head = model.classifier[-1].weight.detach().clone()
    before_stem = model.features[0][0].weight.detach().clone()
    logits = model(images)
    require(tuple(logits.shape) == (len(labels),2) and bool(torch.isfinite(logits).all()), 'Invalid logits')
    loss = nn.functional.cross_entropy(logits,labels,weight=weights)
    require(bool(torch.isfinite(loss)), 'Nonfinite loss')
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    require(all(g is not None and bool(torch.isfinite(g).all()) for g in grads),
            'Missing or nonfinite gradient')
    norm = float(torch.sqrt(sum(g.detach().square().sum() for g in grads)))
    require(norm>0 and norm<float('inf'), 'Invalid gradient norm')
    optimizer.step()
    head_changed = not torch.equal(before_head,model.classifier[-1].weight)
    stem_changed = not torch.equal(before_stem,model.features[0][0].weight)
    require(head_changed and stem_changed, 'Head or backbone did not update')
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'Nonfinite parameters')
    return dict(loss=float(loss.detach()),gradient_norm=norm,head_changed=head_changed,
                backbone_changed=stem_changed,logits_shape=list(logits.shape),
                batch_size=len(labels))


def load_checkpoint(path, expected_config, device):
    checkpoint = torch.load(path,map_location='cpu',weights_only=True)
    require(checkpoint.get('config') == expected_config, 'Checkpoint configuration mismatch')
    model = build_model(expected_config['model']).to(device)
    model.load_state_dict(checkpoint['model_state'],strict=True)
    return model.eval()


def export_previews(dataset, output_dir, seed):
    """Two patients/class; first/middle/last lexicographic path, not anatomical rank."""
    from torchvision.models import ConvNeXt_Tiny_Weights
    from torchvision.transforms import functional as F, InterpolationMode
    require(dataset.split == 'train', 'Crop previews must use training data')
    generator = random.Random(seed)
    preview_dir = output_dir/'crop_previews'
    preview_dir.mkdir()
    by_patient = {}
    for i,row in enumerate(dataset.records):
        by_patient.setdefault(row['patient_id'],[]).append(i)
    selected=[]
    for label in ('NC','AD'):
        patients=sorted(p for p,indices in by_patient.items() if dataset.records[indices[0]]['label']==label)
        generator.shuffle(patients)
        for patient in patients[:2]:
            indices=sorted(by_patient[patient],key=lambda i:dataset.records[i]['file_path'])
            selected.extend(indices[pos] for pos in sorted({0,len(indices)//2,len(indices)-1}))
    results=[]
    official=ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms()
    for number,index in enumerate(selected):
        row=dataset.records[index]
        gray=dataset.read_gray(index)
        actual=convnext_transform(gray)
        expected=official(gray.convert('RGB'))
        require(torch.equal(actual,expected), 'Preprocessing differs from official preset')
        crop=convnext_crop(gray)
        resized=F.resize(gray.convert('RGB'),[236],interpolation=InterpolationMode.BILINEAR,antialias=True)
        left,top=round((resized.width-224)/2),round((resized.height-224)/2)
        annotated=resized.copy()
        ImageDraw.Draw(annotated).rectangle((left,top,left+223,top+223),outline='red',width=2)
        originals=[gray.convert('RGB'),annotated,crop]
        canvas=Image.new('RGB',(sum(im.width for im in originals)+40,max(im.height for im in originals)+40),'white')
        draw=ImageDraw.Draw(canvas);x=10
        for im,label in zip(originals,('Original','Resize + crop box','224 x 224 crop')):
            canvas.paste(im,(x,30));draw.text((x,8),label,fill='black');x+=im.width+10
        name=f'crop_previews/sample_{number:02d}.png'
        canvas.save(output_dir/name)
        results.append(dict(**row,preview=name,original_size=list(gray.size),
                            crop_box_on_resized=[left,top,left+224,top+224],
                            preprocessing_max_abs_diff=float((actual-expected).abs().max()),
                            sha256=sha256(output_dir/name)))
    (output_dir/'crop_previews.json').write_text(json.dumps(results,indent=2)+'\n')
    return results


def run_check(manifest, summary, weights_path, weights_summary, output_dir,
              device='cuda', batch_size=4, seed=42, expected_lengths=None,
              expected_weight_sha256=OFFICIAL_SHA256):
    """CLI uses real cohort/digest; offline tests supply synthetic manifests/states."""
    require(device in ('cpu','cuda'), 'Use cpu or cuda')
    require(batch_size>0, 'batch_size must be positive')
    require(device!='cuda' or torch.cuda.is_available(), 'CUDA requested but unavailable')
    output_dir=Path(output_dir)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    datasets={s:ConvNeXtSliceDataset(manifest,s,summary) for s in ('train','validation','test')}
    lengths={s:len(ds) for s,ds in datasets.items()}
    require(expected_lengths is None or lengths==expected_lengths, 'Frozen cohort lengths mismatch')
    if expected_lengths is not None:
        for split,ad,nc in [('train',133,275),('validation',44,92),('test',44,92)]:
            ds=datasets[split]; counts=Counter(r['patient_id'] for r in ds.records)
            labels={r['patient_id']:r['label'] for r in ds.records}
            require(set(counts.values())=={20} and Counter(labels.values())=={'AD':ad,'NC':nc},
                    'Frozen patient/class/slice counts mismatch')
    digest=verify_weights(weights_path,weights_summary,expected_weight_sha256)
    state=torch.load(weights_path,map_location='cpu',weights_only=True)
    model=build_model('convnext_tiny',num_classes=1000)
    model.load_state_dict(state,strict=True)
    val_batch=next(iter(make_loader(datasets['validation'],batch_size,seed)))
    validate_batch(val_batch,datasets['validation'].records)
    equivalence=compare_official(model,state,val_batch['image'])
    del state
    # Only the class-specific linear head is replaced; all other pretrained tensors remain.
    torch.manual_seed(seed)
    model.classifier[-1]=nn.Linear(768,2)
    nn.init.trunc_normal_(model.classifier[-1].weight,std=0.02)
    nn.init.zeros_(model.classifier[-1].bias)
    weighting=class_weight_config(datasets['train'].records,'balanced')
    config=dict(model='convnext_tiny',num_classes=2,seed=seed,batch_size=batch_size,
        optimizer='AdamW',learning_rate=1e-4,weight_decay=0.01,precision='float32',
        full_finetuning=True,stochastic_depth_probability=0.1,steps=2,
        class_weight_details=weighting,preprocessing=deepcopy(CONVNEXT_PREPROCESSING),
        weights_sha256=digest,weight_name=WEIGHT_NAME,validation_loss='unweighted CE',
        deterministic_algorithms=True,tf32=False,
        input_sha256=dict(manifest=sha256(manifest),summary=sha256(summary)))
    (output_dir/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    if device=='cuda':
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    model=model.to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0.01)
    weights=torch.tensor(weighting['weights'],dtype=torch.float32,device=device)
    loader=make_loader(datasets['train'],batch_size,seed)
    require(len(loader)>=2,'Need two distinct training batches')
    iterator=iter(loader);steps=[]
    for step in range(2):
        batch=next(iterator);validate_batch(batch,datasets['train'].records)
        images,labels=batch['image'].to(device),batch['label'].to(device)
        if device=='cuda':torch.cuda.synchronize()
        start=time.perf_counter()
        result=train_step(model,optimizer,images,labels,weights)
        if device=='cuda':torch.cuda.synchronize()
        result.update(seconds=time.perf_counter()-start,paths=batch['file_path'])
        steps.append(result);logging.info('Step %d: %s',step+1,result)
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    before={name:value.detach().clone() for name,value in model.state_dict().items()}
    images=val_batch['image'].to(device)
    with torch.no_grad():
        logits=model(images)
        loss=nn.functional.cross_entropy(logits,val_batch['label'].to(device))
    require(tuple(logits.shape)==(len(images),2) and bool(torch.isfinite(logits).all())
            and bool(torch.isfinite(loss)), 'Invalid validation output')
    require(all(torch.equal(before[name],value) for name,value in model.state_dict().items())
            and all(p.grad is None for p in model.parameters()),
            'Validation changed model or created gradients')
    del before
    training_memory=(dict(allocated_bytes=torch.cuda.max_memory_allocated(),
                          reserved_bytes=torch.cuda.max_memory_reserved()) if device=='cuda' else None)
    checkpoint=output_dir/'smoke_checkpoint.pt'
    torch.save(dict(config=config,model_state={k:v.detach().cpu() for k,v in model.state_dict().items()}),checkpoint)
    expected=logits.detach().cpu()
    del optimizer,model,logits
    if device=='cuda':torch.cuda.empty_cache()
    restored=load_checkpoint(checkpoint,config,device)
    with torch.no_grad(): actual=restored(images).cpu()
    require(torch.allclose(expected,actual,rtol=1e-5,atol=1e-6), 'Checkpoint reload mismatch')
    # Timing has two inference warm-ups followed by five measured repeats, no optimizer updates.
    with torch.no_grad():
        for _ in range(2):restored(images)
        times=[]
        for _ in range(5):
            if device=='cuda':torch.cuda.synchronize()
            start=time.perf_counter();restored(images)
            if device=='cuda':torch.cuda.synchronize()
            times.append(time.perf_counter()-start)
    previews=ConvNeXtSliceDataset(manifest,'train',summary)
    preview_results=export_previews(previews,output_dir,seed)
    memory=(dict(allocated_bytes=torch.cuda.max_memory_allocated(),
                 reserved_bytes=torch.cuda.max_memory_reserved()) if device=='cuda' else None)
    require(memory is None or max(memory.values())<=20*1024**3, 'Peak memory exceeds 20 GiB smoke target')
    require(datasets['test'].read_calls==0,'Test pixels were read')
    return dict(status='passed',scope='Small-batch engineering check, not accuracy evaluation',
        official_pretrained_digest=digest==OFFICIAL_SHA256,weights_sha256=digest,
        official_model_equivalence=equivalence,preprocessing_matches_official=True,
        config=config,steps=steps,validation_loss=float(loss),validation_state_unchanged=True,
        validation_grad_enabled=False,checkpoint_reload_equal=True,
        reload_max_abs_diff=float((expected-actual).abs().max()),
        checkpoint_sha256=sha256(checkpoint),
        parameter_count=sum(p.numel() for p in restored.parameters()),
        trainable_parameter_count=sum(p.numel() for p in restored.parameters() if p.requires_grad),
        dataset_lengths=lengths,class_weight_details=weighting,
        read_counts={**{s:ds.read_calls for s,ds in datasets.items()},'training_preview_reads':previews.read_calls},
        observed_paths={s:ds.observed_paths for s,ds in datasets.items()},
        crop_preview_count=len(preview_results),crop_visual_review='pending human review; not an automated anatomy verdict',
        device=device,gpu_name=torch.cuda.get_device_name() if device=='cuda' else None,
        training_validation_peak_cuda_memory=training_memory,peak_cuda_memory=memory,
        optimization_timing_scope='Synchronized steps include diagnostics, exclude image loading/transfers; not benchmark',
        inference_timing=dict(warmups=2,repeats=5,batch_size=len(images),seconds=times,
            scope='eval/no_grad, synchronized GPU, excludes I/O/preprocessing/transfers; smoke estimate only'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','summary','weights','weights-summary','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.FileHandler(args.output_dir/'convnext_check.log'),logging.StreamHandler(sys.stdout)])
    result=dict(generated_at=datetime.now(timezone.utc).isoformat(),command=sys.argv,code_version=code_version())
    try:
        import torchvision
        from torchvision.models import convnext_tiny
        from torchvision.transforms._presets import ImageClassification
        result['versions']=dict(python=platform.python_version(),torch=str(torch.__version__),
            torchvision=str(torchvision.__version__),pillow=PIL.__version__,cuda_build=torch.version.cuda)
        result['code_sha256']={n:sha256(Path(__file__).with_name(n)) for n in
            ['check_convnext.py','modules.py','dataset.py','check_dataloader.py','train.py','run_n2.slurm']}
        result['reference_source_sha256']={k:sha256(inspect.getsourcefile(v)) for k,v in
            [('torchvision_convnext',inspect.unwrap(convnext_tiny)),('torchvision_preprocessing',ImageClassification)]}
        result['input_sha256']={k:sha256(v) for k,v in
            [('split_samples',args.manifest),('split_summary',args.summary),('weights_summary',args.weights_summary)]}
        result.update(run_check(args.manifest,args.summary,args.weights,args.weights_summary,args.output_dir,
            args.device,args.batch_size,args.seed,{'train':8160,'validation':2720,'test':2720}))
        result['output_sha256']={p.name:sha256(p) for p in args.output_dir.iterdir()
            if p.name in ('config.json','crop_previews.json')}
        logging.info('PASS (crop visual review still required): %s',result['read_counts'])
    except Exception as exc:
        result.update(status='failed',error_count=1,error=f'{type(exc).__name__}: {exc}')
        logging.exception('ConvNeXt check failed')
        raise
    finally:
        (args.output_dir/'convnext_check_summary.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__': main()
