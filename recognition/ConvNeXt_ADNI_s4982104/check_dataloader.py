"""Check frozen ADNI batches without training or reading test images."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import platform
import subprocess
import sys

import PIL
import torch
from torch.utils.data import DataLoader
from dataset import ADNISliceDataset, LABEL_MAP, PREPROCESSING


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class AuditedDataset(ADNISliceDataset):
    """Retain pre-collation evidence without extra image reads (workers=0)."""
    def __init__(self, manifest, split, summary):
        super().__init__(manifest, split, summary)
        self.read_calls = 0
        self.observed = {}
        self.by_path = {r['file_path']: r for r in self.records}

    def __getitem__(self, index):
        self.read_calls += 1
        item = super().__getitem__(index)
        self.observed[item['file_path']] = item['image'].clone()
        return item


def make_loader(dataset, batch_size, seed):
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=dataset.split == 'train', num_workers=0,
                      drop_last=False, generator=torch.Generator().manual_seed(seed))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_batch(batch, dataset):
    images, labels = batch['image'], batch['label']
    n = len(labels)
    require(list(images.shape) == [n,1,240,256], 'Invalid image shape')
    require(list(labels.shape) == [n] and labels.dtype == torch.int64, 'Invalid labels')
    require(images.dtype == torch.float32 and bool(torch.isfinite(images).all()), 'Invalid pixels')
    require(float(images.min()) >= 0 and float(images.max()) <= 1, 'Pixels outside [0,1]')
    require(all(len(batch[k]) == n for k in ('file_path','patient_id','image_id')), 'Metadata length mismatch')
    for i, path in enumerate(batch['file_path']):
        require(path in dataset.by_path, 'Path outside requested split')
        row = dataset.by_path[path]
        require(row['split'] == dataset.split, 'Split mismatch')
        require(labels[i].item() == LABEL_MAP[row['label']], 'Label mismatch')
        for key in ('patient_id','image_id'):
            require(batch[key][i] == row[key], f'{key} mismatch')
        require(path in dataset.observed and torch.equal(images[i],dataset.observed[path]),
                'Image/path correspondence mismatch')
    return dict(image_shape=list(images.shape), label_shape=list(labels.shape),
                image_dtype=str(images.dtype),label_dtype=str(labels.dtype),
                device=str(images.device), pixel_min=float(images.min()),pixel_max=float(images.max()),
                finite=True,correspondence_passed=True,
                label_counts={label:int((labels == value).sum()) for label,value in LABEL_MAP.items()})


def run_check(manifest, summary, batch_size=4, seed=42, expected_lengths=None):
    train = AuditedDataset(manifest,'train',summary)
    validation = AuditedDataset(manifest,'validation',summary)
    with Path(manifest).open(newline='') as f:
        rows = list(csv.DictReader(f))
    lengths = dict(Counter(r['split'] for r in rows))
    if expected_lengths is not None:
        require(lengths == expected_lengths,
                f'Dataset lengths {lengths} do not match frozen cohort {expected_lengths}')
    splits = ('train','validation','test')
    patients = {s:{r['patient_id'] for r in rows if r['split']==s} for s in splits}
    overlaps = {f'{a}_{b}':len(patients[a]&patients[b])
                for a,b in [('train','validation'),('train','test'),('validation','test')]}
    require(not any(overlaps.values()), 'Patient leakage across splits')
    train_loader = make_loader(train,batch_size,seed)
    val_loader = make_loader(validation,batch_size,seed)
    first_train = next(iter(train_loader))
    train_result = validate_batch(first_train,train)
    logging.info('Train batch: %s', train_result)
    first_val = next(iter(val_loader))
    val_result = validate_batch(first_val,validation)
    logging.info('Validation batch: %s', val_result)
    repeat_val = next(iter(val_loader))
    validate_batch(repeat_val,validation)
    repeat_equal = all(torch.equal(first_val[k],repeat_val[k]) if torch.is_tensor(first_val[k])
                       else first_val[k] == repeat_val[k] for k in first_val)
    require(repeat_equal, 'Validation first batch changed on fresh iterator')
    logging.info('Validation fresh-iterator repeat: identical; test image reads: 0')
    return dict(status='passed',error_count=0,
        scope='First training and validation batches only; not a full image audit',
        preprocessing=PREPROCESSING,batch_size=batch_size,seed=seed,num_workers=0,drop_last=False,
        shuffle={'train':True,'validation':False},
        dataset_lengths=dict(Counter(r['split'] for r in rows)),
        patient_intersections=overlaps,batches={'train':train_result,'validation':val_result},
        validation_repeat_identical=repeat_equal,
        read_counts=dict(train=train.read_calls,validation=validation.read_calls,test=0,
                         total=train.read_calls+validation.read_calls,
                         unique_images=len(set(train.observed)|set(validation.observed))))


def code_version():
    base = Path(__file__).resolve().parent
    def git(*args):
        return subprocess.check_output(['git','-C',str(base),*args],stderr=subprocess.DEVNULL).decode().strip()
    try:
        return dict(commit=git('rev-parse','HEAD'),dirty=bool(git('status','--porcelain')))
    except (subprocess.CalledProcessError,FileNotFoundError):
        return dict(commit=None,dirty=None,reason='Deployed directory is not a Git checkout')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--source-version',type=Path,help='Optional verified source checkout provenance')
    args = p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    output = args.output_dir/'loader_check_summary.json'
    log = args.output_dir/'loader_check.log'
    if output.exists() or log.exists():
        p.error('Output evidence already exists; use a new output directory')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(log),logging.StreamHandler(sys.stdout)])
    result = dict(generated_at=datetime.now(timezone.utc).isoformat(),command=sys.argv,
        versions=dict(python=platform.python_version(),pytorch=torch.__version__,pillow=PIL.__version__,
                      torch_cuda_build=torch.version.cuda),code_version=code_version(),
        code_sha256={name:sha256(Path(__file__).with_name(name)) for name in ('check_dataloader.py','dataset.py')},
        output_dir=str(args.output_dir.resolve()))
    try:
        result['input_sha256'] = dict(split_samples=sha256(args.manifest),
                                      split_summary=sha256(args.summary))
        if args.source_version:
            source = json.loads(args.source_version.read_text())
            require(source['code_sha256'] == result['code_sha256'],'Source code hash mismatch')
            result['source_checkout_version'] = source
        logging.info('Versions: %s; inputs: %s',result['versions'],result['input_sha256'])
        result.update(run_check(args.manifest,args.summary,args.batch_size,args.seed,
                                expected_lengths={'train':8160,'validation':2720,'test':2720}))
        logging.info('PASS: %s',result['read_counts'])
    except Exception as e:
        result.update(status='failed',error_count=1,error=f'{type(e).__name__}: {e}')
        logging.exception('Batch check failed')
        raise
    finally:
        output.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
