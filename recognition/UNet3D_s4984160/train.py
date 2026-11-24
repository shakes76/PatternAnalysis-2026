"""Refine the standard 3D U-Net with train-only class weights and longer learning."""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from support.baseline_model import make_model
from dataset import HipMRIDataset, get_data_pairs, split_by_patient


def cache(pairs, workers):
    data = HipMRIDataset(pairs, target_shape=(64, 128, 128))
    loader = DataLoader(data, batch_size=1, num_workers=workers, shuffle=False)
    cached = []
    for i, (image, mask) in enumerate(loader):
        cached.append((image[0].contiguous(), mask[0].to(torch.uint8).contiguous()))
        if (i+1) % 25 == 0 or i+1 == len(pairs):
            print(f'Cached {i+1}/{len(pairs)}', flush=True)
    return cached


def sample(image, mask, positions_by_label=None):
    """50% full volumes; 25% rectum-centred and 25% prostate-centred crops."""
    draw = random.random()
    if draw < .5:
        return image, mask, 'full'
    label = 4 if draw < .75 else 5
    positions = (positions_by_label[label] if positions_by_label is not None
                 else torch.nonzero(mask == label))
    if len(positions) == 0:
        return image, mask, 'full_missing_label'
    centre = positions[random.randrange(len(positions))].tolist()
    shape = (32, 64, 64)
    starts = [max(0, min(centre[a]+random.randint(-size//4, size//4)-size//2,
                         mask.shape[a]-size)) for a, size in enumerate(shape)]
    region = tuple(slice(start, start+size) for start, size in zip(starts, shape))
    return image[(slice(None),)+region], mask[region], f'class_{label}'


class Loss(nn.Module):
    def __init__(self, train_counts):
        super().__init__()
        counts = torch.as_tensor(train_counts, dtype=torch.float32)
        if counts.shape != (6,) or (counts <= 0).any():
            raise ValueError('All six classes must occur in the training cache')
        # Square-root inverse frequency, Body weight=1, capped for stability.
        # Counts come from TRAIN volumes only, never validation or test.
        self.register_buffer('ce_weights', (counts[1]/counts).sqrt().clamp(.25,25.))
        self.register_buffer('dice_weights', torch.tensor([1.,1.,1.,2.,2.]))

    def forward(self, logits, target):
        logits = logits.float()
        ce = F.cross_entropy(logits, target, weight=self.ce_weights)
        probs = logits.softmax(1)[:, 1:]
        truth = F.one_hot(target, 6).movedim(-1, 1)[:, 1:].float()
        dims = (0,2,3,4)
        dice = (2*(probs*truth).sum(dims)+1e-5)/(probs.sum(dims)+truth.sum(dims)+1e-5)
        weights = self.dice_weights
        return .5*ce + .5*((1-dice)*weights).sum()/weights.sum()


@torch.inference_mode()
def validate(model, cached):
    model.eval()
    scores = [[] for _ in range(6)]
    total = torch.zeros(6,6,dtype=torch.int64)
    for image, target in cached:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            prediction = model(image.unsqueeze(0).cuda()).argmax(1)[0].cpu()
        matrix = torch.bincount((target.long()*6+prediction).reshape(-1), minlength=36).reshape(6,6)
        total += matrix
        for c in range(6):
            denominator = int(matrix[c].sum()+matrix[:,c].sum())
            if denominator:
                scores[c].append(2*int(matrix[c,c])/denominator)
    dice = [sum(v)/len(v) if v else float('nan') for v in scores]
    return dice, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init-checkpoint', required=True,
                        help='Continue learned weights with a new optimizer/loss; not exact optimizer resume')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=.0005)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--max-train-batches', type=int, default=0)
    parser.add_argument('--max-val-cases', type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Slurm GPU required')
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    random.seed(42)
    torch.backends.cudnn.benchmark = True
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args), seed=42, base_channels=8, normalization='instance',
                  target_shape=[64,128,128], full_probability=.5,
                  rectum_crop_probability=.25, prostate_crop_probability=.25,
                  crop_shape=[32,64,64], ce_weight_rule='sqrt(train_body_count/train_class_count), clamp [0.25,25]',
                  foreground_dice_weights=[1,1,1,2,2], split='patient_seed42',
                  scheduler_start_epoch=30, scheduler_min_prostate_dice=.05,
                  scheduler_patience=10, min_early_stop_epoch=60, early_stop_patience=25)
    (out/'config.json').write_text(json.dumps(config, indent=2))
    train, val, _ = split_by_patient(get_data_pairs(), seed=42)
    if args.max_train_batches:
        train = train[:args.max_train_batches]
    if args.max_val_cases:
        val = val[:args.max_val_cases]
    (out/'split.json').write_text(json.dumps(dict(
        train=[dict(patient=p[0], case=p[1].name) for p in train],
        validation=[dict(patient=p[0], case=p[1].name) for p in val]), indent=2))
    print(f'Train {len(train)}, validation {len(val)}; test not loaded', flush=True)
    train_cache, val_cache = cache(train, args.workers), cache(val, args.workers)
    counts = sum((torch.bincount(mask.reshape(-1).long(), minlength=6)
                  for _, mask in train_cache), torch.zeros(6,dtype=torch.int64))
    # Compute positive-voxel locations once instead of scanning masks every epoch.
    positions_cache = [{label: torch.nonzero(mask == label) for label in (4,5)}
                       for _, mask in train_cache]
    model = make_model().cuda()
    if args.init_checkpoint:
        source = torch.load(args.init_checkpoint, map_location='cpu', weights_only=True)
        state = source['model_state_dict']
        required = model.state_dict()
        ignored = [k for k in state if k not in required]
        if any(not k.endswith(('running_mean','running_var','num_batches_tracked')) for k in ignored):
            raise ValueError(f'Unexpected initialization keys: {ignored}')
        model.load_state_dict({k:v for k,v in state.items() if k in required}, strict=True)
        config['source_epoch'] = source['epoch']
        print(f'Continue weights from source epoch {source["epoch"]}; new optimizer and learning rate', flush=True)
    else:
        print('Training FROM SCRATCH; no old checkpoint loaded', flush=True)
    criterion = Loss(counts).cuda()
    config['train_voxel_counts'] = counts.tolist()
    config['ce_weights'] = criterion.ce_weights.cpu().tolist()
    (out/'config.json').write_text(json.dumps(config, indent=2))
    print('Train-only class weights:', config['ce_weights'], flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.5, patience=10)
    best, best_foreground, stale, reached = -1., -1., 0, 0
    fields = ['epoch','train_loss','lr','seconds','prostate_precision','prostate_recall',
              'prostate_volume_ratio'] + [f'val_dice_{c}' for c in range(6)]
    with (out/'history.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1,args.epochs+1):
            start = time.perf_counter()
            model.train()
            order = list(range(len(train_cache)))
            random.shuffle(order)
            loss_sum = torch.zeros((), device='cuda')
            sampled = {}
            for index in order:
                image, mask, kind = sample(*train_cache[index], positions_by_label=positions_cache[index])
                sampled[kind] = sampled.get(kind,0)+1
                image, target = image.unsqueeze(0).cuda(), mask.unsqueeze(0).long().cuda()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = criterion(model(image), target)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 12.)
                optimizer.step()
                loss_sum.add_(loss.detach())
            dice, matrix = validate(model, val_cache)
            tp = int(matrix[5,5]); gt = int(matrix[5].sum()); pred = int(matrix[:,5].sum())
            row = dict(epoch=epoch, train_loss=float(loss_sum)/len(order),
                       lr=optimizer.param_groups[0]['lr'], seconds=time.perf_counter()-start,
                       prostate_precision=tp/pred if pred else 0., prostate_recall=tp/gt if gt else 0.,
                       prostate_volume_ratio=pred/gt if gt else 0.,
                       **{f'val_dice_{c}':v for c,v in enumerate(dice)})
            writer.writerow(row); f.flush()
            print(json.dumps(dict(row, sampled=sampled)), flush=True)
            foreground = sum(dice[1:])/5
            primary_improved = dice[5] > best+1e-8
            tied_better = abs(dice[5]-best) <= 1e-8 and foreground > best_foreground+1e-8
            stale = 0 if primary_improved else stale+1
            if primary_improved or tied_better:
                best, best_foreground = dice[5], foreground
                checkpoint = dict(model_state_dict=model.state_dict(), epoch=epoch,
                                  val_prostate_dice=best, val_dice=dice,
                                  model_config=dict(base_channels=8,normalization='instance'),
                                  training_config=config, init_checkpoint=args.init_checkpoint)
                torch.save(checkpoint, out/'best_model.pt')
                torch.save(matrix, out/'best_validation_confusion.pt')
                print(f'BEST epoch={epoch}, prostate Dice={best:.4f}', flush=True)
            # Do not decay the LR just because the rare class has not emerged yet.
            if epoch >= 30 and dice[5] > .05:
                scheduler.step(dice[5])
            torch.save(dict(model_state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(), epoch=epoch,
                            model_config=dict(base_channels=8,normalization='instance'),
                            best_prostate_dice=best, training_config=config,
                            python_rng=random.getstate(), torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all()), out/'last_model.pt')
            reached = reached+1 if dice[5] >= .70 and dice[4] > 0 else 0
            if reached >= 3:
                print('Validation prostate Dice >=0.70 for three epochs; stopping for review', flush=True)
                break
            if epoch >= 60 and best > .05 and stale >= 25:
                print('Validation plateau: stopping for targeted review', flush=True)
                break
    print(f'Finished; best validation prostate Dice={best:.4f}', flush=True)


if __name__ == '__main__':
    main()
