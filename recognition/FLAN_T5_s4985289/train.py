"""Single-GPU FLAN-T5 training (full or LoRA) with held-out validation loss."""
import argparse
import json
import random
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
from dataset import load_splits, get_tokenizer, tokenize_splits, MODEL_ID
from modules import create_model, parameter_counts


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['full', 'lora'], required=True)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--grad-accum', type=int, default=4)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--train-limit', type=int, default=None)
    p.add_argument('--val-limit', type=int, default=None)
    p.add_argument('--source-len', type=int, default=256)
    p.add_argument('--target-len', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output-dir', default=None)
    a = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable: submit this training with Slurm, not on login0')
    if a.epochs < 1 or a.grad_accum < 1:
        raise ValueError('epochs and grad-accum must be >= 1')
    seed_everything(a.seed)
    out = Path(a.output_dir or f'outputs/{a.mode}')
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = get_tokenizer()
    ds = tokenize_splits(load_splits(a.train_limit, a.val_limit), tokenizer,
                         a.source_len, a.target_len)
    model = create_model(a.mode).to('cuda')
    counts = parameter_counts(model)
    print('Parameter counts:', counts, flush=True)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=-100)
    train = DataLoader(ds['train'], batch_size=a.batch_size, shuffle=True, collate_fn=collator)
    val = DataLoader(ds['validation'], batch_size=a.batch_size, collate_fn=collator)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                   lr=a.lr or (2e-4 if a.mode == 'lora' else 3e-5))
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    history = []
    best = float('inf')
    for epoch in range(a.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for i, batch in enumerate(train):
            batch = {k: v.to('cuda') for k, v in batch.items()}
            loss = model(**batch).loss
            total_loss += loss.item()
            (loss / a.grad_accum).backward()
            if (i + 1) % a.grad_accum == 0 or i + 1 == len(train):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if (i + 1) % 100 == 0:
                print(f'epoch={epoch + 1} batch={i+1}/{len(train)} loss={loss.item():.4f}', flush=True)
        model.eval()
        loss_sum = 0.0
        with torch.inference_mode():
            for batch in val:
                batch = {k: v.to('cuda') for k, v in batch.items()}
                loss_sum += model(**batch).loss.item()
        record = {'epoch': epoch + 1, 'train_loss': total_loss / len(train),
                  'validation_loss': loss_sum / len(val)}
        history.append(record)
        print(record, flush=True)
        if record['validation_loss'] < best:
            best = record['validation_loss']
            model.save_pretrained(out / 'best')
            tokenizer.save_pretrained(out / 'best')
        (out / 'history.json').write_text(json.dumps(history, indent=2))
    profile = {'mode': a.mode, **counts, 'train_seconds': time.perf_counter() - start,
               'peak_gpu_gb': torch.cuda.max_memory_allocated() / (1024**3),
               'seed': a.seed, 'config': vars(a)}
    (out / 'train_profile.json').write_text(json.dumps(profile, indent=2))
    # Save training curve, rather than inventing loss/ROUGE results.
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for key in ('train_loss', 'validation_loss'):
            plt.plot([r['epoch'] for r in history], [r[key] for r in history], label=key)
        plt.xlabel('Epoch')
        plt.ylabel('Cross-entropy loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / 'loss_curve.png', dpi=160)
        plt.close()
    except ImportError:
        print('matplotlib unavailable: history.json saved, plot omitted')
    print(json.dumps(profile, indent=2))


if __name__ == '__main__':
    main()
