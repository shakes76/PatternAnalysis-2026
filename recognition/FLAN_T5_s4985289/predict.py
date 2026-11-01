"""Required test driver: load saved weights and evaluate on validation/test."""
import argparse
import json
from dataset import load_splits, get_tokenizer, MODEL_ID
from modules import load_checkpoint
from evaluate_utils import run_generation


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['zero', 'lora', 'full'], required=True)
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--split', choices=['validation', 'test'], default='validation')
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--source-len', type=int, default=256)
    p.add_argument('--target-len', type=int, default=128)
    p.add_argument('--output-dir', default=None)
    a = p.parse_args()
    if a.mode != 'zero' and not a.checkpoint:
        p.error('--checkpoint is required for lora/full')
    ds = load_splits()
    split = ds[a.split]
    if a.limit is not None:
        split = split.select(range(min(a.limit, len(split))))
    model = load_checkpoint(a.mode, a.checkpoint, MODEL_ID)
    tokenizer = get_tokenizer(a.checkpoint or MODEL_ID)
    output = a.output_dir or f'outputs/eval_{a.mode}_{a.split}'
    metrics, rows = run_generation(model, tokenizer, split,
                                   batch_size=a.batch_size, source_len=a.source_len,
                                   max_new_tokens=a.target_len, output_dir=output)
    print(json.dumps(metrics, indent=2))
    for row in rows[:3]:
        print('\nINPUT:', row['source'], '\nREFERENCE:', row['reference'],
              '\nPREDICTION:', row['prediction'])


if __name__ == '__main__':
    main()
