"""Zero-shot evaluation on the *official validation* split."""
import argparse
import json
from dataset import get_tokenizer, load_splits, MODEL_ID
from modules import create_model
from evaluate_utils import run_generation


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--val-limit', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--source-len', type=int, default=256)
    p.add_argument('--target-len', type=int, default=128)
    p.add_argument('--output-dir', default='outputs/zero')
    a = p.parse_args()
    ds = load_splits(val_limit=a.val_limit)
    metrics, _ = run_generation(create_model('zero'), get_tokenizer(MODEL_ID),
                                ds['validation'], batch_size=a.batch_size,
                                source_len=a.source_len, max_new_tokens=a.target_len,
                                output_dir=a.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
