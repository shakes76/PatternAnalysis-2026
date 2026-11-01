"""Leakage-aware LaymanRRG text data, with the official fixed splits."""
from datasets import load_dataset
from transformers import AutoTokenizer

DATASET_ID = 'BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track'
MODEL_ID = 'google/flan-t5-base'
PROMPT = 'Rewrite this radiology report in simple, accurate, patient-friendly language without adding new findings: '


def get_tokenizer(model_id=MODEL_ID):
    return AutoTokenizer.from_pretrained(model_id)


def load_splits(train_limit=None, val_limit=None, test_limit=None):
    """Use publisher-supplied splits. Limits select first N only for smoke tests."""
    ds = load_dataset(DATASET_ID)
    for split, limit in [('train', train_limit), ('validation', val_limit), ('test', test_limit)]:
        if limit is not None:
            ds[split] = ds[split].select(range(min(limit, len(ds[split]))))
    return ds


def tokenize_splits(ds, tokenizer, source_len=256, target_len=128):
    def convert(batch):
        inputs = [PROMPT + str(s) for s in batch['radiology_report']]
        enc = tokenizer(inputs, max_length=source_len, truncation=True)
        tgt = tokenizer(text_target=[str(s) for s in batch['layman_report']],
                        max_length=target_len, truncation=True)
        enc['labels'] = tgt['input_ids']
        return enc
    return ds.map(convert, batched=True,
                  remove_columns=ds['train'].column_names, desc='Tokenising')


def verify_exact_duplicates(ds):
    """Detect exact same input texts crossing official splits (report, not patient ID)."""
    sets = {k: set(s.strip().casefold() for s in ds[k]['radiology_report']) for k in ds}
    return {f'{a}-{b}': len(sets[a] & sets[b])
            for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]}


if __name__ == '__main__':
    ds = load_splits(100, 20, 20)
    print(ds)
    print('Exact-text split overlaps:', verify_exact_duplicates(ds))
    tok = get_tokenizer()
    print(tokenize_splits(ds, tok)['train'][0].keys())
