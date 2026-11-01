"""Deterministic ROUGE-1/2/L/Lsum evaluation and inference profiling."""
import json
import time
from pathlib import Path
import torch
import evaluate
from torch.utils.data import DataLoader
from dataset import PROMPT
from modules import parameter_counts


def score_predictions(predictions, references):
    rouge = evaluate.load('rouge')
    return {key: float(val) for key, val in rouge.compute(
        predictions=predictions, references=references, use_stemmer=True).items()}


def run_generation(model, tokenizer, data, batch_size=8, source_len=256,
                   max_new_tokens=128, output_dir=None, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    rows = []
    data_loader = DataLoader(list(range(len(data))), batch_size=batch_size, shuffle=False)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    produced_tokens = 0
    for indices in data_loader:
        examples = [data[int(i)] for i in indices]
        sources = [PROMPT + str(row['radiology_report']) for row in examples]
        inputs = tokenizer(sources, padding=True, truncation=True,
                           max_length=source_len, return_tensors='pt').to(device)
        with torch.inference_mode():
            tokens = model.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=1)
        decoded = tokenizer.batch_decode(tokens, skip_special_tokens=True)
        produced_tokens += int((tokens != tokenizer.pad_token_id).sum().item())
        for row, pred in zip(examples, decoded):
            rows.append({'source': row['radiology_report'],
                         'reference': row['layman_report'], 'prediction': pred})
    if device == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    metrics = score_predictions([r['prediction'] for r in rows],
                                [r['reference'] for r in rows])
    metrics.update(parameter_counts(model))
    metrics.update({'samples': len(rows), 'seconds': elapsed,
                    'seconds_per_report': elapsed / max(1, len(rows)),
                    'generated_tokens_per_second': produced_tokens / max(elapsed, 1e-9),
                    'peak_gpu_gb': torch.cuda.max_memory_allocated() / (1024**3) if device == 'cuda' else None})
    if output_dir:
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / 'metrics.json').write_text(json.dumps(metrics, indent=2))
        with (p / 'predictions.jsonl').open('w') as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return metrics, rows
