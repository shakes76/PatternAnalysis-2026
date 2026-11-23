# Inference resource comparison

All three models used the same 32 development reports, verified by matching
ordered-report hashes, on an NVIDIA A100-PCIE-40GB. Runs used float32, batch
size one, two warm-up calls, and identical greedy generation settings with a
1,024-new-token cap. LoRA adapters were not merged into the base model.

| Measurement | Untuned baseline | LoRA | Full fine-tuning |
|---|---:|---:|---:|
| Total parameters | 76,961,152 | 77,305,216 | 76,961,152 |
| Generated tokens | 1,726 | 892 | 1,007 |
| Measured generation time (s) | 19.304 | 12.400 | 11.176 |
| Generated tokens/s | 89.41 | 71.93 | 90.11 |
| Mean latency/report (s) | 0.603 | 0.388 | 0.349 |
| Median latency/report (s) | 0.184 | 0.256 | 0.244 |
| 95th-percentile latency/report (s) | 0.793 | 0.874 | 0.848 |
| Peak CUDA allocated memory (GiB) | 0.325 | 0.301 | 0.300 |
| Peak CUDA reserved memory (GiB) | 0.377 | 0.336 | 0.336 |

## Measurement scope

Timing covers synchronized `model.generate` calls only. Loading, tokenisation,
host-to-device transfer, decoding, warm-up, and ROUGE scoring are excluded.
Generated token counts exclude T5's initial decoder-start token and include EOS
when present. Peak allocated memory includes the resident model, pretokenised
input tensors, and generation allocations. It is not total process/device VRAM
and is separate from the training peaks of 5.26 and 6.16 GiB.

Results are a single-pass benchmark on a small development workload, not a
latency guarantee or population estimate. Generated lengths differ across
models, so lower mean report latency alone does not establish faster computation
for a fixed output length. The baseline has a lower median but higher mean,
consistent with a slower tail; the raw per-example diagnostics should accompany
any interpretation of that tail.

## Interpretation

Full fine-tuning produces approximately 90.11 tokens/s versus 71.93 for unmerged
LoRA, with similar peak allocated inference memory in this workload. LoRA's
training memory savings therefore should not be presented as equivalent
inference savings. All three fit easily on the allocated A100 for this benchmark.
The comparison does not test merged adapters, larger batches, other GPUs, or
CPU/Apple MPS inference. No model or generation settings were tuned using final
evaluation results.

Provenance: `results/inference_profiles/baseline.json`, `lora_all.json`, and
`full_all.json`; these raw files remain excluded from Git. The common ordered
report hash is
`3733b385aa7e056308883cdc2e2e8c43f42aa7fc09d58cde6e81a808c295d9cb`.
