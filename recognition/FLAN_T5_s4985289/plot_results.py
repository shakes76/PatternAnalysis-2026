# import json
# from pathlib import Path
# import numpy as np
# import matplotlib.pyplot as plt

# base = Path(__file__).resolve().parent / "results"

# models = {
#     "Zero-shot": base / "eval_zero_validation/metrics.json",
#     "LoRA": base / "eval_lora_validation/metrics.json",
#     "Full": base / "eval_full_validation/metrics.json",
# }

# metrics = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
# values = {
#     name: json.loads(path.read_text())
#     for name, path in models.items()
# }

# x = np.arange(len(metrics))
# width = 0.25

# fig, ax = plt.subplots(figsize=(9, 5))

# for i, (name, data) in enumerate(values.items()):
#     ax.bar(x + (i - 1) * width,
#            [data[m] for m in metrics],
#            width, label=name)

# ax.set_xticks(x)
# ax.set_xticklabels(["ROUGE-1", "ROUGE-2", "ROUGE-L", "ROUGE-Lsum"])
# ax.set_ylabel("ROUGE Score")
# ax.set_title("FLAN-T5 Validation ROUGE Comparison")
# ax.set_ylim(0, 0.85)
# ax.legend()

# fig.tight_layout()

# output = base / "rouge_comparison.png"
# fig.savefig(output, dpi=300)
# plt.close(fig)

# print(f"Saved to: {output}")

from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent / 'results'
FIG = BASE / 'figures'
FIG.mkdir(parents=True, exist_ok=True)


def read_json(relative):
    path = BASE / relative
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}\nDownload the relevant Rangpur results first.')
    return json.loads(path.read_text(encoding='utf-8'))


def save(fig, name):
    path = FIG / name
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def annotate_bars(ax, decimals=3):
    for group in ax.containers:
        ax.bar_label(group, fmt=f'%.{decimals}f', padding=3, fontsize=8)


models = {
    'Zero-shot': read_json('eval_zero_validation/metrics.json'),
    'LoRA': read_json('eval_lora_validation/metrics.json'),
    'Full': read_json('eval_full_validation/metrics.json'),
}

history = {
    'LoRA': read_json('lora/history.json'),
    'Full': read_json('full/history.json'),
}
profiles = {
    'LoRA': read_json('lora/train_profile.json'),
    'Full': read_json('full/train_profile.json'),
}

# Figure 1: ROUGE
metrics = [('rouge1', 'ROUGE-1'), ('rouge2', 'ROUGE-2'),
           ('rougeL', 'ROUGE-L'), ('rougeLsum', 'ROUGE-Lsum')]
fig, ax = plt.subplots(figsize=(9.5, 5.5))
xs = list(range(len(metrics)))
width = .25
for i, (name, data) in enumerate(models.items()):
    ax.bar([x + (i-1)*width for x in xs],
           [data[key] for key, _ in metrics], width, label=name)
ax.set_xticks(xs, [label for _, label in metrics])
ax.set_ylim(0, 0.9)
ax.set_ylabel('ROUGE F1 score')
ax.set_title('FLAN-T5 Validation ROUGE Comparison (n = 10,000)')
ax.legend()
annotate_bars(ax)
fig.tight_layout()
save(fig, '01_rouge_comparison.png')

# Figure 2: Training & validation loss, both training strategies
fig, ax = plt.subplots(figsize=(9.5, 5.5))
for mode, rows in history.items():
    epochs = [r['epoch'] for r in rows]
    ax.plot(epochs, [r['train_loss'] for r in rows], 'o-', label=f'{mode} train')
    ax.plot(epochs, [r['validation_loss'] for r in rows], 's--', label=f'{mode} validation')
ax.set_xticks(sorted(set(r['epoch'] for rows in history.values() for r in rows)))
ax.set_xlabel('Epoch')
ax.set_ylabel('Cross-entropy loss')
ax.set_title('FLAN-T5 Training and Validation Loss')
ax.grid(axis='y', alpha=.25)
ax.legend()
fig.tight_layout()
save(fig, '02_loss_comparison.png')

# Figure 3: Training resources, two units shown on separate axes (no misleading shared scale)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
labels = list(profiles)
train_minutes = [profiles[k]['train_seconds'] / 60 for k in labels]
train_vram = [profiles[k]['peak_gpu_gb'] for k in labels]
axes[0].bar(labels, train_minutes)
axes[0].set_ylabel('Minutes')
axes[0].set_title('Training time')
axes[0].set_ylim(0, max(train_minutes) * 1.2)
annotate_bars(axes[0], decimals=1)
axes[1].bar(labels, train_vram)
axes[1].set_ylabel('Peak allocated GPU memory (GiB)')
axes[1].set_title('Training GPU memory')
axes[1].set_ylim(0, max(train_vram) * 1.2)
annotate_bars(axes[1], decimals=2)
fig.suptitle('FLAN-T5 Fine-tuning Computational Cost')
fig.tight_layout()
save(fig, '03_training_cost.png')


# Figure 4: Inference performance, two units on separate axes
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
labels = list(models)
latencies = [models[k]['seconds_per_report'] for k in labels]
throughput = [models[k]['generated_tokens_per_second'] for k in labels]
axes[0].bar(labels, latencies)
axes[0].set_ylabel('Seconds per report (lower is better)')
axes[0].set_title('Inference latency')
axes[0].set_ylim(0, max(latencies) * 1.2)
annotate_bars(axes[0], decimals=3)
axes[1].bar(labels, throughput)
axes[1].set_ylabel('Generated tokens per second (higher is better)')
axes[1].set_title('Generation throughput')
axes[1].set_ylim(0, max(throughput) * 1.2)
annotate_bars(axes[1], decimals=1)
fig.suptitle('FLAN-T5 Validation Inference Efficiency (n = 10,000)')
fig.tight_layout()
save(fig, '04_inference_efficiency.png')

print('\nAll four report figures generated successfully.')
print('Note: trainable_params from full inference metadata reflects model flags, not actual gradient updates.')










