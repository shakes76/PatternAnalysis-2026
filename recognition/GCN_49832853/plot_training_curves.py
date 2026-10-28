"""Reconstruct *sampled* learning curves from the ORIGINAL console training logs.

Only epochs 1, 10, ..., 100 were printed. Do not interpret the lines as
full per-epoch histories or as curves from another training run.

Use: python plot_training_curves.py
Plots and CSV are saved into figures/ next to this script.
"""
from pathlib import Path
import csv
import matplotlib.pyplot as plt

EPOCHS = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
HISTORY = {
    'MLP': {
        'train_loss': [1.3880, 0.9860, 0.7207, 0.6346, 0.5969, 0.5777, 0.5641, 0.5535, 0.5452, 0.5391, 0.5351],
        'val_accuracy': [0.4324, 0.6520, 0.7301, 0.7499, 0.7648, 0.7675, 0.7721, 0.7741, 0.7773, 0.7770, 0.7795],
        'val_macro_f1': [0.3222, 0.5292, 0.7029, 0.7240, 0.7438, 0.7466, 0.7532, 0.7554, 0.7580, 0.7578, 0.7602],
    },
    'GCN': {
        'train_loss': [1.3833, 0.8931, 0.5364, 0.3884, 0.3323, 0.3051, 0.2920, 0.2846, 0.2754, 0.2701, 0.2668],
        'val_accuracy': [0.4546, 0.7343, 0.8543, 0.8805, 0.9003, 0.9054, 0.9092, 0.9090, 0.9110, 0.9134, 0.9154],
        'val_macro_f1': [0.3553, 0.6421, 0.8435, 0.8719, 0.8945, 0.9007, 0.9042, 0.9041, 0.9064, 0.9092, 0.9112],
    }
}


def main():
    figures = Path(__file__).resolve().parent / 'figures'
    figures.mkdir(exist_ok=True)

    fields = ['model', 'epoch', 'train_loss', 'val_accuracy', 'val_macro_f1']
    with (figures / 'sampled_training_history.csv').open('w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for model, metrics in HISTORY.items():
            for index, epoch in enumerate(EPOCHS):
                writer.writerow({
                    'model': model,
                    'epoch': epoch,
                    'train_loss': metrics['train_loss'][index],
                    'val_accuracy': metrics['val_accuracy'][index],
                    'val_macro_f1': metrics['val_macro_f1'][index],
                })

    for key, label, filename in [
        ('train_loss', 'Training cross-entropy loss', 'sampled_training_loss.png'),
        ('val_macro_f1', 'Validation macro-F1', 'sampled_validation_macro_f1.png'),
    ]:
        fig, ax = plt.subplots(figsize=(8.2, 4.9))
        for model, metrics in HISTORY.items():
            ax.plot(EPOCHS, metrics[key], marker='o', markersize=4, linewidth=2, label=model)
        ax.set(xlabel='Epoch (logged checkpoints)', ylabel=label, title=f'{label}: GCN vs MLP')
        ax.set_xticks(EPOCHS)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.text(0.5, 0.01, 'Data: original training console logs (epochs 1 and every 10 epochs); not a full per-epoch trace.',
                 ha='center', fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        outfile = figures / filename
        fig.savefig(outfile, dpi=180)
        plt.close(fig)
        print(f'Saved: {outfile}')
    print('Note: best validation checkpoint was epoch 99; plotted epoch 100 is the logged end-of-run score.')


if __name__ == '__main__':
    main()
