"""Plot raw epoch history with explicit partial-run and absent-class labels."""
import argparse
import io
import json
import math
from pathlib import Path


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_history(run_dir):
    directory = Path(run_dir).expanduser().resolve()
    metadata = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    config = metadata['config']
    if not isinstance(config.get('partial_run'), bool):
        raise ValueError('config must explicitly record boolean partial_run')
    history = json.loads((directory / 'history.json').read_text(encoding='utf-8'))
    if not isinstance(history, list) or not history:
        raise ValueError('History must contain at least one epoch')
    for expected_epoch, entry in enumerate(history, start=1):
        if type(entry.get('epoch')) is not int or entry['epoch'] != expected_epoch:
            raise ValueError('History epochs must be consecutive integers beginning at 1')
        for phase in ('train', 'validation'):
            result = entry[phase]
            if not _number(result['loss']):
                raise ValueError('History loss must be finite')
            for metric in ('dice', 'iou'):
                values = result[metric]
                if not isinstance(values, list) or len(values) != 6:
                    raise ValueError('Expected six class scores per metric')
                if any(value is not None and (not _number(value) or not 0 <= value <= 1) for value in values):
                    raise ValueError('Metric values must be null or finite scores in [0, 1]')
    return config, history


def metric_series(history, phase, metric, label):
    return [math.nan if entry[phase][metric][label] is None else entry[phase][metric][label]
            for entry in history]


def plot_history(run_dir, output=None):
    """Export loss and train/validation per-class Dice/IoU; never overwrite a file."""
    directory = Path(run_dir).expanduser().resolve()
    destination = Path(output).expanduser().resolve() if output else directory / 'training_curves.png'
    if destination.suffix.lower() != '.png':
        raise ValueError('Output must have a .png extension')
    if destination.exists():
        raise FileExistsError(f'Output already exists: {destination}')
    config, history = load_history(directory)
    # Lazy import lets history validation run without a plotting installation.
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    epochs = [entry['epoch'] for entry in history]
    colors = plt.get_cmap('tab10').colors[:6]
    fig = plt.figure(figsize=(12, 9), facecolor='white')
    grid = fig.add_gridspec(3, 2, height_ratios=(1.05, 1, 1))
    loss_axis = fig.add_subplot(grid[0, :])
    axes = [loss_axis]
    loss_axis.plot(epochs, [entry['train']['loss'] for entry in history],
                   color='#1769AA', marker='o', label='Train')
    loss_axis.plot(epochs, [entry['validation']['loss'] for entry in history],
                   color='#D55E00', marker='s', label='Validation')
    loss_axis.set_title('Training and validation loss', fontsize=11)
    loss_axis.set_ylabel('Mean batch loss (CE + Dice)')
    loss_axis.legend(fontsize=9)
    for row, metric in ((1, 'dice'), (2, 'iou')):
        for column, phase in enumerate(('train', 'validation')):
            axis = fig.add_subplot(grid[row, column])
            axes.append(axis)
            undefined = []
            for label, color in enumerate(colors):
                values = metric_series(history, phase, metric, label)
                if all(math.isnan(value) for value in values):
                    undefined.append(str(label))
                axis.plot(epochs, values, color=color, marker='o', markersize=3, linewidth=1.3)
            axis.set_title(f'{phase.title()} {"Dice" if metric == "dice" else "IoU"}', fontsize=11)
            axis.set_ylabel('Score')
            axis.set_ylim(-0.02, 1.02)
            if undefined:
                axis.text(0.02, 0.92, 'Undefined throughout: ' + ', '.join(undefined),
                          transform=axis.transAxes, fontsize=8, va='top', color='#555555')
    for axis in axes:
        axis.set_xlabel('Epoch')
        axis.grid(alpha=0.2)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        if len(epochs) <= 10:
            axis.set_xticks(epochs)
        if len(epochs) == 1:
            axis.set_xlim(epochs[0] - 0.5, epochs[0] + 0.5)
        axis.tick_params(labelsize=8)
    status = 'PARTIAL RUN - smoke check only' if config['partial_run'] else 'Full-split run'
    architecture = config.get('architecture', 'Segmentation model')
    width = config.get('model_config', {}).get('base_channels', '?')
    fig.suptitle(f'{architecture} training history (base width {width})\n{status}',
                 fontsize=14, y=0.98)
    fig.legend([Line2D([0], [0], color=color, marker='o', markersize=4) for color in colors],
               [f'Label {label}' for label in range(6)], loc='lower center',
               ncol=6, fontsize=9, bbox_to_anchor=(0.5, 0.015), frameon=False)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.87, bottom=0.11, hspace=0.53, wspace=0.22)
    try:
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=160, facecolor='white',
                    metadata={'Description': status + '; raw epoch values; null metrics are gaps'})
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as handle:
            handle.write(buffer.getvalue())
    finally:
        plt.close(fig)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='New PNG path; default is run-dir/training_curves.png')
    args = parser.parse_args()
    output = plot_history(args.run_dir, args.output)
    print('Saved training curves:', output)
    print('History plot: PASS')


if __name__ == '__main__':
    main()
