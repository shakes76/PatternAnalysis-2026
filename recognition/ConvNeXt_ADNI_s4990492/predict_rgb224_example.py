"""Single validation-slice demonstration of the frozen final model."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from dataset import RGB224Dataset
from modules import ConvNeXtTinyRGB


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--row', type=int, default=0, help='Zero-based validation manifest row')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output', type=Path, default=base/'runs/final_prediction_example')
    args = parser.parse_args()
    checkpoint_path = base/'runs/local_rgb224_lr3e6_imagenet_20261026_201343_178908/best_convnext.pth'
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if digest != '6b89086579452034b671dff60844f8416829bce60a6fb9a6c23609a877425cd5':
        raise ValueError('Final checkpoint hash mismatch')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    manifest = base.parents[1]/'local_data/splits_local/val.csv'
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != checkpoint['config']['manifest_sha256']['val']:
        raise ValueError('Validation manifest hash mismatch')
    data = RGB224Dataset(manifest, training=False)
    if not 0 <= args.row < len(data):
        raise ValueError('Row outside validation manifest')
    if args.output.exists():
        raise FileExistsError(f'Preserving existing output: {args.output}')
    torch.set_num_threads(2)
    model = ConvNeXtTinyRGB()
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.to(args.device).eval()
    image, label = data[args.row]
    with torch.inference_mode():
        logits = model(image.unsqueeze(0).to(args.device))
        if not torch.isfinite(logits).all():
            raise ValueError('Non-finite logits')
        prob_ad = logits.softmax(1)[0, 1].item()
    predicted = int(prob_ad >= 0.5)
    names = ['NC', 'AD']
    result = dict(split='validation', manifest_row=args.row, checkpoint_sha256=digest,
                  true_class=names[label.item()], predicted_class=names[predicted],
                  prob_ad=prob_ad, threshold=0.5, probabilities_calibrated=False,
                  device=args.device, note='One-slice demonstration, not a performance estimate')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Undo normalization to display exactly the padded RGB224 model input.
    display = image * torch.tensor([.229,.224,.225])[:,None,None] + torch.tensor([.485,.456,.406])[:,None,None]
    fig, ax = plt.subplots(figsize=(6,6), layout='constrained')
    ax.imshow(display.permute(1,2,0).clamp(0,1).numpy())
    ax.set_title(f"Validation slice | True: {names[label.item()]} | Predicted: {names[predicted]}\nP(AD) = {prob_ad:.4f}; threshold = 0.50 (uncalibrated)", fontsize=10)
    ax.axis('off')
    args.output.mkdir(parents=True, exist_ok=False)
    fig.savefig(args.output/'prediction.png', dpi=160)
    plt.close(fig)
    text = json.dumps(result, indent=2, allow_nan=False)
    (args.output/'prediction.json').write_text(text+'\n', encoding='utf-8')
    print(text)
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
