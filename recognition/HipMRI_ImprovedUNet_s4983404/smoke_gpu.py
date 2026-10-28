"""Short real-data overfit diagnostic; train split only, no generalisation claims."""
from pathlib import Path
import json
import logging
import torch
from dataset import ROOT, pair_paths, load_pair
from modules import build_model
from metrics import segmentation_loss, confusion_counts, scores


def main() -> None:
    output = Path('runs/overfit_strict')
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    examples = []
    for image_path, label_path in pair_paths(ROOT, 'train'):
        image, labels = load_pair(image_path, label_path)
        if (labels == 5).sum() > 100:
            examples.append((image, labels, image_path.name))
        if len(examples) == 2:
            break
    images = torch.stack([example[0] for example in examples]).cuda()
    targets = torch.nn.functional.one_hot(
        torch.stack([example[1] for example in examples]), 6).permute(0, 3, 1, 2).float().cuda()
    results = {}
    for name in ('basic', 'improved'):
        model = build_model(name, width=8).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
        initial = float(segmentation_loss(model(images), targets).detach())
        for step in range(800 if name == 'basic' else 150):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = segmentation_loss(model(images), targets)
            loss.backward()
            optimizer.step()
            if step % 100 == 0:
                logging.info('%s step=%d loss=%.4f', name, step, float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            logits = model(images)
            final = float(segmentation_loss(logits, targets))
        results[name] = {'initial_loss': initial, 'final_loss': final,
                         **scores(confusion_counts(logits, targets).cpu())}
        if final >= initial*0.5:
            raise AssertionError(f'{name}: tiny-set loss failed to halve')
        if results[name]['dice'][5] < 0.9:
            raise AssertionError(f'{name}: label-5 overfit Dice below 0.9')
    results['samples'] = [example[2] for example in examples]
    (output/'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    logging.info('%s', results)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    main()
