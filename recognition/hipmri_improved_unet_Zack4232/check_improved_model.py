"""Check model wiring on training slices; this is not a quality evaluation."""

import argparse
import io
from pathlib import Path

import torch

from dataset import HipMRIDataset
from losses import CrossEntropyDiceLoss
from modules import UNet2D, ImprovedUNet2D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-checkpoint', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Run this check inside a GPU job.')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    baseline = UNet2D()
    assert sum(p.numel() for p in baseline.parameters()) == 7759686
    saved = torch.load(args.baseline_checkpoint, map_location='cpu', weights_only=True)
    baseline.load_state_dict(saved['model_state_dict'], strict=True)
    print('Baseline checkpoint compatibility: PASS', flush=True)
    del baseline, saved

    data = HipMRIDataset(split='train')
    model = ImprovedUNet2D().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = CrossEntropyDiceLoss(dice_weight=1.0)
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    print('Improved model parameters:', sum(p.numel() for p in model.parameters()), flush=True)
    torch.cuda.reset_peak_memory_stats()
    for key, expected_width in [((4, 0, 30), 128), ((19, 1, 30), 144)]:
        index = next(i for i, r in enumerate(data.samples)
                     if (r.case_id, r.week, r.slice_id) == key)
        sample = data[index]
        image = sample['image'].unsqueeze(0).cuda()
        target = sample['mask'].unsqueeze(0).cuda()
        assert image.shape == (1, 1, 256, expected_width)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        before = model.segmentation_heads[0].weight.detach().clone()
        logits = model(image)
        assert logits.shape == (1, 6, 256, expected_width)
        assert torch.isfinite(logits).all()
        loss = criterion(logits, target)
        assert torch.isfinite(loss)
        loss.backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f'Missing gradient: {name}'
            assert torch.isfinite(parameter.grad).all(), f'Invalid gradient: {name}'
        for head in model.segmentation_heads:
            assert head.weight.grad.abs().sum() > 0
        optimizer.step()
        assert not torch.equal(before, model.segmentation_heads[0].weight)
        model.eval()
        with torch.no_grad():
            first = model(image)
            second = model(image)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        print(f'Case/week/slice: {key}; input: {tuple(image.shape)}; '
              f'output: {tuple(logits.shape)}; loss: {loss.item():.6f}', flush=True)
        print('Finite gradients, three active heads, update, eval repeatability: PASS', flush=True)

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = ImprovedUNet2D().cuda().eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored(image), first, rtol=0, atol=0)
    print('State dictionary round trip: PASS', flush=True)
    print('Peak allocated GPU memory (GiB):',
          torch.cuda.max_memory_allocated() / 1024**3, flush=True)
    print('Smoke test complete. No validation or test slices were read.', flush=True)
    print('Memory includes check overhead and is not a training benchmark.', flush=True)


if __name__ == '__main__':
    main()
