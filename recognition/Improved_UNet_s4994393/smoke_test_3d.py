"""Check the production model on synthetic data or a real MRI patch."""

import argparse
import tempfile
from pathlib import Path
import numpy as np
import torch
from dataset import PatchDataset, discover_pairs, split_pairs
from predict import sliding_window_predict
from modules import ImprovedUNet3D, deep_supervision_loss


def main():
    """Check gradients, optimizer updates, checkpoint reload and window coverage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    model = ImprovedUNet3D(6, 4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    if args.data_root:
        pairs = discover_pairs(args.data_root)
        training, validation = split_pairs(pairs)
        train_ids = {Path(p[0]).name.split("_")[0] for p in training}
        val_ids = {Path(p[0]).name.split("_")[0] for p in validation}
        assert train_ids.isdisjoint(val_ids)
        x, y = PatchDataset(pairs, (16, 16, 16))[0]
        x, y = x[None].to(device), y[None].to(device)
        print(f"Matched {len(pairs)} volumes; patient split is disjoint")
    else:
        x = torch.randn(1, 1, 17, 19, 21, device=device)
        y = torch.randint(0, 6, (1, 17, 19, 21), device=device)
    model.train()
    before = model.output.weight.detach().clone()
    outputs = model(x)
    assert outputs[0].shape == (1, 6, *y.shape[1:])
    loss = deep_supervision_loss(outputs, y)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    assert not torch.equal(before, model.output.weight)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "checkpoint.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        restored = ImprovedUNet3D(6, 4).to(device)
        restored.load_state_dict(checkpoint["model"])
        model.eval()
        restored.eval()
        with torch.no_grad():
            assert torch.equal(model(x)[0], restored(x)[0])

    class ConstantModel(torch.nn.Module):
        def forward(self, tensor):
            logits = torch.zeros(tensor.shape[0], 6, *tensor.shape[2:], device=tensor.device)
            logits[:, 3] = 10
            return logits, logits

    volume = np.ones((13, 19, 21), dtype=np.float32)
    prediction = sliding_window_predict(ConstantModel(), volume, (16, 16, 16), 6, device)
    assert prediction.shape == volume.shape and np.all(prediction == 3)
    prediction = sliding_window_predict(model, volume, (16, 16, 16), 6, device)
    assert prediction.shape == volume.shape and prediction.min() >= 0 and prediction.max() < 6
    print(f"SMOKE TEST PASSED ({device}); loss={loss.item():.5f}")


if __name__ == "__main__":
    main()
