"""Check installed libraries and CPU/CUDA GCN operations without a dataset.

The tiny artificial tensors below check software compatibility only. They are
not Facebook data, model training, or assessment performance evidence.
"""

import argparse
import io
import platform
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
import torch_geometric
from sklearn.metrics import accuracy_score
from torch_geometric.nn import GCNConv


def check_operators(device: torch.device) -> None:
    """Exercise matrix operations, graph convolution and autograd on a device."""
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device)
    edges = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=device)
    layer = GCNConv(2, 2).to(device)
    output = layer(features, edges)
    assert output.shape == (2, 2)
    assert torch.isfinite(output).all()
    output.sum().backward()
    for parameter in layer.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    product = features @ features.T
    assert torch.allclose(product, torch.eye(2, device=device))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    print(f"PASS: tensor operations, GCNConv forward and autograd on {device}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-cuda", action="store_true", help="Fail if CUDA is unavailable."
    )
    args = parser.parse_args()
    print(f"Python: {platform.python_version()}; executable: {sys.executable}")
    for name, version in (
        ("PyTorch", torch.__version__),
        ("PyG", torch_geometric.__version__),
        ("NumPy", np.__version__),
        ("scikit-learn", sklearn.__version__),
        ("Matplotlib", matplotlib.__version__),
    ):
        print(f"{name}: {version}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    torch.manual_seed(42)
    check_operators(torch.device("cpu"))
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        check_operators(torch.device("cuda:0"))
    elif args.require_cuda:
        raise RuntimeError("CUDA was required but torch.cuda.is_available() is False")
    else:
        print("CUDA unavailable; CPU checks only")
    assert accuracy_score([0, 1], [0, 1]) == 1.0
    assert np.array_equal(np.asarray([0, 1]), np.arange(2))
    array = np.asarray([1.0, 2.0], dtype=np.float32)
    assert np.array_equal(torch.from_numpy(array).numpy(), array)
    figure, axes = plt.subplots()
    axes.plot([0, 1], [0, 1])
    with io.BytesIO() as buffer:
        figure.savefig(buffer, format="png")
        assert buffer.tell() > 0
    plt.close(figure)
    print("PASS: metrics, NumPy/PyTorch bridge and headless PNG rendering")
    print("Environment checks complete; no dataset or experiment was run.")


if __name__ == "__main__":
    main()
