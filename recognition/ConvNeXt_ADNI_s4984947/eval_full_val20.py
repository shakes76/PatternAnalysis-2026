"""Evaluate saved best checkpoints on ALL validation slices."""
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import ADNIDataset
from dataset_v2 import build_splits
from modules import CNNBaseline, ConvNeXtClassifier
from train_v3 import validate


def main():
    assert torch.cuda.is_available(), "GPU required"
    device = torch.device("cuda")

    splits = build_splits(seed=3710)

    # All validation slices, not the selected 6 per scan.
    rows = splits["val"]

    loader = DataLoader(
        ADNIDataset(rows),
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    experiments = [
        (
            "CNN V3",
            CNNBaseline,
            "runs/baseline_v3_15epochs/best.pt"
        ),
        (
            "ConvNeXt V5",
            ConvNeXtClassifier,
            "runs/convnext_v5_smoothing20_newsplit/best.pt"
        )
    ]

    results = {}

    for name, model_class, checkpoint_path in experiments:
        model = model_class().to(device)
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )
        model.load_state_dict(checkpoint["model_state_dict"])

        result = validate(
            model,
            loader,
            nn.CrossEntropyLoss(),
            device
        )

        results[name] = result

        cm = result["slice"]["confusion_TN_FP_FN_TP"]
        tn, fp, fn, tp = cm

        print(f"\n=== {name}: Full Validation ===", flush=True)
        print("Slices:", result["validation_slices"], flush=True)
        print("Patients:", result["validation_patients"], flush=True)
        print("Slice Accuracy:",
              f'{result["slice"]["accuracy"]:.2%}', flush=True)
        print("Slice Macro-F1:",
              result["slice"]["macro_f1"], flush=True)
        print("Patient Accuracy:",
              f'{result["patient"]["accuracy"]:.2%}', flush=True)
        print("AD Recall:",
              f'{tp / max(1, tp + fn):.2%}', flush=True)
        print("Confusion:", cm, flush=True)
        del model
        torch.cuda.empty_cache()

    output = Path("runs/full_validation_20_slices.json")
    output.write_text(json.dumps(results, indent=2))
    print("\nSaved:", output, flush=True)


if __name__ == "__main__":
    main()
