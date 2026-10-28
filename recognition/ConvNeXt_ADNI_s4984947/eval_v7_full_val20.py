import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import ADNIDataset
from dataset_v2 import build_splits
from train_v3 import validate
from v7_pretrained import load_custom_pretrained


def main():
    assert torch.cuda.is_available(), "GPU required"
    device = torch.device("cuda")

    # Only validation patients; the test set remains untouched.
    splits = build_splits(seed=3710)
    loader = DataLoader(
        ADNIDataset(splits["val"]),
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Recreate the pretrained architecture, then load the
    # best fine-tuned checkpoint from epoch 13.
    model = load_custom_pretrained(
        "pretrained/convnext_tiny-983f1562.pth",
        num_classes=2
    ).to(device)

    checkpoint = torch.load(
        "runs/convnext_v7_pretrained20_newsplit/best.pt",
        map_location="cpu",
        weights_only=True
    )
    model.load_state_dict(
        checkpoint["model_state_dict"], strict=True
    )
    model.eval()

    result = validate(
        model, loader,
        nn.CrossEntropyLoss(label_smoothing=0.05),
        device
    )

    tn, fp, fn, tp = result["slice"]["confusion_TN_FP_FN_TP"]

    print("\n===== V7 FULL VALIDATION =====")
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Validation slices:", result["validation_slices"])
    print("Validation patients:", result["validation_patients"])
    print("Slice Accuracy:",
          f'{result["slice"]["accuracy"]:.2%}')
    print("Slice Macro-F1:",
          result["slice"]["macro_f1"])
    print("Patient Accuracy:",
          f'{result["patient"]["accuracy"]:.2%}')
    print("AD Recall:",
          f'{tp / max(1, tp + fn):.2%}')
    print("NC Recall:",
          f'{tn / max(1, tn + fp):.2%}')
    print("Confusion [TN, FP, FN, TP]:",
          [tn, fp, fn, tp])
    print("Validation Loss:", result["loss"])

    out = Path(
        "runs/convnext_v7_pretrained20_newsplit/full_val20.json"
    )
    out.write_text(json.dumps(result, indent=2))
    print("Saved:", out)


if __name__ == "__main__":
    main()
