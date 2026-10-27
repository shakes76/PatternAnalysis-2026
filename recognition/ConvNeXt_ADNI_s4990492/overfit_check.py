"""Check whether the baseline can fit a small fixed training batch."""

from pathlib import Path

import torch
from torch import nn

from dataset import ADNISliceDataset
from modules import SimpleCNN


def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)

    if not torch.cuda.is_available():
        raise RuntimeError("Run this script in a Slurm GPU job.")

    base = Path(__file__).resolve().parent
    dataset = ADNISliceDataset(base / "splits" / "train.csv")

    # Select eight different patients from each class.
    generator = torch.Generator().manual_seed(42)
    order = torch.randperm(
        len(dataset), generator=generator
    ).tolist()

    selected = []
    seen_patients = set()
    counts = {0: 0, 1: 0}

    for index in order:
        row = dataset.rows[index]
        label = int(row["target"])
        patient = row["patient_id"]

        if counts[label] >= 8 or patient in seen_patients:
            continue

        selected.append(index)
        seen_patients.add(patient)
        counts[label] += 1

        if counts == {0: 8, 1: 8}:
            break

    if len(selected) != 16:
        raise RuntimeError("Could not select 16 distinct patients.")

    samples = [dataset[index] for index in selected]
    images = torch.stack([sample[0] for sample in samples]).cuda()
    labels = torch.stack([sample[1] for sample in samples]).cuda()

    model = SimpleCNN().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("GPU:", torch.cuda.get_device_name(0))
    print("Fixed batch shape:", tuple(images.shape))
    print("Distinct patients:", len(seen_patients))
    print("Class counts:", counts)

    def report(step):
        model.eval()
        with torch.no_grad():
            logits = model(images)
            loss = criterion(logits, labels).item()
            accuracy = (
                logits.argmax(dim=1) == labels
            ).float().mean().item()

        print(
            f"Step {step:03d} | "
            f"fixed-batch loss {loss:.4f} | "
            f"fixed-batch accuracy {accuracy:.1%}",
            flush=True,
        )
        model.train()

    report(0)

    for step in range(1, 301):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)

        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Non-finite loss at step {step}")

        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            report(step)

    print("Check finished; no checkpoint saved.")
    print("These results concern training samples only.")


if __name__ == "__main__":
    main()