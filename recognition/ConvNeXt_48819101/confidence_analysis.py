"""Confidence analysis: how confident is the model when it is right vs wrong?"""
import torch
from dataset import get_dataloaders
from modules import ConvNeXt

CKPT = "convnext3.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    model = ConvNeXt(num_classes=1).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
    model.eval()

    loader = get_dataloaders(num_workers=0)[2]  # test loader (not shuffled)
    probs = []
    with torch.no_grad():
        for x, _ in loader:
            probs += torch.sigmoid(model(x.to(device)).squeeze(1)).tolist()
    labels = [r[1] for r in loader.dataset.records]

    conf = [max(p, 1 - p) for p in probs]  # probability given to the predicted class
    right = [int(p >= 0.5) == y for p, y in zip(probs, labels)]
    c_right = [c for c, r in zip(conf, right) if r]
    c_wrong = [c for c, r in zip(conf, right) if not r]
    avg = lambda xs: sum(xs) / len(xs) if xs else float("nan")

    acc, mean_conf = len(c_right) / len(right), avg(conf)
    print(f"{len(right)} slices | accuracy {acc:.4f}")
    print(f"mean confidence, all:     {mean_conf:.4f}  (confidence - accuracy = {mean_conf - acc:+.4f}, positive = overconfident)")
    print(f"mean confidence, correct: {avg(c_right):.4f}  (n={len(c_right)})")
    print(f"mean confidence, wrong:   {avg(c_wrong):.4f}  (n={len(c_wrong)})")
    print(f"wrong predictions with confidence >= 0.9: {sum(c >= 0.9 for c in c_wrong)}")


if __name__ == "__main__":
    main()