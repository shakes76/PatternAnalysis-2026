import torch


@torch.no_grad()
def collect_probs(model, loader, device):
    """Softmax probabilities and true labels for every item in a loader."""
    model.eval()
    probs, labels = [], []
    for images, y in loader:
        logits = model(images.to(device))
        probs.append(torch.softmax(logits, dim=1).cpu())
        labels.append(y)
    return torch.cat(probs), torch.cat(labels)
