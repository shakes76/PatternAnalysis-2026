import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from modules import convnext
from dataset import get_dataloaders

# checkpoint path and class names (label 0 = NC, label 1 = AD)
CKPT = "final.pth"
NAMES = ["NC", "AD"]

# shows the regions that push the model to a given class
def gradcam(model, x, cls):
    f = x
    for i in range(4):
        f = model.stages[i](model.downsample_layers[i](f))
    f.retain_grad()
    out = model.head(model.norm(f.mean([-2, -1])))
    out[range(len(cls)), cls].sum().backward()
    cam = F.relu((f.grad.mean([2, 3], keepdim=True) * f).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=224, mode="bilinear")
    return (cam / (cam.amax((2, 3), keepdim=True) + 1e-8))[:, 0].detach().cpu().numpy()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    model = convnext().cuda().eval()
    model.load_state_dict(torch.load(CKPT, map_location="cuda"))
    test_loader = get_dataloaders()[-1]

    # collecting predictions: probability of AD for every test slice
    probs, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            probs.append(model(x.cuda()).softmax(1)[:, 1].cpu())
            labels.append(y)
    probs, labels = torch.cat(probs).numpy(), torch.cat(labels).numpy()
    # threshold at 0.5 to get the predicted class
    preds = (probs > 0.5).astype(int)

    # constructing the 2 x 2 confusion matrix
    cm = confusion_matrix(labels, preds)
    print(f"test images: {len(labels)}  acc {(preds == labels).mean():.4f}  AUROC {roc_auc_score(labels, probs):.4f}")
    print(classification_report(labels, preds, target_names=NAMES, digits=4))
    print(cm)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    # write the count in each cell
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xticks([0, 1], NAMES)
    plt.yticks([0, 1], NAMES)
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.title("confusion matrix")
    plt.tight_layout()
    plt.savefig("predict_confusion.png")

    # choosing sample images: 4 correct, 4 wrong
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(np.where(preds == labels)[0], 4, replace=False),
                          rng.choice(np.where(preds != labels)[0], 4, replace=False)])
    x = torch.stack([test_loader.dataset[i][0] for i in idx]).cuda()
    # computes gradcam for each
    cams = gradcam(model, x, torch.tensor(preds[idx]).cuda())
    imgs = x[:, 0].cpu().numpy() * 0.5 + 0.5

    # sample figure
    _, ax = plt.subplots(2, 8, figsize=(20, 5.5))
    for k, i in enumerate(idx):
        ax[0, k].imshow(imgs[k], cmap="gray")
        # green if correct, red if wrong
        ax[0, k].set_title(f"true {NAMES[labels[i]]} | pred {NAMES[preds[i]]}\np(AD) {probs[i]:.2f}",
                           color="green" if preds[i] == labels[i] else "red", fontsize=10)
        # heatmap at 40% opacity, red/yellow = high influence, blue = low
        ax[1, k].imshow(imgs[k], cmap="gray")
        ax[1, k].imshow(cams[k], cmap="jet", alpha=0.4)
        ax[1, k].set_title("Grad-CAM", fontsize=10)
    for a in ax.ravel():
        a.axis("off")
    plt.suptitle("left 4: correct predictions, right 4: wrong predictions")
    plt.tight_layout()
    plt.savefig("predict_samples.png")