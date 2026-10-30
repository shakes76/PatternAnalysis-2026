import torch
from dataset import get_dataloaders
from modules import ConvNeXt
 
CKPT = "convnext3.pt"
PID = None  # patient ID as a string to inspect; None = the missed AD patient with the lowest mean P(AD)
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
 
    # patient id -> [(slice number, P(AD), true label), ...]
    patients = {}
    for p, (path, label, pid) in zip(probs, loader.dataset.records):
        patients.setdefault(pid, []).append((int(path.split("_")[-1].split(".")[0]), p, label))
 
    mean = lambda pid: sum(s[1] for s in patients[pid]) / len(patients[pid])
    missed = sorted((q for q in patients if patients[q][0][2] == 1 and mean(q) < 0.45), key=mean)
    print(f"Missed AD patients (mean P(AD) < 0.5): {missed}")
    avg_thing = 0
    for i in range(len(missed)):
        pid = PID or (missed[i] if missed else None)
        if pid is None:
            return
        slices = sorted(patients[pid])
        called_ad = sum(p >= 0.45 for _, p, _ in slices)
        print(f"\nPatient {pid}: mean P(AD) {mean(pid):.3f}, {called_ad}/{len(slices)} slices called AD")
        avg_thing += mean(pid)
    print(f"Average mean P(AD) {avg_thing/len(missed)} Lowest P(AD) {mean(missed[0])}")
 
 
if __name__ == "__main__":
    main()
