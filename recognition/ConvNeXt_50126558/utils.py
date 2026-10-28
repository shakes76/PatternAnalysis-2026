# check if there is leakage of patients between train, validation and test
import os
from dataset import get_dataloaders

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    tr, va, te = get_dataloaders()
    ids = lambda l: {os.path.basename(p).rsplit("_", 1)[0] for p, _ in l.dataset.samples}
    print("train&test:", len(ids(tr) & ids(te)), "val&test:", len(ids(va) & ids(te)), "train&val:", len(ids(tr) & ids(va)))