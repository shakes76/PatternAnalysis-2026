import argparse , os , json, random ,re
from pathlib import Path
import numpy as np
import  nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

NUM_CLASSES = 6
CLASS_NAMES = ["background", "body", "bone", "bladder", "rectum", "prostate"]
PROSTATE = 5
DEFAULT_ROOT = (
    "data/2021-09-20_Dowling_Jason_51392v2/data/HipMRI_study_complete_release_v1")

Scan_Re = re.compile(r"([A-Za-z]+_?\d+)_(Week\d+)")

#match every Mri image and labels to its own by using scan id.
def find_pairs(root=DEFAULT_ROOT):
    root = Path(root)

    img_dir = root / "semantic_MRs_anon"
    lbl_dir = root / "semantic_labels_anon"
    labels = {}
    for path in lbl_dir.glob("*.nii.gz"):
        match = Scan_Re.search(path.name)
        if match:
            labels[match.group(0)] = path


    pairs = []
    for path in sorted(img_dir.glob("*.nii.gz")):
        match = Scan_Re.search(path.name)
        if match is None:
            continue
        scan_id = match.group(0)
        if scan_id not in labels:
             raise FileNotFoundError(f"No label volume found for {path.name}")
        pairs.append({
            "case":match.group(1),
            "scan":scan_id,
            "image":str(path),
            "label":str(labels[scan_id])
        })

    if not pairs:
        raise FileNotFoundError(f"No image-label pairs found in {root}")
    return pairs


#Split the dataset into train, validation and test sets by patient id.
def patient_split(pairs,val_frac = 0.15,test_frac = 0.15 ,seed=42):
    cases = sorted({p["case"]for p in pairs})
    random.Random(seed).shuffle(cases)

    n_test = max(1,round(len(cases)*test_frac))
    n_val = max(1,round(len(cases)*val_frac))
    groups = {
        "test":set(cases[:n_test]),
        "val":set(cases[n_test:n_test+n_val]),
        "train":set(cases[n_test+n_val:]),
    }

    assert not (groups["train"] & groups["val"])
    assert not (groups["train"] & groups["test"])
    assert not (groups["val"] & groups["test"])

    return {name: [p for p in pairs if p["case"] in ids]
            for name, ids in groups.items()}


def save_split(split, path="outputs/split.json"):
    """Save the patient IDs and scans in each split, for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        name: {
            "patients": sorted({p["case"] for p in items}),
            "scans": [p["scan"] for p in items],
        }
        for name, items in split.items()
    }
    path.write_text(json.dumps(summary, indent=2))
    return path


def load_volume(path):
    vol = np.asarray(nib.load(path).dataobj,dtype=np.float32)
    if vol.ndim == 4:
        vol = vol[...,0]
    return vol

def preprocess(image,label,scale= 0.5):
    img = torch.from_numpy(image)[None,None]
    lbl = torch.from_numpy(label)[None,None]

    if scale != 1.0:
        size = [max(1,round(s*scale)) for s in img.shape[2:]]
        img = F.interpolate(img,size=size,mode="trilinear",align_corners=False)
        lbl = F.interpolate(lbl,size=size,mode="nearest")

    img = img[0]
    img = (img - img.mean()) / (img.std() + 1e-8)

    lbl = lbl[0,0].round().long()
    if lbl.min() <0 or lbl.max()>= NUM_CLASSES:
        raise ValueError(f"Label volume has values outside [0,{NUM_CLASSES-1}]")
    return img, lbl

#These part is for data augmentation, including random flipping and scaling of the image and label volumes.

def augment(img,lbl,flip_axis = 0):

    if torch.rand(1) <0.5:
        img = torch.flip(img,[flip_axis+1])
        lbl = torch.flip(lbl,[flip_axis])
    scale = 1.0 + 0.1 * (2 * torch.rand(1).item() - 1)
    shift = 0.1 * (2 * torch.rand(1).item() - 1)
    return img * scale + shift, lbl

# Randomly crop a volume and its label to the given size, with a 50% chance of centering the crop on the prostate region.
def random_crop(img,lbl,size):
    dims = lbl.shape
    if any(c > d for c, d in zip(size, dims)):
        raise ValueError(f"Crop size {size} is larger than volume size {dims}")
    
    fg = (lbl == PROSTATE).nonzero()
    if len(fg) and torch.rand(1).item() < 0.5:
        centre = fg[torch.randint(len(fg), (1,)).item()].tolist()
        start = [min(max(c - s // 2, 0), d - s)
                 for c, s, d in zip(centre, size, dims)]
    else:
        start = [torch.randint(0, d - s + 1, (1,)).item()
                 for s, d in zip(size, dims)]
 
    sl = tuple(slice(st, st + s) for st, s in zip(start, size))
    return img[(slice(None),) + sl], lbl[sl]
class HipMRIDataset2D(Dataset):

    def __init__(self,pairs,scale=0.5,train=False,crop_size=None):
        self.pairs = pairs
        self.train = train
        self.crop_size = tuple(crop_size) if crop_size else None
        self.images,self.labels = [],[]
        self.index = []

        for v,p in enumerate(pairs):
            image,label = load_volume(p["image"]),load_volume(p["label"])
            if image.shape != label.shape:
                raise ValueError(f"Image and label shapes do not match for {p['scan']}: "
                                 f"image:{image.shape} != label:{label.shape}")
            img,lbl = preprocess(image,label,scale)
            self.images.append(img)
            self.labels.append(lbl.to(torch.uint8))
            self.index.extend((v,z)for z in range(lbl.shape[-1]))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        v,z = self.index[idx]
        img = self.images[v][...,z]
        lbl = self.labels[v][...,z].long()

        if self.train:
            img,lbl = augment(img,lbl)
            if self.crop_size:
                img,lbl = random_crop(img,lbl,self.crop_size)
        p = self.pairs[v]
        return {"image": img, "label": lbl, "scan": p["scan"], "case": p["case"], "slice": z}    

class HipMRIDataset3D(Dataset):
    def __init__(self,pairs,scale = 0.5,train = False,crop_size =None,cache = False):
        self.pairs = pairs
        self.scale = scale
        self.train = train
        self.crop_size = tuple(crop_size)if crop_size else None
        self.cache = cache
        self.cache_data = {}

    def __len__(self):
        return len(self.pairs)

    def load(self,idx):
        if idx in self.cache_data:
            return self.cache_data[idx]
        p = self.pairs[idx]
        image,label = load_volume(p["image"]),load_volume(p["label"])
        if image.shape != label.shape:
            raise ValueError(f"Image and label shapes do not match for {p['scan']}: "
                             f"image:{image.shape} != label:{label.shape}")

        out = preprocess(image,label,self.scale)
        if self.cache:
            self.cache_data[idx] = out
        return out

    def __getitem__(self,idx):
        img,lbl = self.load(idx)
        if self.train:
            img,lbl = augment(img,lbl)
            if self.crop_size:
                img,lbl = random_crop(img,lbl,self.crop_size)
        return {"image": img, "label": lbl,"scan":self.pairs[idx]["scan"],"case":self.pairs[idx]["case"]}


def get_dataloader(root=DEFAULT_ROOT,dims =2,batch_size=1,scale=0.5,crop_size=None,num_workers=0,subset =0,seed = 42):
    if dims not in(2,3):
         raise ValueError("dims must be 2 or 3")
    split = patient_split(find_pairs(root),seed=seed)
    if subset:
        split = {k: v[:subset]for k,v in split.items()}

    Ds = HipMRIDataset2D if dims == 2 else HipMRIDataset3D
    train_ds = Ds(split["train"],scale=scale,train=True,crop_size=crop_size)
    val_ds = Ds(split["val"],scale=scale,train=False,crop_size=crop_size)
    test_ds = Ds(split["test"],scale=scale,train=False,crop_size=crop_size)

    loaders = {
        "train": DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=num_workers),
        "val": DataLoader(val_ds,batch_size=batch_size,shuffle=False,num_workers=num_workers),
        "test": DataLoader(test_ds,batch_size=batch_size,shuffle=False,num_workers=num_workers)
    }
    return loaders , split

def audit(root, scale, subset):
    pairs = find_pairs(root)
    split = patient_split(pairs)
    print(f"Found {len(pairs)} scans from {len({p['case'] for p in pairs})} patients")
    for name, items in split.items():
        print(f"  {name:5s}: {len({p['case'] for p in items}):2d} patients, "
              f"{len(items):3d} scans")
    print("Split saved to", save_split(split))
 
    first = nib.load(pairs[0]["image"])
    print("Orientation (axis codes):", nib.aff2axcodes(first.affine))
   
    ds = HipMRIDataset3D(split["train"][:subset] if subset else split["train"], scale=scale)
    counts = torch.zeros(NUM_CLASSES)
    n_slices = n_prostate_slice = 0
    for i in range(len(ds)):
        img, lbl = ds.load(i)
        counts += torch.bincount(lbl.flatten(), minlength=NUM_CLASSES).float()
        n_slices += lbl.shape[-1]
        n_prostate_slice += int((lbl == PROSTATE).any(dim=0).any(dim=0).sum())
    print(f"Volume after scale={scale}: image {tuple(img.shape)}, label {tuple(lbl.shape)}")
    print(f"2D slices: {n_slices} training slices of {tuple(lbl.shape[:2])}, "
        f"{n_prostate_slice} ({100 * n_prostate_slice / n_slices:.1f}%) contain prostate")
    print(f"Class voxel fractions over {len(ds)} training scans:")
    for name, frac in zip(CLASS_NAMES, counts / counts.sum()):
        print(f"  {name:10s} {100 * frac:6.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HipMRI 3D data audit")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--subset", type=int, default=0, help="Limit to this many scans per split (0=all)")
    parser.add_argument("--quick", action="store_true",
                        help="quick test run on 4 scans")

    args = parser.parse_args()
    if args.quick:
        args.subset = 4

    audit(args.root, args.scale, args.subset)