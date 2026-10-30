import nibabel as nib
import os,glob,torch,random
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F

class HipMRIDataset(Dataset):
    def __init__(self,data_path,split="train"):
        self.image_dir = os.path.join(data_path, "semantic_MRs")
        self.mask_dir = os.path.join(data_path,"semantic_labels_only")\

        all_files = sorted(glob.glob(os.path.join(self.image_dir, "*_LFOV.nii.gz")))

        #extract patient ID form Filenames
        patients = sorted(set(os.path.basename(f).split("_")[0] for f in all_files))
        rng = random.Random(42)
        rng.shuffle(patients)

        #Splitting patients into train,validation and testing
        n = len(patients)
        n_train = int(n*0.70)
        n_val = int(n*0.15)

        patient_splits = {"train": set(patients[:n_train]),
                          "validate": set(patients[n_train:n_train + n_val]),
                          "test": set(patients[n_train + n_val:])}

        if split not in patient_splits:
            raise ValueError(f"Unknown split: {split}")

        selected_patients = patient_splits[split]
        self.image_files = [f for f in all_files
        if os.path.basename(f).split("_")[0] in selected_patients]
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        #get the MRI files
        image_path = self.image_files[idx]

        #find its segmentation mask
        filename = os.path.basename(image_path)
        mask_filename = filename.replace("_LFOV","_SEMANTIC")
        mask_path = os.path.join(self.mask_dir, mask_filename)

        #load MRI and mask using nibabel
        image = nib.load(image_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()

        #normalize MRI intensity using zscore
        image = (image - image.mean()) / (image.std() + 1e-8)

        #Conversion of image into pytorch
        image =torch.tensor(image,dtype=torch.float32)
        mask = torch.tensor(mask,dtype=torch.long)

        #Add dimension to the image
        image = (F.interpolate(image.unsqueeze(0).unsqueeze(0),
                              size = (128,128,64),
                              mode = "trilinear",
                              align_corners=False
                               ).squeeze(0))

        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=(128, 128,64),
            mode="nearest"
        ).squeeze(0).squeeze(0).long()

        return image,mask


if __name__ == "__main__":
    data_path = "/home/groups/comp3710/HipMRI_Study_open"

    for split in ["train", "validate", "test"]:
        dataset = HipMRIDataset(data_path, split=split)
        print(f"{split}: {len(dataset)} volumes")