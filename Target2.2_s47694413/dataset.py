from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from pathlib import Path
import nibabel as nib
import torch.nn.functional as F
'''
Dataset class for loading images and masks from the OASIS dataset.
'''
class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_path, mask_path):
        self.image_path = Path(image_path)
        self.mask_path = Path(mask_path)

        self.image_files = sorted(self.image_path.glob("*.nii.gz"))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]

        # Convert image filename to mask filename
        mask_name = img_path.name.replace("case_", "seg_", 1)
        mask_file = self.mask_path / mask_name

        # Load images  
        image = torch.as_tensor(nib.load(str(img_path)).get_fdata())
        mask = torch.as_tensor(nib.load(str(mask_file)).get_fdata())
        #image = self.to_tensor(image)
        image = image.unsqueeze(0).unsqueeze(0)  # Add channel dimension
        mask = mask.unsqueeze(0).unsqueeze(0)  # Add channel dimension
        #mask = torch.from_numpy(mask.astype(np.int64))

        image = F.interpolate(image, size=(256, 128), mode='bilinear', align_corners=False).squeeze(0)
        mask = F.interpolate(mask.float(), size=(256, 128), mode='nearest').squeeze(0).squeeze(0).long()
        return image, mask


def get_dataloaders():
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "Data" / "HIP MRI" / "keras_slices_data"


    train_path = DATA_DIR / "keras_slices_train"

    # Training dataset
    HipTrainset = Dataset(image_path=DATA_DIR / "keras_slices_train",mask_path=DATA_DIR / "keras_slices_seg_train")
    HipTrainLoader = torch.utils.data.DataLoader(HipTrainset, batch_size=16, shuffle=True)

    # Validation dataset
    HipValset = Dataset(image_path=DATA_DIR / "keras_slices_validate",mask_path=DATA_DIR / "keras_slices_seg_validate")
    HipValLoader = torch.utils.data.DataLoader(HipValset, batch_size=16, shuffle=False)

    # Test dataset
    HipTestset = Dataset(image_path=DATA_DIR / "keras_slices_test",mask_path=DATA_DIR / "keras_slices_seg_test")
    HipTestLoader = torch.utils.data.DataLoader(HipTestset, batch_size=16,shuffle=False)
    return HipTrainLoader, HipValLoader, HipTestLoader, HipTrainset, HipValset, HipTestset