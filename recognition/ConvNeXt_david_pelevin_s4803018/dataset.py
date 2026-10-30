import torch, torchvision
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

## Calculated using the ADNIDatasetFull class
full_dataset_mean = torch.tensor(29.5524, dtype=torch.float32)
full_dataset_std = torch.tensor(57.5198, dtype=torch.float32)

class ADNIDataset(Dataset):
    """
    Constructs a dataset from a given subset of the ADNI data.

    Dataset class creation learnt in previous demo
    """
    def __init__(self, root_directory, secondary_root_directory, ad_directory, nc_directory, patients=None):
        """
        Initalises the dataset.
        
        Assumes the following directory structure (the quotation marks should not be present):

            root_directory
                /ad_directory
                    "patient_id"_"slice_id".jpeg
                /nc_directory
                    "patient_id"_"slice_id".jpeg
            secondary_root_directory
                /ad_directory
                    "patient_id"_"slice_id".jpeg
                /nc_directory
                    "patient_id"_"slice_id".jpeg

        Parameters:
            root_directory (str): The name of the first directory holding ADNI data
            secondary_root_directory (str): The name of the second directory holding ADNI data
            ad_directory (str): The name of the subdirectory containing Alzheimer's Disease slices
            nc_directory (str): The name of the subdirectory containing Cognitively Normal slices
            patients (list[str]): The patients to include in this split.
                                  Can be left as None to include all patients.
        """
        self.images = []
        self.labels = []
        self.file_names = []

        ## Construct the paths to retrieve the slices
        ad_path = Path(root_directory + "/" + ad_directory)
        nc_path = Path(root_directory + "/" + nc_directory)
        ad_path_2 = Path(secondary_root_directory + "/" + ad_directory)
        nc_path_2 = Path(secondary_root_directory + "/" + nc_directory)

        for path in sorted(ad_path.glob("*.jpeg")):
            ## Filenames are "patient_id"_"slice_id".jpeg
            patient_id = path.name.split(".")[0].split("_")[0]

            if (patients != None) and (patient_id not in patients):
                ## Not in this split, ignore it.
                continue

            image = Image.open(path).convert("L")

            ## Transformation of image to tensor adapted from
            ## https://docs.pytorch.org/vision/main/generated/torchvision.transforms.functional.pil_to_tensor.html
            image_tensor = torchvision.transforms.functional.pil_to_tensor(image).to(torch.float32)

            self.images.append(image_tensor)
            self.labels.append(torch.tensor(1))
            self.file_names.append(path.name.split(".")[0])

        for path in sorted(nc_path.glob("*.jpeg")):
            ## Filenames are "patient_id"_"slice_id".jpeg
            patient_id = path.name.split(".")[0].split("_")[0]

            if (patients != None) and (patient_id not in patients):
                ## Not in this split, ignore it.
                continue
            
            image = Image.open(path).convert("L")

            ## Transformation of image to tensor adapted from
            ## https://docs.pytorch.org/vision/main/generated/torchvision.transforms.functional.pil_to_tensor.html
            image_tensor = torchvision.transforms.functional.pil_to_tensor(image).to(torch.float32)

            self.images.append(image_tensor)
            self.labels.append(torch.tensor(0))
            self.file_names.append(path.name.split(".")[0])
        
        for path in sorted(ad_path_2.glob("*.jpeg")):
            ## Filenames are "patient_id"_"slice_id".jpeg
            patient_id = path.name.split(".")[0].split("_")[0]

            if (patients != None) and (patient_id not in patients):
                ## Not in this split, ignore it.
                continue

            image = Image.open(path).convert("L")

            ## Transformation of image to tensor adapted from
            ## https://docs.pytorch.org/vision/main/generated/torchvision.transforms.functional.pil_to_tensor.html
            image_tensor = torchvision.transforms.functional.pil_to_tensor(image).to(torch.float32)

            self.images.append(image_tensor)
            self.labels.append(torch.tensor(1))
            self.file_names.append(path.name.split(".")[0])

        for path in sorted(nc_path_2.glob("*.jpeg")):
            ## Filenames are "patient_id"_"slice_id".jpeg
            patient_id = path.name.split(".")[0].split("_")[0]

            if (patients != None) and (patient_id not in patients):
                ## Not in this split, ignore it.
                continue
            
            image = Image.open(path).convert("L")
            
            ## Transformation of image to tensor adapted from
            ## https://docs.pytorch.org/vision/main/generated/torchvision.transforms.functional.pil_to_tensor.html
            image_tensor = torchvision.transforms.functional.pil_to_tensor(image).to(torch.float32)

            self.images.append(image_tensor)
            self.labels.append(torch.tensor(0))
            self.file_names.append(path.name.split(".")[0])

    def __len__(self):
        """
        Returns the number of images in the dataset.

        Returns:
            int: The length of the dataset
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieves the item in the dataset with that index.

        Parameters:
            idx (int): The index of the item to be retrieved.

        Returns:
            Tensor (C, H, W), Tensor (int): The image pixel tensor and
            class label tensor of the image.
        """
        return self.images[idx], self.labels[idx]

    def getfilename(self, idx):
        """
        Retrieves the filename of the item in the dataset with that index.

        Parameters:
            idx (int): The index of the item whose name is to be retrieve.
    
        Returns:
            str: The filename of the image (without the path).
        """
        return self.file_names[idx]


def normalise_image(image, dtype=torch.float32):
    """
    Normalises the images based off the global distribution.

    Based off code from Shakes for previous demo

    Parameters:
        image (Tensor (C, H, W)): The image to normalise
        dtype: The datatype to store the image as

    Returns:
        image (Tensor (C, H, W)): The normalised image tensor
    """
    image = image.to(dtype)
    _mean = full_dataset_mean.to(dtype=dtype, device=image.device)
    _std = full_dataset_std.to(dtype=dtype, device=image.device)
    return (image - _mean) / _std

## Based off code from Shakes for previous demo
def preprocess_image(image, crop_pad=32):
    """
    Preprocesses the images by normalising, randomly flipping,
    and then performing a padded crop.

    Based off code from Shakes for previous demo

    Parameters:
        image (Tensor (C, H, W)): The image to normalise
        crop_pad (int): The size of the possible translation when cropping

    Returns:
        image (Tensor (C, H, W)): The normalised image tensor
    """
    image = normalise_image(image)

    ## horizontal flip
    if torch.rand(()) < 0.5:
        image = torch.flip(image, dims=[2])

    _, height, width = image.shape

    ## pad (reflect) then random crop back to the original size
    image = F.pad(image.unsqueeze(0), [crop_pad, crop_pad, crop_pad, crop_pad], mode='reflect')
    image = image.squeeze(0)
    top = int(torch.randint(0, 2 * crop_pad + 1, (1,)))
    left = int(torch.randint(0, 2 * crop_pad + 1, (1,)))
    image = image[:, top:top + height, left:left + width]

    return image

class ADNI_train(ADNIDataset):
    """
    Constructs a ADNIDataset with full image preprocessing.
    """
    def __getitem__(self, idx):
        """
        Retrieves the item in the dataset with that index.
        Has preprocessing built into it.
        
        Parameters:
            idx (int): The index of the item to be retrieved.

        Returns:
            Tensor (C, H, W), Tensor (int): The image pixel tensor and
            class label tensor of the image.
        """
        image, label = self.images[idx], self.labels[idx]
        image = preprocess_image(image)
        return image, label

class ADNI_test(ADNIDataset):
    """
    Constructs a ADNIDataset with only normalisation.
    """
    def __getitem__(self, idx):
        """
        Retrieves the item in the dataset with that index.
        Has normalisation built into it.
        
        Parameters:
            idx (int): The index of the item to be retrieved.

        Returns:
            Tensor (C, H, W), Tensor (int): The image pixel tensor and
            class label tensor of the image.
        """
        image, label = self.images[idx], self.labels[idx]
        image = normalise_image(image)
        return image, label

class ADNIDatasetFull(Dataset):
    """
    Constructs a dataset from all the ADNI data to determine normalisation constants.

    Dataset class creation learnt in previous demo
    """
    def __init__(self, train_root_directory, test_root_directory, ad_directory, nc_directory):
        """
        Initalises the dataset and calculates the mean and std of the dataset.
        
        Assumes the following directory structure (the quotation marks should not be present):

            root_directory
                /ad_directory
                    "patient_id"_"slice_id".jpeg
                /nc_directory
                    "patient_id"_"slice_id".jpeg
            secondary_root_directory
                /ad_directory
                    "patient_id"_"slice_id".jpeg
                /nc_directory
                    "patient_id"_"slice_id".jpeg

        Parameters:
            root_directory (str): The name of the first directory holding ADNI data
            secondary_root_directory (str): The name of the second directory holding ADNI data
            ad_directory (str): The name of the subdirectory containing Alzheimer's Disease slices
            nc_directory (str): The name of the subdirectory containing Cognitively Normal slices
        """
        self.images = []
        self.labels = []
        self.mean = 0
        self.std = 0

        total_sum = 0.0
        total_sum_sq = 0.0
        total_pixels = 0

        train_ad_path = Path(train_root_directory + "/" + ad_directory)
        train_nc_path = Path(train_root_directory + "/" + nc_directory)
        test_ad_path = Path(test_root_directory + "/" + ad_directory)
        test_nc_path = Path(test_root_directory + "/" + nc_directory)

        paths = [train_ad_path, train_nc_path, test_ad_path, test_nc_path]

        for i in range(4):
            for path in sorted(paths[i].glob("*.jpeg")):
                image = Image.open(path).convert("L")

                ## Transformation of image to tensor adapted from
                ## https://docs.pytorch.org/vision/main/generated/torchvision.transforms.functional.pil_to_tensor.html
                image_tensor = torchvision.transforms.functional.pil_to_tensor(image).to(torch.float32)

                self.images.append(image_tensor)
                self.labels.append(torch.tensor(1 - (i % 2)))

                total_sum += image_tensor.sum()
                total_sum_sq += (image_tensor**2).sum()
                total_pixels += image_tensor.size(1) * image_tensor.size(2)
            print("Path complete")
        
        self.mean = total_sum / total_pixels
        self.std = torch.sqrt((total_sum_sq / total_pixels) - (self.mean**2))

    def __len__(self):
        """
        Returns the number of images in the dataset.

        Returns:
            int: The length of the dataset
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieves the item in the dataset with that index.

        Parameters:
            idx (int): The index of the item to be retrieved.

        Returns:
            Tensor (C, H, W), Tensor (int): The image pixel tensor and
            class label tensor of the image.
        """
        return self.images[idx], self.labels[idx]

    def get_mean(self):
        """
        Retrieves the mean of the dataset.

        Returns:
            Tensor (int): A Tensor containing the mean of pixels in the dataset.
        """
        return self.mean

    def get_std(self):
        """
        Retrieves the std of the dataset.

        Returns:
            Tensor (int): A Tensor containing the std of pixels in the dataset.
        """
        return self.std

def record_unique_patients(directory):
    """
    Creates a list of all patients within the directory.

    Parameters:
        directory (Path): The path to check for slices.
                          Slices must be in the given folder.

    Returns:
        list[str]: A list of unique patients.
    """
    patients = []

    for path in sorted(directory.glob("*.jpeg")):
        file_name = path.name.split(".")[0]
        patient = file_name.split("_")[0]
        if patient not in patients: patients.append(patient)

    return patients

def data_audit(train_root_directory, test_root_directory, ad_directory, nc_directory):
    """
    Determines the number of patients in each subsplit and
    checks the splits for any shared patients.

    Assumes the following directory structure (the quotation marks should not be present):
    
        root_directory
            /ad_directory
                "patient_id"_"slice_id".jpeg
            /nc_directory
                "patient_id"_"slice_id".jpeg
        secondary_root_directory
            /ad_directory
                "patient_id"_"slice_id".jpeg
            /nc_directory
                "patient_id"_"slice_id".jpeg

    Parameters:
        train_root_directory (str): The name of the directory holding ADNI training split
        test_root_directory (str): The name of the directory holding ADNI test split
        ad_directory (str): The name of the subdirectory containing Alzheimer's Disease slices
        nc_directory (str): The name of the subdirectory containing Cognitively Normal slices
    """
    all_patients = [0, 0, 0, 0]
    patient_groups = ["AD Train", "NC Train", "AD Test", "NC Test"]

    all_patients[0] = set(record_unique_patients(Path(train_root_directory + "/" + ad_directory)))
    all_patients[1] = set(record_unique_patients(Path(train_root_directory + "/" + nc_directory)))
    all_patients[2] = set(record_unique_patients(Path(test_root_directory + "/" + ad_directory)))
    all_patients[3] = set(record_unique_patients(Path(test_root_directory + "/" + nc_directory)))

    for i in range(4):
        print(f"{patient_groups[i]} has {len(all_patients[i])} unique patients")

        for j in range(i + 1, 4):
            duplicates = all_patients[i] & all_patients[j]

            if len(duplicates) > 0:
                print(f"Patients found across {patient_groups[i]} and {patient_groups[j]}.")

def train_val_split(root_directory, ad_directory, nc_directory):
    """
    Splits the given data into training and validation splits.

    The split will be 80/20.

    Parameters:
        root_directory (str): The name of the directory holding ADNI training split
        ad_directory (str): The name of the subdirectory containing Alzheimer's Disease slices
        nc_directory (str): The name of the subdirectory containing Cognitively Normal slices
    
    Returns:
        list[str], list[str]: Lists containing the patients for the train split and validation split
    """
    ad_patients = record_unique_patients(Path(root_directory + "/" + ad_directory))
    nc_patients = record_unique_patients(Path(root_directory + "/" + nc_directory))

    ## Perform split
    ad_train, ad_val = train_test_split(ad_patients, test_size=0.2, random_state=50)
    nc_train, nc_val = train_test_split(nc_patients, test_size=0.2, random_state=50)

    ## Combine AD and NC components
    train_split = ad_train + nc_train
    val_split = ad_val + nc_val

    return train_split, val_split

def train_val_test_split(train_directory, test_directory, ad_directory, nc_directory):
    """
    Splits the given data into training, validation, and test splits.

    The training/test split will be 67/33 to match the original splits.
    The training/validation split will be 80/20.

    Parameters:
        train_directory (str): The name of the directory holding ADNI training split
        test_directory (str): The name of the directory holding ADNI test split
        ad_directory (str): The name of the subdirectory containing Alzheimer's Disease slices
        nc_directory (str): The name of the subdirectory containing Cognitively Normal slices
    
    Returns:
        list[str], list[str], list[str]: Lists containing the patients for the train split,
                                         validation split, and test split.
    """
    ## Combines AD and NC patients to perform split
    ad_patients = record_unique_patients(Path(train_directory + "/" + ad_directory)) + record_unique_patients(Path(test_directory + "/" + ad_directory))
    nc_patients = record_unique_patients(Path(train_directory + "/" + nc_directory)) + record_unique_patients(Path(test_directory + "/" + nc_directory))

    ## Training/Test Split
    ad_train, ad_test = train_test_split(ad_patients, test_size=0.33, random_state=50)
    nc_train, nc_test = train_test_split(nc_patients, test_size=0.33, random_state=50)

    ## Training/Validation Split
    ad_train, ad_val = train_test_split(ad_train, test_size=0.2, random_state=50)
    nc_train, nc_val = train_test_split(nc_train, test_size=0.2, random_state=50)

    ## Combine AD and NC components
    train_split = ad_train + nc_train
    val_split = ad_val + nc_val
    test_split = ad_test + nc_test

    return train_split, val_split, test_split