from collections import defaultdict
import random

import numpy as np
import utils
import torch
import torch.nn as nn
import os
import re
import nibabel as nib


# resize and focus on the info, maybe use attention net

# return data for model
root2d = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
MRI3d = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"
Segments3d = "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"

testNii = "C:/Users/hawki/OneDrive/Desktop/Uni/2026/Comp_Patterns/Report/seg_042_week_0_slice_9.nii"

#For remote desktop
# root2d = "/home/s4801967/PatternAnalysis-2026/MRI"

# for local
# root2d = "C:/Users/hawki/Uni/2026/MRI"

def get_2d_data(path, truth = False):
    """
    Load 2D medical image data from the specified directory.
    path = directory containing the NIfTI files
    truth = boolean indicating whether the data is ground truth (segmentation masks) or not
    """
    if truth == True:
        pattern = re.compile(r"seg_(\d+)_week_(\d+)_slice_(\d+).nii")
    else:
        pattern = re.compile(r"case_(\d+)_week_(\d+)_slice_(\d+).nii")
    
    volumes = defaultdict(list)
    shapes = {}
    count = 0
    
    for file in os.listdir(path):
        # if count > 10:
        #     break
        # count +=1
        match = pattern.match(file)
        if match:
            patient_id = int(match.group(1))
            week = int(match.group(2))
            slice_num = int(match.group(3))
            volumes[(patient_id, week)].append((slice_num, os.path.join(path, file)))

    for key in volumes:
        volumes[key].sort(key=lambda x: x[0])

    images = {}
    
    for key, slices in volumes.items():
        image_name = [slice_path for _, slice_path in slices]
        if truth:
            images[key] = utils.load_data_2D(image_name, normImage=False, categorical=True, dtype=np.uint8)
        else:
            images[key] = utils.load_data_2D(image_name, normImage=True)

    return images

def get_training_2d():
    """
    Load and format 2D training data
    """
    loc = root2d + "/keras_slices_train"
    loc2 = root2d + "/keras_slices_seg_train"


    return UNet2d_Data(
        get_2d_data(loc),
        get_2d_data(loc2, truth=True)
    )

def get_validation_2d():
    """
    Load and format 2D validation data
    """
    loc = root2d + "/keras_slices_validate"
    loc2 = root2d + "/keras_slices_seg_validate"
    return UNet2d_Data(
        get_2d_data(loc),
        get_2d_data(loc2, truth=True)
    )

def get_testing_2d():
    """
    Load and format 2D testing data
    """
    loc = root2d + "/keras_slices_test"
    loc2 = root2d + "/keras_slices_seg_test"
    return UNet2d_Data(
        get_2d_data(loc),
        get_2d_data(loc2, truth=True)
    )

class UNet2d_Data(torch.utils.data.Dataset):
    def __init__(self, images, masks):
        """images: dict of 2D images, masks: dict of corresponding segmentation masks"""
        self.samples = []
        widths = set()

        mask_ids = []
        img_ids = []


        for key in images:
            if key not in masks:
                continue

            volume = images[key]
            ground_truth = masks[key]
            try:
                for i in range(volume.shape[0]):
                    self.samples.append((volume[i], ground_truth[i]))
            except Exception as e:
                print(f"Error processing key {key}: {e}")
                continue

        print("all removed")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        Resizes the image and ground truth to a target size of 256x144.
        """
        image, ground_truth = self.samples[idx]

        targetH = 256
        targetW = 144

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        image = nn.functional.interpolate(image, size=(targetH, targetW), mode='bilinear', align_corners=False).squeeze(0)
        
        ground_truth = torch.tensor(ground_truth, dtype=torch.float32).permute(2, 0, 1)
        ground_truth = torch.argmax(ground_truth, dim=0).long()
        ground_truth = nn.functional.interpolate(ground_truth.unsqueeze(0).unsqueeze(0).float(), size=(targetH, targetW), mode='nearest').squeeze(0).squeeze(0).long()
        return image, ground_truth
    
    def get_proportion(self):
        """
        Calculate the proportion of each class in the dataset.
        Returns a dictionary with class labels as keys and their corresponding proportions as values.
        """
        num_classes = self.samples[0][1].shape[-1]
        class_counts = torch.zeros(num_classes)

        for _, gt in self.samples:
            gt = torch.tensor(gt, dtype=torch.float32)
            class_counts += gt.sum(dim=(0, 1))

        total = class_counts.sum()
        weights = total / (class_counts + 1e-8)
        weights = weights / weights.mean()

        return weights.float()




















# 3D data loading
root3d_data = "/home/groups/comp3710/HipMRI_Study_open/semantic_MRs"
root3d_truth = "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only"

# root3d_data = "C:/Users/hawki/Uni/2026/MRI/semantic_MRs"
# root3d_truth = "C:/Users/hawki/Uni/2026/MRI/semantic_labels_only"

def get_3d_data(path1 = root3d_data, path2 = root3d_truth):
    """
    Load 3D medical image data from the specified directories and return training, validation, and testing datasets.
    """
    images = load_3d_data(path1)
    masks = load_3d_data(path2, truth=True)

    # (train_images, train_masks), (val_images, val_masks), (test_images, test_masks) = seperate(images, masks)
    train, validate, test = seperate(images, masks)
    print(f"Training set size: {len(train[0])}")
    dtrain = UNet3d_Data(train[0], train[1])
    dvalidate = UNet3d_Data(validate[0], validate[1])
    dtest = UNet3d_Data(test[0], test[1])

    return dtrain, dvalidate, dtest

def load_3d_data(path, truth = False):
    """
    Load 3D medical image data from the specified directory.
    path = directory containing the NIfTI files
    truth = boolean indicating whether the data is ground truth (segmentation masks) or not
    """
    if truth:
        pattern = re.compile(r"(\w+)(\d+)_Week(\d)_SEMANTIC.nii.gz")
    else:
        pattern = re.compile(r"(\w+)(\d+)_Week(\d)_LFOV.nii.gz")

    volumes = defaultdict(list)
    count = 0
    for file in os.listdir(path):
        # count += 1
        # if count > 10:
        #     break
        match = pattern.match(file)
        if match:
            patient_id = match.group(1) + match.group(2)
            week = int(match.group(3))
            file_path = os.path.join(path, file)

            volumes[(patient_id, week)].append(file_path)

    images = {}
    for key, slices in volumes.items():
        image_name = [slice_path for slice_path in slices]
        if truth:
            images[key] = utils.load_data_3D(image_name, normImage=False, dtype=np.uint8)
        else:
            images[key] = utils.load_data_3D(image_name, normImage=True)

        
    return images

def seperate(images, masks):
    """
    Separate the provided images and masks into training, validation, and testing datasets based on unique patient IDs and random selection.
    """
    random.seed(42)
    train_images = {}
    train_masks = {}
    val_images = {}
    val_masks = {}
    test_images = {}
    test_masks = {}

    unique_patients = sorted({patient_id for patient_id, week in images.keys()})
    
    keys = unique_patients
    random.shuffle(keys)

    n = len(keys)

    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_keys = keys[:train_end]
    val_keys = keys[train_end:val_end]
    test_keys = keys[val_end:]

    for key in images.keys():
        patient_id, week = key
        if patient_id in train_keys:
            train_images[key] = images[key]
            train_masks[key] = masks[key]
        elif patient_id in val_keys:
            val_images[key] = images[key]
            val_masks[key] = masks[key]
        elif patient_id in test_keys:
            test_images[key] = images[key]
            test_masks[key] = masks[key]
   
    return (train_images, train_masks), (val_images, val_masks), (test_images, test_masks)

class UNet3d_Data(torch.utils.data.Dataset):
    def __init__(self, images, masks):
        self.samples = []
        for key in images:
            volume = images[key]
            ground_truth = masks[key]
            print(f"Volume shape: {volume.shape}, Ground truth shape: {ground_truth.shape}")
            self.samples.append((volume, ground_truth))

                    
        image, gt = self.samples[0]

        print(image.shape)
        print(gt.shape)
        print(np.unique(gt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, ground_truth = self.samples[idx]
 
        targetH = 256
        targetW = 256
        targetD = 128

        image = torch.tensor(image, dtype=torch.float32)
        ground_truth = torch.tensor(ground_truth, dtype=torch.long)

        image = nn.functional.interpolate(image.unsqueeze(0), size=(targetD, targetW, targetH), mode='trilinear', align_corners=False).squeeze(0)

        ground_truth = nn.functional.interpolate(ground_truth.unsqueeze(0).float(), size=(targetD, targetW, targetH), mode='nearest').squeeze(0).squeeze(0).long()
        
        return image, ground_truth

    def get_proportion(self):
        """
        Calculate the proportion of each class in the dataset.
        Returns a dictionary with class labels as keys and their corresponding proportions as values.
        """
        class_counts = None
        for _, gt in self.samples:
            gt = torch.as_tensor(gt)

            if class_counts is None:
                n_classes = int(gt.max()) + 1
                class_counts = torch.zeros(n_classes)

            for c in range(len(class_counts)):
                class_counts[c] += (gt == c).sum()

        total = class_counts.sum()

        weights = total / (class_counts + 1e-8)
        weights = weights / weights.mean()
        return weights.float()
