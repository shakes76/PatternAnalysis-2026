import os

import numpy as np
import nibabel as nib
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def to_channels(arr: np.ndarray, dtype=np.uint8)-> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + (len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1
    return res

# load medical image functions
def load_data_2D(imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, early_stop=False):

    '''
    Load medical image data from names, cases list provided into a list for each.
    This function pre-allocates 4D arrays for conv2d to avoid excessive memory usage.
    normImage: bool (normalise the image 0.0-1.0)
    early_stop: Stop loading pre-maturely, leaves arrays mostly empty, for quick loading and testing scripts.
    '''
    affines = []
    # get fixed size
    num = len(imageNames)
    first_case = nib.load(imageNames[0]).get_fdata(caching='unchanged')
    if len(first_case.shape) == 3:
        first_case = first_case[:,:,0] # sometimes extra dims, remove
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, channels = first_case.shape
        images = np.zeros((num, rows, cols, channels), dtype=dtype)
    else:
        rows, cols = first_case.shape
        images = np.zeros((num, rows, cols), dtype=dtype)
    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged') # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 3:
            inImage = inImage[:,:,0] # sometimes extra dims in HipMRI_study data
        inImage = inImage.astype(dtype)
        if normImage:
            inImage = (inImage- inImage.mean()) / (inImage.std() + 1e-8)
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i,:,:,:] = inImage
        else:
            images[i,:,:] = inImage

        affines.append(affine)

        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images

def load_data_3D(imageNames, normImage=False, categorical=False, dtype=np.float32, getAffines=False, orient=False, early_stop=False): 
    '''
    Load medical image data from names, cases list provided into a list for each.
    This function pre-allocates 5D arrays for conv3d to avoid excessive memory usage.
    normImage: bool (normalise the image 0.0-1.0)
    orient: Apply orientation and resample image? Good for images with large slice thickness or anisotropic resolution
    dtype: Type of the data. If dtype=np.uint8, it is assumed that the data is labels
    early_stop: Stop loading pre-maturely? Leaves arrays mostly empty, for quick loading and testing scripts.
    '''
    affines = []

    interp = 'linear'
    if dtype == np.uint8: # assume labels
        interp = 'nearest'

    # get fixed size
    num = len(imageNames)
    niftiImage = nib.load(imageNames[0])
    first_case = niftiImage.get_fdata(caching='unchanged')
    if len(first_case.shape) == 4:
        first_case = first_case[:,:,:,0] # sometimes extra dims, remove
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
    else:
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)

    for i, inName in enumerate(tqdm(imageNames)):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged') # read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 4:
            inImage = inImage[:,:,:,0] # sometimes extra dims in HipMRI_study data
            inImage = inImage[:,:,:depth] # clip slices
            inImage = inImage.astype(dtype)
        if normImage:
            inImage = (inImage- inImage.mean()) / (inImage.std() + 1e-8)
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            images[i,:inImage.shape[0],:inImage.shape[1],:inImage.shape[1],:inImage.shape[2],:] = inImage
        else:
            images[i,:inImage.shape[0],:inImage.shape[1],:inImage.shape[2]] = inImage

        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images

def testview():
    root = "C:/Users/hawki/Uni/2026/MRI/keras_slices_seg_validate"
    shapes = {}
    for idx, file in enumerate(os.listdir(root)):
        if file.endswith(".nii.gz"):
            mask = load_data_2D([os.path.join(root, file)], normImage=False, categorical=True, dtype=np.uint8)
            if mask.shape not in shapes:
                shapes[idx] = mask.shape

    root2 = "C:/Users/hawki/Uni/2026/MRI/keras_slices_validate"
    unshaped = {}
    for idx, file in enumerate(os.listdir(root2)):
        if file.endswith(".nii.gz"):
            img = load_data_2D([os.path.join(root2, file)])
            unshaped[idx] = img.shape

    for i in range(len(shapes)):
        if i == 0:
            print(shapes[i][1:3], unshaped[i][1:3])
        if shapes[i][1:3] != unshaped[i][1:3]:
            print(f"Shape mismatch for index {i}: mask shape {shapes[i]}, image shape {unshaped[i]}")

    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            if shapes[i][1:3] != shapes[j][1:3]:
                print(f"Shape mismatch between index {i} and {j}: mask shape {shapes[i]}, mask shape {shapes[j]}") 

     

def testview3d():
    img = nib.load("W041_Week0_LFOV.nii")
    data = img.get_fdata()
    print(data.shape)

    plt.imshow(data, cmap="gray")
    plt.colorbar()
    plt.show()

testview()
