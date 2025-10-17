import os
from glob import glob

from monai.utils import first
import matplotlib.pyplot as plt
import torch
import numpy as np
from monai.losses import DiceLoss
from tqdm import tqdm
import nibabel as nib
import torch.nn.functional as F

from monai.transforms import (
    Compose,
    EnsureChannelFirstD,
    LoadImaged,
    Resized,
    ToTensord,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    SpatialPadD,
    DivisiblePadD,
    RandAdjustContrastd, 
    RandScaleIntensityd,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    RandFlipd,
    RandRotated,
    RandZoomd,
    RandGaussianNoised,
    RandAffined,
    ConcatItemsd,
    Lambdad,


)
from monai.data import DataLoader, Dataset, CacheDataset
from monai.utils import set_determinism

base_dir = r'path/to/split' # Data folders directory with patient NIfTI files
sample_path = r'path/to/sample_patient'
cache_path = r'path/to/cache.pt' #To save preprocessed data

def prepare(base_dir, pixdim=(1.0, 1.0, 2.5), a_min=-100, a_max=200, cache=True, spatial_size = (128, 128, 48), num_samples = 8):

    set_determinism(seed=0) # Ensure reproducibility

    path_train_volumes = sorted(glob(os.path.join(base_dir, "OptimTrainVol", "*.nii.gz")))
    path_train_seg_tumors = sorted(glob(os.path.join(base_dir, "OptimTrainSegTumors", "*.nii.gz")))
    path_train_seg_liver = sorted(glob(os.path.join(base_dir, "OptimTrainSegLiverPred", "*.nii.gz")))  # Predicted Liver Mask
    
    path_val_volumes = sorted(glob(os.path.join(base_dir, "OptimValVol", "*.nii.gz")))
    path_val_seg_tumors = sorted(glob(os.path.join(base_dir, "OptimValSegTumors", "*.nii.gz")))
    path_val_seg_liver = sorted(glob(os.path.join(base_dir, "OptimValSegLiverPred", "*.nii.gz")))

    path_test_volumes = sorted(glob(os.path.join(base_dir, "OptimTestVol", "*.nii.gz")))
    path_test_seg_tumors = sorted(glob(os.path.join(base_dir, "OptimTestSegTumors", "*.nii.gz")))
    path_test_seg_liver = sorted(glob(os.path.join(base_dir, "OptimTestSegLiverPred", "*.nii.gz")))

    # **Pair Volumes with Corresponding Segmentations**
    train_files = [{"vol": vol, "seg_tumor": seg_tumor, "seg_liver": seg_liver} 
                   for vol, seg_tumor, seg_liver in zip(path_train_volumes, path_train_seg_tumors, path_train_seg_liver)]

    val_files = [{"vol": vol, "seg_tumor": seg_tumor, "seg_liver": seg_liver} 
                 for vol, seg_tumor, seg_liver in zip(path_val_volumes, path_val_seg_tumors, path_val_seg_liver)]

    test_files = [{"vol": vol, "seg_tumor": seg_tumor, "seg_liver": seg_liver} 
                  for vol, seg_tumor, seg_liver in zip(path_test_volumes, path_test_seg_tumors, path_test_seg_liver)]

    print(f"Found {len(path_train_volumes)} volumes, {len(path_train_seg_tumors)} tumors, {len(path_train_seg_liver)} livers for training.")
    print(f"Found {len(path_test_volumes)} volumes, {len(path_test_seg_tumors)} tumors, {len(path_test_seg_liver)} livers for testing.")

    print(f"Final train file triplets: {len(train_files)}")
    print(f"Final val file triplets: {len(val_files)}")
    print(f"Final test file triplets: {len(test_files)}")

    train_transforms = Compose([
        LoadImaged(keys=["vol", "seg_tumor", "seg_liver"]),
        EnsureChannelFirstD(keys=["vol", "seg_tumor", "seg_liver"]),
        Spacingd(keys=["vol", "seg_tumor", "seg_liver"], pixdim=pixdim, mode=("bilinear", "nearest", "nearest")),
        Orientationd(keys=["vol", "seg_tumor", "seg_liver"], axcodes="RAS"),

        ScaleIntensityRanged(
            keys=["vol"], a_min=a_min, a_max=a_max,
            b_min=0.0, b_max=1.0, clip=True
        ),        

        CropForegroundd(keys=["vol", "seg_tumor", "seg_liver"], source_key="seg_liver"), #Crops a tight bounding box around the non-zero region in seg_liver
        DivisiblePadD(keys=["vol", "seg_tumor", "seg_liver"], k=16),

        RandCropByPosNegLabeld(
            keys=["vol", "seg_tumor","seg_liver"],
            label_key="seg_tumor",   # crop around the liver region
            spatial_size=spatial_size,
            pos=0.8, # With probability pos/(pos+neg), it chooses a center inside the liver region
            neg=0.2,
            num_samples=num_samples,
            image_key="vol",
            allow_smaller=True
        ),

        RandFlipd(keys=["vol", "seg_tumor"], spatial_axis=[0, 1], prob=0.4),
        RandRotated(keys=["vol", "seg_tumor"], range_x=np.pi/18, range_y=np.pi/18, range_z=np.pi/36, prob=0.4),
        RandZoomd(keys=["vol", "seg_tumor"], min_zoom=0.9, max_zoom=1.1, prob=0.3),
        RandScaleIntensityd(keys=["vol"], factors=0.1, prob=0.2),
        RandGaussianNoised(keys=["vol"], prob=0.05, mean=0.0, std=0.01), 
        
        RandScaleIntensityd(keys=["vol"], factors=0.1, prob=0.15),
        RandShiftIntensityd(keys=["vol"], offsets=0.1, prob=0.15),
        RandAdjustContrastd(keys=["vol"], prob=0.15, gamma=(0.8, 1.2)),

        ConcatItemsd(keys=["vol", "seg_liver"], name="conc_image", dim=0), # Add predicted masks as 2nd channel

        ToTensord(keys=["conc_image", "seg_tumor"])
    ])

    
    

    validation_transforms = Compose([
        LoadImaged(keys=["vol", "seg_tumor", "seg_liver"]),
        EnsureChannelFirstD(keys=["vol", "seg_tumor", "seg_liver"]),

        Spacingd(keys=["vol", "seg_tumor", "seg_liver"], pixdim=pixdim, mode=("bilinear", "nearest", "nearest")),
        Orientationd(keys=["vol", "seg_tumor", "seg_liver"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["vol"], a_min=a_min, a_max=a_max, b_min=0.0, b_max=1.0, clip=True),

        CropForegroundd(keys=["vol", "seg_tumor", "seg_liver"], source_key="seg_liver"), #Crops a tight bounding box around the non-zero region in seg_liverCrops a tight bounding box around the non-zero region in seg_liver

        DivisiblePadD(keys=["vol", "seg_tumor","seg_liver"], k=16, mode="constant"),
        
        ConcatItemsd(keys=["vol", "seg_liver"], name="conc_image", dim=0),

        ToTensord(keys=["conc_image", "seg_tumor"])
    ])


    # Cache dataset for faster loading
    if cache:
        train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
        val_ds = CacheDataset(data=val_files, transform=validation_transforms, cache_rate=1.0)
        test_ds = CacheDataset(data=test_files, transform=validation_transforms, cache_rate=1.0)
        
    else:
        train_ds = Dataset(data=train_files, transform=train_transforms)
        val_ds = Dataset(data=val_files, transform=validation_transforms)
        test_ds = Dataset(data=test_files, transform=validation_transforms)

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader  # Now returns 3 separate sets
