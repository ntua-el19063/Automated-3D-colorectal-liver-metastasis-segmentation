# Calculate median pixdim values for a specific dataset

import os
import nibabel as nib
import numpy as np
from glob import glob

original_ct_dir = r'path/to/dataset' 

nifti_files = sorted(glob(os.path.join(original_ct_dir, "*.nii.gz")))

pixdim_list = []

# Loop through each file and extract voxel spacing
for file in nifti_files:
    nii_img = nib.load(file)
    pixdim = nii_img.header.get_zooms()  
    
    if len(pixdim) >= 3:  
        pixdim_list.append(pixdim[:3])  

pixdim_array = np.array(pixdim_list)

median_pixdim = np.median(pixdim_array, axis=0)

min_pixdim = np.min(pixdim_array, axis=0)

max_pixdim = np.max(pixdim_array, axis=0)

# Print results
print(f" Median Pixel Dimensions from Train CT Scans: {median_pixdim}")
print(f" Minimum Pixel Dimensions from Train CT Scans: {min_pixdim}")
print(f" Maximum Pixel Dimensions from Train CT Scans: {max_pixdim}")
print(f" Suggested pixdim for preprocessing: pixdim={tuple(median_pixdim)}")