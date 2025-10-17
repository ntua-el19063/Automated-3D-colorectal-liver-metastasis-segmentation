# Plot the depth among the patient volumes for a specific dataset

import os
import nibabel as nib  
import matplotlib.pyplot as plt
from collections import Counter

train_vol_dir = r'path/to/data'

depths = []

# Loop over all NIfTI (.nii.gz) files
for fname in os.listdir(train_vol_dir):
    if fname.endswith('.nii') or fname.endswith('.nii.gz'):
        filepath = os.path.join(train_vol_dir, fname)
        img = nib.load(filepath)
        data = img.get_fdata()
        
        # Assuming data shape is (H, W, D) or (C, H, W, D)
        if data.ndim == 4:  # If channel-first
            depth = data.shape[3]
        elif data.ndim == 3:
            depth = data.shape[2]
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")
        
        depths.append(depth)

depth_counts = Counter(depths)

# Plot bar chart
plt.figure(figsize=(10, 5))
plt.bar(depth_counts.keys(), depth_counts.values(), color='skyblue')
plt.xlabel('Depth (Number of Slices)')
plt.ylabel('Number of Volumes')
plt.title('Distribution of Depths in TrainVol')
plt.grid(True, linestyle='--', alpha=0.5)

plt.xticks(ticks=list(range(200, 1, -10)))  # Show labels every 5 slices
plt.show()
