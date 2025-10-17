#Create optimized dataset with patients with more than 10 tumor-positive slices

import os
import nibabel as nib
import numpy as np
from collections import defaultdict
from shutil import copy2

def filter_and_save_tumor_rich_patients(ct_folder, label_folder, liv_folder, output_dir, min_tumor_slices=10):
    os.makedirs(os.path.join(output_dir, "OptimTestVol"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "OptimTestSegTumors"), exist_ok=True)    
    os.makedirs(os.path.join(output_dir, "OptimTestSegLiver"), exist_ok=True)


    approved_patients = []
    tumor_voxel_percentages = []
    depths = []

    print(f"\n Filtering patients with ≥{min_tumor_slices} tumor-containing slices...")
    for filename in sorted(os.listdir(ct_folder)):
        if not (filename.endswith(".nii") or filename.endswith(".nii.gz")):
            continue

        patient_id = filename.replace("_ct.nii.gz", "").replace("_ct.nii", "")
        ct_path = os.path.join(ct_folder, filename)
        label_path = os.path.join(label_folder, f"{patient_id}_seg_combined.nii.gz")

        if not os.path.exists(label_path):
            print(f" Missing label for {patient_id}, skipping.")
            continue
        
        liv_label_path = os.path.join(liv_folder, f"{patient_id}_Liver.nii.gz")

        if not os.path.exists(label_path):
            print(f"  Missing liver label for {patient_id}, skipping.")
            continue

        ct_img = nib.load(ct_path)
        label_img = nib.load(label_path)
        liv_img = nib.load(liv_label_path)
        ct_data = ct_img.get_fdata()
        label_data = label_img.get_fdata()
        liv_label_data = liv_img.get_fdata()


        if ct_data.shape != label_data.shape:
            print(f" Shape mismatch for {patient_id}, skipping.")
            continue

        depth = ct_data.shape[2]
        tumor_mask = (label_data > 0).astype(np.uint8)

        tumor_slices = np.count_nonzero(np.any(tumor_mask, axis=(0, 1)))
        tumor_voxels = np.sum(tumor_mask)
        total_voxels = np.prod(tumor_mask.shape)
        voxel_percentage = (tumor_voxels / total_voxels) * 100

        if tumor_slices >= min_tumor_slices:
            approved_patients.append(patient_id)
            depths.append(depth)
            tumor_voxel_percentages.append(voxel_percentage)

            # Save to OptimizedTestSet
            copy2(ct_path, os.path.join(output_dir, "OptimTestVol", f"{patient_id}_ct.nii.gz"))
            copy2(label_path, os.path.join(output_dir, "OptimTestSegTumors", f"{patient_id}_seg_combined.nii.gz"))
            copy2(liv_label_path, os.path.join(output_dir, "OptimTestSegLiver", f"{patient_id}_Liver.nii.gz"))


    # Print summary
    print(f"\n Total patients approved: {len(approved_patients)}")
    if approved_patients:
        print("\n Approved Patient Depths:")
        for pid, d in zip(approved_patients, depths):
            print(f"{pid:<20} | Depth: {d}")

        print(f"\n Average tumor voxel % across approved patients: {np.mean(tumor_voxel_percentages):.2f}%")

    else:
        print("No patients met the tumor slice threshold.")

# Example usage
if __name__ == "__main__":
    data_path = r"path/to/data"

    in_ct = os.path.join(data_path, "TestVol") #volume folders
    in_labels = os.path.join(data_path, "TestSegTumorsComb") #liver labels
    in_liv = os.path.join(data_path, "TestSegLiver") #tumor labels
    out_dir = data_path

    filter_and_save_tumor_rich_patients(
        ct_folder=in_ct,
        label_folder=in_labels,
        liv_folder = in_liv,
        output_dir=out_dir,
        min_tumor_slices=10
    )
