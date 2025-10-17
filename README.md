# Automated 3D Segmentation of Liver and Colorectal Liver Metastases (CRLM)

> **Thesis Project** — Development of a reliable deep learning pipeline for liver and tumor segmentation from 3D CT scans using the MONAI framework.

## Project Overview
This project focuses on the **automatic segmentation of the liver and colorectal liver metastases (CRLM)** from CT scans through a **two-stage deep learning pipeline**:
1. Liver segmentation
2. Tumor segmentation guided by the predicted liver mask.

The pipeline was built using MONAI and PyTorch, with careful attention to **preprocessing**, **data augmentation**, and **dataset curation** to handle class imbalance, depth variation, and tumor heterogeneity.

## Key Results
| Task                    | Model         | Dice Score | Notes                                           |
|--------------------------|---------------|------------|-------------------------------------------------|
| Liver segmentation       | SegResNet     | **0.968**  | State-of-the-art performance                    |
| Tumor segmentation (CRLM)| SegResNet     | **0.674**  | Competitive given limited data & GPU resources  |

## Tools & Frameworks
- **:contentReference[oaicite:1]{index=1}** – preprocessing, augmentations, models  
- **:contentReference[oaicite:2]{index=2}** – training and inference  
- **:contentReference[oaicite:3]{index=3}** – experiment tracking  
- **:contentReference[oaicite:4]{index=4}** – remote training  
- Local GPU: NVIDIA GeForce GTX 1650 Ti (4GB)  
- Remote GPU: NVIDIA GeForce RTX 4080 (16GB)
