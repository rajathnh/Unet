import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PancreasDataset(Dataset):
    def __init__(self, images_dir, masks_dir, is_train=False):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.is_train = is_train
        
        self.files = sorted([f.name for f in self.images_dir.glob('*.npy')])

        if self.is_train:
            self.transform = A.Compose([
                # 1. Geometry / Shape Changes
                A.Rotate(limit=35, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                
                # 2. Distortions (Updated arguments for new Albumentations version)
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
                A.GridDistortion(p=0.3),
                A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=0.3),

                # 3. Texture / Pixel Level
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(var_limit=(0.001, 0.01), p=0.2),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                
                # 4. Normalization
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img_path = self.images_dir / self.files[i]
        mask_path = self.masks_dir / self.files[i]
        
        # Load Data
        img = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path)
        mask[mask > 0] = 1 
        mask = mask.astype(np.uint8)

        # Handle dimensions (H, W)
        if len(img.shape) == 3:
            img = img.squeeze(0)
            
        # Apply Augmentations
        augmented = self.transform(image=img, mask=mask)
        img_t = augmented['image']
        mask_t = augmented['mask']
        
        # Convert Mask to Long for PyTorch (0 or 1 integers)
        mask_t = mask_t.long()
        
        # Ensure image has channel dim (1, H, W)
        if len(img_t.shape) == 2:
            img_t = img_t.unsqueeze(0)

        return img_t, mask_t