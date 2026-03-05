Dataset link to start training directly(2d dataset)-https://drive.google.com/drive/folders/1qUvLPEPlaM3KkNVI1GKTAfc4Z0jz1YuU?usp=drive_link

Original 3d dataset link - https://drive.google.com/file/d/1YZQFSonulXuagMIfbJkZeTFJ6qEUuUxL/view?usp=drive_link<br>
Steps to derieve 2d data from the original 3d dataset:<br>
1 - Extract the Task07_Pancreas.tar file<br>
2 - Set the directory of the extracted folder in the prepare_dataset.py file and run it.<br>
3 - Run split_data.py to get 80/20 or 70/15/15 split , whichever you want.<br>
4 - Run the Unet.ipynb file.<br>
Note: Before running the scripts, make sure to update the dataset_root path in prepare_dataset.py and the SOURCE_DIR in split_data.py to match your local machine<br>
To train the model, open Unet.ipynb file and run each block seperatly.<br>
Keep all the files in the same root directory.<br>

## Training Hyperparameters

| Hyperparameter / Setting | Value |
|--------------------------|------|
| Batch Size | 4 |
| Epochs | 200 |
| Initial Learning Rate | 1e-4 (0.0001) |
| Optimizer | AdamW (Weight Decay: 1e-5) |
| LR Scheduler | Cosine Annealing (Min LR: 1e-6) |
| Loss Function | Combo Loss: 50% Cross-Entropy + 50% Dice Loss |
| Input Image Size | 512x512 |
| CT Intensity Clipping | [-100, 240] Hounsfield Units (Soft tissue window) |
| Train / Validation Split | 80% / 20% (Random State: 42) |
| Data Augmentations | Rotation (50%), Horizontal Flip (50%), Distortions (30%) |