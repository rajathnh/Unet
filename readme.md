## Dataset

### 2D Dataset (Ready for Training)
Dataset link:  
https://drive.google.com/drive/folders/1qUvLPEPlaM3KkNVI1GKTAfc4Z0jz1YuU?usp=drive_link

### Original 3D Dataset
Download link:  
https://drive.google.com/file/d/1YZQFSonulXuagMIfbJkZeTFJ6qEUuUxL/view?usp=drive_link


## Steps to Generate the 2D Dataset from the 3D Dataset

1. Extract the `Task07_Pancreas.tar` file.
2. Open **prepare_dataset.py** and set the directory path to the extracted dataset.
3. Run `prepare_dataset.py` to convert the 3D volumes into 2D slices.
4. Run `split_data.py` to create the dataset split:
   - **80 / 20** (Train / Validation)  
   - **70 / 15 / 15** (Train / Validation / Test)
5. Open **Unet.ipynb** and run the cells sequentially to start training.


## Important Notes

- Update the **`dataset_root`** path in `prepare_dataset.py` to match your local dataset directory.
- Update the **`SOURCE_DIR`** path in `split_data.py` before running the script.
- Keep all project files in the **same root directory**.
- Run the notebook cells **sequentially** when training the model.

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
| Pooling | 2x2 |
| Pooling Type | Max Pooling |
| Padding | 1(Maintains spatial dimensions during convolution)|
| Activation Functions | ReLU (Hidden layers), Sigmoid (Attention Gate output) |
| Kernel Size | 3x3 (Core convolutions), 1x1 (Output & Attention Gates) |


## Implementation Differences from Original Papers

### U-Net (2015 Paper vs Our Implementation)

- **Padding:** The original paper uses *unpadded convolutions*, causing feature maps to shrink and requiring cropping before concatenation. Our implementation uses **padded convolutions (padding=1)** to maintain spatial dimensions.
- **Batch Normalization:** The original architecture does **not include BatchNorm**. Our implementation adds **`nn.BatchNorm2d` in each DoubleConv block** for improved training stability.
- **Optimizer:** The paper uses **SGD with momentum 0.99**, while our implementation uses **AdamW with a Cosine Annealing learning rate scheduler**.
- **Loss Function:** The original uses **pixel-wise cross-entropy with weight maps** for boundary separation. Our implementation uses **Combo Loss (50% Cross-Entropy + 50% Dice Loss)**.

### Attention U-Net (2018 Paper vs Our Implementation)

- **Dimensionality:** The original paper proposes a **3D Attention U-Net** using 3D convolutions for volumetric CT data. Our implementation adapts this into a **2D pipeline using `nn.Conv2d` on extracted slices**.
- **Deep Supervision:** The paper applies **deep supervision at multiple scales**. Our implementation **computes loss only on the final output layer**.
- **Loss Function:** The original model uses **Sorensen–Dice loss** across semantic classes. Our implementation uses **Combo Loss (50% Cross-Entropy + 50% Dice Loss)**.
- **Optimizer:** The paper uses the **Adam optimizer**, while our implementation uses **AdamW with weight decay (1e-5)**.