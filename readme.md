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


| Hyperparameter | This Repo (Our Implementation) | U-Net (Ronneberger et al., 2015) | Attention U-Net (Oktay et al., 2018) | Reason (Only if Different) |
|---|---|---|---|---|
| Batch Size | 4 | 1 (tile-based / single-image updates used in original experiments) | 2–4 (small batches for 3D patches) | Memory / GPU practicality for slice-wise 2D training |
| Epochs | 200 | Not explicitly fixed (train until convergence / configured per experiment) | Not explicitly fixed (varies by experiment) | — |
| Initial LR | 1e-4 | Not explicitly reported (SGD used) | Adam default variants reported; not exact LR in general | Empirical modern choice |
| Optimizer | AdamW (weight_decay=1e-5) | SGD with momentum 0.99 | Adam | AdamW improves generalization + weight decay regularization |
| LR Scheduler | Cosine Annealing (min_lr=1e-6) | None / not specified in base paper | Not mandated (varies) | Practical modern training design |
| Loss Function | Combo Loss = 0.5 × CE + 0.5 × Dice | Pixel-wise softmax + Cross-Entropy with weight maps (emphasize borders) | Sørensen–Dice / Dice-based losses used | Combo helps overlap metrics and boundary performance |
| Input Image / Patch | 512 × 512 (2D slices) | 572 × 572 tiles (2D input; output crops due to valid convolutions) | Volumetric patches (3D inputs) | We use 2D slices to reduce memory and reuse a 2D pipeline |
| CT Intensity Clipping | [-100, 240] HU (soft tissue window) | N/A (original U-Net used microscopy imagery) | Varies by CT experiments; normalization used | Domain-specific preprocessing for CT |
| Train / Val / Test Split | 80% / 20% (Random State: 42) or 70/15/15 | Varies by dataset / experiment | Varies by dataset / experiment | — |
| Data Augmentation | Rotation (50%), Horizontal Flip (50%), Distortions (30%) | Heavy elastic deformations + basic augmentations | Augmentations vary per experiment (often volumetric for 3D) | Simplified spatial augmentations |
| Pooling | 2 × 2 | 2 × 2 Max Pooling | 2 × 2 (or 2 × 2 × 2 for 3D) | — |
| Pooling Type | Max Pooling | Max Pooling | Max / volumetric pooling for 3D | — |
| Padding in Convs | padding = 1 (spatial dimensions preserved) | Unpadded / valid 3×3 convolutions (feature map shrinkage; cropping required) | Valid convolutions with cropping / tiling strategies | Padding avoids cropping and simplifies skip connections |
| Activation Functions | ReLU (hidden), Sigmoid for attention outputs | ReLU in intermediate layers; final Softmax for multi-class | ReLU in convolutions; sigmoid-style gating in attention blocks | — |
| Kernel Sizes | 3×3 (core convs), 1×1 (output & attention gates) | 3×3 convs in core; 1×1 for final mapping | 3×3 convs (3D kernels in volumetric case) and 1×1 for gates | — |
| Batch Normalization | Added (nn.BatchNorm2d) | Not used in original U-Net | Commonly used in many implementations | Added for modern training stability |
| Dimensionality | 2D slices (Conv2d) | 2D U-Net | 3D Attention U-Net (volumetric convolutions) | Adapted to 2D to manage memory |
| Deep Supervision | Loss computed only on final output | Not used in original U-Net | Used at multiple scales in the paper | Simplified training setup |
| Weight Decay | 1e-5 | Not specified | Not mandated | Regularization for AdamW |