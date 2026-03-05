import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from attn_unet.attn_unet_model import AttU_Net
# from unet.unet_model import UNet 

from medical_dataset import PancreasDataset

TRAIN_DIR = "./msd_split/train"
VAL_DIR = "./msd_split/val"

BATCH_SIZE = 4       
LEARNING_RATE = 1e-4 # Starting LR
EPOCHS = 200         # Need time for augmentations to work
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = 'best_model1.pth'

# --- COMBO LOSS (Dice + CrossEntropy) ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: (B, 2, H, W) raw logits
        # targets: (B, H, W) class indices
        
        # Apply Softmax to get probabilities
        inputs = F.softmax(inputs, dim=1)
        
        # We only care about Class 1 (Pancreas)
        inputs = inputs[:, 1] # (B, H, W)
        
        # Flatten
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

def train_model():
    print(f"--- Starting Training Pipeline ---")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}")

    # 1. Initialize Model
    # Change to UNet(...) if you want to test standard U-Net
    model = AttU_Net(n_channels=1, n_classes=2).to(DEVICE)

    # 2. Load Data
    print("Initializing Datasets...")
    train_set = PancreasDataset(f"{TRAIN_DIR}/images", f"{TRAIN_DIR}/labels", is_train=True)
    val_set = PancreasDataset(f"{VAL_DIR}/images", f"{VAL_DIR}/labels", is_train=False)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Training on {len(train_set)} images | Validating on {len(val_set)} images")

    # 3. Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    # 4. Loss Functions
    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss()

    best_val_loss = float('inf')

    # 5. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        # --- TRAIN STEP ---
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{EPOCHS}', unit='batch') as pbar:
            for images, masks in train_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                
                # Forward
                outputs = model(images)
                
                # Calculate Combo Loss
                loss_ce = criterion_ce(outputs, masks)
                loss_dice = criterion_dice(outputs, masks)
                loss = 0.5 * loss_ce + 0.5 * loss_dice
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                pbar.update(1)
        
        avg_train_loss = train_loss / len(train_loader)

        # --- VALIDATION STEP ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)
                
                outputs = model(images)
                
                loss_ce = criterion_ce(outputs, masks)
                loss_dice = criterion_dice(outputs, masks)
                loss = 0.5 * loss_ce + 0.5 * loss_dice
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Step the scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\n[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.2e}")

        # --- SAVE BEST MODEL ---
        if avg_val_loss < best_val_loss:
            print(f"==> Validation Improved ({best_val_loss:.4f} -> {avg_val_loss:.4f}). Saving model...")
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        
        print("-" * 50)

    print(f"Training Complete. Best Validation Loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    # Windows fix for multiprocessing in DataLoader
    import multiprocessing
    multiprocessing.freeze_support()
    train_model()