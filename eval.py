import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- IMPORTS ---
# Ensure this matches the model you trained (AttU_Net or UNet)
from attn_unet.attn_unet_model import AttU_Net
# from unet.unet_model import UNet

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VAL_DIR = "./msd_split/train" 
MODEL_PATH = "best_model1.pth" 

# --- DICE CALCULATION FUNCTION ---
def calculate_dice(pred, target):
    """Calculates Dice score, handling empty masks correctly."""
    # If both prediction and target are empty, it's a perfect score of 1.0
    if pred.sum() == 0 and target.sum() == 0:
        return 1.0
    
    # Standard Dice calculation
    intersection = (pred * target).sum()
    dice = (2. * intersection) / (pred.sum() + target.sum() + 1e-6)
    return dice

# --- MAIN EVALUATION FUNCTION ---
def evaluate_model():
    print(f"--- Starting Simple Evaluation ---")
    print(f"Model: {MODEL_PATH}")
    print(f"Dataset: {VAL_DIR}")
    
    # 1. Load the trained model
    model = AttU_Net(n_channels=1, n_classes=2).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"FATAL ERROR loading model: {e}")
        print("Ensure the model class (AttU_Net vs UNet) matches the saved file.")
        return
    model.eval()
    
    # 2. Get the list of validation image files
    image_dir = Path(VAL_DIR) / "images"
    label_dir = Path(VAL_DIR) / "labels"
    files = sorted(list(image_dir.glob("*.npy")))
    
    if not files:
        print(f"FATAL ERROR: No .npy files found in '{image_dir}'.")
        return
        
    dice_scores = []
    
    # 3. Loop through each validation image
    with torch.no_grad():
        for f in tqdm(files, desc="Evaluating"):
            # Load image and mask
            img = np.load(f).astype(np.float32)
            mask = np.load(label_dir / f.name).astype(np.uint8)
            
            # Normalize labels (MSD has 0, 1, 2 -> we want 0, 1)
            mask[mask > 0] = 1
            
            # Prepare tensor for the model: (H, W) -> (1, 1, H, W)
            img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(DEVICE)
            
            # Get model prediction
            output = model(img_tensor)
            
            # Convert output to a binary mask (0s and 1s)
            pred_mask = torch.softmax(output, dim=1).argmax(dim=1).cpu().numpy()[0]
            
            # Calculate the Dice score for this slice
            score = calculate_dice(pred_mask, mask)
            dice_scores.append(score)

    # 4. Print the final average score
    avg_dice = np.mean(dice_scores)
    
    print("\n" + "="*30)
    print("      EVALUATION COMPLETE")
    print("="*30)
    print(f"Average Dice Score: {avg_dice:.4f}")
    print("="*30)

if __name__ == "__main__":
    evaluate_model()