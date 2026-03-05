# visualize.py

import torch
from torch.utils.data import DataLoader, random_split
from attn_unet.attn_unet_model import AttU_Net
from medical_dataset import PancreasDataset
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "./msd_split/val"
MODEL_PATH = 'best_model1.pth'
NUM_IMAGES_TO_SHOW = 5

def visualize_predictions():
    # 1. Load Model
    model = AttU_Net(n_channels=1, n_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 2. Load the exact same validation set
    full_dataset = PancreasDataset(f"{DATA_DIR}/images", f"{DATA_DIR}/labels")
    n_val = int(len(full_dataset) * 0.1)
    n_train = len(full_dataset) - n_val
    torch.manual_seed(42)
    _, val_set = random_split(full_dataset, [n_train, n_val])
    
    loader = DataLoader(val_set, batch_size=NUM_IMAGES_TO_SHOW, shuffle=True)

    # 3. Get one batch of images and predict
    images, true_masks = next(iter(loader))
    images = images.to(DEVICE)
    
    with torch.no_grad():
        pred_logits = model(images)
        pred_masks = torch.softmax(pred_logits, dim=1).argmax(dim=1)

    # Move data to CPU for plotting
    images = images.cpu().numpy()
    true_masks = true_masks.cpu().numpy()
    pred_masks = pred_masks.cpu().numpy()

    # 4. Plot the results
    fig, axes = plt.subplots(NUM_IMAGES_TO_SHOW, 3, figsize=(12, NUM_IMAGES_TO_SHOW * 4))
    fig.suptitle("Model Predictions vs. Ground Truth", fontsize=16)
    
    for i in range(NUM_IMAGES_TO_SHOW):
        # Plot Original Image
        axes[i, 0].imshow(images[i, 0], cmap='gray')
        axes[i, 0].set_title(f"Original Image #{i+1}")
        axes[i, 0].axis('off')

        # Plot Ground Truth
        axes[i, 1].imshow(images[i, 0], cmap='gray')
        axes[i, 1].imshow(np.ma.masked_where(true_masks[i] == 0, true_masks[i]), cmap='spring', alpha=0.6)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')

        # Plot Model Prediction
        axes[i, 2].imshow(images[i, 0], cmap='gray')
        axes[i, 2].imshow(np.ma.masked_where(pred_masks[i] == 0, pred_masks[i]), cmap='cool', alpha=0.6)
        axes[i, 2].set_title("Model Prediction")
        axes[i, 2].axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    visualize_predictions()