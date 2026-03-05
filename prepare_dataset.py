import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION (UPDATED FOR YOUR PATHS) ---
# We use r"..." to handle Windows backslashes correctly
dataset_root = Path(r"C:\Users\Raj\Downloads\Task07_Pancreas\Task07_Pancreas")

# Input folders
images_dir = dataset_root / "imagesTr"
labels_dir = dataset_root / "labelsTr"

# Output folder (We will create this inside your Downloads folder next to the main one)
# You can change this if you want it somewhere else
output_path = dataset_root.parent / "pancreas_slices_2d"

# Create output directories
output_images_dir = output_path / "images"
output_labels_dir = output_path / "labels"
output_images_dir.mkdir(parents=True, exist_ok=True)
output_labels_dir.mkdir(parents=True, exist_ok=True)

print(f"Reading data from: {dataset_root}")
print(f"Saving slices to:  {output_path}")

# --- MAIN LOGIC ---

# 1. Get all .nii.gz files
all_image_files = sorted(list(images_dir.glob("*.nii.gz")))

# 2. FILTER OUT THE JUNK (The Fix)
# We remove any file that starts with "._"
valid_image_files = [f for f in all_image_files if not f.name.startswith("._")]

print(f"Found {len(valid_image_files)} valid CT scans.")

for img_file in tqdm(valid_image_files):
    # Construct the corresponding label filename
    # Labels have the same name as images
    lbl_file = labels_dir / img_file.name
    
    # Check if label exists (just in case)
    if not lbl_file.exists():
        print(f"Warning: Label not found for {img_file.name}, skipping.")
        continue

    # Get the base name (e.g., "pancreas_001")
    base_name = img_file.name.replace(".nii.gz", "")

    try:
        # Load the 3D volumes
        img_vol = nib.load(img_file).get_fdata()
        lbl_vol = nib.load(lbl_file).get_fdata()
    except Exception as e:
        print(f"Error loading {img_file.name}: {e}")
        continue

    # --- Preprocessing (Clip Intensities) ---
    # Clip to standard soft tissue/pancreas range [-100, 240]
    img_vol = np.clip(img_vol, -100, 240)

    # --- Slicing Loop ---
    # Iterate through the Z-axis (height/depth)
    for i in range(img_vol.shape[2]):
        image_slice_2d = img_vol[:, :, i]
        label_slice_2d = lbl_vol[:, :, i]

        # --- Filter Empty Slices ---
        # Only save if there is a pancreas (label 1) or tumor (label 2) in this slice
        if np.sum(label_slice_2d) > 0:
            
            slice_filename = f"{base_name}_slice_{i:03d}.npy"
            
            # Save as numpy arrays
            np.save(output_images_dir / slice_filename, image_slice_2d)
            np.save(output_labels_dir / slice_filename, label_slice_2d)

print("\nProcessing Complete!")