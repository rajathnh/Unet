#for 80/20 split
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- CONFIG ---
# 1. Where are your current 2D slices?
SOURCE_DIR = Path("E:\Pancreas\Task07_Pancreas\pancreas_slices_2d")

# 2. Where do you want the new split data to go?
DEST_DIR = Path("./msd_split")

# --- SCRIPT ---

def split_dataset():
    # Create destination folders
    train_img_dir = DEST_DIR / "train" / "images"
    train_lbl_dir = DEST_DIR / "train" / "labels"
    val_img_dir = DEST_DIR / "val" / "images"
    val_lbl_dir = DEST_DIR / "val" / "labels"
    
    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_lbl_dir.mkdir(parents=True, exist_ok=True)
    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    source_img_dir = SOURCE_DIR / "images"
    source_lbl_dir = SOURCE_DIR / "labels"
    
    # 1. Get all unique patient IDs
    all_files = list(source_img_dir.glob("*.npy"))
    # Logic: "pancreas_001_slice_033.npy" -> "pancreas_001"
    patient_ids = sorted(list(set([f.name.split('_slice')[0] for f in all_files])))
    
    print(f"Found {len(patient_ids)} unique patients.")
    
    # 2. Split the PATIENT IDs
    train_ids, val_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)
    
    print(f"Splitting into {len(train_ids)} training patients and {len(val_ids)} validation patients.")
    
    # 3. Move the files
    print("Moving files...")
    for f in tqdm(all_files):
        patient_id = f.name.split('_slice')[0]
        label_file = source_lbl_dir / f.name
        
        if patient_id in train_ids:
            # Move to train folder
            shutil.move(str(f), str(train_img_dir / f.name))
            if label_file.exists():
                shutil.move(str(label_file), str(train_lbl_dir / f.name))
        elif patient_id in val_ids:
            # Move to val folder
            shutil.move(str(f), str(val_img_dir / f.name))
            if label_file.exists():
                shutil.move(str(label_file), str(val_lbl_dir / f.name))

    print("\nSplit complete!")
    print(f"Training data is in: {DEST_DIR / 'train'}")
    print(f"Validation data is in: {DEST_DIR / 'val'}")

if __name__ == "__main__":
    split_dataset()

#for 70/15/15 split
# import os
# import shutil
# from pathlib import Path
# from sklearn.model_selection import train_test_split
# from tqdm import tqdm

# # --- CONFIG ---
# SOURCE_DIR = Path("./pancreas_slices_2d_small") # Where your raw 2D slices are
# DEST_DIR = Path("./msd_final_split")      # The new top-level folder

# def split_dataset_final():
#     # Create destination folders for all three sets
#     (DEST_DIR / "train" / "images").mkdir(parents=True, exist_ok=True)
#     (DEST_DIR / "train" / "labels").mkdir(parents=True, exist_ok=True)
#     (DEST_DIR / "val" / "images").mkdir(parents=True, exist_ok=True)
#     (DEST_DIR / "val" / "labels").mkdir(parents=True, exist_ok=True)
#     (DEST_DIR / "test" / "images").mkdir(parents=True, exist_ok=True)
#     (DEST_DIR / "test" / "labels").mkdir(parents=True, exist_ok=True)
    
#     source_img_dir = SOURCE_DIR / "images"
#     source_lbl_dir = SOURCE_DIR / "labels"
    
#     # 1. Get all unique patient IDs
#     all_files = list(source_img_dir.glob("*.npy"))
#     patient_ids = sorted(list(set([f.name.split('_slice')[0] for f in all_files])))
    
#     # 2. First split: Carve off the 15% Test Set
#     # This leaves 85% for train+val
#     train_val_ids, test_ids = train_test_split(patient_ids, test_size=0.15, random_state=42)
    
#     # 3. Second split: Split the remaining 85% into train and val
#     # A 15% validation set from the original total is roughly 17.6% of the remaining 85%
#     # (0.15 / 0.85 = 0.176)
#     train_ids, val_ids = train_test_split(train_val_ids, test_size=0.1765, random_state=42)
    
#     print(f"Splitting into {len(train_ids)} train, {len(val_ids)} val, and {len(test_ids)} test patients.")
    
#     # 4. Move all the files
#     print("Moving files into train, val, and test folders...")
#     for f in tqdm(all_files):
#         patient_id = f.name.split('_slice')[0]
#         label_file = source_lbl_dir / f.name
        
#         # Determine destination
#         if patient_id in train_ids:
#             dest_img_dir = DEST_DIR / "train" / "images"
#             dest_lbl_dir = DEST_DIR / "train" / "labels"
#         elif patient_id in val_ids:
#             dest_img_dir = DEST_DIR / "val" / "images"
#             dest_lbl_dir = DEST_DIR / "val" / "labels"
#         elif patient_id in test_ids:
#             dest_img_dir = DEST_DIR / "test" / "images"
#             dest_lbl_dir = DEST_DIR / "test" / "labels"
#         else:
#             continue
            
#         # Move the image and its corresponding label
#         shutil.move(str(f), str(dest_img_dir / f.name))
#         if label_file.exists():
#             shutil.move(str(label_file), str(dest_lbl_dir / f.name))

#     print("\nSplit complete!")

# if __name__ == "__main__":
#     split_dataset_final()