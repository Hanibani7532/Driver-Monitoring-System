import os
import shutil
import random
from pathlib import Path

# Identify the project root directory relative to the script's location
current_dir = Path(__file__).resolve().parent.parent 

# Define exact paths based on the established folder structure:
# Desktop > Driver_Monitoring_System > Data > Row > state-farm-distracted-driver-detection > imgs > train
raw_data_path = current_dir / "Data" / "Row" / "state-farm-distracted-driver-detection" / "imgs" / "train"
processed_path = current_dir / "Data" / "processed" / "classification_data"

print(f"Searching for raw data at: {raw_data_path}")

# Verify if the raw data directory exists before proceeding
if not raw_data_path.exists():
    print("\n❌ Error: Raw data folder not found!")
    print(f"Please verify folder naming (case sensitivity). Looking for: {raw_data_path}")
else:
    # Initialize 10 standard classes (c0 to c9) for the distraction dataset
    classes = [f'c{i}' for i in range(10)]
    
    for cls in classes:
        # Create destination directories for training and validation splits
        (processed_path / 'train' / cls).mkdir(parents=True, exist_ok=True)
        (processed_path / 'val' / cls).mkdir(parents=True, exist_ok=True)

        src_folder = raw_data_path / cls
        if src_folder.exists():
            # Filter and collect only image files
            images = [f for f in os.listdir(src_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            random.shuffle(images)

            # Limit to 500 images per class for faster processing and upload management
            limit = min(len(images), 500) 
            split_idx = int(limit * 0.8) # 80-20 Train/Val Split
            
            train_imgs = images[:split_idx]
            val_imgs = images[split_idx:limit]

            print(f"✅ Processing {cls}: Copying {len(train_imgs)} images to training set...")

            # Copy files to their respective processed directories
            for img in train_imgs:
                shutil.copy(src_folder / img, processed_path / 'train' / cls / img)
            for img in val_imgs:
                shutil.copy(src_folder / img, processed_path / 'val' / cls / img)
        else:
            print(f"⚠️ Warning: Class folder {cls} was not found in the source directory.")

    print("\n✨ Processing Complete! You can now ZIP 'Data/processed/classification_data' for cloud upload.")
