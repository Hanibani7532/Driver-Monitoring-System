import os
import shutil
import random
from pathlib import Path

# Script ki location se project root dhoondna
current_dir = Path(__file__).resolve().parent.parent 

# Aapke bataye hue folder structure ke mutabiq exact path
# Desktop > Driver_Monitoring_System > Data > Row > state-farm-distracted-driver-detection > imgs > train
raw_data_path = current_dir / "Data" / "Row" / "state-farm-distracted-driver-detection" / "imgs" / "train"
processed_path = current_dir / "Data" / "processed" / "classification_data"

print(f"Searching data at: {raw_data_path}")

# Check karein ke folder waqai wahan hai ya nahi
if not raw_data_path.exists():
    print("\n❌ Error: Folder abhi bhi nahi mila!")
    print(f"Please check karein ke kya aapka folder 'Data' (D capital) hai ya 'data'?")
    print(f"Script currently yahan dhoond rahi hai: {raw_data_path}")
else:
    classes = [f'c{i}' for i in range(10)]
    
    for cls in classes:
        # Create processed folders
        (processed_path / 'train' / cls).mkdir(parents=True, exist_ok=True)
        (processed_path / 'val' / cls).mkdir(parents=True, exist_ok=True)

        src_folder = raw_data_path / cls
        if src_folder.exists():
            # Sirf images ko filter karein
            images = [f for f in os.listdir(src_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            random.shuffle(images)

            # 500 images per class limit (Upload asaan karne ke liye)
            limit = min(len(images), 500) 
            split_idx = int(limit * 0.8)
            
            train_imgs = images[:split_idx]
            val_imgs = images[split_idx:limit]

            print(f"✅ Processing {cls}: {len(train_imgs)} train images copy ho rahi hain...")

            for img in train_imgs:
                shutil.copy(src_folder / img, processed_path / 'train' / cls / img)
            for img in val_imgs:
                shutil.copy(src_folder / img, processed_path / 'val' / cls / img)
        else:
            print(f"⚠️ Warning: Class folder {cls} nahi mila.")

    print("\n✨ Done! 'Data/processed/classification_data' folder ko ZIP karein aur Drive par upload karein.")