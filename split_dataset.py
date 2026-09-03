import os
import shutil
import random

def split_data(source_dir, train_dir, val_dir, split_ratio=0.8):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    for class_folder in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_folder)

        if not os.path.isdir(class_path):
            continue

        # Only include files (not directories)
        images = [img for img in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, img))]
        random.shuffle(images)

        split_point = int(len(images) * split_ratio)
        train_images = images[:split_point]
        val_images = images[split_point:]

        train_class_path = os.path.join(train_dir, class_folder)
        val_class_path = os.path.join(val_dir, class_folder)

        os.makedirs(train_class_path, exist_ok=True)
        os.makedirs(val_class_path, exist_ok=True)

        for img in train_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(train_class_path, img)
            shutil.copyfile(src, dst)

        for img in val_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(val_class_path, img)
            shutil.copyfile(src, dst)

    print("✅ Dataset split complete.")

# ---- EDIT THESE PATHS BELOW ----
source_dir = "dataset/PlantVillage"
train_dir = "dataset/train"
val_dir = "dataset/val"

split_data(source_dir, train_dir, val_dir)
