import os
import shutil
import random
import kagglehub

path = kagglehub.dataset_download("aliochilov/wheatdataset")

print("Path to dataset files:", path)

source_dir = 'Wheat'
train_dir = 'train'
test_dir = 'test'

print(f"Source dir: {source_dir}")
print(f"Train dir: {train_dir}")
print(f"Test dir: {test_dir}")

if not os.path.exists(source_dir):
    print(f"Error: Source directory {source_dir} does not exist!")
    exit(1)

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

classes = ['healthy', 'sick']  
print(f"Classes: {classes}")

train_split = 0.8

for class_name in classes:
    class_path = os.path.join(source_dir, class_name)
    print(f"Checking path: {class_path}")
    
    if not os.path.exists(class_path):
        print(f"Error: Directory {class_path} does not exist!")
        continue
    
    train_class_dir = os.path.join(train_dir, class_name)
    test_class_dir = os.path.join(test_dir, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)
    
    images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg', '.jfif'))]
    print(f"Found {len(images)} images in {class_path}")
    
    random.shuffle(images)
    
    train_size = int(len(images) * train_split)
    
    train_images = images[:train_size]
    test_images = images[train_size:]
    
    for img in train_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(train_class_dir, img)
        )
    
    for img in test_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(test_class_dir, img)
        )

print("Разделение завершено!")