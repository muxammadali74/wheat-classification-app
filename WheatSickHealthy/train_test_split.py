import os
import shutil
import random


source_dir = '/dataset/Wheat'
train_dir = '/dataset/HealthySickWheat/train'
test_dir = '/dataset/HealthySickWheat/test'


print(f"Source dir: {source_dir}")
print(f"Train dir: {train_dir}")
print(f"Test dir: {test_dir}")

os.makedirs(train_dir, exist_ok=True,)
os.makedirs(test_dir, exist_ok=True,)

classes = ['healthy', 'sick']

train_split = 0.8

for className in classes:
    os.makedirs(os.path.join(train_dir, className), exist_ok=True)
    os.makedirs(os.path.join(test_dir, className), exist_ok=True)

    class_path = os.path.join(source_dir, className)
    images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg', '.jfif'))]

    random.shuffle(images)

    train_size = int(len(images) *  train_split)

    train_images = images[:train_size]
    test_images = images[train_size:]


    for img  in train_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(train_dir, className, img)
        )

    for img  in test_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(test_dir, className, img)
        )

print("Разделение завершено!")