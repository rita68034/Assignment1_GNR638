# Paths
#dataset_path = r'C:\Users\priya\Downloads\UCMerced_LandUse' # Replace with your dataset path
#output_path = r"C:\Users\priya\Documents\split_dataset"  # Replace with your desired output path
import os
import shutil
import random

# Paths
dataset_path = r'C:\Users\priya\Downloads\UCMerced_LandUse\UCMerced_LandUse\Images' # Replace with your dataset path
output_path = r"C:\Users\priya\Documents\split_dataset" 

# Split ratios
train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2

# Create output directories
splits = ['train', 'val', 'test']
for split in splits:
    for category in os.listdir(dataset_path):
        os.makedirs(os.path.join(output_path, split, category), exist_ok=True)

# Split the data
for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)
    if not os.path.isdir(category_path):
        continue

    # Get all image files and shuffle them
    images = [img for img in os.listdir(category_path) if img.endswith(('.jpg', '.png', '.tif', '.jpeg'))]
    images.sort()  # Ensure consistent ordering
    random.seed(42)  # For reproducibility
    random.shuffle(images)

    # Check if any images were found
    if len(images) == 0:
        print(f"Warning: No images found in category {category}")
        continue

    print(f"Images in {category}: {images[:5]}...")  # Print first 5 images for sanity check

    # Calculate split indices
    total_images = len(images)
    train_end = int(train_ratio * total_images)
    val_end = train_end + int(val_ratio * total_images)

    # Assign images to splits
    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    # Debug print to verify the split
    print(f"Total images: {total_images}, Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")

    # Move files
    for img in train_images:
        src = os.path.join(category_path, img)
        dest = os.path.join(output_path, 'train', category, img)
        print(f"Copying {src} to {dest}")  # Debugging copy
        shutil.copy(src, dest)
    for img in val_images:
        src = os.path.join(category_path, img)
        dest = os.path.join(output_path, 'val', category, img)
        print(f"Copying {src} to {dest}")
        shutil.copy(src, dest)
    for img in test_images:
        src = os.path.join(category_path, img)
        dest = os.path.join(output_path, 'test', category, img)
        print(f"Copying {src} to {dest}")
        shutil.copy(src, dest)

print("Dataset split completed!")
