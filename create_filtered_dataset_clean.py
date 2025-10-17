# create_filtered_dataset_clean.py
import shutil
from pathlib import Path

def create_clean_filtered_dataset():
    """Create a clean filtered dataset without symlink issues"""
    
    base_path = Path('/mnt/34B471F7B471BBC4/CSO_project/datasets')
    
    print("🔄 Creating CLEAN filtered dataset...")
    
    # Remove any existing filtered directories
    filtered_train_path = base_path / 'filtered_train2017'
    filtered_val_path = base_path / 'filtered_val2017'
    
    if filtered_train_path.exists():
        shutil.rmtree(filtered_train_path)
    if filtered_val_path.exists():
        shutil.rmtree(filtered_val_path)
    
    # Create new directories
    filtered_train_path.mkdir(exist_ok=True)
    filtered_val_path.mkdir(exist_ok=True)
    
    # Create label directories
    (filtered_train_path / 'labels').mkdir(exist_ok=True)
    (filtered_val_path / 'labels').mkdir(exist_ok=True)
    
    # Copy training images that have labels (from ORIGINAL labels directory)
    print("📁 Copying training images with labels...")
    original_train_labels = base_path / 'labels' / 'train2017'
    original_train_images = base_path / 'train2017'
    
    train_count = 0
    for label_file in original_train_labels.glob('*.txt'):
        image_file = original_train_images / f"{label_file.stem}.jpg"
        if image_file.exists():
            # Copy image to new location
            shutil.copy2(image_file, filtered_train_path / image_file.name)
            # Copy label to new location
            shutil.copy2(label_file, filtered_train_path / 'labels' / label_file.name)
            train_count += 1
    
    print(f"✓ Copied {train_count} training images with labels")
    
    # Copy validation images that have labels (from ORIGINAL labels directory)
    print("📁 Copying validation images with labels...")
    original_val_labels = base_path / 'labels' / 'val2017'
    original_val_images = base_path / 'validation' / 'val2017'
    
    val_count = 0
    for label_file in original_val_labels.glob('*.txt'):
        image_file = original_val_images / f"{label_file.stem}.jpg"
        if image_file.exists():
            # Copy image to new location
            shutil.copy2(image_file, filtered_val_path / image_file.name)
            # Copy label to new location
            shutil.copy2(label_file, filtered_val_path / 'labels' / label_file.name)
            val_count += 1
    
    print(f"✓ Copied {val_count} validation images with labels")
    
    # Create YAML for filtered dataset
    create_filtered_yaml(train_count, val_count)
    
    return train_count, val_count

def create_filtered_yaml(train_count, val_count):
    """Create YAML for the filtered dataset"""
    
    yaml_content = f"""# Filtered COCO 2017 dataset with 30 classes
# Contains only images that have labels for the selected classes
path: /mnt/34B471F7B471BBC4/CSO_project/datasets

# Training - all images here have corresponding labels
train: filtered_train2017
val: filtered_val2017

# Number of classes
nc: 30

# Class names
names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'backpack', 'umbrella', 'handbag', 'tie',
        'skis', 'snowboard', 'sports ball', 'kite',
        'banana', 'apple', 'sandwich']
"""
    
    with open('coco_filtered.yaml', 'w') as f:
        f.write(yaml_content)
    
    print(f"✓ Created coco_filtered.yaml")
    print(f"📊 Filtered dataset: {train_count} train, {val_count} val images")

if __name__ == "__main__":
    train_count, val_count = create_clean_filtered_dataset()
    
    print(f"\n🎉 CLEAN FILTERED DATASET CREATED!")
    print(f"📁 Training: {train_count} images (all have labels)")
    print(f"📁 Validation: {val_count} images (all have labels)")
    print(f"📊 No more 'background' images!")
    print(f"\n🚀 Now run: python train_simple.py")