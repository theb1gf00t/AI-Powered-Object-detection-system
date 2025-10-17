# final_fix_yolo_structure.py
import shutil
from pathlib import Path

def create_exact_yolo_structure():
    """Create EXACT YOLO directory structure"""
    
    base_path = Path('/mnt/34B471F7B471BBC4/CSO_project/datasets')
    
    print("🎯 CREATING EXACT YOLO STRUCTURE")
    print("=" * 50)
    
    # Remove any existing directories
    yolo_train = base_path / 'images' / 'train'
    yolo_val = base_path / 'images' / 'val'
    yolo_train_labels = base_path / 'labels' / 'train'
    yolo_val_labels = base_path / 'labels' / 'val'
    
    # Clean up
    for dir_path in [yolo_train, yolo_val, yolo_train_labels, yolo_val_labels]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("✓ Created YOLO directory structure")
    
    # Copy images and labels to YOLO structure
    print("📁 Copying training data to YOLO structure...")
    
    # Training data
    original_train_images = base_path / 'filtered_train2017'
    original_train_labels = base_path / 'filtered_train2017' / 'labels'
    
    train_count = 0
    for image_file in original_train_images.glob('*.jpg'):
        # Copy image
        shutil.copy2(image_file, yolo_train / image_file.name)
        
        # Copy corresponding label
        label_file = original_train_labels / f"{image_file.stem}.txt"
        if label_file.exists():
            shutil.copy2(label_file, yolo_train_labels / f"{image_file.stem}.txt")
            train_count += 1
    
    print(f"✓ Copied {train_count} training images & labels")
    
    # Validation data
    print("📁 Copying validation data to YOLO structure...")
    
    original_val_images = base_path / 'filtered_val2017'
    original_val_labels = base_path / 'filtered_val2017' / 'labels'
    
    val_count = 0
    for image_file in original_val_images.glob('*.jpg'):
        # Copy image
        shutil.copy2(image_file, yolo_val / image_file.name)
        
        # Copy corresponding label
        label_file = original_val_labels / f"{image_file.stem}.txt"
        if label_file.exists():
            shutil.copy2(label_file, yolo_val_labels / f"{image_file.stem}.txt")
            val_count += 1
    
    print(f"✓ Copied {val_count} validation images & labels")
    
    # Create FINAL YAML
    create_final_yolo_yaml(train_count, val_count)
    
    return train_count, val_count

def create_final_yolo_yaml(train_count, val_count):
    """Create YAML that uses exact YOLO structure"""
    
    yaml_content = f"""# YOLO dataset configuration
# Exact structure that YOLO expects
path: /mnt/34B471F7B471BBC4/CSO_project/datasets

# YOLO expects these exact directory names
train: images/train
val: images/val

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
    
    with open('coco_yolo_exact.yaml', 'w') as f:
        f.write(yaml_content)
    
    print(f"✓ Created coco_yolo_exact.yaml")
    print(f"📊 Dataset: {train_count} train, {val_count} val")

def verify_yolo_structure():
    """Verify the YOLO structure is correct"""
    
    base_path = Path('/mnt/34B471F7B471BBC4/CSO_project/datasets')
    
    print("\n🔍 VERIFYING YOLO STRUCTURE:")
    print("Expected structure:")
    print("datasets/")
    print("├── images/")
    print("│   ├── train/       # Training images")
    print("│   └── val/         # Validation images")
    print("└── labels/")
    print("    ├── train/       # Training labels")
    print("    └── val/         # Validation labels")
    print()
    
    # Check structure
    train_images = len(list((base_path / 'images' / 'train').glob('*.jpg')))
    train_labels = len(list((base_path / 'labels' / 'train').glob('*.txt')))
    val_images = len(list((base_path / 'images' / 'val').glob('*.jpg')))
    val_labels = len(list((base_path / 'labels' / 'val').glob('*.txt')))
    
    print(f"Training: {train_images} images, {train_labels} labels")
    print(f"Validation: {val_images} images, {val_labels} labels")
    
    if train_images == train_labels and val_images == val_labels:
        print("✅ PERFECT! YOLO structure is correct!")
    else:
        print("❌ Structure mismatch!")

if __name__ == "__main__":
    train_count, val_count = create_exact_yolo_structure()
    verify_yolo_structure()
    
    print(f"\n🎯 FINAL SOLUTION READY!")
    print(f"🚀 Run: python train_final_yolo.py")