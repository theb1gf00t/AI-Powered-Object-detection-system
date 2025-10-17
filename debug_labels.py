# debug_labels.py
from pathlib import Path

def debug_label_structure():
    """Debug why YOLO can't find the labels"""
    
    base_path = Path('/mnt/34B471F7B471BBC4/CSO_project/datasets')
    
    print("🔍 DEBUGGING LABEL STRUCTURE")
    print("=" * 50)
    
    # Check training structure
    train_images = base_path / 'train2017'
    train_labels = train_images / 'labels'
    
    print(f"Training directory: {train_images}")
    print(f"Training labels: {train_labels}")
    
    # Count files
    image_files = list(train_images.glob('*.jpg'))
    label_files = list(train_labels.glob('*.txt'))
    
    print(f"Training images: {len(image_files)}")
    print(f"Training labels: {len(label_files)}")
    
    # Check if they match
    if len(image_files) > 0 and len(label_files) > 0:
        # Check first few images have corresponding labels
        sample_images = image_files[:5]
        for img in sample_images:
            label_file = train_labels / f"{img.stem}.txt"
            exists = "✅" if label_file.exists() else "❌"
            print(f"  {img.name} -> {label_file.name} {exists}")
    
    # Check validation structure
    val_images = base_path / 'validation' / 'val2017'
    val_labels = val_images / 'labels'
    
    print(f"\nValidation directory: {val_images}")
    print(f"Validation labels: {val_labels}")
    
    val_image_files = list(val_images.glob('*.jpg'))
    val_label_files = list(val_labels.glob('*.txt'))
    
    print(f"Validation images: {len(val_image_files)}")
    print(f"Validation labels: {len(val_label_files)}")
    
    # Check YAML file
    print(f"\n📁 YAML file: coco_final.yaml")
    with open('coco_final.yaml', 'r') as f:
        yaml_content = f.read()
        print("YAML content:")
        print("-" * 30)
        for line in yaml_content.split('\n')[:10]:
            print(f"  {line}")

def check_symlinks():
    """Check if we need to use symlinks instead"""
    base_path = Path('/mnt/34B471F7B471BBC4/CSO_project/datasets')
    
    print(f"\n🔗 Checking symlink option...")
    
    # Create symlinks from original labels to the new structure
    original_train_labels = base_path / 'labels' / 'train2017'
    target_train_labels = base_path / 'train2017' / 'labels'
    
    # Remove existing and create symlinks
    import shutil
    if target_train_labels.exists():
        shutil.rmtree(target_train_labels)
    
    target_train_labels.symlink_to(original_train_labels, target_is_directory=True)
    print(f"✓ Created symlink: {target_train_labels} -> {original_train_labels}")
    
    # Do the same for validation
    original_val_labels = base_path / 'labels' / 'val2017'
    target_val_labels = base_path / 'validation' / 'val2017' / 'labels'
    
    if target_val_labels.exists():
        shutil.rmtree(target_val_labels)
    
    target_val_labels.symlink_to(original_val_labels, target_is_directory=True)
    print(f"✓ Created symlink: {target_val_labels} -> {original_val_labels}")

if __name__ == "__main__":
    debug_label_structure()
    
    print(f"\n🔄 Trying symlink solution...")
    check_symlinks()
    
    print(f"\n🎯 Now delete cache files and try again:")
    print(f"rm /mnt/34B471F7B471BBC4/CSO_project/datasets/*.cache")
    print(f"python train_final_fixed.py")