import json
from pathlib import Path

def filter_coco_dataset(original_annotations_path, output_annotations_path, selected_classes):
    """
    Filter COCO dataset to only include selected classes
    """
    print(f"Loading COCO annotations from: {original_annotations_path}")
    
    with open(original_annotations_path, 'r') as f:
        coco_data = json.load(f)
    
    # Get class IDs for selected classes
    selected_class_ids = []
    class_name_to_id = {}
    class_id_to_name = {}
    
    for category in coco_data['categories']:
        if category['name'] in selected_classes:
            selected_class_ids.append(category['id'])
            class_name_to_id[category['name']] = category['id']
            class_id_to_name[category['id']] = category['name']
    
    print(f"Selected {len(selected_class_ids)} classes: {list(class_name_to_id.keys())}")
    
    # Filter annotations that contain selected classes
    selected_image_ids = set()
    filtered_annotations = []
    
    for annotation in coco_data['annotations']:
        if annotation['category_id'] in selected_class_ids:
            filtered_annotations.append(annotation)
            selected_image_ids.add(annotation['image_id'])
    
    # Filter images
    filtered_images = [
        img for img in coco_data['images'] 
        if img['id'] in selected_image_ids
    ]
    
    # Filter categories
    filtered_categories = [
        cat for cat in coco_data['categories']
        if cat['id'] in selected_class_ids
    ]
    
    # Create filtered dataset
    filtered_data = {
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': filtered_categories
    }
    
    # Save filtered annotations
    with open(output_annotations_path, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    print(f"✓ Filtered dataset saved to: {output_annotations_path}")
    print(f"✓ Selected {len(filtered_images)} images and {len(filtered_annotations)} annotations")
    print(f"✓ Remaining classes: {[cat['name'] for cat in filtered_categories]}")
    
    return filtered_data

# Your 30 selected COCO classes
SELECTED_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'backpack', 'umbrella', 'handbag', 'tie',
    'skis', 'snowboard', 'sports ball', 'kite',
    'banana', 'apple', 'sandwich'
]

def filter_both_train_val():
    """Filter both train and validation datasets"""
    
    # Paths - adjust these to your actual COCO 2017 paths
    base_path = Path('/path/to/coco2017')  # Change this to your COCO path
    annotations_path = base_path / 'annotations'
    
    # Filter training set
    print("Filtering TRAIN dataset...")
    filter_coco_dataset(
        original_annotations_path=annotations_path / 'instances_train2017.json',
        output_annotations_path=annotations_path / 'instances_train2017_filtered.json',
        selected_classes=SELECTED_CLASSES
    )
    
    print("\n" + "="*50 + "\n")
    
    # Filter validation set
    print("Filtering VALIDATION dataset...")
    filter_coco_dataset(
        original_annotations_path=annotations_path / 'instances_val2017.json',
        output_annotations_path=annotations_path / 'instances_val2017_filtered.json',
        selected_classes=SELECTED_CLASSES
    )

def create_custom_yaml_config():
    """Create custom YAML configuration for filtered dataset"""
    
    config_content = f"""# Custom COCO 2017 dataset with 30 classes
path: /path/to/coco2017  # Change this to your dataset path
train: train2017
val: val2017

# Number of classes
nc: 30

# Class names
names: {SELECTED_CLASSES}
"""
    
    with open('custom_coco.yaml', 'w') as f:
        f.write(config_content)
    
    print("✓ Created custom_coco.yaml configuration file")

if __name__ == "__main__":
    # Run filtering
    filter_both_train_val()
    
    # Create YAML config
    create_custom_yaml_config()
    
    print("\n🎉 Dataset filtering complete!")
    print("Next steps:")
    print("1. Update the path in custom_coco.yaml to your actual COCO directory")
    print("2. Use custom_coco.yaml in your YOLO training script")
