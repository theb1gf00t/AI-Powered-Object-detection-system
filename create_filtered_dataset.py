# train_filtered.py
from ultralytics import YOLO
import torch
import time
from pathlib import Path

def clear_cache():
    """Clear cache files"""
    cache_files = [
        '/mnt/34B471F7B471BBC4/CSO_project/datasets/*.cache',
        '/mnt/34B471F7B471BBC4/CSO_project/datasets/filtered_train2017.cache',
        '/mnt/34B471F7B471BBC4/CSO_project/datasets/filtered_val2017.cache',
    ]
    
    print("🧹 Clearing cache files...")
    for pattern in cache_files:
        for cache_file in Path('/').glob(pattern[1:]):
            try:
                cache_file.unlink()
                print(f"   ✓ {cache_file.name}")
            except:
                pass

def setup_training():
    print("=== TRAINING ON FILTERED DATASET ===")
    
    # Clear cache
    clear_cache()
    
    # Load model
    model = YOLO('yolov8s.pt')
    
    # Training configuration
    training_config = {
        'data': 'coco_filtered.yaml',
        'epochs': 30,
        'imgsz': 416,
        'batch': 16,
        'workers': 8,
        'device': 0,
        'lr0': 0.01,
        'save': True,
        'amp': True,
        'optimizer': 'SGD',
        'cos_lr': False,
        'close_mosaic': 5,
        'val': True,
        'patience': 10,
        'verbose': True,
    }
    
    return model, training_config

def train_filtered():
    model, config = setup_training()
    
    print("\n" + "=" * 60)
    print("🚀 TRAINING ON FILTERED DATASET")
    print("=" * 60)
    print(f"📊 Dataset: 90,702 training images (ALL HAVE LABELS)")
    print(f"📊 Validation: 3,822 images (ALL HAVE LABELS)")
    print(f"🎯 Classes: 30 selected COCO classes")
    print(f"⏱️  Expected: 60-90 minutes")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ No GPU detected!")
        return None, None
    
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()
    
    start_time = time.time()
    print(f"⏰ Started at: {time.strftime('%H:%M:%S')}")
    
    try:
        results = model.train(**config)
        end_time = time.time()
        
        training_duration = (end_time - start_time) / 60
        print(f"\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"⏱️  Total time: {training_duration:.2f} minutes")
        print(f"📁 Best model: {results.best}")
        
        return model, results
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\n⚠️  GPU Out of Memory! Reducing batch size...")
            config['batch'] = 8
            torch.cuda.empty_cache()
            results = model.train(**config)
            
            end_time = time.time()
            training_duration = (end_time - start_time) / 60
            print(f"\n🎉 TRAINING COMPLETED!")
            print(f"⏱️  Total time: {training_duration:.2f} minutes")
            return model, results
        else:
            print(f"❌ Training error: {e}")
            raise e

if __name__ == "__main__":
    # First create the filtered dataset
    print("Creating filtered dataset...")
    exec(open('create_filtered_dataset.py').read())
    
    # Then train
    model, results = train_filtered()