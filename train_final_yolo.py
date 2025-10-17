# train_final_yolo.py
from ultralytics import YOLO
import torch
import time
from pathlib import Path

def clear_all_cache():
    """Clear ALL cache files"""
    print("🧹 Clearing ALL cache files...")
    cache_patterns = [
        '/mnt/34B471F7B471BBC4/CSO_project/datasets/*.cache',
        '/mnt/34B471F7B471BBC4/CSO_project/datasets/images/*.cache',
        '/mnt/34B471F7B471BBC4/CSO_project/datasets/labels/*.cache',
    ]
    
    for pattern in cache_patterns:
        for cache_file in Path('/').glob(pattern[1:]):
            try:
                cache_file.unlink()
                print(f"   ✓ {cache_file.name}")
            except:
                pass

def train_final_yolo():
    print("=== FINAL YOLO TRAINING ===")
    
    # Clear cache
    clear_all_cache()
    
    # Load model
    model = YOLO('yolov8s.pt')
    
    print("\n" + "=" * 60)
    print("🚀 FINAL ATTEMPT - EXACT YOLO STRUCTURE")
    print("=" * 60)
    print("Using: coco_yolo_exact.yaml")
    print("Structure: images/train/, labels/train/, images/val/, labels/val/")
    print("This SHOULD work!")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ No GPU!")
        return
    
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()
    
    start_time = time.time()
    print(f"⏰ Started at: {time.strftime('%H:%M:%S')}")
    
    try:
        results = model.train(
            data='coco_yolo_exact.yaml',
            epochs=30,
            imgsz=416,
            batch=16,
            workers=8,
            device=0,
            lr0=0.01,
            save=True,
            amp=True,
            optimizer='SGD',
            val=True,
            patience=10,
            verbose=True
        )
        
        end_time = time.time()
        training_duration = (end_time - start_time) / 60
        
        print(f"\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"⏱️  Total time: {training_duration:.2f} minutes")
        print(f"📁 Best model: {results.best}")
        
    except Exception as e:
        print(f"❌ Training error: {e}")

if __name__ == "__main__":
    # First create the exact YOLO structure
    print("Creating exact YOLO structure...")
    exec(open('final_fix_yolo_structure.py').read())
    
    # Then train
    train_final_yolo()