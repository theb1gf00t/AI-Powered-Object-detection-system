# class_detection_counter.py
from ultralytics import YOLO
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict
import json

def count_detections_in_test_set():
    """Count how many of each class is detected in the entire test set"""
    
    # Load your trained model
    model = YOLO('/mnt/34B471F7B471BBC4/CSO_project/runs/detect/train7/weights/best.pt')
    
    # Get test images
    test_images_path = Path('/mnt/34B471F7B471BBC4/CSO_project/datasets/test_dataset/test2017')
    all_images = list(test_images_path.glob('*.jpg'))
    
    if not all_images:
        print("❌ No test images found!")
        return None
    
    print(f"🔍 Analyzing {len(all_images)} test images...")
    print("This will count detections for all 30 classes in your model!")
    
    # Initialize counters
    class_counter = Counter()
    image_detection_count = defaultdict(int)  # How many images contain each class
    confidence_scores = defaultdict(list)     # Store confidence scores for each class
    
    # Process all images
    total_detections = 0
    images_with_detections = 0
    
    for i, img_path in enumerate(all_images):
        if i % 100 == 0:
            print(f"📊 Processed {i}/{len(all_images)} images...")
        
        # Run inference
        results = model(img_path, verbose=False)
        result = results[0]
        
        # Get detections for this image
        boxes = result.boxes
        if len(boxes) > 0:
            images_with_detections += 1
            total_detections += len(boxes)
            
            # Track which classes were detected in this image
            detected_classes_in_image = set()
            
            for box in boxes:
                class_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                class_name = model.names[class_id]
                
                # Count the detection
                class_counter[class_name] += 1
                confidence_scores[class_name].append(confidence)
                detected_classes_in_image.add(class_name)
            
            # Count images that contain each class
            for class_name in detected_classes_in_image:
                image_detection_count[class_name] += 1
    
    print(f"\n✅ Analysis Complete!")
    print(f"📊 Total images processed: {len(all_images)}")
    print(f"📦 Images with detections: {images_with_detections}")
    print(f"🎯 Total objects detected: {total_detections}")
    
    return {
        'class_counts': class_counter,
        'image_counts': image_detection_count,
        'confidence_scores': confidence_scores,
        'total_images': len(all_images),
        'images_with_detections': images_with_detections,
        'total_detections': total_detections
    }

def print_detection_summary(results):
    """Print a detailed summary of detections"""
    
    class_counts = results['class_counts']
    image_counts = results['image_counts']
    confidence_scores = results['confidence_scores']
    
    print("\n" + "="*70)
    print("🎯 DETECTION SUMMARY - ALL TEST IMAGES")
    print("="*70)
    
    # Sort classes by detection count (descending)
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{'Class':<20} {'Count':<8} {'Avg Conf':<10} {'Images':<8} {'% of Total':<10}")
    print("-" * 70)
    
    for class_name, count in sorted_classes:
        avg_confidence = np.mean(confidence_scores[class_name]) if confidence_scores[class_name] else 0
        image_count = image_counts.get(class_name, 0)
        percentage = (count / results['total_detections']) * 100
        
        print(f"{class_name:<20} {count:<8} {avg_confidence:.3f}     {image_count:<8} {percentage:>6.1f}%")
    
    print("-" * 70)
    print(f"{'TOTAL':<20} {results['total_detections']:<8} {'':<10} {results['images_with_detections']:<8} {'100%':>10}")

def plot_detection_distribution(results):
    """Plot the distribution of detections across classes"""
    
    class_counts = results['class_counts']
    
    # Sort classes by count
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    classes = [item[0] for item in sorted_classes]
    counts = [item[1] for item in sorted_classes]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Bar chart of detection counts
    bars = ax1.bar(classes, counts, color='skyblue', alpha=0.8)
    ax1.set_title('Number of Detections per Class', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Detections')
    ax1.set_xlabel('Class Name')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Pie chart of detection distribution
    total = sum(counts)
    percentages = [(count/total)*100 for count in counts]
    
    # Only show top 15 in pie chart for readability
    if len(classes) > 15:
        top_classes = classes[:15]
        top_percentages = percentages[:15]
        other_percentage = sum(percentages[15:])
        pie_labels = top_classes + ['Other']
        pie_sizes = top_percentages + [other_percentage]
    else:
        pie_labels = classes
        pie_sizes = percentages
    
    wedges, texts, autotexts = ax2.pie(pie_sizes, labels=pie_labels, autopct='%1.1f%%',
                                      startangle=90, textprops={'fontsize': 8})
    ax2.set_title('Detection Distribution (%)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('detection_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confidence_distribution(results):
    """Plot confidence score distribution for each class"""
    
    confidence_scores = results['confidence_scores']
    
    # Get classes with at least some detections
    classes_with_detections = [cls for cls, scores in confidence_scores.items() if scores]
    
    if not classes_with_detections:
        print("No confidence data to plot!")
        return
    
    # Calculate average confidence for each class
    avg_confidences = {cls: np.mean(scores) for cls, scores in confidence_scores.items() if scores}
    
    # Sort by average confidence
    sorted_classes = sorted(avg_confidences.items(), key=lambda x: x[1], reverse=True)
    classes = [item[0] for item in sorted_classes]
    avg_conf = [item[1] for item in sorted_classes]
    
    # Create plot
    plt.figure(figsize=(12, 8))
    bars = plt.bar(classes, avg_conf, color='lightcoral', alpha=0.8)
    plt.title('Average Confidence Score per Class', fontsize=14, fontweight='bold')
    plt.ylabel('Average Confidence Score')
    plt.xlabel('Class Name')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_detection_results(results):
    """Save detection results to CSV files"""
    
    class_counts = results['class_counts']
    image_counts = results['image_counts']
    confidence_scores = results['confidence_scores']
    
    # Create DataFrame for summary
    data = []
    for class_name in class_counts.keys():
        data.append({
            'class_name': class_name,
            'detection_count': class_counts[class_name],
            'image_count': image_counts.get(class_name, 0),
            'average_confidence': np.mean(confidence_scores[class_name]) if confidence_scores[class_name] else 0,
            'min_confidence': np.min(confidence_scores[class_name]) if confidence_scores[class_name] else 0,
            'max_confidence': np.max(confidence_scores[class_name]) if confidence_scores[class_name] else 0,
            'std_confidence': np.std(confidence_scores[class_name]) if confidence_scores[class_name] else 0
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('detection_count', ascending=False)
    df.to_csv('detection_summary.csv', index=False)
    
    # Save raw confidence scores
    confidence_data = []
    for class_name, scores in confidence_scores.items():
        for score in scores:
            confidence_data.append({
                'class_name': class_name,
                'confidence_score': score
            })
    
    confidence_df = pd.DataFrame(confidence_data)
    confidence_df.to_csv('confidence_scores.csv', index=False)
    
    print("💾 Results saved to:")
    print("   - detection_summary.csv")
    print("   - confidence_scores.csv")

def quick_class_analysis():
    """Quick analysis with progress updates"""
    
    model = YOLO('/mnt/34B471F7B471BBC4/CSO_project/runs/detect/train7/weights/best.pt')
    test_images_path = Path('/mnt/34B471F7B471BBC4/CSO_project/datasets/test_dataset/test2017')
    all_images = list(test_images_path.glob('*.jpg'))
    
    print(f"🔍 Quick analysis of {len(all_images)} images...")
    
    class_counter = Counter()
    
    for i, img_path in enumerate(all_images[:500]):  # Limit to 500 for quick test
        if i % 50 == 0:
            print(f"   Processed {i}/500 images...")
        
        results = model(img_path, verbose=False)
        result = results[0]
        
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0].cpu().numpy())
            class_name = model.names[class_id]
            class_counter[class_name] += 1
    
    print("\n🎯 QUICK DETECTION COUNT (First 500 images):")
    for class_name, count in class_counter.most_common():
        print(f"   {class_name}: {count}")

if __name__ == "__main__":
    print("🎯 CLASS DETECTION COUNTER")
    print("Analyzing all test images to count detections for each class...")
    
    # Option 1: Quick analysis (500 images)
    # quick_class_analysis()
    
    # Option 2: Full analysis (all images)
    results = count_detections_in_test_set()
    
    if results:
        # Print summary
        print_detection_summary(results)
        
        # Create plots
        plot_detection_distribution(results)
        plot_confidence_distribution(results)
        
        # Save results
        save_detection_results(results)
        
        print(f"\n🎉 Analysis completed!")
        print(f"📊 Found {results['total_detections']} total objects across {results['images_with_detections']} images")