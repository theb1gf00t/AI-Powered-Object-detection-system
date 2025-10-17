# performance_4k_test.py
from ultralytics import YOLO
import time
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import psutil
import subprocess
import threading
from collections import deque
import os

class SystemMonitor:
    def __init__(self):
        self.cpu_readings = deque()
        self.ram_readings = deque()
        self.gpu_readings = deque()
        self.monitoring = False
        
    def get_gpu_usage(self):
        """Get GPU usage percentage using nvidia-smi"""
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                gpu_data = result.stdout.strip().split(', ')
                if len(gpu_data) >= 3:
                    return {
                        'gpu_utilization': float(gpu_data[0]),
                        'memory_used': float(gpu_data[1]),
                        'memory_total': float(gpu_data[2]),
                        'memory_percent': (float(gpu_data[1]) / float(gpu_data[2])) * 100,
                        'temperature': float(gpu_data[3]) if len(gpu_data) >= 4 else 0
                    }
        except:
            pass
        return None
    
    def get_system_usage(self):
        """Get CPU and RAM usage"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'ram_percent': psutil.virtual_memory().percent,
            'ram_used_gb': psutil.virtual_memory().used / 1e9,
            'ram_total_gb': psutil.virtual_memory().total / 1e9
        }
    
    def monitor_continuously(self, duration=300):  # Longer duration for 4k images
        """Monitor system usage continuously"""
        self.cpu_readings.clear()
        self.ram_readings.clear()
        self.gpu_readings.clear()
        self.monitoring = True
        
        start_time = time.time()
        while self.monitoring and (time.time() - start_time) < duration:
            # Get system usage
            system_usage = self.get_system_usage()
            self.cpu_readings.append(system_usage['cpu_percent'])
            self.ram_readings.append(system_usage['ram_percent'])
            
            # Get GPU usage
            gpu_usage = self.get_gpu_usage()
            if gpu_usage:
                self.gpu_readings.append(gpu_usage['gpu_utilization'])
            
            time.sleep(1)  # Sample every second for longer test
    
    def get_average_usage(self):
        """Get average usage during monitoring period"""
        cpu_avg = np.mean(list(self.cpu_readings)) if self.cpu_readings else 0
        ram_avg = np.mean(list(self.ram_readings)) if self.ram_readings else 0
        gpu_avg = np.mean(list(self.gpu_readings)) if self.gpu_readings else 0
        
        return {
            'cpu_avg': cpu_avg,
            'ram_avg': ram_avg,
            'gpu_avg': gpu_avg,
            'cpu_max': max(self.cpu_readings) if self.cpu_readings else 0,
            'ram_max': max(self.ram_readings) if self.ram_readings else 0,
            'gpu_max': max(self.gpu_readings) if self.gpu_readings else 0,
            'samples': len(self.cpu_readings)
        }

def get_test_images(num_images=4000):
    """Get test images, create more if needed"""
    test_images_path = Path('/mnt/34B471F7B471BBC4/CSO_project/datasets/test_dataset/test2017')
    
    if not test_images_path.exists():
        print(f"❌ Test path not found: {test_images_path}")
        return None
    
    all_images = list(test_images_path.glob('*.jpg'))
    
    if len(all_images) < num_images:
        print(f"⚠️  Only found {len(all_images)} images, using all available")
        return all_images
    else:
        return all_images[:num_images]

def benchmark_inference(device='cuda', num_images=4000):
    """Benchmark inference speed with system monitoring for 4k images"""
    
    model = YOLO('/mnt/34B471F7B471BBC4/CSO_project/runs/detect/train7/weights/best.pt')
    
    # Get test images
    all_images = get_test_images(num_images)
    
    if not all_images:
        print("❌ No test images found!")
        return None
    
    actual_count = len(all_images)
    print(f"🧪 Benchmarking on {device.upper()} with {actual_count} images...")
    
    # Initialize system monitor
    monitor = SystemMonitor()
    
    # Warm-up run (5 images)
    print("   🔥 Warm-up run...")
    for img_path in all_images[:5]:
        _ = model(img_path, device=device, verbose=False)
    
    # Benchmark inference with system monitoring
    times = []
    
    # Start monitoring in a separate thread (longer duration)
    estimated_duration = actual_count * 0.1  # Conservative estimate
    monitor_thread = threading.Thread(target=monitor.monitor_continuously, args=(estimated_duration * 2,))
    monitor_thread.start()
    
    start_total = time.time()
    
    for i, img_path in enumerate(all_images):
        start_time = time.time()
        
        results = model(img_path, device=device, verbose=False)
        
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        times.append(inference_time)
        
        # Progress reporting
        if i % 100 == 0:
            elapsed = time.time() - start_total
            images_remaining = actual_count - i
            eta = (elapsed / (i + 1)) * images_remaining if i > 0 else 0
            avg_time = np.mean(times) if times else 0
            current_fps = 1000 / avg_time if avg_time > 0 else 0
            
            print(f"   📊 Progress: {i}/{actual_count} | "
                  f"Avg: {avg_time:.1f}ms | FPS: {current_fps:.1f} | "
                  f"ETA: {eta/60:.1f} min")
    
    total_time = time.time() - start_total
    
    # Stop monitoring
    monitor.monitoring = False
    monitor_thread.join()
    
    # Get system usage statistics
    system_stats = monitor.get_average_usage()
    
    stats = {
        'device': device,
        'total_images': actual_count,
        'avg_time_ms': np.mean(times),
        'std_time_ms': np.std(times),
        'min_time_ms': np.min(times),
        'max_time_ms': np.max(times),
        'total_time_s': total_time,
        'fps': actual_count / total_time,
        'all_times': times,
        'system_usage': system_stats
    }
    
    print(f"✅ {device.upper()} Benchmark Complete!")
    print(f"   ⏱️  Average time: {stats['avg_time_ms']:.2f} ms per image")
    print(f"   🎯 FPS: {stats['fps']:.2f}")
    print(f"   📅 Total time: {stats['total_time_s']/60:.2f} minutes")
    print(f"   💻 CPU Usage: {system_stats['cpu_avg']:.1f}%")
    print(f"   🧠 RAM Usage: {system_stats['ram_avg']:.1f}%")
    if device == 'cuda':
        print(f"   🎮 GPU Usage: {system_stats['gpu_avg']:.1f}%")
        print(f"   📊 Samples collected: {system_stats['samples']}")
    
    return stats

def run_4k_comparison():
    """Run comparison between CPU and GPU with 4k images"""
    
    print("🚀 STARTING 4K IMAGE CPU vs GPU PERFORMANCE COMPARISON")
    print("=" * 70)
    print("⚠️  This will take a while! Estimated times:")
    print("   - GPU: 5-15 minutes")
    print("   - CPU: 45-90 minutes")
    print("=" * 70)
    
    results = {}
    
    # Benchmark GPU first (faster)
    print("\n🎮 Testing GPU Performance with 4000 images...")
    gpu_stats = benchmark_inference(device='cuda', num_images=4000)
    if gpu_stats:
        results['GPU'] = gpu_stats
    
    # Ask before starting CPU test (since it's much longer)
    print(f"\n⏰ GPU test completed in {gpu_stats['total_time_s']/60:.1f} minutes")
    response = input("Start CPU test? This may take 45-90 minutes. (y/n): ")
    
    if response.lower() == 'y':
        print("\n💻 Testing CPU Performance with 4000 images...")
        cpu_stats = benchmark_inference(device='cpu', num_images=4000)
        if cpu_stats:
            results['CPU'] = cpu_stats
    else:
        print("Skipping CPU test.")
    
    return results

def plot_4k_comparison(results):
    """Plot comprehensive performance comparison for 4k images"""
    
    if not results or len(results) < 1:
        print("❌ Not enough data to plot")
        return
    
    devices = list(results.keys())
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'4K Image Performance Comparison ({results[devices[0]]["total_images"]} images)', 
                 fontsize=16, fontweight='bold')
    
    colors = {'GPU': '#4ECDC4', 'CPU': '#FF6B6B'}
    
    # Plot 1: Average Inference Time
    avg_times = [results[device]['avg_time_ms'] for device in devices]
    bars1 = ax1.bar(devices, avg_times, color=[colors[d] for d in devices], alpha=0.8)
    ax1.set_title('Average Inference Time per Image', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time (milliseconds)')
    ax1.grid(True, alpha=0.3)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f} ms', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: FPS Comparison
    fps_values = [results[device]['fps'] for device in devices]
    bars2 = ax2.bar(devices, fps_values, color=[colors[d] for d in devices], alpha=0.8)
    ax2.set_title('Frames Per Second (FPS)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('FPS')
    ax2.grid(True, alpha=0.3)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Total Processing Time
    total_times_min = [results[device]['total_time_s'] / 60 for device in devices]
    bars3 = ax3.bar(devices, total_times_min, color=[colors[d] for d in devices], alpha=0.8)
    ax3.set_title('Total Processing Time', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Time (minutes)')
    ax3.grid(True, alpha=0.3)
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f} min', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Speedup Factor (if both devices tested)
    if len(devices) == 2:
        speedup = results['CPU']['avg_time_ms'] / results['GPU']['avg_time_ms']
        bars4 = ax4.bar(['Speedup'], [speedup], color='#FFD166', alpha=0.8)
        ax4.set_title('GPU Speedup Factor', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Times Faster')
        ax4.grid(True, alpha=0.3)
        ax4.text(0, speedup/2, f'{speedup:.1f}x', ha='center', va='center', 
                fontsize=24, fontweight='bold', color='white')
    else:
        # Show system usage instead
        device = devices[0]
        sys_usage = results[device]['system_usage']
        usage_data = [sys_usage['cpu_avg'], sys_usage['ram_avg']]
        if device == 'GPU':
            usage_data.append(sys_usage['gpu_avg'])
        
        labels = ['CPU', 'RAM', 'GPU'][:len(usage_data)]
        bars4 = ax4.bar(labels, usage_data, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
        ax4.set_title(f'{device} System Usage', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Usage (%)')
        ax4.set_ylim(0, 100)
        ax4.grid(True, alpha=0.3)
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('4k_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def save_detailed_results(results):
    """Save detailed results to CSV"""
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"performance_4k_{timestamp}.csv"
    
    data = []
    for device, stats in results.items():
        sys_usage = stats['system_usage']
        data.append({
            'device': device,
            'total_images': stats['total_images'],
            'avg_time_ms': stats['avg_time_ms'],
            'std_time_ms': stats['std_time_ms'],
            'min_time_ms': stats['min_time_ms'],
            'max_time_ms': stats['max_time_ms'],
            'total_time_s': stats['total_time_s'],
            'total_time_min': stats['total_time_s'] / 60,
            'fps': stats['fps'],
            'cpu_usage_avg': sys_usage['cpu_avg'],
            'ram_usage_avg': sys_usage['ram_avg'],
            'gpu_usage_avg': sys_usage['gpu_avg'] if device == 'GPU' else 0,
            'samples_collected': sys_usage['samples']
        })
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"💾 Detailed results saved to: {filename}")
    
    # Also save raw timing data
    for device, stats in results.items():
        timing_filename = f"timing_data_{device}_{timestamp}.csv"
        timing_df = pd.DataFrame({'inference_time_ms': stats['all_times']})
        timing_df.to_csv(timing_filename, index=False)
        print(f"💾 Raw timing data saved to: {timing_filename}")

if __name__ == "__main__":
    print("🎯 4K IMAGE PERFORMANCE COMPARISON")
    print("Testing with 4000 images for accurate results...")
    
    results = run_4k_comparison()
    
    if results:
        # Plot results
        plot_4k_comparison(results)
        
        # Save detailed results
        save_detailed_results(results)
        
        print(f"\n🎉 4K Comparison Completed!")
        for device, stats in results.items():
            print(f"   {device}: {stats['total_time_s']/60:.1f} minutes, {stats['fps']:.1f} FPS")
    else:
        print("❌ No results collected")