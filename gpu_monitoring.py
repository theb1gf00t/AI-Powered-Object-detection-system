# gpu_monitor_simple.py
import torch
import time
import psutil
import subprocess
from datetime import datetime

def get_gpu_info_nvidia_smi():
    """Get GPU info using nvidia-smi command"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            gpu_data = result.stdout.strip().split(', ')
            if len(gpu_data) >= 4:
                return {
                    'utilization': float(gpu_data[0]),
                    'memory_used': float(gpu_data[1]),
                    'memory_total': float(gpu_data[2]),
                    'temperature': float(gpu_data[3])
                }
    except:
        pass
    return None

def get_pytorch_gpu_info():
    """Get PyTorch GPU information"""
    if torch.cuda.is_available():
        return {
            'allocated_gb': torch.cuda.memory_allocated(0) / 1e9,
            'reserved_gb': torch.cuda.memory_reserved(0) / 1e9,
            'total_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
        }
    return None

def get_system_info():
    """Get system information"""
    return {
        'cpu_percent': psutil.cpu_percent(),
        'ram_percent': psutil.virtual_memory().percent,
        'ram_used_gb': psutil.virtual_memory().used / 1e9,
        'ram_total_gb': psutil.virtual_memory().total / 1e9,
    }

def monitor_training():
    """Main monitoring function"""
    print("🖥️  GPU & SYSTEM MONITOR")
    print("=" * 70)
    print("Monitoring RTX 3050 6GB during YOLOv8 training...")
    print("Expected: GPU ~80-95%, Memory ~4-5GB, High CPU during data loading")
    print("=" * 70)
    print("Time     | GPU%  | GPU Mem  | GPU Temp | CPU%  | RAM%  | PyTorch Mem")
    print("=" * 70)
    
    start_time = time.time()
    
    try:
        while True:
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            # Get GPU info from nvidia-smi
            gpu_info = get_gpu_info_nvidia_smi()
            
            # Get PyTorch GPU info
            torch_info = get_pytorch_gpu_info()
            
            # Get system info
            system_info = get_system_info()
            
            # Format display
            if gpu_info:
                gpu_util = f"{gpu_info['utilization']:4.0f}%"
                gpu_mem = f"{gpu_info['memory_used']:3.0f}/{gpu_info['memory_total']:2.0f}MB"
                gpu_temp = f"{gpu_info['temperature']:2.0f}°C"
            else:
                gpu_util = "  N/A"
                gpu_mem = "     N/A"
                gpu_temp = " N/A"
            
            cpu_usage = f"{system_info['cpu_percent']:4.0f}%"
            ram_usage = f"{system_info['ram_percent']:4.0f}%"
            
            if torch_info:
                torch_mem = f"{torch_info['allocated_gb']:5.2f}GB"
            else:
                torch_mem = "   N/A"
            
            print(f"{timestamp} | {gpu_util} | {gpu_mem} | {gpu_temp} | {cpu_usage} | {ram_usage} | {torch_mem}")
            
            time.sleep(5)  # Update every 5 seconds
            
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n🛑 Monitoring stopped after {elapsed/60:.1f} minutes")

def quick_status():
    """Quick one-time status check"""
    print("⚡ QUICK GPU STATUS CHECK")
    print("=" * 50)
    
    # PyTorch info
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Current memory
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"Current - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    
    # nvidia-smi info
    gpu_info = get_gpu_info_nvidia_smi()
    if gpu_info:
        print(f"GPU Utilization: {gpu_info['utilization']}%")
        print(f"GPU Memory: {gpu_info['memory_used']}/{gpu_info['memory_total']} MB")
        print(f"GPU Temperature: {gpu_info['temperature']}°C")
    
    # System info
    print(f"CPU Cores: {psutil.cpu_count()}")
    print(f"RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
    print("=" * 50)

if __name__ == "__main__":
    print("Choose monitoring mode:")
    print("1. Quick status check")
    print("2. Continuous monitoring (run during training)")
    
    choice = input("Enter 1 or 2 (default 2): ").strip()
    
    if choice == "1":
        quick_status()
    else:
        monitor_training()