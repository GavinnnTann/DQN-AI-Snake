"""
CUDA Compatibility Check Script for Snake Game DQN.

This script checks for CUDA compatibility, installs the appropriate PyTorch version,
and verifies that everything is working correctly for GPU acceleration.
"""

import os
import sys
import subprocess
import platform

def print_header(text):
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)

def check_python_version():
    print_header("Checking Python Version")
    version = sys.version.split()[0]
    print(f"Python version: {version}")
    if tuple(map(int, version.split('.'))) < (3, 6):
        print("WARNING: Python 3.6 or higher is recommended for PyTorch with CUDA.")
    return version

def check_cuda_availability():
    print_header("Checking CUDA Availability")
    
    # Try using nvidia-smi to check CUDA installation
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print(result.stdout)
        cuda_available = True
    except FileNotFoundError:
        print("NVIDIA System Management Interface (nvidia-smi) not found.")
        print("This likely means CUDA is not installed or not in PATH.")
        cuda_available = False
    
    if not cuda_available:
        print("\nCUDA does not appear to be available on this system.")
        print("You can still use the DQN with CPU, but training will be slower.")
        print("To install CUDA, visit: https://developer.nvidia.com/cuda-downloads")
        return None, None
    
    # Try to determine CUDA version
    cuda_version = None
    cuda_capability = None
    
    try:
        for line in result.stdout.split('\n'):
            if 'CUDA Version:' in line:
                cuda_version = line.split('CUDA Version:')[1].strip()
            if 'Compute' in line and 'capability' in line.lower():
                cuda_capability = line.strip()
    except:
        pass
        
    print(f"CUDA Version: {cuda_version or 'Unknown'}")
    print(f"Compute Capability: {cuda_capability or 'Unknown'}")
    
    return cuda_version, cuda_capability

def check_pytorch_installation():
    print_header("Checking PyTorch Installation")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            print(f"PyTorch CUDA version: {cuda_version}")
            print(f"Number of CUDA devices: {torch.cuda.device_count()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
            
            # Show memory information
            props = torch.cuda.get_device_properties(0)
            print(f"GPU Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"Compute capability: {props.major}.{props.minor}")
            
            # Verify with a small tensor operation
            print("\nVerifying CUDA with a simple test...")
            x = torch.rand(10000, 10000).cuda()
            y = torch.rand(10000, 10000).cuda()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            z = x @ y  # Matrix multiplication
            end.record()
            torch.cuda.synchronize()
            print(f"GPU calculation time: {start.elapsed_time(end):.2f} ms")
            
            return True, torch.__version__, cuda_version
        else:
            print("PyTorch is installed but CUDA is not available.")
            print("This might be because:")
            print("  1. You have the CPU-only version of PyTorch")
            print("  2. CUDA is not properly installed or configured")
            print("  3. Your GPU is not CUDA-compatible")
            
            return False, torch.__version__, None
    
    except ImportError:
        print("PyTorch is not installed.")
        return False, None, None

def install_pytorch_with_cuda(cuda_version):
    print_header(f"Installing PyTorch with CUDA {cuda_version}")
    
    # Map CUDA version to PyTorch wheel
    cuda_map = {
        "12.5": "cu121",  # Use cu121 for CUDA 12.5
        "12.4": "cu121",  # Use cu121 for CUDA 12.4
        "12.3": "cu121",  # Use cu121 for CUDA 12.3
        "12.2": "cu121",  # Use cu121 for CUDA 12.2
        "12.1": "cu121",  # Use cu121 for CUDA 12.1
        "12.0": "cu118",  # Use cu118 for CUDA 12.0
        "11.8": "cu118",
        "11.7": "cu117", 
        "11.6": "cu116",
        "11.3": "cu113",
        "11.1": "cu111",
        "11.0": "cu110",
        "10.2": "cu102",
        "10.1": "cu101",
        "10.0": "cu100",
        "9.2": "cu92",
    }
    
    # Try to find the best match for the CUDA version
    best_match = None
    
    # Clean up the cuda version string to extract just the number
    # Handle formats like "12.5     |" or other unusual formats
    cuda_version_clean = ''.join(c for c in cuda_version if c.isdigit() or c == '.')
    # Extract the first part that looks like a version number
    import re
    version_match = re.search(r'\d+\.\d+', cuda_version_clean)
    if version_match:
        cuda_version_num = float(version_match.group(0))
    else:
        # Fallback if regex fails
        try:
            cuda_version_num = float(cuda_version_clean.split()[0])
        except:
            print(f"Warning: Could not parse CUDA version '{cuda_version}' properly.")
            cuda_version_num = 0.0
    
    print(f"Parsed CUDA version: {cuda_version_num}")
    
    for ver, suffix in cuda_map.items():
        if float(ver) <= cuda_version_num:
            best_match = suffix
            break
    
    if not best_match:
        print(f"Could not find a matching PyTorch build for CUDA {cuda_version}")
        print("Available options are:", ", ".join(cuda_map.keys()))
        return False
    
    print(f"Found compatible PyTorch build for CUDA {cuda_version}: {best_match}")
    
    # Ask user if they want to install
    response = input("Do you want to install PyTorch with CUDA support? (y/n): ")
    if response.lower() != 'y':
        print("Installation skipped.")
        return False
    
    # Install PyTorch with CUDA
    # Modern PyTorch uses a different format - install the latest version with CUDA support
    cmd = [
        sys.executable, "-m", "pip", "install", 
        f"torch", "torchvision", "torchaudio",
        f"--index-url", f"https://download.pytorch.org/whl/{best_match}"
    ]
    
    print("\nRunning command:")
    print(" ".join(cmd))
    
    try:
        subprocess.run(cmd, check=True)
        print("\nPyTorch with CUDA has been installed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError installing PyTorch: {e}")
        return False

def main():
    print_header("CUDA Compatibility Check for Snake Game DQN")
    print(f"Operating System: {platform.system()} {platform.release()}")
    
    # Check Python version
    python_version = check_python_version()
    
    # Check CUDA availability
    cuda_version, cuda_capability = check_cuda_availability()
    
    # Check PyTorch installation
    has_pytorch_cuda, pytorch_version, pytorch_cuda = check_pytorch_installation()
    
    # Summary
    print_header("Summary")
    print(f"Python: {python_version}")
    print(f"CUDA: {'Available' if cuda_version else 'Not available'} {cuda_version or ''}")
    print(f"PyTorch: {pytorch_version or 'Not installed'}")
    print(f"PyTorch CUDA: {'Available' if has_pytorch_cuda else 'Not available'} {pytorch_cuda or ''}")
    
    # Recommendations
    print_header("Recommendations")
    
    if cuda_version and not has_pytorch_cuda:
        print("CUDA is available but PyTorch is not using it.")
        install_pytorch_with_cuda(cuda_version)
    elif not cuda_version:
        print("CUDA is not available. The DQN will run on CPU only, which will be slower.")
        print("If you have an NVIDIA GPU, consider installing CUDA from:")
        print("https://developer.nvidia.com/cuda-downloads")
    else:
        print("Your system is properly configured for GPU-accelerated DQN training!")
        print("You can adjust GPU settings in constants.py:")
        print("- USE_CUDA: Enable/disable GPU usage")
        print("- GPU_BATCH_SIZE: Increase for better GPU utilization")
    
    print("\nTo run the game with GPU acceleration:")
    print("1. Make sure USE_CUDA = True in constants.py")
    print("2. Run main.py normally")
    
if __name__ == "__main__":
    main()