import torch
from constants import USE_CUDA, GPU_BATCH_SIZE, CPU_BATCH_SIZE, DQN_BATCH_SIZE

# Set global device variable
device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")

def check_gpu_availability():
    """
    Check if GPU/CUDA is available for PyTorch and configure settings accordingly.
    Returns:
        tuple: (bool for GPU availability, appropriate batch size)
    """
    cuda_available = torch.cuda.is_available()
    
    # Print GPU information
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"Available GPUs: {device_count}")
        
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_capability = torch.cuda.get_device_capability(i)
            print(f"  [{i}] {gpu_name} (Compute Capability: {gpu_capability[0]}.{gpu_capability[1]})")
        
        if USE_CUDA:
            print("CUDA is available and enabled. Using GPU for training.")
            batch_size = GPU_BATCH_SIZE
            
            # Set PyTorch to use GPU (already done by default device selection)
            device = torch.device("cuda")
            print(f"Using device: {device}")
            
            # Set memory management options for better performance
            try:
                torch.backends.cudnn.benchmark = True
                print("cuDNN benchmark mode enabled")
            except:
                print("Could not set cuDNN benchmark mode")
                
            return True, batch_size
        else:
            print("CUDA is available but disabled in settings. Using CPU.")
            torch.cuda.is_available = lambda: False  # Override CUDA availability check
            return False, CPU_BATCH_SIZE
    else:
        print("No GPU found. Using CPU for training.")
        return False, CPU_BATCH_SIZE

def get_optimal_batch_size():
    """
    Returns the optimal batch size based on available hardware.
    """
    is_gpu, batch_size = check_gpu_availability()
    return batch_size

# If DQN_BATCH_SIZE is not defined in constants.py, use this function to set it
def get_batch_size():
    """
    Get the appropriate batch size based on hardware and settings.
    Returns:
        int: Batch size to use for training
    """
    if 'DQN_BATCH_SIZE' in globals():
        return DQN_BATCH_SIZE
    else:
        return get_optimal_batch_size()