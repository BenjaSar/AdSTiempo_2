import torch
import sys

print("=" * 70)
print("CUDA SETUP VERIFICATION")
print("=" * 70)

# Python version
print(f"\n✓ Python version: {sys.version.split()[0]}")

# PyTorch version
print(f"✓ PyTorch version: {torch.__version__}")

# CUDA availability
cuda_available = torch.cuda.is_available()
print(f"✓ CUDA available: {cuda_available}")

if cuda_available:
    # CUDA version
    print(f"✓ CUDA version: {torch.version.cuda}")
    
    # cuDNN version
    print(f"✓ cuDNN version: {torch.backends.cudnn.version()}")
    
    # Number of GPUs
    n_gpus = torch.cuda.device_count()
    print(f"✓ Number of GPUs: {n_gpus}")
    
    # GPU details
    for i in range(n_gpus):
        print(f"\n  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        print(f"    Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")
    
    # Test computation
    print("\n" + "=" * 70)
    print("RUNNING GPU TEST")
    print("=" * 70)
    
    device = torch.device('cuda:0')
    
    # Create tensors
    x = torch.randn(1000, 1000).to(device)
    y = torch.randn(1000, 1000).to(device)
    
    # Perform computation
    import time
    start = time.time()
    z = torch.matmul(x, y)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    
    print(f"✓ GPU computation successful!")
    print(f"✓ Time for 1000x1000 matrix multiplication: {gpu_time*1000:.2f} ms")
    
    # CPU comparison
    x_cpu = x.cpu()
    y_cpu = y.cpu()
    start = time.time()
    z_cpu = torch.matmul(x_cpu, y_cpu)
    cpu_time = time.time() - start
    
    print(f"✓ CPU computation time: {cpu_time*1000:.2f} ms")
    print(f"✓ Speedup: {cpu_time/gpu_time:.2f}x faster on GPU")
    
else:
    print("\n⚠️  CUDA not available!")
    print("    Running on CPU only")
    print("\n    Possible reasons:")
    print("    1. No NVIDIA GPU detected")
    print("    2. CUDA drivers not installed")
    print("    3. PyTorch installed without CUDA support")
    print("\n    To install PyTorch with CUDA:")
    print("    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70 + "\n")