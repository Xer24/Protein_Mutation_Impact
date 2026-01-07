import torch
import psutil

print("="*60)
print("SYSTEM CHECK")
print("="*60)

# CPU RAM
ram_gb = psutil.virtual_memory().total / (1024**3)
print(f"Total RAM: {ram_gb:.1f} GB")
print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")

# GPU
if torch.cuda.is_available():
    print(f"\n✅ CUDA GPU available: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
else:
    print("\n❌ No CUDA GPU detected - will use CPU")

# Mac GPU (MPS)
if torch.backends.mps.is_available():
    print("\n✅ Mac GPU (MPS) available")
else:
    print("\n❌ Mac GPU not available")