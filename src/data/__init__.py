import numpy as np
import json

# Load your data
y = np.load("data/processed/gfp_y.npy")
X = np.load("data/processed/gfp_delta_X.npy")

print(f"Total samples: {len(y)}")
print(f"Class distribution: {np.bincount(y)}")
print(f"Class 1 (deleterious) rate: {y.mean():.3f}")
print(f"\nX stats:")
print(f"  Shape: {X.shape}")
print(f"  Mean: {X.mean():.6f}")
print(f"  Std: {X.std():.6f}")
print(f"  Min: {X.min():.6f}")
print(f"  Max: {X.max():.6f}")

# Check if you have metrics
try:
    with open("artifacts/metrics.json") as f:
        metrics = json.load(f)
    print(f"\nCurrent metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
except:
    print("\nNo metrics.json found")