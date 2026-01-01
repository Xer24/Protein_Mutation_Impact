import joblib
import numpy as np


clf = joblib.load("artifacts/mlp_gfp.joblib")
X = np.load("data/processed/gfp_delta_X.npy")
p = clf.predict_proba(X)[:, 1]

print("prob min/mean/max:", p.min(), p.mean(), p.max())
print("fraction > 0.9:", (p > 0.9).mean())
print("fraction < 0.1:", (p < 0.1).mean())



