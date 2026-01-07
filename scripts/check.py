"""
Check model performance at the optimal threshold from the plot
"""
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# FIX: Go up one level since script is in scripts/ folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # ← Added .parent

# Load test data
X_test = np.load(PROJECT_ROOT / "artifacts" / "train_test" / "X_test.npy")
y_test = np.load(PROJECT_ROOT / "artifacts" / "train_test" / "y_test.npy")

# Load model
model_path = PROJECT_ROOT / "artifacts" / "models" / "rf_smote_bundle.joblib"
bundle = joblib.load(model_path)

scaler = bundle["scaler"]
model = bundle.get("calibrated_model", bundle["model"])

# Scale and predict
X_test_scaled = scaler.transform(X_test)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

print("="*80)
print("THRESHOLD ANALYSIS")
print("="*80)

# Test different thresholds
thresholds = [0.108, 0.2, 0.3, 0.5, 0.7]

for threshold in thresholds:
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n{'='*80}")
    print(f"THRESHOLD = {threshold}")
    print(f"{'='*80}")
    print(f"F1 Score: {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"               Predicted")
    print(f"               0      1")
    print(f"Actual 0    [{cm[0,0]:4d}, {cm[0,1]:4d}]  (Tolerated)")
    print(f"       1    [{cm[1,0]:4d}, {cm[1,1]:4d}]  (Deleterious)")
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate metrics
        if (tn + fp) > 0:
            tolerated_recall = tn / (tn + fp)
            print(f"\nTolerated correctly identified: {tn} / {tn + fp} = {tolerated_recall:.2%}")
        
        if (tp + fn) > 0:
            deleterious_recall = tp / (tp + fn)
            print(f"Deleterious correctly identified: {tp} / {tp + fn} = {deleterious_recall:.2%}")
        
        if (tp + fp) > 0:
            precision = tp / (tp + fp)
            print(f"Precision (when predicting deleterious): {precision:.2%}")

print("\n" + "="*80)
print("PROBABILITY DISTRIBUTION STATS")
print("="*80)
print(f"Min probability: {y_proba.min():.4f}")
print(f"Max probability: {y_proba.max():.4f}")
print(f"Mean probability: {y_proba.mean():.4f}")
print(f"Median probability: {np.median(y_proba):.4f}")

print(f"\nClass 0 (Tolerated) probabilities:")
print(f"  Mean: {y_proba[y_test==0].mean():.4f}")
print(f"  Min:  {y_proba[y_test==0].min():.4f}")
print(f"  Max:  {y_proba[y_test==0].max():.4f}")

print(f"\nClass 1 (Deleterious) probabilities:")
print(f"  Mean: {y_proba[y_test==1].mean():.4f}")
print(f"  Min:  {y_proba[y_test==1].min():.4f}")
print(f"  Max:  {y_proba[y_test==1].max():.4f}")