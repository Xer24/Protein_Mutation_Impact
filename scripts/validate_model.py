"""
validate_model.py - Comprehensive validation
"""
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
TEST_DIR = PROJECT_ROOT / "artifacts" / "train_test"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "rf_smote_bundle.joblib"


def main():
    # Load test data
    X_test = np.load(TEST_DIR / "X_test.npy")
    y_test = np.load(TEST_DIR / "y_test.npy")
    
    # Load model
    bundle = joblib.load(MODEL_PATH)
    scaler = bundle["scaler"]
    model = bundle.get("calibrated_model", bundle["model"])
    threshold = bundle["recommended_threshold"]
    
    # Scale and predict
    X_test_scaled = scaler.transform(X_test)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Evaluate
    print("="*60)
    print("FINAL MODEL VALIDATION")
    print("="*60)
    print(f"Threshold: {threshold:.3f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Tolerated', 'Deleterious']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Plot probability distribution
    plt.figure(figsize=(10, 5))
    plt.hist(y_proba[y_test==0], bins=50, alpha=0.6, label='Tolerated', color='blue')
    plt.hist(y_proba[y_test==1], bins=50, alpha=0.6, label='Deleterious', color='red')
    plt.axvline(threshold, color='black', linestyle='--', linewidth=2, 
                label=f'Threshold ({threshold:.3f})')
    plt.xlabel('P(deleterious)')
    plt.ylabel('Count')
    plt.title('Final Model - Probability Distribution')
    plt.legend()
    plt.savefig(PROJECT_ROOT / "artifacts" / "figures" / "final_validation.png", dpi=150)
    print("\n✅ Validation plot saved")


if __name__ == "__main__":
    main()