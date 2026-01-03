import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    average_precision_score,
    confusion_matrix
)
from imblearn.over_sampling import SMOTE  # To help with classifier balancing
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "artifacts" / "train_test"
OUT_DIR = PROJECT_ROOT / "artifacts" / "models"

def main():
    print("="*60)
    print("TRAINING WITH SMOTE BALANCING")
    print("="*60)
    
    # Load splits
    print("\nLoading data...")
    X_train = np.load(DATA_DIR / "X_train.npy")
    X_test = np.load(DATA_DIR / "X_test.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Check ORIGINAL class imbalance
    print(f"\n📊 ORIGINAL class distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} ({count/len(y_train)*100:.1f}%)")
    
    # Scale features FIRST
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE to CREATE SYNTHETIC SAMPLES
    print("\n🔄 Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"\n✅ BALANCED class distribution:")
    unique, counts = np.unique(y_train_balanced, return_counts=True)
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} ({count/len(y_train_balanced)*100:.1f}%)")
    
    # Train on BALANCED data
    print("\nTraining Random Forest on BALANCED data...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    clf.fit(X_train_balanced, y_train_balanced)
    
    # Evaluate on ORIGINAL test set
    print("\n" + "="*60)
    print("EVALUATION ON ORIGINAL TEST SET")
    print("="*60)
    
    # Old version 
    """
    y_pred = clf.predict(X_test_scaled)
    y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
    """
    y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]
    threshold = 0.8  # try 0.6, 0.7, 0.8
    y_pred = (y_pred_proba >= threshold).astype(int)


    # Metrics
    auroc = roc_auc_score(y_test, y_pred_proba)
    auprc = average_precision_score(y_test, y_pred_proba)
    
    print(f"\n✅ AUROC: {auroc:.4f}")
    print(f"✅ AUPRC: {auprc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"\nInterpretation:")
    print(f"  True Negatives:  {cm[0,0]} out of {cm[0,0]+cm[0,1]} (tolerated correctly identified)")
    print(f"  False Positives: {cm[0,1]} out of {cm[0,0]+cm[0,1]} (tolerated wrongly called harmful)")
    print(f"  False Negatives: {cm[1,0]} out of {cm[1,0]+cm[1,1]} (harmful wrongly called tolerated)")
    print(f"  True Positives:  {cm[1,1]} out of {cm[1,0]+cm[1,1]} (harmful correctly identified)")
    
    # Calculate per-class accuracy
    class_0_recall = cm[0,0] / (cm[0,0] + cm[0,1])
    class_1_recall = cm[1,1] / (cm[1,0] + cm[1,1])
    print(f"\n📈 Per-class performance:")
    print(f"  Class 0 recall: {class_0_recall:.2%} ({cm[0,0]}/{cm[0,0]+cm[0,1]} tolerated detected)")
    print(f"  Class 1 recall: {class_1_recall:.2%} ({cm[1,1]}/{cm[1,0]+cm[1,1]} harmful detected)")
    
    # Save model
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {"scaler": scaler, "model": clf}
    joblib.dump(bundle, OUT_DIR / "rf_smote_bundle.joblib")

    
    print(f"\n✅ Model saved to {OUT_DIR}")
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()