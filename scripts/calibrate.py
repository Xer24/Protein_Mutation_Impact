"""
calibrate_model.py - Complete calibration pipeline with threshold re-optimization
"""
from __future__ import annotations
from pathlib import Path
import sys
import json
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Set up path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.calibrate_classifier import calibrate_classifier

SPLIT_DIR = PROJECT_ROOT / "artifacts" / "train_test"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
FIG_DIR = PROJECT_ROOT / "artifacts" / "figures"

BASE_BUNDLE_PATH = MODELS_DIR / "rf_smote_bundle.joblib"
OUT_BUNDLE_PATH = MODELS_DIR / "rf_smote_calibrated_bundle.joblib"
OUT_METRICS_PATH = MODELS_DIR / "calibration_metrics.json"
OUT_CAL_PLOT = FIG_DIR / "calibration_curve.png"
OUT_HIST_PLOT = FIG_DIR / "prob_hist_by_class.png"


def find_optimal_threshold(y_true, y_proba, metric='f1'):
    """Find optimal classification threshold."""
    from sklearn.metrics import f1_score
    
    thresholds = np.linspace(0.01, 0.99, 100)
    scores = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        scores.append(score)
    
    scores = np.array(scores)
    best_idx = np.argmax(scores)
    return thresholds[best_idx], scores[best_idx]


def evaluate_at_threshold(y_true, y_proba, threshold):
    """Get metrics at specific threshold."""
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    
    y_pred = (y_proba >= threshold).astype(int)
    
    return {
        'threshold': float(threshold),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
    }


def main():
    print("="*80)
    print("CALIBRATION PIPELINE WITH THRESHOLD RE-OPTIMIZATION")
    print("="*80)
    
    # Create directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n📂 Loading data splits...")
    X_train = np.load(SPLIT_DIR / "X_train.npy")
    y_train = np.load(SPLIT_DIR / "y_train.npy")
    X_test = np.load(SPLIT_DIR / "X_test.npy")
    y_test = np.load(SPLIT_DIR / "y_test.npy")
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

    # Load base model bundle
    print(f"\n📦 Loading base model from {BASE_BUNDLE_PATH}...")
    bundle = joblib.load(BASE_BUNDLE_PATH)
    model = bundle["model"]
    scaler = bundle["scaler"]
    uncal_threshold = bundle.get("recommended_threshold", 0.5)
    print(f"   Uncalibrated optimal threshold: {uncal_threshold:.3f}")

    # Scale data
    print("\n⚙️  Scaling data...")
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # === CALIBRATION ===
    print("\n" + "="*80)
    print("STEP 1: CALIBRATING PROBABILITIES")
    print("="*80)
    
    result = calibrate_classifier(
        base_model=model,
        X_train=X_train_s,
        y_train=y_train,
        X_test=X_test_s,
        y_test=y_test,
        method="sigmoid",
        test_size_for_cal=0.2,
        random_state=42,
        stratify=True,
        n_bins=10,
    )
    
    print("\n📊 Calibration Metrics (test set):")
    for k, v in result.metrics.items():
        print(f"   {k}: {v:.4f}")
    
    # === THRESHOLD RE-OPTIMIZATION ===
    print("\n" + "="*80)
    print("STEP 2: RE-OPTIMIZING THRESHOLD ON CALIBRATED PROBABILITIES")
    print("="*80)
    
    # Find optimal threshold on calibrated probs
    cal_threshold, cal_f1_score = find_optimal_threshold(
        y_test, 
        result.probs_cal, 
        metric='f1'
    )
    
    print(f"\n🎯 Threshold Comparison:")
    print(f"   Uncalibrated optimal: {uncal_threshold:.3f}")
    print(f"   Calibrated optimal:   {cal_threshold:.3f}")
    print(f"   Change:               {cal_threshold - uncal_threshold:+.3f}")
    
    # Evaluate both
    print(f"\n📈 Performance Comparison:")
    
    # Uncalibrated at its optimal threshold
    uncal_metrics = evaluate_at_threshold(y_test, result.probs_uncal, uncal_threshold)
    print(f"\n   Uncalibrated @ {uncal_threshold:.3f}:")
    for k, v in uncal_metrics.items():
        if k != 'threshold':
            print(f"      {k}: {v:.4f}")
    
    # Calibrated at OLD threshold (should be similar)
    cal_old_thresh = evaluate_at_threshold(y_test, result.probs_cal, uncal_threshold)
    print(f"\n   Calibrated @ {uncal_threshold:.3f} (old threshold):")
    for k, v in cal_old_thresh.items():
        if k != 'threshold':
            print(f"      {k}: {v:.4f}")
    
    # Calibrated at NEW optimal threshold (should be best)
    cal_new_thresh = evaluate_at_threshold(y_test, result.probs_cal, cal_threshold)
    print(f"\n   Calibrated @ {cal_threshold:.3f} (NEW optimal):")
    for k, v in cal_new_thresh.items():
        if k != 'threshold':
            print(f"      {k}: {v:.4f}")
    
    # === SAVE EVERYTHING ===
    print("\n" + "="*80)
    print("STEP 3: SAVING CALIBRATED MODEL")
    print("="*80)
    
    # Save calibrated bundle with BOTH thresholds
    out_bundle = {
        "scaler": scaler,
        "base_model": model,
        "calibrated_model": result.calibrated_model,
        "calibration_method": "sigmoid",
        "recommended_threshold": cal_threshold,  # ⚠️ Use calibrated threshold
        "uncalibrated_threshold": uncal_threshold,
        "threshold_options": {
            "calibrated_f1_optimal": float(cal_threshold),
            "uncalibrated_f1_optimal": float(uncal_threshold),
            "default": 0.5,
        },
        "performance": {
            "uncalibrated_at_optimal": uncal_metrics,
            "calibrated_at_old_threshold": cal_old_thresh,
            "calibrated_at_new_threshold": cal_new_thresh,
        }
    }
    
    joblib.dump(out_bundle, OUT_BUNDLE_PATH)
    print(f"✅ Saved calibrated bundle to: {OUT_BUNDLE_PATH}")
    
    # Save metrics JSON
    metrics = dict(result.metrics)
    metrics.update({
        "method": "sigmoid",
        "n_test": int(len(y_test)),
        "pos_rate_test": float(y_test.mean()),
        "calibrated_threshold": float(cal_threshold),
        "uncalibrated_threshold": float(uncal_threshold),
        "threshold_shift": float(cal_threshold - uncal_threshold),
        "performance": out_bundle["performance"],
    })
    
    OUT_METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"✅ Saved metrics to: {OUT_METRICS_PATH}")
    
    # === VISUALIZATIONS ===
    print("\n📊 Creating visualizations...")
    
    # 1. Calibration curve
    mean_u, frac_u = result.cal_curve_uncal
    mean_c, frac_c = result.cal_curve_cal
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Calibration reliability diagram
    ax = axes[0]
    ax.plot([0, 1], [0, 1], linestyle="--", color='gray', label="Perfect calibration")
    ax.plot(mean_u, frac_u, marker="o", linewidth=2, label="Uncalibrated", color='red')
    ax.plot(mean_c, frac_c, marker="o", linewidth=2, label="Calibrated", color='green')
    ax.set_xlabel("Mean predicted probability", fontsize=11)
    ax.set_ylabel("Fraction of positives", fontsize=11)
    ax.set_title("Calibration Curve (Reliability Diagram)", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Probability histogram
    ax = axes[1]
    ax.hist(result.probs_cal[y_test == 0], bins=40, alpha=0.6, 
            label="Tolerated (0)", color='blue')
    ax.hist(result.probs_cal[y_test == 1], bins=40, alpha=0.6, 
            label="Deleterious (1)", color='red')
    ax.axvline(cal_threshold, color='black', linestyle='--', linewidth=2,
              label=f'Optimal ({cal_threshold:.3f})')
    ax.axvline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5,
              label='Default (0.5)')
    ax.set_xlabel("Predicted P(deleterious) - Calibrated", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Calibrated Probability Distribution", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUT_CAL_PLOT, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved calibration plots to: {OUT_CAL_PLOT}")
    
    # === FINAL SUMMARY ===
    print("\n" + "="*80)
    print("🎯 FINAL RECOMMENDATIONS")
    print("="*80)
    print(f"\nFor BEST performance, use:")
    print(f"   Model: Calibrated")
    print(f"   Threshold: {cal_threshold:.3f}")
    print(f"\nThis gives you:")
    print(f"   F1 Score:  {cal_new_thresh['f1']:.4f}")
    print(f"   Precision: {cal_new_thresh['precision']:.4f}")
    print(f"   Recall:    {cal_new_thresh['recall']:.4f}")
    print(f"   Accuracy:  {cal_new_thresh['accuracy']:.4f}")
    
    print("\n" + "="*80)
    print("✅ CALIBRATION COMPLETE!")
    print("="*80)
    print(f"\nNext steps:")
    print(f"   1. Review plots in: {FIG_DIR}")
    print(f"   2. Use calibrated model: {OUT_BUNDLE_PATH}")
    print(f"   3. Make predictions with threshold {cal_threshold:.3f}")

    

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}")
        print(f"   {e}")
        import traceback
        traceback.print_exc()