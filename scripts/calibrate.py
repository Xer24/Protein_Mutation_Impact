from __future__ import annotations
from pathlib import Path
import sys

import json
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Set up path BEFORE importing from src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Now import from src
from src.models.calibrate_classifier import calibrate_classifier

SPLIT_DIR = PROJECT_ROOT / "artifacts" / "train_test"
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
FIG_DIR = PROJECT_ROOT / "artifacts" / "figures"

# File paths 
BASE_BUNDLE_PATH = MODELS_DIR / "rf_smote_bundle.joblib"

OUT_BUNDLE_PATH = MODELS_DIR / "rf_smote_calibrated_bundle.joblib"
OUT_METRICS_PATH = MODELS_DIR / "calibration_metrics.json"
OUT_CAL_PLOT = FIG_DIR / "calibration_curve.png"
OUT_HIST_PLOT = FIG_DIR / "prob_hist_by_class.png"


def main():
    print("Starting calibration script...")
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created directories: {MODELS_DIR}, {FIG_DIR}")

    # Load split arrays
    print("Loading data splits...")
    X_train = np.load(SPLIT_DIR / "X_train.npy")
    y_train = np.load(SPLIT_DIR / "y_train.npy")
    X_test = np.load(SPLIT_DIR / "X_test.npy")
    y_test = np.load(SPLIT_DIR / "y_test.npy")
    print(f"Loaded: X_train shape={X_train.shape}, X_test shape={X_test.shape}")

    # Load model bundle (expects {"model": ..., "scaler": ...})
    print(f"Loading model bundle from {BASE_BUNDLE_PATH}...")
    bundle = joblib.load(BASE_BUNDLE_PATH)
    model = bundle["model"]
    scaler = bundle["scaler"]
    print("Model and scaler loaded successfully")

    # Apply the same preprocessing used during training
    print("Scaling data...")
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    print("Data scaled")

    # Calibrate
    print("Starting calibration...")
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
    print("Calibration complete!")

    print("\n=== Calibration metrics (test set) ===")
    for k, v in result.metrics.items():
        print(f"{k}: {v:.4f}")
    print("Note: Brier should go DOWN after calibration (lower is better).")

    # Save calibrated bundle
    print("\nSaving calibrated bundle...")
    out_bundle = {
        "scaler": scaler,
        "base_model": model,
        "calibrated_model": result.calibrated_model,
        "calibration_method": "sigmoid",
    }
    joblib.dump(out_bundle, OUT_BUNDLE_PATH)
    print(f"Saved to {OUT_BUNDLE_PATH}")

    # Save metrics JSON
    print("Saving metrics...")
    metrics = dict(result.metrics)
    metrics.update({
        "method": "sigmoid",
        "n_test": int(len(y_test)),
        "pos_rate_test": float(y_test.mean()),
    })
    OUT_METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Saved to {OUT_METRICS_PATH}")

    # Plot calibration curve (reliability diagram)
    print("Creating calibration curve plot...")
    mean_u, frac_u = result.cal_curve_uncal
    mean_c, frac_c = result.cal_curve_cal

    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
    plt.plot(mean_u, frac_u, marker="o", label="Uncalibrated")
    plt.plot(mean_c, frac_c, marker="o", label="Calibrated (sigmoid)")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration curve (test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_CAL_PLOT, dpi=300)
    plt.close()
    print(f"Saved to {OUT_CAL_PLOT}")

    # Histogram of predicted probabilities by class (using CALIBRATED probs)
    print("Creating probability histogram...")
    p = result.probs_cal
    plt.figure()
    plt.hist(p[y_test == 0], bins=30, alpha=0.6, label="Tolerated (0)")
    plt.hist(p[y_test == 1], bins=30, alpha=0.6, label="Harmful (1)")
    plt.xlabel("Predicted P(harmful) (calibrated)")
    plt.ylabel("Count")
    plt.title("Predicted probability distributions by class (test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_HIST_PLOT, dpi=300)
    plt.close()
    print(f"Saved to {OUT_HIST_PLOT}")

    print("\n=== All artifacts saved successfully! ===")
    print("  ", OUT_BUNDLE_PATH)
    print("  ", OUT_METRICS_PATH)
    print("  ", OUT_CAL_PLOT)
    print("  ", OUT_HIST_PLOT)


if __name__ == "__main__":
    print("Script started!")
    try:
        main()
    except Exception as e:
        print(f"\n!!! ERROR OCCURRED !!!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        import traceback
        traceback.print_exc()