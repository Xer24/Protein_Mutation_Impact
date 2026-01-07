from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.model_selection import train_test_split


@dataclass
class CalibrationResult:
    calibrated_model: Any
    metrics: Dict[str, float]
    cal_curve_uncal: Tuple[np.ndarray, np.ndarray]  # (mean_pred, frac_pos)
    cal_curve_cal: Tuple[np.ndarray, np.ndarray]    # (mean_pred, frac_pos)
    probs_uncal: np.ndarray
    probs_cal: np.ndarray


def calibrate_classifier(
    base_model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    method: str = "sigmoid",
    test_size_for_cal: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
    n_bins: int = 10,
) -> CalibrationResult:
    """
    Calibrates probabilities from an ALREADY-FITTED base_model using a held-out 
    calibration split from the training set.

    IMPORTANT: 
    - base_model should ALREADY be fitted before calling this function
    - test set is never used for calibration fitting
    - Only the calibration layer is fit, not the base model
    """
    
    # ⚠️ CRITICAL FIX: Check if model is already fitted
    try:
        # Try to predict - if it fails, model isn't fitted
        _ = base_model.predict(X_train[:1])
    except Exception as e:
        raise ValueError(
            "base_model must be already fitted before calling calibrate_classifier! "
            f"Got error: {e}"
        )
    
    # Split training data for calibration
    strat = y_train if stratify else None
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train,
        y_train,
        test_size=test_size_for_cal,
        random_state=random_state,
        stratify=strat,
    )

    # ⚠️ REMOVED: Do NOT re-fit the base model!
    # base_model.fit(X_fit, y_fit)  # ❌ OLD CODE - THIS WAS THE BUG

    # Get uncalibrated predictions on calibration set
    # We need these to train the calibration function
    print(f"Calibrating on {len(X_cal)} samples...")
    
    # ✅ NEW: Use CalibratedClassifierCV with cv='prefit'
    # This tells sklearn the model is already fitted
    calibrator = CalibratedClassifierCV(
        estimator=base_model, 
        method=method, 
        cv='prefit'  # ⚠️ CRITICAL: Use 'prefit' to avoid re-fitting!
    )
    
    # Fit ONLY the calibration layer on calibration data
    calibrator.fit(X_cal, y_cal)

    # Predict probabilities on test set
    p_uncal = base_model.predict_proba(X_test)[:, 1]
    p_cal = calibrator.predict_proba(X_test)[:, 1]

    # Discrimination metrics (should be identical or very close)
    auroc_uncal = roc_auc_score(y_test, p_uncal)
    auroc_cal = roc_auc_score(y_test, p_cal)
    auprc_uncal = average_precision_score(y_test, p_uncal)
    auprc_cal = average_precision_score(y_test, p_cal)

    # Calibration metric (lower is better - this should improve)
    brier_uncal = brier_score_loss(y_test, p_uncal)
    brier_cal = brier_score_loss(y_test, p_cal)

    # Reliability curves
    frac_u, mean_u = calibration_curve(y_test, p_uncal, n_bins=n_bins, strategy="uniform")
    frac_c, mean_c = calibration_curve(y_test, p_cal, n_bins=n_bins, strategy="uniform")

    metrics = {
        "auroc_uncal": float(auroc_uncal),
        "auroc_cal": float(auroc_cal),
        "auprc_uncal": float(auprc_uncal),
        "auprc_cal": float(auprc_cal),
        "brier_uncal": float(brier_uncal),
        "brier_cal": float(brier_cal),
        "delta_auroc": float(auroc_cal - auroc_uncal),  # Should be ~0
        "delta_brier": float(brier_cal - brier_uncal),  # Should be negative (improvement)
    }

    return CalibrationResult(
        calibrated_model=calibrator,
        metrics=metrics,
        cal_curve_uncal=(mean_u, frac_u),
        cal_curve_cal=(mean_c, frac_c),
        probs_uncal=p_uncal,
        probs_cal=p_cal,
    )