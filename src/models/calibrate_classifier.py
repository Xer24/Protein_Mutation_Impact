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
    Fits base_model on a subset of training data, then calibrates probabilities
    using a held-out calibration split from the training set.

    IMPORTANT: test set is never used for calibration fitting.
    """
    strat = y_train if stratify else None
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train,
        y_train,
        test_size=test_size_for_cal,
        random_state=random_state,
        stratify=strat,
    )

    # Fit base model
    base_model.fit(X_fit, y_fit)

    # Calibrate on calibration split
    calibrator = CalibratedClassifierCV(
        estimator=base_model, 
        method=method, 
        cv=2
    )
    calibrator.fit(X_cal, y_cal)

    # Predict probabilities on test
    p_uncal = base_model.predict_proba(X_test)[:, 1]
    p_cal = calibrator.predict_proba(X_test)[:, 1]

    # Discrimination metrics
    auroc_uncal = roc_auc_score(y_test, p_uncal)
    auroc_cal = roc_auc_score(y_test, p_cal)
    auprc_uncal = average_precision_score(y_test, p_uncal)
    auprc_cal = average_precision_score(y_test, p_cal)

    # Calibration metric (lower is better)
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
    }

    return CalibrationResult(
        calibrated_model=calibrator,
        metrics=metrics,
        cal_curve_uncal=(mean_u, frac_u),
        cal_curve_cal=(mean_c, frac_c),
        probs_uncal=p_uncal,
        probs_cal=p_cal,
    )