from __future__ import annotations

from typing import Dict, Sequence, Optional
import numpy as np
import joblib

from src.mutations.features import embed_wt_and_mutants
from src.features.esm2_embedding import ESM2Config


def predict_mutation_effects(
    wt_seq: str,
    mutations: Sequence[str],
    model_path: str,
    *,
    strict: bool = True,
    cfg: Optional[ESM2Config] = None,
    use_calibrated: bool = True,  # NEW
    use_optimal_threshold: bool = True,  # NEW
    custom_threshold: Optional[float] = None,  # NEW
) -> Dict[str, np.ndarray]:
    """
    Predict deleterious probability for each mutation.
    
    Args:
        wt_seq: Wild-type sequence
        mutations: List of mutations (e.g., ['A23V', 'S65T'])
        model_path: Path to model bundle (.joblib)
        strict: Strict mutation parsing
        cfg: ESM2 config
        use_calibrated: Use calibrated model if available
        use_optimal_threshold: Use saved optimal threshold for binary predictions
        custom_threshold: Override with custom threshold (0-1)
    
    Returns:
        Dictionary with mutations, sequences, probabilities, and predictions
    """
    # Load model bundle
    bundle = joblib.load(model_path)
    
    # Extract components
    scaler = bundle["scaler"]
    
    # Choose calibrated or base model
    if use_calibrated and "calibrated_model" in bundle:
        model = bundle["calibrated_model"]
        print("ℹ️  Using calibrated model")
    else:
        model = bundle["model"]
        print("ℹ️  Using base model (uncalibrated)")
    
    # Determine threshold
    if custom_threshold is not None:
        threshold = custom_threshold
        print(f"ℹ️  Using custom threshold: {threshold:.3f}")
    elif use_optimal_threshold and "recommended_threshold" in bundle:
        threshold = bundle["recommended_threshold"]
        print(f"ℹ️  Using optimal threshold: {threshold:.3f}")
    else:
        threshold = 0.5
        print(f"ℹ️  Using default threshold: {threshold:.3f}")
    
    # Embed and compute deltas
    feats = embed_wt_and_mutants(
        wt_seq=wt_seq,
        mutations=mutations,
        strict=strict,
        cfg=cfg,
    )
    
    X = feats["delta_embeddings"]  # (N, D)
    
    # Scale features (CRITICAL - must match training!)
    X_scaled = scaler.transform(X)
    
    # Predict probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)[:, 1]
    else:
        # Fallback for models without probas
        probs = model.decision_function(X_scaled)
    
    # Binary predictions using threshold
    predictions = (probs >= threshold).astype(int)
    
    # Prepare output
    return {
        "mutations": feats["mutations"],
        "mutant_seqs": feats["mutant_seqs"],
        "delta_l2": feats["delta_l2"],
        "prob_deleterious": probs,
        "predicted_deleterious": predictions,  # NEW: binary predictions
        "threshold_used": threshold,  # NEW: for transparency
        "model_type": "calibrated" if (use_calibrated and "calibrated_model" in bundle) else "uncalibrated",
    }