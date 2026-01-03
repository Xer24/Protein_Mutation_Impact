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
) -> Dict[str, np.ndarray]:
    """
    Predict deleterious probability for each mutation.
    """
    # 1) load classifier
    clf = joblib.load(model_path)

    # 2) embed + compute deltas
    feats = embed_wt_and_mutants(
        wt_seq=wt_seq,
        mutations=mutations,
        strict=strict,
        cfg=cfg,
    )

    X = feats["delta_embeddings"]  # (N, D)

    # 3) predict
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)[:, 1]
    else:
        # fallback (e.g. SVM without probas)
        probs = clf.decision_function(X)

    return {
        "mutations": feats["mutations"],
        "mutant_seqs": feats["mutant_seqs"],
        "delta_l2": feats["delta_l2"],
        "prob_deleterious": probs,
    }
