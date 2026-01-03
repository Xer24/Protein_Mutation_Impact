# src/features/mutation_features.py

from __future__ import annotations

from typing import Any, Dict, Sequence, Optional
import numpy as np

from src.mutations.parse_mutation import apply_mutation
from src.features.esm2_embedding import embed_sequences_esm2, ESM2Config


def embed_wt_and_mutants(
    wt_seq: str,
    mutations: Sequence[str],
    *,
    strict: bool = True,
    cfg: Optional[ESM2Config] = None,
) -> Dict[str, Any]:
    """
    Given a WT sequence and mutation strings, returns embeddings and deltas.

    Returns:
      {
        wt_seq,
        mutations,
        mutant_seqs,
        wt_embedding,
        mutant_embeddings,
        delta_embeddings,
        delta_l2
      }
    """
    mutant_seqs = [apply_mutation(wt_seq, m, strict=strict) for m in mutations]

    all_seqs = [wt_seq] + mutant_seqs
    X = embed_sequences_esm2(all_seqs, cfg=cfg)

    wt_emb = X[0]
    mut_embs = X[1:]

    deltas = mut_embs - wt_emb[None, :]
    delta_l2 = np.linalg.norm(deltas, axis=1)

    return {
        "wt_seq": wt_seq,
        "mutations": list(mutations),
        "mutant_seqs": mutant_seqs,
        "wt_embedding": wt_emb,
        "mutant_embeddings": mut_embs,
        "delta_embeddings": deltas,
        "delta_l2": delta_l2,
    }
