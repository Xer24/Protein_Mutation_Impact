# src/mutations/features.py

from __future__ import annotations

from typing import Any, Dict, Sequence, Optional
import numpy as np
import re
from pathlib import Path

from src.features.esm2_embedding import embed_sequences_esm2, ESM2Config

# Amino acid validation
AA20 = set("ACDEFGHIKLMNPQRSTVWY")
MUT_RE = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")


def apply_mutation(wt_seq: str, mutation: str, strict: bool = True) -> str:
    """
    Apply a single mutation (e.g., 'A206T') to WT sequence.
    """
    mutation = mutation.strip().upper()
    
    m = MUT_RE.match(mutation)
    if not m:
        raise ValueError(f"Invalid mutation format: {mutation!r}. Expected format like 'A206T'")
    
    from_aa, pos_str, to_aa = m.groups()
    pos = int(pos_str)
    
    if pos < 1:
        raise ValueError(f"Position must be >= 1 in {mutation!r}")
    
    if from_aa == to_aa:
        raise ValueError(f"Invalid mutation (from==to): {mutation!r}")
    
    idx = pos - 1
    if idx < 0 or idx >= len(wt_seq):
        raise ValueError(
            f"Mutation {mutation} out of range for sequence length {len(wt_seq)}"
        )
    
    if strict and wt_seq[idx] != from_aa:
        raise ValueError(
            f"WT mismatch for {mutation}: expected '{from_aa}' at position {pos}, "
            f"found '{wt_seq[idx]}'"
        )
    
    mutant_seq = wt_seq[:idx] + to_aa + wt_seq[idx + 1:]
    return mutant_seq


def embed_wt_and_mutants(
    wt_seq: str,
    mutations: Sequence[str],
    *,
    strict: bool = True,
    cfg: Optional[ESM2Config] = None,
) -> Dict[str, Any]:
    """
    Given a WT sequence and mutation strings, returns embeddings and deltas.
    """
    # CRITICAL FIX: Default to t12 model (480 dims) to match training
    if cfg is None:
        cfg = ESM2Config(
            model_name="esm2_t30_35M_UR50D",  # ← MATCH TRAINING
            cache_path=Path("artifacts/esm2_cache.joblib"),
            device="auto",
            batch_size=4,
            max_len=None,
        )
    
    # Apply mutations
    mutant_seqs = [apply_mutation(wt_seq, m, strict=strict) for m in mutations]

    # Embed all sequences
    all_seqs = [wt_seq] + mutant_seqs
    X = embed_sequences_esm2(all_seqs, cfg=cfg)

    wt_emb = X[0]
    mut_embs = X[1:]

    # Compute deltas
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